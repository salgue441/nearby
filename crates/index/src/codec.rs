//! Little-endian, checksummed encoding shared by the index file formats.
//!
//! Every format is a sequence of fixed-width values ending with the CRC-32
//! of everything before it. [`Decoder`] turns every way the input can be
//! wrong (cut short, checksum mismatch, a length too large to allocate)
//! into [`Error::Corruption`], so a damaged file never panics or aborts.

use std::io::{Read, Write};

use nearby_core::checksum::{CrcReader, CrcWriter};
use nearby_core::{Error, Metric, Result};

/// Values converted per read or write of a large array, bounding the
/// temporary byte buffer.
const CHUNK: usize = 1 << 16;

/// Writes values, keeping a running checksum.
pub(crate) struct Encoder<W: Write> {
    w: CrcWriter<W>,
}

impl<W: Write> Encoder<W> {
    pub(crate) fn new(w: W) -> Self {
        Self {
            w: CrcWriter::new(w),
        }
    }

    pub(crate) fn bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.w.write_all(bytes)?;
        Ok(())
    }

    pub(crate) fn u8(&mut self, value: u8) -> Result<()> {
        self.bytes(&[value])
    }

    /// A count or size, which every format stores as a `u32`.
    ///
    /// # Panics
    ///
    /// If `value` doesn't fit in a `u32`.
    pub(crate) fn u32(&mut self, value: usize) -> Result<()> {
        let value = u32::try_from(value).expect("sizes fit in u32");
        self.bytes(&value.to_le_bytes())
    }

    pub(crate) fn u64(&mut self, value: u64) -> Result<()> {
        self.bytes(&value.to_le_bytes())
    }

    pub(crate) fn metric(&mut self, metric: Metric) -> Result<()> {
        self.u8(match metric {
            Metric::L2 => 0,
            Metric::Cosine => 1,
            Metric::Dot => 2,
        })
    }

    pub(crate) fn u64s(&mut self, values: &[u64]) -> Result<()> {
        let mut buf = Vec::with_capacity(CHUNK.min(values.len()) * 8);
        for chunk in values.chunks(CHUNK) {
            buf.clear();
            for value in chunk {
                buf.extend_from_slice(&value.to_le_bytes());
            }
            self.bytes(&buf)?;
        }
        Ok(())
    }

    pub(crate) fn f32s(&mut self, values: &[f32]) -> Result<()> {
        let mut buf = Vec::with_capacity(CHUNK.min(values.len()) * 4);
        for chunk in values.chunks(CHUNK) {
            buf.clear();
            for value in chunk {
                buf.extend_from_slice(&value.to_le_bytes());
            }
            self.bytes(&buf)?;
        }
        Ok(())
    }

    /// Writes the checksum of everything so far and flushes.
    pub(crate) fn finish(self) -> Result<()> {
        let crc = self.w.crc();
        let mut w = self.w.into_inner();
        w.write_all(&crc.to_le_bytes())?;
        w.flush()?;
        Ok(())
    }
}

/// Reads values written by an [`Encoder`], checking the checksum at the end.
pub(crate) struct Decoder<R: Read> {
    r: CrcReader<R>,
    /// Names the format in corruption errors, e.g. "HNSW index".
    what: &'static str,
}

impl<R: Read> Decoder<R> {
    pub(crate) fn new(r: R, what: &'static str) -> Self {
        Self {
            r: CrcReader::new(r),
            what,
        }
    }

    /// A corruption error naming this format.
    pub(crate) fn corrupt(&self, reason: &str) -> Error {
        Error::corruption(self.what, reason)
    }

    /// Reads exactly `buf.len()` bytes; running out of data is corruption.
    pub(crate) fn fill(&mut self, buf: &mut [u8]) -> Result<()> {
        self.r.read_exact(buf).map_err(|err| {
            if err.kind() == std::io::ErrorKind::UnexpectedEof {
                self.corrupt("truncated")
            } else {
                Error::Io(err)
            }
        })
    }

    pub(crate) fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut buf = [0; N];
        self.fill(&mut buf)?;
        Ok(buf)
    }

    /// Checks the 8-byte magic that starts a format.
    pub(crate) fn magic(&mut self, magic: [u8; 8]) -> Result<()> {
        if self.array::<8>()? == magic {
            Ok(())
        } else {
            Err(self.corrupt("unrecognized format"))
        }
    }

    pub(crate) fn u8(&mut self) -> Result<u8> {
        Ok(self.array::<1>()?[0])
    }

    /// A count or size stored as a `u32`.
    pub(crate) fn u32(&mut self) -> Result<usize> {
        Ok(u32::from_le_bytes(self.array()?) as usize)
    }

    pub(crate) fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    pub(crate) fn metric(&mut self) -> Result<Metric> {
        match self.u8()? {
            0 => Ok(Metric::L2),
            1 => Ok(Metric::Cosine),
            2 => Ok(Metric::Dot),
            _ => Err(self.corrupt("unknown metric")),
        }
    }

    /// An empty `Vec` with room for `n` items, or corruption if `n` is too
    /// large to allocate: a damaged length field must not abort.
    pub(crate) fn vec<T>(&self, n: usize) -> Result<Vec<T>> {
        let mut v = Vec::new();
        v.try_reserve_exact(n)
            .map_err(|_| self.corrupt("length too large to allocate"))?;
        Ok(v)
    }

    pub(crate) fn bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut bytes = self.vec(n)?;
        bytes.resize(n, 0);
        self.fill(&mut bytes)?;
        Ok(bytes)
    }

    pub(crate) fn u64s(&mut self, n: usize) -> Result<Vec<u64>> {
        let mut values = self.vec(n)?;
        let mut buf = vec![0u8; CHUNK.min(n) * 8];
        while values.len() < n {
            let take = (n - values.len()).min(CHUNK);
            let bytes = &mut buf[..take * 8];
            self.fill(bytes)?;
            values.extend(
                bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| u64::from_le_bytes(*b)),
            );
        }
        Ok(values)
    }

    pub(crate) fn f32s(&mut self, n: usize) -> Result<Vec<f32>> {
        let mut values = self.vec(n)?;
        let mut buf = vec![0u8; CHUNK.min(n) * 4];
        while values.len() < n {
            let take = (n - values.len()).min(CHUNK);
            let bytes = &mut buf[..take * 4];
            self.fill(bytes)?;
            values.extend(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b)),
            );
        }
        Ok(values)
    }

    /// Reads the stored checksum and compares it with the computed one.
    pub(crate) fn finish(self) -> Result<()> {
        let computed = self.r.crc();
        let what = self.what;
        let mut stored = [0; 4];
        self.r
            .into_inner()
            .read_exact(&mut stored)
            .map_err(|_| Error::corruption(what, "truncated"))?;
        if u32::from_le_bytes(stored) == computed {
            Ok(())
        } else {
            Err(Error::corruption(what, "checksum mismatch"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoded() -> Vec<u8> {
        let mut out = Vec::new();
        let mut e = Encoder::new(&mut out);
        e.bytes(b"NBTEST01").unwrap();
        e.u8(7).unwrap();
        e.u32(70_000).unwrap();
        e.u64(u64::MAX - 1).unwrap();
        e.metric(Metric::Cosine).unwrap();
        e.u64s(&[1, 2, 3]).unwrap();
        e.f32s(&[0.5, -1.25]).unwrap();
        e.finish().unwrap();
        out
    }

    #[test]
    fn values_round_trip() {
        let bytes = encoded();
        let mut d = Decoder::new(bytes.as_slice(), "test file");
        d.magic(*b"NBTEST01").unwrap();
        assert_eq!(d.u8().unwrap(), 7);
        assert_eq!(d.u32().unwrap(), 70_000);
        assert_eq!(d.u64().unwrap(), u64::MAX - 1);
        assert_eq!(d.metric().unwrap(), Metric::Cosine);
        assert_eq!(d.u64s(3).unwrap(), vec![1, 2, 3]);
        assert_eq!(d.f32s(2).unwrap(), vec![0.5, -1.25]);
        d.finish().unwrap();
    }

    #[test]
    fn damage_is_corruption() {
        let good = encoded();
        let mut flipped = good.clone();
        flipped[10] ^= 1;
        let mut d = Decoder::new(flipped.as_slice(), "test file");
        d.magic(*b"NBTEST01").unwrap();
        let _ = (d.u8(), d.u32(), d.u64(), d.metric(), d.u64s(3), d.f32s(2));
        assert!(d.finish().unwrap_err().to_string().contains("checksum"));

        let mut short = Decoder::new(&good[..12], "test file");
        short.magic(*b"NBTEST01").unwrap();
        short.u8().unwrap();
        let err = short.u32().unwrap_err();
        assert!(err.to_string().contains("truncated"), "{err}");

        let err = Decoder::new(&b"NBOTHER1"[..], "test file")
            .magic(*b"NBTEST01")
            .unwrap_err();
        assert!(err.to_string().contains("unrecognized format"), "{err}");
        let err = Decoder::new(&[9u8][..], "test file").metric().unwrap_err();
        assert!(err.to_string().contains("unknown metric"), "{err}");
    }

    #[test]
    fn huge_lengths_are_corruption_not_aborts() {
        let d = Decoder::new(&[][..], "test file");
        assert!(matches!(
            d.vec::<u64>(usize::MAX),
            Err(Error::Corruption { .. })
        ));
    }
}
