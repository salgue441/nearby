//! CRC-32 (IEEE 802.3), the checksum guarding every file Nearby writes.
//!
//! [`crc32`] checks a buffer in one go; [`CrcWriter`] and [`CrcReader`]
//! compute it on the fly while streaming a file too big to buffer.

use std::io::{self, Read, Write};

/// Lookup table for the reflected polynomial `0xEDB8_8320`, one entry per byte value.
const TABLE: [u32; 256] = table();

const fn table() -> [u32; 256] {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 == 0 {
                crc >> 1
            } else {
                0xEDB8_8320 ^ (crc >> 1)
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// A CRC-32 computed incrementally over several chunks.
#[derive(Debug, Clone, Copy)]
pub struct Crc32 {
    state: u32,
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32 {
    /// The checksum of no bytes yet.
    pub fn new() -> Self {
        Self { state: !0 }
    }

    /// Adds `bytes` to the checksum.
    pub fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state =
                TABLE[((self.state ^ u32::from(byte)) & 0xFF) as usize] ^ (self.state >> 8);
        }
    }

    /// The checksum of every byte added so far.
    pub fn value(&self) -> u32 {
        !self.state
    }
}

/// The CRC-32 of `bytes`, as used by zip, PNG and Ethernet.
///
/// # Examples
///
/// ```
/// assert_eq!(nearby_core::checksum::crc32(b"123456789"), 0xCBF4_3926);
/// ```
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = Crc32::new();
    crc.update(bytes);
    crc.value()
}

/// Passes writes through to `W`, keeping the CRC-32 of everything written.
#[derive(Debug)]
pub struct CrcWriter<W> {
    inner: W,
    crc: Crc32,
}

impl<W: Write> CrcWriter<W> {
    /// Wraps `inner`.
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            crc: Crc32::new(),
        }
    }

    /// The checksum of every byte written so far.
    pub fn crc(&self) -> u32 {
        self.crc.value()
    }

    /// Unwraps the writer.
    pub fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: Write> Write for CrcWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.crc.update(&buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Passes reads through from `R`, keeping the CRC-32 of everything read.
#[derive(Debug)]
pub struct CrcReader<R> {
    inner: R,
    crc: Crc32,
}

impl<R: Read> CrcReader<R> {
    /// Wraps `inner`.
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            crc: Crc32::new(),
        }
    }

    /// The checksum of every byte read so far.
    pub fn crc(&self) -> u32 {
        self.crc.value()
    }

    /// Unwraps the reader.
    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R: Read> Read for CrcReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.crc.update(&buf[..n]);
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn incremental_equals_one_shot() {
        let mut crc = Crc32::new();
        crc.update(b"1234");
        crc.update(b"");
        crc.update(b"56789");
        assert_eq!(crc.value(), crc32(b"123456789"));
    }

    #[test]
    fn detects_a_flipped_bit() {
        let mut data = b"nearby".to_vec();
        let before = crc32(&data);
        data[2] ^= 0x10;
        assert_ne!(crc32(&data), before);
    }

    #[test]
    fn writer_and_reader_track_what_passes_through() {
        let mut writer = CrcWriter::new(Vec::new());
        writer.write_all(b"1234").unwrap();
        writer.write_all(b"56789").unwrap();
        assert_eq!(writer.crc(), 0xCBF4_3926);
        let bytes = writer.into_inner();

        let mut reader = CrcReader::new(bytes.as_slice());
        let mut out = Vec::new();
        reader.read_to_end(&mut out).unwrap();
        assert_eq!(
            (out.as_slice(), reader.crc()),
            (&b"123456789"[..], 0xCBF4_3926)
        );
    }
}
