//! Little-endian encoding of the integers and floats in every file.

/// Appends little-endian values to a byte buffer.
pub(crate) trait PutLe {
    fn put_u8(&mut self, value: u8);
    fn put_u32(&mut self, value: u32);
    fn put_u64(&mut self, value: u64);
    fn put_f32s(&mut self, values: &[f32]);
}

impl PutLe for Vec<u8> {
    fn put_u8(&mut self, value: u8) {
        self.push(value);
    }

    fn put_u32(&mut self, value: u32) {
        self.extend_from_slice(&value.to_le_bytes());
    }

    fn put_u64(&mut self, value: u64) {
        self.extend_from_slice(&value.to_le_bytes());
    }

    fn put_f32s(&mut self, values: &[f32]) {
        self.reserve(values.len() * 4);
        for value in values {
            self.extend_from_slice(&value.to_le_bytes());
        }
    }
}

/// Reads little-endian values from the front of a byte slice.
///
/// Every read returns `None` once the input runs out, so callers can turn a
/// truncated file into a corruption error with the context they have.
#[derive(Debug)]
pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    /// Bytes not read yet.
    pub(crate) fn remaining(&self) -> usize {
        self.bytes.len()
    }

    /// The next `n` bytes.
    pub(crate) fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if n > self.bytes.len() {
            return None;
        }
        let (head, tail) = self.bytes.split_at(n);
        self.bytes = tail;
        Some(head)
    }

    fn array<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.take(N)?.try_into().ok()
    }

    pub(crate) fn u8(&mut self) -> Option<u8> {
        Some(self.array::<1>()?[0])
    }

    pub(crate) fn u32(&mut self) -> Option<u32> {
        self.array().map(u32::from_le_bytes)
    }

    pub(crate) fn u64(&mut self) -> Option<u64> {
        self.array().map(u64::from_le_bytes)
    }

    /// Reads `out.len()` floats into `out`.
    pub(crate) fn f32s_into(&mut self, out: &mut [f32]) -> Option<()> {
        let bytes = self.take(out.len() * 4)?;
        for (value, chunk) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
            *value = f32::from_le_bytes(*chunk);
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_round_trip() {
        let mut buf = Vec::new();
        buf.put_u8(7);
        buf.put_u32(0xDEAD_BEEF);
        buf.put_u64(u64::MAX - 1);
        buf.put_f32s(&[1.5, -0.25]);

        let mut r = Reader::new(&buf);
        assert_eq!(r.u8(), Some(7));
        assert_eq!(r.u32(), Some(0xDEAD_BEEF));
        assert_eq!(r.u64(), Some(u64::MAX - 1));
        let mut floats = [0.0; 2];
        r.f32s_into(&mut floats).unwrap();
        assert_eq!(floats, [1.5, -0.25]);
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn short_input_reads_none_and_consumes_nothing() {
        let mut r = Reader::new(&[1, 2, 3]);
        assert_eq!(r.u32(), None);
        assert_eq!(r.remaining(), 3);
        assert_eq!(r.take(3), Some(&[1, 2, 3][..]));
        assert_eq!(r.u8(), None);
    }
}
