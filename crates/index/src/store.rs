//! In-memory storage for fixed-dimension vectors.

/// Vectors stored back to back in one buffer: vector `i` is
/// `data[i * dim..(i + 1) * dim]`.
///
/// One contiguous allocation keeps the CPU cache and prefetcher happy, and
/// the internal position `i` doubles as the vector's internal id.
/// Callers validate vectors before pushing; this type only stores them.
#[derive(Debug, Clone)]
pub(crate) struct VectorStore {
    dim: usize,
    data: Vec<f32>,
}

impl VectorStore {
    /// Creates an empty store with room for `capacity` vectors.
    ///
    /// # Panics
    ///
    /// If `dim` is 0.
    pub(crate) fn with_capacity(dim: usize, capacity: usize) -> Self {
        assert!(dim > 0, "dimension must be at least 1");
        Self {
            dim,
            data: Vec::with_capacity(capacity * dim),
        }
    }

    /// Number of values in each vector.
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    /// Number of vectors stored.
    pub(crate) fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    /// Appends a vector. Its internal id is the previous `len()`.
    pub(crate) fn push(&mut self, vector: &[f32]) {
        debug_assert_eq!(vector.len(), self.dim, "caller must validate the vector");
        self.data.extend_from_slice(vector);
    }

    /// The vector with internal id `i`. Panics if `i` is out of range.
    #[inline]
    pub(crate) fn get(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_get() {
        let mut store = VectorStore::with_capacity(2, 0);
        store.push(&[1.0, 2.0]);
        store.push(&[3.0, 4.0]);
        assert_eq!(store.len(), 2);
        assert_eq!(store.get(1), &[3.0, 4.0]);
    }
}
