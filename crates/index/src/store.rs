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

    /// A store holding `data`, whose length must be a multiple of `dim`.
    pub(crate) fn from_vec(dim: usize, data: Vec<f32>) -> Self {
        assert!(dim > 0, "dimension must be at least 1");
        assert!(
            data.len().is_multiple_of(dim),
            "data must hold whole vectors"
        );
        Self { dim, data }
    }

    /// Every vector, back to back.
    pub(crate) fn as_slice(&self) -> &[f32] {
        &self.data
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

    /// Asks the CPU to start loading vector `i` into cache, without waiting.
    ///
    /// Graph searches read vectors at random addresses, so each read would
    /// otherwise stall on main memory. Prefetching all of a node's unvisited
    /// neighbors before measuring any of them overlaps those loads: about
    /// 1.6x faster HNSW searches on a 200k x 128-d dataset.
    ///
    /// Only a hint: it never changes results, and it is a no-op on CPUs other
    /// than x86-64. Panics if `i` is out of range, like [`VectorStore::get`].
    #[inline]
    pub(crate) fn prefetch(&self, i: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

            // One request per 64-byte cache line (16 f32s).
            for line in self.get(i).chunks(16) {
                // SAFETY: `_mm_prefetch` only needs SSE, which every x86-64
                // CPU has. A prefetch is a hint that never faults, and the
                // pointer comes from a live slice anyway.
                unsafe { _mm_prefetch::<_MM_HINT_T0>(line.as_ptr().cast()) };
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = self.get(i);
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

    #[test]
    fn prefetch_does_not_change_contents() {
        // 40 values per vector: spans three cache lines, the last partial.
        let mut store = VectorStore::with_capacity(40, 2);
        let a: Vec<f32> = (0..40u8).map(f32::from).collect();
        store.push(&a);
        store.push(&a);
        store.prefetch(0);
        store.prefetch(1);
        assert_eq!(store.get(1), a.as_slice());
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn prefetch_checks_bounds() {
        let store = VectorStore::with_capacity(2, 0);
        store.prefetch(0);
    }
}
