//! Vector index algorithms: exact (flat) search, HNSW and product quantization.

use nearby_core::{Result, ScoredPoint};

mod flat;
pub use flat::FlatIndex;

/// What every index can do once it's built.
///
/// Building is left out on purpose: each index builds differently (HNSW has
/// parameters, PQ must be trained first), so that stays on the concrete types.
/// `Send + Sync` lets segments be searched from several threads at once.
pub trait VectorIndex: Send + Sync {
    /// Number of dimensions every vector in this index has.
    fn dim(&self) -> usize;

    /// Number of vectors in the index.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns up to `k` points closest to `query`, closest first.
    ///
    /// # Errors
    ///
    /// [`nearby_core::Error::DimensionMismatch`] if `query` has the wrong
    /// length, or [`nearby_core::Error::InvalidVector`] if it contains NaN.
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<ScoredPoint>>;
}
