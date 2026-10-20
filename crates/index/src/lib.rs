//! Vector index algorithms.
//!
//! - [`FlatIndex`]: exact search. Slow on large data, but always correct,
//!   so it is the ground truth the approximate indexes are measured against.
//! - [`Hnsw`]: approximate search over a layered proximity graph. Much
//!   faster on large data; recall is tuned with [`HnswParams`].
//!
//! Product quantization is planned and will implement the same
//! [`VectorIndex`] trait.

use nearby_core::{Result, ScoredPoint};

mod flat;
mod hnsw;
mod parallel;
mod store;

pub use flat::FlatIndex;
pub use hnsw::{Hnsw, HnswParams};

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

    /// True if the index holds no vectors.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns up to `k` points closest to `query`, closest first.
    ///
    /// Fewer than `k` points come back if the index holds fewer than `k`
    /// vectors, and `k == 0` always returns an empty `Vec`. Points at the
    /// same distance are ordered by ascending id.
    ///
    /// # Errors
    ///
    /// [`nearby_core::Error::DimensionMismatch`] if `query` has the wrong
    /// length, or [`nearby_core::Error::InvalidVector`] if it contains NaN
    /// or infinite values.
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<ScoredPoint>>;
}
