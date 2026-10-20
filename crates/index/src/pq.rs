//! Product quantization: compressed vectors and approximate distances.
//!
//! Jégou, Douze & Schmid, "Product quantization for nearest neighbor search"
//! (2011).
//!
//! A vector is split into `subspaces` equal slices. For each subspace,
//! k-means learns 256 centroids from training data, and every slice is then
//! replaced by the index of its nearest centroid: one byte. A 128-d `f32`
//! vector (512 bytes) becomes 16 bytes with 16 subspaces.
//!
//! A search never decodes the codes. It first measures the query's slices
//! against every centroid once, filling a `subspaces x 256` table, then each
//! stored vector's distance is just `subspaces` table lookups added up
//! (asymmetric distance computation: the query stays exact, only the stored
//! vectors are approximated).

mod index;
mod kmeans;
mod params;
mod quantizer;

pub use self::index::PqIndex;
pub use self::params::PqParams;
pub use self::quantizer::ProductQuantizer;

/// Centroids per subspace. 256 makes every code exactly one byte.
const CENTROIDS: usize = 256;
