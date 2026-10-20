//! Exact nearest-neighbor search: compare the query with every stored vector.
//!
//! Each query costs O(N * dim), so this gets slow on large data, but it is
//! always correct. That makes it the ground truth that every approximate
//! index (HNSW, PQ) is measured against.

use crate::VectorIndex;
use nearby_core::distance::{cosine_distance_with_norms, norm};
use nearby_core::{Metric, PointId, Result, ScoredPoint, TopK, validate_vector};

/// An exact (brute-force) vector index.
///
/// Vectors are stored back to back in one contiguous buffer, which keeps a
/// full scan cache-friendly. Ids live in a parallel `Vec`: the vector at
/// position `i` belongs to `ids[i]`.
///
/// With [`Metric::Cosine`], each vector's norm is also stored at insert time,
/// so a comparison costs one dot product instead of three. Vectors themselves
/// are stored unmodified, so [`FlatIndex::get`] returns exactly what was added.
///
/// # Examples
///
/// ```
/// use nearby_core::Metric;
/// use nearby_index::{FlatIndex, VectorIndex};
///
/// let mut index = FlatIndex::new(2, Metric::L2);
/// index.add(1, &[0.0, 0.0])?;
/// index.add(2, &[5.0, 5.0])?;
///
/// let results = index.search(&[1.0, 1.0], 1)?;
/// assert_eq!(results[0].id, 1);
/// # Ok::<(), nearby_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct FlatIndex {
    dim: usize,
    metric: Metric,
    data: Vec<f32>,
    ids: Vec<PointId>,
    /// `norms[i]` is the norm of vector `i`. Same length as `ids` for
    /// [`Metric::Cosine`], always empty for the other metrics.
    norms: Vec<f32>,
}

impl FlatIndex {
    /// Creates an empty index.
    ///
    /// # Panics
    ///
    /// If `dim` is 0.
    pub fn new(dim: usize, metric: Metric) -> Self {
        assert!(dim > 0, "dimension must be at least 1");
        Self {
            dim,
            metric,
            data: Vec::new(),
            ids: Vec::new(),
            norms: Vec::new(),
        }
    }

    /// Whether this index keeps per-vector norms (see `norms`).
    fn stores_norms(&self) -> bool {
        self.metric == Metric::Cosine
    }

    /// Creates an empty index with room for `capacity` vectors, so loading
    /// a dataset of known size doesn't reallocate.
    ///
    /// # Panics
    ///
    /// If `dim` is 0.
    pub fn with_capacity(dim: usize, metric: Metric, capacity: usize) -> Self {
        let mut index = Self::new(dim, metric);

        index.data.reserve(capacity * dim);
        index.ids.reserve(capacity);
        if index.stores_norms() {
            index.norms.reserve(capacity);
        }
        index
    }

    /// Adds one vector.
    ///
    /// Duplicate ids are not detected here; the engine's id map handles that later.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if `vector` has the wrong length, or
    /// [`Error::InvalidVector`] if it contains NaN or infinite values.
    /// On error the index is left unchanged.
    ///
    /// [`Error::DimensionMismatch`]: nearby_core::Error::DimensionMismatch
    /// [`Error::InvalidVector`]: nearby_core::Error::InvalidVector
    pub fn add(&mut self, id: PointId, vector: &[f32]) -> Result<()> {
        validate_vector(vector, self.dim)?;
        self.data.extend_from_slice(vector);
        self.ids.push(id);
        if self.stores_norms() {
            self.norms.push(norm(vector));
        }

        Ok(())
    }

    /// Returns the vector stored under `id`, or `None` if there is none.
    ///
    /// This scans the ids linearly, so it costs O(N). If `id` was added more
    /// than once, the first copy is returned.
    pub fn get(&self, id: PointId) -> Option<&[f32]> {
        let pos = self.ids.iter().position(|&stored| stored == id)?;
        let start = pos * self.dim;
        Some(&self.data[start..start + self.dim])
    }

    /// Every vector as `(id, vector)`, in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (PointId, &[f32])> {
        self.ids
            .iter()
            .copied()
            .zip(self.data.chunks_exact(self.dim))
    }
}

impl VectorIndex for FlatIndex {
    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        keep: &dyn Fn(PointId) -> bool,
    ) -> Result<Vec<ScoredPoint>> {
        validate_vector(query, self.dim)?;

        let mut top = TopK::new(k);
        let vectors = self.data.chunks_exact(self.dim).zip(&self.ids);

        if self.stores_norms() {
            let query_norm = norm(query);
            for ((vector, &id), &vector_norm) in vectors.zip(&self.norms) {
                if keep(id) {
                    let d = cosine_distance_with_norms(query, vector, query_norm, vector_norm);
                    top.push(id, d);
                }
            }
        } else {
            for (vector, &id) in vectors.filter(|&(_, &id)| keep(id)) {
                top.push(id, self.metric.distance(query, vector));
            }
        }

        Ok(top.into_sorted())
    }
}

#[cfg(test)]
mod tests {
    use nearby_core::Error;

    use super::*;

    /// Five points on a line: id i sits at x = i.
    fn line_index(metric: Metric) -> FlatIndex {
        let mut index = FlatIndex::new(2, metric);
        for i in 0..5u8 {
            index.add(u64::from(i), &[f32::from(i), 0.0]).unwrap();
        }
        index
    }

    fn ids(results: &[ScoredPoint]) -> Vec<PointId> {
        results.iter().map(|p| p.id).collect()
    }

    #[test]
    fn finds_nearest_in_order() {
        let index = line_index(Metric::L2);
        let results = index.search(&[2.2, 0.0], 3).unwrap();
        assert_eq!(ids(&results), vec![2, 3, 1]);
    }

    #[test]
    fn exact_match_comes_first() {
        let index = line_index(Metric::L2);
        let results = index.search(&[4.0, 0.0], 1).unwrap();
        assert_eq!(ids(&results), vec![4]);
        assert!(results[0].distance.abs() < 1e-6);
    }

    #[test]
    fn k_larger_than_index_returns_everything() {
        let index = line_index(Metric::L2);
        assert_eq!(index.search(&[0.0, 0.0], 100).unwrap().len(), 5);
    }

    #[test]
    fn empty_index_returns_nothing() {
        let index = FlatIndex::new(3, Metric::L2);
        assert!(index.is_empty());
        assert_eq!(index.search(&[1.0, 2.0, 3.0], 5).unwrap(), Vec::new());
    }

    #[test]
    fn rejects_wrong_dimension() {
        let mut index = FlatIndex::new(3, Metric::L2);
        assert!(matches!(
            index.add(1, &[1.0, 2.0]),
            Err(Error::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
        assert!(matches!(
            index.search(&[1.0], 1),
            Err(Error::DimensionMismatch { .. })
        ));
        assert!(index.is_empty(), "a rejected vector must not be stored");
    }

    #[test]
    fn cosine_ignores_length() {
        let mut index = FlatIndex::new(2, Metric::Cosine);
        index.add(1, &[10.0, 0.0]).unwrap(); // same direction, far away
        index.add(2, &[1.0, 1.0]).unwrap(); // 45 degrees off, closer in L2
        let results = index.search(&[1.0, 0.0], 2).unwrap();
        assert_eq!(ids(&results), vec![1, 2]);
    }

    #[test]
    fn get_returns_stored_vectors() {
        let index = line_index(Metric::L2);
        assert_eq!(index.get(3), Some(&[3.0, 0.0][..]));
        assert_eq!(index.get(5), None);
    }

    #[test]
    fn iter_yields_vectors_in_insertion_order() {
        let index = line_index(Metric::L2);
        let points: Vec<(PointId, &[f32])> = index.iter().collect();
        assert_eq!(points.len(), 5);
        assert_eq!(points[3], (3, &[3.0, 0.0][..]));
    }

    #[test]
    fn cosine_index_stores_one_norm_per_vector() {
        let index = line_index(Metric::Cosine);
        assert_eq!(index.norms, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn other_metrics_store_no_norms() {
        for metric in [Metric::L2, Metric::Dot] {
            let index = line_index(metric);
            assert!(index.norms.is_empty(), "{metric:?}");
            assert_eq!(index.norms.capacity(), 0, "{metric:?}");
        }
        let index = FlatIndex::with_capacity(4, Metric::L2, 1_000);
        assert_eq!(index.norms.capacity(), 0, "no memory reserved either");
    }

    #[test]
    fn rejected_vector_stores_no_norm() {
        let mut index = FlatIndex::new(2, Metric::Cosine);
        index.add(1, &[3.0, 4.0]).unwrap();
        assert!(index.add(2, &[f32::NAN, 0.0]).is_err());
        assert!(index.add(3, &[1.0]).is_err());
        assert_eq!(index.norms, vec![5.0]);
    }
}
