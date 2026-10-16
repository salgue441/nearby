//! Exact nearest-neighbor search: compare the query with every stored vector.
//!
//! Each query costs O(N * dim), so this gets slow on large data, but it is
//! always correct. That makes it the ground truth that every approximate
//! index (HNSW, PQ) is measured against.

use crate::VectorIndex;
use nearby_core::{Metric, PointId, Result, ScoredPoint, TopK, validate_vector};

#[derive(Debug, Clone)]
pub struct FlatIndex {
    dim: usize,
    metric: Metric,
    data: Vec<f32>,
    ids: Vec<PointId>,
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
        }
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
        index
    }

    /// Adds one vector.
    ///
    /// Duplicate ids are not detected here; the engine's id map handles that later.
    ///
    /// # Errors
    ///
    /// If `vector` has the wrong length or contains NaN/infinite values.
    pub fn add(&mut self, id: PointId, vector: &[f32]) -> Result<()> {
        validate_vector(vector, self.dim)?;
        self.data.extend_from_slice(vector);
        self.ids.push(id);

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
}

impl VectorIndex for FlatIndex {
    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<ScoredPoint>> {
        validate_vector(query, self.dim)?;

        let mut top = TopK::new(k);
        for (vector, &id) in self.data.chunks_exact(self.dim).zip(&self.ids) {
            top.push(id, self.metric.distance(query, vector));
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
}
