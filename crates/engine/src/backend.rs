//! The indexes a collection can search with, behind one interface.

use std::num::NonZeroUsize;

use nearby_core::{PointId, Result};
use nearby_index::{FlatIndex, Hnsw, VectorIndex};

use crate::config::{CollectionConfig, IndexConfig};

/// An index the engine can both search and append to.
///
/// [`VectorIndex`] covers searching. Building differs per index, so it is
/// left out of that trait; this one adds the single way the engine builds:
/// appending batches of points.
pub(crate) trait IndexBackend: VectorIndex + std::fmt::Debug {
    /// Appends `points`, using up to `threads` threads where the index can.
    fn insert_batch(&mut self, points: &[(PointId, &[f32])], threads: NonZeroUsize) -> Result<()>;
}

impl IndexBackend for FlatIndex {
    fn insert_batch(&mut self, points: &[(PointId, &[f32])], _: NonZeroUsize) -> Result<()> {
        points
            .iter()
            .try_for_each(|&(id, vector)| self.add(id, vector))
    }
}

impl IndexBackend for Hnsw {
    fn insert_batch(&mut self, points: &[(PointId, &[f32])], threads: NonZeroUsize) -> Result<()> {
        Hnsw::insert_batch(self, points, threads)
    }
}

/// An empty index as `config` describes, with room for `capacity` points.
pub(crate) fn new_index(
    config: &CollectionConfig,
    capacity: usize,
) -> Result<Box<dyn IndexBackend>> {
    Ok(match &config.index {
        IndexConfig::Flat => Box::new(FlatIndex::with_capacity(
            config.dim,
            config.metric,
            capacity,
        )),
        IndexConfig::Hnsw(hnsw) => Box::new(Hnsw::with_capacity(
            config.dim,
            config.metric,
            hnsw.params(),
            capacity,
        )?),
    })
}

#[cfg(test)]
mod tests {
    use nearby_core::Metric;

    use super::*;
    use crate::config::HnswConfig;

    #[test]
    fn every_backend_inserts_and_searches() {
        for index in [IndexConfig::Flat, IndexConfig::Hnsw(HnswConfig::default())] {
            let config = CollectionConfig {
                dim: 2,
                metric: Metric::L2,
                index,
            };
            let mut backend = new_index(&config, 0).unwrap();
            backend
                .insert_batch(&[(0, &[0.0, 0.0]), (1, &[5.0, 5.0])], NonZeroUsize::MIN)
                .unwrap();
            assert_eq!(backend.len(), 2);
            assert_eq!(backend.search(&[4.0, 4.0], 1).unwrap()[0].id, 1);
        }
    }

    #[test]
    fn invalid_hnsw_settings_are_rejected() {
        let config = CollectionConfig {
            dim: 2,
            metric: Metric::L2,
            index: IndexConfig::Hnsw(HnswConfig {
                m: 1,
                ..HnswConfig::default()
            }),
        };
        assert!(new_index(&config, 0).is_err());
    }
}
