//! The indexes a collection can search with, behind one interface.

use std::io::{Read, Write};
use std::num::NonZeroUsize;
use std::path::Path;

use nearby_core::{Error, PointId, Result};
use nearby_index::{FlatIndex, Hnsw, VectorIndex};

use crate::config::{CollectionConfig, IndexConfig};
use crate::pq::PqBackend;

/// An index the engine can both search and append to.
///
/// [`VectorIndex`] covers searching. Building differs per index, so it is
/// left out of that trait; this one adds the single way the engine builds:
/// appending batches of points.
pub(crate) trait IndexBackend: VectorIndex + std::fmt::Debug {
    /// Appends `points`, using up to `threads` threads where the index can.
    fn insert_batch(&mut self, points: &[(PointId, &[f32])], threads: NonZeroUsize) -> Result<()>;

    /// The vector in `slot`.
    ///
    /// # Panics
    ///
    /// If `slot` is out of range.
    fn vector(&self, slot: u32) -> &[f32];

    /// If the index's distances are approximate: how many candidates to
    /// fetch per requested result, to be re-ranked by exact distance.
    /// `None` means search results are final.
    fn rerank(&self) -> Option<NonZeroUsize> {
        None
    }

    /// Whether [`IndexBackend::save`] is supported.
    fn can_save(&self) -> bool {
        false
    }

    /// Writes the index for a snapshot (see `crate::snapshot`).
    ///
    /// # Panics
    ///
    /// If [`IndexBackend::can_save`] is false.
    fn save(&self, _w: &mut dyn Write) -> Result<()> {
        unreachable!("this index cannot be saved")
    }
}

impl IndexBackend for FlatIndex {
    fn insert_batch(&mut self, points: &[(PointId, &[f32])], _: NonZeroUsize) -> Result<()> {
        points
            .iter()
            .try_for_each(|&(id, vector)| self.add(id, vector))
    }

    fn vector(&self, slot: u32) -> &[f32] {
        self.vector_at(slot as usize).expect("slot in range")
    }
}

impl IndexBackend for Hnsw {
    fn insert_batch(&mut self, points: &[(PointId, &[f32])], threads: NonZeroUsize) -> Result<()> {
        Hnsw::insert_batch(self, points, threads)
    }

    fn vector(&self, slot: u32) -> &[f32] {
        self.vector_at(slot as usize).expect("slot in range")
    }

    fn can_save(&self) -> bool {
        true
    }

    fn save(&self, w: &mut dyn Write) -> Result<()> {
        self.write_to(w)
    }
}

/// An empty index as `config` describes, with room for `capacity` points.
/// An index that spills to disk does so in `dir`.
pub(crate) fn new_index(
    config: &CollectionConfig,
    capacity: usize,
    dir: &Path,
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
        IndexConfig::Pq(pq) => Box::new(PqBackend::new(config.dim, config.metric, pq, dir)?),
    })
}

/// Restores an index that [`IndexBackend::save`] wrote for a collection set
/// up as `config`, spilling to `dir` where the index does. The index must
/// hold exactly `slots` points, point `s` in slot `s`.
///
/// `Ok(None)` means the saved index doesn't apply: it was saved with other
/// settings, or `config` names an index that is never saved.
///
/// # Errors
///
/// [`nearby_core::Error::Corruption`] if the data is damaged or doesn't
/// match `slots`, or an I/O error.
pub(crate) fn load(
    config: &CollectionConfig,
    dir: &Path,
    r: &mut dyn Read,
    slots: usize,
) -> Result<Option<Box<dyn IndexBackend>>> {
    match &config.index {
        IndexConfig::Flat => Ok(None),
        IndexConfig::Hnsw(_) => {
            let index = Hnsw::read_from(r)?;
            if index.dim() != config.dim || index.metric() != config.metric {
                return Ok(None);
            }
            // The engine indexes slots: node `s` must be slot `s`.
            if index.len() != slots || index.iter().enumerate().any(|(s, (id, _))| id != s as u64) {
                return Err(Error::corruption(
                    "HNSW index",
                    "index does not match the slot table",
                ));
            }
            Ok(Some(Box::new(index)))
        }
        IndexConfig::Pq(pq) => {
            let backend = PqBackend::load(config.dim, config.metric, pq, dir, r, slots)?;
            Ok(backend.map(|b| Box::new(b) as Box<dyn IndexBackend>))
        }
    }
}

#[cfg(test)]
mod tests {
    use nearby_core::Metric;

    use super::*;
    use crate::config::{HnswConfig, PqConfig};

    #[test]
    fn every_backend_inserts_searches_and_lends_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let pq = IndexConfig::Pq(PqConfig {
            subspaces: 2,
            ..PqConfig::default()
        });
        for index in [
            IndexConfig::Flat,
            IndexConfig::Hnsw(HnswConfig::default()),
            pq,
        ] {
            let config = CollectionConfig {
                dim: 2,
                metric: Metric::L2,
                index,
            };
            let mut backend = new_index(&config, 0, dir.path()).unwrap();
            backend
                .insert_batch(&[(0, &[0.0, 0.0]), (1, &[5.0, 5.0])], NonZeroUsize::MIN)
                .unwrap();
            assert_eq!(backend.len(), 2);
            assert_eq!(backend.search(&[4.0, 4.0], 1).unwrap()[0].id, 1);
            assert_eq!(backend.vector(1), &[5.0, 5.0]);
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
        assert!(new_index(&config, 0, Path::new(".")).is_err());
    }
}
