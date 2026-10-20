//! A named set of points: durable storage, the id map and an index.

use std::fs;
use std::path::Path;

use nearby_core::{Error, PointId, Result, ScoredPoint, validate_vector};
use nearby_storage::{CollectionStore, Op, StoreOptions};
use serde::Serialize;

use crate::EngineOptions;
use crate::backend::{IndexBackend, new_index};
use crate::config::CollectionConfig;
use crate::points::PointTable;

/// The config's file name inside a collection directory.
const CONFIG_FILE: &str = "collection.toml";

/// Dead slots tolerated before the index is rebuilt without them, unless
/// they outnumber the live points.
const MIN_DEAD_FOR_REBUILD: usize = 1024;

/// A collection's points, kept durable, indexed and searchable.
///
/// Every write goes to the write-ahead log first, then into the in-memory id
/// map and index. Replacing or deleting a point leaves its old vector in the
/// index, where searches skip it; once dead vectors outnumber live ones, the
/// index is rebuilt without them.
///
/// Writes need `&mut self`; searches only `&self`, so many can run at once.
#[derive(Debug)]
pub struct Collection {
    name: String,
    config: CollectionConfig,
    options: EngineOptions,
    store: CollectionStore,
    points: PointTable,
    /// Indexes slots, not point ids: a point's id is `points.live_id(slot)`.
    index: Box<dyn IndexBackend>,
}

/// A summary of a collection, for listing and monitoring.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CollectionInfo {
    /// The collection's name.
    pub name: String,
    /// Its settings.
    pub config: CollectionConfig,
    /// Number of points.
    pub points: usize,
    /// Segments on disk.
    pub segments: usize,
    /// Writes in the log since the last flush.
    pub wal_ops: usize,
}

impl Collection {
    /// Creates an empty collection in `dir`.
    pub(crate) fn create(
        dir: &Path,
        name: &str,
        config: CollectionConfig,
        options: EngineOptions,
    ) -> Result<Self> {
        if config.dim == 0 {
            return Err(Error::InvalidParameter(
                "dimension must be at least 1".to_owned(),
            ));
        }
        let index = new_index(&config, 0)?;
        fs::create_dir_all(dir)?;
        config.write(&dir.join(CONFIG_FILE))?;
        let store = CollectionStore::create(dir, config.dim, store_options(&options))?;
        Ok(Self {
            name: name.to_owned(),
            points: PointTable::new(config.dim),
            config,
            options,
            store,
            index,
        })
    }

    /// Whether `dir` holds a collection.
    pub(crate) fn exists(dir: &Path) -> bool {
        dir.join(CONFIG_FILE).is_file()
    }

    /// Opens the collection in `dir`: replays its storage and rebuilds its index.
    pub(crate) fn open(dir: &Path, name: &str, options: EngineOptions) -> Result<Self> {
        let config = CollectionConfig::read(&dir.join(CONFIG_FILE))?;
        let mut points = PointTable::new(config.dim);
        let store = CollectionStore::open(dir, store_options(&options), |op| match op {
            Op::Upsert { id, vector } => {
                points.upsert(id, vector);
            }
            Op::Delete { id } => {
                points.delete(id);
            }
        })?;
        if store.dim() != config.dim {
            return Err(Error::corruption(
                dir,
                format!(
                    "storage holds {}-d vectors but the config says {}",
                    store.dim(),
                    config.dim
                ),
            ));
        }
        let mut collection = Self {
            name: name.to_owned(),
            index: new_index(&config, 0)?,
            config,
            options,
            store,
            points,
        };
        collection.rebuild_index()?;
        Ok(collection)
    }

    /// The collection's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The collection's settings.
    pub fn config(&self) -> &CollectionConfig {
        &self.config
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the collection holds no points.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A summary of the collection.
    pub fn info(&self) -> CollectionInfo {
        CollectionInfo {
            name: self.name.clone(),
            config: self.config.clone(),
            points: self.len(),
            segments: self.store.segment_count(),
            wal_ops: self.store.wal_ops(),
        }
    }

    /// Point `id`'s vector, if it exists.
    pub fn get(&self, id: PointId) -> Option<&[f32]> {
        self.points.get(id)
    }

    /// Inserts `points`, replacing any that already exist. If an id appears
    /// more than once, its last vector wins.
    ///
    /// The batch is one durable write: with [`Durability::Sync`] it is on
    /// disk when this returns.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] or [`Error::InvalidVector`] for the first
    /// bad vector, in which case nothing is written; or an I/O error.
    ///
    /// [`Durability::Sync`]: nearby_storage::Durability::Sync
    pub fn upsert(&mut self, points: &[(PointId, &[f32])]) -> Result<()> {
        for &(_, vector) in points {
            validate_vector(vector, self.config.dim)?;
        }
        let ops: Vec<Op<'_>> = points
            .iter()
            .map(|&(id, vector)| Op::Upsert { id, vector })
            .collect();
        self.store.write(&ops)?;
        let slots: Vec<(PointId, &[f32])> = points
            .iter()
            .map(|&(id, vector)| (u64::from(self.points.upsert(id, vector)), vector))
            .collect();
        self.index.insert_batch(&slots, self.options.threads)?;
        self.maintain()
    }

    /// Deletes the points with these ids. Ids that don't exist are ignored.
    /// Returns how many points were deleted.
    ///
    /// # Errors
    ///
    /// An I/O error, in which case nothing is deleted.
    pub fn delete(&mut self, ids: &[PointId]) -> Result<usize> {
        let mut existing: Vec<PointId> = ids
            .iter()
            .copied()
            .filter(|&id| self.points.get(id).is_some())
            .collect();
        existing.sort_unstable();
        existing.dedup();
        if existing.is_empty() {
            return Ok(0);
        }
        let ops: Vec<Op<'_>> = existing.iter().map(|&id| Op::Delete { id }).collect();
        self.store.write(&ops)?;
        for &id in &existing {
            self.points.delete(id);
        }
        self.maintain()?;
        Ok(existing.len())
    }

    /// Returns up to `k` points closest to `query`, closest first.
    ///
    /// Points at the same distance come back in the order they were last
    /// written.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] or [`Error::InvalidVector`] if `query`
    /// is not a valid vector for this collection.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<ScoredPoint>> {
        validate_vector(query, self.config.dim)?;
        let live = |slot: PointId| self.points.live_id(slot as u32).is_some();
        let hits = if self.points.dead() == 0 {
            self.index.search(query, k)?
        } else {
            self.index.search_filtered(query, k, &live)?
        };
        Ok(hits
            .into_iter()
            .filter_map(|hit| {
                Some(ScoredPoint {
                    id: self.points.live_id(hit.id as u32)?,
                    distance: hit.distance,
                })
            })
            .collect())
    }

    /// Moves the write-ahead log into a segment, so reopening replays less.
    ///
    /// # Errors
    ///
    /// An I/O error.
    pub fn flush(&mut self) -> Result<()> {
        self.store.flush()
    }

    /// Housekeeping after a write: flushes a long log, compacts when
    /// segments pile up, and rebuilds the index when it is mostly dead.
    fn maintain(&mut self) -> Result<()> {
        if self.store.wal_ops() >= self.options.flush_after_ops {
            self.store.flush()?;
            if self.store.segment_count() > self.options.max_segments {
                self.store.compact()?;
            }
        }
        let dead = self.points.dead();
        if dead >= MIN_DEAD_FOR_REBUILD && dead > self.points.len() {
            tracing::debug!(collection = %self.name, dead, "rebuilding index without dead points");
            self.rebuild_index()?;
        }
        Ok(())
    }

    /// Drops dead slots and builds a fresh index over the live points.
    fn rebuild_index(&mut self) -> Result<()> {
        self.points = self.points.compacted();
        let mut index = new_index(&self.config, self.points.len())?;
        let slots: Vec<(PointId, &[f32])> = self
            .points
            .live_slots()
            .map(|(slot, vector)| (u64::from(slot), vector))
            .collect();
        index.insert_batch(&slots, self.options.threads)?;
        self.index = index;
        Ok(())
    }
}

fn store_options(options: &EngineOptions) -> StoreOptions {
    StoreOptions {
        durability: options.durability,
    }
}

#[cfg(test)]
mod tests {
    use nearby_core::Metric;
    use nearby_storage::Durability;

    use super::*;
    use crate::config::{HnswConfig, IndexConfig};

    fn options() -> EngineOptions {
        EngineOptions {
            durability: Durability::NoSync,
            ..EngineOptions::default()
        }
    }

    fn config(index: IndexConfig) -> CollectionConfig {
        CollectionConfig {
            dim: 2,
            metric: Metric::L2,
            index,
        }
    }

    fn ids(results: &[ScoredPoint]) -> Vec<PointId> {
        results.iter().map(|p| p.id).collect()
    }

    fn both_indexes() -> [IndexConfig; 2] {
        [IndexConfig::Flat, IndexConfig::Hnsw(HnswConfig::default())]
    }

    #[test]
    fn upserts_deletes_and_searches() {
        for index in both_indexes() {
            let dir = tempfile::tempdir().unwrap();
            let mut c = Collection::create(dir.path(), "c", config(index), options()).unwrap();
            c.upsert(&[(1, &[0.0, 0.0]), (2, &[1.0, 0.0]), (3, &[5.0, 5.0])])
                .unwrap();
            assert_eq!(ids(&c.search(&[0.9, 0.0], 2).unwrap()), vec![2, 1]);

            c.upsert(&[(2, &[9.0, 9.0])]).unwrap();
            assert_eq!(c.get(2), Some(&[9.0, 9.0][..]));
            assert_eq!(ids(&c.search(&[0.9, 0.0], 2).unwrap()), vec![1, 3]);

            assert_eq!(
                c.delete(&[1, 1, 42]).unwrap(),
                1,
                "duplicates and unknown ids"
            );
            assert_eq!(ids(&c.search(&[0.0, 0.0], 5).unwrap()), vec![3, 2]);
            assert_eq!(c.len(), 2);
        }
    }

    #[test]
    fn state_survives_reopening() {
        for index in both_indexes() {
            let dir = tempfile::tempdir().unwrap();
            let mut c = Collection::create(dir.path(), "c", config(index), options()).unwrap();
            c.upsert(&[(1, &[1.0, 1.0]), (2, &[2.0, 2.0])]).unwrap();
            c.flush().unwrap();
            c.upsert(&[(3, &[3.0, 3.0]), (1, &[10.0, 10.0])]).unwrap();
            c.delete(&[2]).unwrap();
            let before = c.search(&[0.0, 0.0], 10).unwrap();
            drop(c);

            let c = Collection::open(dir.path(), "c", options()).unwrap();
            assert_eq!(c.len(), 2);
            assert_eq!(c.get(1), Some(&[10.0, 10.0][..]));
            assert_eq!(c.search(&[0.0, 0.0], 10).unwrap(), before);
            assert_eq!(c.info().segments, 1);
        }
    }

    #[test]
    fn invalid_vectors_write_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let mut c =
            Collection::create(dir.path(), "c", config(IndexConfig::Flat), options()).unwrap();
        assert!(matches!(
            c.upsert(&[(1, &[1.0, 1.0]), (2, &[f32::NAN, 0.0])]),
            Err(Error::InvalidVector)
        ));
        assert!(matches!(
            c.upsert(&[(1, &[1.0])]),
            Err(Error::DimensionMismatch { .. })
        ));
        assert!(matches!(
            c.search(&[1.0, 2.0, 3.0], 1),
            Err(Error::DimensionMismatch { .. })
        ));
        assert!(c.is_empty());
        assert_eq!(c.info().wal_ops, 0);
    }

    #[test]
    fn long_logs_are_flushed_and_segments_compacted() {
        let dir = tempfile::tempdir().unwrap();
        let options = EngineOptions {
            flush_after_ops: 10,
            max_segments: 3,
            ..options()
        };
        let mut c =
            Collection::create(dir.path(), "c", config(IndexConfig::Flat), options).unwrap();
        for id in 0..100u64 {
            c.upsert(&[(id, &[id as f32, 0.0])]).unwrap();
            let info = c.info();
            assert!(info.wal_ops < 10, "flushed at 10 ops");
            assert!(info.segments <= 3, "compacted beyond 3 segments");
        }
        drop(c);
        assert_eq!(
            Collection::open(dir.path(), "c", options).unwrap().len(),
            100
        );
    }

    #[test]
    fn mostly_dead_indexes_are_rebuilt() {
        let dir = tempfile::tempdir().unwrap();
        let mut c =
            Collection::create(dir.path(), "c", config(IndexConfig::Flat), options()).unwrap();
        let points: Vec<(PointId, [f32; 2])> =
            (0..3_000u64).map(|id| (id, [id as f32, 1.0])).collect();
        let batch: Vec<(PointId, &[f32])> = points.iter().map(|(id, v)| (*id, &v[..])).collect();
        c.upsert(&batch).unwrap();
        let doomed: Vec<PointId> = (0..2_000).collect();
        c.delete(&doomed).unwrap();
        assert_eq!(
            c.points.dead(),
            0,
            "2000 dead > 1000 live triggers a rebuild"
        );
        assert_eq!(c.index.len(), 1_000);
        assert_eq!(ids(&c.search(&[0.0, 1.0], 1).unwrap()), vec![2_000]);
    }

    #[test]
    fn create_rejects_bad_configs() {
        let dir = tempfile::tempdir().unwrap();
        let zero = CollectionConfig {
            dim: 0,
            ..config(IndexConfig::Flat)
        };
        assert!(Collection::create(&dir.path().join("a"), "a", zero, options()).is_err());
        let bad_m = config(IndexConfig::Hnsw(HnswConfig {
            m: 1,
            ..HnswConfig::default()
        }));
        assert!(Collection::create(&dir.path().join("b"), "b", bad_m, options()).is_err());
        assert!(
            !dir.path().join("b").exists(),
            "nothing is created for a bad config"
        );
    }
}
