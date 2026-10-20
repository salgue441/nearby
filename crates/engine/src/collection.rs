//! A named set of points: durable storage, the id map and an index.

use std::fs;
use std::path::Path;

use nearby_core::{Error, PointId, Result, ScoredPoint, TopK, validate_vector};
use nearby_storage::{CollectionStore, Op, StoreOptions};
use serde::Serialize;

use crate::EngineOptions;
use crate::backend::{IndexBackend, new_index};
use crate::config::CollectionConfig;
use crate::points::PointTable;
use crate::snapshot;
use crate::vectors::VectorFile;

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
    /// The storage generation of the snapshot on disk, if one is current.
    snapshot_generation: Option<u64>,
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
        config.validate()?;
        fs::create_dir_all(dir)?;
        VectorFile::remove_leftovers(dir)?;
        let index = new_index(&config, 0, dir)?;
        config.write(&dir.join(CONFIG_FILE))?;
        let store = CollectionStore::create(dir, config.dim, store_options(&options))?;
        Ok(Self {
            name: name.to_owned(),
            points: PointTable::new(),
            config,
            options,
            store,
            index,
            snapshot_generation: None,
        })
    }

    /// Whether `dir` holds a collection.
    pub(crate) fn exists(dir: &Path) -> bool {
        dir.join(CONFIG_FILE).is_file()
    }

    /// Opens the collection in `dir`: replays its storage and rebuilds its index.
    ///
    /// If a snapshot taken at the current storage generation exists, it is
    /// loaded instead, and only the log written since is replayed.
    pub(crate) fn open(dir: &Path, name: &str, options: EngineOptions) -> Result<Self> {
        let config = CollectionConfig::read(&dir.join(CONFIG_FILE))?;
        VectorFile::remove_leftovers(dir)?;
        let generation = CollectionStore::read_generation(dir)?;
        if let Some((points, index)) = snapshot::load(dir, &config, generation) {
            return Self::open_from_snapshot(dir, name, config, options, points, index);
        }
        let mut points = PointTable::new();
        // Every vector replayed, by slot, until the index holds the live ones.
        let mut replayed: Vec<f32> = Vec::new();
        let store = CollectionStore::open(dir, store_options(&options), |op| match op {
            Op::Upsert { id, vector } => {
                points.upsert(id);
                replayed.extend_from_slice(vector);
            }
            Op::Delete { id } => {
                points.delete(id);
            }
        })?;
        check_dim(&store, &config)?;
        let dim = config.dim;
        let (points, index) = build_index(&config, &options, dir, &points, |slot| {
            let start = slot as usize * dim;
            &replayed[start..start + dim]
        })?;
        Ok(Self {
            name: name.to_owned(),
            config,
            options,
            store,
            points,
            index,
            snapshot_generation: None,
        })
    }

    /// Opens a collection whose state at the current storage generation is
    /// `points` and `index`, replaying the log written since on top.
    fn open_from_snapshot(
        dir: &Path,
        name: &str,
        config: CollectionConfig,
        options: EngineOptions,
        points: PointTable,
        index: Box<dyn IndexBackend>,
    ) -> Result<Self> {
        let mut logged = Vec::new();
        let store = CollectionStore::open_log_only(dir, store_options(&options), |op| {
            logged.push(Logged::from(op));
        })?;
        check_dim(&store, &config)?;
        let mut collection = Self {
            name: name.to_owned(),
            snapshot_generation: Some(store.generation()),
            config,
            options,
            store,
            points,
            index,
        };
        collection.apply_logged(&logged)?;
        tracing::info!(
            collection = %name,
            points = collection.len(),
            replayed = logged.len(),
            "restored the index from a snapshot"
        );
        Ok(collection)
    }

    /// Applies ops replayed from the log, which are already durable, to the
    /// in-memory state.
    fn apply_logged(&mut self, ops: &[Logged]) -> Result<()> {
        let mut inserted: Vec<(PointId, &[f32])> = Vec::new();
        for op in ops {
            match op {
                Logged::Upsert(id, vector) => {
                    let slot = self.points.upsert(*id);
                    inserted.push((u64::from(slot), vector));
                }
                Logged::Delete(id) => {
                    self.points.delete(*id);
                }
            }
        }
        self.index.insert_batch(&inserted, self.options.threads)
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
        self.points.slot(id).map(|slot| self.index.vector(slot))
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
            .map(|&(id, vector)| (u64::from(self.points.upsert(id)), vector))
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
            .filter(|&id| self.points.slot(id).is_some())
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
        let rerank = self.index.rerank();
        let candidates = rerank.map_or(k, |factor| k.saturating_mul(factor.get()));
        let live = |slot: PointId| self.points.live_id(slot as u32).is_some();
        let mut hits = if self.points.dead() == 0 {
            self.index.search(query, candidates)?
        } else {
            self.index.search_filtered(query, candidates, &live)?
        };
        if rerank.is_some() {
            hits = self.rerank(query, k, &hits);
        }
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

    /// The `k` best of `candidates` (slots) by exact distance to `query`.
    fn rerank(&self, query: &[f32], k: usize, candidates: &[ScoredPoint]) -> Vec<ScoredPoint> {
        let mut top = TopK::new(k);
        for hit in candidates {
            let vector = self.index.vector(hit.id as u32);
            top.push(hit.id, self.config.metric.distance(query, vector));
        }
        top.into_sorted()
    }

    /// Moves the write-ahead log into a segment, so reopening replays less.
    ///
    /// # Errors
    ///
    /// An I/O error.
    pub fn flush(&mut self) -> Result<()> {
        self.store.flush()?;
        if self
            .snapshot_generation
            .is_some_and(|g| g != self.store.generation())
        {
            // Stale now, and would be ignored anyway: reclaim the space.
            snapshot::remove(self.store.dir());
            self.snapshot_generation = None;
        }
        Ok(())
    }

    /// Flushes the log and, for an HNSW collection, saves a snapshot of the
    /// index, so the next open loads it instead of rebuilding. Called on
    /// shutdown.
    ///
    /// # Errors
    ///
    /// An I/O error. The collection stays consistent either way: the
    /// snapshot is only a cache.
    pub fn checkpoint(&mut self) -> Result<()> {
        self.store.flush()?;
        let generation = self.store.generation();
        if self.snapshot_generation == Some(generation) {
            return Ok(());
        }
        if snapshot::save(
            self.store.dir(),
            generation,
            &self.points,
            self.index.as_ref(),
        )? {
            self.snapshot_generation = Some(generation);
        }
        Ok(())
    }

    /// Housekeeping after a write: flushes a long log, compacts when
    /// segments pile up, and rebuilds the index when it is mostly dead.
    fn maintain(&mut self) -> Result<()> {
        if self.store.wal_ops() >= self.options.flush_after_ops {
            self.flush()?;
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
        let (points, index) = build_index(
            &self.config,
            &self.options,
            self.store.dir(),
            &self.points,
            |slot| self.index.vector(slot),
        )?;
        self.points = points;
        self.index = index;
        Ok(())
    }
}

/// A compacted copy of `points` and a fresh index over its live points,
/// reading each one's vector from `vector`.
fn build_index<'v>(
    config: &CollectionConfig,
    options: &EngineOptions,
    dir: &Path,
    points: &PointTable,
    vector: impl Fn(u32) -> &'v [f32],
) -> Result<(PointTable, Box<dyn IndexBackend>)> {
    let compacted = points.compacted();
    let mut index = new_index(config, compacted.len(), dir)?;
    // Slot `s` in the compacted table is the `s`-th live slot here.
    let batch: Vec<(PointId, &[f32])> = points
        .live_slots()
        .enumerate()
        .map(|(new_slot, slot)| (new_slot as PointId, vector(slot)))
        .collect();
    index.insert_batch(&batch, options.threads)?;
    Ok((compacted, index))
}

/// An op replayed from the log, owned so it outlives the replay callback.
#[derive(Debug)]
enum Logged {
    Upsert(PointId, Vec<f32>),
    Delete(PointId),
}

impl From<Op<'_>> for Logged {
    fn from(op: Op<'_>) -> Self {
        match op {
            Op::Upsert { id, vector } => Self::Upsert(id, vector.to_vec()),
            Op::Delete { id } => Self::Delete(id),
        }
    }
}

/// Checks that storage and config agree on the dimension.
fn check_dim(store: &CollectionStore, config: &CollectionConfig) -> Result<()> {
    if store.dim() == config.dim {
        return Ok(());
    }
    Err(Error::corruption(
        store.dir(),
        format!(
            "storage holds {}-d vectors but the config says {}",
            store.dim(),
            config.dim
        ),
    ))
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
    use crate::config::{HnswConfig, IndexConfig, PqConfig};

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

    /// One config per index type. The PQ collection trains at 256 points,
    /// so with fewer it searches exactly.
    fn every_index() -> [IndexConfig; 3] {
        [
            IndexConfig::Flat,
            IndexConfig::Hnsw(HnswConfig::default()),
            IndexConfig::Pq(PqConfig {
                subspaces: 2,
                training_points: 256,
                rerank: 2,
            }),
        ]
    }

    #[test]
    fn upserts_deletes_and_searches() {
        for index in every_index() {
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
        for index in every_index() {
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

    /// Names of the vector spill files in `dir`.
    fn spill_files(dir: &Path) -> Vec<String> {
        fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with("vectors-"))
            .collect()
    }

    #[test]
    fn pq_collections_keep_one_spill_file_through_rebuilds() {
        let dir = tempfile::tempdir().unwrap();
        let pq = IndexConfig::Pq(PqConfig {
            subspaces: 2,
            training_points: 256,
            rerank: 2,
        });
        let mut c = Collection::create(dir.path(), "c", config(pq), options()).unwrap();
        let points: Vec<(PointId, [f32; 2])> =
            (0..3_000u64).map(|id| (id, [id as f32, 1.0])).collect();
        let batch: Vec<(PointId, &[f32])> = points.iter().map(|(id, v)| (*id, &v[..])).collect();
        c.upsert(&batch).unwrap();
        let before = spill_files(dir.path());
        assert_eq!(before.len(), 1);

        let doomed: Vec<PointId> = (0..2_000).collect();
        c.delete(&doomed).unwrap();
        assert_eq!(c.points.dead(), 0, "rebuilt");
        let after = spill_files(dir.path());
        assert_eq!(after.len(), 1, "the old file went with the old index");
        assert_ne!(after, before);
        assert_eq!(c.get(2_500), Some(&[2_500.0, 1.0][..]));

        drop(c);
        assert_eq!(spill_files(dir.path()), Vec::<String>::new());
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

    fn hnsw() -> IndexConfig {
        IndexConfig::Hnsw(HnswConfig::default())
    }

    #[test]
    fn a_checkpoint_lets_reopening_skip_the_rebuild() {
        let dir = tempfile::tempdir().unwrap();
        let mut c = Collection::create(dir.path(), "c", config(hnsw()), options()).unwrap();
        c.upsert(&[(1, &[0.0, 0.0]), (2, &[1.0, 1.0]), (3, &[2.0, 2.0])])
            .unwrap();
        c.upsert(&[(2, &[5.0, 5.0])]).unwrap();
        c.delete(&[3]).unwrap();
        c.checkpoint().unwrap();
        let before = c.search(&[0.0, 0.0], 10).unwrap();
        drop(c);
        assert!(dir.path().join("index.snapshot").is_file());

        let c = Collection::open(dir.path(), "c", options()).unwrap();
        assert!(c.snapshot_generation.is_some(), "loaded from the snapshot");
        assert_eq!(c.search(&[0.0, 0.0], 10).unwrap(), before);
        assert_eq!(c.get(2), Some(&[5.0, 5.0][..]));
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn writes_after_a_checkpoint_are_replayed_onto_the_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let mut c = Collection::create(dir.path(), "c", config(hnsw()), options()).unwrap();
        c.upsert(&[(1, &[1.0, 1.0]), (2, &[2.0, 2.0])]).unwrap();
        c.checkpoint().unwrap();
        // Then a crash: these reach only the log.
        c.upsert(&[(3, &[3.0, 3.0]), (1, &[9.0, 9.0])]).unwrap();
        c.delete(&[2]).unwrap();
        drop(c);

        let c = Collection::open(dir.path(), "c", options()).unwrap();
        assert!(c.snapshot_generation.is_some(), "loaded from the snapshot");
        assert_eq!(c.len(), 2);
        assert_eq!(c.get(1), Some(&[9.0, 9.0][..]));
        assert_eq!(c.get(2), None);
        assert_eq!(ids(&c.search(&[3.0, 3.0], 5).unwrap()), vec![3, 1]);
    }

    #[test]
    fn a_flush_after_a_checkpoint_retires_the_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let mut c = Collection::create(dir.path(), "c", config(hnsw()), options()).unwrap();
        c.upsert(&[(1, &[1.0, 1.0])]).unwrap();
        c.checkpoint().unwrap();
        c.upsert(&[(2, &[2.0, 2.0])]).unwrap();
        c.flush().unwrap();
        assert!(
            !dir.path().join("index.snapshot").exists(),
            "stale, so removed"
        );
        drop(c);

        let c = Collection::open(dir.path(), "c", options()).unwrap();
        assert!(c.snapshot_generation.is_none(), "rebuilt instead");
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn flat_collections_checkpoint_without_a_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let mut c =
            Collection::create(dir.path(), "c", config(IndexConfig::Flat), options()).unwrap();
        c.upsert(&[(1, &[1.0, 1.0])]).unwrap();
        c.checkpoint().unwrap();
        assert!(!dir.path().join("index.snapshot").exists());
        assert_eq!(c.info().wal_ops, 0);
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
        // 16 subspaces don't divide 2 dimensions.
        let bad_pq = config(IndexConfig::Pq(PqConfig::default()));
        assert!(Collection::create(&dir.path().join("c"), "c", bad_pq, options()).is_err());
        assert!(
            !dir.path().join("b").exists(),
            "nothing is created for a bad config"
        );
    }

    /// 2000 pseudo-random 8-d points, ids `0..2000`.
    fn pq_points() -> Vec<(PointId, Vec<f32>)> {
        let mut state = 0x2545_F491_u32;
        (0..2_000)
            .map(|id| {
                let v = (0..8)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        (state % 1000) as f32 / 10.0
                    })
                    .collect();
                (id, v)
            })
            .collect()
    }

    #[test]
    fn pq_collections_rerank_with_exact_distances() {
        let dir = tempfile::tempdir().unwrap();
        let pq = IndexConfig::Pq(PqConfig {
            subspaces: 4,
            training_points: 500,
            rerank: 10,
        });
        let config = CollectionConfig {
            dim: 8,
            ..config(pq)
        };
        let mut c = Collection::create(dir.path(), "c", config, options()).unwrap();
        let points = pq_points();
        let batch: Vec<(PointId, &[f32])> = points.iter().map(|(id, v)| (*id, &v[..])).collect();
        c.upsert(&batch).unwrap();
        assert!(c.index.rerank().is_some(), "trained");

        // Every point finds itself, at its exact distance of 0.
        let found = points
            .iter()
            .filter(|(id, v)| {
                c.search(v, 1).unwrap()[0]
                    == ScoredPoint {
                        id: *id,
                        distance: 0.0,
                    }
            })
            .count();
        assert!(found >= 1_990, "{found} of 2000 found themselves");

        let query = &points[7].1;
        let hits = c.search(query, 5).unwrap();
        for hit in &hits {
            let exact = Metric::L2.distance(query, c.get(hit.id).unwrap());
            assert_eq!(hit.distance, exact, "distances are exact");
        }

        c.upsert(&[(7, &[500.0; 8])]).unwrap();
        c.delete(&[hits[1].id]).unwrap();
        let after = c.search(query, 5).unwrap();
        assert!(after.iter().all(|p| p.id != 7 && p.id != hits[1].id));
        drop(c);

        // Retraining on reopen sees the same points in the same order.
        let c = Collection::open(dir.path(), "c", options()).unwrap();
        assert_eq!(c.search(query, 5).unwrap(), after);
    }
}
