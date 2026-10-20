//! One collection's durable state: a write-ahead log plus segments.

use std::fs;
use std::path::{Path, PathBuf};

use nearby_core::{Error, Result};

use crate::manifest::{self, Manifest};
use crate::merge;
use crate::segment::{self, Changes};
use crate::wal::{self, Durability, Op, Wal};

/// Options for opening or creating a [`CollectionStore`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StoreOptions {
    /// When writes are forced to disk.
    pub durability: Durability,
}

/// The durable record of one collection's points.
///
/// Writes are appended to a write-ahead log. [`CollectionStore::flush`] turns
/// the log's net changes into an immutable segment and starts a new log, and
/// [`CollectionStore::compact`] merges every segment into one. On
/// [`CollectionStore::open`], the segments are replayed oldest first, then the
/// log, which rebuilds exactly the state the last acknowledged write left.
///
/// The store keeps no points in memory; whoever opens it rebuilds its own
/// state from the replayed ops. Every file lives in one directory, next to a
/// `MANIFEST` naming the live ones. Other files there are left alone, so the
/// caller can keep its own metadata alongside.
///
/// # Examples
///
/// ```
/// use nearby_storage::{CollectionStore, Op, StoreOptions};
///
/// let dir = tempfile::tempdir()?;
/// let mut store = CollectionStore::create(dir.path(), 2, StoreOptions::default())?;
/// store.write(&[Op::Upsert { id: 7, vector: &[1.0, 2.0], payload: &[] }])?;
/// drop(store);
///
/// let mut ids = Vec::new();
/// CollectionStore::open(dir.path(), StoreOptions::default(), |op| {
///     if let Op::Upsert { id, .. } = op {
///         ids.push(id);
///     }
/// })?;
/// assert_eq!(ids, [7]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct CollectionStore {
    dir: PathBuf,
    manifest: Manifest,
    wal: Wal,
    durability: Durability,
}

impl CollectionStore {
    /// Creates an empty store for `dim`-dimensional vectors in `dir`, which
    /// is created if needed.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if `dim` is 0 or `dir` already holds a
    /// store, or an I/O error.
    pub fn create(dir: &Path, dim: usize, options: StoreOptions) -> Result<Self> {
        if dim == 0 {
            return Err(Error::InvalidParameter(
                "dimension must be at least 1".to_owned(),
            ));
        }
        if Self::exists(dir) {
            return Err(Error::InvalidParameter(format!(
                "{} already holds a store",
                dir.display()
            )));
        }
        fs::create_dir_all(dir)?;
        let mut manifest = Manifest {
            dim,
            next_file: 1,
            wal: String::new(),
            segments: Vec::new(),
        };
        manifest.wal = manifest.new_wal_name();
        let wal = Wal::create(&dir.join(&manifest.wal), options.durability)?;
        manifest.write(dir)?;
        Ok(Self {
            dir: dir.to_owned(),
            manifest,
            wal,
            durability: options.durability,
        })
    }

    /// Whether `dir` holds a store.
    pub fn exists(dir: &Path) -> bool {
        dir.join(manifest::FILE_NAME).is_file()
    }

    /// Opens the store in `dir`, calling `apply` with every op it holds, in
    /// order: each segment's changes, oldest segment first, then the log.
    ///
    /// An incomplete record at the end of the log, left by a crash during a
    /// write, is dropped. Leftover files from an interrupted flush or
    /// compaction are deleted.
    ///
    /// # Errors
    ///
    /// An I/O error (`NotFound` if `dir` holds no store), or
    /// [`Error::Corruption`] if the manifest or a segment is damaged.
    pub fn open(dir: &Path, options: StoreOptions, apply: impl FnMut(Op<'_>)) -> Result<Self> {
        Self::open_replaying(dir, options, true, apply)
    }

    /// Like [`CollectionStore::open`], but replays only the log, skipping the
    /// segments.
    ///
    /// For callers that kept their own snapshot of the state the segments
    /// hold, taken when [`CollectionStore::generation`] was what
    /// [`CollectionStore::read_generation`] still reports.
    ///
    /// # Errors
    ///
    /// The same as [`CollectionStore::open`].
    pub fn open_log_only(
        dir: &Path,
        options: StoreOptions,
        apply: impl FnMut(Op<'_>),
    ) -> Result<Self> {
        Self::open_replaying(dir, options, false, apply)
    }

    /// The [`CollectionStore::generation`] of the store in `dir`, without
    /// opening it.
    ///
    /// # Errors
    ///
    /// An I/O error (`NotFound` if `dir` holds no store), or
    /// [`Error::Corruption`] if the manifest is damaged.
    pub fn read_generation(dir: &Path) -> Result<u64> {
        Ok(Manifest::read(dir)?.next_file)
    }

    fn open_replaying(
        dir: &Path,
        options: StoreOptions,
        replay_segments: bool,
        mut apply: impl FnMut(Op<'_>),
    ) -> Result<Self> {
        let manifest = Manifest::read(dir)?;
        if replay_segments {
            for name in &manifest.segments {
                segment::read(&dir.join(name), manifest.dim, &mut apply)?;
            }
        }
        let wal = Wal::open(
            &dir.join(&manifest.wal),
            manifest.dim,
            options.durability,
            &mut apply,
        )?;
        let store = Self {
            dir: dir.to_owned(),
            manifest,
            wal,
            durability: options.durability,
        };
        store.remove_unreferenced_files()?;
        Ok(store)
    }

    /// Identifies what the segments hold: it changes whenever a flush or
    /// compaction rewrites them, and never otherwise. Writes to the log
    /// leave it alone.
    pub fn generation(&self) -> u64 {
        self.manifest.next_file
    }

    /// The dimension every vector must have.
    pub fn dim(&self) -> usize {
        self.manifest.dim
    }

    /// The directory the store lives in.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Ops in the current log, which [`CollectionStore::flush`] would move
    /// into a segment.
    pub fn wal_ops(&self) -> usize {
        self.wal.ops()
    }

    /// Number of segments, which [`CollectionStore::compact`] would merge.
    pub fn segment_count(&self) -> usize {
        self.manifest.segments.len()
    }

    /// Appends `ops` to the log as one write. With [`Durability::Sync`] they
    /// are on disk when this returns.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if an upserted vector has the wrong
    /// length, in which case nothing is written; or an I/O error.
    pub fn write(&mut self, ops: &[Op<'_>]) -> Result<()> {
        wal::check_dims(ops, self.dim())?;
        self.wal.append(ops)
    }

    /// Moves the log's net changes into a new segment and starts an empty
    /// log. Does nothing if the log is empty.
    ///
    /// # Errors
    ///
    /// An I/O error. The store stays usable and consistent either way: until
    /// the new manifest is written, the old log remains the live one.
    pub fn flush(&mut self) -> Result<()> {
        if self.wal.ops() == 0 {
            return Ok(());
        }
        self.wal.sync()?;
        let mut changes = Changes::default();
        Wal::read(self.wal.path(), self.dim(), |op| changes.apply(op))?;

        let mut manifest = self.manifest.clone();
        let segment = manifest.new_segment_name();
        changes.write(&self.dir.join(&segment), self.dim())?;
        manifest.segments.push(segment);
        manifest.wal = manifest.new_wal_name();
        let wal = Wal::create(&self.dir.join(&manifest.wal), self.durability)?;
        self.commit(manifest)?;
        self.wal = wal;
        Ok(())
    }

    /// Flushes the log, then merges every segment into one that holds only
    /// the live points. Does nothing more if there is at most one segment.
    ///
    /// The merge streams through the segments in id order, so its memory use
    /// doesn't grow with the collection.
    ///
    /// # Errors
    ///
    /// The same as [`CollectionStore::flush`].
    pub fn compact(&mut self) -> Result<()> {
        self.flush()?;
        if self.manifest.segments.len() <= 1 {
            return Ok(());
        }
        let inputs: Vec<PathBuf> = self
            .manifest
            .segments
            .iter()
            .map(|name| self.dir.join(name))
            .collect();
        let mut manifest = self.manifest.clone();
        let segment = manifest.new_segment_name();
        merge::merge(&inputs, self.dim(), &self.dir.join(&segment))?;
        manifest.segments = vec![segment];
        self.commit(manifest)
    }

    /// Makes `manifest` the live one, then deletes the files it no longer names.
    fn commit(&mut self, manifest: Manifest) -> Result<()> {
        manifest.write(&self.dir)?;
        self.manifest = manifest;
        self.remove_unreferenced_files()
    }

    /// Deletes store files in the directory that the manifest doesn't name.
    fn remove_unreferenced_files(&self) -> Result<()> {
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if Manifest::is_store_file(name) && !self.manifest.references(name) {
                tracing::debug!(file = name, "removing unreferenced store file");
                fs::remove_file(entry.path())?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use nearby_core::PointId;

    use super::*;

    /// Opens the store and returns the live points the replay leaves.
    fn reopen(dir: &Path) -> (CollectionStore, BTreeMap<PointId, Vec<f32>>) {
        let mut points = BTreeMap::new();
        let store = CollectionStore::open(dir, StoreOptions::default(), |op| match op {
            Op::Upsert { id, vector, .. } => {
                points.insert(id, vector.to_vec());
            }
            Op::Delete { id } => {
                points.remove(&id);
            }
        })
        .unwrap();
        (store, points)
    }

    fn upsert(store: &mut CollectionStore, id: PointId, x: f32) {
        store
            .write(&[Op::Upsert {
                id,
                vector: &[x, x],
                payload: &[],
            }])
            .unwrap();
    }

    fn delete(store: &mut CollectionStore, id: PointId) {
        store.write(&[Op::Delete { id }]).unwrap();
    }

    fn files(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn writes_survive_reopening() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = CollectionStore::create(dir.path(), 2, StoreOptions::default()).unwrap();
        upsert(&mut store, 1, 1.0);
        upsert(&mut store, 2, 2.0);
        upsert(&mut store, 1, 3.0);
        delete(&mut store, 2);
        drop(store);

        let (store, points) = reopen(dir.path());
        assert_eq!(points, BTreeMap::from([(1, vec![3.0, 3.0])]));
        assert_eq!(store.wal_ops(), 4);
        assert_eq!(store.dim(), 2);
    }

    #[test]
    fn flushed_and_logged_changes_combine_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = CollectionStore::create(dir.path(), 2, StoreOptions::default()).unwrap();
        upsert(&mut store, 1, 1.0);
        upsert(&mut store, 2, 2.0);
        store.flush().unwrap();
        delete(&mut store, 1);
        upsert(&mut store, 3, 3.0);
        store.flush().unwrap();
        upsert(&mut store, 2, 9.0);
        assert_eq!((store.segment_count(), store.wal_ops()), (2, 1));
        drop(store);

        let (_, points) = reopen(dir.path());
        assert_eq!(
            points,
            BTreeMap::from([(2, vec![9.0, 9.0]), (3, vec![3.0, 3.0])])
        );
    }

    #[test]
    fn compaction_keeps_live_points_in_one_segment() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = CollectionStore::create(dir.path(), 2, StoreOptions::default()).unwrap();
        for id in 0..10 {
            upsert(&mut store, id, id as f32);
            store.flush().unwrap();
        }
        for id in 0..5 {
            delete(&mut store, id);
        }
        store.compact().unwrap();
        assert_eq!((store.segment_count(), store.wal_ops()), (1, 0));
        drop(store);

        assert_eq!(
            files(dir.path()),
            vec!["MANIFEST", "seg-000024.seg", "wal-000023.log"]
        );
        let (_, points) = reopen(dir.path());
        assert_eq!(
            points.keys().copied().collect::<Vec<_>>(),
            vec![5, 6, 7, 8, 9]
        );
    }

    #[test]
    fn flushing_an_empty_log_changes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = CollectionStore::create(dir.path(), 2, StoreOptions::default()).unwrap();
        store.flush().unwrap();
        store.compact().unwrap();
        assert_eq!(store.segment_count(), 0);
        assert_eq!(files(dir.path()), vec!["MANIFEST", "wal-000001.log"]);
    }

    #[test]
    fn leftovers_are_removed_and_other_files_kept() {
        let dir = tempfile::tempdir().unwrap();
        let store = CollectionStore::create(dir.path(), 2, StoreOptions::default()).unwrap();
        drop(store);
        for name in [
            "seg-000099.seg",
            "wal-000098.log",
            "MANIFEST.tmp",
            "collection.toml",
        ] {
            fs::write(dir.path().join(name), b"junk").unwrap();
        }
        reopen(dir.path());
        assert_eq!(
            files(dir.path()),
            vec!["MANIFEST", "collection.toml", "wal-000001.log"]
        );
    }

    #[test]
    fn create_rejects_bad_arguments() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            CollectionStore::create(dir.path(), 0, StoreOptions::default()),
            Err(Error::InvalidParameter(_))
        ));
        CollectionStore::create(dir.path(), 2, StoreOptions::default()).unwrap();
        assert!(CollectionStore::exists(dir.path()));
        let err = CollectionStore::create(dir.path(), 2, StoreOptions::default()).unwrap_err();
        assert!(err.to_string().contains("already holds a store"), "{err}");
    }

    #[test]
    fn write_rejects_wrong_dimensions_atomically() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = CollectionStore::create(dir.path(), 2, StoreOptions::default()).unwrap();
        let err = store
            .write(&[
                Op::Upsert {
                    id: 1,
                    vector: &[1.0, 1.0],
                    payload: &[],
                },
                Op::Upsert {
                    id: 2,
                    vector: &[1.0],
                    payload: &[],
                },
            ])
            .unwrap_err();
        assert!(matches!(err, Error::DimensionMismatch { .. }));
        assert_eq!(store.wal_ops(), 0, "nothing was written");
    }

    #[test]
    fn opening_a_missing_store_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let err = CollectionStore::open(dir.path(), StoreOptions::default(), |_| {}).unwrap_err();
        assert!(matches!(err, Error::Io(e) if e.kind() == std::io::ErrorKind::NotFound));
    }

    #[test]
    fn a_damaged_segment_fails_to_open() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = CollectionStore::create(dir.path(), 2, StoreOptions::default()).unwrap();
        upsert(&mut store, 1, 1.0);
        store.flush().unwrap();
        drop(store);
        let segment = dir.path().join("seg-000002.seg");
        let mut bytes = fs::read(&segment).unwrap();
        bytes[10] ^= 0xFF;
        fs::write(&segment, bytes).unwrap();

        let err = CollectionStore::open(dir.path(), StoreOptions::default(), |_| {}).unwrap_err();
        assert!(matches!(err, Error::Corruption { .. }), "{err}");
    }

    #[test]
    fn generation_changes_only_when_segments_do() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = CollectionStore::create(dir.path(), 2, StoreOptions::default()).unwrap();
        let g0 = store.generation();
        upsert(&mut store, 1, 1.0);
        assert_eq!(store.generation(), g0, "log writes keep it");
        store.flush().unwrap();
        let g1 = store.generation();
        assert_ne!(g1, g0);
        store.flush().unwrap();
        assert_eq!(store.generation(), g1, "an empty flush keeps it");
        upsert(&mut store, 2, 2.0);
        store.compact().unwrap();
        assert_ne!(store.generation(), g1);
        assert_eq!(
            CollectionStore::read_generation(dir.path()).unwrap(),
            store.generation()
        );
    }

    #[test]
    fn log_only_opening_skips_the_segments() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = CollectionStore::create(dir.path(), 2, StoreOptions::default()).unwrap();
        upsert(&mut store, 1, 1.0);
        store.flush().unwrap();
        upsert(&mut store, 2, 2.0);
        delete(&mut store, 1);
        drop(store);

        let mut seen = Vec::new();
        let store = CollectionStore::open_log_only(dir.path(), StoreOptions::default(), |op| {
            seen.push(format!("{op:?}"));
        })
        .unwrap();
        assert_eq!(seen.len(), 2, "only the two logged ops: {seen:?}");
        assert_eq!(store.segment_count(), 1);
    }

    #[test]
    fn no_sync_writes_are_still_replayed() {
        let dir = tempfile::tempdir().unwrap();
        let options = StoreOptions {
            durability: Durability::NoSync,
        };
        let mut store = CollectionStore::create(dir.path(), 2, options).unwrap();
        upsert(&mut store, 4, 4.0);
        drop(store);
        let (_, points) = reopen(dir.path());
        assert_eq!(points.len(), 1);
    }
}
