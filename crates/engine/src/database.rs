//! Every collection under one data directory.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, PoisonError, RwLock};

use nearby_core::{Error, Result};

use crate::EngineOptions;
use crate::collection::Collection;
use crate::config::CollectionConfig;

/// Longest allowed collection name.
const MAX_NAME_LEN: usize = 64;

/// A shared handle to one collection: many readers or one writer at a time.
pub type CollectionHandle = Arc<RwLock<Collection>>;

/// The collections stored under one data directory, each in
/// `<data_dir>/collections/<name>/`.
///
/// All methods take `&self`, so one `Database` can be shared by every
/// request handler. Collections are locked individually: searches in one
/// collection run alongside each other and alongside writes to others.
///
/// # Examples
///
/// ```
/// use nearby_engine::{CollectionConfig, Database, EngineOptions, IndexConfig};
///
/// let dir = tempfile::tempdir()?;
/// let db = Database::open(dir.path(), EngineOptions::default())?;
/// let config = CollectionConfig { dim: 2, metric: Default::default(), index: IndexConfig::Flat };
/// let docs = db.create_collection("docs", config)?;
///
/// docs.write().unwrap().upsert(&[(1, &[0.0, 0.0]), (2, &[5.0, 5.0])])?;
/// let hits = docs.read().unwrap().search(&[4.0, 4.0], 1)?;
/// assert_eq!(hits[0].id, 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct Database {
    collections_dir: PathBuf,
    options: EngineOptions,
    collections: RwLock<BTreeMap<String, CollectionHandle>>,
}

impl Database {
    /// Opens the database in `data_dir`, creating the directory if needed,
    /// and opens every collection in it.
    ///
    /// # Errors
    ///
    /// An I/O error, or [`Error::Corruption`] if a collection's files are
    /// damaged.
    pub fn open(data_dir: &Path, options: EngineOptions) -> Result<Self> {
        let collections_dir = data_dir.join("collections");
        fs::create_dir_all(&collections_dir)?;
        let mut collections = BTreeMap::new();
        for entry in fs::read_dir(&collections_dir)? {
            let entry = entry?;
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            if !is_valid_name(&name) || !Collection::exists(&entry.path()) {
                continue;
            }
            let collection = Collection::open(&entry.path(), &name, options)?;
            tracing::info!(collection = %name, points = collection.len(), "opened collection");
            collections.insert(name, Arc::new(RwLock::new(collection)));
        }
        Ok(Self {
            collections_dir,
            options,
            collections: RwLock::new(collections),
        })
    }

    /// The names of every collection, sorted.
    pub fn list(&self) -> Vec<String> {
        self.read_map().keys().cloned().collect()
    }

    /// The collection called `name`.
    ///
    /// # Errors
    ///
    /// [`Error::CollectionNotFound`] if there is none.
    pub fn collection(&self, name: &str) -> Result<CollectionHandle> {
        self.read_map()
            .get(name)
            .cloned()
            .ok_or_else(|| Error::CollectionNotFound(name.to_owned()))
    }

    /// Creates an empty collection.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if the name is invalid (it must be 1 to 64
    /// ASCII letters, digits, `-` or `_`) or the config is;
    /// [`Error::CollectionExists`] if the name is taken; or an I/O error.
    pub fn create_collection(
        &self,
        name: &str,
        config: CollectionConfig,
    ) -> Result<CollectionHandle> {
        if !is_valid_name(name) {
            return Err(Error::InvalidParameter(format!(
                "collection names are 1 to {MAX_NAME_LEN} ASCII letters, digits, `-` or `_`, \
                 got `{name}`"
            )));
        }
        let mut map = self.write_map();
        if map.contains_key(name) {
            return Err(Error::CollectionExists(name.to_owned()));
        }
        let dir = self.collections_dir.join(name);
        // A directory without a collection is a leftover from a failed create.
        if dir.exists() {
            fs::remove_dir_all(&dir)?;
        }
        let collection =
            Collection::create(&dir, name, config, self.options).inspect_err(|_| {
                let _ = fs::remove_dir_all(&dir);
            })?;
        let handle = Arc::new(RwLock::new(collection));
        map.insert(name.to_owned(), Arc::clone(&handle));
        tracing::info!(collection = %name, "created collection");
        Ok(handle)
    }

    /// Deletes the collection called `name` and all its files.
    ///
    /// # Errors
    ///
    /// [`Error::CollectionNotFound`] if there is none, or an I/O error.
    pub fn drop_collection(&self, name: &str) -> Result<()> {
        let mut map = self.write_map();
        let handle = map
            .remove(name)
            .ok_or_else(|| Error::CollectionNotFound(name.to_owned()))?;
        // Wait for in-flight requests on it to finish before its files go.
        drop(handle.write().unwrap_or_else(PoisonError::into_inner));
        fs::remove_dir_all(self.collections_dir.join(name))?;
        tracing::info!(collection = %name, "dropped collection");
        Ok(())
    }

    /// Checkpoints every collection (see [`Collection::checkpoint`]): flushes
    /// its log and saves HNSW indexes, so the next start is quick.
    ///
    /// # Errors
    ///
    /// The first I/O error; the remaining collections are still checkpointed.
    pub fn checkpoint_all(&self) -> Result<()> {
        let mut first_error = None;
        for handle in self.read_map().values() {
            let mut collection = handle.write().unwrap_or_else(PoisonError::into_inner);
            if let Err(err) = collection.checkpoint() {
                tracing::error!(collection = %collection.name(), %err, "checkpoint failed");
                first_error.get_or_insert(err);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn read_map(&self) -> std::sync::RwLockReadGuard<'_, BTreeMap<String, CollectionHandle>> {
        self.collections
            .read()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn write_map(&self) -> std::sync::RwLockWriteGuard<'_, BTreeMap<String, CollectionHandle>> {
        self.collections
            .write()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

/// Whether `name` can name a collection (and so a directory, on any OS).
fn is_valid_name(name: &str) -> bool {
    (1..=MAX_NAME_LEN).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

#[cfg(test)]
mod tests {
    use nearby_core::Metric;
    use nearby_storage::Durability;

    use super::*;
    use crate::config::IndexConfig;

    fn options() -> EngineOptions {
        EngineOptions {
            durability: Durability::NoSync,
            ..EngineOptions::default()
        }
    }

    fn config(dim: usize) -> CollectionConfig {
        CollectionConfig {
            dim,
            metric: Metric::Cosine,
            index: IndexConfig::Flat,
        }
    }

    #[test]
    fn collections_persist_across_reopening() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path(), options()).unwrap();
        db.create_collection("b-docs", config(2)).unwrap();
        let a = db.create_collection("a_images", config(3)).unwrap();
        a.write().unwrap().upsert(&[(7, &[1.0, 2.0, 3.0])]).unwrap();
        drop((a, db));

        let db = Database::open(dir.path(), options()).unwrap();
        assert_eq!(db.list(), vec!["a_images", "b-docs"]);
        let a = db.collection("a_images").unwrap();
        let a = a.read().unwrap();
        assert_eq!(a.get(7), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(a.config().metric, Metric::Cosine);
    }

    #[test]
    fn names_are_validated_and_unique() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path(), options()).unwrap();
        for bad in ["", "a b", "../escape", "ü", &"x".repeat(65)] {
            assert!(
                matches!(
                    db.create_collection(bad, config(2)),
                    Err(Error::InvalidParameter(_))
                ),
                "{bad:?}"
            );
        }
        db.create_collection("docs", config(2)).unwrap();
        assert!(matches!(
            db.create_collection("docs", config(2)),
            Err(Error::CollectionExists(_))
        ));
    }

    #[test]
    fn dropping_removes_the_collection_and_its_files() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path(), options()).unwrap();
        db.create_collection("docs", config(2)).unwrap();
        db.drop_collection("docs").unwrap();
        assert_eq!(db.list(), Vec::<String>::new());
        assert!(!dir.path().join("collections/docs").exists());
        assert!(matches!(
            db.collection("docs"),
            Err(Error::CollectionNotFound(_))
        ));
        assert!(matches!(
            db.drop_collection("docs"),
            Err(Error::CollectionNotFound(_))
        ));
        db.create_collection("docs", config(4)).unwrap();
    }

    #[test]
    fn a_failed_create_leaves_nothing_behind() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path(), options()).unwrap();
        assert!(db.create_collection("bad", config(0)).is_err());
        assert!(!dir.path().join("collections/bad").exists());
        assert_eq!(db.list(), Vec::<String>::new());
    }

    #[test]
    fn stray_directories_are_ignored() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("collections/not-a-collection")).unwrap();
        fs::create_dir_all(dir.path().join("collections/bad name")).unwrap();
        let db = Database::open(dir.path(), options()).unwrap();
        assert_eq!(db.list(), Vec::<String>::new());
        db.create_collection("not-a-collection", config(2)).unwrap();
    }

    #[test]
    fn checkpoint_all_moves_every_log_into_a_segment() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path(), options()).unwrap();
        for name in ["a", "b"] {
            let c = db.create_collection(name, config(2)).unwrap();
            c.write().unwrap().upsert(&[(1, &[1.0, 1.0])]).unwrap();
        }
        db.checkpoint_all().unwrap();
        for name in ["a", "b"] {
            let info = db.collection(name).unwrap().read().unwrap().info();
            assert_eq!((info.wal_ops, info.segments), (0, 1));
        }
    }
}
