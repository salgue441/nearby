//! A spill file for full-precision vectors that don't need to stay in RAM.

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use memmap2::Mmap;
use nearby_core::Result;

/// File names start with this; anything else in the directory is left alone.
const PREFIX: &str = "vectors-";
/// Not `.tmp`: storage deletes `*.tmp` files it finds as its own leftovers.
const SUFFIX: &str = ".spill";

/// Vectors appended to a memory-mapped file in a collection directory, so a
/// compressed index can keep only its codes in RAM.
///
/// Reading a vector is a memory access into the map. Pages come from the OS
/// page cache, which holds what fits and evicts what doesn't, so the process
/// itself never owns the vectors.
///
/// The file is scratch space, not storage: the write-ahead log and segments
/// stay the source of truth. Values are stored in native byte order, since
/// the file never outlives this process; it is deleted on drop, and
/// leftovers from a crash are deleted by [`VectorFile::remove_leftovers`].
#[derive(Debug)]
pub(crate) struct VectorFile {
    path: PathBuf,
    dim: usize,
    len: usize,
    writer: BufWriter<File>,
    /// The handle `map` is created from.
    file: File,
    /// The first `len` vectors. `None` while the file is empty, since an
    /// empty file can't be mapped everywhere.
    map: Option<Mmap>,
}

impl VectorFile {
    /// Creates an empty file in `dir` for `dim`-dimensional vectors.
    pub(crate) fn create(dir: &Path, dim: usize) -> Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!("{PREFIX}{}-{n}{SUFFIX}", std::process::id()));
        let writer = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let file = File::open(&path)?;
        Ok(Self {
            path,
            dim,
            len: 0,
            writer: BufWriter::new(writer),
            file,
            map: None,
        })
    }

    /// Deletes files left in `dir` by a process that didn't shut down
    /// cleanly. Call before creating any file there.
    pub(crate) fn remove_leftovers(dir: &Path) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(PREFIX) && name.ends_with(SUFFIX) {
                fs::remove_file(entry.path())?;
            }
        }
        Ok(())
    }

    /// Number of vectors in the file.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Appends `vectors`, which callers have validated, and maps them.
    pub(crate) fn append<'a>(
        &mut self,
        vectors: impl IntoIterator<Item = &'a [f32]>,
    ) -> Result<()> {
        for vector in vectors {
            debug_assert_eq!(vector.len(), self.dim, "caller validates vectors");
            for value in vector {
                self.writer.write_all(&value.to_ne_bytes())?;
            }
            self.len += 1;
        }
        self.writer.flush()?;
        if self.len > 0 {
            // SAFETY: the file is private to this process: its name is
            // unique, and it is deleted on drop. It is only ever appended to,
            // past the end of any existing map, so mapped bytes never change
            // while a map is alive.
            self.map = Some(unsafe { Mmap::map(&self.file)? });
        }
        Ok(())
    }

    /// The `index`-th vector appended.
    ///
    /// # Panics
    ///
    /// If `index` is out of range.
    pub(crate) fn get(&self, index: usize) -> &[f32] {
        assert!(index < self.len, "vector {index} of {}", self.len);
        let width = self.dim * size_of::<f32>();
        let map = self.map.as_ref().expect("a non-empty file is mapped");
        let bytes = &map[index * width..(index + 1) * width];
        // SAFETY: every bit pattern is a valid `f32`, and the bytes were
        // written from `f32`s in native byte order.
        let (head, floats, tail) = unsafe { bytes.align_to::<f32>() };
        // Maps start on a page boundary and every vector at a multiple of 4.
        assert!(head.is_empty() && tail.is_empty(), "vectors are aligned");
        floats
    }
}

impl Drop for VectorFile {
    fn drop(&mut self) {
        // Unmap first: some platforms refuse to delete a mapped file.
        self.map = None;
        if let Err(err) = fs::remove_file(&self.path) {
            tracing::warn!(path = %self.path.display(), %err, "cannot remove vector file");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nearby_storage::{CollectionStore, Op, StoreOptions};

    #[test]
    fn appended_vectors_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let mut file = VectorFile::create(dir.path(), 3).unwrap();
        file.append([&[1.0, 2.0, 3.0][..], &[4.0, 5.0, 6.0]])
            .unwrap();
        file.append([&[7.0, 8.0, 9.0][..]]).unwrap();
        assert_eq!(file.len(), 3);
        assert_eq!(file.get(1), &[4.0, 5.0, 6.0]);
        assert_eq!(file.get(2), &[7.0, 8.0, 9.0]);
    }

    #[test]
    fn empty_batches_are_fine() {
        let dir = tempfile::tempdir().unwrap();
        let mut file = VectorFile::create(dir.path(), 2).unwrap();
        file.append([]).unwrap();
        assert_eq!(file.len(), 0);
        file.append([&[1.0, 2.0][..]]).unwrap();
        file.append([]).unwrap();
        assert_eq!(file.get(0), &[1.0, 2.0]);
    }

    #[test]
    fn files_are_deleted_on_drop_and_leftovers_on_request() {
        let dir = tempfile::tempdir().unwrap();
        let mut file = VectorFile::create(dir.path(), 1).unwrap();
        file.append([&[1.0][..]]).unwrap();
        let second = VectorFile::create(dir.path(), 1).unwrap();
        assert_ne!(file.path, second.path, "names never collide");
        drop(file);
        assert_eq!(
            fs::read_dir(dir.path()).unwrap().count(),
            1,
            "only the second"
        );

        drop(second);
        // As if a crashed process had left its file behind.
        fs::write(dir.path().join("vectors-1-0.spill"), [0; 4]).unwrap();
        fs::write(dir.path().join("MANIFEST"), "keep").unwrap();
        VectorFile::remove_leftovers(dir.path()).unwrap();
        let left: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(left, vec!["MANIFEST"]);
    }

    #[test]
    fn storage_leaves_spill_files_alone() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = CollectionStore::create(dir.path(), 1, StoreOptions::default()).unwrap();
        let mut file = VectorFile::create(dir.path(), 1).unwrap();
        file.append([&[1.0][..]]).unwrap();
        // A flush commits a new manifest and sweeps files storage owns.
        store
            .write(&[Op::Upsert {
                id: 1,
                vector: &[1.0],
                payload: &[],
            }])
            .unwrap();
        store.flush().unwrap();
        assert!(file.path.exists());
    }

    #[test]
    #[should_panic(expected = "vector 1 of 1")]
    fn reading_past_the_end_panics() {
        let dir = tempfile::tempdir().unwrap();
        let mut file = VectorFile::create(dir.path(), 1).unwrap();
        file.append([&[1.0][..]]).unwrap();
        let _ = file.get(1);
    }
}
