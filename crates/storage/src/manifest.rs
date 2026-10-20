//! The manifest: which files make up a collection's current state.
//!
//! A small text file, replaced atomically whenever the set of files changes:
//!
//! ```text
//! nearby-manifest 1
//! dim 128
//! next 7
//! wal wal-000006.log
//! segment seg-000003.seg
//! segment seg-000005.seg
//! ```
//!
//! Segments are listed oldest first. Files in the directory that the manifest
//! doesn't name are leftovers from an interrupted flush or compaction.

use std::path::Path;

use nearby_core::{Error, Result};

use crate::fsio::atomic_write;

/// The manifest's file name inside a collection directory.
pub(crate) const FILE_NAME: &str = "MANIFEST";

const HEADER: &str = "nearby-manifest 1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Manifest {
    pub dim: usize,
    /// Number for the next file created, so names are never reused.
    pub next_file: u64,
    pub wal: String,
    /// Oldest first.
    pub segments: Vec<String>,
}

impl Manifest {
    /// A fresh name for a write-ahead log, reserving its number.
    pub(crate) fn new_wal_name(&mut self) -> String {
        let n = self.reserve();
        format!("wal-{n:06}.log")
    }

    /// A fresh name for a segment, reserving its number.
    pub(crate) fn new_segment_name(&mut self) -> String {
        let n = self.reserve();
        format!("seg-{n:06}.seg")
    }

    fn reserve(&mut self) -> u64 {
        let n = self.next_file;
        self.next_file += 1;
        n
    }

    /// Whether `name` is a file this store creates (logs, segments and
    /// temporary files), as opposed to files other code keeps alongside.
    pub(crate) fn is_store_file(name: &str) -> bool {
        let numbered = |prefix: &str, ext: &str| {
            name.strip_prefix(prefix)
                .and_then(|rest| rest.strip_suffix(ext))
                .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
        };
        numbered("wal-", ".log") || numbered("seg-", ".seg") || is_tmp(name)
    }

    /// Whether the manifest refers to `name`.
    pub(crate) fn references(&self, name: &str) -> bool {
        self.wal == name || self.segments.iter().any(|s| s == name)
    }

    pub(crate) fn write(&self, dir: &Path) -> Result<()> {
        let mut text = format!(
            "{HEADER}\ndim {}\nnext {}\nwal {}\n",
            self.dim, self.next_file, self.wal
        );
        text.extend(self.segments.iter().map(|s| format!("segment {s}\n")));
        atomic_write(&dir.join(FILE_NAME), text.as_bytes())
    }

    /// Reads the manifest in `dir`.
    ///
    /// # Errors
    ///
    /// An I/O error if it can't be read (`NotFound` if there is none), or
    /// [`Error::Corruption`] if it isn't a valid manifest.
    pub(crate) fn read(dir: &Path) -> Result<Self> {
        let path = dir.join(FILE_NAME);
        let text = std::fs::read_to_string(&path)?;
        Self::parse(&text).map_err(|reason| Error::corruption(path, reason))
    }

    fn parse(text: &str) -> std::result::Result<Self, String> {
        let mut lines = text.lines();
        if lines.next() != Some(HEADER) {
            return Err("not a manifest, or an unsupported version".to_owned());
        }
        let (mut dim, mut next_file, mut wal) = (None, None, None);
        let mut segments = Vec::new();
        for line in lines {
            let (key, value) = line
                .split_once(' ')
                .ok_or_else(|| format!("malformed line `{line}`"))?;
            let number = || value.parse().map_err(|_| format!("bad number in `{line}`"));
            match key {
                "dim" => dim = Some(number()?),
                "next" => next_file = Some(number()? as u64),
                "wal" => wal = Some(value.to_owned()),
                "segment" => segments.push(value.to_owned()),
                _ => return Err(format!("unknown key `{key}`")),
            }
        }
        let manifest = Self {
            dim: dim.ok_or("missing `dim`")?,
            next_file: next_file.ok_or("missing `next`")?,
            wal: wal.ok_or("missing `wal`")?,
            segments,
        };
        if manifest.dim == 0 {
            return Err("dimension is 0".to_owned());
        }
        let names = std::iter::once(&manifest.wal).chain(&manifest.segments);
        if let Some(bad) = names
            .into_iter()
            .find(|n| !Self::is_store_file(n) || is_tmp(n))
        {
            return Err(format!("bad file name `{bad}`"));
        }
        Ok(manifest)
    }
}

/// Whether `name` is a temporary file left by [`atomic_write`].
fn is_tmp(name: &str) -> bool {
    Path::new(name)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("tmp"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Manifest {
        Manifest {
            dim: 128,
            next_file: 7,
            wal: "wal-000006.log".to_owned(),
            segments: vec!["seg-000003.seg".to_owned(), "seg-000005.seg".to_owned()],
        }
    }

    #[test]
    fn round_trips_through_a_file() {
        let dir = tempfile::tempdir().unwrap();
        sample().write(dir.path()).unwrap();
        assert_eq!(Manifest::read(dir.path()).unwrap(), sample());
    }

    #[test]
    fn names_are_never_reused() {
        let mut m = sample();
        assert_eq!(m.new_segment_name(), "seg-000007.seg");
        assert_eq!(m.new_wal_name(), "wal-000008.log");
        assert_eq!(m.next_file, 9);
    }

    #[test]
    fn recognizes_its_own_files() {
        for name in ["wal-000001.log", "seg-123456.seg", "MANIFEST.tmp"] {
            assert!(Manifest::is_store_file(name), "{name}");
        }
        for name in ["collection.toml", "MANIFEST", "wal-.log", "seg-12a.seg"] {
            assert!(!Manifest::is_store_file(name), "{name}");
        }
        assert!(sample().references("seg-000005.seg"));
        assert!(!sample().references("seg-000004.seg"));
    }

    #[test]
    fn rejects_damaged_manifests() {
        let good = "nearby-manifest 1\ndim 4\nnext 2\nwal wal-000001.log\n";
        assert!(Manifest::parse(good).is_ok());
        let cases = [
            ("nearby-manifest 9\n", "version"),
            (
                "nearby-manifest 1\nnext 2\nwal wal-000001.log\n",
                "missing `dim`",
            ),
            ("nearby-manifest 1\ndim x\n", "bad number"),
            ("nearby-manifest 1\ncolor blue\n", "unknown key"),
            (
                "nearby-manifest 1\ndim 0\nnext 2\nwal wal-000001.log\n",
                "dimension is 0",
            ),
            (
                "nearby-manifest 1\ndim 4\nnext 2\nwal ../../etc/passwd\n",
                "bad file name",
            ),
        ];
        for (text, expected) in cases {
            let err = Manifest::parse(text).unwrap_err();
            assert!(err.contains(expected), "{text:?}: {err}");
        }
    }
}
