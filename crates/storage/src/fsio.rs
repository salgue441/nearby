//! Durable file replacement.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use nearby_core::Result;

/// Replaces `path` with `bytes` so that a crash leaves either the old file or
/// the new one, never a mix: the bytes go to a temporary file next to it,
/// which is synced and then renamed over `path`.
///
/// # Errors
///
/// If writing, syncing or renaming fails. A leftover temporary file is
/// harmless; [`crate::CollectionStore::open`] removes it.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    atomic_write_with(path, |w| Ok(w.write_all(bytes)?))
}

/// Like [`atomic_write`], but streams the contents: `write` fills a buffered
/// writer to the temporary file. For files too large to build in memory.
///
/// # Errors
///
/// If `write` fails, or writing, syncing or renaming does. `path` is left
/// untouched then.
pub fn atomic_write_with(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> Result<()>,
) -> Result<()> {
    let tmp = path.with_extension("tmp");
    let mut writer = BufWriter::new(File::create(&tmp)?);
    write(&mut writer)?;
    let file = writer
        .into_inner()
        .map_err(std::io::IntoInnerError::into_error)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&tmp, path)?;
    sync_dir(path.parent().unwrap_or_else(|| Path::new(".")))
}

/// Makes renames and new files in `dir` durable.
///
/// Unix needs the directory itself synced; Windows can't open a directory
/// as a file, and its renames are already durable once the files are synced.
#[cfg_attr(
    not(unix),
    expect(clippy::unnecessary_wraps, reason = "syncing can only fail on Unix")
)]
pub(crate) fn sync_dir(dir: &Path) -> Result<()> {
    #[cfg(unix)]
    File::open(dir)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_write_creates_and_replaces() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("MANIFEST");
        atomic_write(&path, b"one").unwrap();
        atomic_write(&path, b"two").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"two");
        assert!(
            !path.with_extension("tmp").exists(),
            "no temporary file is left"
        );
    }
}
