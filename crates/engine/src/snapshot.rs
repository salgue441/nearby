//! Snapshots: a collection's id map and index, saved so a restart can skip
//! rebuilding the index.
//!
//! A snapshot is a cache. It is tagged with the storage generation it was
//! taken at, right after a flush, and only used while storage still reports
//! that generation; then only the log written since needs replaying. A
//! missing, stale or damaged snapshot just means a full rebuild.
//!
//! ```text
//! magic "NBSNAP02", generation u64, slot count u64
//! for each slot: point id u64, live u8
//! payload count u64, then by ascending point id: id u64, length u32, payload
//! crc32 of the above u32
//! the index, in its own self-checking format (see `IndexBackend::save`)
//! ```
//!
//! A version 1 snapshot (`NBSNAP01`, from before payloads) counts as stale.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

use nearby_core::checksum::{CrcReader, CrcWriter};
use nearby_core::{Error, PointId, Result};

use crate::backend::{self, IndexBackend};
use crate::config::CollectionConfig;
use crate::payload::{self, Payload};
use crate::points::PointTable;

/// The snapshot's file name inside a collection directory.
const FILE_NAME: &str = "index.snapshot";

const MAGIC: &[u8; 8] = b"NBSNAP02";
const MAGIC_V1: &[u8; 8] = b"NBSNAP01";

fn path(dir: &Path) -> PathBuf {
    dir.join(FILE_NAME)
}

/// Saves `points` and `index` as the state at storage `generation`.
///
/// Returns `false`, writing nothing, for indexes that can't be saved (a
/// flat index rebuilds faster than it would load).
pub(crate) fn save(
    dir: &Path,
    generation: u64,
    points: &PointTable,
    index: &dyn IndexBackend,
) -> Result<bool> {
    if !index.can_save() {
        return Ok(false);
    }
    nearby_storage::atomic_write_with(&path(dir), |w| {
        let mut header = CrcWriter::new(&mut *w);
        header.write_all(MAGIC)?;
        header.write_all(&generation.to_le_bytes())?;
        header.write_all(&(points.slot_count() as u64).to_le_bytes())?;
        for (id, live) in points.slots() {
            header.write_all(&id.to_le_bytes())?;
            header.write_all(&[u8::from(live)])?;
        }
        let mut payloads: Vec<(PointId, Payload)> = points.payloads().collect();
        payloads.sort_unstable_by_key(|&(id, _)| id);
        header.write_all(&(payloads.len() as u64).to_le_bytes())?;
        for (id, payload) in payloads {
            let bytes = payload::encode(&payload)?;
            header.write_all(&id.to_le_bytes())?;
            header.write_all(&(bytes.len() as u32).to_le_bytes())?;
            header.write_all(&bytes)?;
        }
        let crc = header.crc();
        w.write_all(&crc.to_le_bytes())?;
        index.save(w)
    })?;
    Ok(true)
}

/// Loads the snapshot in `dir` if it was taken at storage `generation` and
/// fits `config`. Anything else, including damage, yields `None` (logged),
/// and the stale or damaged file is removed.
pub(crate) fn load(
    dir: &Path,
    config: &CollectionConfig,
    generation: u64,
) -> Option<(PointTable, Box<dyn IndexBackend>)> {
    let path = path(dir);
    if !path.is_file() {
        return None;
    }
    let outcome = File::open(&path)
        .map_err(Error::from)
        .and_then(|file| read(BufReader::new(file), dir, config, generation));
    match outcome {
        Ok(Some(state)) => return Some(state),
        Ok(None) => {
            tracing::info!(path = %path.display(), "snapshot is stale; rebuilding the index");
        }
        Err(err) => tracing::warn!(path = %path.display(), %err, "ignoring damaged snapshot"),
    }
    remove(dir);
    None
}

/// Deletes the snapshot in `dir`, if any. Failing to is only logged: a
/// stale snapshot is ignored anyway.
pub(crate) fn remove(dir: &Path) {
    let path = path(dir);
    if let Err(err) = std::fs::remove_file(&path)
        && err.kind() != std::io::ErrorKind::NotFound
    {
        tracing::warn!(path = %path.display(), %err, "cannot remove snapshot");
    }
}

/// Reads a snapshot. `Ok(None)` means it is valid but doesn't apply.
fn read(
    r: impl Read,
    dir: &Path,
    config: &CollectionConfig,
    generation: u64,
) -> Result<Option<(PointTable, Box<dyn IndexBackend>)>> {
    let corrupt = |reason: &str| Error::corruption(FILE_NAME, reason);
    let mut header = CrcReader::new(r);
    let mut magic = [0; 8];
    read_exact(&mut header, &mut magic)?;
    if &magic == MAGIC_V1 {
        return Ok(None);
    }
    if &magic != MAGIC {
        return Err(corrupt("not a snapshot"));
    }
    if read_u64(&mut header)? != generation {
        return Ok(None);
    }
    let count = usize::try_from(read_u64(&mut header)?).map_err(|_| corrupt("bad slot count"))?;
    let mut slots: Vec<(PointId, bool)> = Vec::new();
    slots
        .try_reserve_exact(count)
        .map_err(|_| corrupt("slot count too large"))?;
    for _ in 0..count {
        let id = read_u64(&mut header)?;
        let mut live = [0];
        read_exact(&mut header, &mut live)?;
        slots.push((id, live[0] != 0));
    }
    let payload_count = read_u64(&mut header)?;
    let mut payloads = HashMap::new();
    for _ in 0..payload_count {
        let id = read_u64(&mut header)?;
        let len = read_u32(&mut header)? as usize;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .map_err(|_| corrupt("payload too large"))?;
        bytes.resize(len, 0);
        read_exact(&mut header, &mut bytes)?;
        let payload = payload::decode(&bytes)?.ok_or_else(|| corrupt("empty payload"))?;
        payloads.insert(id, payload);
    }
    let computed = header.crc();
    let mut r = header.into_inner();
    if read_u32(&mut r)? != computed {
        return Err(corrupt("checksum mismatch"));
    }

    let Some(index) = backend::load(config, dir, &mut r, count).map_err(|err| match err {
        Error::Corruption { reason, .. } => corrupt(&reason),
        other => other,
    })?
    else {
        return Ok(None);
    };
    Ok(Some((PointTable::from_slots(&slots, payloads), index)))
}

fn read_exact(r: &mut impl Read, buf: &mut [u8]) -> Result<()> {
    r.read_exact(buf).map_err(|err| {
        if err.kind() == std::io::ErrorKind::UnexpectedEof {
            Error::corruption(FILE_NAME, "truncated")
        } else {
            Error::Io(err)
        }
    })
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut buf = [0; 8];
    read_exact(r, &mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut buf = [0; 4];
    read_exact(r, &mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use nearby_core::Metric;

    use super::*;
    use crate::backend::new_index;
    use crate::config::{HnswConfig, IndexConfig, PqConfig};
    use crate::payload::Value;

    fn config() -> CollectionConfig {
        CollectionConfig {
            dim: 2,
            metric: Metric::L2,
            index: IndexConfig::Hnsw(HnswConfig::default()),
        }
    }

    /// Five points, one replaced and one deleted: 6 slots, 4 live.
    fn state() -> (PointTable, Box<dyn IndexBackend>) {
        let mut points = PointTable::new();
        let mut index = new_index(&config(), 0, Path::new(".")).unwrap();
        let mut add = |points: &mut PointTable, id: PointId, x: f32| {
            let v = [x, x];
            let tag = Payload::from([("tag".to_owned(), Value::String(format!("{id}")))]);
            let slot = points.upsert(id, (id == 1).then_some(&tag));
            index
                .insert_batch(&[(u64::from(slot), &v)], NonZeroUsize::MIN)
                .unwrap();
        };
        for id in 0..5 {
            add(&mut points, id, id as f32);
        }
        add(&mut points, 2, 20.0);
        points.delete(4);
        (points, index)
    }

    #[test]
    fn saved_state_loads_back() {
        let dir = tempfile::tempdir().unwrap();
        let (points, index) = state();
        assert!(save(dir.path(), 7, &points, index.as_ref()).unwrap());

        let (loaded, loaded_index) = load(dir.path(), &config(), 7).unwrap();
        assert_eq!((loaded.len(), loaded.slot_count()), (4, 6));
        let slot = loaded.slot(2).unwrap();
        assert_eq!(loaded_index.vector(slot), &[20.0, 20.0]);
        assert_eq!(loaded.slot(4), None);
        assert!(
            loaded.payloads().eq(points.payloads()),
            "payloads come back"
        );
        assert!(loaded.payload(1).is_some());
        assert_eq!(
            loaded_index.search(&[1.0, 1.0], 6).unwrap(),
            index.search(&[1.0, 1.0], 6).unwrap()
        );
    }

    #[test]
    fn a_stale_snapshot_is_ignored_and_removed() {
        let dir = tempfile::tempdir().unwrap();
        let (points, index) = state();
        save(dir.path(), 7, &points, index.as_ref()).unwrap();
        assert!(load(dir.path(), &config(), 8).is_none());
        assert!(!path(dir.path()).exists());
    }

    #[test]
    fn a_damaged_snapshot_is_ignored_and_removed() {
        let dir = tempfile::tempdir().unwrap();
        let (points, index) = state();
        save(dir.path(), 7, &points, index.as_ref()).unwrap();
        let mut bytes = std::fs::read(path(dir.path())).unwrap();
        for at in [20, bytes.len() - 3] {
            bytes[at] ^= 0xFF;
            std::fs::write(path(dir.path()), &bytes).unwrap();
            assert!(load(dir.path(), &config(), 7).is_none(), "damage at {at}");
            bytes[at] ^= 0xFF;
        }
    }

    #[test]
    fn flat_indexes_are_not_saved() {
        let dir = tempfile::tempdir().unwrap();
        let flat = CollectionConfig {
            index: IndexConfig::Flat,
            ..config()
        };
        let index = new_index(&flat, 0, dir.path()).unwrap();
        assert!(!save(dir.path(), 1, &PointTable::new(), index.as_ref()).unwrap());
        assert!(!path(dir.path()).exists());
    }

    #[test]
    fn a_snapshot_for_another_config_is_not_used() {
        let dir = tempfile::tempdir().unwrap();
        let (points, index) = state();
        save(dir.path(), 7, &points, index.as_ref()).unwrap();
        let cosine = CollectionConfig {
            metric: Metric::Cosine,
            ..config()
        };
        assert!(load(dir.path(), &cosine, 7).is_none());
    }

    fn pq_config(lists: usize) -> CollectionConfig {
        CollectionConfig {
            dim: 4,
            metric: Metric::L2,
            index: IndexConfig::Pq(PqConfig {
                subspaces: 2,
                training_points: 300,
                rerank: 2,
                lists,
                probe: 3,
            }),
        }
    }

    /// `n` PQ points with ids `0..n`, the last one deleted.
    fn pq_state(
        config: &CollectionConfig,
        dir: &Path,
        n: u64,
    ) -> (PointTable, Box<dyn IndexBackend>) {
        let mut points = PointTable::new();
        let mut index = new_index(config, 0, dir).unwrap();
        let vectors: Vec<[f32; 4]> = (0..n)
            .map(|i| {
                let x = (i * 7919 % 1000) as f32;
                [x, (i % 13) as f32, x / 3.0, (i % 101) as f32]
            })
            .collect();
        let batch: Vec<(PointId, &[f32])> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (u64::from(points.upsert(i as PointId, None)), &v[..]))
            .collect();
        index.insert_batch(&batch, NonZeroUsize::MIN).unwrap();
        points.delete(n - 1);
        (points, index)
    }

    #[test]
    fn pq_backends_load_back_in_every_state() {
        // Collecting (too few points to train), codes, and clustered codes.
        for (lists, n) in [(0, 100), (0, 600), (8, 600)] {
            let dir = tempfile::tempdir().unwrap();
            let config = pq_config(lists);
            let (points, index) = pq_state(&config, dir.path(), n);
            assert!(save(dir.path(), 3, &points, index.as_ref()).unwrap());

            let (loaded, loaded_index) = load(dir.path(), &config, 3).unwrap();
            assert_eq!(
                (loaded.len(), loaded.slot_count()),
                (points.len(), points.slot_count())
            );
            assert_eq!(
                loaded_index.rerank(),
                index.rerank(),
                "lists {lists}, n {n}"
            );
            assert_eq!(loaded_index.vector(42), index.vector(42));
            for q in [[0.0; 4], [500.0, 6.0, 170.0, 50.0]] {
                assert_eq!(
                    loaded_index.search(&q, 10).unwrap(),
                    index.search(&q, 10).unwrap()
                );
            }
        }
    }

    #[test]
    fn a_pq_snapshot_with_other_clustering_is_not_used() {
        let dir = tempfile::tempdir().unwrap();
        let (points, index) = pq_state(&pq_config(8), dir.path(), 600);
        save(dir.path(), 3, &points, index.as_ref()).unwrap();
        assert!(load(dir.path(), &pq_config(0), 3).is_none());
    }
}
