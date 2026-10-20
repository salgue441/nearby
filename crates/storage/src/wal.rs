//! The write-ahead log: every write is appended here before it is applied.
//!
//! A log is a sequence of records:
//!
//! ```text
//! ┌─────────────┬────────────┬──────────────────────────────────────────┐
//! │ len: u32    │ crc32: u32 │ payload (len bytes)                      │
//! └─────────────┴────────────┴──────────────────────────────────────────┘
//! payload: tag u8 (1 = upsert, 2 = delete, 3 = upsert with a payload), id u64,
//!          for an upsert: value count u32, then the f32 values,
//!          and for tag 3: payload length u32, then the payload bytes
//! ```
//!
//! A crash can cut the last record short. When a log is opened, replay stops
//! at the first record that is incomplete or fails its checksum, and the file
//! is truncated there, so later appends start from a clean end.

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use nearby_core::{Error, PointId, Result};

use crate::codec::{PutLe, Reader};
use crate::fsio::sync_dir;
use nearby_core::checksum::crc32;

const TAG_UPSERT: u8 = 1;
const TAG_DELETE: u8 = 2;
/// An upsert with a non-empty payload. Upserts without one keep tag 1, so
/// their records are the same as before payloads existed.
const TAG_UPSERT_PAYLOAD: u8 = 3;

/// Bytes before each record's payload: its length and checksum.
const HEADER_LEN: usize = 8;

/// One change to a collection's points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op<'a> {
    /// Inserts the point, or replaces it if the id exists.
    Upsert {
        /// The point's id.
        id: PointId,
        /// Its vector.
        vector: &'a [f32],
        /// Bytes stored with the point, empty for none. Storage doesn't
        /// interpret them.
        payload: &'a [u8],
    },
    /// Removes the point, if it exists.
    Delete {
        /// The point's id.
        id: PointId,
    },
}

/// When appended records are forced to disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Durability {
    /// Every write is synced before it returns: nothing acknowledged is lost
    /// in a crash or power failure.
    #[default]
    Sync,
    /// Writes reach the operating system but are not synced. Survives a
    /// process crash, not a power failure. Much faster for bulk loads.
    NoSync,
}

/// An open log file, positioned at its end.
#[derive(Debug)]
pub(crate) struct Wal {
    path: PathBuf,
    file: File,
    durability: Durability,
    /// Records in the log, including replayed ones.
    ops: usize,
}

impl Wal {
    /// Creates an empty log at `path`, which must not exist.
    pub(crate) fn create(path: &Path, durability: Durability) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .create_new(true)
            .open(path)?;
        file.sync_all()?;
        sync_dir(path.parent().unwrap_or_else(|| Path::new(".")))?;
        Ok(Self {
            path: path.to_owned(),
            file,
            durability,
            ops: 0,
        })
    }

    /// Opens the log at `path`, calls `apply` with every complete record in
    /// order, and truncates anything after the last one.
    ///
    /// `dim` is the length every upserted vector must have.
    pub(crate) fn open(
        path: &Path,
        dim: usize,
        durability: Durability,
        apply: impl FnMut(Op<'_>),
    ) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let (valid_len, ops) = replay(&bytes, dim, apply);
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;
        if valid_len < bytes.len() {
            tracing::warn!(
                path = %path.display(),
                dropped_bytes = bytes.len() - valid_len,
                "write-ahead log ends with an incomplete record; truncating it"
            );
            file.set_len(valid_len as u64)?;
            file.sync_all()?;
        }
        file.seek(SeekFrom::End(0))?;
        Ok(Self {
            path: path.to_owned(),
            file,
            durability,
            ops,
        })
    }

    /// Calls `apply` with every complete record of the log at `path`.
    pub(crate) fn read(path: &Path, dim: usize, apply: impl FnMut(Op<'_>)) -> Result<()> {
        let bytes = std::fs::read(path)?;
        replay(&bytes, dim, apply);
        Ok(())
    }

    /// Appends `ops` as one write, synced according to the log's durability.
    pub(crate) fn append(&mut self, ops: &[Op<'_>]) -> Result<()> {
        let mut buf = Vec::new();
        for op in ops {
            encode(op, &mut buf);
        }
        self.file.write_all(&buf)?;
        if self.durability == Durability::Sync {
            self.file.sync_data()?;
        }
        self.ops += ops.len();
        Ok(())
    }

    /// Forces every appended record to disk, whatever the durability.
    pub(crate) fn sync(&self) -> Result<()> {
        self.file.sync_data()?;
        Ok(())
    }

    /// Records in the log.
    pub(crate) fn ops(&self) -> usize {
        self.ops
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

/// Appends one record for `op` to `buf`.
fn encode(op: &Op<'_>, buf: &mut Vec<u8>) {
    let mut payload = Vec::new();
    match *op {
        Op::Upsert {
            id,
            vector,
            payload: bytes,
        } => {
            payload.put_u8(if bytes.is_empty() {
                TAG_UPSERT
            } else {
                TAG_UPSERT_PAYLOAD
            });
            payload.put_u64(id);
            payload.put_u32(u32::try_from(vector.len()).expect("vector length fits in u32"));
            payload.put_f32s(vector);
            if !bytes.is_empty() {
                payload.put_u32(u32::try_from(bytes.len()).expect("payload length fits in u32"));
                payload.extend_from_slice(bytes);
            }
        }
        Op::Delete { id } => {
            payload.put_u8(TAG_DELETE);
            payload.put_u64(id);
        }
    }
    buf.put_u32(u32::try_from(payload.len()).expect("record length fits in u32"));
    buf.put_u32(crc32(&payload));
    buf.extend_from_slice(&payload);
}

/// Calls `apply` for each valid record at the front of `bytes`. Returns how
/// many bytes those records span and how many there were.
fn replay(bytes: &[u8], dim: usize, mut apply: impl FnMut(Op<'_>)) -> (usize, usize) {
    let mut reader = Reader::new(bytes);
    let mut vector = vec![0.0; dim];
    let mut ops = 0;
    loop {
        let start = bytes.len() - reader.remaining();
        let Some(payload) = next_payload(&mut reader) else {
            return (start, ops);
        };
        match decode(payload, &mut vector) {
            Some(op) => apply(op),
            None => return (start, ops),
        }
        ops += 1;
    }
}

/// The next record's payload, if it is complete and its checksum matches.
fn next_payload<'a>(reader: &mut Reader<'a>) -> Option<&'a [u8]> {
    if reader.remaining() < HEADER_LEN {
        return None;
    }
    let len = reader.u32()? as usize;
    let crc = reader.u32()?;
    let payload = reader.take(len)?;
    (crc32(payload) == crc).then_some(payload)
}

/// Decodes a payload whose checksum matched. `vector` must have the
/// collection's dimension; an upsert of any other length is rejected.
fn decode<'v>(payload: &'v [u8], vector: &'v mut [f32]) -> Option<Op<'v>> {
    let mut r = Reader::new(payload);
    let tag = r.u8()?;
    let id = r.u64()?;
    let op = match tag {
        TAG_UPSERT | TAG_UPSERT_PAYLOAD => {
            if r.u32()? as usize != vector.len() {
                return None;
            }
            r.f32s_into(vector)?;
            let payload = if tag == TAG_UPSERT_PAYLOAD {
                let len = r.u32()? as usize;
                r.take(len)?
            } else {
                &[]
            };
            Op::Upsert {
                id,
                vector,
                payload,
            }
        }
        TAG_DELETE => Op::Delete { id },
        _ => return None,
    };
    (r.remaining() == 0).then_some(op)
}

/// Checks that every upserted vector has `dim` values.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] for the first one that doesn't.
pub(crate) fn check_dims(ops: &[Op<'_>], dim: usize) -> Result<()> {
    for op in ops {
        if let Op::Upsert { vector, .. } = op
            && vector.len() != dim
        {
            return Err(Error::DimensionMismatch {
                expected: dim,
                got: vector.len(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Owned copy of an op, so replayed ops can be collected.
    #[derive(Debug, PartialEq)]
    enum Owned {
        Upsert(PointId, Vec<f32>),
        Delete(PointId),
    }

    fn owned(op: Op<'_>) -> Owned {
        match op {
            Op::Upsert { id, vector, .. } => Owned::Upsert(id, vector.to_vec()),
            Op::Delete { id } => Owned::Delete(id),
        }
    }

    fn reopen(path: &Path) -> (Wal, Vec<Owned>) {
        let mut seen = Vec::new();
        let wal = Wal::open(path, 2, Durability::Sync, |op| seen.push(owned(op))).unwrap();
        (wal, seen)
    }

    #[test]
    fn appended_ops_replay_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");
        let mut wal = Wal::create(&path, Durability::Sync).unwrap();
        wal.append(&[
            Op::Upsert {
                id: 1,
                vector: &[1.0, 2.0],
                payload: &[],
            },
            Op::Delete { id: 9 },
        ])
        .unwrap();
        wal.append(&[Op::Upsert {
            id: 1,
            vector: &[3.0, 4.0],
            payload: &[],
        }])
        .unwrap();
        drop(wal);

        let (wal, seen) = reopen(&path);
        assert_eq!(
            seen,
            vec![
                Owned::Upsert(1, vec![1.0, 2.0]),
                Owned::Delete(9),
                Owned::Upsert(1, vec![3.0, 4.0]),
            ]
        );
        assert_eq!(wal.ops(), 3);
    }

    #[test]
    fn a_torn_tail_is_truncated_and_appends_continue() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");
        let mut wal = Wal::create(&path, Durability::NoSync).unwrap();
        wal.append(&[Op::Delete { id: 1 }, Op::Delete { id: 2 }])
            .unwrap();
        drop(wal);
        let full = std::fs::metadata(&path).unwrap().len();
        // Cut the second record short, as a crash mid-write would.
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(full - 3)
            .unwrap();

        let (mut wal, seen) = reopen(&path);
        assert_eq!(seen, vec![Owned::Delete(1)]);
        wal.append(&[Op::Delete { id: 3 }]).unwrap();
        drop(wal);
        let (_, seen) = reopen(&path);
        assert_eq!(seen, vec![Owned::Delete(1), Owned::Delete(3)]);
    }

    #[test]
    fn a_corrupted_record_ends_replay() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");
        let mut wal = Wal::create(&path, Durability::Sync).unwrap();
        for id in 0..3 {
            wal.append(&[Op::Delete { id }]).unwrap();
        }
        drop(wal);
        let mut bytes = std::fs::read(&path).unwrap();
        let record = bytes.len() / 3;
        bytes[record + HEADER_LEN + 2] ^= 0xFF; // inside the second payload
        std::fs::write(&path, &bytes).unwrap();

        let (wal, seen) = reopen(&path);
        assert_eq!(seen, vec![Owned::Delete(0)]);
        assert_eq!(wal.ops(), 1);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), record as u64);
    }

    #[test]
    fn upserts_of_the_wrong_dimension_are_not_replayed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");
        let mut wal = Wal::create(&path, Durability::Sync).unwrap();
        wal.append(&[Op::Upsert {
            id: 1,
            vector: &[1.0, 2.0, 3.0],
            payload: &[],
        }])
        .unwrap();
        drop(wal);
        let (_, seen) = reopen(&path);
        assert_eq!(seen, Vec::new(), "a 3-d vector in a 2-d log is corruption");
    }

    #[test]
    fn payloads_replay_with_their_upserts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");
        let mut wal = Wal::create(&path, Durability::Sync).unwrap();
        wal.append(&[
            Op::Upsert {
                id: 1,
                vector: &[1.0, 2.0],
                payload: b"tagged",
            },
            Op::Upsert {
                id: 2,
                vector: &[3.0, 4.0],
                payload: &[],
            },
        ])
        .unwrap();
        drop(wal);
        let mut payloads = Vec::new();
        Wal::read(&path, 2, |op| {
            if let Op::Upsert { id, payload, .. } = op {
                payloads.push((id, payload.to_vec()));
            }
        })
        .unwrap();
        assert_eq!(payloads, vec![(1, b"tagged".to_vec()), (2, Vec::new())]);
    }

    #[test]
    fn upserts_without_a_payload_keep_the_original_record() {
        let mut with = Vec::new();
        encode(
            &Op::Upsert {
                id: 1,
                vector: &[1.0],
                payload: b"x",
            },
            &mut with,
        );
        let mut without = Vec::new();
        encode(
            &Op::Upsert {
                id: 1,
                vector: &[1.0],
                payload: &[],
            },
            &mut without,
        );
        // Each record starts with its length and checksum; the tag follows.
        assert_eq!((with[8], without[8]), (TAG_UPSERT_PAYLOAD, TAG_UPSERT));
        assert_eq!(without.len(), 8 + 1 + 8 + 4 + 4, "no payload fields");
    }

    #[test]
    fn check_dims_names_the_bad_length() {
        let ops = [
            Op::Delete { id: 1 },
            Op::Upsert {
                id: 2,
                vector: &[1.0],
                payload: &[],
            },
        ];
        assert!(matches!(
            check_dims(&ops, 2),
            Err(Error::DimensionMismatch {
                expected: 2,
                got: 1
            })
        ));
        assert!(check_dims(&ops[..1], 2).is_ok());
    }

    #[test]
    fn create_refuses_to_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.log");
        Wal::create(&path, Durability::Sync).unwrap();
        assert!(Wal::create(&path, Durability::Sync).is_err());
    }
}
