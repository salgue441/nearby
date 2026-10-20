//! Merging segments into one, streamed, so compaction runs in constant memory.
//!
//! A segment keeps its upserted and deleted ids sorted, so each one can be
//! walked in id order with a buffered reader per section. A k-way merge over
//! those walks visits every id once, keeps the entry from the newest segment
//! that has it, and drops deletes: the result is the oldest segment, with
//! nothing older left for a delete to remove.
//!
//! The output's sections can't be written in place until the number of live
//! points is known, so one pass over the merge streams vectors and payloads
//! to scratch files and keeps only ids and payload lengths (12 bytes per
//! point) in memory; the segment is then assembled from them, checksummed on
//! the way, and renamed into place.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use nearby_core::checksum::{CrcReader, CrcWriter};
use nearby_core::{Error, PointId, Result};

use crate::fsio::atomic_write_with;
use crate::segment::{MAGIC, MAGIC_V1};

/// One change read from a segment.
#[derive(Debug, PartialEq)]
enum Entry {
    Upsert { vector: Vec<f32>, payload: Vec<u8> },
    Delete,
}

/// Walks one segment's changes in ascending id order.
#[derive(Debug)]
struct Cursor {
    path: PathBuf,
    dim: usize,
    ids: BufReader<File>,
    vectors: BufReader<File>,
    /// `None` for a version 1 segment, which has no payloads.
    payloads: Option<(BufReader<File>, BufReader<File>)>,
    deletes: BufReader<File>,
    upserts_left: u64,
    deletes_left: u64,
    next_upsert: Option<PointId>,
    next_delete: Option<PointId>,
}

impl Cursor {
    /// Opens the segment at `path` after checking its checksum and header.
    fn open(path: &Path, dim: usize) -> Result<Self> {
        let corrupt = |reason: &str| Error::corruption(path, reason);
        let len = fs::metadata(path)?.len();
        verify_checksum(path, len)?;

        let mut header = BufReader::new(File::open(path)?);
        let mut magic = [0; 8];
        read_exact(&mut header, &mut magic, path)?;
        let has_payloads = match &magic {
            m if m == MAGIC => true,
            m if m == MAGIC_V1 => false,
            _ => return Err(corrupt("not a segment file")),
        };
        let file_dim = read_u32(&mut header, path)? as usize;
        if file_dim != dim {
            return Err(corrupt(&format!(
                "segment holds {file_dim}-d vectors, expected {dim}"
            )));
        }
        let upserts = read_u64(&mut header, path)?;
        let deletes = read_u64(&mut header, path)?;
        let payload_bytes = if has_payloads {
            read_u64(&mut header, path)?
        } else {
            0
        };
        let header_len: u64 = if has_payloads { 36 } else { 28 };
        let offsets = (|| {
            let vectors = header_len.checked_add(upserts.checked_mul(8)?)?;
            let lengths = vectors.checked_add(upserts.checked_mul(dim as u64 * 4)?)?;
            let payloads = lengths.checked_add(if has_payloads { upserts * 4 } else { 0 })?;
            let deletes_at = payloads.checked_add(payload_bytes)?;
            let end = deletes_at.checked_add(deletes.checked_mul(8)?)?;
            Some((vectors, lengths, payloads, deletes_at, end))
        })();
        let Some((vectors_at, lengths_at, payloads_at, deletes_at, end)) = offsets else {
            return Err(corrupt("length does not match the header"));
        };
        if end + 4 != len {
            return Err(corrupt("length does not match the header"));
        }

        let section = |offset: u64| -> Result<BufReader<File>> {
            let mut file = File::open(path)?;
            file.seek(SeekFrom::Start(offset))?;
            Ok(BufReader::new(file))
        };
        let mut cursor = Self {
            path: path.to_owned(),
            dim,
            ids: section(header_len)?,
            vectors: section(vectors_at)?,
            payloads: if has_payloads {
                Some((section(lengths_at)?, section(payloads_at)?))
            } else {
                None
            },
            deletes: section(deletes_at)?,
            upserts_left: upserts,
            deletes_left: deletes,
            next_upsert: None,
            next_delete: None,
        };
        cursor.next_upsert = cursor.read_upsert_id()?;
        cursor.next_delete = cursor.read_delete_id()?;
        Ok(cursor)
    }

    fn read_upsert_id(&mut self) -> Result<Option<PointId>> {
        if self.upserts_left == 0 {
            return Ok(None);
        }
        self.upserts_left -= 1;
        read_u64(&mut self.ids, &self.path).map(Some)
    }

    fn read_delete_id(&mut self) -> Result<Option<PointId>> {
        if self.deletes_left == 0 {
            return Ok(None);
        }
        self.deletes_left -= 1;
        read_u64(&mut self.deletes, &self.path).map(Some)
    }

    /// The smallest id not yet taken, if any.
    fn peek(&self) -> Option<PointId> {
        match (self.next_upsert, self.next_delete) {
            (Some(u), Some(d)) => Some(u.min(d)),
            (u, d) => u.or(d),
        }
    }

    /// Takes the change for [`Cursor::peek`]'s id.
    fn take(&mut self) -> Result<Entry> {
        if self.next_upsert.is_some() && self.next_upsert == self.peek() {
            let mut bytes = vec![0; self.dim * 4];
            read_exact(&mut self.vectors, &mut bytes, &self.path)?;
            let vector = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            let payload = match &mut self.payloads {
                Some((lengths, payloads)) => {
                    let len = read_u32(lengths, &self.path)? as usize;
                    let mut payload = vec![0; len];
                    read_exact(payloads, &mut payload, &self.path)?;
                    payload
                }
                None => Vec::new(),
            };
            self.next_upsert = self.read_upsert_id()?;
            Ok(Entry::Upsert { vector, payload })
        } else {
            self.next_delete = self.read_delete_id()?;
            Ok(Entry::Delete)
        }
    }
}

/// Merges `inputs`, oldest first, into a new segment at `output` holding
/// only the live points.
///
/// # Errors
///
/// [`Error::Corruption`] if an input is damaged; an I/O error if reading or
/// writing fails. `output` is only replaced once it is complete and synced.
pub(crate) fn merge(inputs: &[PathBuf], dim: usize, output: &Path) -> Result<()> {
    let scratch = Scratch {
        vectors: output.with_extension("vectors.tmp"),
        payloads: output.with_extension("payloads.tmp"),
    };
    let mut ids: Vec<PointId> = Vec::new();
    let mut lengths: Vec<u32> = Vec::new();
    let mut vectors = BufWriter::new(File::create(&scratch.vectors)?);
    let mut payloads = BufWriter::new(File::create(&scratch.payloads)?);
    for_each_live(inputs, dim, |id, vector, payload| {
        ids.push(id);
        lengths.push(u32::try_from(payload.len()).expect("payload length fits in u32"));
        for value in vector {
            vectors.write_all(&value.to_le_bytes())?;
        }
        payloads.write_all(payload)?;
        Ok(())
    })?;
    vectors.flush()?;
    payloads.flush()?;
    drop((vectors, payloads));

    let payload_bytes: u64 = lengths.iter().map(|&n| u64::from(n)).sum();
    atomic_write_with(output, |w| {
        let mut out = CrcWriter::new(&mut *w);
        out.write_all(MAGIC)?;
        out.write_all(
            &u32::try_from(dim)
                .expect("dimension fits in u32")
                .to_le_bytes(),
        )?;
        out.write_all(&(ids.len() as u64).to_le_bytes())?;
        out.write_all(&0u64.to_le_bytes())?;
        out.write_all(&payload_bytes.to_le_bytes())?;
        for id in &ids {
            out.write_all(&id.to_le_bytes())?;
        }
        std::io::copy(&mut BufReader::new(File::open(&scratch.vectors)?), &mut out)?;
        for len in &lengths {
            out.write_all(&len.to_le_bytes())?;
        }
        std::io::copy(
            &mut BufReader::new(File::open(&scratch.payloads)?),
            &mut out,
        )?;
        let crc = out.crc();
        w.write_all(&crc.to_le_bytes())?;
        Ok(())
    })
}

/// Scratch files for the sections a merge writes before it knows where they
/// go, deleted when the merge ends either way. Their `.tmp` extension also
/// marks them as leftovers for the next open, should the process die.
struct Scratch {
    vectors: PathBuf,
    payloads: PathBuf,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        for path in [&self.vectors, &self.payloads] {
            if let Err(err) = fs::remove_file(path)
                && err.kind() != std::io::ErrorKind::NotFound
            {
                tracing::warn!(path = %path.display(), %err, "cannot remove merge scratch file");
            }
        }
    }
}

/// Calls `live` with each id's newest change across `inputs` (oldest
/// first), in ascending id order, skipping ids whose newest change deletes
/// them.
fn for_each_live(
    inputs: &[PathBuf],
    dim: usize,
    mut live: impl FnMut(PointId, &[f32], &[u8]) -> Result<()>,
) -> Result<()> {
    let mut cursors = inputs
        .iter()
        .map(|path| Cursor::open(path, dim))
        .collect::<Result<Vec<_>>>()?;
    // Smallest id first; for one id, the newest segment (highest index) first.
    let mut heap: BinaryHeap<Reverse<(PointId, Reverse<usize>)>> = cursors
        .iter()
        .enumerate()
        .filter_map(|(i, c)| Some(Reverse((c.peek()?, Reverse(i)))))
        .collect();
    while let Some(Reverse((id, Reverse(newest)))) = heap.pop() {
        let entry = cursors[newest].take()?;
        if let Some(next) = cursors[newest].peek() {
            heap.push(Reverse((next, Reverse(newest))));
        }
        // Older segments' changes to this id are superseded: skip them.
        while let Some(&Reverse((other_id, Reverse(older)))) = heap.peek() {
            if other_id != id {
                break;
            }
            heap.pop();
            cursors[older].take()?;
            if let Some(next) = cursors[older].peek() {
                heap.push(Reverse((next, Reverse(older))));
            }
        }
        if let Entry::Upsert { vector, payload } = entry {
            live(id, &vector, &payload)?;
        }
    }
    Ok(())
}

/// Checks the segment's trailing checksum, reading it once front to back.
fn verify_checksum(path: &Path, len: u64) -> Result<()> {
    let corrupt = |reason: &str| Error::corruption(path, reason);
    let Some(body_len) = len.checked_sub(4) else {
        return Err(corrupt("file is too short"));
    };
    let mut file = File::open(path)?;
    let mut reader = CrcReader::new(BufReader::new(&mut file).take(body_len));
    std::io::copy(&mut reader, &mut std::io::sink())?;
    let computed = reader.crc();
    let mut stored = [0; 4];
    file.seek(SeekFrom::Start(body_len))?;
    file.read_exact(&mut stored)?;
    if computed == u32::from_le_bytes(stored) {
        Ok(())
    } else {
        Err(corrupt("checksum mismatch"))
    }
}

fn read_exact(r: &mut impl Read, buf: &mut [u8], path: &Path) -> Result<()> {
    r.read_exact(buf).map_err(|err| {
        if err.kind() == std::io::ErrorKind::UnexpectedEof {
            Error::corruption(path, "truncated")
        } else {
            Error::Io(err)
        }
    })
}

fn read_u32(r: &mut impl Read, path: &Path) -> Result<u32> {
    let mut buf = [0; 4];
    read_exact(r, &mut buf, path)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read, path: &Path) -> Result<u64> {
    let mut buf = [0; 8];
    read_exact(r, &mut buf, path)?;
    Ok(u64::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use nearby_core::checksum::crc32;

    use super::*;
    use crate::segment::{self, Changes};
    use crate::wal::Op;

    /// A change to write into a test segment.
    enum Change {
        /// Upserts the id with vector `[x, -x]` and the payload.
        Up(PointId, f32, &'static [u8]),
        Del(PointId),
    }
    use Change::{Del, Up};

    fn write(dir: &Path, name: &str, ops: &[Change]) -> PathBuf {
        let mut changes = Changes::default();
        for op in ops {
            match *op {
                Up(id, x, payload) => changes.apply(Op::Upsert {
                    id,
                    vector: &[x, -x],
                    payload,
                }),
                Del(id) => changes.apply(Op::Delete { id }),
            }
        }
        let path = dir.join(name);
        changes.write(&path, 2).unwrap();
        path
    }

    /// What reading `inputs` in order and keeping the net live points gives.
    fn expected(inputs: &[PathBuf]) -> Vec<String> {
        let mut changes = Changes::default();
        for path in inputs {
            segment::read(path, 2, |op| changes.apply(op)).unwrap();
        }
        changes.drop_deletes();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("in-memory.seg");
        changes.write(&path, 2).unwrap();
        read_ops(&path)
    }

    fn read_ops(path: &Path) -> Vec<String> {
        let mut ops = Vec::new();
        segment::read(path, 2, |op| ops.push(format!("{op:?}"))).unwrap();
        ops
    }

    #[test]
    fn merging_keeps_each_ids_newest_change() {
        let dir = tempfile::tempdir().unwrap();
        let inputs = vec![
            write(
                dir.path(),
                "a.seg",
                &[Up(1, 1.0, b""), Up(2, 2.0, b"two"), Up(3, 3.0, b"")],
            ),
            write(
                dir.path(),
                "b.seg",
                &[Up(2, 20.0, b""), Del(3), Up(5, 5.0, b"five")],
            ),
            write(dir.path(), "c.seg", &[Del(1), Up(4, 4.0, b""), Del(9)]),
        ];
        let out = dir.path().join("out.seg");
        merge(&inputs, 2, &out).unwrap();
        assert_eq!(read_ops(&out), expected(&inputs));
        assert_eq!(read_ops(&out).len(), 3, "ids 2, 4 and 5 are live");
    }

    #[test]
    fn a_merged_segment_matches_one_written_in_memory() {
        let dir = tempfile::tempdir().unwrap();
        let inputs: Vec<PathBuf> = (0..4)
            .map(|s| {
                let ops: Vec<Change> = (0..300u64)
                    .map(|i| {
                        let id = (i * 7 + s * 13) % 500;
                        if (i + s) % 5 == 0 {
                            Del(id)
                        } else {
                            Up(
                                id,
                                (s * 1000 + i) as f32,
                                if i % 3 == 0 { b"tagged" } else { b"" },
                            )
                        }
                    })
                    .collect();
                write(dir.path(), &format!("{s}.seg"), &ops)
            })
            .collect();
        let out = dir.path().join("out.seg");
        merge(&inputs, 2, &out).unwrap();

        let mut changes = Changes::default();
        for path in &inputs {
            segment::read(path, 2, |op| changes.apply(op)).unwrap();
        }
        changes.drop_deletes();
        let reference = dir.path().join("reference.seg");
        changes.write(&reference, 2).unwrap();
        let merged = fs::read(&out).unwrap();
        assert_eq!(merged, fs::read(&reference).unwrap(), "byte for byte");
        let (body, crc) = merged.split_last_chunk::<4>().unwrap();
        assert_eq!(crc32(body), u32::from_le_bytes(*crc));
    }

    #[test]
    fn a_damaged_input_aborts_the_merge() {
        let dir = tempfile::tempdir().unwrap();
        let good = write(dir.path(), "a.seg", &[Up(1, 1.0, b"")]);
        let bad = write(dir.path(), "b.seg", &[Up(2, 2.0, b"")]);
        let mut bytes = fs::read(&bad).unwrap();
        bytes[20] ^= 1;
        fs::write(&bad, bytes).unwrap();
        let out = dir.path().join("out.seg");
        let err = merge(&[good, bad], 2, &out).unwrap_err();
        assert!(err.to_string().contains("checksum"), "{err}");
        assert!(!out.exists());
    }

    #[test]
    fn version_1_inputs_merge_too() {
        let dir = tempfile::tempdir().unwrap();
        let mut v1 = Vec::new();
        v1.extend_from_slice(MAGIC_V1);
        v1.extend_from_slice(&2u32.to_le_bytes());
        v1.extend_from_slice(&1u64.to_le_bytes());
        v1.extend_from_slice(&0u64.to_le_bytes());
        v1.extend_from_slice(&7u64.to_le_bytes());
        for x in [7.0f32, -7.0] {
            v1.extend_from_slice(&x.to_le_bytes());
        }
        let crc = crc32(&v1);
        v1.extend_from_slice(&crc.to_le_bytes());
        let old = dir.path().join("old.seg");
        fs::write(&old, v1).unwrap();
        let new = write(dir.path(), "new.seg", &[Up(8, 8.0, b"p")]);
        let out = dir.path().join("out.seg");
        merge(&[old, new], 2, &out).unwrap();
        assert_eq!(read_ops(&out).len(), 2);
    }
}
