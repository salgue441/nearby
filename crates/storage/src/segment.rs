//! Segments: immutable files holding a batch of net changes.
//!
//! ```text
//! ┌──────────────────┬──────────┬──────────────┬──────────────┐
//! │ magic "NBSEG001" │ dim: u32 │ upserts: u64 │ deletes: u64 │
//! ├──────────────────┴──────────┴──────────────┴──────────────┤
//! │ upserted ids: u64 × upserts                               │
//! │ vectors: f32 × dim × upserts                              │
//! │ deleted ids: u64 × deletes                                │
//! ├───────────────────────────────────────────────────────────┤
//! │ crc32 of everything above: u32                            │
//! └───────────────────────────────────────────────────────────┘
//! ```
//!
//! Upserted and deleted ids are disjoint and each sorted, so a segment's
//! changes can be applied in any order, and the same changes always produce
//! the same bytes.

use std::collections::BTreeMap;
use std::path::Path;

use nearby_core::{Error, PointId, Result};

use crate::codec::{PutLe, Reader};
use crate::crc::crc32;
use crate::fsio::atomic_write;
use crate::wal::Op;

const MAGIC: &[u8; 8] = b"NBSEG001";

/// The net effect of a sequence of ops: for each id touched, its final vector,
/// or `None` if it ended deleted.
#[derive(Debug, Default)]
pub(crate) struct Changes {
    points: BTreeMap<PointId, Option<Vec<f32>>>,
}

impl Changes {
    /// Records `op`, replacing whatever was recorded for its id.
    pub(crate) fn apply(&mut self, op: Op<'_>) {
        match op {
            Op::Upsert { id, vector } => self.points.insert(id, Some(vector.to_vec())),
            Op::Delete { id } => self.points.insert(id, None),
        };
    }

    /// Forgets every delete. Used when the changes become the oldest segment,
    /// where there is nothing older left for a delete to remove.
    pub(crate) fn drop_deletes(&mut self) {
        self.points.retain(|_, vector| vector.is_some());
    }

    /// Writes the changes to a new segment file at `path`, atomically.
    pub(crate) fn write(&self, path: &Path, dim: usize) -> Result<()> {
        let upserts: Vec<(PointId, &[f32])> = self
            .points
            .iter()
            .filter_map(|(&id, v)| Some((id, v.as_deref()?)))
            .collect();
        let deletes: Vec<PointId> = self
            .points
            .iter()
            .filter_map(|(&id, v)| v.is_none().then_some(id))
            .collect();

        let mut buf = Vec::with_capacity(32 + upserts.len() * (8 + dim * 4) + deletes.len() * 8);
        buf.extend_from_slice(MAGIC);
        buf.put_u32(u32::try_from(dim).expect("dimension fits in u32"));
        buf.put_u64(upserts.len() as u64);
        buf.put_u64(deletes.len() as u64);
        for &(id, _) in &upserts {
            buf.put_u64(id);
        }
        for &(_, vector) in &upserts {
            buf.put_f32s(vector);
        }
        for &id in &deletes {
            buf.put_u64(id);
        }
        buf.put_u32(crc32(&buf));
        atomic_write(path, &buf)
    }
}

/// Calls `apply` with every change in the segment at `path`: upserts first,
/// then deletes.
///
/// # Errors
///
/// [`Error::Corruption`] if the file is truncated, fails its checksum, has
/// the wrong magic, or holds vectors of a dimension other than `dim`; an I/O
/// error if it can't be read.
pub(crate) fn read(path: &Path, dim: usize, mut apply: impl FnMut(Op<'_>)) -> Result<()> {
    let bytes = std::fs::read(path)?;
    let corrupt = |reason: &str| Error::corruption(path, reason);
    let Some((body, stored_crc)) = bytes.split_last_chunk::<4>() else {
        return Err(corrupt("file is too short"));
    };
    if crc32(body) != u32::from_le_bytes(*stored_crc) {
        return Err(corrupt("checksum mismatch"));
    }

    let mut r = Reader::new(body);
    if r.take(MAGIC.len()) != Some(MAGIC) {
        return Err(corrupt("not a segment file"));
    }
    let header = (|| Some((r.u32()? as usize, r.u64()?, r.u64()?)))();
    let Some((file_dim, upserts, deletes)) = header else {
        return Err(corrupt("truncated header"));
    };
    if file_dim != dim {
        return Err(corrupt(&format!(
            "segment holds {file_dim}-d vectors, expected {dim}"
        )));
    }
    let expected = upserts
        .checked_mul(8 + dim as u64 * 4)
        .and_then(|n| n.checked_add(deletes.checked_mul(8)?));
    if expected != Some(r.remaining() as u64) {
        return Err(corrupt("length does not match the header"));
    }

    // Sizes now match the header, so the reads below can't run out.
    let upserts = upserts as usize;
    let ids = r.take(upserts * 8).expect("length checked");
    let mut vectors = Reader::new(r.take(upserts * dim * 4).expect("length checked"));
    let mut vector = vec![0.0; dim];
    for id in ids.as_chunks::<8>().0 {
        vectors.f32s_into(&mut vector).expect("length checked");
        apply(Op::Upsert {
            id: u64::from_le_bytes(*id),
            vector: &vector,
        });
    }
    while let Some(id) = r.u64() {
        apply(Op::Delete { id });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(path: &Path, dim: usize) -> Result<Vec<String>> {
        let mut seen = Vec::new();
        read(path, dim, |op| seen.push(format!("{op:?}")))?;
        Ok(seen)
    }

    fn sample() -> Changes {
        let mut changes = Changes::default();
        changes.apply(Op::Upsert {
            id: 5,
            vector: &[1.0, 2.0],
        });
        changes.apply(Op::Delete { id: 3 });
        changes.apply(Op::Upsert {
            id: 1,
            vector: &[0.5, 0.5],
        });
        changes.apply(Op::Delete { id: 5 }); // the later op wins
        changes.apply(Op::Upsert {
            id: 3,
            vector: &[9.0, 9.0],
        });
        changes
    }

    #[test]
    fn changes_net_out_and_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seg.seg");
        sample().write(&path, 2).unwrap();
        assert_eq!(
            collect(&path, 2).unwrap(),
            vec![
                format!(
                    "{:?}",
                    Op::Upsert {
                        id: 1,
                        vector: &[0.5, 0.5]
                    }
                ),
                format!(
                    "{:?}",
                    Op::Upsert {
                        id: 3,
                        vector: &[9.0, 9.0]
                    }
                ),
                format!("{:?}", Op::Delete { id: 5 }),
            ]
        );
    }

    #[test]
    fn drop_deletes_keeps_only_upserts() {
        let mut changes = sample();
        changes.drop_deletes();
        assert_eq!(changes.points.len(), 2);
        assert!(changes.points.values().all(Option::is_some));
    }

    #[test]
    fn same_changes_write_identical_files() {
        let dir = tempfile::tempdir().unwrap();
        sample().write(&dir.path().join("a.seg"), 2).unwrap();
        sample().write(&dir.path().join("b.seg"), 2).unwrap();
        assert_eq!(
            std::fs::read(dir.path().join("a.seg")).unwrap(),
            std::fs::read(dir.path().join("b.seg")).unwrap()
        );
    }

    #[test]
    fn damage_is_reported_as_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seg.seg");
        sample().write(&path, 2).unwrap();
        let good = std::fs::read(&path).unwrap();

        let mut flipped = good.clone();
        flipped[30] ^= 1;
        let cases: [(&[u8], &str); 3] = [
            (&good[..good.len() - 1], "checksum"),
            (&flipped, "checksum"),
            (&good[..2], "too short"),
        ];
        for (bytes, expected) in cases {
            std::fs::write(&path, bytes).unwrap();
            let err = collect(&path, 2).unwrap_err();
            assert!(matches!(err, Error::Corruption { .. }), "{err}");
            assert!(err.to_string().contains(expected), "{err}");
        }

        std::fs::write(&path, &good).unwrap();
        let err = collect(&path, 3).unwrap_err().to_string();
        assert!(err.contains("2-d vectors, expected 3"), "{err}");
    }

    #[test]
    fn an_empty_segment_is_valid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seg.seg");
        Changes::default().write(&path, 4).unwrap();
        assert_eq!(collect(&path, 4).unwrap(), Vec::<String>::new());
    }
}
