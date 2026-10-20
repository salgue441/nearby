//! Saving an HNSW index to bytes and loading it back, without a rebuild.
//!
//! All values are little-endian:
//!
//! ```text
//! magic "NBHNSW01"
//! dim u32, metric u8 (0 L2, 1 cosine, 2 dot)
//! m u32, m0 u32, ef_construction u32, ef_search u32, seed u64
//! rng state u64, node count u64, entry point u64 (u64::MAX if empty), max level u32
//! vectors f32 × dim × count
//! norms f32 × count (cosine only)
//! external ids u64 × count
//! levels u8 × count
//! for each node, for each layer 0..=level: link count u16, links u32 × count
//! crc32 of everything above u32
//! ```
//!
//! Loading checks every structural invariant searches rely on, so a damaged
//! file is reported as [`Error::Corruption`] instead of panicking later.

use std::io::{Read, Write};
use std::sync::Mutex;

use nearby_core::{Metric, Result};

use super::graph::Graph;
use super::{Hnsw, HnswParams, MAX_LEVEL};
use crate::codec::{Decoder, Encoder};
use crate::rng::SplitMix64;
use crate::store::VectorStore;

const MAGIC: [u8; 8] = *b"NBHNSW01";

impl Hnsw {
    /// Writes the whole index, graph and vectors, to `w`, so
    /// [`Hnsw::read_from`] can restore it without a rebuild.
    ///
    /// Pass a buffered writer: the index is written in many small pieces.
    ///
    /// # Errors
    ///
    /// If writing fails.
    pub fn write_to(&self, w: impl Write) -> Result<()> {
        let mut e = Encoder::new(w);
        let params = &self.params;
        e.bytes(&MAGIC)?;
        e.u32(self.store.dim())?;
        e.metric(self.metric)?;
        for value in [
            params.m,
            params.m0,
            params.ef_construction,
            params.ef_search,
        ] {
            e.u32(value)?;
        }
        e.u64(params.seed)?;
        e.u64(self.rng.state())?;
        e.u64(self.ids.len() as u64)?;
        e.u64(self.entry_point.map_or(u64::MAX, u64::from))?;
        e.u32(self.max_level)?;

        e.f32s(self.store.as_slice())?;
        e.f32s(&self.norms)?;
        e.u64s(&self.ids)?;
        let count = self.ids.len() as u32;
        // Levels never exceed MAX_LEVEL (16), so each fits in a byte.
        let levels: Vec<u8> = (0..count).map(|n| self.graph.level(n) as u8).collect();
        e.bytes(&levels)?;
        for node in 0..count {
            for level in 0..=self.graph.level(node) {
                let links = self.graph.neighbors(node, level);
                // Lists hold at most m0 <= 1024 links.
                e.bytes(&(links.len() as u16).to_le_bytes())?;
                for link in links {
                    e.bytes(&link.to_le_bytes())?;
                }
            }
        }
        e.finish()
    }

    /// Restores an index written by [`Hnsw::write_to`].
    ///
    /// Pass a buffered reader: the index is read in many small pieces.
    ///
    /// # Errors
    ///
    /// [`Error::Corruption`] if the data is truncated, fails its checksum or
    /// describes an invalid graph; an I/O error if reading fails.
    ///
    /// [`Error::Corruption`]: nearby_core::Error::Corruption
    pub fn read_from(r: impl Read) -> Result<Self> {
        let mut d = Decoder::new(r, "HNSW index");
        d.magic(MAGIC)?;
        let dim = d.u32()?;
        let metric = d.metric()?;
        let params = HnswParams {
            m: d.u32()?,
            m0: d.u32()?,
            ef_construction: d.u32()?,
            ef_search: d.u32()?,
            seed: d.u64()?,
        };
        params.validate().map_err(|e| d.corrupt(&e.to_string()))?;
        if dim == 0 {
            return Err(d.corrupt("dimension is 0"));
        }
        let rng = SplitMix64::new(d.u64()?);
        let count = d.u64()?;
        if count > 1 << 32 {
            return Err(d.corrupt("more than 2^32 nodes"));
        }
        let count = count as usize;
        let entry = d.u64()?;
        let max_level = d.u32()?;

        let size = count
            .checked_mul(dim)
            .ok_or_else(|| d.corrupt("size overflow"))?;
        let vectors = d.f32s(size)?;
        let norms = d.f32s(if metric == Metric::Cosine { count } else { 0 })?;
        let ids = d.u64s(count)?;
        let levels = d.bytes(count)?;
        let graph = read_graph(&mut d, &params, &levels)?;
        let entry_point = match (count, entry) {
            (0, u64::MAX) => None,
            (0, _) => return Err(d.corrupt("an empty index has an entry point")),
            (_, e) if e < count as u64 => Some(e as u32),
            _ => return Err(d.corrupt("entry point out of range")),
        };
        let top = levels.iter().copied().max().map_or(0, usize::from);
        if max_level != top || entry_point.is_some_and(|e| graph.level(e) != top) {
            return Err(d.corrupt("entry point is not on the top layer"));
        }
        d.finish()?;

        Ok(Self {
            params,
            metric,
            store: VectorStore::from_vec(dim, vectors),
            norms,
            ids,
            graph,
            entry_point,
            max_level,
            rng,
            visited_pool: Mutex::new(Vec::new()),
        })
    }
}

/// Reads every node's links, checking each list's length and targets.
fn read_graph(d: &mut Decoder<impl Read>, params: &HnswParams, levels: &[u8]) -> Result<Graph> {
    let count = levels.len();
    let mut graph = Graph::with_capacity(params.m0, count);
    for &level in levels {
        let level = usize::from(level);
        if level > MAX_LEVEL {
            return Err(d.corrupt("node level above the maximum"));
        }
        graph.add_node(level);
    }
    let mut links = Vec::new();
    for node in 0..count as u32 {
        for level in 0..=graph.level(node) {
            let len = usize::from(u16::from_le_bytes(d.array()?));
            if len > params.max_links(level) {
                return Err(d.corrupt("neighbor list longer than allowed"));
            }
            links.clear();
            for _ in 0..len {
                let link = u32::from_le_bytes(d.array()?);
                if link as usize >= count || usize::from(levels[link as usize]) < level {
                    return Err(d.corrupt("link to a node not on that layer"));
                }
                links.push(link);
            }
            graph.set_neighbors(node, level, &links);
        }
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use nearby_core::{Error, PointId};

    use super::*;
    use crate::VectorIndex;

    fn index(metric: Metric, n: usize) -> Hnsw {
        let mut rng = SplitMix64::new(5);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..6).map(|_| rng.next_f64() as f32).collect())
            .collect();
        let batch: Vec<(PointId, &[f32])> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (1_000 + i as PointId, v.as_slice()))
            .collect();
        let mut index = Hnsw::new(6, metric, HnswParams::with_m(5)).unwrap();
        index
            .insert_batch(&batch, NonZeroUsize::new(4).unwrap())
            .unwrap();
        index
    }

    fn bytes(index: &Hnsw) -> Vec<u8> {
        let mut out = Vec::new();
        index.write_to(&mut out).unwrap();
        out
    }

    #[test]
    fn round_trips_exactly() {
        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            let original = index(metric, 800);
            let restored = Hnsw::read_from(bytes(&original).as_slice()).unwrap();
            assert_eq!(bytes(&restored), bytes(&original), "{metric:?}");
            assert_eq!(restored.max_level(), original.max_level());
            for q in [[0.1; 6], [0.9; 6], [0.3, 0.7, 0.2, 0.5, 0.5, 0.1]] {
                assert_eq!(
                    restored.search(&q, 10).unwrap(),
                    original.search(&q, 10).unwrap()
                );
            }
        }
    }

    #[test]
    fn a_restored_index_keeps_growing_like_the_original() {
        let mut original = index(Metric::L2, 300);
        let mut restored = Hnsw::read_from(bytes(&original).as_slice()).unwrap();
        for i in 0..50u16 {
            let v = [f32::from(i) / 50.0; 6];
            original.insert(u64::from(i), &v).unwrap();
            restored.insert(u64::from(i), &v).unwrap();
        }
        assert_eq!(
            bytes(&restored),
            bytes(&original),
            "same rng state, same graph"
        );
    }

    #[test]
    fn empty_indexes_round_trip() {
        let empty = Hnsw::new(3, Metric::L2, HnswParams::default()).unwrap();
        let restored = Hnsw::read_from(bytes(&empty).as_slice()).unwrap();
        assert!(restored.is_empty());
        assert_eq!(restored.search(&[0.0; 3], 5).unwrap(), Vec::new());
    }

    #[test]
    fn damage_is_detected() {
        let good = bytes(&index(Metric::L2, 200));
        let mut flipped = good.clone();
        flipped[good.len() / 2] ^= 0x40;
        let mut bad_magic = good.clone();
        bad_magic[0] = b'X';
        let cases: [(&[u8], &str); 4] = [
            (&good[..good.len() - 1], "truncated"),
            (&good[..10], "truncated"),
            (&flipped, ""),
            (&bad_magic, "unrecognized format"),
        ];
        for (data, expected) in cases {
            let err = Hnsw::read_from(data).unwrap_err();
            assert!(matches!(err, Error::Corruption { .. }), "{err}");
            assert!(err.to_string().contains(expected), "{err}");
        }
    }

    #[test]
    fn a_huge_length_field_is_an_error_not_an_abort() {
        let mut data = bytes(&index(Metric::L2, 10));
        // The node count sits after magic (8), dim (4), metric (1),
        // four u32 params (16), seed (8) and rng state (8).
        let offset = 8 + 4 + 1 + 16 + 8 + 8;
        data[offset..offset + 8].copy_from_slice(&(1u64 << 32).to_le_bytes());
        // dim is 6, so 2^32 nodes need 96 GiB: either the allocation fails
        // or the data runs out, and both are corruption.
        let err = Hnsw::read_from(data.as_slice()).unwrap_err();
        assert!(matches!(err, Error::Corruption { .. }), "{err}");
    }
}
