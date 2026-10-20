//! A searchable collection of product-quantized vectors.

use std::num::NonZeroUsize;

use nearby_core::{PointId, Result, ScoredPoint, TopK, validate_vector};

use super::ProductQuantizer;
use crate::VectorIndex;
use crate::parallel::parallel_map;

/// Vectors encoded per parallel task in [`PqIndex::add_batch`].
const ENCODE_CHUNK: usize = 1024;

/// An index that stores only product-quantization codes.
///
/// Every vector costs [`ProductQuantizer::code_len`] bytes (16 for 128-d
/// vectors with the default 16 subspaces, 32x smaller than `f32`s), plus its
/// id. A search scans every code like [`FlatIndex`](crate::FlatIndex) does,
/// but each comparison is a few table lookups instead of a full distance
/// computation. Distances are approximate, and so is the ranking.
///
/// The original vectors are not kept, so results can't be re-ranked with
/// exact distances here.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use nearby_core::Metric;
/// use nearby_index::{PqIndex, PqParams, ProductQuantizer, VectorIndex};
///
/// // 1000 points on a 40 x 25 grid, quantized as 2 one-byte codes.
/// let points: Vec<f32> = (0..1_000u16)
///     .flat_map(|i| [f32::from(i % 40), f32::from(i / 40)])
///     .collect();
/// let params = PqParams::with_subspaces(2);
/// let pq = ProductQuantizer::train(2, Metric::L2, &params, &points, NonZeroUsize::MIN)?;
///
/// let mut index = PqIndex::new(pq);
/// for (id, vector) in points.chunks_exact(2).enumerate() {
///     index.add(id as u64, vector)?;
/// }
/// assert_eq!(index.search(&[3.0, 2.0], 1)?[0].id, 83);
/// # Ok::<(), nearby_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct PqIndex {
    quantizer: ProductQuantizer,
    /// Every code back to back: vector `i`'s code is
    /// `codes[i * code_len..(i + 1) * code_len]`.
    codes: Vec<u8>,
    /// `ids[i]` is the external id of vector `i`.
    ids: Vec<PointId>,
}

impl PqIndex {
    /// Creates an empty index that encodes vectors with `quantizer`.
    pub fn new(quantizer: ProductQuantizer) -> Self {
        Self::with_capacity(quantizer, 0)
    }

    /// Creates an empty index with room for `capacity` vectors.
    pub fn with_capacity(quantizer: ProductQuantizer, capacity: usize) -> Self {
        Self {
            codes: Vec::with_capacity(capacity * quantizer.code_len()),
            ids: Vec::with_capacity(capacity),
            quantizer,
        }
    }

    /// The quantizer vectors are encoded with.
    pub fn quantizer(&self) -> &ProductQuantizer {
        &self.quantizer
    }

    /// Bytes of codes stored per vector, ids not included.
    pub fn bytes_per_vector(&self) -> usize {
        self.quantizer.code_len()
    }

    /// Encodes and adds one vector.
    ///
    /// Duplicate ids are not detected here; the engine's id map handles that later.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if `vector` has the wrong length, or
    /// [`Error::InvalidVector`] if it contains NaN or infinite values.
    /// On error the index is left unchanged.
    ///
    /// [`Error::DimensionMismatch`]: nearby_core::Error::DimensionMismatch
    /// [`Error::InvalidVector`]: nearby_core::Error::InvalidVector
    pub fn add(&mut self, id: PointId, vector: &[f32]) -> Result<()> {
        self.add_batch(&[(id, vector)], NonZeroUsize::MIN)
    }

    /// Encodes and adds many vectors on up to `threads` threads. The result
    /// is the same as adding them one at a time, in order.
    ///
    /// # Errors
    ///
    /// The same as [`PqIndex::add`], for any of the vectors. On error
    /// nothing is added.
    pub fn add_batch(&mut self, points: &[(PointId, &[f32])], threads: NonZeroUsize) -> Result<()> {
        for (_, vector) in points {
            validate_vector(vector, self.quantizer.dim())?;
        }
        let chunks: Vec<&[(PointId, &[f32])]> = points.chunks(ENCODE_CHUNK).collect();
        let encoded = parallel_map(chunks.len(), threads, |i| self.encode_all(chunks[i]));
        for codes in encoded {
            self.codes.extend_from_slice(&codes);
        }
        self.ids.extend(points.iter().map(|&(id, _)| id));
        Ok(())
    }

    /// The codes of validated `points`, back to back.
    fn encode_all(&self, points: &[(PointId, &[f32])]) -> Vec<u8> {
        let code_len = self.quantizer.code_len();
        let mut codes = vec![0; points.len() * code_len];
        for ((_, vector), code) in points.iter().zip(codes.chunks_exact_mut(code_len)) {
            self.quantizer.encode_into(vector, code);
        }
        codes
    }
}

impl VectorIndex for PqIndex {
    fn dim(&self) -> usize {
        self.quantizer.dim()
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        keep: &dyn Fn(PointId) -> bool,
    ) -> Result<Vec<ScoredPoint>> {
        validate_vector(query, self.quantizer.dim())?;
        let table = self.quantizer.distance_table(query);
        let mut top = TopK::new(k);
        let codes = self.codes.chunks_exact(self.quantizer.code_len());
        for (code, &id) in codes.zip(&self.ids).filter(|&(_, &id)| keep(id)) {
            top.push(id, table.distance(code));
        }
        Ok(top.into_sorted())
    }
}

#[cfg(test)]
mod tests {
    use nearby_core::{Error, Metric};

    use super::*;
    use crate::FlatIndex;
    use crate::pq::PqParams;
    use crate::rng::SplitMix64;

    const DIM: usize = 16;

    fn random(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        (0..n * DIM).map(|_| rng.next_f64() as f32).collect()
    }

    fn batch(data: &[f32]) -> Vec<(PointId, &[f32])> {
        data.as_chunks::<DIM>()
            .0
            .iter()
            .enumerate()
            .map(|(i, v)| (i as PointId, v.as_slice()))
            .collect()
    }

    fn index(metric: Metric, data: &[f32], threads: usize) -> PqIndex {
        let params = PqParams {
            subspaces: 8,
            iterations: 4,
            ..PqParams::default()
        };
        let threads = NonZeroUsize::new(threads).unwrap();
        let pq = ProductQuantizer::train(DIM, metric, &params, data, threads).unwrap();
        let mut index = PqIndex::new(pq);
        index.add_batch(&batch(data), threads).unwrap();
        index
    }

    #[test]
    fn batch_and_single_adds_store_the_same_codes() {
        let data = random(1_500, 1);
        let batched = index(Metric::L2, &data, 8);
        let mut single = PqIndex::new(batched.quantizer().clone());
        for (id, v) in batch(&data) {
            single.add(id, v).unwrap();
        }
        assert_eq!(single.codes, batched.codes);
        assert_eq!(single.ids, batched.ids);
    }

    #[test]
    fn search_ranks_by_table_distance() {
        let data = random(1_000, 2);
        let index = index(Metric::L2, &data, 2);
        let query = &random(1, 3)[..];
        let results = index.search(query, 10).unwrap();
        assert_eq!(results.len(), 10);
        assert!(results.windows(2).all(|w| w[0].distance <= w[1].distance));
        let table = index.quantizer.distance_table(query);
        let code = |id: PointId| {
            let len = index.bytes_per_vector();
            &index.codes[id as usize * len..(id as usize + 1) * len]
        };
        for p in &results {
            assert_eq!(p.distance, table.distance(code(p.id)));
        }
    }

    #[test]
    fn recall_is_far_better_than_chance() {
        let data = random(1_000, 4);
        let queries = random(30, 5);
        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            let pq = index(metric, &data, 4);
            let mut flat = FlatIndex::new(DIM, metric);
            for (id, v) in batch(&data) {
                flat.add(id, v).unwrap();
            }
            // How often the exact nearest neighbor is in PQ's top 10.
            let hits = queries
                .as_chunks::<DIM>()
                .0
                .iter()
                .map(<[f32; DIM]>::as_slice)
                .filter(|q| {
                    let truth = flat.search(q, 1).unwrap()[0].id;
                    pq.search(q, 10).unwrap().iter().any(|p| p.id == truth)
                })
                .count();
            assert!(hits >= 24, "{metric:?}: {hits}/30 (chance: 1 in 100)");
        }
    }

    #[test]
    fn rejected_batches_add_nothing() {
        let data = random(500, 6);
        let mut index = index(Metric::L2, &data, 1);
        let mut bad = random(10, 7);
        bad[25] = f32::NAN;
        assert!(matches!(
            index.add_batch(&batch(&bad), NonZeroUsize::MIN),
            Err(Error::InvalidVector)
        ));
        assert!(matches!(
            index.add(1, &[0.0; 3]),
            Err(Error::DimensionMismatch { .. })
        ));
        assert_eq!(index.len(), 500);
        assert_eq!(index.codes.len(), 500 * index.bytes_per_vector());
    }

    #[test]
    fn empty_index_and_k_zero_return_nothing() {
        let data = random(300, 8);
        let empty = PqIndex::new(index(Metric::L2, &data, 1).quantizer.clone());
        assert_eq!(empty.search(&data[..DIM], 5).unwrap(), Vec::new());
        assert!(empty.is_empty());
        let full = index(Metric::L2, &data, 1);
        assert_eq!(full.search(&data[..DIM], 0).unwrap(), Vec::new());
    }
}
