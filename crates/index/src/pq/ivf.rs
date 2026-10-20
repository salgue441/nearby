//! IVF-PQ: product-quantized codes grouped into an inverted file.

use std::num::NonZeroUsize;

use nearby_core::distance::normalize;
use nearby_core::{Error, Metric, PointId, Result, ScoredPoint, TopK, validate_vector};

use super::kmeans::{kmeans, nearest};
use super::{PqParams, ProductQuantizer};
use crate::VectorIndex;
use crate::parallel::parallel_map;

/// Vectors assigned and encoded per parallel task in [`IvfPqIndex::add_batch`].
const ENCODE_CHUNK: usize = 1024;

/// Settings for [`IvfPqIndex::train`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IvfPqParams {
    /// How every vector is compressed.
    pub pq: PqParams,
    /// Clusters (inverted lists) the vectors are split into. Each needs a
    /// few dozen training vectors to be placed well. Typical: around
    /// `sqrt(n)` for `n` vectors.
    pub lists: usize,
    /// Lists a search scans, nearest to the query first. More: higher
    /// recall, slower searches. Can be changed later with
    /// [`IvfPqIndex::set_probe`].
    pub probe: usize,
}

impl Default for IvfPqParams {
    fn default() -> Self {
        Self {
            pq: PqParams::default(),
            lists: 256,
            probe: 16,
        }
    }
}

impl IvfPqParams {
    /// Checks that the parameters can index `dim`-dimensional vectors.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if the PQ parameters don't fit `dim`, or
    /// `lists` or `probe` is 0.
    pub fn validate(&self, dim: usize) -> Result<()> {
        self.pq.validate(dim)?;
        if self.lists == 0 {
            return Err(Error::InvalidParameter(
                "lists must be at least 1".to_owned(),
            ));
        }
        if self.probe == 0 {
            return Err(Error::InvalidParameter(
                "probe must be at least 1".to_owned(),
            ));
        }
        Ok(())
    }
}

/// One cluster's codes and ids, in insertion order.
#[derive(Debug, Clone, Default)]
struct InvertedList {
    ids: Vec<PointId>,
    /// Code `i` is `codes[i * code_len..(i + 1) * code_len]`.
    codes: Vec<u8>,
}

/// Product-quantized vectors grouped by their nearest coarse centroid, so a
/// search scans only the lists nearest the query.
///
/// Training learns `lists` coarse centroids with k-means, then a
/// [`ProductQuantizer`]. Each vector is stored in its nearest centroid's
/// list as a PQ code of the vector itself, not of its offset from the
/// centroid: that keeps one distance table per query instead of one per
/// list, at a small cost in accuracy, which re-ranking recovers anyway.
///
/// A search ranks the centroids by the metric's distance to the query,
/// scans the nearest `probe` lists like [`PqIndex`](crate::PqIndex) scans
/// everything, and returns the best codes. With `probe == lists` it finds
/// exactly what a `PqIndex` would.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use nearby_core::Metric;
/// use nearby_index::{IvfPqIndex, IvfPqParams, PqParams, VectorIndex};
///
/// // 1000 points on a 40 x 25 grid, in 10 lists, 2-byte codes.
/// let points: Vec<f32> = (0..1_000u16)
///     .flat_map(|i| [f32::from(i % 40), f32::from(i / 40)])
///     .collect();
/// let params = IvfPqParams {
///     pq: PqParams::with_subspaces(2),
///     lists: 10,
///     probe: 3,
/// };
/// let mut index = IvfPqIndex::train(2, Metric::L2, &params, &points, NonZeroUsize::MIN)?;
/// for (id, vector) in points.chunks_exact(2).enumerate() {
///     index.add(id as u64, vector)?;
/// }
/// assert_eq!(index.search(&[3.0, 2.0], 1)?[0].id, 83);
/// # Ok::<(), nearby_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct IvfPqIndex {
    quantizer: ProductQuantizer,
    /// Coarse centroids back to back, `dim` values each. With
    /// [`Metric::Cosine`] they sit among normalized vectors.
    centroids: Vec<f32>,
    lists: Vec<InvertedList>,
    probe: usize,
    len: usize,
}

impl IvfPqIndex {
    /// Learns coarse centroids and a quantizer from `training`, which holds
    /// training vectors of `dim` values each, back to back. Training runs on
    /// up to `threads` threads, and its result doesn't depend on how many.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if `params` don't fit `dim`, or there are
    /// fewer training vectors than [`ProductQuantizer::MIN_TRAINING`] or
    /// than `lists`; [`Error::InvalidVector`] if a training vector contains
    /// NaN or infinite values.
    pub fn train(
        dim: usize,
        metric: Metric,
        params: &IvfPqParams,
        training: &[f32],
        threads: NonZeroUsize,
    ) -> Result<Self> {
        params.validate(dim)?;
        let quantizer = ProductQuantizer::train(dim, metric, &params.pq, training, threads)?;
        let n = training.len() / dim;
        if n < params.lists {
            return Err(Error::InvalidParameter(format!(
                "{} lists need at least as many training vectors, got {n}",
                params.lists
            )));
        }
        let mut training = training.to_vec();
        if metric == Metric::Cosine {
            training.chunks_exact_mut(dim).for_each(normalize);
        }
        let centroids = kmeans(
            &training,
            dim,
            params.lists,
            params.pq.iterations,
            params.pq.seed,
            threads,
        );
        Ok(Self {
            quantizer,
            centroids,
            lists: vec![InvertedList::default(); params.lists],
            probe: params.probe.min(params.lists),
            len: 0,
        })
    }

    /// The quantizer vectors are encoded with.
    pub fn quantizer(&self) -> &ProductQuantizer {
        &self.quantizer
    }

    /// Number of inverted lists.
    pub fn lists(&self) -> usize {
        self.lists.len()
    }

    /// Lists a search scans.
    pub fn probe(&self) -> usize {
        self.probe
    }

    /// Changes how many lists later searches scan, from 1 to
    /// [`IvfPqIndex::lists`]; values outside that range are clamped.
    pub fn set_probe(&mut self, probe: usize) {
        self.probe = probe.clamp(1, self.lists.len());
    }

    /// Encodes and adds one vector.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if `vector` has the wrong length, or
    /// [`Error::InvalidVector`] if it contains NaN or infinite values. On
    /// error the index is left unchanged.
    pub fn add(&mut self, id: PointId, vector: &[f32]) -> Result<()> {
        self.add_batch(&[(id, vector)], NonZeroUsize::MIN)
    }

    /// Assigns, encodes and adds many vectors on up to `threads` threads.
    /// The result is the same as adding them one at a time, in order.
    ///
    /// # Errors
    ///
    /// The same as [`IvfPqIndex::add`], for any of the vectors. On error
    /// nothing is added.
    pub fn add_batch(&mut self, points: &[(PointId, &[f32])], threads: NonZeroUsize) -> Result<()> {
        let dim = self.quantizer.dim();
        for (_, vector) in points {
            validate_vector(vector, dim)?;
        }
        let chunks: Vec<&[(PointId, &[f32])]> = points.chunks(ENCODE_CHUNK).collect();
        let encoded = parallel_map(chunks.len(), threads, |i| {
            chunks[i]
                .iter()
                .map(|&(_, vector)| {
                    let mut code = vec![0; self.quantizer.code_len()];
                    self.quantizer.encode_into(vector, &mut code);
                    (self.list_of(vector), code)
                })
                .collect::<Vec<_>>()
        });
        for (&(id, _), (list, code)) in points.iter().zip(encoded.into_iter().flatten()) {
            let list = &mut self.lists[list];
            list.ids.push(id);
            list.codes.extend_from_slice(&code);
        }
        self.len += points.len();
        Ok(())
    }

    /// The list a validated vector belongs to: its nearest coarse centroid.
    fn list_of(&self, vector: &[f32]) -> usize {
        let dim = self.quantizer.dim();
        if self.quantizer.metric() == Metric::Cosine {
            let mut unit = vector.to_vec();
            normalize(&mut unit);
            nearest(&self.centroids, dim, &unit).0 as usize
        } else {
            nearest(&self.centroids, dim, vector).0 as usize
        }
    }

    /// The `probe` lists nearest to a validated `query`, nearest first.
    fn lists_to_scan(&self, query: &[f32]) -> Vec<usize> {
        let metric = self.quantizer.metric();
        let mut ranked: Vec<(f32, usize)> = self
            .centroids
            .chunks_exact(self.quantizer.dim())
            .map(|centroid| metric.distance(query, centroid))
            .enumerate()
            .map(|(list, distance)| (distance, list))
            .collect();
        let probe = self.probe.min(ranked.len());
        if probe < ranked.len() {
            ranked.select_nth_unstable_by(probe, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            ranked.truncate(probe);
        }
        ranked.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        ranked.into_iter().map(|(_, list)| list).collect()
    }
}

impl VectorIndex for IvfPqIndex {
    fn dim(&self) -> usize {
        self.quantizer.dim()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        keep: &dyn Fn(PointId) -> bool,
    ) -> Result<Vec<ScoredPoint>> {
        validate_vector(query, self.quantizer.dim())?;
        let table = self.quantizer.distance_table(query);
        let code_len = self.quantizer.code_len();
        let mut top = TopK::new(k);
        for list in self.lists_to_scan(query) {
            let list = &self.lists[list];
            let codes = list.codes.chunks_exact(code_len);
            for (code, &id) in codes.zip(&list.ids).filter(|&(_, &id)| keep(id)) {
                top.push(id, table.distance(code));
            }
        }
        Ok(top.into_sorted())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PqIndex;
    use crate::rng::SplitMix64;

    const DIM: usize = 8;

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

    fn params(lists: usize, probe: usize) -> IvfPqParams {
        IvfPqParams {
            pq: PqParams {
                subspaces: 4,
                iterations: 4,
                ..PqParams::default()
            },
            lists,
            probe,
        }
    }

    fn index(metric: Metric, data: &[f32], lists: usize, probe: usize) -> IvfPqIndex {
        let threads = NonZeroUsize::new(4).unwrap();
        let mut index =
            IvfPqIndex::train(DIM, metric, &params(lists, probe), data, threads).unwrap();
        index.add_batch(&batch(data), threads).unwrap();
        index
    }

    #[test]
    fn probing_every_list_matches_a_flat_pq_scan() {
        let data = random(1_000, 1);
        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            let ivf = index(metric, &data, 16, 16);
            let mut pq = PqIndex::new(ivf.quantizer().clone());
            pq.add_batch(&batch(&data), NonZeroUsize::MIN).unwrap();
            for query in random(10, 2).as_chunks::<DIM>().0 {
                assert_eq!(
                    ivf.search(query, 10).unwrap(),
                    pq.search(query, 10).unwrap(),
                    "{metric:?}"
                );
            }
        }
    }

    #[test]
    fn every_vector_lands_in_exactly_one_list() {
        let data = random(2_000, 3);
        let ivf = index(Metric::L2, &data, 32, 4);
        assert_eq!(ivf.len(), 2_000);
        let mut ids: Vec<PointId> = ivf.lists.iter().flat_map(|l| l.ids.clone()).collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..2_000).collect::<Vec<_>>());
        assert!(ivf.lists.iter().filter(|l| !l.ids.is_empty()).count() > 16);
    }

    #[test]
    fn a_vector_is_found_in_its_own_list_with_one_probe() {
        let data = random(1_000, 4);
        let ivf = index(Metric::L2, &data, 16, 1);
        let found = batch(&data)
            .iter()
            .filter(|(id, v)| ivf.search(v, 10).unwrap().iter().any(|p| p.id == *id))
            .count();
        // Its own list is the nearest centroid's, unless PQ error pushes it
        // out of the top 10 there.
        assert!(found >= 900, "{found} of 1000");
    }

    #[test]
    fn batches_and_threads_do_not_change_the_index() {
        let data = random(1_500, 5);
        let one = {
            let mut index =
                IvfPqIndex::train(DIM, Metric::L2, &params(8, 2), &data, NonZeroUsize::MIN)
                    .unwrap();
            for (id, v) in batch(&data) {
                index.add(id, v).unwrap();
            }
            index
        };
        let many = index(Metric::L2, &data, 8, 2);
        assert_eq!(one.centroids, many.centroids);
        for (a, b) in one.lists.iter().zip(&many.lists) {
            assert_eq!((&a.ids, &a.codes), (&b.ids, &b.codes));
        }
    }

    #[test]
    fn probe_is_clamped_and_filters_apply() {
        let data = random(600, 6);
        let mut ivf = index(Metric::L2, &data, 8, 100);
        assert_eq!(ivf.probe(), 8);
        ivf.set_probe(0);
        assert_eq!(ivf.probe(), 1);
        ivf.set_probe(8);
        let even = ivf
            .search_filtered(&data[..DIM], 20, &|id| id % 2 == 0)
            .unwrap();
        assert_eq!(even.len(), 20);
        assert!(even.iter().all(|p| p.id % 2 == 0));
    }

    #[test]
    fn bad_settings_and_too_little_training_are_rejected() {
        let data = random(300, 7);
        for p in [params(0, 1), params(4, 0)] {
            assert!(IvfPqIndex::train(DIM, Metric::L2, &p, &data, NonZeroUsize::MIN).is_err());
        }
        let err = IvfPqIndex::train(DIM, Metric::L2, &params(400, 1), &data, NonZeroUsize::MIN)
            .unwrap_err();
        assert!(err.to_string().contains("400 lists"), "{err}");
    }
}
