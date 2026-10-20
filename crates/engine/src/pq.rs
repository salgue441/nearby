//! The PQ backend: exact search until there is enough data to train a
//! quantizer, product-quantized codes after.

use std::num::NonZeroUsize;
use std::path::Path;

use nearby_core::{Metric, PointId, Result, ScoredPoint};
use nearby_index::{FlatIndex, IvfPqIndex, IvfPqParams, PqIndex, ProductQuantizer, VectorIndex};

use crate::backend::IndexBackend;
use crate::config::PqConfig;
use crate::vectors::VectorFile;

/// A PQ index that trains itself once enough points have arrived.
///
/// A quantizer can't be trained before there is data, so points first go
/// into a [`FlatIndex`] and are searched exactly. When the
/// `training_points`-th point arrives, the quantizer (and, with `lists`, the
/// clusters) are trained on the first `training_points` points, in
/// insertion order, every point is encoded, and the flat index is dropped.
/// Training on a fixed prefix keeps the result independent of how writes
/// were batched.
///
/// The full vectors, needed for re-ranking and lookups, go to a
/// [`VectorFile`] on disk, so once trained only the codes stay in memory.
#[derive(Debug)]
pub(crate) struct PqBackend {
    metric: Metric,
    config: PqConfig,
    state: State,
    vectors: VectorFile,
}

#[derive(Debug)]
enum State {
    /// Not enough points to train on yet.
    Collecting(FlatIndex),
    Trained(Codes),
}

/// Trained codes: scanned in full, or grouped into clusters.
#[derive(Debug)]
enum Codes {
    Flat(PqIndex),
    Ivf(IvfPqIndex),
}

impl Codes {
    /// Trains on the first `config.training_points` vectors of `flat` and
    /// encodes all of them.
    fn train(
        flat: &FlatIndex,
        metric: Metric,
        config: &PqConfig,
        threads: NonZeroUsize,
    ) -> Result<Self> {
        let training: Vec<f32> = flat
            .iter()
            .take(config.training_points)
            .flat_map(|(_, vector)| vector)
            .copied()
            .collect();
        let mut codes = if config.lists == 0 {
            let quantizer =
                ProductQuantizer::train(flat.dim(), metric, &config.params(), &training, threads)?;
            Self::Flat(PqIndex::with_capacity(quantizer, flat.len()))
        } else {
            let params = IvfPqParams {
                pq: config.params(),
                lists: config.lists,
                probe: config.probe,
            };
            Self::Ivf(IvfPqIndex::train(
                flat.dim(),
                metric,
                &params,
                &training,
                threads,
            )?)
        };
        let points: Vec<(PointId, &[f32])> = flat.iter().collect();
        codes.add_batch(&points, threads)?;
        Ok(codes)
    }

    fn add_batch(&mut self, points: &[(PointId, &[f32])], threads: NonZeroUsize) -> Result<()> {
        match self {
            Self::Flat(pq) => pq.add_batch(points, threads),
            Self::Ivf(ivf) => ivf.add_batch(points, threads),
        }
    }

    fn index(&self) -> &dyn VectorIndex {
        match self {
            Self::Flat(pq) => pq,
            Self::Ivf(ivf) => ivf,
        }
    }
}

impl PqBackend {
    /// An empty backend for `dim`-dimensional vectors, spilling them to `dir`.
    ///
    /// # Errors
    ///
    /// [`nearby_core::Error::InvalidParameter`] if `config` doesn't fit `dim`,
    /// or an I/O error creating the vector file.
    pub(crate) fn new(dim: usize, metric: Metric, config: &PqConfig, dir: &Path) -> Result<Self> {
        config.validate(dim)?;
        Ok(Self {
            vectors: VectorFile::create(dir, dim)?,
            metric,
            config: config.clone(),
            state: State::Collecting(FlatIndex::new(dim, metric)),
        })
    }

    /// Whether the quantizer has been trained.
    #[cfg(test)]
    fn is_trained(&self) -> bool {
        matches!(self.state, State::Trained(_))
    }

    fn index(&self) -> &dyn VectorIndex {
        match &self.state {
            State::Collecting(flat) => flat,
            State::Trained(codes) => codes.index(),
        }
    }
}

impl VectorIndex for PqBackend {
    fn dim(&self) -> usize {
        self.index().dim()
    }

    fn len(&self) -> usize {
        self.index().len()
    }

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        keep: &dyn Fn(PointId) -> bool,
    ) -> Result<Vec<ScoredPoint>> {
        self.index().search_filtered(query, k, keep)
    }
}

impl IndexBackend for PqBackend {
    fn insert_batch(&mut self, points: &[(PointId, &[f32])], threads: NonZeroUsize) -> Result<()> {
        self.vectors
            .append(points.iter().map(|&(_, vector)| vector))?;
        match &mut self.state {
            State::Trained(codes) => codes.add_batch(points, threads)?,
            State::Collecting(flat) => {
                flat.insert_batch(points, threads)?;
                if flat.len() >= self.config.training_points {
                    let codes = Codes::train(flat, self.metric, &self.config, threads)?;
                    self.state = State::Trained(codes);
                }
            }
        }
        debug_assert_eq!(self.vectors.len(), self.len(), "one vector per slot");
        Ok(())
    }

    fn vector(&self, slot: u32) -> &[f32] {
        self.vectors.get(slot as usize)
    }

    fn rerank(&self) -> Option<NonZeroUsize> {
        let trained = matches!(self.state, State::Trained(_));
        trained.then(|| NonZeroUsize::new(self.config.rerank).expect("validated in new"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 8;

    fn config(training_points: usize) -> PqConfig {
        PqConfig {
            subspaces: 4,
            training_points,
            rerank: 3,
            lists: 0,
            probe: 1,
        }
    }

    /// `n` points with ids `0..n` on a pseudo-random but fixed layout.
    fn points(n: usize) -> Vec<f32> {
        let mut state = 0x9E37_79B9_u32;
        (0..n * DIM)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state % 1000) as f32 / 10.0
            })
            .collect()
    }

    fn insert(backend: &mut PqBackend, data: &[f32], batch: usize) {
        let all: Vec<(PointId, &[f32])> = data
            .as_chunks::<DIM>()
            .0
            .iter()
            .enumerate()
            .map(|(i, v)| (i as PointId, v.as_slice()))
            .collect();
        for chunk in all.chunks(batch) {
            backend.insert_batch(chunk, NonZeroUsize::MIN).unwrap();
        }
    }

    #[test]
    fn searches_are_exact_until_training() {
        let dir = tempfile::tempdir().unwrap();
        let data = points(299);
        let mut backend = PqBackend::new(DIM, Metric::L2, &config(300), dir.path()).unwrap();
        insert(&mut backend, &data, 100);
        assert!(!backend.is_trained());
        assert_eq!(backend.rerank(), None, "exact results need no re-ranking");
        let query = &data[5 * DIM..6 * DIM];
        assert_eq!(backend.search(query, 1).unwrap()[0].id, 5);
        assert_eq!(backend.search(query, 1).unwrap()[0].distance, 0.0);
    }

    #[test]
    fn the_quantizer_is_trained_once_enough_points_arrive() {
        let dir = tempfile::tempdir().unwrap();
        let data = points(1_000);
        let mut backend = PqBackend::new(DIM, Metric::L2, &config(300), dir.path()).unwrap();
        insert(&mut backend, &data, 1_000);
        assert!(backend.is_trained());
        assert_eq!(backend.len(), 1_000, "every point is encoded");
        assert_eq!(
            backend.vector(123),
            &data[123 * DIM..124 * DIM],
            "full vectors come back from disk"
        );
        assert_eq!(backend.rerank(), NonZeroUsize::new(3));
    }

    #[test]
    fn batching_does_not_change_the_trained_index() {
        let dir = tempfile::tempdir().unwrap();
        let data = points(1_000);
        let mut whole = PqBackend::new(DIM, Metric::L2, &config(300), dir.path()).unwrap();
        insert(&mut whole, &data, 1_000);
        let mut pieces = PqBackend::new(DIM, Metric::L2, &config(300), dir.path()).unwrap();
        insert(&mut pieces, &data, 7);
        for query in data.as_chunks::<DIM>().0.iter().step_by(50) {
            assert_eq!(
                whole.search(query, 10).unwrap(),
                pieces.search(query, 10).unwrap()
            );
        }
    }

    #[test]
    fn clusters_probed_in_full_match_a_full_scan() {
        let dir = tempfile::tempdir().unwrap();
        let data = points(1_000);
        let mut full = PqBackend::new(DIM, Metric::L2, &config(300), dir.path()).unwrap();
        insert(&mut full, &data, 1_000);
        let clustered = PqConfig {
            lists: 8,
            probe: 8,
            ..config(300)
        };
        let mut ivf = PqBackend::new(DIM, Metric::L2, &clustered, dir.path()).unwrap();
        insert(&mut ivf, &data, 250);
        assert!(matches!(ivf.state, State::Trained(Codes::Ivf(_))));
        for query in data.as_chunks::<DIM>().0.iter().step_by(50) {
            assert_eq!(
                full.search(query, 10).unwrap(),
                ivf.search(query, 10).unwrap()
            );
        }
    }

    #[test]
    fn bad_settings_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        assert!(PqBackend::new(DIM, Metric::L2, &config(10), dir.path()).is_err());
        let odd = PqConfig {
            subspaces: 3,
            ..config(300)
        };
        assert!(PqBackend::new(DIM, Metric::L2, &odd, dir.path()).is_err());
    }
}
