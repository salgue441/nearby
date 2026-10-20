//! Learning codebooks, encoding vectors, and building distance tables.

use std::num::NonZeroUsize;

use nearby_core::distance::{dot, l2_squared, normalize};
use nearby_core::{Error, Metric, Result, validate_vector};

use super::kmeans::{kmeans, nearest};
use super::{CENTROIDS, PqParams};
use crate::parallel::parallel_map;

/// Compresses vectors into `subspaces`-byte codes, one byte per slice.
///
/// Each subspace has its own codebook of 256 centroids, learned from
/// training data by [`ProductQuantizer::train`]. A trained quantizer is
/// immutable and can encode any number of vectors.
///
/// With [`Metric::Cosine`], vectors are normalized before training and
/// encoding, so the codes approximate directions; with the other metrics
/// they approximate the vectors themselves.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use nearby_core::Metric;
/// use nearby_index::{PqParams, ProductQuantizer};
///
/// // 512 training vectors of dimension 4, quantized as 2 one-byte codes.
/// let training: Vec<f32> = (0..2_048u16).map(|i| f32::from(i % 97)).collect();
/// let params = PqParams::with_subspaces(2);
/// let pq = ProductQuantizer::train(4, Metric::L2, &params, &training, NonZeroUsize::MIN)?;
///
/// let codes = pq.encode(&[1.0, 2.0, 3.0, 4.0])?;
/// assert_eq!(codes.len(), 2);
/// assert_eq!(pq.decode(&codes).len(), 4);
/// # Ok::<(), nearby_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    dim: usize,
    metric: Metric,
    subspaces: usize,
    /// Every codebook back to back: centroid `c` of subspace `s` is the
    /// `sub_dim()` values starting at `(s * CENTROIDS + c) * sub_dim()`.
    codebooks: Vec<f32>,
}

impl ProductQuantizer {
    /// Learns one codebook per subspace from `training`, which holds
    /// training vectors of `dim` values each, back to back.
    ///
    /// Subspaces are trained independently on up to `threads` threads.
    /// The result depends only on the inputs and [`PqParams::seed`], never on
    /// `threads`. Tens of thousands of training vectors are usually plenty;
    /// more mostly cost training time.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if `params` don't fit `dim`,
    /// `training.len()` is not a multiple of `dim`, or there are fewer than
    /// 256 training vectors; [`Error::InvalidVector`] if a training vector
    /// contains NaN or infinite values.
    pub fn train(
        dim: usize,
        metric: Metric,
        params: &PqParams,
        training: &[f32],
        threads: NonZeroUsize,
    ) -> Result<Self> {
        params.validate(dim)?;
        if !training.len().is_multiple_of(dim) {
            return Err(Error::InvalidParameter(format!(
                "training data holds {} values, not a multiple of the dimension ({dim})",
                training.len()
            )));
        }
        let n = training.len() / dim;
        if n < CENTROIDS {
            return Err(Error::InvalidParameter(format!(
                "training needs at least {CENTROIDS} vectors, got {n}"
            )));
        }
        for vector in training.chunks_exact(dim) {
            validate_vector(vector, dim)?;
        }

        let mut training = training.to_vec();
        if metric == Metric::Cosine {
            training.chunks_exact_mut(dim).for_each(normalize);
        }
        let sub_dim = dim / params.subspaces;
        let codebooks = parallel_map(params.subspaces, threads, |s| {
            let slices: Vec<f32> = training
                .chunks_exact(dim)
                .flat_map(|v| &v[s * sub_dim..(s + 1) * sub_dim])
                .copied()
                .collect();
            let seed = params.seed.wrapping_add(s as u64);
            kmeans(&slices, sub_dim, CENTROIDS, params.iterations, seed)
        });
        Ok(Self {
            dim,
            metric,
            subspaces: params.subspaces,
            codebooks: codebooks.concat(),
        })
    }

    /// Number of values in each vector.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The metric this quantizer was trained for.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Length of every code in bytes: one per subspace.
    pub fn code_len(&self) -> usize {
        self.subspaces
    }

    fn sub_dim(&self) -> usize {
        self.dim / self.subspaces
    }

    /// Subspace `s`'s 256 centroids, back to back.
    fn codebook(&self, s: usize) -> &[f32] {
        let len = CENTROIDS * self.sub_dim();
        &self.codebooks[s * len..(s + 1) * len]
    }

    /// Encodes `vector`: each slice becomes the index of its nearest centroid.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if `vector` has the wrong length, or
    /// [`Error::InvalidVector`] if it contains NaN or infinite values.
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        validate_vector(vector, self.dim)?;
        let mut code = vec![0; self.code_len()];
        self.encode_into(vector, &mut code);
        Ok(code)
    }

    /// [`ProductQuantizer::encode`] for a vector the caller has validated,
    /// writing into `code`, which must be [`ProductQuantizer::code_len`] long.
    pub(super) fn encode_into(&self, vector: &[f32], code: &mut [u8]) {
        let mut normalized;
        let vector = if self.metric == Metric::Cosine {
            normalized = vector.to_vec();
            normalize(&mut normalized);
            &normalized
        } else {
            vector
        };
        let sub_dim = self.sub_dim();
        for (s, (byte, slice)) in code
            .iter_mut()
            .zip(vector.chunks_exact(sub_dim))
            .enumerate()
        {
            // `nearest` returns an index below CENTROIDS (256), so it fits in a byte.
            *byte = nearest(self.codebook(s), sub_dim, slice).0 as u8;
        }
    }

    /// The vector a code stands for: each byte's centroid, concatenated.
    /// With [`Metric::Cosine`] that approximates the normalized vector.
    ///
    /// # Panics
    ///
    /// If `code` is not [`ProductQuantizer::code_len`] bytes long.
    pub fn decode(&self, code: &[u8]) -> Vec<f32> {
        assert_eq!(code.len(), self.code_len(), "wrong code length");
        let sub_dim = self.sub_dim();
        code.iter()
            .enumerate()
            .flat_map(|(s, &c)| {
                let start = usize::from(c) * sub_dim;
                &self.codebook(s)[start..start + sub_dim]
            })
            .copied()
            .collect()
    }

    /// Measures a validated `query` against every centroid once, so the
    /// distance to any code is a few table lookups.
    pub(super) fn distance_table(&self, query: &[f32]) -> DistanceTable {
        let mut normalized;
        let query = if self.metric == Metric::Cosine {
            normalized = query.to_vec();
            normalize(&mut normalized);
            &normalized
        } else {
            query
        };
        let sub_dim = self.sub_dim();
        let mut entries = Vec::with_capacity(self.subspaces * CENTROIDS);
        for (s, slice) in query.chunks_exact(sub_dim).enumerate() {
            entries.extend(self.codebook(s).chunks_exact(sub_dim).map(
                |centroid| match self.metric {
                    Metric::L2 => l2_squared(slice, centroid),
                    Metric::Dot | Metric::Cosine => -dot(slice, centroid),
                },
            ));
        }
        DistanceTable {
            entries,
            // Cosine distance between unit vectors is `1 - dot`.
            offset: if self.metric == Metric::Cosine {
                1.0
            } else {
                0.0
            },
        }
    }
}

/// One query's distances to every centroid of every subspace.
#[derive(Debug)]
pub(super) struct DistanceTable {
    /// `entries[s * CENTROIDS + c]`: the query's slice `s` against centroid `c`.
    entries: Vec<f32>,
    /// Added to every distance (1 for cosine, 0 otherwise).
    offset: f32,
}

impl DistanceTable {
    /// The approximate distance from the query to the vector `code` stands for.
    #[inline]
    pub(super) fn distance(&self, code: &[u8]) -> f32 {
        let sum: f32 = self
            .entries
            .as_chunks::<CENTROIDS>()
            .0
            .iter()
            .zip(code)
            .map(|(row, &c)| row[usize::from(c)])
            .sum();
        self.offset + sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    fn random(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        (0..n * dim).map(|_| rng.next_f64() as f32).collect()
    }

    /// Few iterations keep debug-mode tests fast.
    fn test_params(subspaces: usize) -> PqParams {
        PqParams {
            subspaces,
            iterations: 4,
            ..PqParams::default()
        }
    }

    fn train(metric: Metric, subspaces: usize, data: &[f32], threads: usize) -> ProductQuantizer {
        ProductQuantizer::train(
            8,
            metric,
            &test_params(subspaces),
            data,
            NonZeroUsize::new(threads).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn training_does_not_depend_on_the_thread_count() {
        let data = random(1_000, 8, 1);
        let one = train(Metric::L2, 4, &data, 1);
        let many = train(Metric::L2, 4, &data, 8);
        assert_eq!(one.codebooks, many.codebooks);
    }

    #[test]
    fn decoding_approximates_the_vector() {
        let data = random(2_000, 8, 2);
        let pq = train(Metric::L2, 4, &data, 4);
        let mean_error: f32 = data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|v| l2_squared(v, &pq.decode(&pq.encode(v).unwrap())))
            .sum::<f32>()
            / 2_000.0;
        // A random point is ~1.33 away (squared) from another; 256 centroids
        // per 2-d slice cut that a lot.
        assert!(mean_error < 0.05, "mean squared error {mean_error}");
    }

    #[test]
    fn table_distances_equal_distances_to_the_decoded_vector() {
        let data = random(1_000, 8, 3);
        let query = &random(1, 8, 4)[..];
        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            let pq = train(metric, 2, &data, 2);
            let table = pq.distance_table(query);
            for v in data.as_chunks::<8>().0.iter().take(20) {
                let code = pq.encode(v).unwrap();
                let decoded = pq.decode(&code);
                let mut unit_query = query.to_vec();
                if metric == Metric::Cosine {
                    normalize(&mut unit_query);
                }
                let expected = match metric {
                    Metric::L2 => l2_squared(&unit_query, &decoded),
                    Metric::Dot => -dot(&unit_query, &decoded),
                    Metric::Cosine => 1.0 - dot(&unit_query, &decoded),
                };
                let got = table.distance(&code);
                assert!(
                    (got - expected).abs() < 1e-4,
                    "{metric:?}: {got} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn cosine_codes_ignore_vector_length() {
        let data = random(1_000, 8, 5);
        let pq = train(Metric::Cosine, 4, &data, 2);
        let v = &data[..8];
        let scaled: Vec<f32> = v.iter().map(|x| x * 10.0).collect();
        assert_eq!(pq.encode(v).unwrap(), pq.encode(&scaled).unwrap());
    }

    #[test]
    fn rejects_bad_training_data() {
        let threads = NonZeroUsize::MIN;
        let params = PqParams::with_subspaces(2);
        let too_few = random(255, 8, 6);
        let err = ProductQuantizer::train(8, Metric::L2, &params, &too_few, threads).unwrap_err();
        assert!(err.to_string().contains("at least 256"), "{err}");

        let ragged = random(300, 8, 7)[..8 * 300 - 3].to_vec();
        let err = ProductQuantizer::train(8, Metric::L2, &params, &ragged, threads).unwrap_err();
        assert!(err.to_string().contains("not a multiple"), "{err}");

        let mut nan = random(300, 8, 8);
        nan[100] = f32::NAN;
        assert!(matches!(
            ProductQuantizer::train(8, Metric::L2, &params, &nan, threads),
            Err(Error::InvalidVector)
        ));
    }

    #[test]
    fn encode_validates_its_input() {
        let pq = train(Metric::L2, 2, &random(300, 8, 9), 1);
        assert!(matches!(
            pq.encode(&[0.0; 7]),
            Err(Error::DimensionMismatch { .. })
        ));
        assert!(matches!(
            pq.encode(&[f32::INFINITY; 8]),
            Err(Error::InvalidVector)
        ));
    }
}
