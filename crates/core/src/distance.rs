//! Distance functions between vectors.
//!
//! These are the hottest functions in the whole database: a single search
//! calls them thousands of times. This version is portable scalar code that
//! keeps several independent running sums so the compiler can auto-vectorize
//! it; Milestone 3 adds explicit SIMD versions behind the same functions.
//!
//! All functions expect slices of equal length. That is checked in debug
//! builds only, because callers validate dimensions at the API boundary.

use crate::types::Metric;

impl Metric {
    /// Distance between `a` and `b` under this metric.
    /// Lower means closer.
    ///
    /// With [`Metric::Dot`] the result can be negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use nearby_core::Metric;
    ///
    /// assert_eq!(Metric::L2.distance(&[0.0, 0.0], &[3.0, 4.0]), 25.0);
    /// assert_eq!(Metric::Dot.distance(&[1.0, 2.0], &[3.0, 4.0]), -11.0);
    /// ```
    #[inline]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::L2 => l2_squared(a, b),
            Metric::Cosine => cosine_distance(a, b),
            Metric::Dot => -dot(a, b),
        }
    }
}

/// Number of independent running sums in [`sum_lanes`].
///
/// 8 `f32`s fill one AVX register, or two SSE registers.
const LANES: usize = 8;

/// `sum(term(a[i], b[i]))`, written so the compiler can vectorize it.
///
/// Float addition is not associative, so LLVM may not reorder a plain
/// `.iter().sum()` into SIMD lanes: every addition waits for the previous
/// one. Keeping `LANES` independent sums makes that reordering explicit,
/// which lets it vectorize (about 3x faster on 128-d vectors).
///
/// The summation order differs from a left-to-right sum, so results can
/// differ from it in the last bits.
#[inline]
fn sum_lanes(a: &[f32], b: &[f32], term: impl Fn(f32, f32) -> f32) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have the same dimension");

    let (a_chunks, a_tail) = a.as_chunks::<LANES>();
    let (b_chunks, b_tail) = b.as_chunks::<LANES>();

    let mut acc = [0.0f32; LANES];
    for (x, y) in a_chunks.iter().zip(b_chunks) {
        for i in 0..LANES {
            acc[i] += term(x[i], y[i]);
        }
    }

    let tail: f32 = a_tail.iter().zip(b_tail).map(|(&x, &y)| term(x, y)).sum();
    acc.iter().sum::<f32>() + tail
}

/// Dot product: `sum(a[i] * b[i])`.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    sum_lanes(a, b, |x, y| x * y)
}

/// Squared Euclidean distance: `sum((a[i] - b[i])^2)`.
///
/// Skipping the square root keeps the same ranking and saves time.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    sum_lanes(a, b, |x, y| {
        let d = x - y;
        d * d
    })
}

/// Euclidean length of `v`.
#[inline]
pub fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// `1 - cosine_similarity(a, b)`, in `[0, 2]`.
///
/// A zero vector has no direction, so by convention it is treated as
/// orthogonal to everything and gets distance `1.0`.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    cosine_distance_with_norms(a, b, norm(a), norm(b))
}

/// [`cosine_distance`] with the norms already known, so a comparison costs
/// one dot product instead of three.
///
/// Indexes store each vector's norm once at insert time and compute the
/// query's norm once per search. `norm_a` and `norm_b` must be
/// [`norm`]`(a)` and [`norm`]`(b)`; given those, the result is bit-for-bit
/// the same as [`cosine_distance`].
#[inline]
pub fn cosine_distance_with_norms(a: &[f32], b: &[f32], norm_a: f32, norm_b: f32) -> f32 {
    let denom = norm_a * norm_b;
    if denom <= f32::MIN_POSITIVE {
        return 1.0;
    }

    (1.0 - dot(a, b) / denom).clamp(0.0, 2.0)
}

/// Scales `v` in place to unit length. Leaves a zero vector unchanged.
///
/// # Examples
///
/// ```
/// use nearby_core::distance::normalize;
///
/// let mut v = [3.0, 4.0];
/// normalize(&mut v);
/// assert_eq!(v, [0.6, 0.8]);
/// ```
///
/// For unit vectors, cosine distance is just `1 - dot`, so normalizing once
/// at insert time makes every later comparison cheaper.
pub fn normalize(v: &mut [f32]) {
    let n = norm(v);
    if n > f32::MIN_POSITIVE {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    /// Lengths around the lane width (empty, tail only, exact chunks, chunks
    /// plus tail) and real dataset dimensions (SIFT 128, GIST 960).
    fn lengths() -> impl Iterator<Item = usize> {
        (0..=40).chain([127, 128, 129, 960])
    }

    /// Tiny deterministic pseudo-random generator in `[-1, 1)`.
    fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    /// Left-to-right reference sum in `f64`, plus the sum of `|term|`, which
    /// bounds the rounding error of any `f32` summation order.
    fn reference(a: &[f32], b: &[f32], term: impl Fn(f64, f64) -> f64) -> (f64, f64) {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| term(f64::from(x), f64::from(y)))
            .fold((0.0, 0.0), |(sum, abs), t| (sum + t, abs + t.abs()))
    }

    fn assert_close(got: f32, (expected, scale): (f64, f64), what: &str) {
        let tolerance = 1e-5 * scale + 1e-6;
        assert!(
            (f64::from(got) - expected).abs() <= tolerance,
            "{what}: got {got}, expected {expected}"
        );
    }

    #[test]
    fn kernels_match_reference_at_every_length() {
        for n in lengths() {
            let a = pseudo_random(n, 1 + n as u64);
            let b = pseudo_random(n, 1_000 + n as u64);
            assert_close(
                dot(&a, &b),
                reference(&a, &b, |x, y| x * y),
                &format!("dot n={n}"),
            );
            assert_close(
                l2_squared(&a, &b),
                reference(&a, &b, |x, y| (x - y) * (x - y)),
                &format!("l2 n={n}"),
            );
        }
    }

    #[test]
    fn kernels_are_exact_on_small_integers() {
        // Small integer sums are exact in any order, so this catches a
        // dropped, doubled or misaligned element exactly.
        for n in lengths() {
            let a: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
            let b: Vec<f32> = (0..n).map(|i| (i * 3 % 5) as f32 - 2.0).collect();
            let dot_expected: f32 = (0..n).map(|i| a[i] * b[i]).sum();
            let l2_expected: f32 = (0..n).map(|i| (a[i] - b[i]) * (a[i] - b[i])).sum();
            assert_eq!(dot(&a, &b), dot_expected, "dot n={n}");
            assert_eq!(l2_squared(&a, &b), l2_expected, "l2 n={n}");
        }
    }

    #[test]
    fn cosine_with_norms_is_identical_to_cosine() {
        for n in lengths() {
            let a = pseudo_random(n, 7 + n as u64);
            let b = pseudo_random(n, 77 + n as u64);
            assert_eq!(
                cosine_distance_with_norms(&a, &b, norm(&a), norm(&b)).to_bits(),
                cosine_distance(&a, &b).to_bits(),
                "n={n}"
            );
        }
    }

    #[test]
    fn cosine_with_norms_keeps_zero_vector_convention() {
        let zero = [0.0; 9];
        let v = pseudo_random(9, 3);
        assert_eq!(cosine_distance_with_norms(&zero, &v, 0.0, norm(&v)), 1.0);
    }

    #[test]
    fn dot_product() {
        assert!(approx_eq(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0));
        assert!(approx_eq(dot(&[1.0, 0.0], &[0.0, 1.0]), 0.0));
    }

    #[test]
    fn l2_squared_distance() {
        assert!(approx_eq(l2_squared(&[0.0, 0.0], &[3.0, 4.0]), 25.0));
        assert!(approx_eq(l2_squared(&[1.0, 2.0], &[1.0, 2.0]), 0.0));
    }

    #[test]
    fn cosine_distance_basics() {
        let a = [1.0, 0.0];
        assert!(approx_eq(cosine_distance(&a, &a), 0.0), "same direction");
        assert!(
            approx_eq(cosine_distance(&a, &[0.0, 1.0]), 1.0),
            "orthogonal"
        );
        assert!(
            approx_eq(cosine_distance(&a, &[-1.0, 0.0]), 2.0),
            "opposite"
        );
        assert!(
            approx_eq(cosine_distance(&a, &[5.0, 0.0]), 0.0),
            "length is ignored"
        );
    }

    #[test]
    fn cosine_with_zero_vector_is_one() {
        assert!(approx_eq(cosine_distance(&[0.0, 0.0], &[1.0, 2.0]), 1.0));
        assert!(approx_eq(cosine_distance(&[0.0, 0.0], &[0.0, 0.0]), 1.0));
    }

    #[test]
    fn distances_are_symmetric() {
        let a = [0.3, -1.2, 4.0, 0.5];
        let b = [2.0, 0.1, -0.7, 1.5];
        for metric in [Metric::L2, Metric::Cosine, Metric::Dot] {
            assert!(
                approx_eq(metric.distance(&a, &b), metric.distance(&b, &a)),
                "{metric:?} is not symmetric"
            );
        }
    }

    #[test]
    fn dot_metric_ranks_larger_products_first() {
        let q = [1.0, 1.0];
        let near = Metric::Dot.distance(&q, &[2.0, 2.0]);
        let far = Metric::Dot.distance(&q, &[0.5, 0.5]);
        assert!(near < far);
    }

    #[test]
    fn normalize_gives_unit_length() {
        let mut v = [3.0, 4.0];
        normalize(&mut v);
        assert!(approx_eq(norm(&v), 1.0));
        assert!(approx_eq(v[0], 0.6) && approx_eq(v[1], 0.8));

        let mut zero = [0.0, 0.0];
        normalize(&mut zero);
        assert!(approx_eq(zero[0], 0.0) && approx_eq(zero[1], 0.0));
    }
}
