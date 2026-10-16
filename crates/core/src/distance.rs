//! Distance functions between vectors.
//!
//! These are the hottest functions in the whole database: a single search
//! calls them thousands of times. This first version is plain scalar code;
//! Milestone 3 adds SIMD versions behind the same functions.
//!
//! All functions expect slices of equal length. That is checked in debug
//! builds only, because callers validate dimensions at the API boundary.

use crate::types::Metric;

impl Metric {
    /// Distance between `a` and `b` under this metric.
    /// Lower means closer.
    #[inline]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::L2 => l2_squared(a, b),
            Metric::Cosine => cosine_distance(a, b),
            Metric::Dot => -dot(a, b),
        }
    }
}

/// Dot product: `sum(a[i] * b[i])`.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have the same dimension");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Squared Euclidean distance: `sum((a[i] - b[i])^2)`.
///
/// Skipping the square root keeps the same ranking and saves time.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have the same dimension");
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
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
    let denom = norm(a) * norm(b);
    if denom <= f32::MIN_POSITIVE {
        return 1.0;
    }

    (1.0 - dot(a, b) / denom).clamp(0.0, 2.0)
}

/// Scales `v` in place to unit length. Leaves a zero vector unchanged.
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
