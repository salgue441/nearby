//! Behavior of `nearby-core`'s public API, as seen from another crate.

use std::path::PathBuf;

use nearby_core::distance::{cosine_distance, dot, l2_squared, norm, normalize};
use nearby_core::{Error, Metric, OrdF32, PointId, ScoredPoint, TopK, validate_vector};

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

/// Tiny deterministic pseudo-random generator in `[-1, 1)`, so tests need no extra crate.
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

// ---- Metric -----------------------------------------------------------------

#[test]
fn default_metric_is_l2() {
    assert_eq!(Metric::default(), Metric::L2);
}

#[test]
fn metric_dispatches_to_the_matching_function() {
    let a = pseudo_random(16, 1);
    let b = pseudo_random(16, 2);
    assert!(approx_eq(Metric::L2.distance(&a, &b), l2_squared(&a, &b)));
    assert!(approx_eq(
        Metric::Cosine.distance(&a, &b),
        cosine_distance(&a, &b)
    ));
    assert!(approx_eq(Metric::Dot.distance(&a, &b), -dot(&a, &b)));
}

#[test]
fn self_distance_is_minimal_for_l2_and_cosine() {
    let a = pseudo_random(8, 3);
    assert!(approx_eq(Metric::L2.distance(&a, &a), 0.0));
    assert!(approx_eq(Metric::Cosine.distance(&a, &a), 0.0));
}

#[test]
fn cosine_distance_stays_in_range() {
    for seed in 0..50 {
        let a = pseudo_random(8, seed);
        let b = pseudo_random(8, seed + 1_000);
        let d = cosine_distance(&a, &b);
        assert!((0.0..=2.0).contains(&d), "seed {seed}: {d}");
    }
}

#[test]
fn distances_of_empty_vectors() {
    assert!(approx_eq(dot(&[], &[]), 0.0));
    assert!(approx_eq(l2_squared(&[], &[]), 0.0));
    assert!(approx_eq(cosine_distance(&[], &[]), 1.0), "no direction");
}

#[test]
fn norm_is_euclidean_length() {
    assert!(approx_eq(norm(&[3.0, 4.0]), 5.0));
    assert!(approx_eq(norm(&[0.0, 0.0, 0.0]), 0.0));
}

#[test]
fn normalize_preserves_direction() {
    let original = pseudo_random(8, 7);
    let mut v = original.clone();
    normalize(&mut v);
    assert!(approx_eq(norm(&v), 1.0));
    assert!(approx_eq(cosine_distance(&original, &v), 0.0));
}

#[test]
fn for_unit_vectors_cosine_equals_one_minus_dot() {
    let mut a = pseudo_random(8, 11);
    let mut b = pseudo_random(8, 12);
    normalize(&mut a);
    normalize(&mut b);
    assert!(approx_eq(cosine_distance(&a, &b), 1.0 - dot(&a, &b)));
}

// ---- validate_vector --------------------------------------------------------

#[test]
fn validate_accepts_empty_vector_for_dim_zero() {
    assert!(validate_vector(&[], 0).is_ok());
}

#[test]
fn validate_rejects_too_long_vector() {
    assert!(matches!(
        validate_vector(&[1.0, 2.0, 3.0], 2),
        Err(Error::DimensionMismatch {
            expected: 2,
            got: 3
        })
    ));
}

#[test]
fn validate_rejects_negative_infinity_anywhere() {
    assert!(matches!(
        validate_vector(&[1.0, 2.0, f32::NEG_INFINITY], 3),
        Err(Error::InvalidVector)
    ));
}

#[test]
fn validate_checks_dimension_before_values() {
    assert!(matches!(
        validate_vector(&[f32::NAN], 2),
        Err(Error::DimensionMismatch { .. })
    ));
}

// ---- Error ------------------------------------------------------------------

#[test]
fn error_messages() {
    let cases = [
        (
            Error::DimensionMismatch {
                expected: 3,
                got: 2,
            },
            "dimension mismatch: expected 3, got 2",
        ),
        (
            Error::InvalidVector,
            "vector contains NaN or infinite values",
        ),
        (Error::PointNotFound(42), "point `42` not found"),
        (
            Error::CollectionNotFound("docs".to_owned()),
            "collection `docs` not found",
        ),
        (
            Error::Config("bad port".to_owned()),
            "invalid config: bad port",
        ),
    ];
    for (err, expected) in cases {
        assert_eq!(err.to_string(), expected);
    }
}

#[test]
fn corruption_constructor_fills_fields() {
    let err = Error::corruption(PathBuf::from("wal/0001.log"), String::from("bad crc"));
    match err {
        Error::Corruption { path, reason } => {
            assert_eq!(path, PathBuf::from("wal/0001.log"));
            assert_eq!(reason, "bad crc");
        }
        other => panic!("expected Corruption, got {other:?}"),
    }
}

#[test]
fn io_errors_convert_and_keep_their_message() {
    let io = std::io::Error::new(std::io::ErrorKind::NotFound, "no such segment");
    let err: Error = io.into();
    assert!(matches!(err, Error::Io(_)));
    assert_eq!(err.to_string(), "no such segment");
}

#[test]
fn question_mark_converts_io_errors() {
    fn open() -> nearby_core::Result<()> {
        Err(std::io::Error::other("disk on fire"))?;
        Ok(())
    }
    assert!(matches!(open(), Err(Error::Io(_))));
}

// ---- OrdF32 -----------------------------------------------------------------

#[test]
fn ord_f32_orders_every_value() {
    let mut values = [
        OrdF32(f32::NAN),
        OrdF32(1.0),
        OrdF32(f32::INFINITY),
        OrdF32(0.0),
        OrdF32(-0.0),
        OrdF32(f32::NEG_INFINITY),
    ];
    values.sort();
    let sorted: Vec<f32> = values.iter().map(|v| v.0).collect();

    assert_eq!(sorted[0], f32::NEG_INFINITY);
    assert!(
        sorted[1] == 0.0 && sorted[1].is_sign_negative(),
        "-0 before +0"
    );
    assert!(sorted[2] == 0.0 && sorted[2].is_sign_positive());
    assert_eq!(sorted[3], 1.0);
    assert_eq!(sorted[4], f32::INFINITY);
    assert!(sorted[5].is_nan(), "NaN sorts last");
}

#[test]
fn ord_f32_equality_is_reflexive_even_for_nan() {
    assert_eq!(OrdF32(f32::NAN), OrdF32(f32::NAN));
    assert_ne!(OrdF32(0.0), OrdF32(-0.0));
}

// ---- TopK -------------------------------------------------------------------

#[test]
fn new_topk_is_empty() {
    let top = TopK::new(3);
    assert_eq!(top.k(), 3);
    assert_eq!(top.len(), 0);
    assert!(top.is_empty());
    assert!(!top.is_full());
    assert_eq!(top.worst(), None);
}

#[test]
fn len_never_exceeds_k() {
    let mut top = TopK::new(3);
    for (id, d) in pseudo_random(100, 5).into_iter().enumerate() {
        top.push(id as PointId, d);
        assert!(top.len() <= 3);
    }
    assert!(top.is_full());
}

#[test]
fn zero_k_is_always_full_and_never_has_a_worst() {
    let mut top = TopK::new(0);
    assert!(top.is_full());
    assert!(!top.push(1, -100.0));
    assert_eq!(top.worst(), None);
}

#[test]
fn ties_are_broken_by_smaller_id() {
    let mut top = TopK::new(2);
    top.push(30, 1.0);
    top.push(10, 1.0);
    top.push(20, 1.0);
    let ids: Vec<PointId> = top.into_sorted().iter().map(|p| p.id).collect();
    assert_eq!(ids, vec![10, 20]);
}

#[test]
fn tie_with_worst_and_larger_id_is_rejected() {
    let mut top = TopK::new(1);
    assert!(top.push(5, 1.0));
    assert!(!top.push(6, 1.0), "equal distance, larger id");
    assert!(top.push(4, 1.0), "equal distance, smaller id");
}

#[test]
fn result_does_not_depend_on_insertion_order() {
    let candidates: Vec<(PointId, f32)> = pseudo_random(200, 9)
        .into_iter()
        .enumerate()
        .map(|(id, d)| (id as PointId, (d * 4.0).round())) // many ties
        .collect();

    let collect = |order: &mut dyn Iterator<Item = &(PointId, f32)>| {
        let mut top = TopK::new(15);
        for &(id, d) in order {
            top.push(id, d);
        }
        top.into_sorted()
    };

    let forward = collect(&mut candidates.iter());
    let backward = collect(&mut candidates.iter().rev());
    assert_eq!(forward, backward);
}

#[test]
fn negative_distances_are_supported() {
    // The Dot metric produces negative distances.
    let mut top = TopK::new(2);
    top.push(1, -1.0);
    top.push(2, -5.0);
    top.push(3, 0.5);
    assert_eq!(
        top.into_sorted(),
        vec![
            ScoredPoint {
                id: 2,
                distance: -5.0
            },
            ScoredPoint {
                id: 1,
                distance: -1.0
            },
        ]
    );
}
