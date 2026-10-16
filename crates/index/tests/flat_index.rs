//! Behavior of `FlatIndex` through the public API, as seen from another crate.

use nearby_core::{Error, Metric, PointId, ScoredPoint};
use nearby_index::{FlatIndex, VectorIndex};

const METRICS: [Metric; 3] = [Metric::L2, Metric::Cosine, Metric::Dot];

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

fn ids(results: &[ScoredPoint]) -> Vec<PointId> {
    results.iter().map(|p| p.id).collect()
}

/// Reference answer: score every vector and fully sort.
fn brute_force(data: &[f32], dim: usize, metric: Metric, query: &[f32], k: usize) -> Vec<PointId> {
    let mut scored: Vec<(f32, PointId)> = data
        .chunks_exact(dim)
        .enumerate()
        .map(|(i, v)| (metric.distance(query, v), i as PointId))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

#[test]
fn matches_a_full_sort_for_every_metric() {
    const DIM: usize = 8;
    const N: usize = 500;
    let data = pseudo_random(N * DIM, 1);

    for metric in METRICS {
        let mut index = FlatIndex::with_capacity(DIM, metric, N);
        for (i, v) in data.as_chunks::<DIM>().0.iter().enumerate() {
            index.add(i as PointId, v).unwrap();
        }

        for seed in 0..10 {
            let query = pseudo_random(DIM, 1_000 + seed);
            let got = index.search(&query, 10).unwrap();
            assert_eq!(
                ids(&got),
                brute_force(&data, DIM, metric, &query, 10),
                "{metric:?}, query seed {seed}"
            );
        }
    }
}

#[test]
fn results_are_sorted_and_carry_correct_distances() {
    let mut index = FlatIndex::new(3, Metric::L2);
    let vectors = pseudo_random(30, 2);
    for (i, v) in vectors.as_chunks::<3>().0.iter().enumerate() {
        index.add(i as PointId, v).unwrap();
    }

    let query = [0.1, -0.2, 0.3];
    let results = index.search(&query, 5).unwrap();
    assert!(results.windows(2).all(|w| w[0].distance <= w[1].distance));
    for p in &results {
        let stored = index.get(p.id).unwrap();
        assert!((p.distance - Metric::L2.distance(&query, stored)).abs() < 1e-6);
    }
}

#[test]
fn cosine_distances_are_bit_identical_to_the_metric() {
    // The index uses stored norms; the result must not change because of it.
    const DIM: usize = 13; // not a multiple of the kernel's lane width
    let data = pseudo_random(200 * DIM, 4);
    let mut index = FlatIndex::new(DIM, Metric::Cosine);
    for (i, v) in data.as_chunks::<DIM>().0.iter().enumerate() {
        index.add(i as PointId, v).unwrap();
    }

    let query = pseudo_random(DIM, 5);
    for p in index.search(&query, 200).unwrap() {
        let expected = Metric::Cosine.distance(&query, index.get(p.id).unwrap());
        assert_eq!(p.distance.to_bits(), expected.to_bits(), "id {}", p.id);
    }
}

#[test]
fn cosine_handles_zero_vectors() {
    let mut index = FlatIndex::new(2, Metric::Cosine);
    index.add(1, &[0.0, 0.0]).unwrap();
    index.add(2, &[1.0, 0.0]).unwrap();
    index.add(3, &[-1.0, 0.0]).unwrap();

    let results = index.search(&[2.0, 0.0], 3).unwrap();
    assert_eq!(ids(&results), vec![2, 1, 3]);
    assert!(
        (results[1].distance - 1.0).abs() < 1e-6,
        "zero vector is orthogonal"
    );

    let from_zero = index.search(&[0.0, 0.0], 3).unwrap();
    assert!(from_zero.iter().all(|p| (p.distance - 1.0).abs() < 1e-6));
}

#[test]
fn reports_dim_and_len() {
    let mut index = FlatIndex::new(4, Metric::Cosine);
    assert_eq!(index.dim(), 4);
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());

    index.add(9, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(index.len(), 1);
    assert!(!index.is_empty());
}

#[test]
fn with_capacity_behaves_like_new() {
    let mut index = FlatIndex::with_capacity(2, Metric::L2, 100);
    assert!(index.is_empty());
    index.add(1, &[1.0, 1.0]).unwrap();
    assert_eq!(ids(&index.search(&[1.0, 1.0], 1).unwrap()), vec![1]);
}

#[test]
#[should_panic(expected = "dimension must be at least 1")]
fn new_panics_on_zero_dimension() {
    let _ = FlatIndex::new(0, Metric::L2);
}

#[test]
#[should_panic(expected = "dimension must be at least 1")]
fn with_capacity_panics_on_zero_dimension() {
    let _ = FlatIndex::with_capacity(0, Metric::L2, 10);
}

#[test]
fn huge_k_returns_everything_sorted() {
    let mut index = FlatIndex::new(1, Metric::L2);
    for id in [4, 0, 3, 1, 2] {
        index.add(id, &[id as f32]).unwrap();
    }
    let results = index.search(&[0.0], usize::MAX).unwrap();
    assert_eq!(ids(&results), vec![0, 1, 2, 3, 4]);
}

#[test]
fn k_zero_returns_nothing() {
    let mut index = FlatIndex::new(1, Metric::L2);
    index.add(1, &[1.0]).unwrap();
    assert_eq!(index.search(&[1.0], 0).unwrap(), Vec::new());
}

#[test]
fn equal_distances_are_ordered_by_id() {
    let mut index = FlatIndex::new(1, Metric::L2);
    // Ids 7 and 3 are both at distance 1 from the query; added out of order.
    index.add(7, &[1.0]).unwrap();
    index.add(3, &[-1.0]).unwrap();
    index.add(5, &[10.0]).unwrap();
    assert_eq!(ids(&index.search(&[0.0], 2).unwrap()), vec![3, 7]);
}

#[test]
fn dot_metric_prefers_larger_products() {
    let mut index = FlatIndex::new(2, Metric::Dot);
    index.add(1, &[1.0, 0.0]).unwrap();
    index.add(2, &[3.0, 0.0]).unwrap();
    index.add(3, &[-2.0, 0.0]).unwrap();
    let results = index.search(&[1.0, 0.0], 3).unwrap();
    assert_eq!(ids(&results), vec![2, 1, 3]);
    assert!((results[0].distance - -3.0).abs() < 1e-6);
}

#[test]
fn rejects_invalid_vectors_without_storing_them() {
    let mut index = FlatIndex::new(2, Metric::L2);
    assert!(matches!(
        index.add(1, &[f32::NAN, 0.0]),
        Err(Error::InvalidVector)
    ));
    assert!(matches!(
        index.add(2, &[0.0, f32::INFINITY]),
        Err(Error::InvalidVector)
    ));
    assert!(index.is_empty());
    assert_eq!(index.get(1), None);
}

#[test]
fn rejects_invalid_queries() {
    let mut index = FlatIndex::new(2, Metric::L2);
    index.add(1, &[0.0, 0.0]).unwrap();
    assert!(matches!(
        index.search(&[f32::NAN, 0.0], 1),
        Err(Error::InvalidVector)
    ));
    assert!(matches!(
        index.search(&[0.0, 0.0, 0.0], 1),
        Err(Error::DimensionMismatch {
            expected: 2,
            got: 3
        })
    ));
}

#[test]
fn get_looks_up_by_id_not_position() {
    let mut index = FlatIndex::new(2, Metric::L2);
    index.add(100, &[1.0, 2.0]).unwrap();
    index.add(7, &[3.0, 4.0]).unwrap();
    assert_eq!(index.get(7), Some(&[3.0, 4.0][..]));
    assert_eq!(index.get(100), Some(&[1.0, 2.0][..]));
    assert_eq!(index.get(0), None);
    assert_eq!(index.get(1), None);
}

#[test]
fn get_returns_first_copy_of_a_duplicate_id() {
    let mut index = FlatIndex::new(1, Metric::L2);
    index.add(1, &[1.0]).unwrap();
    index.add(1, &[2.0]).unwrap();
    assert_eq!(index.len(), 2, "duplicates are not detected by the index");
    assert_eq!(index.get(1), Some(&[1.0][..]));
}

#[test]
fn usable_as_a_shared_trait_object() {
    let mut flat = FlatIndex::new(2, Metric::L2);
    flat.add(1, &[0.0, 0.0]).unwrap();
    flat.add(2, &[1.0, 1.0]).unwrap();

    let index: std::sync::Arc<dyn VectorIndex> = std::sync::Arc::new(flat);
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let index = std::sync::Arc::clone(&index);
            std::thread::spawn(move || index.search(&[0.9, 0.9], 1).unwrap())
        })
        .collect();

    for handle in handles {
        assert_eq!(ids(&handle.join().unwrap()), vec![2]);
    }
}
