//! Behavior of `Hnsw` through the public API, as seen from another crate.
//!
//! Recall against exact search is covered in `recall.rs`; this file covers
//! the API contract: errors, edge cases, determinism and thread safety.

use std::sync::Arc;

use nearby_core::{Error, Metric, PointId, ScoredPoint};
use nearby_index::{FlatIndex, Hnsw, HnswParams, VectorIndex};

const DIM: usize = 8;

/// Deterministic pseudo-random vectors in `[0, 1)^dim`, no extra crate needed.
fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
}

fn build(metric: Metric, params: HnswParams, data: &[Vec<f32>]) -> Hnsw {
    let mut index = Hnsw::new(DIM, metric, params).unwrap();
    for (i, v) in data.iter().enumerate() {
        index.insert(i as PointId, v).unwrap();
    }
    index
}

fn ids(results: &[ScoredPoint]) -> Vec<PointId> {
    results.iter().map(|p| p.id).collect()
}

fn invalid_parameter(result: Result<Hnsw, Error>) -> String {
    match result {
        Err(err @ Error::InvalidParameter(_)) => err.to_string(),
        other => panic!("expected InvalidParameter, got {other:?}"),
    }
}

#[test]
fn rejects_invalid_construction_parameters() {
    let err = invalid_parameter(Hnsw::new(0, Metric::L2, HnswParams::default()));
    assert!(err.contains("dimension"), "{err}");

    let bad_m0 = HnswParams {
        m0: 4_096,
        ..HnswParams::default()
    };
    let err = invalid_parameter(Hnsw::new(DIM, Metric::L2, bad_m0));
    assert!(err.contains("m0"), "{err}");

    let bad_ef = HnswParams {
        ef_construction: 0,
        ..HnswParams::default()
    };
    invalid_parameter(Hnsw::with_capacity(DIM, Metric::L2, bad_ef, 100));
}

#[test]
fn rejects_invalid_vectors_and_queries() {
    let mut index = Hnsw::new(2, Metric::L2, HnswParams::default()).unwrap();
    assert!(matches!(
        index.insert(1, &[f32::INFINITY, 0.0]),
        Err(Error::InvalidVector)
    ));
    index.insert(1, &[0.0, 0.0]).unwrap();
    assert!(matches!(
        index.search(&[0.0, 0.0, 0.0], 1),
        Err(Error::DimensionMismatch {
            expected: 2,
            got: 3
        })
    ));
    assert!(matches!(
        index.search(&[f32::NAN, 0.0], 1),
        Err(Error::InvalidVector)
    ));
    assert_eq!(index.len(), 1);
}

#[test]
fn reports_external_ids_and_exact_distances() {
    // External ids are arbitrary, not insertion positions.
    let data = random_vectors(300, DIM, 1);
    let mut hnsw = Hnsw::new(DIM, Metric::L2, HnswParams::default()).unwrap();
    let mut flat = FlatIndex::new(DIM, Metric::L2);
    for (i, v) in data.iter().enumerate() {
        let id = 1_000_000 + 7 * i as PointId;
        hnsw.insert(id, v).unwrap();
        flat.add(id, v).unwrap();
    }

    let query = &random_vectors(1, DIM, 2)[0];
    for p in hnsw.search(query, 10).unwrap() {
        let stored = flat.get(p.id).expect("returned id was inserted");
        assert_eq!(
            p.distance,
            Metric::L2.distance(query, stored),
            "id {}",
            p.id
        );
    }
}

#[test]
fn small_index_with_large_ef_is_exact() {
    // With ef_search >= len on a connected graph, HNSW explores everything.
    let data = random_vectors(200, DIM, 3);
    let mut hnsw = build(Metric::L2, HnswParams::default(), &data);
    hnsw.set_ef_search(200);
    let mut flat = FlatIndex::new(DIM, Metric::L2);
    for (i, v) in data.iter().enumerate() {
        flat.add(i as PointId, v).unwrap();
    }

    for query in random_vectors(20, DIM, 4) {
        assert_eq!(
            ids(&hnsw.search(&query, 10).unwrap()),
            ids(&flat.search(&query, 10).unwrap())
        );
    }
}

#[test]
fn k_larger_than_len_returns_every_point_sorted() {
    let data = random_vectors(20, DIM, 5);
    let index = build(Metric::L2, HnswParams::default(), &data);
    let results = index.search(&data[0], 1_000).unwrap();
    assert_eq!(results.len(), 20);
    assert!(results.windows(2).all(|w| w[0].distance <= w[1].distance));
    assert_eq!(results[0].id, 0);
}

#[test]
fn dot_metric_matches_exact_search() {
    let data = random_vectors(500, DIM, 6);
    let mut hnsw = build(Metric::Dot, HnswParams::default(), &data);
    hnsw.set_ef_search(500);
    let mut flat = FlatIndex::new(DIM, Metric::Dot);
    for (i, v) in data.iter().enumerate() {
        flat.add(i as PointId, v).unwrap();
    }

    let query = &random_vectors(1, DIM, 7)[0];
    assert_eq!(
        ids(&hnsw.search(query, 5).unwrap()),
        ids(&flat.search(query, 5).unwrap())
    );
}

#[test]
fn same_seed_and_order_give_identical_results() {
    let data = random_vectors(500, DIM, 8);
    let a = build(Metric::L2, HnswParams::default(), &data);
    let b = build(Metric::L2, HnswParams::default(), &data);
    assert_eq!(a.max_level(), b.max_level());
    for query in random_vectors(10, DIM, 9) {
        assert_eq!(a.search(&query, 10).unwrap(), b.search(&query, 10).unwrap());
    }
}

#[test]
fn inserts_and_searches_can_interleave() {
    let data = random_vectors(200, DIM, 10);
    let mut index = Hnsw::new(DIM, Metric::L2, HnswParams::default()).unwrap();
    for (i, v) in data.iter().enumerate() {
        index.insert(i as PointId, v).unwrap();
        // The point just inserted is immediately findable.
        assert_eq!(index.search(v, 1).unwrap()[0].id, i as PointId);
    }
}

#[test]
fn usable_as_a_shared_trait_object() {
    let data = random_vectors(500, DIM, 11);
    let index: Arc<dyn VectorIndex> = Arc::new(build(Metric::L2, HnswParams::default(), &data));
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let index = Arc::clone(&index);
            let data = data.clone();
            std::thread::spawn(move || {
                for i in (t..500).step_by(4) {
                    assert_eq!(index.search(&data[i], 1).unwrap()[0].id, i as PointId);
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
}
