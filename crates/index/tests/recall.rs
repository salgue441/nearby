//! HNSW recall measured against exact search, using only the public API.
//!
//! Integration tests live in `tests/` and compile as a separate crate, so they
//! can only use what `nearby_index` exports, exactly like a real user would.

use std::num::NonZeroUsize;

use nearby_core::{Metric, PointId};
use nearby_index::{FlatIndex, Hnsw, HnswParams, VectorIndex};

const DIM: usize = 32;
const K: usize = 10;

/// Deterministic uniform vectors in [0, 1)^dim (`SplitMix64`), no extra crate needed.
fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
    };
    (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
}

fn build(metric: Metric, params: HnswParams, data: &[Vec<f32>]) -> (Hnsw, FlatIndex) {
    let mut hnsw = Hnsw::new(DIM, metric, params).unwrap();
    let mut flat = FlatIndex::new(DIM, metric);
    for (i, v) in data.iter().enumerate() {
        hnsw.insert(i as PointId, v).unwrap();
        flat.add(i as PointId, v).unwrap();
    }
    (hnsw, flat)
}

/// Mean fraction of the exact top-k that HNSW also returns.
fn mean_recall(hnsw: &Hnsw, flat: &FlatIndex, queries: &[Vec<f32>]) -> f64 {
    let total: usize = queries
        .iter()
        .map(|q| {
            let truth = flat.search(q, K).unwrap();
            let found = hnsw.search(q, K).unwrap();
            found
                .iter()
                .filter(|p| truth.iter().any(|t| t.id == p.id))
                .count()
        })
        .sum();
    total as f64 / (queries.len() * K) as f64
}

fn test_params() -> HnswParams {
    // A smaller ef_construction keeps debug-mode tests fast.
    HnswParams {
        ef_construction: 100,
        ..HnswParams::default()
    }
}

#[test]
fn hnsw_recall_is_high_on_random_data() {
    let data = random_vectors(2_000, DIM, 1);
    let queries = random_vectors(50, DIM, 2);
    let (hnsw, flat) = build(Metric::L2, test_params(), &data);

    let recall = mean_recall(&hnsw, &flat, &queries);
    assert!(recall >= 0.95, "recall@{K} = {recall:.3}");
}

#[test]
fn larger_ef_search_does_not_hurt_recall() {
    let data = random_vectors(2_000, DIM, 3);
    let queries = random_vectors(50, DIM, 4);
    let (mut hnsw, flat) = build(Metric::L2, test_params(), &data);

    hnsw.set_ef_search(10);
    let low = mean_recall(&hnsw, &flat, &queries);
    hnsw.set_ef_search(200);
    let high = mean_recall(&hnsw, &flat, &queries);
    assert!(high >= low, "ef=10: {low:.3}, ef=200: {high:.3}");
    assert!(high >= 0.99, "ef=200 should be nearly exact, got {high:.3}");
}

#[test]
fn works_with_cosine_distance() {
    let data = random_vectors(1_000, DIM, 5);
    let queries = random_vectors(30, DIM, 6);
    let (hnsw, flat) = build(Metric::Cosine, test_params(), &data);

    let recall = mean_recall(&hnsw, &flat, &queries);
    assert!(recall >= 0.95, "recall@{K} = {recall:.3}");
}

#[test]
fn every_point_finds_itself() {
    let data = random_vectors(500, DIM, 7);
    let (hnsw, _) = build(Metric::L2, test_params(), &data);
    for (i, v) in data.iter().enumerate() {
        assert_eq!(hnsw.search(v, 1).unwrap()[0].id, i as PointId);
    }
}

#[test]
fn batch_insert_recall_matches_single_inserts() {
    let data = random_vectors(2_000, DIM, 8);
    let queries = random_vectors(50, DIM, 9);
    let (single, flat) = build(Metric::L2, test_params(), &data);

    let points: Vec<(PointId, &[f32])> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (i as PointId, v.as_slice()))
        .collect();
    let mut batch = Hnsw::new(DIM, Metric::L2, test_params()).unwrap();
    batch
        .insert_batch(&points, NonZeroUsize::new(4).unwrap())
        .unwrap();

    let single_recall = mean_recall(&single, &flat, &queries);
    let batch_recall = mean_recall(&batch, &flat, &queries);
    assert!(batch_recall >= 0.95, "batch recall@{K} = {batch_recall:.3}");
    assert!(
        batch_recall >= single_recall - 0.02,
        "batch {batch_recall:.3} vs single inserts {single_recall:.3}"
    );
}
