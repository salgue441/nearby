//! `search_filtered` on every index, through the public API.

use std::num::NonZeroUsize;

use nearby_core::{Metric, PointId, ScoredPoint};
use nearby_index::{FlatIndex, Hnsw, HnswParams, PqIndex, PqParams, ProductQuantizer, VectorIndex};

const DIM: usize = 8;

/// Deterministic pseudo-random vectors in `[0, 1)^DIM`.
fn random_vectors(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    (0..n).map(|_| (0..DIM).map(|_| next()).collect()).collect()
}

fn ids(results: &[ScoredPoint]) -> Vec<PointId> {
    results.iter().map(|p| p.id).collect()
}

fn odd(id: PointId) -> bool {
    id % 2 == 1
}

fn indexes(data: &[Vec<f32>]) -> Vec<(&'static str, Box<dyn VectorIndex>)> {
    let mut flat = FlatIndex::new(DIM, Metric::L2);
    let mut hnsw = Hnsw::new(DIM, Metric::L2, HnswParams::default()).unwrap();
    for (i, v) in data.iter().enumerate() {
        flat.add(i as PointId, v).unwrap();
        hnsw.insert(i as PointId, v).unwrap();
    }
    let training: Vec<f32> = data.concat();
    let pq = ProductQuantizer::train(
        DIM,
        Metric::L2,
        &PqParams {
            subspaces: 4,
            iterations: 4,
            ..PqParams::default()
        },
        &training,
        NonZeroUsize::MIN,
    )
    .unwrap();
    let mut pq = PqIndex::new(pq);
    for (i, v) in data.iter().enumerate() {
        pq.add(i as PointId, v).unwrap();
    }
    vec![
        ("flat", Box::new(flat)),
        ("hnsw", Box::new(hnsw)),
        ("pq", Box::new(pq)),
    ]
}

#[test]
fn rejected_points_are_never_returned() {
    let data = random_vectors(600, 1);
    for (name, index) in indexes(&data) {
        for query in random_vectors(10, 2) {
            let results = index.search_filtered(&query, 10, &odd).unwrap();
            assert_eq!(results.len(), 10, "{name}");
            assert!(
                results.iter().all(|p| odd(p.id)),
                "{name}: {:?}",
                ids(&results)
            );
            assert!(results.windows(2).all(|w| w[0].distance <= w[1].distance));
        }
    }
}

#[test]
fn rejecting_everything_returns_nothing() {
    let data = random_vectors(300, 3);
    for (name, index) in indexes(&data) {
        let results = index.search_filtered(&data[0], 5, &|_| false).unwrap();
        assert_eq!(results, Vec::new(), "{name}");
    }
}

#[test]
fn accepting_everything_matches_search() {
    let data = random_vectors(300, 4);
    for (name, index) in indexes(&data) {
        let query = &random_vectors(1, 5)[0];
        assert_eq!(
            index.search_filtered(query, 10, &|_| true).unwrap(),
            index.search(query, 10).unwrap(),
            "{name}"
        );
    }
}

#[test]
fn filtered_flat_search_is_exact_over_the_kept_points() {
    let data = random_vectors(500, 6);
    let mut all = FlatIndex::new(DIM, Metric::L2);
    let mut kept = FlatIndex::new(DIM, Metric::Cosine);
    let mut kept_l2 = FlatIndex::new(DIM, Metric::L2);
    let mut all_cosine = FlatIndex::new(DIM, Metric::Cosine);
    for (i, v) in data.iter().enumerate() {
        let id = i as PointId;
        all.add(id, v).unwrap();
        all_cosine.add(id, v).unwrap();
        if odd(id) {
            kept_l2.add(id, v).unwrap();
            kept.add(id, v).unwrap();
        }
    }
    let query = &random_vectors(1, 7)[0];
    assert_eq!(
        all.search_filtered(query, 20, &odd).unwrap(),
        kept_l2.search(query, 20).unwrap()
    );
    // Cosine keeps per-vector norms; filtering must keep them aligned.
    assert_eq!(
        all_cosine.search_filtered(query, 20, &odd).unwrap(),
        kept.search(query, 20).unwrap()
    );
}

#[test]
fn filtered_hnsw_recall_holds_when_half_the_points_are_rejected() {
    let data = random_vectors(2_000, 8);
    let mut hnsw = Hnsw::new(DIM, Metric::L2, HnswParams::default()).unwrap();
    let mut flat = FlatIndex::new(DIM, Metric::L2);
    for (i, v) in data.iter().enumerate() {
        hnsw.insert(i as PointId, v).unwrap();
        flat.add(i as PointId, v).unwrap();
    }
    let queries = random_vectors(50, 9);
    let found: usize = queries
        .iter()
        .map(|q| {
            let truth = ids(&flat.search_filtered(q, 10, &odd).unwrap());
            let got = hnsw.search_filtered(q, 10, &odd).unwrap();
            got.iter().filter(|p| truth.contains(&p.id)).count()
        })
        .sum();
    let recall = found as f64 / 500.0;
    assert!(recall >= 0.95, "filtered recall@10 = {recall:.3}");
}
