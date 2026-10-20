//! The engine through its public API: random workloads against a model,
//! recovery, and concurrent use.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::thread;

use nearby_core::{Metric, PointId};
use nearby_engine::{
    CollectionConfig, Database, Durability, EngineOptions, HnswConfig, IndexConfig,
};

const DIM: usize = 4;

fn options() -> EngineOptions {
    EngineOptions {
        durability: Durability::NoSync,
        flush_after_ops: 300,
        max_segments: 3,
        ..EngineOptions::default()
    }
}

fn config(index: IndexConfig) -> CollectionConfig {
    CollectionConfig {
        dim: DIM,
        metric: Metric::L2,
        index,
    }
}

/// A tiny deterministic generator, so the test needs no extra crate.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn vector(&mut self) -> Vec<f32> {
        (0..DIM)
            .map(|_| (self.next() % 1000) as f32 / 10.0)
            .collect()
    }
}

/// The ids of the `k` points in `model` closest to `query`, ties by id.
fn brute_force(model: &BTreeMap<PointId, Vec<f32>>, query: &[f32], k: usize) -> Vec<PointId> {
    let mut all: Vec<(f32, PointId)> = model
        .iter()
        .map(|(&id, v)| (Metric::L2.distance(query, v), id))
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    all.into_iter().take(k).map(|(_, id)| id).collect()
}

#[test]
fn a_flat_collection_matches_brute_force_through_writes_and_restarts() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path(), options()).unwrap();
    db.create_collection("c", config(IndexConfig::Flat))
        .unwrap();
    let mut model = BTreeMap::new();
    let mut rng = Lcg(7);

    for step in 0..3_000 {
        let c = db.collection("c").unwrap();
        match rng.next() % 10 {
            0..=5 => {
                let id = rng.next() % 500;
                let v = rng.vector();
                c.write().unwrap().upsert(&[(id, &v)]).unwrap();
                model.insert(id, v);
            }
            6..=7 => {
                let id = rng.next() % 500;
                c.write().unwrap().delete(&[id]).unwrap();
                model.remove(&id);
            }
            8 => {
                let query = rng.vector();
                let got: Vec<PointId> = c
                    .read()
                    .unwrap()
                    .search(&query, 5)
                    .unwrap()
                    .iter()
                    .map(|p| p.id)
                    .collect();
                let want = brute_force(&model, &query, 5);
                // Equal distances may order differently; compare distances.
                let dist = |ids: &[PointId]| -> Vec<f32> {
                    ids.iter()
                        .map(|id| Metric::L2.distance(&query, &model[id]))
                        .collect()
                };
                assert_eq!(dist(&got), dist(&want), "step {step}");
            }
            _ => {
                if step % 7 == 0 {
                    drop(c);
                    drop(db);
                    db = Database::open(dir.path(), options()).unwrap();
                }
            }
        }
    }
    let c = db.collection("c").unwrap();
    let c = c.read().unwrap();
    assert_eq!(c.len(), model.len());
    for (id, v) in &model {
        assert_eq!(c.get(*id), Some(v.as_slice()));
    }
}

#[test]
fn an_hnsw_collection_finds_every_live_point_after_churn() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), options()).unwrap();
    let c = db
        .create_collection("c", config(IndexConfig::Hnsw(HnswConfig::default())))
        .unwrap();
    let mut rng = Lcg(11);
    let mut model = BTreeMap::new();
    for _ in 0..4 {
        let batch: Vec<(PointId, Vec<f32>)> = (0..1_000)
            .map(|_| (rng.next() % 1_500, rng.vector()))
            .collect();
        let refs: Vec<(PointId, &[f32])> =
            batch.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        c.write().unwrap().upsert(&refs).unwrap();
        for (id, v) in batch {
            model.insert(id, v);
        }
        let doomed: Vec<PointId> = (0..300).map(|_| rng.next() % 1_500).collect();
        c.write().unwrap().delete(&doomed).unwrap();
        for id in doomed {
            model.remove(&id);
        }
    }
    drop(c);
    drop(db);

    let db = Database::open(dir.path(), options()).unwrap();
    let c = db.collection("c").unwrap();
    let c = c.read().unwrap();
    assert_eq!(c.len(), model.len());
    let found = model
        .iter()
        .filter(|(id, v)| {
            let hits = c.search(v, 3).unwrap();
            hits.iter().any(|p| p.id == **id) && hits.iter().all(|p| model.contains_key(&p.id))
        })
        .count();
    // Random data may hold duplicate vectors; nearly every point must still be found.
    assert!(
        found * 100 >= model.len() * 99,
        "{found} of {} found",
        model.len()
    );
}

#[test]
fn searches_run_while_another_thread_writes() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(Database::open(dir.path(), options()).unwrap());
    db.create_collection("c", config(IndexConfig::Hnsw(HnswConfig::default())))
        .unwrap();

    let writer = {
        let db = Arc::clone(&db);
        thread::spawn(move || {
            let mut rng = Lcg(3);
            for chunk in 0..20u64 {
                let batch: Vec<(PointId, Vec<f32>)> =
                    (0..100).map(|i| (chunk * 100 + i, rng.vector())).collect();
                let refs: Vec<(PointId, &[f32])> =
                    batch.iter().map(|(id, v)| (*id, v.as_slice())).collect();
                db.collection("c")
                    .unwrap()
                    .write()
                    .unwrap()
                    .upsert(&refs)
                    .unwrap();
            }
        })
    };
    let readers: Vec<_> = (0..4)
        .map(|seed| {
            let db = Arc::clone(&db);
            thread::spawn(move || {
                let mut rng = Lcg(100 + seed);
                for _ in 0..200 {
                    let hits = db
                        .collection("c")
                        .unwrap()
                        .read()
                        .unwrap()
                        .search(&rng.vector(), 5)
                        .unwrap();
                    assert!(hits.len() <= 5);
                }
            })
        })
        .collect();
    writer.join().unwrap();
    for reader in readers {
        reader.join().unwrap();
    }
    assert_eq!(db.collection("c").unwrap().read().unwrap().len(), 2_000);
}
