//! Collections end to end: batched writes, and searches with and without
//! payload filters.
//!
//! ```sh
//! cargo bench -p nearby-engine --bench collection
//! ```
//!
//! Writes use `Durability::NoSync`, so they measure the engine (log
//! encoding, the id map, indexing) rather than the disk's sync latency.

use std::hint::black_box;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use nearby_core::{Metric, PointId};
use nearby_engine::{
    CollectionConfig, CollectionHandle, Database, Durability, EngineOptions, Filter, IndexConfig,
    Payload, Range, Value,
};
use tempfile::TempDir;

const DIM: usize = 64;

fn vectors(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    (0..n).map(|_| (0..DIM).map(|_| next()).collect()).collect()
}

/// A payload whose `bucket` is `id % 100`, so `bucket < p` keeps p%.
fn payload(id: usize) -> Payload {
    Payload::from([
        ("bucket".to_owned(), Value::Number((id % 100) as f64)),
        (
            "lang".to_owned(),
            Value::String(if id.is_multiple_of(2) { "en" } else { "es" }.to_owned()),
        ),
    ])
}

/// A fresh database with one empty collection.
fn collection(index: IndexConfig) -> (TempDir, Database, CollectionHandle) {
    let dir = tempfile::tempdir().unwrap();
    let options = EngineOptions {
        durability: Durability::NoSync,
        ..EngineOptions::default()
    };
    let db = Database::open(dir.path(), options).unwrap();
    let config = CollectionConfig {
        dim: DIM,
        metric: Metric::L2,
        index,
    };
    let handle = db.create_collection("bench", config).unwrap();
    (dir, db, handle)
}

fn writes(c: &mut Criterion) {
    let data = vectors(1_000, 1);
    let payloads: Vec<Payload> = (0..data.len()).map(payload).collect();
    let batch: Vec<(PointId, &[f32], Option<&Payload>)> = data
        .iter()
        .zip(&payloads)
        .enumerate()
        .map(|(id, (v, p))| (id as PointId, v.as_slice(), Some(p)))
        .collect();

    let mut group = c.benchmark_group("upsert 1000 points with payloads");
    group.throughput(Throughput::Elements(batch.len() as u64));
    for (name, index) in [
        ("flat", IndexConfig::Flat),
        ("hnsw", IndexConfig::default()),
    ] {
        group.bench_function(name, |bench| {
            bench.iter_batched(
                || collection(index.clone()),
                |state| {
                    state
                        .2
                        .write()
                        .unwrap()
                        .upsert_with_payloads(black_box(&batch))
                        .unwrap();
                    state // dropped outside the timing
                },
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

fn filtered_searches(c: &mut Criterion) {
    const POINTS: usize = 20_000;
    let data = vectors(POINTS + 100, 2);
    let (points, queries) = data.split_at(POINTS);
    let payloads: Vec<Payload> = (0..POINTS).map(payload).collect();
    let batch: Vec<(PointId, &[f32], Option<&Payload>)> = points
        .iter()
        .zip(&payloads)
        .enumerate()
        .map(|(id, (v, p))| (id as PointId, v.as_slice(), Some(p)))
        .collect();
    let (_dir, _db, handle) = collection(IndexConfig::default());
    handle
        .write()
        .unwrap()
        .upsert_with_payloads(&batch)
        .unwrap();
    let collection = handle.read().unwrap();

    let below = |p: f64| Filter::Range {
        field: "bucket".to_owned(),
        range: Range {
            lt: Some(p),
            ..Range::default()
        },
    };
    let cases = [
        ("no filter", None),
        ("filter keeps 100%", Some(below(100.0))),
        ("filter keeps 50%", Some(below(50.0))),
        ("filter keeps 10%", Some(below(10.0))),
        ("filter keeps 1%", Some(below(1.0))),
    ];
    let mut group = c.benchmark_group("hnsw search 20k x 64-d, k=10");
    for (name, filter) in &cases {
        let mut next = queries.iter().cycle();
        group.bench_function(*name, |bench| {
            bench.iter(|| {
                collection
                    .search_where(black_box(next.next().unwrap()), 10, filter.as_ref())
                    .unwrap()
            });
        });
    }
    group.finish();

    let filter = Filter::And(vec![
        below(50.0),
        Filter::Eq {
            field: "lang".to_owned(),
            value: Value::String("en".to_owned()),
        },
    ]);
    let sample = payload(42);
    c.bench_function("filter matches (and of range and eq)", |bench| {
        bench.iter(|| black_box(&filter).matches(Some(black_box(&sample))));
    });
}

criterion_group!(benches, writes, filtered_searches);
criterion_main!(benches);
