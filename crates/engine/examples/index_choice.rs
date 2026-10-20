//! Choosing a collection index: flat, HNSW, PQ and IVF-PQ on the same data.
//!
//! Loads 100,000 64-d embedding-like vectors into one collection per index type,
//! then reports load time, search latency and recall against the exact
//! answer. Flat is exact but scans everything; HNSW is fast and accurate but
//! slowest to build; PQ compresses vectors 16x and re-ranks candidates
//! exactly; IVF-PQ also scans only the clusters nearest each query.
//!
//! ```sh
//! cargo run -p nearby-engine --release --example index_choice
//! ```

use std::time::{Duration, Instant};

use nearby_core::{Metric, PointId};
use nearby_engine::{
    Collection, CollectionConfig, Database, Durability, EngineOptions, HnswConfig, IndexConfig,
    PqConfig,
};

const DIM: usize = 64;
const POINTS: usize = 100_000;
const QUERIES: usize = 300;
const K: usize = 10;

fn main() -> nearby_core::Result<()> {
    let data = embeddings(POINTS + QUERIES, 7);
    let (points, queries) = data.split_at(POINTS);
    let batch: Vec<(PointId, &[f32])> = points
        .iter()
        .enumerate()
        .map(|(id, v)| (id as PointId, v.as_slice()))
        .collect();

    let dir = tempfile::tempdir()?;
    let options = EngineOptions {
        // A bulk load can leave syncing to the OS: writes survive a process
        // crash, though not a power cut.
        durability: Durability::NoSync,
        ..EngineOptions::default()
    };
    let db = Database::open(dir.path(), options)?;
    let choices = [
        ("flat", IndexConfig::Flat),
        ("hnsw", IndexConfig::Hnsw(HnswConfig::default())),
        (
            "pq",
            IndexConfig::Pq(PqConfig {
                lists: 0, // scan every code
                ..PqConfig::default()
            }),
        ),
        ("ivf-pq", IndexConfig::Pq(PqConfig::default())),
    ];

    let mut truth = Vec::new();
    println!(
        "{:<8} {:>10} {:>10} {:>10}",
        "index", "load", "per query", "recall@10"
    );
    for (name, index) in choices {
        let config = CollectionConfig {
            dim: DIM,
            metric: Metric::L2,
            index,
        };
        let handle = db.create_collection(name, config)?;
        let start = Instant::now();
        for chunk in batch.chunks(10_000) {
            handle.write().unwrap().upsert(chunk)?;
        }
        let load = start.elapsed();

        let collection = handle.read().unwrap();
        let (found, time) = search_all(&collection, queries)?;
        if truth.is_empty() {
            truth.clone_from(&found); // flat runs first and is exact
        }
        println!(
            "{name:<8} {:>10.2?} {:>10.2?} {:>10.3}",
            load,
            time / QUERIES as u32,
            recall(&found, &truth)
        );
    }
    Ok(())
}

/// The ids of the `K` nearest points to each query, and the time taken.
fn search_all(
    collection: &Collection,
    queries: &[Vec<f32>],
) -> nearby_core::Result<(Vec<Vec<PointId>>, Duration)> {
    let start = Instant::now();
    let found = queries
        .iter()
        .map(|q| Ok(collection.search(q, K)?.into_iter().map(|p| p.id).collect()))
        .collect::<nearby_core::Result<_>>()?;
    Ok((found, start.elapsed()))
}

/// The share of the true neighbors that were found.
fn recall(found: &[Vec<PointId>], truth: &[Vec<PointId>]) -> f64 {
    let hits: usize = found
        .iter()
        .zip(truth)
        .map(|(f, t)| f.iter().filter(|id| t.contains(id)).count())
        .sum();
    hits as f64 / (truth.len() * K) as f64
}

/// `n` vectors that mix 8 hidden factors, plus a little noise. Like real
/// embeddings, they vary along far fewer directions than they have
/// dimensions, which is what lets compression keep neighbors apart.
/// Deterministic for a given `seed`.
fn embeddings(n: usize, seed: u64) -> Vec<Vec<f32>> {
    const FACTORS: usize = 8;
    let mut rng = Rng(seed);
    let mixing: Vec<f32> = (0..DIM * FACTORS).map(|_| rng.next() * 2.0 - 1.0).collect();
    (0..n)
        .map(|_| {
            let factors: Vec<f32> = (0..FACTORS).map(|_| rng.next() * 2.0 - 1.0).collect();
            mixing
                .as_chunks::<FACTORS>()
                .0
                .iter()
                .map(|row| {
                    let signal: f32 = row.iter().zip(&factors).map(|(m, f)| m * f).sum();
                    signal + (rng.next() - 0.5) * 0.05
                })
                .collect()
        })
        .collect()
}

/// A tiny xorshift generator, so the example needs no extra crate.
struct Rng(u64);

impl Rng {
    /// A float in `[0, 1)`.
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}
