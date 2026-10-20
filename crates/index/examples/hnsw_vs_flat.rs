//! Exact search against HNSW, using the index crate directly.
//!
//! Builds a flat index and an HNSW graph over the same 20,000 clustered
//! vectors, then measures how much of the exact answer HNSW finds (recall)
//! and how fast, as `ef_search` trades speed for accuracy.
//!
//! ```sh
//! cargo run -p nearby-index --release --example hnsw_vs_flat
//! ```

use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use nearby_core::{Metric, PointId};
use nearby_index::{FlatIndex, Hnsw, HnswParams, VectorIndex};

const DIM: usize = 64;
const POINTS: usize = 20_000;
const QUERIES: usize = 200;
const K: usize = 10;

fn main() -> nearby_core::Result<()> {
    let data = clustered_vectors(POINTS + QUERIES, 1);
    let (points, queries) = data.split_at(POINTS);
    let batch: Vec<(PointId, &[f32])> = points
        .iter()
        .enumerate()
        .map(|(id, v)| (id as PointId, v.as_slice()))
        .collect();

    // The flat index compares every vector: slow, but always right.
    let mut flat = FlatIndex::with_capacity(DIM, Metric::L2, POINTS);
    for &(id, vector) in &batch {
        flat.add(id, vector)?;
    }

    // HNSW builds a graph, on every core; the graph is the same either way.
    let threads = std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    let start = Instant::now();
    let mut hnsw = Hnsw::with_capacity(DIM, Metric::L2, HnswParams::default(), POINTS)?;
    hnsw.insert_batch(&batch, threads)?;
    println!(
        "built HNSW over {POINTS} vectors in {:.2?} on {threads} threads\n",
        start.elapsed()
    );

    let (truth, flat_time) = timed(|| search_all(&flat, queries))?;
    println!("{:<14} {:>9} {:>12}", "index", "recall@10", "per query");
    println!(
        "{:<14} {:>9.3} {:>12.2?}",
        "flat",
        1.0,
        flat_time / QUERIES as u32
    );
    for ef in [10, 20, 40, 80, 160] {
        hnsw.set_ef_search(ef);
        let (found, time) = timed(|| search_all(&hnsw, queries))?;
        println!(
            "{:<14} {:>9.3} {:>12.2?}",
            format!("hnsw ef={ef}"),
            recall(&found, &truth),
            time / QUERIES as u32
        );
    }
    Ok(())
}

/// The ids of the `K` nearest points to each query.
fn search_all(
    index: &dyn VectorIndex,
    queries: &[Vec<f32>],
) -> nearby_core::Result<Vec<Vec<PointId>>> {
    queries
        .iter()
        .map(|q| Ok(index.search(q, K)?.into_iter().map(|p| p.id).collect()))
        .collect()
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

fn timed<T>(work: impl FnOnce() -> nearby_core::Result<T>) -> nearby_core::Result<(T, Duration)> {
    let start = Instant::now();
    let value = work()?;
    Ok((value, start.elapsed()))
}

/// `n` vectors scattered around 50 random centers, like embeddings that
/// form topics. Deterministic for a given `seed`.
fn clustered_vectors(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng(seed);
    let centers: Vec<Vec<f32>> = (0..50)
        .map(|_| (0..DIM).map(|_| rng.next() * 10.0).collect())
        .collect();
    (0..n)
        .map(|i| {
            let center = &centers[i % centers.len()];
            center.iter().map(|&c| c + rng.next() - 0.5).collect()
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
