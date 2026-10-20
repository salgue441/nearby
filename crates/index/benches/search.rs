//! Searching and building each index over the same 20,000 vectors.
//!
//! ```sh
//! cargo bench -p nearby-index --bench search
//! ```
//!
//! `search` measures one query (k = 10) per iteration, cycling through 100
//! queries so caches see a realistic mix. `build` measures building an HNSW
//! graph and encoding PQ codes. For recall and latency percentiles on real
//! datasets, use `nearby-bench` instead.

use std::hint::black_box;
use std::num::NonZeroUsize;

use criterion::{Criterion, criterion_group, criterion_main};
use nearby_core::{Metric, PointId};
use nearby_index::{
    FlatIndex, Hnsw, HnswParams, IvfPqIndex, IvfPqParams, PqIndex, PqParams, ProductQuantizer,
    VectorIndex,
};

const DIM: usize = 128;
const POINTS: usize = 20_000;
const QUERIES: usize = 100;
const K: usize = 10;

/// `n` vectors mixing 16 hidden factors, plus a little noise: like real
/// embeddings, they vary along far fewer directions than they have values.
fn embeddings(n: usize, seed: u64) -> Vec<Vec<f32>> {
    const FACTORS: usize = 16;
    let mut state = seed;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    let mixing: Vec<f32> = (0..DIM * FACTORS).map(|_| next()).collect();
    (0..n)
        .map(|_| {
            let factors: Vec<f32> = (0..FACTORS).map(|_| next()).collect();
            mixing
                .as_chunks::<FACTORS>()
                .0
                .iter()
                .map(|row| {
                    row.iter().zip(&factors).map(|(m, f)| m * f).sum::<f32>() + next() * 0.02
                })
                .collect()
        })
        .collect()
}

fn threads() -> NonZeroUsize {
    std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN)
}

fn search(c: &mut Criterion) {
    let data = embeddings(POINTS + QUERIES, 1);
    let (points, queries) = data.split_at(POINTS);
    let batch: Vec<(PointId, &[f32])> = points
        .iter()
        .enumerate()
        .map(|(id, v)| (id as PointId, v.as_slice()))
        .collect();
    let training: Vec<f32> = points.iter().take(10_000).flatten().copied().collect();

    let mut flat = FlatIndex::with_capacity(DIM, Metric::L2, POINTS);
    for &(id, v) in &batch {
        flat.add(id, v).unwrap();
    }
    let mut hnsw = Hnsw::new(DIM, Metric::L2, HnswParams::default()).unwrap();
    hnsw.insert_batch(&batch, threads()).unwrap();
    let quantizer =
        ProductQuantizer::train(DIM, Metric::L2, &PqParams::default(), &training, threads())
            .unwrap();
    let mut pq = PqIndex::new(quantizer);
    pq.add_batch(&batch, threads()).unwrap();
    let ivf_params = IvfPqParams {
        lists: 128,
        probe: 8,
        ..IvfPqParams::default()
    };
    let mut ivf = IvfPqIndex::train(DIM, Metric::L2, &ivf_params, &training, threads()).unwrap();
    ivf.add_batch(&batch, threads()).unwrap();

    let indexes: [(&str, &dyn VectorIndex); 4] = [
        ("flat", &flat),
        ("hnsw ef=64", &hnsw),
        ("pq", &pq),
        ("ivf-pq probe=8", &ivf),
    ];
    let mut group = c.benchmark_group("search 20k x 128-d, k=10");
    for (name, index) in indexes {
        let mut next = queries.iter().cycle();
        group.bench_function(name, |bench| {
            bench.iter(|| index.search(black_box(next.next().unwrap()), K).unwrap());
        });
    }
    group.finish();
}

fn build(c: &mut Criterion) {
    let data = embeddings(5_000, 2);
    let batch: Vec<(PointId, &[f32])> = data
        .iter()
        .enumerate()
        .map(|(id, v)| (id as PointId, v.as_slice()))
        .collect();
    let training: Vec<f32> = data.iter().flatten().copied().collect();
    let quantizer =
        ProductQuantizer::train(DIM, Metric::L2, &PqParams::default(), &training, threads())
            .unwrap();

    let mut group = c.benchmark_group("build 5k x 128-d");
    group.sample_size(10);
    group.bench_function("hnsw insert_batch", |bench| {
        bench.iter(|| {
            let mut hnsw = Hnsw::new(DIM, Metric::L2, HnswParams::default()).unwrap();
            hnsw.insert_batch(black_box(&batch), threads()).unwrap();
            hnsw
        });
    });
    group.bench_function("pq encode", |bench| {
        bench.iter(|| {
            let mut pq = PqIndex::new(quantizer.clone());
            pq.add_batch(black_box(&batch), threads()).unwrap();
            pq
        });
    });
    group.finish();
}

criterion_group!(benches, search, build);
criterion_main!(benches);
