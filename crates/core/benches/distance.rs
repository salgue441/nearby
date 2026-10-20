//! Distance kernels at common embedding sizes.
//!
//! ```sh
//! cargo bench -p nearby-core --bench distance
//! ```
//!
//! Throughput is reported in vector elements per second, so sizes compare
//! directly. The kernels dispatch at runtime: AVX2 + FMA where the CPU has
//! them, a portable loop otherwise.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use nearby_core::distance::{cosine_distance, dot, l2_squared};

/// Two deterministic pseudo-random vectors of `dim` values.
fn vectors(dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    let a = (0..dim).map(|_| next()).collect();
    let b = (0..dim).map(|_| next()).collect();
    (a, b)
}

fn distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance");
    for dim in [128, 384, 768, 1536] {
        let (a, b) = vectors(dim);
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::new("dot", dim), &dim, |bench, _| {
            bench.iter(|| dot(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("l2_squared", dim), &dim, |bench, _| {
            bench.iter(|| l2_squared(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("cosine", dim), &dim, |bench, _| {
            bench.iter(|| cosine_distance(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

criterion_group!(benches, distances);
criterion_main!(benches);
