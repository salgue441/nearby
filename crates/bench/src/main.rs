//! Recall and throughput benchmark on the TEXMEX SIFT datasets.
//!
//! Usage: `cargo run -p nearby-bench --release -- [dataset dir] [k]`
//!
//! The dataset directory defaults to `data/siftsmall` and must contain
//! `<name>_base.fvecs`, `<name>_query.fvecs` and `<name>_groundtruth.ivecs`,
//! where `<name>` is the directory's own name.

mod dataset;
mod metrics;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, ensure};
use nearby_core::{Metric, PointId};
use nearby_index::{FlatIndex, VectorIndex};

use crate::metrics::{LatencyStats, recall_at_k};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .map_or_else(|| PathBuf::from("data/siftsmall"), PathBuf::from);
    let k: usize = args
        .next()
        .map(|s| s.parse())
        .transpose()
        .context("k must be a positive integer")?
        .unwrap_or(10);
    ensure!(k > 0, "k must be at least 1");

    // ---- Load ---------------------------------------------------------------
    let name = dir
        .file_name()
        .and_then(|n| n.to_str())
        .context("dataset path has no directory name")?;
    let base = dataset::read_fvecs(&dir.join(format!("{name}_base.fvecs")))?;
    let queries = dataset::read_fvecs(&dir.join(format!("{name}_query.fvecs")))?;
    let truth = dataset::read_ivecs(&dir.join(format!("{name}_groundtruth.ivecs")))?;

    ensure!(
        queries.dim == base.dim,
        "queries have dimension {}, base vectors have {}",
        queries.dim,
        base.dim
    );
    ensure!(
        truth.len() == queries.len(),
        "{} ground-truth rows for {} queries",
        truth.len(),
        queries.len()
    );
    ensure!(
        truth.iter().all(|row| row.len() >= k),
        "ground truth has fewer than k = {k} neighbors per query"
    );

    println!(
        "dataset    {name}: {} base vectors, {} queries, dim {}",
        base.len(),
        queries.len(),
        base.dim
    );

    // ---- Build --------------------------------------------------------------
    let start = Instant::now();
    let mut index = FlatIndex::with_capacity(base.dim, Metric::L2, base.len());
    for (i, vector) in base.iter().enumerate() {
        index.add(i as PointId, vector)?;
    }
    let build_time = start.elapsed();

    // ---- Search -------------------------------------------------------------
    let mut latencies = Vec::with_capacity(queries.len());
    let mut recall_sum = 0.0;
    for (query, true_ids) in queries.iter().zip(&truth) {
        let start = Instant::now();
        let results = index.search(query, k)?;
        latencies.push(start.elapsed());
        recall_sum += recall_at_k(&results, true_ids, k);
    }
    let recall = recall_sum / queries.len() as f64;
    let stats = LatencyStats::from_samples(latencies).context("the query file is empty")?;

    // ---- Report -------------------------------------------------------------
    println!("index      flat (exact), L2");
    println!("build      {build_time:.2?}");
    println!("recall@{k:<3} {recall:.4}");
    println!("qps        {:.0}", stats.qps());
    println!(
        "latency    mean {:.2?} | p50 {:.2?} | p95 {:.2?} | p99 {:.2?} | max {:.2?}",
        stats.mean, stats.p50, stats.p95, stats.p99, stats.max
    );

    Ok(())
}
