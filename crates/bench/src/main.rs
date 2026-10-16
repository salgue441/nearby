//! Recall and throughput benchmark on the TEXMEX SIFT datasets.
//!
//! Usage: `cargo run -p nearby-bench --release -- [DATASET_DIR] [OPTIONS]`
//! (run with `--help` for the options).
//!
//! The dataset directory defaults to `data/siftsmall` and must contain
//! `<name>_base.fvecs`, `<name>_query.fvecs` and `<name>_groundtruth.ivecs`,
//! where `<name>` is the directory's own name.

mod dataset;
mod metrics;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, bail, ensure};
use nearby_core::{Metric, PointId};
use nearby_index::{FlatIndex, Hnsw, HnswParams, VectorIndex};

use crate::dataset::Dataset;
use crate::metrics::{LatencyStats, recall_at_k};

const USAGE: &str = "\
usage: nearby-bench [DATASET_DIR] [OPTIONS]

  DATASET_DIR              default: data/siftsmall
  --index flat|hnsw|both   which index to benchmark (default: both)
  --k N                    neighbors per query (default: 10)
  --m N                    HNSW links per node (default: 16)
  --ef-construction N      HNSW build quality (default: 200)
  --ef N,N,...             HNSW ef_search values to sweep (default: 10,20,40,80,160,320)";

/// Which indexes to build and query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexChoice {
    Flat,
    Hnsw,
    Both,
}

/// Parsed command line. See [`USAGE`] for each option's meaning and default.
#[derive(Debug, PartialEq)]
struct Args {
    dir: PathBuf,
    index: IndexChoice,
    k: usize,
    m: usize,
    ef_construction: usize,
    /// One result row is reported per value.
    ef_search: Vec<usize>,
}

fn main() -> anyhow::Result<()> {
    let Some(args) = parse_args(std::env::args().skip(1))? else {
        println!("{USAGE}");
        return Ok(());
    };
    let data = Dataset::load(&args.dir, args.k)?;
    println!(
        "dataset  {}: {} base vectors, {} queries, dim {}\n",
        data.name,
        data.base.len(),
        data.queries.len(),
        data.base.dim
    );

    let mut rows = Vec::new();

    if args.index != IndexChoice::Hnsw {
        let start = Instant::now();
        let mut flat = FlatIndex::with_capacity(data.base.dim, Metric::L2, data.base.len());
        for (i, vector) in data.base.iter().enumerate() {
            flat.add(i as PointId, vector)?;
        }
        println!("build    flat: {:.2?}", start.elapsed());
        rows.push(("flat".to_owned(), run_queries(&flat, &data, args.k)?));
    }

    if args.index != IndexChoice::Flat {
        let params = HnswParams {
            m: args.m,
            m0: 2 * args.m,
            ef_construction: args.ef_construction,
            ..HnswParams::default()
        };
        let start = Instant::now();
        let mut hnsw = Hnsw::with_capacity(data.base.dim, Metric::L2, params, data.base.len())?;
        for (i, vector) in data.base.iter().enumerate() {
            hnsw.insert(i as PointId, vector)?;
        }
        println!(
            "build    hnsw (m={}, ef_construction={}): {:.2?}, {} layers",
            args.m,
            args.ef_construction,
            start.elapsed(),
            hnsw.max_level() + 1
        );
        for &ef in &args.ef_search {
            hnsw.set_ef_search(ef);
            rows.push((format!("hnsw ef={ef}"), run_queries(&hnsw, &data, args.k)?));
        }
    }

    println!(
        "\n{:<14} {:>9} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "index",
        format!("recall@{}", args.k),
        "qps",
        "mean",
        "p50",
        "p95",
        "p99",
        "max"
    );
    for (label, (recall, stats)) in &rows {
        println!(
            "{label:<14} {recall:>9.4} {:>9.0} {:>10} {:>10} {:>10} {:>10} {:>10}",
            stats.qps(),
            fmt_duration(stats.mean),
            fmt_duration(stats.p50),
            fmt_duration(stats.p95),
            fmt_duration(stats.p99),
            fmt_duration(stats.max)
        );
    }
    Ok(())
}

/// Formats a latency for the results table, e.g. `412.35µs`.
fn fmt_duration(d: std::time::Duration) -> String {
    format!("{d:.2?}")
}

/// Runs every query once and returns the mean recall@k and latency stats.
///
/// # Errors
///
/// If a search fails, or the dataset has no queries.
fn run_queries(
    index: &dyn VectorIndex,
    data: &Dataset,
    k: usize,
) -> anyhow::Result<(f64, LatencyStats)> {
    let mut latencies = Vec::with_capacity(data.queries.len());
    let mut recall_sum = 0.0;
    for (query, truth) in data.queries.iter().zip(&data.truth) {
        let start = Instant::now();
        let results = index.search(query, k)?;
        latencies.push(start.elapsed());
        recall_sum += recall_at_k(&results, truth, k);
    }
    let stats = LatencyStats::from_samples(latencies).context("the query file is empty")?;
    Ok((recall_sum / data.queries.len() as f64, stats))
}

/// Parses command-line arguments (without the program name).
///
/// Returns `None` if `-h` or `--help` was given, so the caller can print
/// [`USAGE`] and exit. Taking the arguments as a parameter, instead of
/// reading `std::env::args()` here, keeps this testable.
///
/// # Errors
///
/// On an unknown option or index, a missing or non-numeric value, `--k 0`,
/// or an empty `--ef` list. Messages include [`USAGE`].
fn parse_args(raw: impl IntoIterator<Item = String>) -> anyhow::Result<Option<Args>> {
    let mut args = Args {
        dir: PathBuf::from("data/siftsmall"),
        index: IndexChoice::Both,
        k: 10,
        m: 16,
        ef_construction: 200,
        ef_search: vec![10, 20, 40, 80, 160, 320],
    };
    let mut it = raw.into_iter();
    while let Some(arg) = it.next() {
        let mut value = |flag: &str| {
            it.next()
                .with_context(|| format!("{flag} needs a value\n\n{USAGE}"))
        };
        match arg.as_str() {
            "--index" => {
                args.index = match value("--index")?.as_str() {
                    "flat" => IndexChoice::Flat,
                    "hnsw" => IndexChoice::Hnsw,
                    "both" => IndexChoice::Both,
                    other => bail!("unknown index `{other}`\n\n{USAGE}"),
                }
            }
            "--k" => args.k = parse_number(&value("--k")?)?,
            "--m" => args.m = parse_number(&value("--m")?)?,
            "--ef-construction" => {
                args.ef_construction = parse_number(&value("--ef-construction")?)?;
            }
            "--ef" => {
                args.ef_search = value("--ef")?
                    .split(',')
                    .map(parse_number)
                    .collect::<anyhow::Result<_>>()?;
            }
            "-h" | "--help" => return Ok(None),
            flag if flag.starts_with('-') => bail!("unknown option `{flag}`\n\n{USAGE}"),
            _ => args.dir = PathBuf::from(arg),
        }
    }
    ensure!(args.k > 0, "--k must be at least 1");
    ensure!(!args.ef_search.is_empty(), "--ef needs at least one value");
    Ok(Some(args))
}

/// Parses one numeric option value, ignoring surrounding whitespace.
///
/// # Errors
///
/// If `s` is not a non-negative integer that fits in `usize`.
fn parse_number(s: &str) -> anyhow::Result<usize> {
    s.trim()
        .parse()
        .with_context(|| format!("`{s}` is not a positive integer"))
}
