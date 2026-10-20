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

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, bail, ensure};
use nearby_core::{Metric, PointId};
use nearby_index::{FlatIndex, Hnsw, HnswParams, PqIndex, PqParams, ProductQuantizer, VectorIndex};

use crate::dataset::{Dataset, Vectors};
use crate::metrics::{LatencyStats, recall_at_k};

const USAGE: &str = "\
usage: nearby-bench [DATASET_DIR] [OPTIONS]

  DATASET_DIR              default: data/siftsmall
  --index LIST             indexes to benchmark, comma-separated: flat, hnsw, pq,
                           or `both` (flat,hnsw) or `all` (default: both)
  --k N                    neighbors per query (default: 10)
  --m N                    HNSW links per node (default: 16)
  --ef-construction N      HNSW build quality (default: 200)
  --ef N,N,...             HNSW ef_search values to sweep (default: 10,20,40,80,160,320)
                           (a search uses at least k, so smaller values run as k)
  --pq-subspaces N         PQ code bytes per vector; must divide the dimension (default: 16)
  --pq-train N             base vectors PQ is trained on (default: 100000)
  --threads N              threads that build HNSW and PQ (default: all available)";

/// Base vectors read from disk at a time while building. Small enough to
/// add no meaningful memory, large enough that timing each chunk is cheap.
const BUILD_CHUNK: usize = 4096;

/// Which indexes to build and query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IndexSet {
    flat: bool,
    hnsw: bool,
    pq: bool,
}

impl IndexSet {
    /// The default: the exact index and HNSW.
    const BOTH: Self = Self {
        flat: true,
        hnsw: true,
        pq: false,
    };

    /// Parses an `--index` value: a comma-separated list of `flat`, `hnsw`,
    /// `pq`, `both` (flat and hnsw) and `all`.
    ///
    /// # Errors
    ///
    /// On an unknown name, including an empty one.
    fn parse(list: &str) -> anyhow::Result<Self> {
        let mut set = Self {
            flat: false,
            hnsw: false,
            pq: false,
        };
        for name in list.split(',').map(str::trim) {
            match name {
                "flat" => set.flat = true,
                "hnsw" => set.hnsw = true,
                "pq" => set.pq = true,
                "both" => (set.flat, set.hnsw) = (true, true),
                "all" => (set.flat, set.hnsw, set.pq) = (true, true, true),
                other => bail!(
                    "unknown index `{other}`

{USAGE}"
                ),
            }
        }
        Ok(set)
    }
}

/// Parsed command line. See [`USAGE`] for each option's meaning and default.
#[derive(Debug, PartialEq)]
struct Args {
    dir: PathBuf,
    indexes: IndexSet,
    k: usize,
    m: usize,
    ef_construction: usize,
    /// One HNSW result row is reported per value.
    ef_search: Vec<usize>,
    pq_subspaces: usize,
    /// How many base vectors, from the front of the file, train PQ.
    pq_train: usize,
    /// Threads that build the HNSW and PQ indexes.
    threads: NonZeroUsize,
}

/// One line of the results table: a label, the mean recall@k and latencies.
type Row = (String, (f64, LatencyStats));

fn main() -> anyhow::Result<()> {
    let Some(args) = parse_args(std::env::args().skip(1))? else {
        println!("{USAGE}");
        return Ok(());
    };
    let data = Dataset::load(&args.dir, args.k)?;
    println!(
        "dataset  {}: {} base vectors, {} queries, dim {}\n",
        data.name,
        data.base_len(),
        data.queries.len(),
        data.dim(),
    );

    // Each index is built from its own pass over the base file, queried, and
    // dropped before the next one is built, so peak memory is that of the
    // largest single index.
    let mut rows = Vec::new();
    if args.indexes.flat {
        rows.extend(bench_flat(&data, &args)?);
    }
    if args.indexes.hnsw {
        rows.extend(bench_hnsw(&data, &args)?);
    }
    if args.indexes.pq {
        rows.extend(bench_pq(&data, &args)?);
    }
    print_rows(&rows, args.k);
    Ok(())
}

/// Builds and queries the exact index.
fn bench_flat(data: &Dataset, args: &Args) -> anyhow::Result<Vec<Row>> {
    let mut flat = FlatIndex::with_capacity(data.dim(), Metric::L2, data.base_len());
    let build_time = build(data, |batch| {
        batch
            .iter()
            .try_for_each(|&(id, vector)| flat.add(id, vector))
    })?;
    println!("build    flat: {build_time:.2?}");
    Ok(vec![(
        "flat".to_owned(),
        run_queries(&flat, &data.queries, &data.truth, args.k)?,
    )])
}

/// Builds the HNSW index, then queries it once per `ef_search` value.
fn bench_hnsw(data: &Dataset, args: &Args) -> anyhow::Result<Vec<Row>> {
    let params = HnswParams {
        m: args.m,
        m0: 2 * args.m,
        ef_construction: args.ef_construction,
        ..HnswParams::default()
    };
    let mut hnsw = Hnsw::with_capacity(data.dim(), Metric::L2, params, data.base_len())?;
    let build_time = build(data, |batch| hnsw.insert_batch(batch, args.threads))?;
    println!(
        "build    hnsw (m={}, ef_construction={}, threads={}): {build_time:.2?}, {} layers",
        args.m,
        args.ef_construction,
        args.threads,
        hnsw.max_level() + 1
    );
    let sweep = effective_ef(&args.ef_search, args.k);
    if sweep != args.ef_search {
        println!(
            "note     a search uses ef >= k, so values below k = {} run as {}",
            args.k, args.k
        );
    }
    let mut rows = Vec::with_capacity(sweep.len());
    for ef in sweep {
        hnsw.set_ef_search(ef);
        rows.push((
            format!("hnsw ef={ef}"),
            run_queries(&hnsw, &data.queries, &data.truth, args.k)?,
        ));
    }
    Ok(rows)
}

/// Trains a product quantizer on the front of the base file, encodes every
/// base vector with it, and queries the codes.
fn bench_pq(data: &Dataset, args: &Args) -> anyhow::Result<Vec<Row>> {
    let training = data.read_base_prefix(args.pq_train)?;
    let params = PqParams {
        subspaces: args.pq_subspaces,
        ..PqParams::default()
    };
    let start = Instant::now();
    let quantizer = ProductQuantizer::train(
        data.dim(),
        Metric::L2,
        &params,
        &training.data,
        args.threads,
    )?;
    let train_time = start.elapsed();
    let training_len = training.len();
    drop(training);

    let mut pq = PqIndex::with_capacity(quantizer, data.base_len());
    let encode_time = build(data, |batch| pq.add_batch(batch, args.threads))?;
    println!(
        "build    pq (subspaces={}, threads={}): train {train_time:.2?} on {training_len} vectors, \
         encode {encode_time:.2?}, {} bytes/vector",
        args.pq_subspaces,
        args.threads,
        pq.bytes_per_vector()
    );
    Ok(vec![(
        "pq".to_owned(),
        run_queries(&pq, &data.queries, &data.truth, args.k)?,
    )])
}

/// Prints the results table.
fn print_rows(rows: &[Row], k: usize) {
    println!(
        "\n{:<14} {:>9} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "index",
        format!("recall@{k}"),
        "qps",
        "mean",
        "p50",
        "p95",
        "p99",
        "max"
    );
    for (label, (recall, stats)) in rows {
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
}

/// Streams the base vectors into an index through `insert`, one chunk of
/// `(id, vector)` pairs at a time, where each id is the vector's position in
/// the file. Returns the time spent in `insert` alone, excluding file reads.
///
/// Inserts are timed per chunk, not per vector: a flat insert is a short
/// copy, and timing each one would mostly measure the timer. Chunks also let
/// an index insert many vectors in parallel.
///
/// # Errors
///
/// If reading the base file or an insert fails.
fn build(
    data: &Dataset,
    mut insert: impl FnMut(&[(PointId, &[f32])]) -> nearby_core::Result<()>,
) -> anyhow::Result<Duration> {
    let dim = data.dim();
    let mut elapsed = Duration::ZERO;
    data.for_each_base_chunk(BUILD_CHUNK, |first, chunk| {
        let batch: Vec<(PointId, &[f32])> = chunk
            .chunks_exact(dim)
            .enumerate()
            .map(|(offset, vector)| ((first + offset) as PointId, vector))
            .collect();
        let start = Instant::now();
        insert(&batch)?;
        elapsed += start.elapsed();
        Ok(())
    })?;
    Ok(elapsed)
}

/// The `ef_search` values that will actually run, in the order requested.
///
/// A search uses `max(ef_search, k)`, so values below `k` are raised to `k`.
/// Duplicates that result are dropped, so every reported row is distinct and
/// labeled with the `ef` it really used.
fn effective_ef(requested: &[usize], k: usize) -> Vec<usize> {
    let mut sweep = Vec::with_capacity(requested.len());
    for ef in requested.iter().map(|&ef| ef.max(k)) {
        if !sweep.contains(&ef) {
            sweep.push(ef);
        }
    }
    sweep
}

/// Formats a latency for the results table, e.g. `412.35µs`.
fn fmt_duration(d: std::time::Duration) -> String {
    format!("{d:.2?}")
}

/// Runs every query once and returns the mean recall@k and latency stats.
///
/// `truths[q]` is query `q`'s ground truth, closest first.
///
/// # Errors
///
/// If a search fails, or the dataset has no queries.
fn run_queries(
    index: &dyn VectorIndex,
    queries: &Vectors,
    truths: &[Vec<u32>],
    k: usize,
) -> anyhow::Result<(f64, LatencyStats)> {
    let mut latencies = Vec::with_capacity(queries.len());
    let mut recall_sum = 0.0;
    for (query, truth) in queries.iter().zip(truths) {
        let start = Instant::now();
        let results = index.search(query, k)?;
        latencies.push(start.elapsed());
        recall_sum += recall_at_k(&results, truth, k);
    }
    let stats = LatencyStats::from_samples(latencies).context("the query file is empty")?;
    Ok((recall_sum / queries.len() as f64, stats))
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
/// `--threads 0`, or an `--ef` list that is empty or contains 0. Errors
/// about unknown or incomplete options include [`USAGE`].
fn parse_args(raw: impl IntoIterator<Item = String>) -> anyhow::Result<Option<Args>> {
    let mut args = Args {
        dir: PathBuf::from("data/siftsmall"),
        indexes: IndexSet::BOTH,
        k: 10,
        m: 16,
        ef_construction: 200,
        ef_search: vec![10, 20, 40, 80, 160, 320],
        pq_subspaces: 16,
        pq_train: 100_000,
        threads: default_threads(),
    };
    let mut it = raw.into_iter();
    while let Some(arg) = it.next() {
        let mut value = |flag: &str| {
            it.next()
                .with_context(|| format!("{flag} needs a value\n\n{USAGE}"))
        };
        match arg.as_str() {
            "--index" => args.indexes = IndexSet::parse(&value("--index")?)?,
            "--k" => args.k = parse_number(&value("--k")?)?,
            "--m" => args.m = parse_number(&value("--m")?)?,
            "--ef-construction" => {
                args.ef_construction = parse_number(&value("--ef-construction")?)?;
            }
            "--pq-subspaces" => args.pq_subspaces = parse_number(&value("--pq-subspaces")?)?,
            "--pq-train" => args.pq_train = parse_number(&value("--pq-train")?)?,
            "--threads" => {
                args.threads = NonZeroUsize::new(parse_number(&value("--threads")?)?)
                    .context("--threads must be at least 1")?;
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
    ensure!(
        args.ef_search.iter().all(|&ef| ef > 0),
        "--ef values must be at least 1"
    );
    Ok(Some(args))
}

/// Every thread the machine offers, or 1 if that can't be determined.
fn default_threads() -> NonZeroUsize {
    thread::available_parallelism().unwrap_or(NonZeroUsize::MIN)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> anyhow::Result<Option<Args>> {
        parse_args(args.iter().map(|&s| s.to_owned()))
    }

    #[test]
    fn defaults() {
        let args = parse(&[]).unwrap().unwrap();
        assert_eq!(
            args,
            Args {
                dir: PathBuf::from("data/siftsmall"),
                indexes: IndexSet::BOTH,
                k: 10,
                m: 16,
                ef_construction: 200,
                ef_search: vec![10, 20, 40, 80, 160, 320],
                pq_subspaces: 16,
                pq_train: 100_000,
                threads: default_threads(),
            }
        );
    }

    #[test]
    fn every_option() {
        let args = parse(&[
            "data/sift",
            "--index",
            "hnsw",
            "--k",
            "100",
            "--m",
            "32",
            "--ef-construction",
            "400",
            "--threads",
            "3",
            "--pq-subspaces",
            "32",
            "--pq-train",
            "5000",
            "--ef",
            "50, 100,200",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(args.dir, PathBuf::from("data/sift"));
        assert_eq!(
            args.indexes,
            IndexSet {
                flat: false,
                hnsw: true,
                pq: false
            }
        );
        assert_eq!((args.k, args.m, args.ef_construction), (100, 32, 400));
        assert_eq!(args.ef_search, vec![50, 100, 200]);
        assert_eq!(args.threads.get(), 3);
        assert_eq!((args.pq_subspaces, args.pq_train), (32, 5000));
    }

    #[test]
    fn options_may_come_before_the_directory() {
        let args = parse(&["--index", "flat", "data/gist"]).unwrap().unwrap();
        assert_eq!(args.dir, PathBuf::from("data/gist"));
        assert!(args.indexes.flat && !args.indexes.hnsw && !args.indexes.pq);
    }

    #[test]
    fn index_lists_combine() {
        let set = |list: &str| IndexSet::parse(list).unwrap();
        let all = IndexSet {
            flat: true,
            hnsw: true,
            pq: true,
        };
        assert_eq!(set("all"), all);
        assert_eq!(set("pq, both"), all);
        assert_eq!(set("both"), IndexSet::BOTH);
        assert_eq!(
            set("pq"),
            IndexSet {
                flat: false,
                hnsw: false,
                pq: true
            }
        );
    }

    #[test]
    fn help_returns_none() {
        assert!(parse(&["--help"]).unwrap().is_none());
        assert!(parse(&["data/sift", "-h"]).unwrap().is_none());
    }

    #[test]
    fn errors_explain_the_problem() {
        let cases: [(&[&str], &str); 11] = [
            (&["--index", "ivf"], "unknown index `ivf`"),
            (&["--index", "flat,,pq"], "unknown index ``"),
            (&["--pq-train"], "--pq-train needs a value"),
            (&["--frobnicate"], "unknown option `--frobnicate`"),
            (&["--k"], "--k needs a value"),
            (&["--k", "ten"], "`ten` is not"),
            (&["--k", "-3"], "`-3` is not"),
            (&["--k", "0"], "--k must be at least 1"),
            (&["--ef", "10,,20"], "`` is not"),
            (&["--ef", "10,0"], "--ef values must be at least 1"),
            (&["--threads", "0"], "--threads must be at least 1"),
        ];
        for (args, expected) in cases {
            let err = format!("{:#}", parse(args).unwrap_err());
            assert!(err.contains(expected), "{args:?}: got {err}");
        }
    }

    #[test]
    fn usage_is_included_in_option_errors() {
        let err = format!("{:#}", parse(&["--bogus"]).unwrap_err());
        assert!(err.contains("usage: nearby-bench"), "got: {err}");
    }

    #[test]
    fn run_queries_on_an_exact_index_has_perfect_recall() {
        let mut flat = FlatIndex::new(1, Metric::L2);
        for (id, x) in [0.0, 10.0, 20.0, 30.0].into_iter().enumerate() {
            flat.add(id as PointId, &[x]).unwrap();
        }
        let queries = Vectors {
            dim: 1,
            data: vec![1.0, 29.0],
        };
        let truth = vec![vec![0, 1], vec![3, 2]];

        let (recall, stats) = run_queries(&flat, &queries, &truth, 2).unwrap();
        assert!((recall - 1.0).abs() < 1e-9);
        assert_eq!(stats.count, 2);
    }

    #[test]
    fn effective_ef_raises_values_below_k_and_drops_repeats() {
        assert_eq!(effective_ef(&[10, 20, 40, 80], 10), vec![10, 20, 40, 80]);
        assert_eq!(effective_ef(&[10, 20, 40, 160], 50), vec![50, 160]);
        assert_eq!(
            effective_ef(&[200, 100], 1),
            vec![200, 100],
            "order is kept"
        );
        assert_eq!(effective_ef(&[5, 5], 1), vec![5]);
    }

    #[test]
    fn build_inserts_every_base_vector_with_its_position_as_id() {
        // 5000 one-dimensional vectors: more than one BUILD_CHUNK.
        let n = 5_000u32;
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("line");
        std::fs::create_dir_all(&dir).unwrap();
        let record = |value: [u8; 4]| [1u32.to_le_bytes(), value].concat();
        let base: Vec<u8> = (0..n)
            .flat_map(|i| record((i as f32).to_le_bytes()))
            .collect();
        std::fs::write(dir.join("line_base.fvecs"), base).unwrap();
        std::fs::write(dir.join("line_query.fvecs"), record(0f32.to_le_bytes())).unwrap();
        std::fs::write(
            dir.join("line_groundtruth.ivecs"),
            record(0u32.to_le_bytes()),
        )
        .unwrap();
        let data = Dataset::load(&dir, 1).unwrap();

        let mut flat = FlatIndex::new(1, Metric::L2);
        build(&data, |batch| {
            batch.iter().try_for_each(|&(id, v)| flat.add(id, v))
        })
        .unwrap();
        assert_eq!(flat.len(), n as usize);
        assert_eq!(flat.get(4_321), Some(&[4_321.0][..]));
    }

    #[test]
    fn fmt_duration_uses_two_decimals() {
        assert_eq!(
            fmt_duration(std::time::Duration::from_micros(1_500)),
            "1.50ms"
        );
    }
}
