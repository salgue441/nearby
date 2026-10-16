# Nearby

A vector database written in Rust, built from first principles: exact search
first, then approximate indexes (HNSW, product quantization), persistence
and a server.

**Status:** early. Exact (flat) search, the distance kernels, the benchmark
harness and the server's config and logging work. The engine, storage and
the approximate indexes are not implemented yet.

## Architecture

The workspace is split into small crates with one job each. Dependencies only
point downward, toward `nearby-core`.

```text
            nearby-bench          nearby-server
             (binary)               (binary)
              │     │               │       │
              │     ▼               ▼       │
              │  nearby-index   nearby-engine (planned)
              │     │                       │
              ▼     ▼                       ▼
                     nearby-core
```

`nearby-storage` exists as a placeholder and has no dependents yet.

| Crate | Kind | Responsibility |
|---|---|---|
| `nearby-core` | library | Shared types (`PointId`, `Metric`, `ScoredPoint`), the error type, distance kernels and the `TopK` collector. Only code that two or more crates need lives here. |
| `nearby-index` | library | The `VectorIndex` trait and its implementations. Today: `FlatIndex` (exact search). Planned: HNSW, PQ. |
| `nearby-engine` | library | *Planned.* Collections, the id map and query execution on top of indexes and storage. |
| `nearby-storage` | library | *Planned.* Segments, the write-ahead log and recovery. |
| `nearby-server` | binary | Loads config and sets up logging. The network API is not implemented yet. |
| `nearby-bench` | binary | Recall and latency benchmark on the TEXMEX datasets (SIFT, GIST). |

### Design rules

- **Every metric is a distance: lower is closer.** `L2` is squared Euclidean
  distance, `Cosine` is `1 - cosine_similarity`, and `Dot` is the negated dot
  product. Heaps, merges and indexes never need to know which metric is in use.
- **Validate at the boundary, trust inside.** `validate_vector` rejects wrong
  dimensions and NaN/infinite values when data enters an index. Hot paths such
  as the distance kernels only `debug_assert!` their inputs.
- **Results are deterministic.** Candidates at equal distance are ordered by
  ascending id, so the same query always returns the same results.
- **Libraries return `nearby_core::Error`; binaries use `anyhow`.** Callers can
  `match` on specific error variants. Only the binaries add context with
  `anyhow`.
- **Only binaries configure logging.** Library crates just emit `tracing`
  events.

### Exact search (`FlatIndex`)

`FlatIndex` compares the query with every stored vector, which costs
O(N · dim) per query. It's slow on large data but always correct, so it is the
ground truth that approximate indexes are measured against.

- Vectors are stored back to back in one contiguous `Vec<f32>`, so a scan reads
  memory sequentially.
- The best `k` candidates are kept in `TopK`, a max-heap of size `k`. Once it is
  full, rejecting a candidate costs one comparison. Over N candidates that's
  O(N log k), instead of O(N log N) for a full sort.
- With the `Cosine` metric, each vector's norm is stored at insert time, so a
  comparison costs one dot product instead of three.

The distance kernels are portable scalar code that keeps 8 independent running
sums, which lets the compiler vectorize them.

## Building and testing

The project builds on the stable toolchain (see `rust-toolchain.toml`).

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

Clippy runs with the `pedantic` group, and `missing_docs` warns across the
whole workspace (see `[workspace.lints]` in `Cargo.toml`).

Unit tests live next to the code they test, in `#[cfg(test)] mod tests`.
Tests of each library's public API live in `crates/<crate>/tests/`, where
they compile as a separate crate, exactly as a downstream user would see the
API.

## Benchmark

`nearby-bench` builds an index from a dataset's base vectors, runs every query
and reports recall against the dataset's ground truth, plus throughput and
latency.

### Getting a dataset

The bench reads the [TEXMEX](http://corpus-texmex.irisa.fr/) `.fvecs` /
`.ivecs` formats. Extract the archives into `data/`, which is git-ignored:

```sh
mkdir -p data && cd data
curl -O ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
tar -xzf siftsmall.tar.gz
```

| Dataset | Archive | Base vectors | Queries | Dimension | Base file |
|---|---|---|---|---|---|
| SIFT10K | `siftsmall.tar.gz` | 10,000 | 100 | 128 | 5 MB |
| SIFT1M | `sift.tar.gz` | 1,000,000 | 10,000 | 128 | 516 MB |
| GIST1M | `gist.tar.gz` | 1,000,000 | 1,000 | 960 | 3.8 GB |

A dataset directory `<name>/` must contain `<name>_base.fvecs`,
`<name>_query.fvecs` and `<name>_groundtruth.ivecs`. The TEXMEX archives
already extract in that layout.

### Running

```sh
cargo run -p nearby-bench --release -- [dataset dir] [k]
```

- `dataset dir` defaults to `data/siftsmall`.
- `k` is the number of neighbors per query. It defaults to `10` and must not
  exceed the number of ground-truth neighbors per query (100 for the TEXMEX
  sets).

Always use `--release`: debug builds are many times slower and the numbers
are meaningless.

```sh
cargo run -p nearby-bench --release -- data/sift 100
```

The distance kernels speed up another ~20% when compiled for your exact CPU
(AVX2, for example). The resulting binary may not run on other machines, so
this is set per run rather than committed:

```sh
RUSTFLAGS="-C target-cpu=native" cargo run -p nearby-bench --release
```

### Output

```text
dataset    siftsmall: 10000 base vectors, 100 queries, dim 128
index      flat (exact), L2
build      ... (load + insert)
recall@10  1.0000
qps        ...
latency    mean ... | p50 ... | p95 ... | p99 ... | max ...
```

- **build**: time to stream the base file into the index. For a flat index,
  inserting is just a copy, so this is mostly read time.
- **recall@k**: the average fraction of each query's true `k` nearest
  neighbors found among the first `k` results. The flat index is exact, so
  anything below 1.0000 is a bug.
- **qps**: queries per second, run one after another on a single thread.
- **latency**: per-query latency. Percentiles use the nearest-rank method.
  Tail latencies (p95, p99) are what the slowest requests experience.

Base vectors are streamed straight into the index, so peak memory is about
the size of the index, not twice the base file.

## Server

```sh
cargo run -p nearby-server -- [path/to/nearby.toml]
```

The config path defaults to `nearby.toml`. A missing file means all
defaults; unknown keys are rejected. Every key and its default:

```toml
data_dir = "./data"

[log]
level = "info"            # tracing filter directive; RUST_LOG overrides it
console = true
console_format = "full"   # "full", "compact" or "json"
file = true               # JSON lines in rolling files
file_dir = "./logs"
file_prefix = "nearby"
rotation = "daily"        # "minutely", "hourly", "daily" or "never"
max_files = 14            # 0 keeps every file
```

An invalid `RUST_LOG` stops startup with an error rather than being ignored.

## License

Dual-licensed under MIT or Apache-2.0, at your option.
