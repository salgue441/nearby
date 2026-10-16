# Nearby

A vector database written in Rust, built from first principles: exact search
first, then approximate indexes (HNSW, product quantization), persistence
and a server.

**Status:** early. Exact (flat) search, approximate search with HNSW, the
distance kernels, the benchmark harness and the server's config and logging
work. Product quantization, the engine and storage are not implemented yet.

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
| `nearby-index` | library | The `VectorIndex` trait and its implementations: `FlatIndex` (exact search) and `Hnsw` (approximate search). Planned: PQ. |
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

The distance kernels are picked at runtime. On x86-64 CPUs with AVX2 and FMA
(Intel since 2013, AMD since 2015), hand-written SIMD kernels compute about
twice as fast as portable code. Elsewhere, portable code keeps 8 independent
running sums so the compiler can vectorize it (with NEON on aarch64). The
kernels round differently, so distances can differ in the last bits between
CPUs; on any one machine they are deterministic.

### Approximate search (`Hnsw`)

`Hnsw` implements Hierarchical Navigable Small World graphs (Malkov &
Yashunin, 2016). Every vector is a node linked to its nearest neighbors on
layer 0; a random few also live on higher, sparser layers that act as express
lanes. A search starts at the top layer, moves greedily toward the query,
drops a layer whenever it can't get closer, and finishes with a wider search
on layer 0. That costs roughly O(log N) distance computations instead of O(N),
in exchange for recall below 1.0.

- **Layout.** Each vector gets a dense `u32` node id (its insertion position).
  Layer-0 links, where searches spend most of their time, live in one flat
  array with `m0` slots per node; the sparse upper layers use per-node lists.
- **Neighbor selection** uses the paper's heuristic (Algorithm 4): a
  candidate is dropped if a neighbor already chosen is strictly closer to it
  than the new node is. That spreads links across directions, so searches can
  escape tight clusters. Ties are kept, as in hnswlib.
- **Cosine comparisons cost one dot product.** With `Metric::Cosine`, each
  vector's norm is stored at insert time and the query's norm is computed once
  per search, instead of recomputing both norms on every comparison.
- **Searches don't allocate visited sets.** Each one borrows a reusable set
  from a pool, and a generation counter clears it in O(1). Searches take
  `&self`, so many threads can search one index at once.
- **Builds are reproducible.** Layers are drawn from a seeded `SplitMix64`, so
  the same vectors, insertion order and seed always give the same graph.

`HnswParams` holds the knobs:

| Parameter | Default | Effect |
|---|---|---|
| `m` | 16 | Links per node on upper layers. More: better recall, more memory, slower inserts. |
| `m0` | 32 | Links per node on layer 0, usually `2 * m`. At most 1024. |
| `ef_construction` | 200 | Candidates considered per insert. Higher builds a better graph, more slowly. |
| `ef_search` | 64 | Candidates considered per search: the recall/speed dial. Change it any time with `set_ef_search`; a search always uses at least its `k`. |
| `seed` | `0x5EED` | Seed for layer assignment. |

**Known limitation:** many *identical* vectors degrade reachability. They all
tie at distance 0 and link to the same few nodes, and later copies lose their
incoming links when those nodes' lists are pruned: with 100 copies among 400
random vectors, about a third of the copies can be found. This is inherent to
HNSW (hnswlib behaves the same way), so deduplicate before indexing.

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

`nearby-bench` builds a flat index, an HNSW index or both from a dataset's
base vectors, runs every query and reports recall against the dataset's
ground truth, plus throughput and latency. For HNSW it sweeps several
`ef_search` values, so one run shows the whole recall/speed trade-off.

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
cargo run -p nearby-bench --release -- [DATASET_DIR] [OPTIONS]
```

| Option | Default | Meaning |
|---|---|---|
| `DATASET_DIR` | `data/siftsmall` | Dataset directory (see above). |
| `--index flat\|hnsw\|both` | `both` | Which indexes to build and query. |
| `--k N` | `10` | Neighbors per query. At most the ground truth's width (100 for TEXMEX). |
| `--m N` | `16` | HNSW links per node (`m0` is set to `2 * m`). |
| `--ef-construction N` | `200` | HNSW candidates per insert. |
| `--ef N,N,...` | `10,20,40,80,160,320` | HNSW `ef_search` values, each at least 1; one result row each. A search always uses at least `k`, so values below `k` run as `k`; the bench says so and labels each row with the `ef` it actually used. |
| `-h`, `--help` | | Print usage. |

Always use `--release`: debug builds are many times slower and the numbers
are meaningless.

```sh
cargo run -p nearby-bench --release -- data/sift --k 100 --ef 100,200,400
cargo run -p nearby-bench --release -- data/gist --index hnsw --m 32
```

The SIMD distance kernels are chosen at runtime, so a normal build already
uses AVX2 where the CPU has it. Compiling for your exact CPU still helps a
little (5–25% on in-cache distances, ~5% when memory-bound): the runtime
check disappears and the kernels can be inlined. The resulting binary may
not run on other machines, so this is set per run rather than committed:

```sh
RUSTFLAGS="-C target-cpu=native" cargo run -p nearby-bench --release
```

### Output

```text
dataset  siftsmall: 10000 base vectors, 100 queries, dim 128

build    flat: ...
build    hnsw (m=16, ef_construction=200): ..., ... layers

index          recall@10       qps       mean        p50        p95        p99        max
flat              1.0000       ...        ...        ...        ...        ...        ...
hnsw ef=10           ...       ...        ...        ...        ...        ...        ...
hnsw ef=20           ...       ...        ...        ...        ...        ...        ...
...
```

- **build**: time to insert every base vector. For the flat index that's a
  copy; for HNSW it's the graph construction, and the line also reports the
  number of layers.
- **recall@k**: the average fraction of each query's true `k` nearest
  neighbors found among the first `k` results. The flat index is exact, so
  anything below 1.0000 there is a bug. For HNSW, recall rises with `ef`.
- **qps**: queries per second, run one after another on a single thread.
- **mean … max**: per-query latency. Percentiles use the nearest-rank method.
  Tail latencies (p95, p99) are what the slowest requests experience.

HNSW recall depends heavily on the data. Real embeddings and descriptors such
as SIFT are clustered and reach high recall at small `ef`. Uniformly random
high-dimensional vectors are a worst case: all points are nearly equidistant,
so the same settings give much lower recall.

The base file is streamed from disk, in chunks, each time an index is built,
and each index is dropped before the next one is built. So the base set
never sits in memory next to an index, and peak memory is about that of the
largest single index, whichever `--index` is chosen.

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
