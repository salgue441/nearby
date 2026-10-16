//! Loaders for the TEXMEX `.fvecs` / `.ivecs` formats used by SIFT and GIST.
//!
//! A file is a list of records. Each record is a little-endian `i32` dimension
//! followed by that many 4-byte values: `f32` in `.fvecs`, `i32` in `.ivecs`.
//!
//! ```text
//! ┌──────────┬────────────┬────────────┬─────┐
//! │ dim: i32 │ value 0    │ value 1    │ ... │   × number of vectors
//! └──────────┴────────────┴────────────┴─────┘
//! ```

use anyhow::{Context, ensure};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

/// Vectors stored back to back, all with the same dimension.
#[derive(Debug)]
pub struct Vectors {
    /// Number of values in each vector. Always at least 1.
    pub dim: usize,
    /// All vectors concatenated; `data.len()` is a multiple of `dim`.
    pub data: Vec<f32>,
}

impl Vectors {
    /// Number of vectors.
    pub fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    /// Iterates over the vectors, each as a `dim`-long slice.
    pub fn iter(&self) -> std::slice::ChunksExact<'_, f32> {
        self.data.chunks_exact(self.dim)
    }
}

/// Reads a `.fvecs` file (base vectors or queries) into memory.
///
/// For files too big to hold twice (once here, once in an index), stream
/// them with [`VecsReader::open_fvecs`] instead.
///
/// # Errors
///
/// If the file cannot be read, is empty, declares a dimension of 0, is
/// truncated, or mixes records of different dimensions.
pub fn read_fvecs(path: &Path) -> anyhow::Result<Vectors> {
    let mut reader = VecsReader::open_fvecs(path)?;
    let dim = reader.dim();
    let data = read_all(&mut reader).with_context(|| format!("parsing {}", path.display()))?;
    Ok(Vectors { dim, data })
}

/// Reads an `.ivecs` file (ground truth: the ids of each query's true neighbors).
///
/// Values are read as `u32`, since ids are never negative.
///
/// # Errors
///
/// Same as [`read_fvecs`].
pub fn read_ivecs(path: &Path) -> anyhow::Result<Vec<Vec<u32>>> {
    let (file, len) = open(path)?;
    let mut reader = VecsReader::new(
        BufReader::with_capacity(READ_BUFFER, file),
        len,
        u32::from_le_bytes,
    )
    .with_context(|| format!("parsing {}", path.display()))?;

    let mut rows = Vec::with_capacity(reader.len());
    while let Some(row) = reader
        .next_vector()
        .with_context(|| format!("parsing {}", path.display()))?
    {
        rows.push(row.to_vec());
    }
    Ok(rows)
}

/// Read buffer size. Large enough that per-record reads are cheap copies.
const READ_BUFFER: usize = 1 << 20;

/// Opens `path` and returns it with its size in bytes.
fn open(path: &Path) -> anyhow::Result<(File, u64)> {
    let open = || -> std::io::Result<(File, u64)> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        Ok((file, len))
    };
    open().with_context(|| {
        format!(
            "reading {} (is the dataset extracted into data/?)",
            path.display()
        )
    })
}

/// Drains `reader` into one flat `Vec`, sized exactly up front.
fn read_all<R: Read, T: Copy>(reader: &mut VecsReader<R, T>) -> anyhow::Result<Vec<T>> {
    let total = reader
        .len()
        .checked_mul(reader.dim())
        .context("file too large")?;
    let mut values = Vec::with_capacity(total);
    while let Some(vector) = reader.next_vector()? {
        values.extend_from_slice(vector);
    }
    Ok(values)
}

/// Streams the records of a `.fvecs` / `.ivecs` file one at a time.
///
/// The header and file size are checked up front, so [`VecsReader::dim`] and
/// [`VecsReader::len`] are known before any vector is read. Memory use is one
/// record, however large the file, which lets the bench feed base vectors
/// straight into an index without holding a second copy of the dataset.
///
/// Not an [`Iterator`]: each vector borrows the reader's internal buffer,
/// which the next call overwrites.
pub struct VecsReader<R, T> {
    reader: R,
    convert: fn([u8; 4]) -> T,
    dim: usize,
    count: usize,
    /// Index of the next record to read.
    next: usize,
    /// Raw bytes of the current record, header included.
    record: Vec<u8>,
    /// The current record's values, converted.
    vector: Vec<T>,
}

impl<R, T> std::fmt::Debug for VecsReader<R, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VecsReader")
            .field("dim", &self.dim)
            .field("count", &self.count)
            .field("next", &self.next)
            .finish_non_exhaustive()
    }
}

impl VecsReader<BufReader<File>, f32> {
    /// Opens a `.fvecs` file for streaming.
    ///
    /// # Errors
    ///
    /// If the file cannot be opened, or its header or size is invalid (see
    /// [`VecsReader::new`]).
    pub fn open_fvecs(path: &Path) -> anyhow::Result<Self> {
        let (file, len) = open(path)?;
        Self::new(
            BufReader::with_capacity(READ_BUFFER, file),
            len,
            f32::from_le_bytes,
        )
        .with_context(|| format!("parsing {}", path.display()))
    }
}

impl<R: Read, T> VecsReader<R, T> {
    /// Reads the first header from `reader`, which holds `len` bytes in all.
    /// `convert` turns each value's 4 bytes into `T`.
    ///
    /// # Errors
    ///
    /// If `len` is too short for a header, the dimension is 0, or `len` is
    /// not a whole number of records.
    pub fn new(mut reader: R, len: u64, convert: fn([u8; 4]) -> T) -> anyhow::Result<Self> {
        ensure!(len >= 4, "file is too short to contain a header");
        let mut header = [0u8; 4];
        reader.read_exact(&mut header).context("reading header")?;

        let dim = u32::from_le_bytes(header) as usize;
        ensure!(dim > 0, "dimension is 0");

        let record_len = 4 * (1 + dim as u64);
        ensure!(
            len.is_multiple_of(record_len),
            "file size {len} is not a multiple of the record size {record_len} (truncated file?)"
        );

        // The first record's header has been consumed; keep it in place so
        // the first record is checked like every other one.
        let mut record = vec![0u8; usize::try_from(record_len)?];
        record[..4].copy_from_slice(&header);

        Ok(Self {
            reader,
            convert,
            dim,
            count: usize::try_from(len / record_len)?,
            next: 0,
            record,
            vector: Vec::with_capacity(dim),
        })
    }

    /// Number of values in each vector. Always at least 1.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Total number of vectors in the file, including ones already read.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Reads the next vector, or returns `None` after the last one.
    ///
    /// # Errors
    ///
    /// If the reader fails or ends early, or the record declares a
    /// different dimension than the first one.
    pub fn next_vector(&mut self) -> anyhow::Result<Option<&[T]>> {
        if self.next == self.count {
            return Ok(None);
        }

        let i = self.next;
        let unread = if i == 0 {
            &mut self.record[4..]
        } else {
            &mut self.record[..]
        };
        self.reader
            .read_exact(unread)
            .with_context(|| format!("reading record {i}"))?;

        let (words, _) = self.record.as_chunks::<4>();
        let record_dim = u32::from_le_bytes(words[0]) as usize;
        ensure!(
            record_dim == self.dim,
            "record {i} has dimension {record_dim}, expected {}",
            self.dim
        );

        self.vector.clear();
        self.vector
            .extend(words[1..].iter().copied().map(self.convert));
        self.next += 1;

        Ok(Some(&self.vector))
    }
}

/// A benchmark dataset: base vectors, queries, and each query's true neighbors.
///
/// Queries and ground truth are small and loaded whole. The base set is the
/// big one, so it stays on disk and is streamed with
/// [`Dataset::for_each_base_chunk`] each time an index is built. It never
/// sits in memory next to an index, and an index can be dropped before the
/// next one is built, so peak memory is that of the largest single index.
#[derive(Debug)]
pub struct Dataset {
    /// The directory's name, which prefixes every file in it (`siftsmall`).
    pub name: String,
    /// The vectors to search for. Same dimension as the base vectors.
    pub queries: Vectors,
    /// `truth[q]` = ids of query `q`'s nearest base vectors, closest first.
    pub truth: Vec<Vec<u32>>,
    base_path: PathBuf,
    dim: usize,
    base_len: usize,
}

impl Dataset {
    /// Loads `<dir>/<name>_query.fvecs` and `<name>_groundtruth.ivecs`, reads
    /// the header of `<name>_base.fvecs`, where `<name>` is the directory's
    /// own name, and checks that the three files agree with each other.
    ///
    /// # Errors
    ///
    /// If `dir` has no usable name, a file is missing or malformed, the query
    /// and base dimensions differ, there isn't one ground-truth row per
    /// query, or a row has fewer than `k` neighbors.
    pub fn load(dir: &Path, k: usize) -> anyhow::Result<Self> {
        let name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .context("dataset path has no directory name")?
            .to_owned();
        let base_path = dir.join(format!("{name}_base.fvecs"));
        let base = VecsReader::open_fvecs(&base_path)?;
        let queries = read_fvecs(&dir.join(format!("{name}_query.fvecs")))?;
        let truth = read_ivecs(&dir.join(format!("{name}_groundtruth.ivecs")))?;

        ensure!(
            queries.dim == base.dim(),
            "queries have dimension {}, base vectors have {}",
            queries.dim,
            base.dim()
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
        Ok(Self {
            name,
            queries,
            truth,
            base_path,
            dim: base.dim(),
            base_len: base.len(),
        })
    }

    /// Number of values in each vector.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of base vectors.
    pub fn base_len(&self) -> usize {
        self.base_len
    }

    /// Reads the base vectors from disk, front to back, in chunks of up to
    /// `max` vectors stored back to back, and calls `f(first, chunk)` for
    /// each. `first` is the position of the chunk's first vector in the file.
    ///
    /// Each call reads the file again, so it can be called once per index.
    ///
    /// # Panics
    ///
    /// If `max` is 0.
    ///
    /// # Errors
    ///
    /// If the base file can't be read or parsed, no longer matches the shape
    /// seen by [`Dataset::load`], or `f` fails. File errors name the file.
    pub fn for_each_base_chunk(
        &self,
        max: usize,
        mut f: impl FnMut(usize, &[f32]) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        assert!(max > 0, "chunks must hold at least one vector");
        let context = || format!("parsing {}", self.base_path.display());
        let mut reader = VecsReader::open_fvecs(&self.base_path)?;
        ensure!(
            (reader.dim(), reader.len()) == (self.dim, self.base_len),
            "{} changed since it was loaded",
            self.base_path.display()
        );

        let mut chunk = Vec::with_capacity(max * self.dim);
        let mut first = 0;
        loop {
            chunk.clear();
            while chunk.len() < max * self.dim {
                match reader.next_vector().with_context(context)? {
                    Some(vector) => chunk.extend_from_slice(vector),
                    None => break,
                }
            }
            if chunk.is_empty() {
                return Ok(());
            }
            f(first, &chunk)?;
            first += chunk.len() / self.dim;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encodes records the way `.fvecs` files store them.
    fn encode(records: &[&[f32]]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for record in records {
            bytes.extend((record.len() as u32).to_le_bytes());
            for value in *record {
                bytes.extend(value.to_le_bytes());
            }
        }

        bytes
    }

    /// Reads every vector from `reader`, which claims to hold `len` bytes.
    fn parse_from(reader: impl Read, len: u64) -> anyhow::Result<(usize, Vec<f32>)> {
        let mut reader = VecsReader::new(reader, len, f32::from_le_bytes)?;
        let dim = reader.dim();
        Ok((dim, read_all(&mut reader)?))
    }

    /// Parses an in-memory `.fvecs` file.
    fn parse(bytes: &[u8]) -> anyhow::Result<(usize, Vec<f32>)> {
        parse_from(bytes, bytes.len() as u64)
    }

    /// A reader that returns at most one byte per `read` call, the worst
    /// case `read_exact` has to handle.
    struct OneByteAtATime<'a>(&'a [u8]);

    impl Read for OneByteAtATime<'_> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let Some((&first, rest)) = self.0.split_first() else {
                return Ok(0);
            };
            if buf.is_empty() {
                return Ok(0);
            }
            buf[0] = first;
            self.0 = rest;
            Ok(1)
        }
    }

    #[test]
    fn short_reads_are_handled() {
        let bytes = encode(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let (dim, data) = parse_from(OneByteAtATime(&bytes), bytes.len() as u64).unwrap();
        assert_eq!(dim, 3);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn reader_ending_early_is_an_error() {
        // The file shrank between reading its size and reading its contents.
        let bytes = encode(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let err = parse_from(&bytes[..12], bytes.len() as u64).unwrap_err();
        assert!(format!("{err:#}").contains("record 1"), "got: {err:#}");
    }

    #[test]
    fn reader_reports_shape_before_reading() {
        let bytes = encode(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
        let reader = VecsReader::new(&bytes[..], bytes.len() as u64, f32::from_le_bytes).unwrap();
        assert_eq!(reader.dim(), 2);
        assert_eq!(reader.len(), 3);
    }

    #[test]
    fn reader_yields_each_vector_then_none() {
        let bytes = encode(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let mut reader =
            VecsReader::new(&bytes[..], bytes.len() as u64, f32::from_le_bytes).unwrap();
        assert_eq!(reader.next_vector().unwrap(), Some(&[1.0, 2.0][..]));
        assert_eq!(reader.next_vector().unwrap(), Some(&[3.0, 4.0][..]));
        assert_eq!(reader.next_vector().unwrap(), None);
        assert_eq!(reader.next_vector().unwrap(), None, "stays exhausted");
    }

    #[test]
    fn reader_validates_lazily_but_keeps_earlier_vectors_valid() {
        // Record 1 has a bad header: record 0 must still come back intact.
        let mut bytes = encode(&[&[1.0, 2.0]]);
        bytes.extend(encode(&[&[3.0]]));
        bytes.extend(0f32.to_le_bytes());
        let mut reader =
            VecsReader::new(&bytes[..], bytes.len() as u64, f32::from_le_bytes).unwrap();
        assert_eq!(reader.next_vector().unwrap(), Some(&[1.0, 2.0][..]));
        let err = reader.next_vector().unwrap_err();
        assert!(err.to_string().contains("record 1"), "got: {err}");
    }

    #[test]
    fn open_fvecs_streams_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("base.fvecs");
        std::fs::write(&path, encode(&[&[1.0], &[2.0], &[3.0]])).unwrap();

        let mut reader = VecsReader::open_fvecs(&path).unwrap();
        assert_eq!((reader.dim(), reader.len()), (1, 3));
        let mut seen = Vec::new();
        while let Some(v) = reader.next_vector().unwrap() {
            seen.extend_from_slice(v);
        }
        assert_eq!(seen, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn open_fvecs_errors_name_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.fvecs");
        std::fs::write(&path, []).unwrap();
        let err = format!("{:#}", VecsReader::open_fvecs(&path).unwrap_err());
        assert!(err.contains("empty.fvecs"), "got: {err}");
    }

    #[test]
    fn single_record() {
        assert_eq!(parse(&encode(&[&[7.0]])).unwrap(), (1, vec![7.0]));
    }

    #[test]
    fn reads_files_larger_than_the_read_buffer() {
        const DIM: usize = 960; // GIST
        let records: Vec<Vec<f32>> = (0..300)
            .map(|r| (0..DIM).map(|i| (r * DIM + i) as f32).collect())
            .collect();
        let refs: Vec<&[f32]> = records.iter().map(Vec::as_slice).collect();
        let bytes = encode(&refs);
        assert!(
            bytes.len() > READ_BUFFER,
            "test must cross a buffer boundary"
        );

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gist.fvecs");
        std::fs::write(&path, bytes).unwrap();

        let v = read_fvecs(&path).unwrap();
        assert_eq!(v.dim, DIM);
        assert_eq!(v.len(), 300);
        assert_eq!(v.data, records.concat());
    }

    #[test]
    fn parses_records() {
        let bytes = encode(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let (dim, data) = parse(&bytes).unwrap();
        assert_eq!(dim, 3);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn rejects_truncated_file() {
        let mut bytes = encode(&[&[1.0, 2.0], &[3.0, 4.0]]);
        bytes.pop();
        let err = parse(&bytes).unwrap_err();
        assert!(err.to_string().contains("not a multiple"), "got: {err}");
    }

    #[test]
    fn rejects_mixed_dimensions() {
        // Same total size as two 2-dim records, but the second header says 1.
        let mut bytes = encode(&[&[1.0, 2.0]]);
        bytes.extend(encode(&[&[3.0]]));
        bytes.extend(0f32.to_le_bytes());
        let err = parse(&bytes).unwrap_err();
        assert!(err.to_string().contains("record 1"), "got: {err}");
    }

    #[test]
    fn rejects_empty_file() {
        assert!(parse(&[]).is_err());
    }

    #[test]
    fn rejects_zero_dimension() {
        let err = parse(&0u32.to_le_bytes()).unwrap_err();
        assert!(err.to_string().contains("dimension is 0"), "got: {err}");
    }

    #[test]
    fn rejects_negative_dimension() {
        // A negative `i32` header reads as a huge `u32`; no file is that big.
        let mut bytes = (-1i32).to_le_bytes().to_vec();
        bytes.extend(1f32.to_le_bytes());
        assert!(parse(&bytes).is_err());
    }

    #[test]
    fn rejects_header_without_values() {
        let bytes = 3u32.to_le_bytes();
        assert!(parse(&bytes).is_err());
    }

    #[test]
    fn vectors_len_and_iter() {
        let v = Vectors {
            dim: 2,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        assert_eq!(v.len(), 3);
        let rows: Vec<&[f32]> = v.iter().collect();
        assert_eq!(rows, [&[1.0, 2.0][..], &[3.0, 4.0], &[5.0, 6.0]]);
    }

    #[test]
    fn reads_fvecs_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("base.fvecs");
        std::fs::write(&path, encode(&[&[1.5, -2.0], &[0.0, 8.25]])).unwrap();

        let v = read_fvecs(&path).unwrap();
        assert_eq!(v.dim, 2);
        assert_eq!(v.data, vec![1.5, -2.0, 0.0, 8.25]);
    }

    #[test]
    fn reads_ivecs_file_into_rows() {
        let mut bytes = Vec::new();
        for row in [[3u32, 1, 4], [1, 5, 9]] {
            bytes.extend(3u32.to_le_bytes());
            for id in row {
                bytes.extend(id.to_le_bytes());
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("groundtruth.ivecs");
        std::fs::write(&path, bytes).unwrap();

        assert_eq!(
            read_ivecs(&path).unwrap(),
            vec![vec![3, 1, 4], vec![1, 5, 9]]
        );
    }

    #[test]
    fn errors_name_the_file() {
        let dir = tempfile::tempdir().unwrap();

        let missing = dir.path().join("missing.fvecs");
        let err = format!("{:#}", read_fvecs(&missing).unwrap_err());
        assert!(err.contains("missing.fvecs"), "got: {err}");

        let corrupt = dir.path().join("corrupt.ivecs");
        std::fs::write(&corrupt, [1, 0]).unwrap();
        let err = format!("{:#}", read_ivecs(&corrupt).unwrap_err());
        assert!(err.contains("corrupt.ivecs"), "got: {err}");
    }

    /// Writes a TEXMEX-layout dataset named `name` under `root` and returns
    /// its directory.
    fn write_dataset(
        root: &Path,
        name: &str,
        base: &[&[f32]],
        queries: &[&[f32]],
        truth: &[&[u32]],
    ) -> std::path::PathBuf {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("{name}_base.fvecs")), encode(base)).unwrap();
        std::fs::write(dir.join(format!("{name}_query.fvecs")), encode(queries)).unwrap();
        let mut ivecs = Vec::new();
        for row in truth {
            ivecs.extend((row.len() as u32).to_le_bytes());
            for id in *row {
                ivecs.extend(id.to_le_bytes());
            }
        }
        std::fs::write(dir.join(format!("{name}_groundtruth.ivecs")), ivecs).unwrap();
        dir
    }

    #[test]
    fn dataset_loads_all_three_files() {
        let root = tempfile::tempdir().unwrap();
        let dir = write_dataset(
            root.path(),
            "tiny",
            &[&[0.0, 0.0], &[1.0, 1.0], &[2.0, 2.0]],
            &[&[0.1, 0.1]],
            &[&[0, 1, 2]],
        );

        let data = Dataset::load(&dir, 3).unwrap();
        assert_eq!(data.name, "tiny");
        assert_eq!((data.base_len(), data.dim()), (3, 2));
        assert_eq!(data.queries.len(), 1);
        assert_eq!(data.truth, vec![vec![0, 1, 2]]);
    }

    #[test]
    fn base_vectors_stream_in_chunks() {
        let root = tempfile::tempdir().unwrap();
        let base: Vec<[f32; 1]> = (0..5u8).map(|i| [f32::from(i)]).collect();
        let refs: Vec<&[f32]> = base.iter().map(|v| &v[..]).collect();
        let dir = write_dataset(root.path(), "chunks", &refs, &[&[0.0]], &[&[0]]);
        let data = Dataset::load(&dir, 1).unwrap();

        let mut seen = Vec::new();
        data.for_each_base_chunk(2, |first, chunk| {
            seen.push((first, chunk.to_vec()));
            Ok(())
        })
        .unwrap();
        assert_eq!(
            seen,
            vec![(0, vec![0.0, 1.0]), (2, vec![2.0, 3.0]), (4, vec![4.0])]
        );
    }

    #[test]
    fn base_can_be_streamed_again() {
        let root = tempfile::tempdir().unwrap();
        let dir = write_dataset(root.path(), "twice", &[&[1.0], &[2.0]], &[&[0.0]], &[&[0]]);
        let data = Dataset::load(&dir, 1).unwrap();
        for _ in 0..2 {
            let mut total = 0;
            data.for_each_base_chunk(10, |_, chunk| {
                total += chunk.len();
                Ok(())
            })
            .unwrap();
            assert_eq!(total, 2);
        }
    }

    #[test]
    fn base_chunk_errors_stop_the_stream() {
        let root = tempfile::tempdir().unwrap();
        let dir = write_dataset(root.path(), "stop", &[&[1.0], &[2.0]], &[&[0.0]], &[&[0]]);
        let data = Dataset::load(&dir, 1).unwrap();
        let mut calls = 0;
        let err = data
            .for_each_base_chunk(1, |_, _| {
                calls += 1;
                anyhow::bail!("index full")
            })
            .unwrap_err();
        assert_eq!(calls, 1);
        assert_eq!(err.to_string(), "index full");
    }

    #[test]
    fn base_read_errors_name_the_file() {
        let root = tempfile::tempdir().unwrap();
        let dir = write_dataset(root.path(), "bad", &[&[1.0], &[2.0]], &[&[0.0]], &[&[0]]);
        let data = Dataset::load(&dir, 1).unwrap();
        // Corrupt the second record's dimension header (same file size).
        let path = dir.join("bad_base.fvecs");
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[8..12].copy_from_slice(&7u32.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();

        let err = data.for_each_base_chunk(10, |_, _| Ok(())).unwrap_err();
        let err = format!("{err:#}");
        assert!(
            err.contains("bad_base.fvecs") && err.contains("record 1"),
            "got: {err}"
        );
    }

    #[test]
    fn base_changed_since_load_is_an_error() {
        let root = tempfile::tempdir().unwrap();
        let dir = write_dataset(root.path(), "grew", &[&[1.0]], &[&[0.0]], &[&[0]]);
        let data = Dataset::load(&dir, 1).unwrap();
        write_dataset(root.path(), "grew", &[&[1.0], &[2.0]], &[&[0.0]], &[&[0]]);

        let err = data.for_each_base_chunk(10, |_, _| Ok(())).unwrap_err();
        assert!(
            err.to_string().contains("changed since it was loaded"),
            "got: {err}"
        );
    }

    #[test]
    fn dataset_rejects_inconsistent_files() {
        let root = tempfile::tempdir().unwrap();
        let base: &[&[f32]] = &[&[0.0, 0.0], &[1.0, 1.0]];

        let dir = write_dataset(root.path(), "dims", base, &[&[0.0]], &[&[0, 1]]);
        let err = format!("{:#}", Dataset::load(&dir, 1).unwrap_err());
        assert!(err.contains("queries have dimension 1"), "got: {err}");

        let dir = write_dataset(root.path(), "rows", base, &[&[0.0, 0.0]], &[&[0], &[1]]);
        let err = format!("{:#}", Dataset::load(&dir, 1).unwrap_err());
        assert!(
            err.contains("2 ground-truth rows for 1 queries"),
            "got: {err}"
        );

        let dir = write_dataset(root.path(), "short", base, &[&[0.0, 0.0]], &[&[0, 1]]);
        let err = format!("{:#}", Dataset::load(&dir, 5).unwrap_err());
        assert!(err.contains("fewer than k = 5"), "got: {err}");
    }

    #[test]
    fn dataset_errors_name_the_missing_file() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("absent");
        std::fs::create_dir_all(&dir).unwrap();
        let err = format!("{:#}", Dataset::load(&dir, 1).unwrap_err());
        assert!(err.contains("absent_base.fvecs"), "got: {err}");
    }
}
