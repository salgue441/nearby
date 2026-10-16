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
use std::path::Path;

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

/// Reads a `.fvecs` file (base vectors or queries).
///
/// # Errors
///
/// If the file cannot be read, is empty, declares a dimension of 0, is
/// truncated, or mixes records of different dimensions.
pub fn read_fvecs(path: &Path) -> anyhow::Result<Vectors> {
    let bytes = read_file(path)?;
    let (dim, data) = parse_vecs(&bytes, f32::from_le_bytes)
        .with_context(|| format!("parsing {}", path.display()))?;

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
    let bytes = read_file(path)?;
    let (dim, flat) = parse_vecs(&bytes, u32::from_le_bytes)
        .with_context(|| format!("parsing {}", path.display()))?;

    Ok(flat.chunks_exact(dim).map(<[u32]>::to_vec).collect())
}

fn read_file(path: &Path) -> anyhow::Result<Vec<u8>> {
    std::fs::read(path).with_context(|| {
        format!(
            "reading {} (is the dataset extracted into data/?)",
            path.display()
        )
    })
}

/// Parses the shared record layout. `convert` turns each value's 4 bytes into `T`.
/// Returns the dimension and all values back to back.
fn parse_vecs<T>(bytes: &[u8], convert: fn([u8; 4]) -> T) -> anyhow::Result<(usize, Vec<T>)> {
    let (header, _) = bytes
        .split_first_chunk::<4>()
        .context("file is too short to contain a header")?;

    let dim = u32::from_le_bytes(*header) as usize;
    ensure!(dim > 0, "dimension is 0");

    let record_len = 4 * (1 + dim);
    ensure!(
        bytes.len().is_multiple_of(record_len),
        "file size {} is not a multiple of the record size {record_len} (truncated file?)",
        bytes.len()
    );

    let mut values = Vec::with_capacity(bytes.len() / record_len * dim);
    for (i, record) in bytes.chunks_exact(record_len).enumerate() {
        let (words, _) = record.as_chunks::<4>();
        let record_dim = u32::from_le_bytes(words[0]) as usize;

        ensure!(
            record_dim == dim,
            "record {i} has dimension {record_dim}, expected {dim}"
        );

        values.extend(words[1..].iter().copied().map(convert));
    }

    Ok((dim, values))
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

    #[test]
    fn parses_records() {
        let bytes = encode(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let (dim, data) = parse_vecs(&bytes, f32::from_le_bytes).unwrap();
        assert_eq!(dim, 3);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn rejects_truncated_file() {
        let mut bytes = encode(&[&[1.0, 2.0], &[3.0, 4.0]]);
        bytes.pop();
        let err = parse_vecs(&bytes, f32::from_le_bytes).unwrap_err();
        assert!(err.to_string().contains("not a multiple"), "got: {err}");
    }

    #[test]
    fn rejects_mixed_dimensions() {
        // Same total size as two 2-dim records, but the second header says 1.
        let mut bytes = encode(&[&[1.0, 2.0]]);
        bytes.extend(encode(&[&[3.0]]));
        bytes.extend(0f32.to_le_bytes());
        let err = parse_vecs(&bytes, f32::from_le_bytes).unwrap_err();
        assert!(err.to_string().contains("record 1"), "got: {err}");
    }

    #[test]
    fn rejects_empty_file() {
        assert!(parse_vecs(&[], f32::from_le_bytes).is_err());
    }

    #[test]
    fn rejects_zero_dimension() {
        let err = parse_vecs(&0u32.to_le_bytes(), f32::from_le_bytes).unwrap_err();
        assert!(err.to_string().contains("dimension is 0"), "got: {err}");
    }

    #[test]
    fn rejects_negative_dimension() {
        // A negative `i32` header reads as a huge `u32`; no file is that big.
        let mut bytes = (-1i32).to_le_bytes().to_vec();
        bytes.extend(1f32.to_le_bytes());
        assert!(parse_vecs(&bytes, f32::from_le_bytes).is_err());
    }

    #[test]
    fn rejects_header_without_values() {
        let bytes = 3u32.to_le_bytes();
        assert!(parse_vecs(&bytes, f32::from_le_bytes).is_err());
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
}
