//! The crate-wide error type.
//!
//! Library code returns `nearby::Result<T>`. Binaries can wrap
//! it with `anyhow` to add context, but the library itself
//! never uses `anyhow`, so callers can `match` on specific
//! variants.

use std::path::PathBuf;

/// `Result` with [`Error`] as the default error type.
pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("vector contains NaN or infinite values")]
    InvalidVector,

    #[error("point `{0}` not found")]
    PointNotFound(u64),

    #[error("collection `{0}` not found")]
    CollectionNotFound(String),

    #[error("corrupted data in {}: {reason}", path.display())]
    Corruption { path: PathBuf, reason: String },

    #[error("invalid config: {0}")]
    Config(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl Error {
    pub fn corruption(path: impl Into<PathBuf>, reason: impl Into<String>) -> Self {
        Self::Corruption {
            path: path.into(),
            reason: reason.into(),
        }
    }
}

/// Checks that `vector` has `dim` elements and that every element is finite.
///
/// Call this at the API boundary; internal code can then assume valid vectors.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] if the length is wrong, or
/// [`Error::InvalidVector`] if any element is NaN or infinite.
pub fn validate_vector(vector: &[f32], dim: usize) -> Result<()> {
    if vector.len() != dim {
        return Err(Error::DimensionMismatch {
            expected: dim,
            got: vector.len(),
        });
    }

    if !vector.iter().all(|x| x.is_finite()) {
        return Err(Error::InvalidVector);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_valid_vector() {
        assert!(validate_vector(&[1.0, 2.0, 3.0], 3).is_ok());
    }

    #[test]
    fn rejects_wrong_dimension() {
        let err = validate_vector(&[1.0, 2.0], 3).unwrap_err();
        assert!(matches!(
            err,
            Error::DimensionMismatch {
                expected: 3,
                got: 2
            }
        ));
    }

    #[test]
    fn rejects_nan_and_infinity() {
        assert!(matches!(
            validate_vector(&[f32::NAN], 1),
            Err(Error::InvalidVector)
        ));
        assert!(matches!(
            validate_vector(&[f32::INFINITY], 1),
            Err(Error::InvalidVector)
        ));
    }

    #[test]
    fn error_messages_are_readable() {
        let err = Error::corruption("seg-0001.seg", "bad magic");
        assert_eq!(err.to_string(), "corrupted data in seg-0001.seg: bad magic");
    }
}
