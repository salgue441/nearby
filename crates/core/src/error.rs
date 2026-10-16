//! The crate-wide error type.
//!
//! Library code returns `nearby_core::Result<T>`. Binaries can wrap
//! it with `anyhow` to add context, but the library itself
//! never uses `anyhow`, so callers can `match` on specific
//! variants.

use std::path::PathBuf;

/// `Result` with [`Error`] as the default error type.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Every error the `nearby` library crates can return.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A vector's length does not match the dimension of the index or
    /// collection it was given to.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The dimension the index or collection was created with.
        expected: usize,
        /// The length of the vector that was passed in.
        got: usize,
    },

    /// A vector contains NaN or an infinity, which would make every
    /// distance computed from it meaningless.
    #[error("vector contains NaN or infinite values")]
    InvalidVector,

    /// No point with this id exists.
    #[error("point `{0}` not found")]
    PointNotFound(u64),

    /// No collection with this name exists.
    #[error("collection `{0}` not found")]
    CollectionNotFound(String),

    /// On-disk data failed an integrity check (bad magic, checksum, length...).
    ///
    /// Build it with [`Error::corruption`].
    #[error("corrupted data in {}: {reason}", path.display())]
    Corruption {
        /// The file that failed the check.
        path: PathBuf,
        /// What was wrong with it, for humans.
        reason: String,
    },

    /// A configuration value is invalid.
    #[error("invalid config: {0}")]
    Config(String),

    /// An I/O operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl Error {
    /// Shorthand for building [`Error::Corruption`] from anything path-like
    /// and string-like.
    ///
    /// # Examples
    ///
    /// ```
    /// use nearby_core::Error;
    ///
    /// let err = Error::corruption("seg-0001.seg", "bad magic");
    /// assert_eq!(err.to_string(), "corrupted data in seg-0001.seg: bad magic");
    /// ```
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
