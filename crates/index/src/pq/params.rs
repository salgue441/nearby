//! Product quantization training parameters.

use nearby_core::{Error, Result};

/// Tuning knobs for training a [`super::ProductQuantizer`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PqParams {
    /// Slices each vector is split into, which is also the code length in
    /// bytes. Must divide the dimension. More subspaces: better recall, more
    /// memory, slower searches. Typical: `dim / 8` to `dim / 2`.
    pub subspaces: usize,
    /// Maximum k-means iterations per subspace. Training stops earlier once
    /// no training vector changes cluster.
    pub iterations: usize,
    /// Seed for picking the initial centroids, so training is reproducible.
    pub seed: u64,
}

impl Default for PqParams {
    fn default() -> Self {
        Self {
            subspaces: 16,
            iterations: 25,
            seed: 0x5EED,
        }
    }
}

impl PqParams {
    /// Default parameters with a different number of subspaces.
    pub fn with_subspaces(subspaces: usize) -> Self {
        Self {
            subspaces,
            ..Self::default()
        }
    }

    /// Checks that the parameters can quantize `dim`-dimensional vectors.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if `dim` or `subspaces` is 0, `subspaces`
    /// does not divide `dim`, or `iterations` is 0.
    pub fn validate(&self, dim: usize) -> Result<()> {
        if dim == 0 {
            return invalid("dimension must be at least 1".to_owned());
        }
        if self.subspaces == 0 || !dim.is_multiple_of(self.subspaces) {
            return invalid(format!(
                "subspaces must divide the dimension ({dim}), got {}",
                self.subspaces
            ));
        }
        if self.iterations == 0 {
            return invalid("iterations must be at least 1".to_owned());
        }
        Ok(())
    }
}

fn invalid(message: String) -> Result<()> {
    Err(Error::InvalidParameter(message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid_for_common_dimensions() {
        for dim in [16, 128, 960] {
            assert!(PqParams::default().validate(dim).is_ok(), "dim {dim}");
        }
    }

    #[test]
    fn rejects_unusable_parameters() {
        let cases = [
            (PqParams::default(), 0, "dimension"),
            (PqParams::with_subspaces(0), 128, "subspaces"),
            (PqParams::with_subspaces(5), 128, "subspaces must divide"),
            (
                PqParams {
                    iterations: 0,
                    ..PqParams::default()
                },
                128,
                "iterations",
            ),
        ];
        for (params, dim, expected) in cases {
            let err = params.validate(dim).unwrap_err().to_string();
            assert!(err.contains(expected), "{params:?}, dim {dim}: {err}");
        }
    }

    #[test]
    fn one_subspace_per_dimension_is_allowed() {
        assert!(PqParams::with_subspaces(8).validate(8).is_ok());
    }
}
