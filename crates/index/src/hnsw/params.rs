//! HNSW build and search parameters.

use nearby_core::{Error, Result};

/// Layer-0 neighbor counts are stored as `u16`, and nobody needs more than this.
const MAX_M0: usize = 1024;

/// Tuning knobs for [`super::Hnsw`].
///
/// The defaults are a solid starting point for 100–1000 dimensional data.
/// See section 4 of the HNSW paper for how each one affects recall and speed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HnswParams {
    /// Max links per node on layers 1 and up. More links: better recall,
    /// more memory, slower inserts. Typical: 8–48.
    pub m: usize,
    /// Max links per node on layer 0, which is denser. Usually `2 * m`.
    pub m0: usize,
    /// Candidate list size while inserting. Higher builds a better graph, slower.
    pub ef_construction: usize,
    /// Candidate list size while searching: the recall vs. speed dial.
    /// Searches always use at least `k`. Can be changed after building.
    pub ef_search: usize,
    /// Seed for the random layer assignment, so builds are reproducible.
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: 64,
            seed: 0x5EED,
        }
    }
}

impl HnswParams {
    /// Default parameters with a different `m` (and `m0 = 2 * m`).
    pub fn with_m(m: usize) -> Self {
        Self {
            m,
            m0: 2 * m,
            ..Self::default()
        }
    }

    /// Checks that the parameters can build a working graph.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if `m < 2`, `m0` is outside `m..=1024`,
    /// or `ef_construction` or `ef_search` is 0.
    pub(super) fn validate(&self) -> Result<()> {
        if self.m < 2 {
            return invalid(format!("m must be at least 2, got {}", self.m));
        }
        if self.m0 < self.m || self.m0 > MAX_M0 {
            return invalid(format!(
                "m0 must be between m ({}) and {MAX_M0}, got {}",
                self.m, self.m0
            ));
        }
        if self.ef_construction == 0 || self.ef_search == 0 {
            return invalid("ef_construction and ef_search must be at least 1".to_owned());
        }
        Ok(())
    }

    /// Max links per node on `level`.
    pub(super) fn max_links(&self, level: usize) -> usize {
        if level == 0 { self.m0 } else { self.m }
    }

    /// `1 / ln(m)`: how quickly the number of nodes shrinks per layer.
    /// With m = 16, about 1 in 16 nodes reaches layer 1, 1 in 256 layer 2, ...
    pub(super) fn level_multiplier(&self) -> f64 {
        1.0 / (self.m as f64).ln()
    }
}

fn invalid(message: String) -> Result<()> {
    Err(Error::InvalidParameter(message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        assert!(HnswParams::default().validate().is_ok());
        assert!(HnswParams::with_m(8).validate().is_ok());
    }

    #[test]
    fn rejects_bad_values() {
        assert!(HnswParams::with_m(1).validate().is_err());
        let p = HnswParams {
            m0: 4,
            ..HnswParams::default()
        };
        assert!(p.validate().is_err(), "m0 < m");
        let p = HnswParams {
            ef_search: 0,
            ..HnswParams::default()
        };
        assert!(p.validate().is_err());
    }
}
