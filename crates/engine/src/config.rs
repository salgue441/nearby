//! How a collection is set up: dimension, metric and index.

use std::path::Path;

use nearby_core::{Error, Metric, Result};
use nearby_index::HnswParams;
use serde::{Deserialize, Serialize};

/// A collection's settings, fixed when it is created.
///
/// Serialized as TOML in the collection's directory, and as JSON by the
/// server, e.g. `{"dim": 128, "metric": "cosine", "index": {"type": "hnsw"}}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollectionConfig {
    /// Number of values in every vector.
    pub dim: usize,
    /// How distances are measured. Defaults to L2.
    #[serde(default, with = "MetricDef")]
    pub metric: Metric,
    /// Which index answers searches. Defaults to HNSW with default parameters.
    #[serde(default)]
    pub index: IndexConfig,
}

/// Which index a collection searches with.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum IndexConfig {
    /// Exact search: compares the query with every point.
    Flat,
    /// Approximate search over an HNSW graph.
    Hnsw(HnswConfig),
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self::Hnsw(HnswConfig::default())
    }
}

/// HNSW settings. Missing fields take [`HnswParams`]'s defaults.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct HnswConfig {
    /// Links per node on upper layers; layer 0 gets `2 * m`.
    pub m: usize,
    /// Candidates considered per insert.
    pub ef_construction: usize,
    /// Candidates considered per search.
    pub ef_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let params = HnswParams::default();
        Self {
            m: params.m,
            ef_construction: params.ef_construction,
            ef_search: params.ef_search,
        }
    }
}

impl HnswConfig {
    /// The index parameters these settings stand for.
    pub fn params(&self) -> HnswParams {
        HnswParams {
            m: self.m,
            m0: 2 * self.m,
            ef_construction: self.ef_construction,
            ef_search: self.ef_search,
            ..HnswParams::default()
        }
    }
}

/// Serde mirror of [`Metric`], which lives in a crate without serde.
#[derive(Serialize, Deserialize)]
#[serde(remote = "Metric", rename_all = "lowercase")]
enum MetricDef {
    L2,
    Cosine,
    Dot,
}

impl CollectionConfig {
    /// Reads the config saved in a collection's directory.
    pub(crate) fn read(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        toml::from_str(&text).map_err(|e| Error::corruption(path, e.to_string()))
    }

    /// Saves the config atomically.
    pub(crate) fn write(&self, path: &Path) -> Result<()> {
        let text = toml::to_string(self).map_err(|e| Error::Config(e.to_string()))?;
        nearby_storage::atomic_write(path, text.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_fields_take_defaults() {
        let config: CollectionConfig = toml::from_str("dim = 4").unwrap();
        assert_eq!(config.metric, Metric::L2);
        assert_eq!(config.index, IndexConfig::Hnsw(HnswConfig::default()));
        assert_eq!(config.index, IndexConfig::default());
    }

    #[test]
    fn every_field_parses() {
        let config: CollectionConfig = toml::from_str(
            "dim = 8\nmetric = \"cosine\"\n[index]\ntype = \"hnsw\"\nm = 8\nef_search = 100\n",
        )
        .unwrap();
        assert_eq!(config.metric, Metric::Cosine);
        let IndexConfig::Hnsw(hnsw) = &config.index else {
            panic!("expected hnsw");
        };
        assert_eq!(
            (hnsw.m, hnsw.ef_construction, hnsw.ef_search),
            (8, 200, 100)
        );
        assert_eq!(hnsw.params().m0, 16);

        let flat: CollectionConfig = toml::from_str("dim = 2\n[index]\ntype = \"flat\"").unwrap();
        assert_eq!(flat.index, IndexConfig::Flat);
    }

    #[test]
    fn unknown_values_are_rejected() {
        for text in [
            "dim = 2\nmetric = \"manhattan\"",
            "dim = 2\ncolor = \"red\"",
            "dim = 2\n[index]\ntype = \"ivf\"",
            "metric = \"l2\"",
        ] {
            assert!(toml::from_str::<CollectionConfig>(text).is_err(), "{text}");
        }
    }

    #[test]
    fn round_trips_through_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("collection.toml");
        let config = CollectionConfig {
            dim: 3,
            metric: Metric::Dot,
            index: IndexConfig::Flat,
        };
        config.write(&path).unwrap();
        assert_eq!(CollectionConfig::read(&path).unwrap(), config);

        std::fs::write(&path, "not toml [").unwrap();
        assert!(matches!(
            CollectionConfig::read(&path),
            Err(Error::Corruption { .. })
        ));
    }
}
