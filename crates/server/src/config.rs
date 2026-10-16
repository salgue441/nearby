//! Server configuration, loaded from a TOML file.
//!
//! Every field has a default, so an empty file (or no file) is a valid config.
//! Unknown keys are rejected so typos fail loudly instead of being ignored.

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub data_dir: PathBuf,
    pub log: LogConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data"),
            log: LogConfig::default(),
        }
    }
}

impl Config {
    /// Loads and parses `path`. Fails if the file is missing.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config file {}", path.display()))?;

        Self::from_toml(&text).with_context(|| format!("parsing config file {}", path.display()))
    }

    /// Like [`Config::load`], but returns the defaults if `path` does not exist.
    pub fn load_or_default(path: &Path) -> anyhow::Result<Self> {
        if path.exists() {
            Self::load(path)
        } else {
            Ok(Self::default())
        }
    }

    pub fn from_toml(text: &str) -> anyhow::Result<Self> {
        Ok(toml::from_str(text)?)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LogConfig {
    pub level: String,
    pub console: bool,
    pub console_format: ConsoleFormat,
    pub file: bool,
    pub file_dir: PathBuf,
    pub file_prefix: String,
    pub rotation: Rotation,
    pub max_files: usize,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: "info".to_owned(),
            console: true,
            console_format: ConsoleFormat::default(),
            file: true,
            file_dir: PathBuf::from("./logs"),
            file_prefix: "nearby".to_owned(),
            rotation: Rotation::default(),
            max_files: 14,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConsoleFormat {
    /// One line per event, with colors. Best for development.
    #[default]
    Full,

    /// Shorter lines, fewer details.
    Compact,

    /// One JSON oject per line. Best for log aggregators (Loki, Datadog, ...)
    Json,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Rotation {
    Minutely,
    Hourly,
    #[default]
    Daily,
    Never,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_file_gives_defaults() {
        let cfg = Config::from_toml("").unwrap();
        assert_eq!(cfg.data_dir, PathBuf::from("./data"));
        assert_eq!(cfg.log.level, "info");
        assert_eq!(cfg.log.rotation, Rotation::Daily);
    }

    #[test]
    fn partial_file_overrides_only_given_fields() {
        let cfg = Config::from_toml(
            r#"
            data_dir = "/var/lib/nearby"
            [log]
            level = "debug"
            console_format = "json"
            "#,
        )
        .unwrap();
        assert_eq!(cfg.data_dir, PathBuf::from("/var/lib/nearby"));
        assert_eq!(cfg.log.level, "debug");
        assert_eq!(cfg.log.console_format, ConsoleFormat::Json);
        assert!(cfg.log.file, "unset fields keep their defaults");
    }

    #[test]
    fn unknown_keys_are_rejected() {
        let err = Config::from_toml("data_dri = \"oops\"").unwrap_err();
        assert!(err.to_string().contains("data_dri"), "got: {err}");
    }

    #[test]
    fn missing_file_falls_back_to_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Config::load_or_default(&dir.path().join("nope.toml")).unwrap();
        assert_eq!(cfg.log.level, "info");
    }
}
