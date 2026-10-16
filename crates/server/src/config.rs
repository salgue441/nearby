//! Server configuration, loaded from a TOML file.
//!
//! Every field has a default, so an empty file (or no file) is a valid config.
//! Unknown keys are rejected so typos fail loudly instead of being ignored.

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Deserialize;

/// Top-level server configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Where collections and their segments are stored. Default: `./data`.
    pub data_dir: PathBuf,
    /// Logging settings, from the `[log]` table.
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
    ///
    /// # Errors
    ///
    /// If the file cannot be read, or its contents fail [`Config::from_toml`].
    /// The error message names the file.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config file {}", path.display()))?;

        Self::from_toml(&text).with_context(|| format!("parsing config file {}", path.display()))
    }

    /// Like [`Config::load`], but returns the defaults if `path` does not exist.
    ///
    /// # Errors
    ///
    /// Same as [`Config::load`] when the file exists. A missing file is not
    /// an error.
    pub fn load_or_default(path: &Path) -> anyhow::Result<Self> {
        if path.exists() {
            Self::load(path)
        } else {
            Ok(Self::default())
        }
    }

    /// Parses a config from TOML text. Missing fields take their defaults.
    ///
    /// # Errors
    ///
    /// If `text` is not valid TOML, has an unknown key, or has a value of
    /// the wrong type (for example an unknown `rotation`).
    pub fn from_toml(text: &str) -> anyhow::Result<Self> {
        Ok(toml::from_str(text)?)
    }
}

/// The `[log]` table: where logs go and how much is written.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LogConfig {
    /// An [`EnvFilter`] directive such as `"info"` or `"nearby_index=debug,info"`.
    /// The `RUST_LOG` environment variable, when set, takes precedence.
    /// Default: `"info"`.
    ///
    /// [`EnvFilter`]: tracing_subscriber::EnvFilter
    pub level: String,
    /// Write logs to stdout. Default: `true`.
    pub console: bool,
    /// How console lines look. Default: [`ConsoleFormat::Full`].
    pub console_format: ConsoleFormat,
    /// Write JSON logs to rolling files in [`LogConfig::file_dir`]. Default: `true`.
    pub file: bool,
    /// Directory for log files, created if missing. Default: `./logs`.
    pub file_dir: PathBuf,
    /// File name prefix; files are named `<prefix>.<date>.log`. Default: `"nearby"`.
    pub file_prefix: String,
    /// How often a new log file is started. Default: [`Rotation::Daily`].
    pub rotation: Rotation,
    /// How many log files to keep; older ones are deleted. `0` keeps all of
    /// them. Default: `14`.
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

/// Format of console log lines. Written in lowercase in TOML (`"json"`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConsoleFormat {
    /// One line per event, with colors. Best for development.
    #[default]
    Full,

    /// Shorter lines, fewer details.
    Compact,

    /// One JSON object per line. Best for log aggregators (Loki, Datadog, ...)
    Json,
}

/// How often the log file rolls over. Written in lowercase in TOML (`"hourly"`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Rotation {
    /// A new file every minute. Mostly useful for testing rotation.
    Minutely,
    /// A new file every hour.
    Hourly,
    /// A new file every day.
    #[default]
    Daily,
    /// A single file that grows forever.
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
