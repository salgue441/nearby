//! Server configuration, loaded from a TOML file.
//!
//! Every field has a default, so an empty file (or no file) is a valid config.
//! Unknown keys are rejected so typos fail loudly instead of being ignored.

use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::Context;
use nearby_engine::{Durability, EngineOptions};
use serde::Deserialize;

/// Top-level server configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Where collections and their segments are stored. Default: `./data`.
    pub data_dir: PathBuf,
    /// Network settings, from the `[server]` table.
    pub server: ServerConfig,
    /// Storage and indexing settings, from the `[engine]` table.
    pub engine: EngineConfig,
    /// Logging settings, from the `[log]` table.
    pub log: LogConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data"),
            server: ServerConfig::default(),
            engine: EngineConfig::default(),
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

/// The `[server]` table: where the HTTP API listens.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    /// Address and port to listen on. Default: `127.0.0.1:7700`, reachable
    /// from this machine only; use `0.0.0.0:7700` to listen everywhere.
    pub listen: SocketAddr,
    /// Largest request body accepted, in bytes. Default: 64 MiB.
    pub max_body_bytes: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: SocketAddr::from(([127, 0, 0, 1], 7700)),
            max_body_bytes: 64 << 20,
        }
    }
}

/// The `[engine]` table: durability and indexing.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineConfig {
    /// `"sync"` forces every write to disk before it is acknowledged;
    /// `"nosync"` leaves that to the OS, surviving a crash of the server but
    /// not of the machine. Default: `"sync"`.
    pub durability: DurabilityConfig,
    /// Threads that build indexes. `0` uses every core. Default: `0`.
    pub threads: usize,
    /// Writes a collection's log holds before it is flushed into a
    /// segment. Default: `50000`.
    pub flush_after_ops: usize,
    /// Segments a collection keeps before they are merged. Default: `8`.
    pub max_segments: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        let defaults = EngineOptions::default();
        Self {
            durability: DurabilityConfig::Sync,
            threads: 0,
            flush_after_ops: defaults.flush_after_ops,
            max_segments: defaults.max_segments,
        }
    }
}

impl EngineConfig {
    /// The engine options these settings stand for.
    pub fn options(&self) -> EngineOptions {
        let defaults = EngineOptions::default();
        EngineOptions {
            durability: match self.durability {
                DurabilityConfig::Sync => Durability::Sync,
                DurabilityConfig::NoSync => Durability::NoSync,
            },
            threads: NonZeroUsize::new(self.threads).unwrap_or(defaults.threads),
            flush_after_ops: self.flush_after_ops.max(1),
            max_segments: self.max_segments.max(1),
        }
    }
}

/// When writes reach the disk. Written in lowercase in TOML (`"nosync"`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DurabilityConfig {
    /// Every write is synced before it is acknowledged.
    #[default]
    Sync,
    /// Writes are left to the OS to sync.
    NoSync,
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

    #[test]
    fn defaults_are_documented_values() {
        let cfg = Config::default();
        assert_eq!(cfg.server.listen.to_string(), "127.0.0.1:7700");
        assert_eq!(cfg.server.max_body_bytes, 64 << 20);
        assert_eq!(cfg.engine.durability, DurabilityConfig::Sync);
        assert_eq!(cfg.engine.threads, 0);
        assert_eq!(cfg.engine.flush_after_ops, 50_000);
        assert_eq!(cfg.engine.max_segments, 8);

        let log = cfg.log;
        assert_eq!(log.level, "info");
        assert!(log.console);
        assert_eq!(log.console_format, ConsoleFormat::Full);
        assert!(log.file);
        assert_eq!(log.file_dir, PathBuf::from("./logs"));
        assert_eq!(log.file_prefix, "nearby");
        assert_eq!(log.rotation, Rotation::Daily);
        assert_eq!(log.max_files, 14);
    }

    #[test]
    fn server_and_engine_tables_parse() {
        let cfg = Config::from_toml(
            r#"
            [server]
            listen = "0.0.0.0:9000"
            [engine]
            durability = "nosync"
            threads = 3
            "#,
        )
        .unwrap();
        assert_eq!(cfg.server.listen.port(), 9000);
        let options = cfg.engine.options();
        assert_eq!(options.durability, Durability::NoSync);
        assert_eq!(options.threads.get(), 3);
        assert_eq!(options.flush_after_ops, 50_000);

        assert!(
            Config::from_toml(
                "[server]
listen = \"nowhere\""
            )
            .is_err()
        );
        assert!(
            Config::from_toml(
                "[engine]
durability = \"maybe\""
            )
            .is_err()
        );
    }

    #[test]
    fn zero_threads_means_every_core() {
        let options = EngineConfig::default().options();
        assert_eq!(options.threads, EngineOptions::default().threads);
    }

    #[test]
    fn every_enum_value_parses_from_lowercase() {
        for (text, format) in [
            ("full", ConsoleFormat::Full),
            ("compact", ConsoleFormat::Compact),
            ("json", ConsoleFormat::Json),
        ] {
            let cfg = Config::from_toml(&format!("log.console_format = \"{text}\"")).unwrap();
            assert_eq!(cfg.log.console_format, format);
        }

        for (text, rotation) in [
            ("minutely", Rotation::Minutely),
            ("hourly", Rotation::Hourly),
            ("daily", Rotation::Daily),
            ("never", Rotation::Never),
        ] {
            let cfg = Config::from_toml(&format!("log.rotation = \"{text}\"")).unwrap();
            assert_eq!(cfg.log.rotation, rotation);
        }
    }

    #[test]
    fn enum_values_are_case_sensitive() {
        assert!(Config::from_toml("log.rotation = \"Daily\"").is_err());
        assert!(Config::from_toml("log.rotation = \"weekly\"").is_err());
    }

    #[test]
    fn unknown_keys_in_log_table_are_rejected() {
        let err = Config::from_toml("[log]\nlevle = \"debug\"").unwrap_err();
        assert!(err.to_string().contains("levle"), "got: {err}");
    }

    #[test]
    fn wrong_value_types_are_rejected() {
        assert!(Config::from_toml("log.console = \"yes\"").is_err());
        assert!(Config::from_toml("log.max_files = -1").is_err());
    }

    #[test]
    fn invalid_toml_is_rejected() {
        assert!(Config::from_toml("data_dir = ").is_err());
    }

    #[test]
    fn load_reads_an_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nearby.toml");
        std::fs::write(&path, "[log]\nmax_files = 3\n").unwrap();

        assert_eq!(Config::load(&path).unwrap().log.max_files, 3);
        assert_eq!(Config::load_or_default(&path).unwrap().log.max_files, 3);
    }

    #[test]
    fn load_fails_on_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let err = Config::load(&dir.path().join("nope.toml")).unwrap_err();
        assert!(err.to_string().contains("nope.toml"), "got: {err}");
    }

    #[test]
    fn load_or_default_still_reports_parse_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.toml");
        std::fs::write(&path, "nonsense = true").unwrap();

        let err = Config::load_or_default(&path).unwrap_err();
        assert!(err.to_string().contains("bad.toml"), "got: {err}");
    }
}
