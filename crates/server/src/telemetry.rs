//! Logging setup: console and/or rolling log files, built on `tracing`.
//!
//! Only the server binary sets up logging. Library crates just use the macros:
//!
//! ```ignore
//! tracing::info!(collection = %name, points = n, "segment flushed");
//! tracing::debug!(?params, "building HNSW");
//! ```
//!
//! and `#[tracing::instrument]` on functions worth timing.

use anyhow::Context;
use tracing::Dispatch;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_appender::rolling::{self, RollingFileAppender};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer, fmt};

use crate::config::{ConsoleFormat, LogConfig, Rotation};

/// Keeps the background file-writer thread alive.
///
/// File logs are written on a separate thread so logging never blocks a query.
/// When this guard is dropped, buffered lines are flushed and the thread stops,
/// so hold it in `main` for the lifetime of the program.
#[must_use = "dropping the guard stops file logging; bind it with `let _guard = ...`"]
#[derive(Debug)]
pub struct LogGuard {
    _file: Option<WorkerGuard>,
}

/// Install the global logger. Call once, at the top of `main`.
///
/// The filter comes from the `RUST_LOG` environment variable if it is set,
/// otherwise from [`LogConfig::level`]. Console and file output are enabled
/// independently.
///
/// # Errors
///
/// If `RUST_LOG` is set but invalid, anything [`build_dispatch`] rejects,
/// or if a global logger is already installed.
pub fn init_logging(cfg: &LogConfig) -> anyhow::Result<LogGuard> {
    let rust_log = match std::env::var(EnvFilter::DEFAULT_ENV) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(err @ std::env::VarError::NotUnicode(_)) => {
            return Err(err).context("reading RUST_LOG");
        }
    };

    let (dispatch, guard) = build_dispatch(cfg, rust_log.as_deref())?;
    dispatch
        .try_init()
        .context("installing the global logger (was it already initialized?)")?;

    Ok(guard)
}

/// Builds the logger described by `cfg` without installing it.
///
/// `rust_log` is the value of `RUST_LOG`, passed in rather than read here so
/// tests don't have to mutate the process environment. When it is `Some`
/// and not blank it overrides [`LogConfig::level`].
///
/// Tests can activate the result for one thread only, with
/// [`tracing::dispatcher::with_default`].
///
/// # Errors
///
/// If the chosen filter is not a valid directive, the log directory cannot be
/// created, or the log file cannot be opened.
pub fn build_dispatch(
    cfg: &LogConfig,
    rust_log: Option<&str>,
) -> anyhow::Result<(Dispatch, LogGuard)> {
    let filter = build_filter(&cfg.level, rust_log)?;

    let console = cfg.console.then(|| match cfg.console_format {
        ConsoleFormat::Full => fmt::layer().with_target(true).boxed(),
        ConsoleFormat::Compact => fmt::layer().compact().with_target(false).boxed(),
        ConsoleFormat::Json => fmt::layer().json().boxed(),
    });

    let (file, guard) = if cfg.file {
        std::fs::create_dir_all(&cfg.file_dir)
            .with_context(|| format!("creating log directory {}", cfg.file_dir.display()))?;

        let mut builder = RollingFileAppender::builder()
            .rotation(match cfg.rotation {
                Rotation::Minutely => rolling::Rotation::MINUTELY,
                Rotation::Hourly => rolling::Rotation::HOURLY,
                Rotation::Daily => rolling::Rotation::DAILY,
                Rotation::Never => rolling::Rotation::NEVER,
            })
            .filename_prefix(&cfg.file_prefix)
            .filename_suffix("log");

        if cfg.max_files > 0 {
            builder = builder.max_log_files(cfg.max_files);
        }

        let appender = builder.build(&cfg.file_dir).context("opening log file")?;
        let (writer, guard) = tracing_appender::non_blocking(appender);
        let layer = fmt::layer()
            .json()
            .with_ansi(false)
            .with_writer(writer)
            .boxed();

        (Some(layer), Some(guard))
    } else {
        (None, None)
    };

    let subscriber = tracing_subscriber::registry()
        .with(filter)
        .with(console)
        .with(file);

    Ok((Dispatch::new(subscriber), LogGuard { _file: guard }))
}

/// Parses `rust_log` if it is set and not blank, otherwise `level`.
///
/// An invalid `rust_log` is an error rather than a silent fallback to
/// `level`, so a typo in the environment doesn't go unnoticed.
fn build_filter(level: &str, rust_log: Option<&str>) -> anyhow::Result<EnvFilter> {
    match rust_log.filter(|value| !value.trim().is_empty()) {
        Some(value) => EnvFilter::try_new(value)
            .with_context(|| format!("invalid {} `{value}`", EnvFilter::DEFAULT_ENV)),
        None => EnvFilter::try_new(level).with_context(|| format!("invalid log level `{level}`")),
    }
}

/// Routes panics through `tracing`, so they end up in the log file too
/// (by default a panic message only goes to stderr).
pub fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let location = info.location().map(ToString::to_string).unwrap_or_default();
        let message = info
            .payload()
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| info.payload().downcast_ref::<String>().map(String::as_str))
            .unwrap_or("<non-string panic payload>");

        tracing::error!(%location, message, "panic");
        default_hook(info);
    }));
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tracing::level_filters::LevelFilter;

    use super::*;

    /// File logging only, into `dir`, with a fixed file name (`test.log`).
    fn file_only(dir: &Path) -> LogConfig {
        LogConfig {
            console: false,
            file_dir: dir.to_path_buf(),
            file_prefix: "test".to_owned(),
            rotation: Rotation::Never,
            ..LogConfig::default()
        }
    }

    #[test]
    fn filter_uses_config_level_when_rust_log_is_unset() {
        let filter = build_filter("debug", None).unwrap();
        assert_eq!(filter.max_level_hint(), Some(LevelFilter::DEBUG));
    }

    #[test]
    fn rust_log_overrides_config_level() {
        let filter = build_filter("info", Some("trace")).unwrap();
        assert_eq!(filter.max_level_hint(), Some(LevelFilter::TRACE));
    }

    #[test]
    fn blank_rust_log_is_treated_as_unset() {
        let filter = build_filter("warn", Some("  ")).unwrap();
        assert_eq!(filter.max_level_hint(), Some(LevelFilter::WARN));
    }

    #[test]
    fn invalid_rust_log_is_an_error_not_a_fallback() {
        let err = build_filter("info", Some("nearby=loud")).unwrap_err();
        assert!(err.to_string().contains("RUST_LOG"), "got: {err}");
    }

    #[test]
    fn invalid_config_level_is_an_error() {
        let err = build_filter("nearby=loud", None).unwrap_err();
        assert!(err.to_string().contains("invalid log level"), "got: {err}");
    }

    #[test]
    fn file_logging_writes_json_lines() {
        let dir = tempfile::tempdir().unwrap();
        let log_dir = dir.path().join("logs");
        let (dispatch, guard) = build_dispatch(&file_only(&log_dir), None).unwrap();

        tracing::dispatcher::with_default(&dispatch, || {
            tracing::info!(points = 3, "segment flushed");
            tracing::debug!("filtered out at the default `info` level");
        });
        drop(dispatch);
        drop(guard); // flushes the background writer

        let text = std::fs::read_to_string(log_dir.join("test.log")).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 1, "got: {text}");
        assert!(lines[0].starts_with('{'), "not JSON: {}", lines[0]);
        assert!(lines[0].contains("segment flushed"));
        assert!(lines[0].contains(r#""points":3"#));
    }

    #[test]
    fn disabled_file_logging_touches_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let log_dir = dir.path().join("logs");
        let cfg = LogConfig {
            file: false,
            ..file_only(&log_dir)
        };

        let (dispatch, _guard) = build_dispatch(&cfg, None).unwrap();
        tracing::dispatcher::with_default(&dispatch, || tracing::error!("nowhere to go"));
        assert!(!log_dir.exists());
    }

    #[test]
    fn unwritable_log_directory_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("not-a-dir");
        std::fs::write(&blocker, "").unwrap();

        let err = build_dispatch(&file_only(&blocker.join("logs")), None).unwrap_err();
        assert!(
            err.to_string().contains("creating log directory"),
            "got: {err}"
        );
    }
}
