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
/// The filter comes from `RUST_LOG` if it is set and valid, otherwise from
/// [`LogConfig::level`]. Console and file output are enabled independently.
///
/// # Errors
///
/// If `cfg.level` is not a valid filter directive, the log directory cannot
/// be created, the log file cannot be opened, or a global logger is already
/// installed.
pub fn init_logging(cfg: &LogConfig) -> anyhow::Result<LogGuard> {
    let filter = match EnvFilter::try_from_default_env() {
        Ok(filter) => filter,
        Err(_) => EnvFilter::try_new(&cfg.level)
            .with_context(|| format!("invalid log level `{}`", cfg.level))?,
    };

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

    tracing_subscriber::registry()
        .with(filter)
        .with(console)
        .with(file)
        .try_init()
        .context("installing the global logger (was it already initialized?)")?;

    Ok(LogGuard { _file: guard })
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
