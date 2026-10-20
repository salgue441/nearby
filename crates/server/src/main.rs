//! Server entry point: loads the config, opens the database and serves the
//! HTTP API until Ctrl-C (or SIGTERM on Unix), then flushes every collection.
//!
//! Usage: `cargo run -p nearby-server -- [path/to/nearby.toml]`

mod api;
mod config;
mod telemetry;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use nearby_engine::Database;

use crate::config::Config;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config_path = std::env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from("nearby.toml"), PathBuf::from);

    let config = Config::load_or_default(&config_path)?;
    let _guard = telemetry::init_logging(&config.log).context("initializing logging")?;

    telemetry::install_panic_hook();
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        config = %config_path.display(),
        data_dir = %config.data_dir.display(),
        "nearby starting"
    );
    tracing::debug!(?config, "full configuration");

    let data_dir = config.data_dir.clone();
    let options = config.engine.options();
    let db = tokio::task::spawn_blocking(move || Database::open(&data_dir, options))
        .await?
        .with_context(|| format!("opening the database in {}", config.data_dir.display()))?;
    let db = Arc::new(db);

    let listener = tokio::net::TcpListener::bind(config.server.listen)
        .await
        .with_context(|| format!("listening on {}", config.server.listen))?;
    tracing::info!(address = %listener.local_addr()?, "serving the HTTP API");

    axum::serve(
        listener,
        api::router(Arc::clone(&db), config.server.max_body_bytes),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .context("serving the HTTP API")?;

    tracing::info!("shutting down: flushing collections");
    tokio::task::spawn_blocking(move || db.flush_all())
        .await?
        .context("flushing collections")?;
    tracing::info!("nearby stopped");
    Ok(())
}

/// Resolves on Ctrl-C, or on SIGTERM on Unix (what service managers send).
async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(err) = tokio::signal::ctrl_c().await {
            tracing::error!(%err, "cannot listen for Ctrl-C");
            std::future::pending::<()>().await;
        }
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(err) => {
                tracing::error!(%err, "cannot listen for SIGTERM");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {}
        () = terminate => {}
    }
}
