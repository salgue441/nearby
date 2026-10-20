//! Server entry point: loads the config, opens the database and serves the
//! HTTP API until Ctrl-C (or SIGTERM on Unix), then flushes every collection.
//!
//! Usage: `cargo run -p nearby-server -- [path/to/nearby.toml]`

mod api;
mod auth;
mod config;
mod telemetry;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use nearby_engine::Database;

use crate::api::ApiOptions;
use crate::auth::ApiKey;
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

    let api_key = std::env::var("NEARBY_API_KEY")
        .ok()
        .or_else(|| config.server.api_key.clone())
        .map(|key| ApiKey::new(&key))
        .transpose()
        .context("invalid API key")?;
    if api_key.is_none() {
        tracing::warn!("no API key configured: every request is accepted");
    }
    let options = ApiOptions {
        max_body_bytes: config.server.max_body_bytes,
        api_key,
    };

    serve(listener, db, options, shutdown_signal()).await?;
    tracing::info!("nearby stopped");
    Ok(())
}

/// Serves the API on `listener` until `shutdown` resolves, then lets
/// in-flight requests finish and flushes every collection.
async fn serve(
    listener: tokio::net::TcpListener,
    db: Arc<Database>,
    options: ApiOptions,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    axum::serve(listener, api::router(Arc::clone(&db), options))
        .with_graceful_shutdown(shutdown)
        .await
        .context("serving the HTTP API")?;

    tracing::info!("shutting down: flushing collections");
    tokio::task::spawn_blocking(move || db.flush_all())
        .await?
        .context("flushing collections")
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

#[cfg(test)]
mod tests {
    use nearby_engine::{Durability, EngineOptions};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    use super::*;

    /// Sends one HTTP/1.1 request over a fresh connection and returns the
    /// status code and body.
    async fn request(
        addr: std::net::SocketAddr,
        method: &str,
        path: &str,
        body: &str,
    ) -> (u16, String) {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        let head = format!(
            "{method} {path} HTTP/1.1\r\nhost: test\r\ncontent-type: application/json\r\n\
             content-length: {}\r\nconnection: close\r\n\r\n",
            body.len()
        );
        stream.write_all(head.as_bytes()).await.unwrap();
        stream.write_all(body.as_bytes()).await.unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).await.unwrap();
        let status = response[9..12].parse().unwrap();
        let body = response
            .split_once("\r\n\r\n")
            .map_or("", |(_, b)| b)
            .to_owned();
        (status, body)
    }

    #[tokio::test]
    async fn shutdown_flushes_every_collection() {
        let dir = tempfile::tempdir().unwrap();
        let options = EngineOptions {
            durability: Durability::NoSync,
            ..EngineOptions::default()
        };
        let db = Arc::new(Database::open(dir.path(), options).unwrap());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let api = ApiOptions {
            max_body_bytes: 1 << 20,
            api_key: None,
        };
        let server = tokio::spawn(serve(listener, db, api, async {
            let _ = stopped.await;
        }));

        let (status, _) = request(addr, "PUT", "/collections/docs", r#"{"dim": 2}"#).await;
        assert_eq!(status, 201);
        let points = r#"{"points": [{"id": 1, "vector": [1, 2]}]}"#;
        let (status, body) = request(addr, "PUT", "/collections/docs/points", points).await;
        assert_eq!((status, body.as_str()), (200, r#"{"upserted":1}"#));

        stop.send(()).unwrap();
        server.await.unwrap().unwrap();

        // The write moved from the log into a segment during shutdown.
        let db = Database::open(dir.path(), options).unwrap();
        let info = db.collection("docs").unwrap().read().unwrap().info();
        assert_eq!((info.points, info.wal_ops, info.segments), (1, 0, 1));
    }
}
