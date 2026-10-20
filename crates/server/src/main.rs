//! Server entry point: loads the config, opens the database and serves the
//! HTTP API until Ctrl-C (or SIGTERM on Unix), then flushes every collection.
//!
//! Usage: `cargo run -p nearby-server -- [path/to/nearby.toml]`

mod api;
mod auth;
mod config;
mod metrics;
mod telemetry;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use axum_server::tls_rustls::RustlsConfig;
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
    let tls = match config.server.tls()? {
        Some((cert, key)) => {
            tracing::info!(cert = %cert.display(), "serving HTTPS");
            Some(tls_config(cert, key).await?)
        }
        None => None,
    };

    serve(listener, db, options, tls, shutdown_signal()).await?;
    tracing::info!("nearby stopped");
    Ok(())
}

/// Loads a PEM certificate chain and private key for HTTPS.
///
/// # Errors
///
/// If either file can't be read or parsed.
async fn tls_config(cert: &Path, key: &Path) -> anyhow::Result<RustlsConfig> {
    // rustls needs a crypto provider chosen once per process; a second call
    // finds one installed already, which is fine.
    let _ = rustls::crypto::ring::default_provider().install_default();
    RustlsConfig::from_pem_file(cert, key)
        .await
        .with_context(|| {
            format!(
                "loading TLS certificate {} and key {}",
                cert.display(),
                key.display()
            )
        })
}

/// Serves the API on `listener`, over HTTPS when `tls` is given, until
/// `shutdown` resolves, then lets in-flight requests finish and flushes
/// every collection.
async fn serve(
    listener: tokio::net::TcpListener,
    db: Arc<Database>,
    options: ApiOptions,
    tls: Option<RustlsConfig>,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    let app = api::router(Arc::clone(&db), options);
    if let Some(tls) = tls {
        let handle = axum_server::Handle::new();
        let stopper = handle.clone();
        tokio::spawn(async move {
            shutdown.await;
            stopper.graceful_shutdown(None);
        });
        axum_server::from_tcp_rustls(listener.into_std()?, tls)?
            .handle(handle)
            .serve(app.into_make_service())
            .await
            .context("serving the HTTPS API")?;
    } else {
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown)
            .await
            .context("serving the HTTP API")?;
    }

    tracing::info!("shutting down: checkpointing collections");
    tokio::task::spawn_blocking(move || db.checkpoint_all())
        .await?
        .context("checkpointing collections")
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
        let server = tokio::spawn(serve(listener, db, api, None, async {
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

    #[tokio::test]
    async fn https_is_served_when_configured() {
        use rustls::pki_types::pem::PemObject;
        use rustls::pki_types::{CertificateDer, ServerName};

        let certs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/tls");
        let tls = tls_config(&certs.join("cert.pem"), &certs.join("key.pem"))
            .await
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Database::open(dir.path(), EngineOptions::default()).unwrap());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let api = ApiOptions {
            max_body_bytes: 1 << 20,
            api_key: None,
        };
        let server = tokio::spawn(serve(listener, db, api, Some(tls), async {
            let _ = stopped.await;
        }));

        // A client that trusts only the CA that signed the test certificate
        // (valid until 2126; its private key was discarded).
        let mut roots = rustls::RootCertStore::empty();
        roots
            .add(CertificateDer::from_pem_file(certs.join("ca.pem")).unwrap())
            .unwrap();
        let client = rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        let connector = tokio_rustls::TlsConnector::from(Arc::new(client));
        let tcp = TcpStream::connect(addr).await.unwrap();
        let name = ServerName::try_from("localhost").unwrap();
        let mut stream = connector.connect(name, tcp).await.unwrap();
        stream
            .write_all(b"GET /health HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).await.unwrap();
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        assert!(response.ends_with(r#"{"status":"ok"}"#), "{response}");

        // Plain HTTP on the same port gets no HTTP answer.
        let mut plain = TcpStream::connect(addr).await.unwrap();
        plain
            .write_all(b"GET /health HTTP/1.1\r\nhost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut answer = Vec::new();
        let _ = plain.read_to_end(&mut answer).await;
        assert!(!answer.starts_with(b"HTTP/1.1 200"));

        stop.send(()).unwrap();
        server.await.unwrap().unwrap();
    }
}
