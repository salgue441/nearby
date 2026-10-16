//! Server entry point. For now it only loads config and
//! sets up logging.
//!
//! Usage: `cargo run -p nearby-server -- [path/to/nearby.toml]`

mod config;
mod telemetry;

use anyhow::Context;
use std::path::PathBuf;

use crate::config::Config;

fn main() -> anyhow::Result<()> {
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
    tracing::warn!("server not implemented yet");

    Ok(())
}
