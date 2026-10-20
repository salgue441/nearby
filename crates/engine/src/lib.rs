//! Collections, the id map and query execution on top of indexes and storage.
//!
//! A [`Database`] holds named [`Collection`]s under one data directory. Each
//! collection has a fixed dimension, metric and index ([`CollectionConfig`]),
//! and supports upserts, deletes, point lookups and nearest-neighbor search:
//!
//! - **Durability.** Every write goes through the collection's write-ahead
//!   log (see `nearby-storage`) before it is applied, and reopening the
//!   database replays it.
//! - **Ids.** Points have caller-chosen [`PointId`](nearby_core::PointId)s.
//!   Writing an existing id replaces the point.
//! - **Deletes.** Indexes only append, so a replaced or deleted point's old
//!   vector stays in the index, where searches skip it; once dead vectors
//!   outnumber live ones, the index is rebuilt without them.
//! - **Recovery.** Opening a collection replays its storage and rebuilds the
//!   index, in parallel where the index allows. An HNSW collection saves a
//!   snapshot on checkpoint, so it only replays the log written since.
//! - **Indexes.** Flat (exact), HNSW, or PQ: compressed codes whose
//!   candidates are re-ranked by exact distance (see [`PqConfig`]).

mod backend;
mod collection;
mod config;
mod database;
mod points;
mod pq;
mod snapshot;

use std::num::NonZeroUsize;

pub use collection::{Collection, CollectionInfo};
pub use config::{CollectionConfig, HnswConfig, IndexConfig, PqConfig};
pub use database::{CollectionHandle, Database};
pub use nearby_storage::Durability;

/// Settings shared by every collection in a [`Database`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineOptions {
    /// When writes are forced to disk.
    pub durability: Durability,
    /// Threads that build indexes.
    pub threads: NonZeroUsize,
    /// Writes a collection's log holds before it is flushed into a segment.
    pub flush_after_ops: usize,
    /// Segments a collection keeps before they are compacted into one.
    pub max_segments: usize,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            durability: Durability::Sync,
            threads: std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN),
            flush_after_ops: 50_000,
            max_segments: 8,
        }
    }
}
