//! On-disk persistence: the write-ahead log, segments and recovery.
//!
//! Each collection lives in its own directory, managed by a
//! [`CollectionStore`]:
//!
//! - Every write is appended to a **write-ahead log** first, as checksummed
//!   records, and (by default) synced before it is acknowledged.
//! - A **flush** turns the log's net changes into an immutable **segment**
//!   file and starts a new log, so the log stays short.
//! - **Compaction** merges every segment into one holding only live points.
//! - A **manifest**, replaced atomically, names the live log and segments, so
//!   a crash at any moment leaves a consistent set of files.
//!
//! Recovery replays the segments oldest first, then the log, dropping an
//! incomplete record a crash may have left at its end.
//!
//! The store is a log of [`Op`]s and keeps nothing in memory; indexes are
//! rebuilt from the replayed ops by whoever opens it.

mod codec;
mod fsio;
mod manifest;
mod segment;
mod store;
mod wal;

pub use fsio::atomic_write;
pub use store::{CollectionStore, StoreOptions};
pub use wal::{Durability, Op};
