//! Shared types and primitives used by every other `nearby` crate.
//!
//! Keep this crate small: only put code here when two or more crates need it.

pub mod distance;
mod error;
mod topk;
mod types;

pub use error::{Error, Result, validate_vector};
pub use topk::{OrdF32, TopK};
pub use types::{Metric, PointId, ScoredPoint};
