//! Small value types shared by every crate.

/// A point's external identifier, chosen by the user.
pub type PointId = u64;

/// How the distance between two vectors is measured.
///
/// Every metric is expressed as a *distance*: lower always means closer.
/// That single rule lets heaps, merges and HNSW ignore which metric is in use.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Metric {
    /// Squared Euclidean distance. Same ranking as Euclidean, without the `sqrt`.
    #[default]
    L2,
    /// `1 - cosine_similarity`, in `[0, 2]`.
    Cosine,
    /// Negative dot product, so larger dot products rank first.
    Dot,
}

/// A search result: which point, and how far it is from the query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredPoint {
    /// The point's id.
    pub id: PointId,
    /// Distance to the query under the index's [`Metric`]. Lower is closer.
    pub distance: f32,
}
