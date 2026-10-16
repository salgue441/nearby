//! Keeping the `k` best candidates out of many.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::types::{PointId, ScoredPoint};

/// An `f32` with a total order, so it can be stored in a `BinaryHeap`.
///
/// `f32` only implements `PartialOrd`, because NaN is not comparable to
/// anything.
///
/// `f32::total_cmp` defines an order for every value (NaN sorts after infinity),
/// which is all a heap needs. Vectors are validated at the API boundary, so NaN
/// distances don't happen in practice.
///
/// Equality follows the same total order, so unlike `f32`, `-0.0 != 0.0`
/// and `NaN == NaN` here.
#[derive(Debug, Clone, Copy)]
pub struct OrdF32(pub f32);

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for OrdF32 {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for OrdF32 {}

/// Heap entry. The derived `Ord` compares `distance` first, then `id`,
/// so ties are broken the same way on every run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Entry {
    distance: OrdF32,
    id: PointId,
}

/// Keeps the `k` closest points seen so far.
///
/// Internally it's a max-heap of size `k`: the *worst* of the current best
/// sits on top. Checking whether a new candidate belongs is O(1), and
/// replacing the worst is O(log k). Over N candidates that's O(N log k),
/// instead of O(N log N) for sorting everything.
///
/// Candidates at the same distance are ranked by ascending id, so results
/// are deterministic regardless of insertion order.
///
/// # Examples
///
/// ```
/// use nearby_core::TopK;
///
/// let mut top = TopK::new(2);
/// top.push(1, 0.9);
/// top.push(2, 0.1);
/// top.push(3, 0.5);
///
/// let ids: Vec<u64> = top.into_sorted().iter().map(|p| p.id).collect();
/// assert_eq!(ids, [2, 3]);
/// ```
#[derive(Debug, Clone)]
pub struct TopK {
    k: usize,
    heap: BinaryHeap<Entry>,
}

impl TopK {
    /// Entries reserved up front. Past this the heap grows on demand, so a
    /// huge `k` (from an untrusted request, say) only costs memory for the
    /// candidates actually kept, instead of failing a giant allocation.
    const MAX_PREALLOC: usize = 1024;

    /// Creates an empty collector that keeps at most `k` candidates.
    ///
    /// Memory for up to 1024 entries is reserved up front; any `k` is safe,
    /// including `usize::MAX`. With `k == 0`, every candidate is rejected.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k.min(Self::MAX_PREALLOC)),
        }
    }

    /// The maximum number of candidates kept.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Number of candidates currently kept. Never more than [`TopK::k`].
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// True if no candidate has been kept yet.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// True once `k` candidates are held. From then on, a new candidate
    /// only gets in by beating [`TopK::worst`].
    pub fn is_full(&self) -> bool {
        self.heap.len() >= self.k
    }

    /// Distance of the worst candidate currently kept, if any.
    ///
    /// HNSW uses this to stop searching: once the closest unexplored node is
    /// farther than `worst()`, nothing better can be found.
    pub fn worst(&self) -> Option<f32> {
        self.heap.peek().map(|e| e.distance.0)
    }

    /// Offers a candidate. Returns `true` if it was kept.
    ///
    /// While not full, every candidate is kept. Once full, a candidate is
    /// kept only if it ranks strictly better than [`TopK::worst`] (closer,
    /// or equally close with a smaller id), and it evicts that worst one.
    pub fn push(&mut self, id: PointId, distance: f32) -> bool {
        let entry = Entry {
            distance: OrdF32(distance),
            id,
        };

        if self.heap.len() < self.k {
            self.heap.push(entry);
            return true;
        }

        // Full (or k == 0): replace the worst only if the new one is better.
        // Assigning through `PeekMut` re-sorts the heap when it goes out of scope.
        if let Some(mut worst) = self.heap.peek_mut()
            && entry < *worst
        {
            *worst = entry;
            return true;
        }

        false
    }

    /// Consumes the heap and returns the results, closest first.
    pub fn into_sorted(self) -> Vec<ScoredPoint> {
        self.heap
            .into_sorted_vec()
            .into_iter()
            .map(|e| ScoredPoint {
                id: e.id,
                distance: e.distance.0,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny deterministic pseudo-random generator, so tests need no extra crate.
    fn pseudo_random(n: usize) -> Vec<f32> {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 40) as f32 / (1u64 << 24) as f32
            })
            .collect()
    }

    #[test]
    fn keeps_the_k_smallest() {
        let distances = pseudo_random(1_000);
        let mut topk = TopK::new(10);
        for (id, &d) in distances.iter().enumerate() {
            topk.push(id as PointId, d);
        }

        // Reference answer: sort everything and take the first 10.
        let mut expected: Vec<(f32, PointId)> = distances
            .iter()
            .enumerate()
            .map(|(id, &d)| (d, id as PointId))
            .collect();
        expected.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let expected_ids: Vec<PointId> = expected[..10].iter().map(|&(_, id)| id).collect();

        let got_ids: Vec<PointId> = topk.into_sorted().iter().map(|p| p.id).collect();
        assert_eq!(got_ids, expected_ids);
    }

    #[test]
    fn results_are_sorted_closest_first() {
        let mut topk = TopK::new(3);
        for (id, d) in [(1, 0.9), (2, 0.1), (3, 0.5), (4, 0.3)] {
            topk.push(id, d);
        }
        let ids: Vec<PointId> = topk.into_sorted().iter().map(|p| p.id).collect();
        assert_eq!(ids, vec![2, 4, 3]);
    }

    #[test]
    fn fewer_candidates_than_k() {
        let mut topk = TopK::new(5);
        topk.push(7, 1.0);
        topk.push(8, 2.0);
        assert!(!topk.is_full());
        assert_eq!(topk.into_sorted().len(), 2);
    }

    #[test]
    fn worst_tracks_the_boundary() {
        let mut topk = TopK::new(2);
        assert_eq!(topk.worst(), None);
        topk.push(1, 0.5);
        topk.push(2, 0.2);
        assert_eq!(topk.worst(), Some(0.5));
        assert!(topk.push(3, 0.1), "better candidate gets in");
        assert_eq!(topk.worst(), Some(0.2));
        assert!(!topk.push(4, 0.9), "worse candidate is rejected");
    }

    #[test]
    fn k_zero_keeps_nothing() {
        let mut topk = TopK::new(0);
        assert!(!topk.push(1, 0.0));
        assert_eq!(topk.into_sorted(), Vec::new());
    }
}
