//! Tracks which nodes a search has already looked at.

/// A visited set that resets in O(1).
///
/// Every search needs a fresh "seen" set. A `HashSet` is slow, and clearing a
/// `Vec<bool>` of size N costs O(N) per query. Instead, each slot stores the
/// generation in which it was last visited; starting a new search just bumps
/// the generation, which makes every old mark stale at once.
#[derive(Debug, Clone, Default)]
pub(super) struct VisitedSet {
    marks: Vec<u32>,
    generation: u32,
}

impl VisitedSet {
    /// Starts a new search over nodes `0..len`.
    pub(super) fn reset(&mut self, len: usize) {
        if self.marks.len() < len {
            self.marks.resize(len, 0);
        }
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            // Wrapped around after 4 billion searches: old marks could collide,
            // so pay for one real clear.
            self.marks.fill(0);
            self.generation = 1;
        }
    }

    /// Whether `node` was visited in the current search. Doesn't mark it.
    #[inline]
    pub(super) fn contains(&self, node: u32) -> bool {
        self.marks[node as usize] == self.generation
    }

    /// Marks `node` as visited. Returns `true` if it wasn't visited before.
    #[inline]
    pub(super) fn insert(&mut self, node: u32) -> bool {
        let mark = &mut self.marks[node as usize];
        if *mark == self.generation {
            false
        } else {
            *mark = self.generation;
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_reports_first_visit_only() {
        let mut v = VisitedSet::default();
        v.reset(10);
        assert!(!v.contains(3));
        assert!(v.insert(3));
        assert!(!v.insert(3));
        assert!(v.contains(3), "contains sees the mark");
        assert!(!v.contains(4), "contains does not mark");
        assert!(v.insert(4));
    }

    #[test]
    fn reset_forgets_previous_search() {
        let mut v = VisitedSet::default();
        v.reset(10);
        v.insert(3);
        v.reset(10);
        assert!(v.insert(3));
    }

    #[test]
    fn reset_grows_to_fit_new_nodes() {
        let mut v = VisitedSet::default();
        v.reset(2);
        v.reset(100);
        assert!(v.insert(99));
    }

    #[test]
    fn survives_generation_wraparound() {
        let mut v = VisitedSet::default();
        v.reset(4);
        v.insert(1);
        v.generation = u32::MAX;
        v.reset(4);
        assert_eq!(v.generation, 1);
        assert!(v.insert(1), "stale mark must not count as visited");
    }
}
