//! The two core HNSW routines: searching one layer, and choosing neighbors.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use nearby_core::{Metric, OrdF32, TopK};

use super::graph::Graph;
use super::visited::VisitedSet;
use crate::store::VectorStore;

/// A node and its distance to whatever we're currently measuring from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Neighbor {
    /// Internal node id.
    pub node: u32,
    /// Distance from the reference point, under the index's metric.
    pub distance: f32,
}

/// Read-only view of the index that the search routines need.
#[derive(Debug, Clone, Copy)]
pub(super) struct SearchCtx<'a> {
    pub store: &'a VectorStore,
    pub graph: &'a Graph,
    pub metric: Metric,
}

impl SearchCtx<'_> {
    /// Measures the distance from `query` to `node`.
    #[inline]
    pub(super) fn neighbor(&self, query: &[f32], node: u32) -> Neighbor {
        Neighbor {
            node,
            distance: self.metric.distance(query, self.store.get(node as usize)),
        }
    }

    /// Best-first search on one layer (Algorithm 2 in the paper).
    ///
    /// Returns up to `ef` nodes closest to `query`, closest first. Always
    /// returns at least the entry points, so the result is never empty.
    pub(super) fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[Neighbor],
        ef: usize,
        level: usize,
        visited: &mut VisitedSet,
    ) -> Vec<Neighbor> {
        visited.reset(self.store.len());
        // What to explore next: a min-heap, so the closest unexplored node pops first.
        let mut candidates = BinaryHeap::new();
        // The best `ef` found so far: a max-heap, so the worst of them is on top.
        let mut results = TopK::new(ef);

        for ep in entry_points {
            visited.insert(ep.node);
            candidates.push(Reverse((OrdF32(ep.distance), ep.node)));
            results.push(u64::from(ep.node), ep.distance);
        }

        while let Some(Reverse((OrdF32(distance), node))) = candidates.pop() {
            // The closest unexplored node is farther than everything we're keeping,
            // so exploring further can't improve the results.
            if results.is_full() && results.worst().is_some_and(|worst| distance > worst) {
                break;
            }
            for &next in self.graph.neighbors(node, level) {
                if !visited.insert(next) {
                    continue;
                }
                let n = self.neighbor(query, next);
                // `push` keeps `n` only if it beats the current worst (or there's room).
                if results.push(u64::from(n.node), n.distance) {
                    candidates.push(Reverse((OrdF32(n.distance), n.node)));
                }
            }
        }

        results
            .into_sorted()
            .into_iter()
            .map(|p| Neighbor {
                node: p.id as u32,
                distance: p.distance,
            })
            .collect()
    }

    /// Greedy search with `ef = 1`: used to descend quickly through the upper layers.
    pub(super) fn greedy_closest(
        &self,
        query: &[f32],
        entry: Neighbor,
        level: usize,
        visited: &mut VisitedSet,
    ) -> Neighbor {
        self.search_layer(query, &[entry], 1, level, visited)[0]
    }

    /// Picks up to `m` neighbors from `candidates` (Algorithm 4 in the paper).
    ///
    /// `candidates` must be sorted closest first, with distances measured from
    /// the node being linked. Instead of simply taking the `m` closest, a
    /// candidate is kept only if it's closer to that node than to every
    /// neighbor already kept. That spreads links across different directions,
    /// so searches can escape tight clusters, which greatly improves recall.
    pub(super) fn select_neighbors(&self, candidates: &[Neighbor], m: usize) -> Vec<Neighbor> {
        let mut selected: Vec<Neighbor> = Vec::with_capacity(m);
        for &candidate in candidates {
            if selected.len() >= m {
                break;
            }
            let vector = self.store.get(candidate.node as usize);
            let is_diverse = selected.iter().all(|kept| {
                self.metric
                    .distance(vector, self.store.get(kept.node as usize))
                    > candidate.distance
            });
            if is_diverse {
                selected.push(candidate);
            }
        }
        selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(points: &[[f32; 2]]) -> VectorStore {
        let mut store = VectorStore::with_capacity(2, points.len());
        for p in points {
            store.push(p);
        }
        store
    }

    #[test]
    fn search_layer_walks_to_the_nearest_node() {
        // Five points on a line, each linked to its neighbors: 0 - 1 - 2 - 3 - 4.
        let store = store(&[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]);
        let mut graph = Graph::with_capacity(2, 5);
        for _ in 0..5 {
            graph.add_node(0);
        }
        graph.set_neighbors(0, 0, &[1]);
        graph.set_neighbors(1, 0, &[0, 2]);
        graph.set_neighbors(2, 0, &[1, 3]);
        graph.set_neighbors(3, 0, &[2, 4]);
        graph.set_neighbors(4, 0, &[3]);
        let ctx = SearchCtx {
            store: &store,
            graph: &graph,
            metric: Metric::L2,
        };
        let mut visited = VisitedSet::default();

        let query = [3.9, 0.0];
        let start = ctx.neighbor(&query, 0);
        assert_eq!(ctx.greedy_closest(&query, start, 0, &mut visited).node, 4);

        let found = ctx.search_layer(&query, &[start], 3, 0, &mut visited);
        let nodes: Vec<u32> = found.iter().map(|n| n.node).collect();
        assert_eq!(nodes, vec![4, 3, 2]);
    }

    #[test]
    fn select_neighbors_prefers_diverse_directions() {
        // Linking a node at the origin. B is close to A, so it adds little;
        // C lies in a different direction, so it's kept even though it's farther.
        let store = store(&[[1.0, 0.0], [1.1, 0.0], [0.0, 1.2]]);
        let graph = Graph::with_capacity(2, 0);
        let ctx = SearchCtx {
            store: &store,
            graph: &graph,
            metric: Metric::L2,
        };
        let origin = [0.0, 0.0];
        let candidates: Vec<Neighbor> = (0..3).map(|n| ctx.neighbor(&origin, n)).collect();

        let selected = ctx.select_neighbors(&candidates, 2);
        let nodes: Vec<u32> = selected.iter().map(|n| n.node).collect();
        assert_eq!(nodes, vec![0, 2]);
    }
}
