//! The two core HNSW routines: searching one layer, and choosing neighbors.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use nearby_core::distance::{cosine_distance_with_norms, norm};
use nearby_core::{Metric, OrdF32, TopK};

use super::graph::Graph;
use super::visited::VisitedSet;
use crate::store::VectorStore;

/// A vector to measure distances from, with its norm when the metric needs it.
///
/// Build it with [`SearchCtx::query`], so the norm always matches the vector.
#[derive(Debug, Clone, Copy)]
pub(super) struct Query<'a> {
    vector: &'a [f32],
    /// `norm(vector)` for [`Metric::Cosine`]; unused (0) for other metrics.
    norm: f32,
}

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
    /// `norms[node]` is the norm of stored vector `node` for
    /// [`Metric::Cosine`], so a cosine comparison costs one dot product
    /// instead of three. Empty for other metrics.
    pub norms: &'a [f32],
}

impl<'a> SearchCtx<'a> {
    /// Prepares `vector` for distance measurements, computing its norm once
    /// if the metric needs it.
    pub(super) fn query<'q>(&self, vector: &'q [f32]) -> Query<'q> {
        let norm = if self.norms.is_empty() {
            0.0
        } else {
            norm(vector)
        };
        Query { vector, norm }
    }

    /// Prepares stored vector `node` for distance measurements, reusing its
    /// stored norm.
    pub(super) fn node_query(&self, node: u32) -> Query<'a> {
        Query {
            vector: self.store.get(node as usize),
            norm: self.norms.get(node as usize).copied().unwrap_or(0.0),
        }
    }

    /// Measures the distance from `query` to `node`.
    ///
    /// With stored norms this is bit-for-bit the same as
    /// [`Metric::distance`], just cheaper.
    #[inline]
    pub(super) fn neighbor(&self, query: &Query<'_>, node: u32) -> Neighbor {
        let vector = self.store.get(node as usize);
        let distance = if self.norms.is_empty() {
            self.metric.distance(query.vector, vector)
        } else {
            cosine_distance_with_norms(query.vector, vector, query.norm, self.norms[node as usize])
        };
        Neighbor { node, distance }
    }

    /// Best-first search on one layer (Algorithm 2 in the paper).
    ///
    /// Returns up to `ef` nodes closest to `query`, closest first. Always
    /// returns at least the entry points, so the result is never empty.
    pub(super) fn search_layer(
        &self,
        query: &Query<'_>,
        entry_points: &[Neighbor],
        ef: usize,
        level: usize,
        visited: &mut VisitedSet,
    ) -> Vec<Neighbor> {
        self.search_layer_filtered(query, entry_points, ef, level, visited, |_| true)
    }

    /// [`SearchCtx::search_layer`], returning only nodes `admit` accepts.
    ///
    /// Rejected nodes still guide the search: they are explored like any
    /// other node that could lead closer, they just never enter the results.
    /// So the result may be empty, and the more nodes are rejected, the more
    /// of the graph a search explores before it has `ef` results.
    pub(super) fn search_layer_filtered(
        &self,
        query: &Query<'_>,
        entry_points: &[Neighbor],
        ef: usize,
        level: usize,
        visited: &mut VisitedSet,
        admit: impl Fn(u32) -> bool,
    ) -> Vec<Neighbor> {
        visited.reset(self.store.len());
        // What to explore next: a min-heap, so the closest unexplored node pops first.
        let mut candidates = BinaryHeap::new();
        // The best `ef` found so far: a max-heap, so the worst of them is on top.
        let mut results = TopK::new(ef);

        for ep in entry_points {
            visited.insert(ep.node);
            candidates.push(Reverse((OrdF32(ep.distance), ep.node)));
            if admit(ep.node) {
                results.push(u64::from(ep.node), ep.distance);
            }
        }

        while let Some(Reverse((OrdF32(distance), node))) = candidates.pop() {
            // The closest unexplored node is farther than everything we're keeping,
            // so exploring further can't improve the results.
            if results.is_full() && results.worst().is_some_and(|worst| distance > worst) {
                break;
            }
            let links = self.graph.neighbors(node, level);
            // Start loading every unvisited neighbor before measuring any of
            // them, so the memory reads overlap (see `VectorStore::prefetch`).
            for &next in links {
                if !visited.contains(next) {
                    self.store.prefetch(next as usize);
                }
            }
            for &next in links {
                if !visited.insert(next) {
                    continue;
                }
                let n = self.neighbor(query, next);
                // Only a node that could enter the results is worth exploring;
                // checking that first keeps `admit` off the common path.
                if results.is_full() && results.worst().is_some_and(|worst| n.distance > worst) {
                    continue;
                }
                let promising = if admit(n.node) {
                    // `push` keeps `n` only if it beats the current worst (or there's room).
                    results.push(u64::from(n.node), n.distance)
                } else {
                    !results.is_full() || results.worst().is_some_and(|worst| n.distance < worst)
                };
                if promising {
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
        query: &Query<'_>,
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
    /// candidate is dropped if some neighbor already kept is strictly closer
    /// to it than the node is. That spreads links across different directions,
    /// so searches can escape tight clusters, which greatly improves recall.
    ///
    /// Ties are kept, as in hnswlib. With a strict comparison, a candidate
    /// identical to the node (distance 0 to both) would always be dropped,
    /// leaving duplicate vectors with a single link each.
    pub(super) fn select_neighbors(&self, candidates: &[Neighbor], m: usize) -> Vec<Neighbor> {
        let mut selected: Vec<Neighbor> = Vec::with_capacity(m);
        for &candidate in candidates {
            if selected.len() >= m {
                break;
            }
            let from = self.node_query(candidate.node);
            let is_diverse = selected
                .iter()
                .all(|kept| self.neighbor(&from, kept.node).distance >= candidate.distance);
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
            norms: &[],
        };
        let mut visited = VisitedSet::default();

        let query = ctx.query(&[3.9, 0.0]);
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
            norms: &[],
        };
        let origin = ctx.query(&[0.0, 0.0]);
        let candidates: Vec<Neighbor> = (0..3).map(|n| ctx.neighbor(&origin, n)).collect();

        let selected = ctx.select_neighbors(&candidates, 2);
        let nodes: Vec<u32> = selected.iter().map(|n| n.node).collect();
        assert_eq!(nodes, vec![0, 2]);
    }

    /// Five points on a line, each linked to its neighbors: 0 - 1 - 2 - 3 - 4.
    fn line() -> (VectorStore, Graph) {
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
        (store, graph)
    }

    fn ctx<'a>(store: &'a VectorStore, graph: &'a Graph) -> SearchCtx<'a> {
        SearchCtx {
            store,
            graph,
            metric: Metric::L2,
            norms: &[],
        }
    }

    fn nodes(found: &[Neighbor]) -> Vec<u32> {
        found.iter().map(|n| n.node).collect()
    }

    #[test]
    fn search_layer_returns_at_most_ef_sorted() {
        let (store, graph) = line();
        let ctx = ctx(&store, &graph);
        let mut visited = VisitedSet::default();
        let query = ctx.query(&[2.2, 0.0]);
        let start = ctx.neighbor(&query, 0);

        let found = ctx.search_layer(&query, &[start], 10, 0, &mut visited);
        assert_eq!(
            nodes(&found),
            vec![2, 3, 1, 4, 0],
            "all 5 reachable, sorted"
        );
        assert!(found.windows(2).all(|w| w[0].distance <= w[1].distance));
    }

    #[test]
    fn search_layer_only_reaches_connected_nodes() {
        let (store, mut graph) = line();
        graph.set_neighbors(2, 0, &[1]); // cut the 2 -> 3 edge
        let ctx = ctx(&store, &graph);
        let mut visited = VisitedSet::default();
        let query = ctx.query(&[4.0, 0.0]);
        let start = ctx.neighbor(&query, 0);

        let found = ctx.search_layer(&query, &[start], 10, 0, &mut visited);
        assert_eq!(nodes(&found), vec![2, 1, 0]);
    }

    #[test]
    fn filtered_search_walks_through_rejected_nodes() {
        // Nodes 1 to 3 are rejected, but the only way from 0 to 4 runs through them.
        let (store, graph) = line();
        let ctx = ctx(&store, &graph);
        let mut visited = VisitedSet::default();
        let query = ctx.query(&[3.9, 0.0]);
        let start = ctx.neighbor(&query, 0);

        let found = ctx.search_layer_filtered(&query, &[start], 2, 0, &mut visited, |n| {
            !(1..=3).contains(&n)
        });
        assert_eq!(nodes(&found), vec![4, 0]);

        let none = ctx.search_layer_filtered(&query, &[start], 2, 0, &mut visited, |_| false);
        assert_eq!(none, Vec::new());
    }

    #[test]
    fn search_layer_accepts_several_entry_points() {
        let (store, graph) = line();
        let ctx = ctx(&store, &graph);
        let mut visited = VisitedSet::default();
        let query = ctx.query(&[0.0, 0.0]);
        let entries = [ctx.neighbor(&query, 4), ctx.neighbor(&query, 1)];

        let found = ctx.search_layer(&query, &entries, 2, 0, &mut visited);
        assert_eq!(nodes(&found), vec![0, 1]);
    }

    #[test]
    fn select_neighbors_edge_cases() {
        let (store, graph) = line();
        let ctx = ctx(&store, &graph);
        let origin = ctx.query(&[0.0, 0.0]);
        let candidates: Vec<Neighbor> = (1..5).map(|n| ctx.neighbor(&origin, n)).collect();

        assert!(ctx.select_neighbors(&candidates, 0).is_empty(), "m = 0");
        assert!(ctx.select_neighbors(&[], 3).is_empty(), "no candidates");
        // All candidates lie in one direction: each is closer to the previous
        // one than to the origin, so only the nearest is kept.
        assert_eq!(nodes(&ctx.select_neighbors(&candidates, 4)), vec![1]);
    }

    #[test]
    fn cosine_with_stored_norms_matches_the_metric_exactly() {
        let points = [[3.0, 4.0], [-1.0, 2.0], [0.5, -0.25], [0.0, 0.0]];
        let store = store(&points);
        let norms: Vec<f32> = points.iter().map(|p| norm(p)).collect();
        let graph = Graph::with_capacity(2, 0);
        let ctx = SearchCtx {
            store: &store,
            graph: &graph,
            metric: Metric::Cosine,
            norms: &norms,
        };

        let raw = [2.0, -7.0];
        let query = ctx.query(&raw);
        for node in 0..4u32 {
            let expected = Metric::Cosine.distance(&raw, &points[node as usize]);
            assert_eq!(
                ctx.neighbor(&query, node).distance.to_bits(),
                expected.to_bits()
            );

            let from_node = ctx.node_query(node);
            let expected = Metric::Cosine.distance(&points[node as usize], &points[0]);
            assert_eq!(
                ctx.neighbor(&from_node, 0).distance.to_bits(),
                expected.to_bits()
            );
        }
    }

    #[test]
    fn select_neighbors_keeps_ties() {
        // Linking a node at the origin. Nodes 0-2 are exact copies of it: each
        // is as close to the kept copies (0) as to the node (0), so all are
        // kept; a strict comparison would keep only the first. Node 3 is as
        // far from the kept copies as from the node (they share a position),
        // so it ties too: a neighbor on top of the node covers no direction.
        let store = store(&[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [5.0, 5.0]]);
        let graph = Graph::with_capacity(2, 0);
        let ctx = ctx(&store, &graph);
        let origin = ctx.query(&[0.0, 0.0]);
        let candidates: Vec<Neighbor> = (0..4).map(|n| ctx.neighbor(&origin, n)).collect();

        assert_eq!(
            nodes(&ctx.select_neighbors(&candidates, 4)),
            vec![0, 1, 2, 3]
        );
    }
}
