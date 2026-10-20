//! Linking new nodes into the graph (Algorithm 1 in the paper).
//!
//! Nodes are inserted in chunks, and each chunk takes two steps:
//!
//! 1. **Planning** chooses every new node's neighbors on every layer it
//!    lives on. It only reads the index, so a chunk's nodes are planned in
//!    parallel. Each one searches the graph as it was before the chunk, and
//!    is measured directly against the chunk's earlier nodes, which that
//!    graph does not link to yet.
//! 2. **Linking** writes the chosen links into the graph in both directions,
//!    pruning any neighbor list that grows past its limit. Lists are pruned
//!    independently of each other, so this runs in parallel too.
//!
//! Neither step depends on how many threads run it, so a build is
//! deterministic. A single insert is a chunk of one node, which makes it
//! exactly the sequential algorithm from the paper.

use std::num::NonZeroUsize;
use std::ops::Range;

use super::Hnsw;
use super::search::Neighbor;
use super::visited::VisitedSet;
use crate::parallel::parallel_map;

/// The most nodes planned against the same graph. Larger chunks leave more
/// work to run in parallel between synchronization points, but each node is
/// measured directly against more of its chunk's earlier nodes.
const MAX_CHUNK: usize = 256;

/// The neighbors chosen for one new node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LinkPlan {
    /// The new node.
    pub node: u32,
    /// `layers[l]` holds the nodes `node` links to on layer `l`.
    pub layers: Vec<Vec<u32>>,
}

impl LinkPlan {
    /// The highest layer the node lives on.
    fn level(&self) -> usize {
        self.layers.len() - 1
    }
}

/// The new links pointing at one node on one layer.
#[derive(Debug)]
struct BackLinks {
    level: usize,
    target: u32,
    /// The new nodes linking to `target`, in ascending order.
    sources: Vec<u32>,
}

impl Hnsw {
    /// Links `nodes`, which must already be stored (see `Hnsw::push`) and be
    /// the most recent ones, into the graph on up to `threads` threads.
    pub(super) fn link_nodes(&mut self, nodes: Range<usize>, threads: NonZeroUsize) {
        let mut start = nodes.start;
        while start < nodes.end {
            let end = nodes.end.min(start + chunk_len(start));
            self.link_chunk(start..end, threads);
            start = end;
        }
    }

    /// Plans every node in `chunk` against the current graph, then links them all.
    fn link_chunk(&mut self, chunk: Range<usize>, threads: NonZeroUsize) {
        // Node ids fit in `u32`: `Hnsw::insert_batch` checks that before storing them.
        let first = chunk.start as u32;
        let plans = parallel_map(chunk.len(), threads, |i| {
            let node = first + i as u32;
            let mut visited = self.take_visited();
            let plan = self.plan_links(node, first..node, &mut visited);
            self.return_visited(visited);
            plan
        });
        self.link(&plans, threads);
    }

    /// Chooses `node`'s neighbors on every layer it lives on.
    ///
    /// `node` must already be stored, but not linked yet. The graph is
    /// searched from the current entry point. `peers` are stored nodes that
    /// are not linked yet either, so no search can reach them; they are
    /// measured directly instead.
    fn plan_links(&self, node: u32, peers: Range<u32>, visited: &mut VisitedSet) -> LinkPlan {
        let ctx = self.ctx();
        let query = ctx.node_query(node);
        let level = self.graph.level(node);
        let ef = self.params.ef_construction;

        // Phase 1: above the new node's top layer, just descend greedily.
        let mut ep = self.entry_point.map(|entry| {
            let start = ctx.neighbor(&query, entry);
            (level + 1..=self.max_level)
                .rev()
                .fold(start, |ep, l| ctx.greedy_closest(&query, ep, l, visited))
        });

        // Phase 2: on every layer the node lives on, choose its neighbors.
        let mut layers = vec![Vec::new(); level + 1];
        for l in (0..=level).rev() {
            let mut candidates = match ep {
                Some(entry) if l <= self.max_level => {
                    let found = ctx.search_layer(&query, &[entry], ef, l, visited);
                    ep = Some(found[0]);
                    found
                }
                _ => Vec::new(),
            };
            if !peers.is_empty() {
                candidates.extend(
                    peers
                        .clone()
                        .filter(|&peer| self.graph.level(peer) >= l)
                        .map(|peer| ctx.neighbor(&query, peer)),
                );
                // Same order as a search's results: by distance, then id.
                candidates
                    .sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.node.cmp(&b.node)));
                candidates.truncate(ef);
            }
            layers[l] = node_ids(&ctx.select_neighbors(&candidates, self.params.m));
        }
        LinkPlan { node, layers }
    }

    /// Writes planned links into the graph in both directions, and promotes a
    /// new node to entry point if it lives above the current top layer.
    fn link(&mut self, plans: &[LinkPlan], threads: NonZeroUsize) {
        for plan in plans {
            for (level, links) in plan.layers.iter().enumerate() {
                self.graph.set_neighbors(plan.node, level, links);
            }
        }

        let back_links = group_back_links(plans);
        let updated = parallel_map(back_links.len(), threads, |i| {
            self.with_back_links(&back_links[i])
        });
        for (b, links) in back_links.iter().zip(&updated) {
            self.graph.set_neighbors(b.target, b.level, links);
        }

        for plan in plans {
            if self.entry_point.is_none() || plan.level() > self.max_level {
                self.entry_point = Some(plan.node);
                self.max_level = plan.level();
            }
        }
    }

    /// `b.target`'s neighbor list on `b.level` once `b.sources` are added.
    ///
    /// If that is more than the layer allows, the list is re-selected with
    /// the same heuristic as a new node's, using distances measured from
    /// *the target*.
    fn with_back_links(&self, b: &BackLinks) -> Vec<u32> {
        let existing = self.graph.neighbors(b.target, b.level);
        let merged = existing.iter().chain(&b.sources).copied();
        let max_links = self.params.max_links(b.level);
        if existing.len() + b.sources.len() <= max_links {
            return merged.collect();
        }
        let ctx = self.ctx();
        let base = ctx.node_query(b.target);
        let mut candidates: Vec<Neighbor> = merged.map(|n| ctx.neighbor(&base, n)).collect();
        candidates.sort_by(|x, y| x.distance.total_cmp(&y.distance));
        node_ids(&ctx.select_neighbors(&candidates, max_links))
    }
}

/// Inverts the planned links: for every node a plan links to on some layer,
/// which new nodes link to it there. Sorted by layer, then target.
fn group_back_links(plans: &[LinkPlan]) -> Vec<BackLinks> {
    let mut edges: Vec<(usize, u32, u32)> = plans
        .iter()
        .flat_map(|plan| {
            plan.layers
                .iter()
                .enumerate()
                .flat_map(move |(level, links)| {
                    links.iter().map(move |&target| (level, target, plan.node))
                })
        })
        .collect();
    edges.sort_unstable();

    let mut groups: Vec<BackLinks> = Vec::new();
    for (level, target, source) in edges {
        match groups.last_mut() {
            Some(g) if g.level == level && g.target == target => g.sources.push(source),
            _ => groups.push(BackLinks {
                level,
                target,
                sources: vec![source],
            }),
        }
    }
    groups
}

/// How many nodes to plan together once `linked` nodes are in the graph.
///
/// Chunks grow with the graph, up to [`MAX_CHUNK`]: while the graph is
/// small, a large chunk would mostly be linked by direct comparison instead
/// of by searching. The schedule depends only on the graph size, never on
/// the thread count.
fn chunk_len(linked: usize) -> usize {
    linked.clamp(1, MAX_CHUNK)
}

fn node_ids(neighbors: &[Neighbor]) -> Vec<u32> {
    neighbors.iter().map(|n| n.node).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_double_until_the_cap() {
        let mut linked = 0;
        let mut lens = Vec::new();
        while linked < 1_000 {
            let len = chunk_len(linked);
            lens.push(len);
            linked += len;
        }
        assert_eq!(lens, [1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 256, 256]);
    }

    #[test]
    fn back_links_are_grouped_by_layer_and_target() {
        let plans = [
            LinkPlan {
                node: 5,
                layers: vec![vec![1, 2], vec![1]],
            },
            LinkPlan {
                node: 6,
                layers: vec![vec![2, 5]],
            },
        ];
        let groups: Vec<(usize, u32, Vec<u32>)> = group_back_links(&plans)
            .into_iter()
            .map(|b| (b.level, b.target, b.sources))
            .collect();
        assert_eq!(
            groups,
            vec![
                (0, 1, vec![5]),
                (0, 2, vec![5, 6]),
                (0, 5, vec![6]),
                (1, 1, vec![5]),
            ]
        );
    }
}
