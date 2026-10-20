//! Linking new nodes into the graph (Algorithm 1 in the paper).
//!
//! An insert happens in two steps:
//!
//! 1. **Planning** searches the graph for the new node's neighbors on every
//!    layer it lives on. It only reads the index.
//! 2. **Linking** writes the chosen links into the graph in both directions,
//!    pruning any neighbor list that grows past its limit.

use super::Hnsw;
use super::search::Neighbor;
use super::visited::VisitedSet;

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
    /// Chooses `node`'s neighbors on every layer it lives on.
    ///
    /// `node` must already be stored (see `Hnsw::push`), but not linked yet.
    pub(super) fn plan_links(&self, node: u32, visited: &mut VisitedSet) -> LinkPlan {
        let ctx = self.ctx();
        let query = ctx.node_query(node);
        let level = self.graph.level(node);
        let mut layers = vec![Vec::new(); level + 1];
        let Some(entry) = self.entry_point else {
            // The first node has nothing to link to.
            return LinkPlan { node, layers };
        };

        // Phase 1: above the new node's top layer, just descend greedily.
        let mut ep = ctx.neighbor(&query, entry);
        for l in (level + 1..=self.max_level).rev() {
            ep = ctx.greedy_closest(&query, ep, l, visited);
        }

        // Phase 2: on every layer the node shares with the graph, choose neighbors.
        for l in (0..=level.min(self.max_level)).rev() {
            let candidates =
                ctx.search_layer(&query, &[ep], self.params.ef_construction, l, visited);
            ep = candidates[0];
            layers[l] = node_ids(&ctx.select_neighbors(&candidates, self.params.m));
        }
        LinkPlan { node, layers }
    }

    /// Writes planned links into the graph in both directions, and promotes a
    /// new node to entry point if it lives above the current top layer.
    pub(super) fn link(&mut self, plans: &[LinkPlan]) {
        for plan in plans {
            for (level, links) in plan.layers.iter().enumerate() {
                self.graph.set_neighbors(plan.node, level, links);
            }
        }

        let back_links = group_back_links(plans);
        let updated: Vec<Vec<u32>> = back_links.iter().map(|b| self.with_back_links(b)).collect();
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

fn node_ids(neighbors: &[Neighbor]) -> Vec<u32> {
    neighbors.iter().map(|n| n.node).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
