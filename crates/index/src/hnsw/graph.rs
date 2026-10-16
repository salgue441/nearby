//! The layered neighbor lists.

/// Adjacency lists for every node on every layer it lives on.
///
/// Layer 0 holds every node and is where searches spend most of their time,
/// so it uses one flat array with `m0` fixed slots per node: node `n`'s
/// neighbors are `layer0[n * m0..n * m0 + layer0_len[n]]`. Upper layers hold
/// few nodes, so plain per-node `Vec`s are fine there.
#[derive(Debug, Clone)]
pub(super) struct Graph {
    m0: usize,
    layer0: Vec<u32>,
    layer0_len: Vec<u16>,
    /// `upper[n][l - 1]` = node `n`'s neighbors on layer `l` (for `l >= 1`).
    upper: Vec<Vec<Vec<u32>>>,
}

impl Graph {
    /// Creates an empty graph with room for `capacity` nodes, each with up
    /// to `m0` links on layer 0.
    pub(super) fn with_capacity(m0: usize, capacity: usize) -> Self {
        Self {
            m0,
            layer0: Vec::with_capacity(capacity * m0),
            layer0_len: Vec::with_capacity(capacity),
            upper: Vec::with_capacity(capacity),
        }
    }

    /// Adds a node with no links that lives on layers `0..=level`.
    /// Its id is the number of nodes added before it.
    pub(super) fn add_node(&mut self, level: usize) {
        self.layer0.resize(self.layer0.len() + self.m0, 0);
        self.layer0_len.push(0);
        self.upper.push(vec![Vec::new(); level]);
    }

    /// The highest layer `node` lives on.
    pub(super) fn level(&self, node: u32) -> usize {
        self.upper[node as usize].len()
    }

    /// `node`'s neighbors on `level`. `node` must live on that layer.
    #[inline]
    pub(super) fn neighbors(&self, node: u32, level: usize) -> &[u32] {
        debug_assert!(
            level <= self.level(node),
            "node {node} is not on layer {level}"
        );
        let n = node as usize;
        if level == 0 {
            let start = n * self.m0;
            &self.layer0[start..start + usize::from(self.layer0_len[n])]
        } else {
            &self.upper[n][level - 1]
        }
    }

    /// Replaces `node`'s neighbor list on `level`.
    ///
    /// Callers keep layer-0 lists within `m0` (that's what pruning is for).
    pub(super) fn set_neighbors(&mut self, node: u32, level: usize, neighbors: &[u32]) {
        let n = node as usize;
        if level == 0 {
            assert!(neighbors.len() <= self.m0, "layer-0 list exceeds m0");
            let start = n * self.m0;
            self.layer0[start..start + neighbors.len()].copy_from_slice(neighbors);
            self.layer0_len[n] = neighbors.len() as u16;
        } else {
            let list = &mut self.upper[n][level - 1];
            list.clear();
            list.extend_from_slice(neighbors);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_nodes_have_no_links() {
        let mut g = Graph::with_capacity(4, 0);
        g.add_node(2);
        assert_eq!(g.level(0), 2);
        assert_eq!(g.neighbors(0, 0), &[] as &[u32]);
        assert_eq!(g.neighbors(0, 2), &[] as &[u32]);
    }

    #[test]
    fn set_and_replace_neighbors() {
        let mut g = Graph::with_capacity(4, 0);
        for _ in 0..5 {
            g.add_node(1);
        }
        g.set_neighbors(0, 0, &[1, 2, 3]);
        g.set_neighbors(0, 1, &[4]);
        assert_eq!(g.neighbors(0, 0), &[1, 2, 3]);
        assert_eq!(g.neighbors(0, 1), &[4]);

        g.set_neighbors(0, 0, &[4]);
        assert_eq!(g.neighbors(0, 0), &[4], "shorter list replaces the old one");
        assert!(g.neighbors(1, 0).is_empty(), "other nodes are untouched");
    }

    #[test]
    #[should_panic(expected = "exceeds m0")]
    fn layer0_list_is_capped() {
        let mut g = Graph::with_capacity(2, 0);
        g.add_node(0);
        g.set_neighbors(0, 0, &[1, 2, 3]);
    }
}
