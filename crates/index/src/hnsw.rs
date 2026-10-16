//! HNSW: Hierarchical Navigable Small World graphs.
//!
//! Malkov & Yashunin, "Efficient and robust approximate nearest neighbor
//! search using Hierarchical Navigable Small World graphs" (2016).
//!
//! Every node links to its nearest neighbors on layer 0. A random few also
//! live on higher, sparser layers that act like express lanes. A search
//! starts at the top, moves greedily toward the query, drops a layer whenever
//! it can't get closer, and finishes with a wider search on layer 0.

mod graph;
mod params;
mod rng;
mod search;
mod visited;

use std::sync::{Mutex, PoisonError};

use nearby_core::{Error, Metric, PointId, Result, ScoredPoint, validate_vector};

use self::graph::Graph;
pub use self::params::HnswParams;
use self::rng::SplitMix64;
use self::search::{Neighbor, SearchCtx};
use self::visited::VisitedSet;
use crate::VectorIndex;
use crate::store::VectorStore;

/// Highest layer a node can be assigned. With m = 16, reaching it naturally
/// would take ~16^16 nodes; the cap only guards against extreme random draws.
const MAX_LEVEL: usize = 16;

/// An approximate nearest-neighbor index over a layered proximity graph.
///
/// Searches cost roughly O(log N) distance computations instead of
/// [`FlatIndex`](crate::FlatIndex)'s O(N), in exchange for recall below 1.0.
/// Recall is tuned with [`HnswParams`]: `m` and `ef_construction` at build
/// time, `ef_search` at query time (see [`Hnsw::set_ef_search`]).
///
/// Builds are deterministic: the same vectors inserted in the same order with
/// the same [`HnswParams::seed`] always produce the same graph.
///
/// Internally each vector gets a dense `u32` node id (its insertion position),
/// which keeps neighbor lists compact; [`ScoredPoint::id`] reports the
/// caller's [`PointId`] instead.
///
/// # Examples
///
/// ```
/// use nearby_core::Metric;
/// use nearby_index::{Hnsw, HnswParams, VectorIndex};
///
/// let mut index = Hnsw::new(2, Metric::L2, HnswParams::default())?;
/// index.insert(1, &[0.0, 0.0])?;
/// index.insert(2, &[5.0, 5.0])?;
///
/// let results = index.search(&[1.0, 1.0], 1)?;
/// assert_eq!(results[0].id, 1);
/// # Ok::<(), nearby_core::Error>(())
/// ```
#[derive(Debug)]
pub struct Hnsw {
    params: HnswParams,
    metric: Metric,
    store: VectorStore,
    /// `ids[node]` is the external id of internal node `node`.
    ids: Vec<PointId>,
    graph: Graph,
    /// A node on the top layer, where every search starts.
    entry_point: Option<u32>,
    max_level: usize,
    rng: SplitMix64,
    /// Reusable visited sets, so searches don't allocate. Each search takes
    /// one and puts it back, which lets many threads search at once.
    visited_pool: Mutex<Vec<VisitedSet>>,
}

impl Hnsw {
    /// Creates an empty index.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if `dim` is 0 or `params` are invalid.
    pub fn new(dim: usize, metric: Metric, params: HnswParams) -> Result<Self> {
        Self::with_capacity(dim, metric, params, 0)
    }

    /// Creates an empty index with room for `capacity` vectors.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if `dim` is 0 or `params` are invalid.
    pub fn with_capacity(
        dim: usize,
        metric: Metric,
        params: HnswParams,
        capacity: usize,
    ) -> Result<Self> {
        if dim == 0 {
            return Err(Error::InvalidParameter(
                "dimension must be at least 1".to_owned(),
            ));
        }
        params.validate()?;
        Ok(Self {
            store: VectorStore::with_capacity(dim, capacity),
            ids: Vec::with_capacity(capacity),
            graph: Graph::with_capacity(params.m0, capacity),
            entry_point: None,
            max_level: 0,
            rng: SplitMix64::new(params.seed),
            visited_pool: Mutex::new(Vec::new()),
            metric,
            params,
        })
    }

    /// The parameters this index was built with. `ef_search` reflects the
    /// latest [`Hnsw::set_ef_search`].
    pub fn params(&self) -> &HnswParams {
        &self.params
    }

    /// The metric distances are measured with.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// The top layer of the graph (0 when every node is on layer 0 only).
    pub fn max_level(&self) -> usize {
        self.max_level
    }

    /// Changes the recall/speed trade-off for later searches. No rebuild needed.
    ///
    /// Larger values explore more of the graph: higher recall, slower
    /// searches. A search always uses at least its `k`, and 0 is raised to 1.
    pub fn set_ef_search(&mut self, ef_search: usize) {
        self.params.ef_search = ef_search.max(1);
    }

    /// Adds one vector and links it into the graph (Algorithm 1 in the paper).
    ///
    /// Duplicate ids are not detected here; the engine's id map handles that later.
    ///
    /// Costs roughly `ef_construction * log N` distance computations. Vectors
    /// can be inserted at any time, including between searches.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if `vector` has the wrong length,
    /// [`Error::InvalidVector`] if it contains NaN or infinite values, or
    /// [`Error::InvalidParameter`] if the index already holds 2^32 vectors.
    /// On error the index is left unchanged.
    pub fn insert(&mut self, id: PointId, vector: &[f32]) -> Result<()> {
        validate_vector(vector, self.store.dim())?;
        let node = u32::try_from(self.ids.len()).map_err(|_| {
            Error::InvalidParameter("an index holds at most 2^32 vectors".to_owned())
        })?;
        let level = self.random_level();
        self.store.push(vector);
        self.ids.push(id);
        self.graph.add_node(level);

        let Some(entry) = self.entry_point else {
            // First node: it becomes the entry point, and there's nothing to link.
            self.entry_point = Some(node);
            self.max_level = level;
            return Ok(());
        };

        let mut visited = self.take_visited();
        let ctx = self.ctx();

        // Phase 1: above the new node's top layer, just descend greedily.
        let mut ep = ctx.neighbor(vector, entry);
        for l in (level + 1..=self.max_level).rev() {
            ep = ctx.greedy_closest(vector, ep, l, &mut visited);
        }

        // Phase 2: on every layer the node lives on, find neighbors and link them.
        for l in (0..=level.min(self.max_level)).rev() {
            let ctx = self.ctx();
            let candidates =
                ctx.search_layer(vector, &[ep], self.params.ef_construction, l, &mut visited);
            let selected = ctx.select_neighbors(&candidates, self.params.m);
            ep = candidates[0];
            self.connect(node, l, &selected);
        }
        self.return_visited(visited);

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(node);
        }
        Ok(())
    }

    /// Links `node` to `selected` on `level` in both directions.
    ///
    /// If a neighbor ends up with too many links, its list is re-selected with
    /// the same heuristic, using distances measured from *that neighbor*.
    fn connect(&mut self, node: u32, level: usize, selected: &[Neighbor]) {
        let links: Vec<u32> = selected.iter().map(|n| n.node).collect();
        self.graph.set_neighbors(node, level, &links);

        let max_links = self.params.max_links(level);
        for neighbor in selected {
            let mut links = self.graph.neighbors(neighbor.node, level).to_vec();
            if links.len() < max_links {
                links.push(node);
            } else {
                let ctx = self.ctx();
                let base = self.store.get(neighbor.node as usize);
                let mut candidates: Vec<Neighbor> = links
                    .iter()
                    .copied()
                    .chain([node])
                    .map(|n| ctx.neighbor(base, n))
                    .collect();
                candidates.sort_by(|a, b| a.distance.total_cmp(&b.distance));
                links = ctx
                    .select_neighbors(&candidates, max_links)
                    .iter()
                    .map(|n| n.node)
                    .collect();
            }
            self.graph.set_neighbors(neighbor.node, level, &links);
        }
    }

    /// Draws a node's top layer: `floor(-ln(U) * mL)` for uniform `U` in (0, 1].
    /// Most nodes get 0; each layer up is about `m` times rarer.
    fn random_level(&mut self) -> usize {
        let u = self.rng.next_f64();
        #[expect(
            clippy::cast_sign_loss,
            reason = "-ln(u) >= 0 for u in (0, 1], so the value is never negative"
        )]
        let level = (-u.ln() * self.params.level_multiplier()).floor() as usize;
        level.min(MAX_LEVEL)
    }

    fn ctx(&self) -> SearchCtx<'_> {
        SearchCtx {
            store: &self.store,
            graph: &self.graph,
            metric: self.metric,
        }
    }

    fn take_visited(&self) -> VisitedSet {
        self.visited_pool
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop()
            .unwrap_or_default()
    }

    fn return_visited(&self, visited: VisitedSet) {
        self.visited_pool
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(visited);
    }
}

impl VectorIndex for Hnsw {
    fn dim(&self) -> usize {
        self.store.dim()
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<ScoredPoint>> {
        validate_vector(query, self.store.dim())?;
        let Some(entry) = self.entry_point else {
            return Ok(Vec::new());
        };
        if k == 0 {
            return Ok(Vec::new());
        }

        let mut visited = self.take_visited();
        let ctx = self.ctx();
        let mut ep = ctx.neighbor(query, entry);
        for l in (1..=self.max_level).rev() {
            ep = ctx.greedy_closest(query, ep, l, &mut visited);
        }
        let ef = self.params.ef_search.max(k);
        let found = ctx.search_layer(query, &[ep], ef, 0, &mut visited);
        self.return_visited(visited);

        Ok(found
            .into_iter()
            .take(k)
            .map(|n| ScoredPoint {
                id: self.ids[n.node as usize],
                distance: n.distance,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_index() -> Hnsw {
        // A 10 x 10 grid; id = y * 10 + x.
        let mut index = Hnsw::new(2, Metric::L2, HnswParams::with_m(4)).unwrap();
        for y in 0..10u8 {
            for x in 0..10u8 {
                let id = u64::from(y) * 10 + u64::from(x);
                index.insert(id, &[f32::from(x), f32::from(y)]).unwrap();
            }
        }
        index
    }

    #[test]
    fn empty_index_returns_nothing() {
        let index = Hnsw::new(3, Metric::L2, HnswParams::default()).unwrap();
        assert!(index.is_empty());
        assert_eq!(index.search(&[1.0, 2.0, 3.0], 5).unwrap(), Vec::new());
    }

    #[test]
    fn single_point() {
        let mut index = Hnsw::new(2, Metric::L2, HnswParams::default()).unwrap();
        index.insert(42, &[1.0, 1.0]).unwrap();
        let results = index.search(&[0.0, 0.0], 3).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 42);
    }

    #[test]
    fn finds_exact_matches_on_a_grid() {
        let index = grid_index();
        assert_eq!(index.len(), 100);
        for id in [0u8, 7, 55, 99] {
            let (x, y) = (f32::from(id % 10), f32::from(id / 10));
            let results = index.search(&[x, y], 1).unwrap();
            assert_eq!(results[0].id, u64::from(id));
        }
    }

    #[test]
    fn results_are_sorted_and_sized() {
        let index = grid_index();
        let results = index.search(&[4.4, 4.4], 10).unwrap();
        assert_eq!(results.len(), 10);
        assert!(results.windows(2).all(|w| w[0].distance <= w[1].distance));
        assert_eq!(results[0].id, 44);
    }

    #[test]
    fn same_seed_builds_the_same_graph() {
        let a = grid_index();
        let b = grid_index();
        assert_eq!(a.max_level(), b.max_level());
        assert_eq!(a.entry_point, b.entry_point);
        assert_eq!(
            a.search(&[3.3, 6.6], 5).unwrap(),
            b.search(&[3.3, 6.6], 5).unwrap()
        );
    }

    #[test]
    fn rejects_bad_input() {
        assert!(matches!(
            Hnsw::new(0, Metric::L2, HnswParams::default()),
            Err(Error::InvalidParameter(_))
        ));
        assert!(matches!(
            Hnsw::new(2, Metric::L2, HnswParams::with_m(1)),
            Err(Error::InvalidParameter(_))
        ));
        let mut index = Hnsw::new(2, Metric::L2, HnswParams::default()).unwrap();
        assert!(matches!(
            index.insert(1, &[1.0]),
            Err(Error::DimensionMismatch { .. })
        ));
        assert!(index.is_empty(), "a rejected vector must not be stored");
    }

    #[test]
    fn layer0_lists_never_exceed_m0() {
        let index = grid_index();
        for node in 0..100 {
            assert!(index.graph.neighbors(node, 0).len() <= index.params.m0);
        }
    }

    /// `n` deterministic pseudo-random vectors in `[0, 1)^dim`.
    fn random_index(n: usize, dim: usize, params: HnswParams) -> Hnsw {
        let mut rng = SplitMix64::new(99);
        let mut index = Hnsw::new(dim, Metric::L2, params).unwrap();
        for id in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.next_f64() as f32).collect();
            index.insert(id as PointId, &v).unwrap();
        }
        index
    }

    /// Checks every structural invariant the search relies on.
    fn assert_graph_invariants(index: &Hnsw) {
        let n = index.len() as u32;
        let mut top = 0;
        for node in 0..n {
            let level = index.graph.level(node);
            top = top.max(level);
            for l in 0..=level {
                let links = index.graph.neighbors(node, l);
                assert!(
                    links.len() <= index.params.max_links(l),
                    "node {node} has {} links on layer {l}",
                    links.len()
                );
                let mut seen = std::collections::HashSet::new();
                for &next in links {
                    assert_ne!(next, node, "node {node} links to itself on layer {l}");
                    assert!(next < n, "node {node} links to missing node {next}");
                    assert!(seen.insert(next), "node {node} links to {next} twice");
                    assert!(
                        index.graph.level(next) >= l,
                        "node {node} links to {next}, which is not on layer {l}"
                    );
                }
            }
        }
        assert_eq!(index.max_level, top, "max_level is the highest node level");
        let entry = index
            .entry_point
            .expect("non-empty index has an entry point");
        assert_eq!(
            index.graph.level(entry),
            top,
            "entry point is on the top layer"
        );
    }

    #[test]
    fn graph_invariants_hold_after_many_inserts() {
        let index = random_index(1_000, 8, HnswParams::with_m(6));
        assert!(
            index.max_level() > 0,
            "1000 nodes with m = 6 use upper layers"
        );
        assert_graph_invariants(&index);
    }

    #[test]
    fn every_node_is_reachable_on_layer_0() {
        // Search only works if layer 0 is connected from the entry point.
        let index = random_index(1_000, 8, HnswParams::with_m(6));
        let mut seen = vec![false; index.len()];
        let mut stack = vec![index.entry_point.unwrap()];
        seen[stack[0] as usize] = true;
        while let Some(node) = stack.pop() {
            for &next in index.graph.neighbors(node, 0) {
                if !std::mem::replace(&mut seen[next as usize], true) {
                    stack.push(next);
                }
            }
        }
        let unreachable = seen.iter().filter(|&&s| !s).count();
        assert_eq!(unreachable, 0, "{unreachable} nodes unreachable on layer 0");
    }

    #[test]
    fn first_insert_becomes_the_entry_point() {
        let mut index = Hnsw::new(1, Metric::L2, HnswParams::default()).unwrap();
        assert_eq!(index.entry_point, None);
        index.insert(7, &[1.0]).unwrap();
        assert_eq!(index.entry_point, Some(0));
        assert_eq!(index.max_level, index.graph.level(0));
    }

    #[test]
    fn k_zero_and_k_beyond_len() {
        let index = grid_index();
        assert!(index.search(&[1.0, 1.0], 0).unwrap().is_empty(), "k = 0");
        let all = index.search(&[1.0, 1.0], 1_000).unwrap();
        assert_eq!(all.len(), 100, "a connected graph reaches every node");
    }

    #[test]
    fn set_ef_search_updates_params_and_clamps_zero() {
        let mut index = grid_index();
        index.set_ef_search(123);
        assert_eq!(index.params().ef_search, 123);
        index.set_ef_search(0);
        assert_eq!(index.params().ef_search, 1);
    }

    #[test]
    fn accessors_report_construction_values() {
        let index = Hnsw::new(3, Metric::Cosine, HnswParams::with_m(8)).unwrap();
        assert_eq!(index.metric(), Metric::Cosine);
        assert_eq!(index.params(), &HnswParams::with_m(8));
        assert_eq!(index.dim(), 3);
        assert_eq!(index.max_level(), 0);
    }

    #[test]
    fn rejected_vectors_leave_the_index_unchanged() {
        let mut index = grid_index();
        let before = (index.len(), index.entry_point, index.max_level);
        assert!(matches!(
            index.insert(1_000, &[f32::NAN, 0.0]),
            Err(Error::InvalidVector)
        ));
        assert_eq!((index.len(), index.entry_point, index.max_level), before);
        assert_graph_invariants(&index);
    }

    #[test]
    fn searches_return_their_visited_set_to_the_pool() {
        let index = grid_index();
        index.search(&[1.0, 1.0], 5).unwrap();
        index.search(&[2.0, 2.0], 5).unwrap();
        let pooled = index.visited_pool.lock().unwrap().len();
        assert_eq!(pooled, 1, "sequential searches reuse one visited set");
    }

    #[test]
    fn random_levels_follow_the_expected_distribution() {
        // P(level >= 1) = 1/m. With m = 16 and 20k draws, expect ~1250.
        let mut index = Hnsw::new(1, Metric::L2, HnswParams::with_m(16)).unwrap();
        let draws = 20_000;
        let upper = (0..draws).filter(|_| index.random_level() >= 1).count();
        let expected = draws / 16;
        assert!(
            upper.abs_diff(expected) < expected / 5,
            "{upper} of {draws} draws reached layer 1, expected about {expected}"
        );
    }
}
