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
mod insert;
mod params;
mod persist;
mod search;
mod visited;

use std::num::NonZeroUsize;
use std::sync::{Mutex, PoisonError};

use nearby_core::distance::norm;
use nearby_core::{Error, Metric, PointId, Result, ScoredPoint, validate_vector};

use self::graph::Graph;
pub use self::params::HnswParams;
use self::search::SearchCtx;
use self::visited::VisitedSet;
use crate::VectorIndex;
use crate::rng::SplitMix64;
use crate::store::VectorStore;

/// Highest layer a node can be assigned. With m = 16, reaching it naturally
/// would take ~16^16 nodes; the cap only guards against extreme random draws.
const MAX_LEVEL: usize = 16;

/// Node ids are `u32`, so an index holds at most 2^32 vectors.
const MAX_NODES: u64 = 1 << 32;

/// An approximate nearest-neighbor index over a layered proximity graph.
///
/// Searches cost roughly O(log N) distance computations instead of
/// [`FlatIndex`](crate::FlatIndex)'s O(N), in exchange for recall below 1.0.
/// Recall is tuned with [`HnswParams`]: `m` and `ef_construction` at build
/// time, `ef_search` at query time (see [`Hnsw::set_ef_search`]).
///
/// Builds are deterministic: the same vectors inserted in the same order with
/// the same [`HnswParams::seed`] always produce the same graph.
/// [`Hnsw::insert_batch`] builds on several threads, and its graph does not
/// depend on how many.
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
    /// `norms[node]` is the norm of vector `node` when the metric is
    /// [`Metric::Cosine`], so comparisons cost one dot product instead of
    /// three. Empty for other metrics.
    norms: Vec<f32>,
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
            norms: if metric == Metric::Cosine {
                Vec::with_capacity(capacity)
            } else {
                Vec::new()
            },
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

    /// Every point as `(id, vector)`, in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (PointId, &[f32])> {
        self.ids
            .iter()
            .enumerate()
            .map(|(node, &id)| (id, self.store.get(node)))
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
        self.insert_batch(&[(id, vector)], NonZeroUsize::MIN)
    }

    /// Adds many vectors at once, linking them into the graph on up to
    /// `threads` threads.
    ///
    /// On a multi-core machine this builds an index several times faster
    /// than calling [`Hnsw::insert`] in a loop, with about the same recall.
    /// Vectors are linked in chunks of up to a few hundred: every vector in a
    /// chunk searches the graph as it was before the chunk, all in parallel,
    /// and is also compared directly with the chunk's earlier vectors.
    ///
    /// The graph is deterministic: the same vectors in the same order, with
    /// the same [`HnswParams::seed`], always give the same graph whatever
    /// `threads` is. It is not the graph that inserting them one at a time
    /// would build.
    ///
    /// # Errors
    ///
    /// The same as [`Hnsw::insert`], for any of the vectors. On error
    /// nothing is inserted.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use std::thread;
    ///
    /// use nearby_core::Metric;
    /// use nearby_index::{Hnsw, HnswParams, VectorIndex};
    ///
    /// let points: Vec<(u64, [f32; 2])> = (0..1_000u16)
    ///     .map(|i| (u64::from(i), [f32::from(i % 40), f32::from(i / 40)]))
    ///     .collect();
    /// let batch: Vec<(u64, &[f32])> = points.iter().map(|(id, v)| (*id, &v[..])).collect();
    ///
    /// let threads = thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    /// let mut index = Hnsw::new(2, Metric::L2, HnswParams::default())?;
    /// index.insert_batch(&batch, threads)?;
    ///
    /// assert_eq!(index.len(), 1_000);
    /// assert_eq!(index.search(&[3.0, 2.0], 1)?[0].id, 83);
    /// # Ok::<(), nearby_core::Error>(())
    /// ```
    pub fn insert_batch(
        &mut self,
        points: &[(PointId, &[f32])],
        threads: NonZeroUsize,
    ) -> Result<()> {
        for (_, vector) in points {
            validate_vector(vector, self.store.dim())?;
        }
        let first = self.ids.len();
        if (first + points.len()) as u64 > MAX_NODES {
            return Err(Error::InvalidParameter(
                "an index holds at most 2^32 vectors".to_owned(),
            ));
        }
        for &(id, vector) in points {
            self.push(id, vector);
        }
        self.link_nodes(first..self.ids.len(), threads);
        Ok(())
    }

    /// Stores a validated vector as a new node, on a randomly drawn top
    /// layer but not linked to anything yet. Its node id is the previous
    /// `len()`, which the caller has checked fits in `u32`.
    fn push(&mut self, id: PointId, vector: &[f32]) {
        let level = self.random_level();
        self.store.push(vector);
        if self.metric == Metric::Cosine {
            self.norms.push(norm(vector));
        }
        self.ids.push(id);
        self.graph.add_node(level);
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
            norms: &self.norms,
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
        // Not the default, which goes through `search_filtered`: with a filter
        // known to keep everything, the check compiles away instead of costing
        // an extra memory read per visited node.
        self.search_admitting(query, k, |_| true)
    }

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        keep: &dyn Fn(PointId) -> bool,
    ) -> Result<Vec<ScoredPoint>> {
        self.search_admitting(query, k, |node| keep(self.ids[node as usize]))
    }
}

impl Hnsw {
    /// Searches for the `k` nodes closest to `query` among those `admit` accepts.
    fn search_admitting(
        &self,
        query: &[f32],
        k: usize,
        admit: impl Fn(u32) -> bool,
    ) -> Result<Vec<ScoredPoint>> {
        validate_vector(query, self.store.dim())?;
        let Some(entry) = self.entry_point else {
            return Ok(Vec::new());
        };
        if k == 0 {
            return Ok(Vec::new());
        }

        let mut visited = self.take_visited();
        let ctx = self.ctx();
        let query = ctx.query(query);
        let mut ep = ctx.neighbor(&query, entry);
        for l in (1..=self.max_level).rev() {
            ep = ctx.greedy_closest(&query, ep, l, &mut visited);
        }
        let ef = self.params.ef_search.max(k);
        let found = ctx.search_layer_filtered(&query, &[ep], ef, 0, &mut visited, admit);
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

    /// Like [`random_index`], but built with [`Hnsw::insert_batch`].
    fn random_batch_index(n: usize, dim: usize, params: HnswParams, threads: usize) -> Hnsw {
        let mut rng = SplitMix64::new(99);
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.next_f64() as f32).collect())
            .collect();
        let batch: Vec<(PointId, &[f32])> = vectors
            .iter()
            .enumerate()
            .map(|(id, v)| (id as PointId, v.as_slice()))
            .collect();
        let mut index = Hnsw::new(dim, Metric::L2, params).unwrap();
        index
            .insert_batch(&batch, NonZeroUsize::new(threads).unwrap())
            .unwrap();
        index
    }

    /// Every node's links on every layer it lives on.
    fn all_links(index: &Hnsw) -> Vec<Vec<u32>> {
        (0..index.len() as u32)
            .flat_map(|node| {
                (0..=index.graph.level(node)).map(move |l| index.graph.neighbors(node, l).to_vec())
            })
            .collect()
    }

    /// Nodes that can't be reached on layer 0 from the entry point.
    fn unreachable_on_layer0(index: &Hnsw) -> usize {
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
        seen.iter().filter(|&&s| !s).count()
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
        let unreachable = unreachable_on_layer0(&index);
        assert_eq!(unreachable, 0, "{unreachable} nodes unreachable on layer 0");
    }

    #[test]
    fn batch_insert_keeps_the_graph_valid_and_connected() {
        // 1000 nodes span every chunk size, up to several full chunks.
        let index = random_batch_index(1_000, 8, HnswParams::with_m(6), 4);
        assert_eq!(index.len(), 1_000);
        assert!(index.max_level() > 0, "upper layers are used");
        assert_graph_invariants(&index);
        let unreachable = unreachable_on_layer0(&index);
        assert_eq!(unreachable, 0, "{unreachable} nodes unreachable on layer 0");
    }

    #[test]
    fn batch_insert_builds_the_same_graph_with_any_thread_count() {
        let params = HnswParams::with_m(6);
        let one = random_batch_index(1_000, 8, params.clone(), 1);
        for threads in [2, 5, 16] {
            let many = random_batch_index(1_000, 8, params.clone(), threads);
            assert_eq!(all_links(&many), all_links(&one), "{threads} threads");
            assert_eq!(
                (many.entry_point, many.max_level),
                (one.entry_point, one.max_level),
                "{threads} threads"
            );
        }
    }

    #[test]
    fn batch_insert_draws_the_same_levels_as_single_inserts() {
        let params = HnswParams::with_m(6);
        let single = random_index(500, 8, params.clone());
        let batch = random_batch_index(500, 8, params, 4);
        let levels = |index: &Hnsw| -> Vec<usize> {
            (0..index.len() as u32)
                .map(|n| index.graph.level(n))
                .collect()
        };
        assert_eq!(levels(&batch), levels(&single));
    }

    #[test]
    fn batch_of_one_is_a_single_insert() {
        let mut single = grid_index();
        let mut batch = grid_index();
        single.insert(100, &[4.5, 4.5]).unwrap();
        batch
            .insert_batch(&[(100, &[4.5, 4.5])], NonZeroUsize::new(8).unwrap())
            .unwrap();
        assert_eq!(all_links(&batch), all_links(&single));
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
    fn iter_yields_points_in_insertion_order() {
        let index = grid_index();
        let points: Vec<(PointId, Vec<f32>)> =
            index.iter().map(|(id, v)| (id, v.to_vec())).collect();
        assert_eq!(points.len(), 100);
        assert_eq!(points[0], (0, vec![0.0, 0.0]));
        assert_eq!(points[23], (23, vec![3.0, 2.0]));
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
    fn norms_are_stored_only_for_cosine() {
        let mut cosine = Hnsw::new(2, Metric::Cosine, HnswParams::default()).unwrap();
        let mut l2 = Hnsw::new(2, Metric::L2, HnswParams::default()).unwrap();
        for (id, v) in [[3.0, 4.0], [1.0, 0.0]].iter().enumerate() {
            cosine.insert(id as PointId, v).unwrap();
            l2.insert(id as PointId, v).unwrap();
        }
        assert_eq!(cosine.norms, vec![5.0, 1.0]);
        assert!(l2.norms.is_empty(), "L2 keeps no norms");
        assert!(
            cosine.insert(9, &[f32::NAN, 0.0]).is_err() && cosine.norms.len() == 2,
            "a rejected vector stores no norm"
        );
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
