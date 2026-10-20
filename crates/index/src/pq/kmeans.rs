//! k-means clustering (Lloyd's algorithm), deterministic for a given seed.

use super::slice_l2;

use crate::rng::SplitMix64;

/// Clusters `n` points of `dim` values each, stored back to back in
/// `points`, into `k` centroids, returned back to back in the same layout.
///
/// Starts from `k` points picked with k-means++ and `seed`, then
/// alternates between assigning every point to its nearest centroid and
/// moving every centroid to the mean of its points, for at most
/// `iterations` rounds or until no point changes cluster. A cluster that
/// ends up empty restarts at the point farthest from its own centroid.
///
/// The same inputs always give the same centroids.
pub(super) fn kmeans(
    points: &[f32],
    dim: usize,
    k: usize,
    iterations: usize,
    seed: u64,
) -> Vec<f32> {
    let n = points.len() / dim;
    debug_assert!(k >= 1 && n >= k, "need at least k = {k} points, got {n}");
    let mut centroids = initial_centroids(points, dim, k, seed);
    let mut assignments = vec![u32::MAX; n];
    let mut distances = vec![0.0; n];
    for _ in 0..iterations {
        if !assign(points, &centroids, dim, &mut assignments, &mut distances) {
            break;
        }
        update(points, dim, &assignments, &mut distances, &mut centroids);
    }
    centroids
}

/// Picks `k` starting centroids with k-means++ (Arthur & Vassilvitskii,
/// 2007): the first is a random point, and each next one is a point drawn
/// with probability proportional to its squared distance from the nearest
/// centroid picked so far. That spreads the starting centroids out, so
/// Lloyd's iterations rarely get stuck with two centroids in one cluster.
///
/// A point already picked is at distance 0, so it is never drawn again
/// while any point is farther away. Once every point is at distance 0
/// (fewer distinct points than `k`), the rest are drawn uniformly.
fn initial_centroids(points: &[f32], dim: usize, k: usize, seed: u64) -> Vec<f32> {
    let n = points.len() / dim;
    let mut rng = SplitMix64::new(seed);
    let first = point(points, dim, rng.next_below(n));
    let mut centroids = Vec::with_capacity(k * dim);
    centroids.extend_from_slice(first);
    let mut distances: Vec<f32> = points
        .chunks_exact(dim)
        .map(|p| slice_l2(p, first))
        .collect();

    while centroids.len() < k * dim {
        let picked = point(points, dim, weighted_pick(&distances, &mut rng));
        centroids.extend_from_slice(picked);
        for (d, p) in distances.iter_mut().zip(points.chunks_exact(dim)) {
            *d = d.min(slice_l2(p, picked));
        }
    }
    centroids
}

/// An index drawn with probability proportional to its weight, or uniformly
/// if every weight is 0.
fn weighted_pick(weights: &[f32], rng: &mut SplitMix64) -> usize {
    let total: f64 = weights.iter().map(|&w| f64::from(w)).sum();
    if total <= 0.0 {
        return rng.next_below(weights.len());
    }
    // `next_f64` is in (0, 1], so the target is in (0, total].
    let target = rng.next_f64() * total;
    let mut cumulative = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cumulative += f64::from(w);
        if w > 0.0 && cumulative >= target {
            return i;
        }
    }
    // Rounding can leave the target just above the final sum.
    weights.iter().rposition(|&w| w > 0.0).unwrap_or(0)
}

/// Assigns every point to its nearest centroid (the lowest index on ties)
/// and records the distance to it. Returns whether any assignment changed.
fn assign(
    points: &[f32],
    centroids: &[f32],
    dim: usize,
    assignments: &mut [u32],
    distances: &mut [f32],
) -> bool {
    let mut changed = false;
    for (i, p) in points.chunks_exact(dim).enumerate() {
        let (best, distance) = nearest(centroids, dim, p);
        changed |= assignments[i] != best;
        assignments[i] = best;
        distances[i] = distance;
    }
    changed
}

/// The index of the centroid nearest to `p` (the lowest on ties) and its
/// squared distance.
pub(super) fn nearest(centroids: &[f32], dim: usize, p: &[f32]) -> (u32, f32) {
    let mut best = (0, f32::INFINITY);
    for (c, centroid) in centroids.chunks_exact(dim).enumerate() {
        let d = slice_l2(p, centroid);
        if d < best.1 {
            best = (c as u32, d);
        }
    }
    best
}

/// Moves every centroid to the mean of its points. An empty cluster takes
/// the point farthest from its centroid instead, which is then no longer a
/// candidate for the next empty cluster.
fn update(
    points: &[f32],
    dim: usize,
    assignments: &[u32],
    distances: &mut [f32],
    centroids: &mut [f32],
) {
    let k = centroids.len() / dim;
    // f64 sums: adding up to millions of f32s would lose precision.
    let mut sums = vec![0.0f64; k * dim];
    let mut counts = vec![0usize; k];
    for (p, &c) in points.chunks_exact(dim).zip(assignments) {
        let c = c as usize;
        counts[c] += 1;
        for (sum, &x) in sums[c * dim..(c + 1) * dim].iter_mut().zip(p) {
            *sum += f64::from(x);
        }
    }

    for c in 0..k {
        let centroid = &mut centroids[c * dim..(c + 1) * dim];
        if counts[c] == 0 {
            let farthest = farthest_point(distances);
            distances[farthest] = 0.0;
            centroid.copy_from_slice(point(points, dim, farthest));
        } else {
            let count = counts[c] as f64;
            for (x, &sum) in centroid.iter_mut().zip(&sums[c * dim..(c + 1) * dim]) {
                *x = (sum / count) as f32;
            }
        }
    }
}

/// The index of the largest distance (the lowest index on ties).
fn farthest_point(distances: &[f32]) -> usize {
    let mut best = 0;
    for (i, &d) in distances.iter().enumerate() {
        if d > distances[best] {
            best = i;
        }
    }
    best
}

fn point(points: &[f32], dim: usize, i: usize) -> &[f32] {
    &points[i * dim..(i + 1) * dim]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sorts centroids so results can be compared without caring about order.
    fn sorted_1d(centroids: &[f32]) -> Vec<f32> {
        let mut c = centroids.to_vec();
        c.sort_by(f32::total_cmp);
        c
    }

    #[test]
    fn finds_well_separated_clusters() {
        // Three tight groups on a line, around 0, 10 and 20.
        let points: Vec<f32> = [0.0, 10.0, 20.0]
            .iter()
            .flat_map(|&c| [c - 0.5, c, c + 0.5])
            .collect();
        let centroids = kmeans(&points, 1, 3, 20, 1);
        assert_eq!(sorted_1d(&centroids), vec![0.0, 10.0, 20.0]);
    }

    #[test]
    fn same_seed_gives_the_same_centroids() {
        let mut rng = SplitMix64::new(5);
        let points: Vec<f32> = (0..2_000).map(|_| rng.next_f64() as f32).collect();
        let a = kmeans(&points, 4, 16, 10, 9);
        let b = kmeans(&points, 4, 16, 10, 9);
        assert_eq!(a, b);
        assert_ne!(a, kmeans(&points, 4, 16, 10, 10), "the seed matters");
    }

    #[test]
    fn weighted_pick_never_draws_zero_weights() {
        let mut rng = SplitMix64::new(4);
        let weights = [0.0, 3.0, 0.0, 1.0];
        let mut counts = [0; 4];
        for _ in 0..4_000 {
            counts[weighted_pick(&weights, &mut rng)] += 1;
        }
        assert_eq!((counts[0], counts[2]), (0, 0));
        assert!((2_700..3_300).contains(&counts[1]), "{counts:?}");
        assert!(weighted_pick(&[0.0; 3], &mut rng) < 3, "all zero: uniform");
    }

    #[test]
    fn initial_centroids_are_distinct_points() {
        let points: Vec<f32> = (0..50u8).map(f32::from).collect();
        let centroids = initial_centroids(&points, 1, 50, 3);
        assert_eq!(
            sorted_1d(&centroids),
            points,
            "a permutation of every point"
        );
    }

    #[test]
    fn empty_clusters_restart_at_the_farthest_point() {
        // Every point is assigned to centroid 0, so centroid 1 is empty and
        // must move to the point farthest from centroid 0.
        let points = [0.0, 1.0, 9.0];
        let assignments = [0, 0, 0];
        let mut distances = [0.0, 1.0, 81.0];
        let mut centroids = [0.0, 100.0];
        update(&points, 1, &assignments, &mut distances, &mut centroids);
        assert_eq!(centroids, [10.0 / 3.0, 9.0]);
        assert_eq!(
            distances[2], 0.0,
            "the point can't restart a second cluster"
        );
    }

    #[test]
    fn duplicate_points_still_fill_every_cluster() {
        // Only two distinct values for four clusters: no centroid may be left
        // where no point is.
        let points = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0];
        let centroids = kmeans(&points, 1, 4, 10, 2);
        assert!(
            centroids.iter().all(|c| [1.0, 5.0].contains(c)),
            "{centroids:?}"
        );
    }

    #[test]
    fn nearest_prefers_the_lowest_index_on_ties() {
        let centroids = [0.0, 0.0, 2.0, 2.0, 0.0, 0.0];
        // Centroids 0 and 2 are the same point.
        assert_eq!(nearest(&centroids, 2, &[0.1, 0.0]).0, 0);
        assert_eq!(nearest(&centroids, 2, &[2.0, 2.0]), (1, 0.0));
    }
}
