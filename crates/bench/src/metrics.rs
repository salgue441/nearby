//! Benchmark measurements: recall and latency.

use nearby_core::{PointId, ScoredPoint};
use std::time::Duration;

/// Fraction of the true `k` nearest neighbors that appear in the first `k` results.
///
/// `truth` is the ground-truth id list for one query, closest first.
/// 1.0 means every true neighbor was found.
pub fn recall_at_k(results: &[ScoredPoint], truth: &[u32], k: usize) -> f64 {
    let truth = &truth[..k.min(truth.len())];
    if truth.is_empty() {
        return 1.0;
    }

    let found = results
        .iter()
        .take(k)
        .filter(|p| truth.iter().any(|&t| PointId::from(t) == p.id))
        .count();

    found as f64 / truth.len() as f64
}

/// Summary of per-query latencies.
#[derive(Debug, Clone)]
pub struct LatencyStats {
    /// Number of samples.
    pub count: usize,
    /// Sum of all samples.
    pub total: Duration,
    /// Arithmetic mean.
    pub mean: Duration,
    /// Median (nearest-rank).
    pub p50: Duration,
    /// 95th percentile (nearest-rank).
    pub p95: Duration,
    /// 99th percentile (nearest-rank).
    pub p99: Duration,
    /// Slowest sample.
    pub max: Duration,
}

impl LatencyStats {
    /// Summarizes `samples`, in any order.
    ///
    /// Returns `None` if there are no samples.
    pub fn from_samples(mut samples: Vec<Duration>) -> Option<Self> {
        let max = *samples.iter().max()?;
        samples.sort_unstable();

        let count = samples.len();
        let total: Duration = samples.iter().sum();

        Some(Self {
            count,
            total,
            mean: total.div_f64(count as f64),
            p50: percentile(&samples, 50.0),
            p95: percentile(&samples, 95.0),
            p99: percentile(&samples, 99.0),
            max,
        })
    }

    /// Queries per second, for queries run one after another on one thread.
    ///
    /// Infinite if every sample was zero.
    pub fn qps(&self) -> f64 {
        self.count as f64 / self.total.as_secs_f64()
    }
}

/// Nearest-rank percentile of already-sorted, non-empty samples.
///
/// Tail latencies (p95, p99) matter more than the mean: they're what the
/// slowest users of a service actually experience.
fn percentile(sorted: &[Duration], p: f64) -> Duration {
    #[expect(
        clippy::cast_sign_loss,
        reason = "p is in [0, 100], so the rank is never negative"
    )]
    let rank = (p / 100.0 * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn points(ids: &[PointId]) -> Vec<ScoredPoint> {
        ids.iter()
            .map(|&id| ScoredPoint { id, distance: 0.0 })
            .collect()
    }

    #[test]
    fn perfect_recall() {
        let r = recall_at_k(&points(&[3, 1, 2]), &[1, 2, 3, 99], 3);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn partial_recall() {
        // Found 1 and 2, missed 3.
        let r = recall_at_k(&points(&[1, 2, 7, 8]), &[1, 2, 3, 4], 4);
        assert!((r - 0.5).abs() < 1e-9);
    }

    #[test]
    fn only_first_k_results_count() {
        let r = recall_at_k(&points(&[9, 1]), &[1, 2], 1);
        assert!(r.abs() < 1e-9, "id 1 is outside the first k = 1 results");
    }

    #[test]
    fn fewer_results_than_k() {
        let r = recall_at_k(&points(&[1]), &[1, 2, 3, 4], 4);
        assert!((r - 0.25).abs() < 1e-9);
    }

    #[test]
    fn truth_shorter_than_k_is_the_denominator() {
        let r = recall_at_k(&points(&[1, 2, 3]), &[1, 2], 3);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn empty_truth_counts_as_perfect() {
        assert!((recall_at_k(&points(&[1, 2]), &[], 2) - 1.0).abs() < 1e-9);
        assert!((recall_at_k(&points(&[1, 2]), &[1, 2], 0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn order_within_the_first_k_does_not_matter() {
        let r = recall_at_k(&points(&[4, 3, 2, 1]), &[1, 2, 3, 4], 4);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn samples_need_not_be_sorted() {
        let samples = [5, 1, 4, 2, 3].map(Duration::from_millis).to_vec();
        let stats = LatencyStats::from_samples(samples).unwrap();
        assert_eq!(stats.count, 5);
        assert_eq!(stats.total, Duration::from_millis(15));
        assert_eq!(stats.p50, Duration::from_millis(3));
        assert_eq!(stats.max, Duration::from_millis(5));
    }

    #[test]
    fn qps_is_count_over_total_time() {
        let samples = vec![Duration::from_millis(250); 8];
        let stats = LatencyStats::from_samples(samples).unwrap();
        assert!((stats.qps() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn percentile_bounds() {
        let sorted = [10, 20, 30].map(Duration::from_millis);
        assert_eq!(percentile(&sorted, 0.0), Duration::from_millis(10));
        assert_eq!(percentile(&sorted, 34.0), Duration::from_millis(20));
        assert_eq!(percentile(&sorted, 100.0), Duration::from_millis(30));
    }

    #[test]
    fn percentiles_use_nearest_rank() {
        let samples: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
        let stats = LatencyStats::from_samples(samples).unwrap();
        assert_eq!(stats.p50, Duration::from_millis(50));
        assert_eq!(stats.p95, Duration::from_millis(95));
        assert_eq!(stats.p99, Duration::from_millis(99));
        assert_eq!(stats.max, Duration::from_millis(100));
        assert_eq!(stats.mean, Duration::from_micros(50_500));
    }

    #[test]
    fn single_sample() {
        let stats = LatencyStats::from_samples(vec![Duration::from_millis(7)]).unwrap();
        assert_eq!(stats.p50, Duration::from_millis(7));
        assert_eq!(stats.p99, Duration::from_millis(7));
    }

    #[test]
    fn no_samples() {
        assert!(LatencyStats::from_samples(Vec::new()).is_none());
    }
}
