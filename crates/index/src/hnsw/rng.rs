//! A tiny seeded random number generator.
//!
//! HNSW only needs random numbers to pick each node's top layer. A seeded
//! generator makes builds reproducible: the same data and seed always give the
//! same graph, which is essential for debugging and fair benchmarks.

/// `SplitMix64` (Steele, Lea & Flood, 2014): fast, tiny, good enough statistically.
#[derive(Debug, Clone)]
pub(super) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Creates a generator. Any seed is fine, including 0.
    pub(super) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next 64 uniformly distributed random bits.
    pub(super) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1]`. Never 0, so `ln()` of it is always finite.
    pub(super) fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = SplitMix64::new(7);
        let mut b = SplitMix64::new(7);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn floats_are_in_range() {
        let mut rng = SplitMix64::new(1);
        for _ in 0..10_000 {
            let x = rng.next_f64();
            assert!(x > 0.0 && x <= 1.0, "{x}");
        }
    }

    #[test]
    fn matches_the_reference_implementation() {
        // First outputs for seed 0 from the reference SplitMix64.
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(rng.next_u64(), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn floats_are_roughly_uniform() {
        let mut rng = SplitMix64::new(3);
        let n = 100_000;
        let mean = (0..n).map(|_| rng.next_f64()).sum::<f64>() / f64::from(n);
        assert!((mean - 0.5).abs() < 0.01, "mean {mean}");
    }
}
