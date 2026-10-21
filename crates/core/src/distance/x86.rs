//! AVX2 + FMA distance kernels for x86-64.
//!
//! The default x86-64 target only guarantees SSE2, so these kernels are
//! compiled with `#[target_feature]` and picked at runtime when
//! [`available`] says the CPU supports them (every Intel CPU since Haswell,
//! 2013, and every AMD CPU since Excavator, 2015).
//!
//! Each kernel keeps 4 independent 8-lane accumulators, 32 floats per loop
//! iteration, so consecutive fused multiply-adds don't wait on each other.
//! On a Tiger Lake laptop that is 2.0x faster than the portable kernels on
//! 128-d vectors, and 2.2x on 960-d ones.
//!
//! Like the portable kernels, these read `min(a.len(), b.len())` values, so
//! mismatched lengths can never read out of bounds.

use std::arch::x86_64::{
    __m256, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps, _mm256_add_ps,
    _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
    _mm256_setzero_ps, _mm256_sub_ps,
};

/// Whether the CPU running this code supports AVX2 and FMA.
///
/// Free when the binary is compiled with both features enabled (for example
/// with `-C target-cpu=native` on a capable CPU). Otherwise it reads a
/// detection result that the standard library caches after the first call.
#[inline]
pub(super) fn available() -> bool {
    cfg!(all(target_feature = "avx2", target_feature = "fma"))
        || (std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma"))
}

/// `sum(a[i] * b[i])` over the first `min(a.len(), b.len())` values.
///
/// Rounds differently from the portable kernel (fused multiply-adds, a
/// different summation order), so results can differ in the last bits.
#[target_feature(enable = "avx2,fma")]
pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a8, a_tail) = a[..n].as_chunks::<8>();
    let (b8, b_tail) = b[..n].as_chunks::<8>();
    let (a32, a8_rest) = a8.as_chunks::<4>();
    let (b32, b8_rest) = b8.as_chunks::<4>();

    let mut acc = [_mm256_setzero_ps(); 4];
    for (xs, ys) in a32.iter().zip(b32) {
        for k in 0..4 {
            acc[k] = _mm256_fmadd_ps(load(&xs[k]), load(&ys[k]), acc[k]);
        }
    }
    for (x, y) in a8_rest.iter().zip(b8_rest) {
        acc[0] = _mm256_fmadd_ps(load(x), load(y), acc[0]);
    }

    let mut sum = horizontal_sum(acc);
    for (x, y) in a_tail.iter().zip(b_tail) {
        sum += x * y;
    }
    sum
}

/// `sum((a[i] - b[i])^2)` over the first `min(a.len(), b.len())` values.
///
/// Rounds differently from the portable kernel (fused multiply-adds, a
/// different summation order), so results can differ in the last bits.
#[target_feature(enable = "avx2,fma")]
pub(super) fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a8, a_tail) = a[..n].as_chunks::<8>();
    let (b8, b_tail) = b[..n].as_chunks::<8>();
    let (a32, a8_rest) = a8.as_chunks::<4>();
    let (b32, b8_rest) = b8.as_chunks::<4>();

    let mut acc = [_mm256_setzero_ps(); 4];
    for (xs, ys) in a32.iter().zip(b32) {
        for k in 0..4 {
            let d = _mm256_sub_ps(load(&xs[k]), load(&ys[k]));
            acc[k] = _mm256_fmadd_ps(d, d, acc[k]);
        }
    }
    for (x, y) in a8_rest.iter().zip(b8_rest) {
        let d = _mm256_sub_ps(load(x), load(y));
        acc[0] = _mm256_fmadd_ps(d, d, acc[0]);
    }

    let mut sum = horizontal_sum(acc);
    for (x, y) in a_tail.iter().zip(b_tail) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// `[a·b, a·a, b·b]` over the first `min(a.len(), b.len())` values, in one
/// pass.
///
/// Each sum runs through the same operations in the same order as [`dot`]
/// would, so each value has the same bits as the matching `dot` call. The
/// 12 accumulators and 2 loaded values fit in the 16 AVX registers.
#[target_feature(enable = "avx2,fma")]
pub(super) fn dot_and_squared_norms(a: &[f32], b: &[f32]) -> [f32; 3] {
    let n = a.len().min(b.len());
    let (a8, a_tail) = a[..n].as_chunks::<8>();
    let (b8, b_tail) = b[..n].as_chunks::<8>();
    let (a32, a8_rest) = a8.as_chunks::<4>();
    let (b32, b8_rest) = b8.as_chunks::<4>();

    let mut ab = [_mm256_setzero_ps(); 4];
    let mut aa = [_mm256_setzero_ps(); 4];
    let mut bb = [_mm256_setzero_ps(); 4];
    for (xs, ys) in a32.iter().zip(b32) {
        for k in 0..4 {
            let (xv, yv) = (load(&xs[k]), load(&ys[k]));
            ab[k] = _mm256_fmadd_ps(xv, yv, ab[k]);
            aa[k] = _mm256_fmadd_ps(xv, xv, aa[k]);
            bb[k] = _mm256_fmadd_ps(yv, yv, bb[k]);
        }
    }
    for (x, y) in a8_rest.iter().zip(b8_rest) {
        let (xv, yv) = (load(x), load(y));
        ab[0] = _mm256_fmadd_ps(xv, yv, ab[0]);
        aa[0] = _mm256_fmadd_ps(xv, xv, aa[0]);
        bb[0] = _mm256_fmadd_ps(yv, yv, bb[0]);
    }

    let mut sums = [horizontal_sum(ab), horizontal_sum(aa), horizontal_sum(bb)];
    for (x, y) in a_tail.iter().zip(b_tail) {
        sums[0] += x * y;
        sums[1] += x * x;
        sums[2] += y * y;
    }
    sums
}

/// Loads 8 floats into a register.
#[inline]
#[target_feature(enable = "avx")]
fn load(v: &[f32; 8]) -> __m256 {
    // SAFETY: `v` is 8 readable `f32`s, and `loadu` has no alignment
    // requirement.
    unsafe { _mm256_loadu_ps(v.as_ptr()) }
}

/// Adds the 4 accumulators, then the 8 lanes of the result.
#[inline]
#[target_feature(enable = "avx")]
fn horizontal_sum(acc: [__m256; 4]) -> f32 {
    let v = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps::<0b01>(s, s));
    _mm_cvtss_f32(s)
}
