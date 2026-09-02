//! Oracle implementations, computed in f64.
//!
//! # Why f64 and not just "the naive version"
//!
//! The obvious oracle is a second f32 implementation written with a different
//! loop structure. That catches transcription errors, but it cannot tell you
//! anything about *accuracy* -- two f32 implementations that agree to the last
//! bit might both be drifting.
//!
//! These accumulate in f64 instead. Comparing against them measures the real
//! error of the f32 kernel against something much closer to exact, which is what
//! you actually want to know when step 9 starts reordering summations for SIMD:
//! the question then is not "did the answer change" (it will) but "did it get
//! worse". This gives a fixed yardstick to answer that against.
//!
//! Compiled under `cfg(test)` or the `reference` feature, so it never reaches
//! the wasm binary.

use alloc::vec;
use alloc::vec::Vec;

/// `out[i, j] = dot(lhs[i, ..], rhs[j, ..])`, accumulated in f64.
pub fn matmul_nt(lhs: &[f32], rhs: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(lhs.len(), m * k);
    assert_eq!(rhs.len(), n * k);
    assert_eq!(out.len(), m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += lhs[i * k + p] as f64 * rhs[j * k + p] as f64;
            }
            out[i * n + j] = acc as f32;
        }
    }
}

pub fn matvec(w: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    assert_eq!(w.len(), rows * cols);
    assert_eq!(x.len(), cols);
    assert_eq!(out.len(), rows);
    for r in 0..rows {
        let mut acc = 0.0f64;
        for c in 0..cols {
            acc += w[r * cols + c] as f64 * x[c] as f64;
        }
        out[r] = acc as f32;
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f64;
    for i in 0..a.len() {
        acc += a[i] as f64 * b[i] as f64;
    }
    acc as f32
}

/// `out = x * rsqrt(mean(x^2) + eps) * weight`, in f64.
pub fn rmsnorm(x: &[f32], weight: &[f32], out: &mut [f32], eps: f32) {
    let mut ss = 0.0f64;
    for v in x {
        ss += (*v as f64) * (*v as f64);
    }
    let scale = 1.0 / libm::sqrt(ss / x.len() as f64 + eps as f64);
    for i in 0..x.len() {
        out[i] = (x[i] as f64 * scale * weight[i] as f64) as f32;
    }
}

/// Max-subtracted softmax in f64.
pub fn softmax(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![0.0; x.len()];
    }
    let mut e: Vec<f64> = x
        .iter()
        .map(|v| libm::exp(*v as f64 - max as f64))
        .collect();
    let sum: f64 = e.iter().sum();
    for v in e.iter_mut() {
        *v /= sum;
    }
    e.into_iter().map(|v| v as f32).collect()
}

/// `x * sigmoid(x)` in f64.
pub fn silu(x: f32) -> f32 {
    let xd = x as f64;
    (xd / (1.0 + libm::exp(-xd))) as f32
}

pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(g, u)| (silu(*g) as f64 * *u as f64) as f32)
        .collect()
}

/// RoPE stated as what it actually is: a complex multiplication.
///
/// Each plane holds a complex number `x[a] + i*x[b]`, and rope multiplies it by
/// `e^(i * pos * theta)`. Writing it this way rather than as the four-term
/// rotation matrix is a genuinely different expression of the same thing, so a
/// sign error in the real kernel's matrix form shows up here.
pub fn rope_head(
    x: &[f32],
    pos: usize,
    head_dim: usize,
    rope_dim: usize,
    freq_base: f32,
    split_half: bool,
) -> Vec<f32> {
    let mut out = x.to_vec();
    let n = rope_dim / 2;
    for i in 0..n {
        let theta = 1.0 / libm::pow(freq_base as f64, 2.0 * i as f64 / rope_dim as f64);
        let angle = pos as f64 * theta;
        let (c, s) = (libm::cos(angle), libm::sin(angle));
        let (a, b) = if split_half {
            (i, i + n)
        } else {
            (2 * i, 2 * i + 1)
        };
        // (re + i*im) * (cos + i*sin)
        let re = x[a] as f64;
        let im = x[b] as f64;
        out[a] = (re * c - im * s) as f32;
        out[b] = (re * s + im * c) as f32;
    }
    debug_assert_eq!(out.len(), head_dim);
    out
}
