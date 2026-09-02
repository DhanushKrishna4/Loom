//! RMSNorm.

use crate::math::rsqrt;

/// `out = x * rsqrt(mean(x^2) + eps) * weight`
///
/// This is **not** LayerNorm. There is no mean subtraction and no bias term --
/// only the scale. Adding a mean subtraction "because that is what normalisation
/// means" produces a model that still generates plausible text while being
/// quietly wrong, which is the worst failure mode available.
///
/// `eps` goes *inside* the square root, added to the mean of squares. Some
/// formulations add it outside; for Qwen2's eps of 1e-6 the difference is small
/// but it is not zero, and it is the kind of thing that shows up as a slow drift
/// across 24 layers.
pub fn rmsnorm(x: &[f32], weight: &[f32], out: &mut [f32], eps: f32) {
    assert_eq!(x.len(), weight.len(), "rmsnorm weight length mismatch");
    assert_eq!(x.len(), out.len(), "rmsnorm output length mismatch");
    debug_assert!(!x.is_empty(), "rmsnorm of an empty vector is undefined");

    // Accumulate the sum of squares in f32 to match the reference implementations.
    // HF's Qwen2RMSNorm upcasts bf16 activations to f32 for exactly this step; we
    // are already f32, so there is nothing to upcast, but the accumulation order
    // still has to match for the tolerance budget to mean anything.
    let mut ss = 0.0f32;
    for v in x {
        ss += v * v;
    }
    let scale = rsqrt(ss / x.len() as f32 + eps);

    for ((o, xi), w) in out.iter_mut().zip(x).zip(weight) {
        *o = xi * scale * w;
    }
}
