//! Softmax, SiLU, and the SwiGLU elementwise step.

use crate::math::exp;

/// In-place softmax, max-subtracted.
///
/// Subtracting the max before exponentiating is not optional. Attention scores
/// reach into the tens, `exp(89)` overflows f32, and the resulting inf/inf gives
/// NaN that propagates through the whole residual stream. Subtracting the max
/// makes every argument <= 0, so every `exp` lands in (0, 1].
///
/// A fully-masked row (every entry -inf) sums to zero; rather than emit NaN this
/// leaves the row at zero and lets the caller notice. Causal attention never
/// produces one -- position `p` always attends to at least itself -- so reaching
/// that branch means the mask is wrong.
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    let mut max = f32::NEG_INFINITY;
    for v in x.iter() {
        if *v > max {
            max = *v;
        }
    }
    if !max.is_finite() {
        // All -inf (or a NaN got in). Leave zeros; see the doc comment.
        x.fill(0.0);
        return;
    }

    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = exp(*v - max);
        sum += *v;
    }

    // sum >= 1 always, because the max element contributes exp(0) = 1.
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// `silu(x) = x * sigmoid(x) = x / (1 + e^-x)`
///
/// Also called Swish. The naive form is numerically safe at both ends without
/// any branching: for very negative `x`, `exp(-x)` saturates to +inf and
/// `x / inf` gives -0.0, which is the correct limit; for very positive `x`,
/// `exp(-x)` underflows to 0 and the result is `x`. No NaN is reachable for
/// finite input.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + exp(-x))
}

pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = silu(*v);
    }
}

/// The elementwise half of SwiGLU: `gate = silu(gate) * up`.
///
/// The full FFN is `silu(x @ W_gate) * (x @ W_up) @ W_down`; this is the middle
/// step, applied in place on the gate projection so the FFN needs two scratch
/// buffers rather than three.
pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    assert_eq!(gate.len(), up.len(), "swiglu length mismatch");
    for (g, u) in gate.iter_mut().zip(up) {
        *g = silu(*g) * u;
    }
}
