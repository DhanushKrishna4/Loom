//! Elementwise helpers. Small, but they are the residual stream.

/// `a += b`
pub fn add_assign(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "add_assign length mismatch");
    for (x, y) in a.iter_mut().zip(b) {
        *x += *y;
    }
}

/// `a *= b`
pub fn mul_assign(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "mul_assign length mismatch");
    for (x, y) in a.iter_mut().zip(b) {
        *x *= *y;
    }
}

/// `a *= s`
pub fn scale(a: &mut [f32], s: f32) {
    for x in a.iter_mut() {
        *x *= s;
    }
}

/// Plain sequential dot product.
///
/// Sequential summation is a deliberate choice for now: it is what the reference
/// implementations do, so the f32 accumulation order matches and the tolerance
/// budget stays honest. Step 9 will want multiple accumulators to break the
/// dependency chain, which changes the rounding -- that is the moment to widen
/// tolerances, with the naive version kept as the oracle.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot length mismatch");
    let mut acc = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        acc += x * y;
    }
    acc
}
