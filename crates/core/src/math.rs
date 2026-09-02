//! The float math `core` does not have.
//!
//! `core` has no `sqrt`, no `exp`, no `sin` -- they all live in `std` because
//! they lower to libm calls. Step 3 needs every one of them: rmsnorm wants
//! `rsqrt`, softmax and SiLU want `exp`, and RoPE's cos/sin table wants `sin`,
//! `cos` and `powf`.
//!
//! # Why libm on every target, including when `std` is available
//!
//! The obvious arrangement is to use `std`'s intrinsics when we have them and
//! fall back to libm only for `no_std`. That would make the native build and the
//! wasm build compute *slightly different numbers* -- host libm and Rust's libm
//! agree to within an ulp, not exactly.
//!
//! An ulp does not matter for tolerance-based tests. It matters for two things
//! this project has explicitly promised:
//!
//! * the golden generation test (fixed prompt, fixed seed, exact token sequence)
//!   only means something for the wasm build if the wasm build does the same
//!   arithmetic as the native build it was validated against;
//! * a shared URL carrying a seed has to reproduce the same output for whoever
//!   opens it, and a near-tie in an argmax or a top-p cut can be decided by the
//!   last mantissa bit.
//!
//! So: one implementation, everywhere, no `cfg`. The cost is that `sqrt` is a
//! software routine rather than the hardware instruction `std` would emit.
//! rmsnorm calls it twice per layer, 48 times per token, against millions of
//! cycles of matmul -- under 0.1% of the forward pass. If step 9's profiles ever
//! disagree, that is the moment to revisit it, with a benchmark in hand.

/// Square root.
#[inline]
pub fn sqrt(x: f32) -> f32 {
    libm::sqrtf(x)
}

/// Reciprocal square root -- the shape rmsnorm actually wants.
///
/// Kept as its own function so the eventual SIMD path (`f32x4` has no rsqrt on
/// wasm128, so this becomes a divide) has one place to change.
#[inline]
pub fn rsqrt(x: f32) -> f32 {
    1.0 / libm::sqrtf(x)
}

/// e^x.
#[inline]
pub fn exp(x: f32) -> f32 {
    libm::expf(x)
}

#[inline]
pub fn sin(x: f32) -> f32 {
    libm::sinf(x)
}

#[inline]
pub fn cos(x: f32) -> f32 {
    libm::cosf(x)
}

/// x^y. RoPE needs it once per (head dim, position) at table-build time.
#[inline]
pub fn powf(x: f32, y: f32) -> f32 {
    libm::powf(x, y)
}

/// Round half away from zero, matching `std`'s `f32::round`.
///
/// # Not the bias trick
///
/// The tempting version is `(x + copysign(0.5, x)) as i32 as f32`: `as i32`
/// truncates toward zero, so biasing by half first looks like it gives
/// round-half-away-from-zero with no function call. It does not, and the failure
/// is silent.
///
/// `0.49999997` is the largest f32 below `0.5`. Adding `0.5` gives `0.99999997`
/// exactly -- which is precisely halfway between the two representable neighbours
/// `0.99999994` and `1.0`, because the exponent stepped up and the mantissa
/// spacing doubled. Ties-to-even picks `1.0`, and the trick returns 1 where
/// `round` returns 0. Every value just under a `.5` boundary that crosses a
/// binade has the same problem.
///
/// In `quantize_block_q8_0` that would push the smallest activations in a block
/// to +/-1 instead of 0 -- a small, permanent, entirely invisible error. libm's
/// `roundf` is correct, and it is also what ggml's own scalar reference uses.
///
/// (ggml's *SIMD* paths use round-to-nearest-even instead, as will ours in step 9
/// when `f32x4.nearest` becomes available -- the two differ only on exact ties,
/// which do not occur in real activation data.)
#[inline]
pub fn round_half_away_from_zero(x: f32) -> f32 {
    libm::roundf(x)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn known_values() {
        assert_eq!(sqrt(0.0), 0.0);
        assert_eq!(sqrt(1.0), 1.0);
        assert_eq!(sqrt(4.0), 2.0);
        assert_eq!(sqrt(1e-8), 1e-4);
        assert!(sqrt(-1.0).is_nan());

        assert_eq!(exp(0.0), 1.0);
        assert!((exp(1.0) - core::f32::consts::E).abs() < 1e-6);
        assert_eq!(exp(f32::NEG_INFINITY), 0.0);

        assert_eq!(sin(0.0), 0.0);
        assert_eq!(cos(0.0), 1.0);
        assert!(sin(core::f32::consts::FRAC_PI_2) > 0.9999995);

        assert_eq!(powf(2.0, 10.0), 1024.0);
        assert_eq!(powf(10.0, 0.0), 1.0);
        // The shape RoPE uses: theta_i = freq_base^(-2i/d).
        assert!((powf(1_000_000.0, -2.0 * 8.0 / 64.0) - 0.031622776).abs() < 1e-7);
    }

    #[test]
    fn sqrt_is_exact_for_perfect_squares() {
        // sqrt is required by IEEE-754 to be correctly rounded, so this must be
        // exact rather than approximate -- if it is not, we have the wrong routine.
        for n in 0..=2048u32 {
            let x = n as f32;
            assert_eq!(sqrt(x * x), x, "sqrt({})", x * x);
        }
    }

    /// Documents how far our libm sits from the host's. Not a correctness
    /// requirement -- we deliberately use libm on both targets *instead of* the
    /// host's -- but a large divergence here would mean one of them is wrong.
    #[test]
    fn agrees_with_the_host_libm() {
        let mut worst: f64 = 0.0;
        let mut check = |ours: f32, theirs: f32, what: &str, at: f32| {
            if theirs.abs() > 1e-30 {
                let rel = ((ours as f64 - theirs as f64) / theirs as f64).abs();
                worst = worst.max(rel);
                assert!(
                    rel < 1e-6,
                    "{what}({at}): {ours} vs host {theirs} (rel {rel:e})"
                );
            }
        };
        for i in -2000..2000 {
            let x = i as f32 / 100.0;
            check(exp(x), x.exp(), "exp", x);
            check(sin(x), x.sin(), "sin", x);
            check(cos(x), x.cos(), "cos", x);
            if x > 0.0 {
                check(sqrt(x), x.sqrt(), "sqrt", x);
                check(powf(x, 1.5), x.powf(1.5), "powf", x);
            }
        }
        // f32::EPSILON is 1.19e-7, so a couple of ulp.
        assert!(worst < 1e-6, "worst relative divergence {worst:e}");
    }

    #[test]
    fn round_matches_std() {
        // Includes the ties, which is the only interesting part.
        for i in -100_000..100_000i32 {
            let x = i as f32 / 2.0; // hits every .0 and .5
            assert_eq!(round_half_away_from_zero(x), x.round(), "round({x})");
        }
        assert_eq!(round_half_away_from_zero(2.5), 3.0);
        assert_eq!(round_half_away_from_zero(-2.5), -3.0);
        assert_eq!(round_half_away_from_zero(0.49999997), 0.0);
    }

    proptest! {
        #[test]
        fn trig_identity_holds(x in -1000.0f32..1000.0) {
            let s = sin(x);
            let c = cos(x);
            prop_assert!((s * s + c * c - 1.0).abs() < 1e-5, "sin^2+cos^2 at {x}");
        }

        #[test]
        fn rsqrt_is_the_reciprocal_of_sqrt(x in 1e-6f32..1e6) {
            prop_assert!(((rsqrt(x) * sqrt(x)) - 1.0).abs() < 1e-5);
        }

        #[test]
        fn exp_is_monotone_and_positive(a in -80.0f32..80.0, b in -80.0f32..80.0) {
            prop_assume!(a < b);
            prop_assert!(exp(a) > 0.0 || a < -70.0);
            prop_assert!(exp(a) <= exp(b), "exp({a}) > exp({b})");
        }

        /// Softmax subtracts the max before exponentiating, so the only inputs
        /// that reach `exp` in practice are <= 0. None of them may overflow.
        #[test]
        fn exp_of_non_positive_is_finite_and_bounded(x in -200.0f32..0.0) {
            let e = exp(x);
            prop_assert!(e.is_finite() && (0.0..=1.0).contains(&e), "exp({x}) = {e}");
        }
    }
}
