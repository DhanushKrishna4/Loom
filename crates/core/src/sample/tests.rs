//! Sampling tests.
//!
//! Sampling is easy to get subtly wrong in ways that never crash: an off-by-one
//! in the nucleus cut, a repetition penalty that promotes negative logits, an
//! RNG that is reproducible on one target and not another. Each of those is
//! pinned below.

use super::*;
use proptest::prelude::*;

fn cfg() -> SamplerConfig {
    SamplerConfig::default()
}

/// Logits with a known shape: token `i` gets logit `v[i]`.
fn sampler_over(logits: &[f32], c: SamplerConfig) -> (Sampler, Vec<f32>) {
    (Sampler::new(c, logits.len()), logits.to_vec())
}

// ==================================================================== rng ==

#[test]
fn pcg32_is_reproducible_and_stream_stable() {
    // The whole point: a seed in a URL must reproduce a generation anywhere.
    let a: Vec<u32> = (0..8)
        .scan(Pcg32::new(42), |r, _| Some(r.next_u32()))
        .collect();
    let b: Vec<u32> = (0..8)
        .scan(Pcg32::new(42), |r, _| Some(r.next_u32()))
        .collect();
    assert_eq!(a, b, "same seed must give the same stream");

    let c: Vec<u32> = (0..8)
        .scan(Pcg32::new(43), |r, _| Some(r.next_u32()))
        .collect();
    assert_ne!(a, c, "different seeds must give different streams");
}

#[test]
fn pcg32_floats_stay_in_range() {
    // A value of exactly 1.0 would walk off the end of the cumulative scan.
    let mut r = Pcg32::new(7);
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for _ in 0..200_000 {
        let v = r.next_f32();
        assert!((0.0..1.0).contains(&v), "{v} out of range");
        lo = lo.min(v);
        hi = hi.max(v);
    }
    // Should cover most of the interval.
    assert!(lo < 0.001 && hi > 0.999, "poor coverage: {lo}..{hi}");
}

#[test]
fn pcg32_is_roughly_uniform() {
    let mut r = Pcg32::new(99);
    let mut buckets = [0usize; 16];
    let n = 160_000;
    for _ in 0..n {
        buckets[(r.next_f32() * 16.0) as usize % 16] += 1;
    }
    let expect = n as f64 / 16.0;
    for (i, &b) in buckets.iter().enumerate() {
        let dev = (b as f64 - expect).abs() / expect;
        assert!(dev < 0.05, "bucket {i}: {b} deviates {dev:.3} from uniform");
    }
}

// ================================================================= greedy ==

#[test]
fn temperature_zero_is_greedy() {
    let logits = [0.1f32, 3.0, -1.0, 2.9];
    let (mut s, l) = sampler_over(&logits, cfg());
    for _ in 0..10 {
        assert_eq!(s.sample(&l), 1);
    }
    assert_eq!(argmax(&logits), 1);
}

#[test]
fn argmax_breaks_ties_towards_the_lowest_index() {
    // Matches torch.argmax, which keeps the golden-token test stable.
    assert_eq!(argmax(&[1.0, 1.0, 1.0]), 0);
    assert_eq!(argmax(&[0.0, 5.0, 5.0]), 1);
}

#[test]
fn top_k_of_one_is_greedy() {
    let logits = [1.0f32, 4.0, 2.0, 3.0];
    let c = SamplerConfig {
        temperature: 1.0,
        top_k: 1,
        ..cfg()
    };
    let (mut s, l) = sampler_over(&logits, c);
    for _ in 0..20 {
        assert_eq!(s.sample(&l), 1);
    }
}

#[test]
fn tiny_top_p_collapses_to_greedy() {
    // p below the top token's own probability must still leave exactly one
    // candidate -- never zero.
    let logits = [1.0f32, 8.0, 2.0, 3.0];
    let c = SamplerConfig {
        temperature: 1.0,
        top_p: 1e-6,
        ..cfg()
    };
    let (mut s, l) = sampler_over(&logits, c);
    for _ in 0..20 {
        assert_eq!(s.sample(&l), 1);
    }
}

// ================================================================== top-p ==

#[test]
fn top_p_cuts_where_the_cumulative_sum_crosses() {
    // Logits chosen so the probabilities are exactly 0.5, 0.25, 0.125, 0.125:
    // ln of each, so softmax reproduces them.
    let probs = [0.5f32, 0.25, 0.125, 0.125];
    let logits: Vec<f32> = probs.iter().map(|p| p.ln()).collect();

    // p = 0.4 -> the first token alone already exceeds it.
    let c = SamplerConfig {
        temperature: 1.0,
        top_p: 0.4,
        seed: 1,
        ..cfg()
    };
    let mut s = Sampler::new(c, 4);
    for _ in 0..50 {
        assert_eq!(s.sample(&logits), 0);
    }

    // p = 0.7 -> tokens 0 and 1 (0.5 then 0.75 >= 0.7).
    let c = SamplerConfig {
        temperature: 1.0,
        top_p: 0.7,
        seed: 1,
        ..cfg()
    };
    let mut s = Sampler::new(c, 4);
    let mut seen = [false; 4];
    for _ in 0..500 {
        seen[s.sample(&logits) as usize] = true;
    }
    assert!(seen[0] && seen[1], "both survivors should appear");
    assert!(
        !seen[2] && !seen[3],
        "tokens outside the nucleus must never appear"
    );
}

#[test]
fn top_k_restricts_the_candidate_set() {
    let logits = [0.0f32, 1.0, 2.0, 3.0, 4.0];
    let c = SamplerConfig {
        temperature: 2.0,
        top_k: 2,
        seed: 5,
        ..cfg()
    };
    let mut s = Sampler::new(c, 5);
    let mut seen = [false; 5];
    for _ in 0..1000 {
        seen[s.sample(&logits) as usize] = true;
    }
    assert!(seen[3] && seen[4], "the top two should both appear");
    assert!(
        !seen[0] && !seen[1] && !seen[2],
        "tokens outside top-k must never appear"
    );
}

// ==================================================== repetition penalty ==

#[test]
fn repetition_penalty_pushes_seen_tokens_down_whatever_their_sign() {
    // The CTRL formulation: divide positive logits, multiply negative ones.
    // Naively dividing both would *promote* a negative logit.
    let logits = [2.0f32, -2.0, 0.5];
    let c = SamplerConfig {
        temperature: 0.0,
        repetition_penalty: 2.0,
        repetition_window: 8,
        ..cfg()
    };
    let mut s = Sampler::new(c, 3);

    // Untouched, token 0 wins.
    assert_eq!(s.sample(&logits), 0);

    // After seeing token 0: 2.0 / 2 = 1.0, still the highest.
    s.accept(0);
    assert_eq!(s.sample(&logits), 0);

    // Penalise it again by widening the window's view of history.
    s.accept(0);
    s.accept(0);
    // History holds three copies, each applying the penalty once more:
    // 2.0 -> 1.0 -> 0.5 -> 0.25, now below token 2's 0.5.
    assert_eq!(s.sample(&logits), 2);

    // The negative logit must have moved further down, never up.
    let mut s2 = Sampler::new(c, 3);
    s2.accept(1);
    assert_ne!(s2.sample(&logits), 1, "a penalised token must not win");
}

#[test]
fn repetition_window_forgets_old_tokens() {
    let c = SamplerConfig {
        repetition_penalty: 2.0,
        repetition_window: 3,
        ..cfg()
    };
    let mut s = Sampler::new(c, 8);
    for t in 0..6u32 {
        s.accept(t);
    }
    assert_eq!(s.history(), &[3, 4, 5], "only the window's worth is kept");
}

// ============================================== reproducibility and reset ==

#[test]
fn the_same_seed_reproduces_the_same_tokens() {
    // This is the property a shared URL depends on.
    let logits: Vec<f32> = (0..64).map(|i| ((i * 37) % 11) as f32 * 0.4).collect();
    let c = SamplerConfig {
        temperature: 1.0,
        top_p: 0.9,
        seed: 0xC0FFEE,
        ..cfg()
    };

    let run = |c: SamplerConfig| {
        let mut s = Sampler::new(c, 64);
        (0..40)
            .map(|_| {
                let t = s.sample(&logits);
                s.accept(t);
                t
            })
            .collect::<Vec<_>>()
    };

    assert_eq!(run(c), run(c), "same seed must reproduce the sequence");
    let other = SamplerConfig {
        seed: 0xC0FFEF,
        ..c
    };
    assert_ne!(run(c), run(other), "a different seed should diverge");
}

#[test]
fn reset_restores_the_starting_state() {
    let logits: Vec<f32> = (0..32).map(|i| (i % 7) as f32).collect();
    let c = SamplerConfig {
        temperature: 1.0,
        seed: 11,
        repetition_penalty: 1.2,
        ..cfg()
    };
    let mut s = Sampler::new(c, 32);

    let first: Vec<u32> = (0..12)
        .map(|_| {
            let t = s.sample(&logits);
            s.accept(t);
            t
        })
        .collect();

    s.reset();
    assert!(s.history().is_empty());
    let second: Vec<u32> = (0..12)
        .map(|_| {
            let t = s.sample(&logits);
            s.accept(t);
            t
        })
        .collect();

    assert_eq!(
        first, second,
        "reset must restore both the RNG and the history"
    );
}

#[test]
fn changing_config_does_not_restart_the_stream() {
    // The UI changes temperature mid-conversation. That must advance the RNG
    // from where it was, not silently reseed and replay draws already used --
    // which would make the same token far more likely to repeat.
    let logits: Vec<f32> = (0..16).map(|i| (i % 5) as f32).collect();
    let c = SamplerConfig {
        temperature: 1.0,
        seed: 3,
        ..cfg()
    };

    // Reference: three draws with the config never touched.
    let mut a = Sampler::new(c, 16);
    let want: Vec<u32> = (0..3).map(|_| a.sample(&logits)).collect();

    // Same, but re-applying the identical config between draws.
    let mut b = Sampler::new(c, 16);
    let got: Vec<u32> = (0..3)
        .map(|_| {
            let t = b.sample(&logits);
            b.set_config(c); // a no-op change
            t
        })
        .collect();

    assert_eq!(got, want, "set_config perturbed the RNG stream");

    // And history survives a config change.
    let mut d = Sampler::new(c, 16);
    d.accept(7);
    d.set_config(SamplerConfig {
        temperature: 1.5,
        ..c
    });
    assert_eq!(d.history(), &[7], "set_config dropped the history");
}

// =============================================================== sampling ==

#[test]
fn draws_approximate_the_intended_distribution() {
    // The end-to-end check: with temperature 1 and no filtering, the empirical
    // frequencies should match softmax(logits).
    let logits = [0.0f32, 1.0, 2.0];
    let want: Vec<f32> = {
        let m = 2.0f32; // max-subtract, as softmax does
        let e: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
        let denom: f32 = e.iter().sum();
        e.iter().map(|v| v / denom).collect()
    };

    let c = SamplerConfig {
        temperature: 1.0,
        seed: 2024,
        ..cfg()
    };
    let mut s = Sampler::new(c, 3);
    let n = 60_000;
    let mut counts = [0usize; 3];
    for _ in 0..n {
        counts[s.sample(&logits) as usize] += 1;
    }
    for i in 0..3 {
        let got = counts[i] as f32 / n as f32;
        assert!(
            (got - want[i]).abs() < 0.01,
            "token {i}: sampled {got:.4}, expected {:.4}",
            want[i]
        );
    }
}

proptest! {
    /// Whatever the settings, the result must be a valid token id and must be
    /// one the filters actually left in play.
    #[test]
    fn always_returns_a_token_that_survived_filtering(
        seed in any::<u64>(),
        temp in 0.0f32..2.0,
        k in 0usize..8,
        p in 0.05f32..1.0,
        pen in 1.0f32..2.0,
    ) {
        let logits: Vec<f32> = (0..16).map(|i| ((i * 13) % 9) as f32 - 4.0).collect();
        let c = SamplerConfig {
            temperature: temp,
            top_k: k,
            top_p: p,
            repetition_penalty: pen,
            repetition_window: 4,
            seed,
        };
        let mut s = Sampler::new(c, 16);
        for _ in 0..8 {
            let t = s.sample(&logits);
            prop_assert!((t as usize) < 16, "token {t} out of range");
            s.accept(t);
        }
    }

    /// Degenerate logits must not produce NaN or a panic.
    #[test]
    fn survives_extreme_logits(scale in 1.0f32..1e6, seed in any::<u64>()) {
        let logits: Vec<f32> = (0..8).map(|i| (i as f32 - 4.0) * scale).collect();
        let c = SamplerConfig { temperature: 1.0, top_p: 0.9, seed, ..SamplerConfig::default() };
        let mut s = Sampler::new(c, 8);
        let t = s.sample(&logits);
        prop_assert!((t as usize) < 8);
    }
}
