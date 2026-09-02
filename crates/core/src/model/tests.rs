//! Forward-pass tests.
//!
//! The layer-by-layer comparison is the point. Comparing only final logits tells
//! you *that* something is wrong; comparing every layer tells you *where*, and
//! the difference is minutes against days.
//!
//! The reference runs the same weights -- dequantised out of the same GGUF --
//! through PyTorch, using transformers' own `apply_rotary_pos_emb` and
//! `repeat_kv`. Same weights means any disagreement is our arithmetic rather
//! than the quantiser's, so the tolerance can be tight.

use super::*;
use crate::gguf::Gguf;
use crate::ops::RopeKind;
use crate::sample::{Sampler, SamplerConfig};
use crate::testutil::npy;
use alloc::vec::Vec;
use std::sync::OnceLock;

const MODEL: &str = "../../models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
/// Must match PROMPT_TOKENS in tools/dump_reference_activations.py.
/// "The capital of France is"
const PROMPT: [u32; 5] = [785, 6722, 315, 9625, 374];

fn gguf() -> Option<&'static Gguf<'static>> {
    static CELL: OnceLock<Option<&'static Gguf<'static>>> = OnceLock::new();
    *CELL.get_or_init(|| {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL);
        let bytes: &'static [u8] = Box::leak(std::fs::read(path).ok()?.into_boxed_slice());
        Some(&*Box::leak(Box::new(Gguf::parse(bytes).ok()?)))
    })
}

fn have_reference() -> bool {
    npy::load("activations/logits.npy").is_some()
}

/// Worst absolute and relative error, ignoring near-zero reference values for
/// the relative figure.
fn worst(got: &[f32], want: &[f32]) -> (f32, f32) {
    let (mut a, mut r) = (0.0f32, 0.0f32);
    for (g, w) in got.iter().zip(want) {
        a = a.max((g - w).abs());
        if w.abs() > 1e-3 {
            r = r.max((g - w).abs() / w.abs());
        }
    }
    (a, r)
}

/// Relative L2 error, `||got - want|| / ||want||`.
///
/// The right metric for a whole activation vector. Worst *element-wise*
/// relative error is meaningless here: a hidden state has plenty of components
/// near zero, and dividing by them turns a 1e-3 absolute difference into a
/// relative error of 5. That is a property of the denominator, not of the
/// implementation.
fn rel_l2(got: &[f32], want: &[f32]) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        num += ((g - w) as f64).powi(2);
        den += (*w as f64).powi(2);
    }
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

/// Cosine similarity, the other view of "is this the same vector".
fn cosine(got: &[f32], want: &[f32]) -> f32 {
    let (mut d, mut a, mut b) = (0.0f64, 0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        d += *g as f64 * *w as f64;
        a += (*g as f64).powi(2);
        b += (*w as f64).powi(2);
    }
    (d / (a.sqrt() * b.sqrt()).max(1e-12)) as f32
}

#[test]
fn weights_load_with_the_expected_shapes() {
    let Some(g) = gguf() else {
        std::eprintln!("note: skipping -- needs {MODEL}");
        return;
    };
    let w = ModelWeights::from_gguf(g).expect("weights should load");
    let c = &w.config;

    assert_eq!(w.layers.len(), 24);
    assert_eq!(c.embedding_length, 896);
    assert_eq!(c.head_count, 14);
    assert_eq!(c.head_count_kv, 2);

    // Qwen2 has QKV biases and no output-projection bias. Getting this wrong
    // does not crash; it just makes every attention score quietly wrong.
    assert!(w.has_attn_bias(), "Qwen2 must have attention biases");
    for l in &w.layers {
        assert_eq!(l.attn_q_bias.as_ref().map(Vec::len), Some(c.q_dim()));
        assert_eq!(l.attn_k_bias.as_ref().map(Vec::len), Some(c.kv_dim()));
        assert_eq!(l.attn_v_bias.as_ref().map(Vec::len), Some(c.kv_dim()));
    }

    // Shapes are (rows, cols) = (out, in). A transposed weight would still load
    // without this check and produce plausible garbage.
    let l0 = &w.layers[0];
    assert_eq!((l0.attn_q.rows(), l0.attn_q.cols()), (896, 896));
    assert_eq!((l0.attn_k.rows(), l0.attn_k.cols()), (128, 896)); // GQA: 2 heads
    assert_eq!((l0.attn_v.rows(), l0.attn_v.cols()), (128, 896));
    assert_eq!((l0.ffn_gate.rows(), l0.ffn_gate.cols()), (4864, 896));
    assert_eq!((l0.ffn_down.rows(), l0.ffn_down.cols()), (896, 4864));
    assert_eq!((w.output.rows(), w.output.cols()), (151_936, 896));
}

#[test]
fn kv_cache_geometry_and_reset() {
    let Some(g) = gguf() else { return };
    let c = g.config().unwrap();
    let mut cache = KvCache::new(&c, 128);
    // 2 tensors x 24 layers x 128 positions x 128 kv_dim x 4 bytes.
    assert_eq!(cache.byte_len(), 2 * 24 * 128 * 128 * 4);
    assert_eq!(cache.max_seq(), 128);
    assert!(cache.is_empty());
    cache.reset();
    assert_eq!(cache.len(), 0);
}

/// The headline test: every intermediate against PyTorch, layer by layer.
///
/// Runs the **unfused** path on purpose. PyTorch computes in f32 against the
/// same dequantised weights, so this is an apples-to-apples check of the
/// architecture and the arithmetic, and it can hold a tight tolerance. The
/// fused path quantises activations to 8 bits and is checked separately, with a
/// metric that makes sense for it.
///
/// `#[ignore]` because it runs a real forward pass through the unfused
/// dequantising path -- about 50 seconds in a debug build, against 0.8 s for the
/// whole rest of the suite. It is a verification you ask for, not a regression
/// guard you pay for on every save:
///
/// ```text
/// cargo test --release -p nano-infer-core -- --ignored --nocapture
/// ```
#[test]
#[ignore = "slow: needs --release and the generated reference activations"]
fn matches_pytorch_layer_by_layer() {
    let Some(g) = gguf() else {
        std::eprintln!("note: skipping -- needs {MODEL}");
        return;
    };
    if !have_reference() {
        std::eprintln!(
            "note: skipping -- run\n  python3 tools/dump_reference_activations.py {MODEL}"
        );
        return;
    }

    let weights = ModelWeights::from_gguf(g).unwrap();
    let cfg = weights.config.clone();
    let max_seq = 16;
    let mut model = Model::new(weights, max_seq, RopeKind::SplitHalf);
    model.set_fused(false); // exact f32 path -- see the doc comment
    let mut ws = Workspace::new(&cfg, max_seq);
    let mut cache = KvCache::new(&cfg, max_seq);
    let mut trace = Trace::default();

    model
        .forward(&PROMPT, &mut ws, &mut cache, Some(&mut trace))
        .expect("forward pass");

    let d = cfg.embedding_length;
    let last = PROMPT.len() - 1;

    // Every name the trace records, in execution order.
    let mut names: Vec<alloc::string::String> = alloc::vec!["hidden.0".into()];
    for i in 0..cfg.block_count {
        names.push(alloc::format!("l{i}.attn_norm"));
        names.push(alloc::format!("l{i}.ffn_norm"));
        names.push(alloc::format!("hidden.{}", i + 1));
    }
    names.push("final_norm".into());

    let mut worst_rel_overall = 0.0f32;
    let mut first_bad: Option<alloc::string::String> = None;

    for name in &names {
        let got = trace
            .get(name)
            .unwrap_or_else(|| panic!("trace is missing {name}"));
        let reference = npy::load(&alloc::format!("activations/{name}.npy"))
            .unwrap_or_else(|| panic!("no reference for {name}"));
        // Reference holds every position; we traced only the last one.
        assert_eq!(reference.data.len(), PROMPT.len() * d, "{name} shape");
        let want = &reference.data[last * d..];

        let (abs, rel) = worst(got, want);
        worst_rel_overall = worst_rel_overall.max(rel);
        if rel > 2e-2 && first_bad.is_none() {
            first_bad = Some(alloc::format!("{name}: abs {abs:e} rel {rel:e}"));
        }
        // Print only the layer boundaries, or this is 73 lines of noise.
        if name.starts_with("hidden.") || name == "final_norm" {
            std::eprintln!("  {name:<14} worst abs {abs:.3e}  worst rel {rel:.3e}");
        }
    }

    if let Some(bad) = first_bad {
        panic!("first layer to diverge -- {bad}");
    }

    // Logits: the reference is [T, vocab].
    let logits_ref = npy::load("activations/logits.npy").unwrap();
    let v = cfg.vocab_size;
    let want = &logits_ref.data[last * v..];
    let got = model.logits(&ws);
    let (abs, rel) = worst(got, want);
    std::eprintln!(
        "  {:<14} worst abs {abs:.3e}  worst rel {rel:.3e}",
        "logits"
    );

    // The prediction itself must agree exactly, not just numerically.
    let ours = model.argmax(&ws);
    let theirs = want
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |a, (i, &x)| {
            if x > a.1 {
                (i, x)
            } else {
                a
            }
        })
        .0 as u32;
    assert_eq!(ours, theirs, "argmax disagrees with PyTorch");
    std::eprintln!("  argmax agrees: token {ours}");

    assert!(
        worst_rel_overall < 2e-2,
        "worst relative error across all layers was {worst_rel_overall:e}"
    );
}

/// The RoPE convention, settled numerically rather than by eye.
#[test]
#[ignore = "slow: needs --release and the generated reference activations"]
fn the_wrong_rope_convention_is_obviously_wrong() {
    let Some(g) = gguf() else { return };
    if !have_reference() {
        return;
    }
    let d = g.config().unwrap().embedding_length;
    let last = PROMPT.len() - 1;
    let reference = npy::load("activations/hidden.1.npy").expect("hidden.1");
    let want = &reference.data[last * d..];

    let mut results = Vec::new();
    for kind in [RopeKind::SplitHalf, RopeKind::Adjacent] {
        let weights = ModelWeights::from_gguf(g).unwrap();
        let cfg = weights.config.clone();
        let mut model = Model::new(weights, 16, kind);
        model.set_fused(false);
        let mut ws = Workspace::new(&cfg, 16);
        let mut cache = KvCache::new(&cfg, 16);
        let mut trace = Trace::default();
        model
            .forward(&PROMPT, &mut ws, &mut cache, Some(&mut trace))
            .unwrap();
        let (_, rel) = worst(trace.get("hidden.1").unwrap(), want);
        std::eprintln!("  {kind:?}: worst relative error after layer 0 = {rel:.3e}");
        results.push(rel);
    }

    // Split-half must match; adjacent must not be close. This is the third and
    // last leg of the RoPE question: HuggingFace uses split-half, our split-half
    // matches HuggingFace, and now -- with weights read out of the GGUF -- the
    // GGUF path agrees too, which means the converter does not permute Q/K for
    // Qwen2 the way it does for Llama.
    assert!(
        results[0] < 2e-2,
        "split-half should match: {:e}",
        results[0]
    );
    assert!(
        results[1] > 0.1,
        "adjacent should be badly wrong: {:e}",
        results[1]
    );
}

/// What the fused kernels cost in accuracy, and what they must not cost.
///
/// Quantising activations to 8 bits is the trade that makes the integer inner
/// loop possible. The question is not whether the numbers move -- they do -- but
/// whether the model still computes the same thing. Relative L2 and cosine
/// similarity answer that; worst element-wise relative error does not, because a
/// hidden state is full of near-zero components.
#[test]
#[ignore = "slow: needs --release and the generated reference activations"]
fn fused_kernels_stay_faithful_to_the_reference() {
    let Some(g) = gguf() else {
        std::eprintln!("note: skipping -- needs {MODEL}");
        return;
    };
    if !have_reference() {
        std::eprintln!("note: skipping -- run tools/dump_reference_activations.py");
        return;
    }

    let cfg = g.config().unwrap();
    let d = cfg.embedding_length;
    let last = PROMPT.len() - 1;
    let max_seq = 16;

    let mut runs = Vec::new();
    for fused in [false, true] {
        let weights = ModelWeights::from_gguf(g).unwrap();
        let mut model = Model::new(weights, max_seq, RopeKind::SplitHalf);
        model.set_fused(fused);
        let mut ws = Workspace::new(&cfg, max_seq);
        let mut cache = KvCache::new(&cfg, max_seq);
        let mut trace = Trace::default();
        model
            .forward(&PROMPT, &mut ws, &mut cache, Some(&mut trace))
            .unwrap();
        let argmax = model.argmax(&ws);
        runs.push((trace, argmax, ws.logits_vec()));
    }

    std::eprintln!("  {:<12} {:>12} {:>12}", "layer", "rel L2", "cosine");
    for probe in [
        "hidden.1",
        "hidden.6",
        "hidden.12",
        "hidden.18",
        "hidden.24",
        "final_norm",
    ] {
        let reference = npy::load(&alloc::format!("activations/{probe}.npy")).unwrap();
        let want = &reference.data[last * d..];
        let got = runs[1].0.get(probe).unwrap();
        let (l2, cos) = (rel_l2(got, want), cosine(got, want));
        std::eprintln!("  {probe:<12} {l2:>12.3e} {cos:>12.6}");
        // Error accumulates with depth, but the direction must not drift.
        assert!(l2 < 0.05, "{probe}: relative L2 {l2:e} is too large");
        assert!(
            cos > 0.998,
            "{probe}: cosine {cos} -- the vector has turned"
        );
    }

    // The only thing that actually has to survive: the prediction.
    let (unfused_argmax, fused_argmax) = (runs[0].1, runs[1].1);
    std::eprintln!("  argmax: unfused {unfused_argmax}, fused {fused_argmax}");
    assert_eq!(
        fused_argmax, unfused_argmax,
        "fused path changed the prediction"
    );

    // Logits should stay well correlated -- that is what keeps sampling stable.
    let cos_logits = cosine(&runs[0].2, &runs[1].2);
    std::eprintln!("  logits cosine (fused vs unfused): {cos_logits:.6}");
    assert!(cos_logits > 0.999, "logits cosine {cos_logits}");
}

// ================================================== KV cache and decoding ==

/// Both layouts must produce bit-identical results.
///
/// They hold the same numbers in a different order, so any difference is an
/// indexing bug -- and an indexing bug in a KV cache produces attention over the
/// wrong positions, which reads as a model that has merely got worse.
#[test]
#[ignore = "slow: needs --release and the model file"]
fn both_kv_layouts_give_identical_logits() {
    let Some(g) = gguf() else {
        std::eprintln!("note: skipping -- needs {MODEL}");
        return;
    };
    let cfg = g.config().unwrap();
    let max_seq = 32;

    let mut runs = Vec::new();
    for layout in [KvLayout::PositionMajor, KvLayout::HeadMajor] {
        let weights = ModelWeights::from_gguf(g).unwrap();
        let model = Model::new(weights, max_seq, RopeKind::SplitHalf);
        let mut ws = Workspace::new(&cfg, max_seq);
        let mut cache = KvCache::with_layout(&cfg, max_seq, layout);
        model.forward(&PROMPT, &mut ws, &mut cache, None).unwrap();
        assert_eq!(cache.layout(), layout);
        runs.push(ws.logits_vec());
    }
    assert_eq!(runs[0], runs[1], "the two KV layouts disagree");
}

/// Incremental decode must equal a single-shot run over the same tokens.
///
/// This is the property the whole cache exists to provide: appending one token
/// to an existing cache has to give exactly what re-running the entire sequence
/// would. Position tracking, cache offsets and `len` bookkeeping all fail here
/// if they are wrong, and nowhere else until the output quietly degrades.
#[test]
#[ignore = "slow: needs --release and the model file"]
fn incremental_decode_equals_a_single_pass() {
    let Some(g) = gguf() else { return };
    let cfg = g.config().unwrap();
    let max_seq = 32;
    let weights = ModelWeights::from_gguf(g).unwrap();
    let model = Model::new(weights, max_seq, RopeKind::SplitHalf);

    // Reference: the whole sequence in one call, from an empty cache.
    let full: Vec<u32> = PROMPT.iter().copied().chain([12095u32, 13]).collect();
    let mut ws = Workspace::new(&cfg, max_seq);
    let mut cache = KvCache::new(&cfg, max_seq);
    model.forward(&full, &mut ws, &mut cache, None).unwrap();
    let want = ws.logits_vec();
    assert_eq!(cache.len(), full.len());

    // Incremental: prompt first, then one token at a time into the same cache.
    let mut ws2 = Workspace::new(&cfg, max_seq);
    let mut cache2 = KvCache::new(&cfg, max_seq);
    model.forward(&PROMPT, &mut ws2, &mut cache2, None).unwrap();
    for (i, &tok) in full[PROMPT.len()..].iter().enumerate() {
        model
            .forward_token(tok, PROMPT.len() + i, &mut ws2, &mut cache2, None)
            .unwrap();
    }
    let got = ws2.logits_vec();

    assert_eq!(cache2.len(), full.len());
    assert_eq!(got, want, "incremental decode diverged from a single pass");
}

/// `reset()` must return the cache to a genuinely empty state.
///
/// A stale `len` would let the next conversation attend to the previous one's
/// tokens -- which looks like the model rambling rather than like a bug.
#[test]
#[ignore = "slow: needs --release and the model file"]
fn reset_makes_the_cache_reusable() {
    let Some(g) = gguf() else { return };
    let cfg = g.config().unwrap();
    let max_seq = 32;
    let weights = ModelWeights::from_gguf(g).unwrap();
    let model = Model::new(weights, max_seq, RopeKind::SplitHalf);
    let mut ws = Workspace::new(&cfg, max_seq);
    let mut cache = KvCache::new(&cfg, max_seq);

    model.forward(&PROMPT, &mut ws, &mut cache, None).unwrap();
    let first = ws.logits_vec();

    // Run something else through the same cache, then reset and repeat.
    model
        .forward(&[9707, 11, 1879], &mut ws, &mut cache, None)
        .unwrap();
    assert_eq!(cache.len(), PROMPT.len() + 3);

    cache.reset();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    model.forward(&PROMPT, &mut ws, &mut cache, None).unwrap();
    let second = ws.logits_vec();

    assert_eq!(second, first, "reset left state behind");
}

#[test]
fn kv_cache_reports_its_own_geometry() {
    // Cheap enough to run without the model: only the config is needed.
    let cfg = crate::gguf::ModelConfig {
        architecture: "qwen2".into(),
        name: None,
        block_count: 24,
        embedding_length: 896,
        feed_forward_length: 4864,
        head_count: 14,
        head_count_kv: 2,
        head_dim: 64,
        context_length: 32768,
        rms_norm_eps: 1e-6,
        rope_freq_base: 1e6,
        rope_dimension_count: None,
        vocab_size: 151_936,
        bos_token_id: None,
        eos_token_id: None,
        pad_token_id: None,
        unk_token_id: None,
        add_bos_token: None,
        tokenizer_model: None,
    };
    for layout in [KvLayout::PositionMajor, KvLayout::HeadMajor] {
        let mut cache = KvCache::with_layout(&cfg, 2048, layout);
        // 2 tensors x 24 layers x 2048 positions x 128 kv_dim x 4 bytes = 48 MiB.
        assert_eq!(cache.byte_len(), 48 * 1024 * 1024);
        assert_eq!(cache.n_layers(), 24);
        assert_eq!(cache.utilisation(), 0.0);

        // Round-trip a distinctive vector through every head and position to
        // prove the two layouts address the same logical cells.
        let k: Vec<f32> = (0..128).map(|i| i as f32).collect();
        cache.fill_for_bench(3, 7, &k, &k);
        for h in 0..2 {
            let got = cache.key(3, 7, h);
            assert_eq!(got, &k[h * 64..(h + 1) * 64], "{layout:?} head {h}");
            assert_eq!(cache.value(3, 7, h), got);
        }
        assert_eq!(cache.len(), 8);
        assert!((cache.utilisation() - 8.0 / 2048.0).abs() < 1e-9);
    }
}

// ====================================================== golden generation ==

/// Fixed prompt, greedy decoding, exact token sequence.
///
/// The cheapest regression guard in the project: any change that perturbs the
/// numerics — a reordered summation, a different kernel, a broken cache — moves
/// these ids, and it says so in one line instead of leaving you to notice that
/// the prose got slightly worse.
///
/// The sequence is not an independent oracle; it is what this engine produced
/// once its every layer had been checked against PyTorch. Treat a diff here as
/// "something changed, go find out what", not automatically as "something broke".
#[test]
#[ignore = "slow: needs --release and the model file"]
fn golden_greedy_generation() {
    let Some(g) = gguf() else {
        std::eprintln!("note: skipping -- needs {MODEL}");
        return;
    };
    // "The capital of France is" -> " Paris. It is the largest city in Europe
    //  and the second largest in the world"
    const EXPECTED: [u32; 16] = [
        12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 304, 279, 1879,
    ];

    let cfg = g.config().unwrap();
    let max_seq = 64;
    let weights = ModelWeights::from_gguf(g).unwrap();
    let model = Model::new(weights, max_seq, RopeKind::SplitHalf);
    let mut ws = Workspace::new(&cfg, max_seq);
    let mut cache = KvCache::new(&cfg, max_seq);
    let mut sampler = Sampler::new(SamplerConfig::default(), cfg.vocab_size);

    model.forward(&PROMPT, &mut ws, &mut cache, None).unwrap();
    let mut got = Vec::new();
    for _ in 0..EXPECTED.len() {
        let t = sampler.sample(model.logits(&ws));
        got.push(t);
        model
            .forward_token(t, cache.len(), &mut ws, &mut cache, None)
            .unwrap();
    }
    assert_eq!(got, EXPECTED, "greedy generation drifted");
}

/// A seeded sampled generation must reproduce exactly, and a different seed
/// must not.
///
/// This is what makes a seed in a shared URL mean anything.
#[test]
#[ignore = "slow: needs --release and the model file"]
fn seeded_sampling_is_reproducible() {
    let Some(g) = gguf() else { return };
    let cfg = g.config().unwrap();
    let max_seq = 64;
    let weights = ModelWeights::from_gguf(g).unwrap();
    let model = Model::new(weights, max_seq, RopeKind::SplitHalf);

    let run = |seed: u64| -> Vec<u32> {
        let mut ws = Workspace::new(&cfg, max_seq);
        let mut cache = KvCache::new(&cfg, max_seq);
        let mut sampler = Sampler::new(
            SamplerConfig {
                temperature: 0.9,
                top_k: 40,
                top_p: 0.95,
                repetition_penalty: 1.1,
                repetition_window: 64,
                seed,
            },
            cfg.vocab_size,
        );
        model.forward(&PROMPT, &mut ws, &mut cache, None).unwrap();
        for &t in &PROMPT {
            sampler.accept(t);
        }
        let mut out = Vec::new();
        for _ in 0..12 {
            let t = sampler.sample(model.logits(&ws));
            out.push(t);
            sampler.accept(t);
            model
                .forward_token(t, cache.len(), &mut ws, &mut cache, None)
                .unwrap();
        }
        out
    };

    assert_eq!(
        run(7),
        run(7),
        "the same seed must reproduce the generation"
    );
    assert_ne!(run(7), run(8), "a different seed should diverge");
}

// ============================================================== telemetry ==

/// Captured attention must be a real probability distribution over exactly the
/// positions the causal mask allows.
///
/// A heatmap is a plausible-looking picture no matter what is in the buffer, so
/// this is the only thing standing between "the visualisation is informative"
/// and "the visualisation is decorative".
#[test]
#[ignore = "slow: needs --release and the model file"]
fn captured_attention_is_a_causal_distribution() {
    let Some(g) = gguf() else {
        std::eprintln!("note: skipping -- needs {MODEL}");
        return;
    };
    let cfg = g.config().unwrap();
    let max_seq = 32;
    let weights = ModelWeights::from_gguf(g).unwrap();
    let model = Model::new(weights, max_seq, RopeKind::SplitHalf);
    let mut ws = Workspace::new(&cfg, max_seq);
    let mut cache = KvCache::new(&cfg, max_seq);

    ws.set_capture_attention(true);
    assert!(ws.captures_attention());
    model.forward(&PROMPT, &mut ws, &mut cache, None).unwrap();

    let len = PROMPT.len();
    for layer in 0..cfg.block_count {
        for head in 0..cfg.head_count {
            let a = ws.attention(layer, head, len);
            assert_eq!(a.len(), len);
            let sum: f32 = a.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "layer {layer} head {head}: sums to {sum}, not 1"
            );
            assert!(
                a.iter().all(|p| (0.0..=1.0).contains(p)),
                "layer {layer} head {head}: probability outside [0,1]"
            );
        }
    }

    // Position 0 can only attend to itself, whatever the layer or head.
    let mut ws0 = Workspace::new(&cfg, max_seq);
    let mut cache0 = KvCache::new(&cfg, max_seq);
    ws0.set_capture_attention(true);
    model
        .forward(&PROMPT[..1], &mut ws0, &mut cache0, None)
        .unwrap();
    for layer in 0..cfg.block_count {
        for head in 0..cfg.head_count {
            assert_eq!(
                ws0.attention(layer, head, 1),
                &[1.0],
                "layer {layer} head {head}"
            );
        }
    }
}

#[test]
#[ignore = "slow: needs --release and the model file"]
fn timings_are_off_until_a_clock_is_supplied() {
    let Some(g) = gguf() else { return };
    let cfg = g.config().unwrap();
    let weights = ModelWeights::from_gguf(g).unwrap();
    let model = Model::new(weights, 32, RopeKind::SplitHalf);
    let mut ws = Workspace::new(&cfg, 32);
    let mut cache = KvCache::new(&cfg, 32);

    // No clock: `core` has none of its own, so instrumentation costs nothing.
    model.forward(&PROMPT, &mut ws, &mut cache, None).unwrap();
    assert_eq!(ws.timings(), OpTimings::default());

    // With one, the buckets fill and add up to the whole step.
    fn clock() -> f64 {
        use std::sync::OnceLock;
        static T0: OnceLock<std::time::Instant> = OnceLock::new();
        T0.get_or_init(std::time::Instant::now)
            .elapsed()
            .as_secs_f64()
            * 1000.0
    }
    ws.set_clock(clock);
    cache.reset();
    model.forward(&PROMPT, &mut ws, &mut cache, None).unwrap();

    let t = ws.timings();
    std::eprintln!(
        "  matmul {:.2} ms · attention {:.2} ms · other {:.2} ms · total {:.2} ms",
        t.matmul_ms,
        t.attention_ms,
        t.other_ms,
        t.total_ms()
    );
    assert!(t.matmul_ms > 0.0, "matmul should dominate, got {t:?}");
    assert!(t.total_ms() > 0.0);
    // The matmuls are ~95% of a decode step; if they are not the biggest bucket,
    // the instrumentation is attributing time to the wrong place.
    assert!(
        t.matmul_ms > t.attention_ms && t.matmul_ms > t.other_ms,
        "matmul should be the largest bucket: {t:?}"
    );
}

// ======================================================= batched prefill ==

/// Batched prefill must equal per-token prefill **bit for bit**.
///
/// Not approximately. The batched matmuls unpack a weight row once and reuse it
/// across the chunk, while `forward_token` unpacks per token — different code,
/// and float multiplication is not associative, so agreeing exactly is a
/// property that had to be designed in rather than hoped for. If this ever
/// drifts, the two paths have started disagreeing about scale folding and every
/// downstream equivalence claim goes with it.
#[test]
#[ignore = "slow: needs --release and the model file"]
fn batched_prefill_matches_per_token_prefill() {
    let Some(g) = gguf() else {
        std::eprintln!("note: skipping -- needs {MODEL}");
        return;
    };
    let cfg = g.config().unwrap();
    let max_seq = 128;
    let weights = ModelWeights::from_gguf(g).unwrap();
    let model = Model::new(weights, max_seq, RopeKind::SplitHalf);

    // Longer than one chunk, so the chunk boundary is covered: a position in
    // chunk 2 must attend to positions in chunk 1.
    let long: Vec<u32> = (0..40).map(|i| PROMPT[i % PROMPT.len()]).collect();

    for tokens in [&PROMPT[..], &long[..]] {
        let mut ws_a = Workspace::new(&cfg, max_seq);
        let mut cache_a = KvCache::new(&cfg, max_seq);
        model
            .forward(tokens, &mut ws_a, &mut cache_a, None)
            .unwrap();

        let mut ws_b = Workspace::new(&cfg, max_seq);
        let mut cache_b = KvCache::new(&cfg, max_seq);
        model.prefill(tokens, &mut ws_b, &mut cache_b).unwrap();

        assert_eq!(cache_a.len(), cache_b.len(), "cache length differs");
        assert_eq!(
            ws_a.logits_vec(),
            ws_b.logits_vec(),
            "batched prefill diverged from per-token at {} tokens",
            tokens.len()
        );
    }
}

/// Prefill then decode must continue correctly across the handover.
///
/// The cache is written by the batched path and then read by the incremental
/// one; an off-by-one in `cache.len` after a chunk would only show up here.
#[test]
#[ignore = "slow: needs --release and the model file"]
fn decode_continues_correctly_after_batched_prefill() {
    let Some(g) = gguf() else { return };
    let cfg = g.config().unwrap();
    let max_seq = 128;
    let weights = ModelWeights::from_gguf(g).unwrap();
    let model = Model::new(weights, max_seq, RopeKind::SplitHalf);
    let full: Vec<u32> = PROMPT.iter().copied().chain([12095u32, 13, 1084]).collect();

    // Reference: everything through the per-token path.
    let mut ws_a = Workspace::new(&cfg, max_seq);
    let mut cache_a = KvCache::new(&cfg, max_seq);
    model.forward(&full, &mut ws_a, &mut cache_a, None).unwrap();

    // Batched prefill for the prompt, then per-token decode for the rest.
    let mut ws_b = Workspace::new(&cfg, max_seq);
    let mut cache_b = KvCache::new(&cfg, max_seq);
    model.prefill(&PROMPT, &mut ws_b, &mut cache_b).unwrap();
    for (i, &tok) in full[PROMPT.len()..].iter().enumerate() {
        model
            .forward_token(tok, PROMPT.len() + i, &mut ws_b, &mut cache_b, None)
            .unwrap();
    }

    assert_eq!(
        ws_a.logits_vec(),
        ws_b.logits_vec(),
        "handover changed the logits"
    );
}
