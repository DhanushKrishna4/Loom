//! The forward pass.
//!
//! Positions are processed one at a time, appending to the KV cache as they go,
//! so prefill and decode are the same code path. Step 7 splits them: prefill
//! becomes a batched matmul over the whole prompt, decode stays as this.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use super::weights::ModelWeights;
use crate::gguf::GgmlType;
use crate::gguf::ModelConfig;
use crate::math::sqrt;
use crate::ops::{self, RopeKind, RopeTable};
use crate::quant::{ActivationQ8, QuantError, UnpackedRow};
use crate::tensor::QuantMatrix;

/// How the KV cache orders its axes.
///
/// Both hold the same numbers; they differ only in which access pattern is
/// contiguous. The spec for this project says to pick by measurement rather than
/// intuition, so both exist and [`KvCache::new`] takes one -- see the benchmark
/// results in the README.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvLayout {
    /// `[layer][position][kv_head][head_dim]`.
    ///
    /// A whole projected K vector is one contiguous write. Reading one head's
    /// history then strides by `kv_dim`, touching `head_dim` floats out of every
    /// `kv_dim`.
    PositionMajor,
    /// `[layer][kv_head][position][head_dim]`.
    ///
    /// One head's history is contiguous, which is what the score loop wants.
    /// The write scatters into `n_kv_heads` separate places instead.
    HeadMajor,
}

/// Preallocated K and V for every layer and position.
///
/// Allocated once at load and never resized: a reallocation mid-generation
/// would invalidate every `Float32Array` view JS holds into wasm memory, which
/// is the failure mode the whole "preallocate everything" rule exists to avoid.
#[derive(Debug)]
pub struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    kv_dim: usize,
    layout: KvLayout,
    /// Number of positions written so far.
    len: usize,
}

impl KvCache {
    pub fn new(config: &ModelConfig, max_seq: usize) -> Self {
        Self::with_layout(config, max_seq, KvLayout::HeadMajor)
    }

    pub fn with_layout(config: &ModelConfig, max_seq: usize, layout: KvLayout) -> Self {
        let kv_dim = config.kv_dim();
        let n = config.block_count * max_seq * kv_dim;
        KvCache {
            k: vec![0.0; n],
            v: vec![0.0; n],
            n_layers: config.block_count,
            n_kv_heads: config.head_count_kv,
            head_dim: config.head_dim,
            max_seq,
            kv_dim,
            layout,
            len: 0,
        }
    }

    /// Positions currently held.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    pub fn layout(&self) -> KvLayout {
        self.layout
    }

    /// Fraction of the allocated cache in use, for the memory panel.
    pub fn utilisation(&self) -> f32 {
        self.len as f32 / self.max_seq as f32
    }

    /// Bytes held. Surfaced so the UI can show the cache filling up.
    pub fn byte_len(&self) -> usize {
        (self.k.len() + self.v.len()) * core::mem::size_of::<f32>()
    }

    /// Forget everything. No reallocation -- this is what starting a new
    /// conversation does, and it must not touch the allocator.
    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Where one head's history starts, and how far apart consecutive positions
    /// are.
    ///
    /// Both layouts reduce to `base + t * stride`, so the attention loop is
    /// identical for either and the layout branch is hoisted out of it. That
    /// also keeps the benchmark honest: the two are being compared on memory
    /// behaviour, not on differing amounts of index arithmetic.
    #[inline]
    fn head_span(&self, layer: usize, kv_head: usize) -> (usize, usize) {
        match self.layout {
            KvLayout::PositionMajor => (
                layer * self.max_seq * self.kv_dim + kv_head * self.head_dim,
                self.kv_dim,
            ),
            KvLayout::HeadMajor => (
                (layer * self.n_kv_heads + kv_head) * self.max_seq * self.head_dim,
                self.head_dim,
            ),
        }
    }

    fn write(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        match self.layout {
            KvLayout::PositionMajor => {
                // One contiguous run: the whole projected vector at once.
                let o = (layer * self.max_seq + pos) * self.kv_dim;
                self.k[o..o + self.kv_dim].copy_from_slice(k);
                self.v[o..o + self.kv_dim].copy_from_slice(v);
            }
            KvLayout::HeadMajor => {
                // Scattered: one run per KV head.
                for h in 0..self.n_kv_heads {
                    let (base, stride) = self.head_span(layer, h);
                    let o = base + pos * stride;
                    let src = h * self.head_dim..(h + 1) * self.head_dim;
                    self.k[o..o + self.head_dim].copy_from_slice(&k[src.clone()]);
                    self.v[o..o + self.head_dim].copy_from_slice(&v[src]);
                }
            }
        }
    }

    /// One KV head's key vector at a position.
    ///
    /// The attention loop uses [`Self::k_span`] instead, to hoist the layout
    /// branch; this is the readable form, for tests and for the attention
    /// visualiser.
    #[inline]
    pub fn key(&self, layer: usize, pos: usize, kv_head: usize) -> &[f32] {
        let (base, stride) = self.head_span(layer, kv_head);
        let o = base + pos * stride;
        &self.k[o..o + self.head_dim]
    }

    #[inline]
    pub fn value(&self, layer: usize, pos: usize, kv_head: usize) -> &[f32] {
        let (base, stride) = self.head_span(layer, kv_head);
        let o = base + pos * stride;
        &self.v[o..o + self.head_dim]
    }

    /// Write K and V for one position, for benchmarks that drive the cache
    /// directly rather than through a forward pass.
    pub fn fill_for_bench(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        self.write(layer, pos, k, v);
        self.len = self.len.max(pos + 1);
    }

    /// Raw K storage plus one head's `(base, stride)`, for the attention loop
    /// and for benchmarks that want to drive it directly.
    #[inline]
    pub fn k_span(&self, layer: usize, kv_head: usize) -> (&[f32], usize, usize) {
        let (base, stride) = self.head_span(layer, kv_head);
        (&self.k, base, stride)
    }

    #[inline]
    pub fn v_span(&self, layer: usize, kv_head: usize) -> (&[f32], usize, usize) {
        let (base, stride) = self.head_span(layer, kv_head);
        (&self.v, base, stride)
    }
}

/// Where a decode step's time went, in milliseconds.
///
/// Accumulated across all layers for one token. The split is coarse on purpose:
/// "which of these three should I optimise" is the question a perf panel needs
/// to answer, and finer buckets would cost more clock calls than they are worth.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct OpTimings {
    /// Every weight projection: Q/K/V/O, gate/up/down, and the unembedding.
    pub matmul_ms: f64,
    /// Scores, softmax, and the weighted sum over V.
    pub attention_ms: f64,
    /// RMSNorm, RoPE, SwiGLU, residual adds -- everything elementwise.
    pub other_ms: f64,
}

impl OpTimings {
    pub fn total_ms(&self) -> f64 {
        self.matmul_ms + self.attention_ms + self.other_ms
    }

    pub fn clear(&mut self) {
        *self = OpTimings::default();
    }
}

/// Positions processed together during prefill.
///
/// The whole prompt at once would be simpler, but the batch buffers scale with
/// it and a long prompt would then allocate mid-generation — which grows wasm
/// linear memory and detaches every `Float32Array` the page is holding. 32 keeps
/// the batch under 2 MiB while still amortising each weight-row unpack 32 ways.
pub const PREFILL_CHUNK: usize = 32;

/// Buffers for the batched prefill path, preallocated like everything else.
#[derive(Debug)]
struct PrefillBatch {
    /// `[t][d]` hidden states for the chunk.
    xs: Vec<f32>,
    xb: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    /// One quantised activation per position in the chunk.
    acts: Vec<ActivationQ8>,
    /// One unpacked weight row, reused across every row of every matmul.
    unpacked: UnpackedRow,
}

impl PrefillBatch {
    fn new(config: &ModelConfig) -> Self {
        let t = PREFILL_CHUNK;
        let d = config.embedding_length;
        let ff = config.feed_forward_length;
        let widest = d.max(ff).max(config.q_dim());
        PrefillBatch {
            xs: vec![0.0; t * d],
            xb: vec![0.0; t * d],
            q: vec![0.0; t * config.q_dim()],
            k: vec![0.0; t * config.kv_dim()],
            v: vec![0.0; t * config.kv_dim()],
            attn_out: vec![0.0; t * config.q_dim()],
            gate: vec![0.0; t * ff],
            up: vec![0.0; t * ff],
            down: vec![0.0; t * d],
            acts: (0..t).map(|_| ActivationQ8::new(widest)).collect(),
            unpacked: UnpackedRow::new(widest),
        }
    }

    fn byte_len(&self) -> usize {
        (self.xs.len()
            + self.xb.len()
            + self.q.len()
            + self.k.len()
            + self.v.len()
            + self.attn_out.len()
            + self.gate.len()
            + self.up.len()
            + self.down.len())
            * core::mem::size_of::<f32>()
            + self.acts.iter().map(ActivationQ8::byte_len).sum::<usize>()
            + self.unpacked.byte_len()
    }
}

/// Scratch buffers, allocated once and reused for every token.
#[derive(Debug)]
pub struct Workspace {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    attn_out: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    /// One dequantised weight row, for the unfused reference path.
    row: Vec<f32>,
    /// Post-softmax attention weights for the current token, laid out
    /// `[layer][head][position]`.
    ///
    /// Allocated unconditionally rather than on demand. It is ~1.4 MiB against a
    /// 24 MiB KV cache, and allocating it lazily would grow wasm linear memory
    /// mid-generation -- which detaches every `Float32Array` the page is holding.
    /// Paying 1.4 MiB to never do that is the right trade.
    attn: Vec<f32>,
    /// Whether to fill `attn`. Off by default: the copy is cheap but not free.
    capture_attention: bool,
    /// Wall clock, supplied by the host. `core` has none of its own.
    clock: Option<fn() -> f64>,
    timings: OpTimings,
    n_heads: usize,
    max_seq: usize,
    batch: PrefillBatch,
    /// The activation vector, quantised once per matmul and reused by every
    /// projection that shares it -- Q/K/V all read the same normed vector, as
    /// do gate and up, so one quantisation serves three matmuls.
    act: ActivationQ8,
    logits: Vec<f32>,
}

impl Workspace {
    pub fn new(config: &ModelConfig, max_seq: usize) -> Self {
        let d = config.embedding_length;
        let widest_row = d.max(config.feed_forward_length);
        Workspace {
            x: vec![0.0; d],
            xb: vec![0.0; d],
            xb2: vec![0.0; d],
            q: vec![0.0; config.q_dim()],
            k: vec![0.0; config.kv_dim()],
            v: vec![0.0; config.kv_dim()],
            att: vec![0.0; max_seq],
            attn_out: vec![0.0; config.q_dim()],
            gate: vec![0.0; config.feed_forward_length],
            up: vec![0.0; config.feed_forward_length],
            row: vec![0.0; widest_row],
            act: ActivationQ8::new(widest_row.max(config.q_dim())),
            attn: vec![0.0; config.block_count * config.head_count * max_seq],
            capture_attention: false,
            clock: None,
            timings: OpTimings::default(),
            n_heads: config.head_count,
            max_seq,
            batch: PrefillBatch::new(config),
            logits: vec![0.0; config.vocab_size],
        }
    }

    /// A copy of the current logits.
    pub fn logits_vec(&self) -> Vec<f32> {
        self.logits.clone()
    }

    /// Record post-softmax attention weights for every layer and head.
    pub fn set_capture_attention(&mut self, on: bool) {
        self.capture_attention = on;
    }

    pub fn captures_attention(&self) -> bool {
        self.capture_attention
    }

    /// Supply a millisecond clock so the forward pass can time itself.
    ///
    /// `core` deliberately has no clock -- it is `no_std` and knows nothing about
    /// the host -- so instrumentation is opt-in and costs nothing when absent.
    pub fn set_clock(&mut self, clock: fn() -> f64) {
        self.clock = Some(clock);
    }

    pub fn timings(&self) -> OpTimings {
        self.timings
    }

    /// One head's attention distribution over positions `0..=pos`.
    pub fn attention(&self, layer: usize, head: usize, len: usize) -> &[f32] {
        let base = (layer * self.n_heads + head) * self.max_seq;
        &self.attn[base..base + len.min(self.max_seq)]
    }

    /// The whole `[layer][head][position]` block, for a zero-copy view.
    pub fn attention_all(&self) -> &[f32] {
        &self.attn
    }

    #[inline]
    fn tick(&self) -> f64 {
        match self.clock {
            Some(f) => f(),
            None => 0.0,
        }
    }

    pub fn byte_len(&self) -> usize {
        (self.x.len()
            + self.xb.len()
            + self.xb2.len()
            + self.q.len()
            + self.k.len()
            + self.v.len()
            + self.att.len()
            + self.attn_out.len()
            + self.gate.len()
            + self.up.len()
            + self.row.len()
            + self.attn.len()
            + self.logits.len())
            * core::mem::size_of::<f32>()
            + self.act.byte_len()
            + self.batch.byte_len()
    }
}

/// Captured intermediate activations, for comparing against a reference.
///
/// Layer-by-layer capture is the whole reason a numerical bug in this file is
/// findable at all: a wrong sign in layer 3 is obvious here and essentially
/// undiagnosable from final logits.
#[derive(Debug, Default)]
pub struct Trace {
    pub records: Vec<(String, Vec<f32>)>,
}

impl Trace {
    pub fn record(&mut self, name: &str, data: &[f32]) {
        self.records.push((String::from(name), data.to_vec()));
    }

    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.records
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_slice())
    }
}

/// Run one projection, fused or not.
///
/// The fused path quantises nothing here -- `a` is already prepared -- so the
/// only cost of the choice is this branch, once per matmul.
#[inline]
fn project(
    fused: bool,
    w: &QuantMatrix<'_>,
    x: &[f32],
    a: &ActivationQ8,
    out: &mut [f32],
    row: &mut [f32],
) -> Result<(), QuantError> {
    if fused && w.ggml_type() != GgmlType::F32 {
        ops::matvec_fused(w, a, out)
    } else {
        ops::matvec_dequant(w, x, out, row)
    }
}

/// A loaded model, ready to run.
#[derive(Debug)]
pub struct Model<'a> {
    pub weights: ModelWeights<'a>,
    pub rope: RopeTable,
    /// Use the fused quantised kernels. On by default; turning it off selects
    /// the dequantise-then-dot reference path, which is what the fused kernels
    /// are tested against.
    fused: bool,
    /// Use the batched unpack-once kernels for prefill. Bit-identical either
    /// way; see [`Self::set_batched_prefill`].
    batched_prefill: bool,
}

impl<'a> Model<'a> {
    /// Build the RoPE table for a maximum sequence length.
    ///
    /// `max_seq` rather than `context_length`: the full 32768-position table is
    /// 8 MiB and the KV cache for that context is 768 MiB, so nothing runs at
    /// full context in a browser anyway.
    pub fn new(weights: ModelWeights<'a>, max_seq: usize, rope_kind: RopeKind) -> Self {
        let c = &weights.config;
        let rope_dim = c.rope_dimension_count.unwrap_or(c.head_dim);
        let rope = RopeTable::new(c.head_dim, rope_dim, max_seq, c.rope_freq_base, rope_kind);
        Model {
            weights,
            rope,
            fused: true,
            // Chosen per target: see the prefill table in the README. On wasm the
            // batched path wins; on native it does not, and native is the dev
            // tool rather than the product.
            batched_prefill: cfg!(target_arch = "wasm32"),
        }
    }

    /// Switch between the fused kernels and the dequantising reference path.
    pub fn set_fused(&mut self, fused: bool) {
        self.fused = fused;
    }

    /// Use the batched (unpack-once) kernels for prefill.
    ///
    /// Both paths produce bit-identical results, so this is purely a speed
    /// choice — and which one wins depends on the target. See the README's
    /// prefill table.
    pub fn set_batched_prefill(&mut self, on: bool) {
        self.batched_prefill = on;
    }

    pub fn is_batched_prefill(&self) -> bool {
        self.batched_prefill
    }

    pub fn is_fused(&self) -> bool {
        self.fused
    }

    pub fn config(&self) -> &ModelConfig {
        &self.weights.config
    }

    /// Run one token at `pos`, appending its K and V to the cache.
    ///
    /// Returns the logits for that position.
    pub fn forward_token(
        &self,
        token: u32,
        pos: usize,
        ws: &mut Workspace,
        cache: &mut KvCache,
        mut trace: Option<&mut Trace>,
    ) -> Result<(), QuantError> {
        let c = &self.weights.config;
        let head_dim = c.head_dim;
        let n_heads = c.head_count;
        let n_kv = c.head_count_kv;
        let group = c.kv_group_size();
        let scale = 1.0 / sqrt(head_dim as f32);
        // Zero cost when no clock was supplied: `tick` returns 0.0 and the
        // subtractions below all come out zero.
        ws.timings.clear();
        let mut t = ws.tick();

        // --- embedding -------------------------------------------------
        self.weights.token_embd.dequant_row(token, &mut ws.x)?;
        if let Some(t) = trace.as_deref_mut() {
            t.record("hidden.0", &ws.x);
        }

        {
            let now = ws.tick();
            ws.timings.matmul_ms += now - t;
            t = now;
        }

        for (li, layer) in self.weights.layers.iter().enumerate() {
            // --- attention ---------------------------------------------
            ops::rmsnorm(&ws.x, &layer.attn_norm, &mut ws.xb, c.rms_norm_eps);
            if let Some(t) = trace.as_deref_mut() {
                t.record(&format!("l{li}.attn_norm"), &ws.xb);
            }

            // One quantisation, three matmuls: Q, K and V all read this same
            // normed vector.
            ws.act.quantize(&ws.xb);
            {
                let now = ws.tick();
                ws.timings.other_ms += now - t;
                t = now;
            }
            project(
                self.fused,
                &layer.attn_q,
                &ws.xb,
                &ws.act,
                &mut ws.q,
                &mut ws.row,
            )?;
            project(
                self.fused,
                &layer.attn_k,
                &ws.xb,
                &ws.act,
                &mut ws.k,
                &mut ws.row,
            )?;
            project(
                self.fused,
                &layer.attn_v,
                &ws.xb,
                &ws.act,
                &mut ws.v,
                &mut ws.row,
            )?;

            {
                let now = ws.tick();
                ws.timings.matmul_ms += now - t;
                t = now;
            }

            // Qwen2's attention biases. Absent for Llama-family models.
            if let Some(b) = &layer.attn_q_bias {
                ops::add_assign(&mut ws.q, b);
            }
            if let Some(b) = &layer.attn_k_bias {
                ops::add_assign(&mut ws.k, b);
            }
            if let Some(b) = &layer.attn_v_bias {
                ops::add_assign(&mut ws.v, b);
            }

            // RoPE on Q and K. Never on V: it carries no positional meaning and
            // rotating it is silently wrong.
            self.rope.apply_projection(&mut ws.q, n_heads, pos);
            self.rope.apply_projection(&mut ws.k, n_kv, pos);

            cache.write(li, pos, &ws.k, &ws.v);
            {
                let now = ws.tick();
                ws.timings.other_ms += now - t;
                t = now;
            }

            for h in 0..n_heads {
                // GQA: consecutive query heads share a KV head. Integer
                // division, not modulo -- getting this backwards interleaves
                // the groups and produces confident nonsense.
                let kv_head = h / group;
                let q_head = &ws.q[h * head_dim..(h + 1) * head_dim];

                // Causal: attend to 0..=pos only, which is implicit in the
                // bound rather than applied as a -inf mask.
                //
                // The layout branch is hoisted here, once per head, so the loops
                // below are the same instructions for either layout.
                let att = &mut ws.att[..=pos];
                let (kbuf, kbase, kstride) = cache.k_span(li, kv_head);
                for (t, a) in att.iter_mut().enumerate() {
                    let o = kbase + t * kstride;
                    *a = ops::dot(q_head, &kbuf[o..o + head_dim]) * scale;
                }
                ops::softmax(att);

                // Snapshot the distribution before it is consumed. One copy of
                // `pos+1` floats per head, only when the UI asked for it.
                if ws.capture_attention {
                    let base = (li * n_heads + h) * ws.max_seq;
                    ws.attn[base..base + att.len()].copy_from_slice(att);
                }

                let out = &mut ws.attn_out[h * head_dim..(h + 1) * head_dim];
                out.fill(0.0);
                let (vbuf, vbase, vstride) = cache.v_span(li, kv_head);
                for (t, &p) in att.iter().enumerate() {
                    let o = vbase + t * vstride;
                    for (dst, vi) in out.iter_mut().zip(&vbuf[o..o + head_dim]) {
                        *dst += p * vi;
                    }
                }
            }

            {
                let now = ws.tick();
                ws.timings.attention_ms += now - t;
                t = now;
            }

            ws.act.quantize(&ws.attn_out);
            project(
                self.fused,
                &layer.attn_output,
                &ws.attn_out,
                &ws.act,
                &mut ws.xb2,
                &mut ws.row,
            )?;
            ops::add_assign(&mut ws.x, &ws.xb2);
            {
                let now = ws.tick();
                ws.timings.matmul_ms += now - t;
                t = now;
            }

            // --- feed-forward ------------------------------------------
            ops::rmsnorm(&ws.x, &layer.ffn_norm, &mut ws.xb, c.rms_norm_eps);
            if let Some(t) = trace.as_deref_mut() {
                t.record(&format!("l{li}.ffn_norm"), &ws.xb);
            }

            // Gate and up share the normed vector too.
            ws.act.quantize(&ws.xb);
            {
                let now = ws.tick();
                ws.timings.other_ms += now - t;
                t = now;
            }
            project(
                self.fused,
                &layer.ffn_gate,
                &ws.xb,
                &ws.act,
                &mut ws.gate,
                &mut ws.row,
            )?;
            project(
                self.fused,
                &layer.ffn_up,
                &ws.xb,
                &ws.act,
                &mut ws.up,
                &mut ws.row,
            )?;
            {
                let now = ws.tick();
                ws.timings.matmul_ms += now - t;
                t = now;
            }
            ops::swiglu(&mut ws.gate, &ws.up);
            ws.act.quantize(&ws.gate);
            {
                let now = ws.tick();
                ws.timings.other_ms += now - t;
                t = now;
            }
            project(
                self.fused,
                &layer.ffn_down,
                &ws.gate,
                &ws.act,
                &mut ws.xb2,
                &mut ws.row,
            )?;
            ops::add_assign(&mut ws.x, &ws.xb2);
            {
                let now = ws.tick();
                ws.timings.matmul_ms += now - t;
                t = now;
            }

            if let Some(t) = trace.as_deref_mut() {
                t.record(&format!("hidden.{}", li + 1), &ws.x);
            }
        }

        cache.len = cache.len.max(pos + 1);

        // --- unembedding ---------------------------------------------
        ops::rmsnorm(&ws.x, &self.weights.output_norm, &mut ws.xb, c.rms_norm_eps);
        if let Some(t) = trace.as_deref_mut() {
            t.record("final_norm", &ws.xb);
        }
        {
            let now = ws.tick();
            ws.timings.other_ms += now - t;
            t = now;
        }
        ws.act.quantize(&ws.xb);
        project(
            self.fused,
            &self.weights.output,
            &ws.xb,
            &ws.act,
            &mut ws.logits,
            &mut ws.row,
        )?;
        ws.timings.matmul_ms += ws.tick() - t;

        // Last use of `trace`, so no reborrow is needed here.
        if let Some(t) = trace {
            t.record("logits", &ws.logits);
        }
        Ok(())
    }

    /// Prefill: process a chunk of positions with batched matmuls.
    ///
    /// Every weight row is read and unpacked **once per chunk** instead of once
    /// per token, which is the difference between prefill costing 463 MB of
    /// memory traffic per token and 463 MB per 32 tokens.
    ///
    /// Only the matmuls batch. Attention still runs per position, because each
    /// query attends to a different prefix and there is no shared work to hoist
    /// — a batched causal attention would be a different kernel, not this one
    /// with a loop moved.
    ///
    /// Logits are produced only for the final position of the final chunk;
    /// nothing else needs them, and the unembedding is the single most expensive
    /// matmul in the model.
    fn prefill_chunk(
        &self,
        tokens: &[u32],
        start_pos: usize,
        ws: &mut Workspace,
        cache: &mut KvCache,
        want_logits: bool,
    ) -> Result<(), QuantError> {
        let c = &self.weights.config;
        let d = c.embedding_length;
        let ff = c.feed_forward_length;
        let q_dim = c.q_dim();
        let kv_dim = c.kv_dim();
        let head_dim = c.head_dim;
        let n_heads = c.head_count;
        let n_kv = c.head_count_kv;
        let group = c.kv_group_size();
        let scale = 1.0 / sqrt(head_dim as f32);
        let t = tokens.len();
        debug_assert!(t <= PREFILL_CHUNK);

        // --- embedding ---------------------------------------------------
        for (i, &tok) in tokens.iter().enumerate() {
            self.weights
                .token_embd
                .dequant_row(tok, &mut ws.batch.xs[i * d..(i + 1) * d])?;
        }

        for (li, layer) in self.weights.layers.iter().enumerate() {
            // --- attention block -----------------------------------------
            for i in 0..t {
                ops::rmsnorm(
                    &ws.batch.xs[i * d..(i + 1) * d],
                    &layer.attn_norm,
                    &mut ws.batch.xb[i * d..(i + 1) * d],
                    c.rms_norm_eps,
                );
                ws.batch.acts[i].quantize(&ws.batch.xb[i * d..(i + 1) * d]);
            }
            let acts = &ws.batch.acts[..t];

            ops::matmul_fused(
                &layer.attn_q,
                acts,
                &mut ws.batch.q[..t * q_dim],
                &mut ws.batch.unpacked,
            )?;
            ops::matmul_fused(
                &layer.attn_k,
                acts,
                &mut ws.batch.k[..t * kv_dim],
                &mut ws.batch.unpacked,
            )?;
            ops::matmul_fused(
                &layer.attn_v,
                acts,
                &mut ws.batch.v[..t * kv_dim],
                &mut ws.batch.unpacked,
            )?;

            for i in 0..t {
                let q = &mut ws.batch.q[i * q_dim..(i + 1) * q_dim];
                if let Some(b) = &layer.attn_q_bias {
                    ops::add_assign(q, b);
                }
                self.rope.apply_projection(q, n_heads, start_pos + i);

                let k = &mut ws.batch.k[i * kv_dim..(i + 1) * kv_dim];
                if let Some(b) = &layer.attn_k_bias {
                    ops::add_assign(k, b);
                }
                self.rope.apply_projection(k, n_kv, start_pos + i);

                let v = &mut ws.batch.v[i * kv_dim..(i + 1) * kv_dim];
                if let Some(b) = &layer.attn_v_bias {
                    ops::add_assign(v, b);
                }
            }

            // Every position's K and V must be in the cache before any of them
            // attends, or a later position in this chunk cannot see an earlier
            // one.
            for i in 0..t {
                cache.write(
                    li,
                    start_pos + i,
                    &ws.batch.k[i * kv_dim..(i + 1) * kv_dim],
                    &ws.batch.v[i * kv_dim..(i + 1) * kv_dim],
                );
            }
            cache.len = cache.len.max(start_pos + t);

            for i in 0..t {
                let pos = start_pos + i;
                for h in 0..n_heads {
                    let kv_head = h / group;
                    let q_head =
                        &ws.batch.q[i * q_dim + h * head_dim..i * q_dim + (h + 1) * head_dim];

                    let att = &mut ws.att[..=pos];
                    let (kbuf, kbase, kstride) = cache.k_span(li, kv_head);
                    for (tt, a) in att.iter_mut().enumerate() {
                        let o = kbase + tt * kstride;
                        *a = ops::dot(q_head, &kbuf[o..o + head_dim]) * scale;
                    }
                    ops::softmax(att);

                    if ws.capture_attention {
                        let base = (li * n_heads + h) * ws.max_seq;
                        ws.attn[base..base + att.len()].copy_from_slice(att);
                    }

                    let out = &mut ws.batch.attn_out
                        [i * q_dim + h * head_dim..i * q_dim + (h + 1) * head_dim];
                    out.fill(0.0);
                    let (vbuf, vbase, vstride) = cache.v_span(li, kv_head);
                    for (tt, &p) in att.iter().enumerate() {
                        let o = vbase + tt * vstride;
                        for (dst, vi) in out.iter_mut().zip(&vbuf[o..o + head_dim]) {
                            *dst += p * vi;
                        }
                    }
                }
            }

            for i in 0..t {
                ws.batch.acts[i].quantize(&ws.batch.attn_out[i * q_dim..(i + 1) * q_dim]);
            }
            ops::matmul_fused(
                &layer.attn_output,
                &ws.batch.acts[..t],
                &mut ws.batch.down[..t * d],
                &mut ws.batch.unpacked,
            )?;
            for i in 0..t * d {
                ws.batch.xs[i] += ws.batch.down[i];
            }

            // --- feed-forward --------------------------------------------
            for i in 0..t {
                ops::rmsnorm(
                    &ws.batch.xs[i * d..(i + 1) * d],
                    &layer.ffn_norm,
                    &mut ws.batch.xb[i * d..(i + 1) * d],
                    c.rms_norm_eps,
                );
                ws.batch.acts[i].quantize(&ws.batch.xb[i * d..(i + 1) * d]);
            }
            let acts = &ws.batch.acts[..t];
            ops::matmul_fused(
                &layer.ffn_gate,
                acts,
                &mut ws.batch.gate[..t * ff],
                &mut ws.batch.unpacked,
            )?;
            ops::matmul_fused(
                &layer.ffn_up,
                acts,
                &mut ws.batch.up[..t * ff],
                &mut ws.batch.unpacked,
            )?;

            for i in 0..t {
                let (g, u) = (i * ff, (i + 1) * ff);
                // Disjoint fields, so the borrow checker allows both at once.
                ops::swiglu(&mut ws.batch.gate[g..u], &ws.batch.up[g..u]);
                ws.batch.acts[i].quantize(&ws.batch.gate[g..u]);
            }
            ops::matmul_fused(
                &layer.ffn_down,
                &ws.batch.acts[..t],
                &mut ws.batch.down[..t * d],
                &mut ws.batch.unpacked,
            )?;
            for i in 0..t * d {
                ws.batch.xs[i] += ws.batch.down[i];
            }
        }

        if want_logits {
            let last = (t - 1) * d;
            ws.x.copy_from_slice(&ws.batch.xs[last..last + d]);
            ops::rmsnorm(&ws.x, &self.weights.output_norm, &mut ws.xb, c.rms_norm_eps);
            ws.act.quantize(&ws.xb);
            project(
                self.fused,
                &self.weights.output,
                &ws.xb,
                &ws.act,
                &mut ws.logits,
                &mut ws.row,
            )?;
        }
        Ok(())
    }

    /// Prefill a whole prompt, in chunks.
    ///
    /// Bit-identical to calling [`Self::forward_token`] for each position — the
    /// batched kernels fold their scales in exactly the same order as the fused
    /// ones, which is asserted in `quant`'s tests and again end to end in
    /// `model`'s.
    pub fn prefill(
        &self,
        tokens: &[u32],
        ws: &mut Workspace,
        cache: &mut KvCache,
    ) -> Result<(), QuantError> {
        if !self.batched_prefill {
            return self.forward(tokens, ws, cache, None);
        }
        let start = cache.len();
        for (ci, chunk) in tokens.chunks(PREFILL_CHUNK).enumerate() {
            let last_chunk = (ci + 1) * PREFILL_CHUNK >= tokens.len();
            self.prefill_chunk(chunk, start + ci * PREFILL_CHUNK, ws, cache, last_chunk)?;
        }
        Ok(())
    }

    /// Run a whole sequence from position `cache.len()`, returning the logits
    /// for the final token.
    pub fn forward(
        &self,
        tokens: &[u32],
        ws: &mut Workspace,
        cache: &mut KvCache,
        mut trace: Option<&mut Trace>,
    ) -> Result<(), QuantError> {
        let start = cache.len();
        for (i, &tok) in tokens.iter().enumerate() {
            // Only the last position's activations are traced; tracing every
            // position of a long prompt would be gigabytes.
            let t = if i + 1 == tokens.len() {
                trace.as_deref_mut()
            } else {
                None
            };
            self.forward_token(tok, start + i, ws, cache, t)?;
        }
        Ok(())
    }

    /// Logits from the most recent [`Self::forward_token`].
    pub fn logits<'w>(&self, ws: &'w Workspace) -> &'w [f32] {
        &ws.logits
    }

    /// Greedy next token.
    pub fn argmax(&self, ws: &Workspace) -> u32 {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in ws.logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        best as u32
    }
}
