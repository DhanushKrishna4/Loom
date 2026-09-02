//! The wasm-bindgen boundary.
//!
//! The rule for this crate: it marshals data and nothing else. If a function in
//! here is doing arithmetic, it belongs in `nano-infer-core`, where it can be
//! tested natively against PyTorch instead of through devtools.
//!
//! # Generation is driven from JS, one step at a time
//!
//! There is deliberately no `generate()` that loops internally. wasm runs on the
//! thread that called it, so a loop in here freezes the tab for the entire
//! generation -- no repaint, no input, no cancel button. Instead JS calls
//! [`Engine::decode_step`] once per frame and stays in control of the schedule.
//!
//! # Lifetimes
//!
//! The parsed model borrows the file buffer, so [`Engine`] would be
//! self-referential. Rather than reach for `unsafe` or a self-ref crate, the
//! buffer is leaked into `&'static [u8]` at load. The consequence, stated plainly:
//! **loading a second model leaks the first.** For a page that loads one model
//! and keeps it, that is the right trade; switching models means reloading the
//! page. `Engine::reset()` is for starting a new conversation, and does not
//! reallocate anything.

use core::cell::RefCell;

use wasm_bindgen::prelude::*;

use nano_infer_core::gguf::{Gguf, ModelConfig};
use nano_infer_core::model::{KvCache, Model, ModelWeights, Workspace};
use nano_infer_core::ops::RopeKind;
use nano_infer_core::quant;
use nano_infer_core::sample::{Sampler, SamplerConfig};
use nano_infer_core::tokenizer::{apply_chat_template, ChatMessage, EncodeOptions, Tokenizer};

extern crate alloc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Call once at startup so Rust panics surface as readable console errors
/// instead of `unreachable executed`.
#[wasm_bindgen(start)]
pub fn start() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

// Resolved once, then called directly. See `now_ms`.
thread_local! {
    static PERF: RefCell<Option<(JsValue, js_sys::Function)>> = const { RefCell::new(None) };
}

/// `performance.now()`, resolved once and then called directly.
///
/// Reached through the global object rather than `window` so this works in a
/// worker too. The lookup is cached because the forward pass calls this ~150
/// times per token for its per-op timings: doing four `Reflect` lookups and a
/// string allocation on every one of those would be instrumentation that
/// changes what it measures.
fn now_ms() -> f64 {
    PERF.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            let global = js_sys::global();
            let perf = js_sys::Reflect::get(&global, &JsValue::from_str("performance")).ok();
            if let Some(p) = perf.filter(|v| !v.is_undefined()) {
                if let Some(f) = js_sys::Reflect::get(&p, &JsValue::from_str("now"))
                    .ok()
                    .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                {
                    *slot = Some((p, f));
                }
            }
        }
        match slot.as_ref() {
            Some((p, f)) => f.call0(p).ok().and_then(|v| v.as_f64()).unwrap_or(0.0),
            None => 0.0,
        }
    })
}

/// One decode step's result.
#[wasm_bindgen]
pub struct DecodeResult {
    token: u32,
    text: String,
    is_eos: bool,
    ms: f64,
    matmul_ms: f64,
    attention_ms: f64,
    other_ms: f64,
}

#[wasm_bindgen]
impl DecodeResult {
    #[wasm_bindgen(getter)]
    pub fn token(&self) -> u32 {
        self.token
    }

    /// The text this step added.
    ///
    /// Often empty, and that is correct rather than a bug: a token can be half a
    /// UTF-8 sequence (any emoji spans several), so bytes are buffered until they
    /// form valid text. Appending this verbatim each step yields the right string.
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.text.clone()
    }

    #[wasm_bindgen(getter, js_name = isEos)]
    pub fn is_eos(&self) -> bool {
        self.is_eos
    }

    /// Total wall time for the step, including sampling and detokenisation.
    #[wasm_bindgen(getter)]
    pub fn ms(&self) -> f64 {
        self.ms
    }

    /// Time inside the weight projections. Expect ~95% of the step.
    #[wasm_bindgen(getter, js_name = matmulMs)]
    pub fn matmul_ms(&self) -> f64 {
        self.matmul_ms
    }

    /// Scores, softmax and the weighted sum over V.
    #[wasm_bindgen(getter, js_name = attentionMs)]
    pub fn attention_ms(&self) -> f64 {
        self.attention_ms
    }

    /// RMSNorm, RoPE, SwiGLU, residuals -- everything elementwise.
    #[wasm_bindgen(getter, js_name = otherMs)]
    pub fn other_ms(&self) -> f64 {
        self.other_ms
    }

    /// Whatever the forward pass did not account for: sampling, detokenisation,
    /// and the boundary crossing itself.
    #[wasm_bindgen(getter, js_name = overheadMs)]
    pub fn overhead_ms(&self) -> f64 {
        (self.ms - self.matmul_ms - self.attention_ms - self.other_ms).max(0.0)
    }
}

/// Counters for the perf panel.
#[wasm_bindgen]
pub struct Stats {
    prefill_tokens: u32,
    prefill_ms: f64,
    decode_tokens: u32,
    decode_ms: f64,
    first_token_ms: f64,
    rolling_tps: f64,
    cache_used: u32,
    cache_capacity: u32,
    cache_bytes: f64,
    workspace_bytes: f64,
    weight_bytes: f64,
}

#[wasm_bindgen]
impl Stats {
    #[wasm_bindgen(getter, js_name = prefillTokens)]
    pub fn prefill_tokens(&self) -> u32 {
        self.prefill_tokens
    }
    #[wasm_bindgen(getter, js_name = prefillMs)]
    pub fn prefill_ms(&self) -> f64 {
        self.prefill_ms
    }
    #[wasm_bindgen(getter, js_name = decodeTokens)]
    pub fn decode_tokens(&self) -> u32 {
        self.decode_tokens
    }
    #[wasm_bindgen(getter, js_name = decodeMs)]
    pub fn decode_ms(&self) -> f64 {
        self.decode_ms
    }
    /// Throughput over the last few tokens.
    ///
    /// A cumulative average is the wrong number for a live readout: it is
    /// dominated by however the generation started and barely moves once a few
    /// dozen tokens are in, so a real slowdown — a longer context, the tab going
    /// to the background — takes far too long to show up. This is the one the UI
    /// displays; [`Self::average_tokens_per_second`] is the whole-run figure.
    #[wasm_bindgen(getter, js_name = tokensPerSecond)]
    pub fn tokens_per_second(&self) -> f64 {
        self.rolling_tps
    }

    /// Throughput over the whole generation. The right number for a benchmark,
    /// the wrong one for a live display.
    #[wasm_bindgen(getter, js_name = averageTokensPerSecond)]
    pub fn average_tokens_per_second(&self) -> f64 {
        if self.decode_ms <= 0.0 {
            0.0
        } else {
            self.decode_tokens as f64 * 1000.0 / self.decode_ms
        }
    }
    /// Time to first token: prefill plus the first decode step. This is what a
    /// user waits through before anything appears, and it is not `prefillMs`:
    /// the first decode step is paid before the first character shows up.
    /// Zero until that token has been produced.
    #[wasm_bindgen(getter, js_name = timeToFirstToken)]
    pub fn time_to_first_token(&self) -> f64 {
        self.first_token_ms
    }
    #[wasm_bindgen(getter, js_name = cacheUsed)]
    pub fn cache_used(&self) -> u32 {
        self.cache_used
    }
    #[wasm_bindgen(getter, js_name = cacheCapacity)]
    pub fn cache_capacity(&self) -> u32 {
        self.cache_capacity
    }
    #[wasm_bindgen(getter, js_name = cacheBytes)]
    pub fn cache_bytes(&self) -> f64 {
        self.cache_bytes
    }
    #[wasm_bindgen(getter, js_name = workspaceBytes)]
    pub fn workspace_bytes(&self) -> f64 {
        self.workspace_bytes
    }
    #[wasm_bindgen(getter, js_name = weightBytes)]
    pub fn weight_bytes(&self) -> f64 {
        self.weight_bytes
    }
}

/// One tensor's identity, for the explorer's picker.
#[wasm_bindgen]
pub struct TensorSummary {
    name: String,
    format: String,
    rows: u32,
    cols: u32,
    blocks: u32,
    bytes: f64,
    bits_per_weight: f64,
    inspectable: bool,
}

#[wasm_bindgen]
impl TensorSummary {
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn format(&self) -> String {
        self.format.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> u32 {
        self.rows
    }
    #[wasm_bindgen(getter)]
    pub fn cols(&self) -> u32 {
        self.cols
    }
    #[wasm_bindgen(getter)]
    pub fn blocks(&self) -> u32 {
        self.blocks
    }
    #[wasm_bindgen(getter)]
    pub fn bytes(&self) -> f64 {
        self.bytes
    }
    #[wasm_bindgen(getter, js_name = bitsPerWeight)]
    pub fn bits_per_weight(&self) -> f64 {
        self.bits_per_weight
    }
    /// False for formats we can size but not decode, so the UI can grey them out
    /// instead of offering a view that would fail.
    #[wasm_bindgen(getter)]
    pub fn inspectable(&self) -> bool {
        self.inspectable
    }
}

/// One block taken apart: `value[i] = scale[i/group] * quant[i] - min[i/group]`.
#[wasm_bindgen]
pub struct BlockView {
    quants: Vec<i32>,
    values: Vec<f32>,
    scales: Vec<f32>,
    mins: Vec<f32>,
    group: u32,
    quant_min: i32,
    quant_max: i32,
}

#[wasm_bindgen]
impl BlockView {
    /// The stored integer per element, with any bias folded in.
    #[wasm_bindgen(getter)]
    pub fn quants(&self) -> Vec<i32> {
        self.quants.clone()
    }
    /// The reconstructed weights.
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<f32> {
        self.values.clone()
    }
    /// Multiplier per scale group.
    #[wasm_bindgen(getter)]
    pub fn scales(&self) -> Vec<f32> {
        self.scales.clone()
    }
    /// Offset per group, subtracted. All zero except for Q4_K, which is affine.
    #[wasm_bindgen(getter)]
    pub fn mins(&self) -> Vec<f32> {
        self.mins.clone()
    }
    /// Elements covered by one scale: 32 usually, 16 for Q6_K.
    #[wasm_bindgen(getter)]
    pub fn group(&self) -> u32 {
        self.group
    }
    #[wasm_bindgen(getter, js_name = quantMin)]
    pub fn quant_min(&self) -> i32 {
        self.quant_min
    }
    #[wasm_bindgen(getter, js_name = quantMax)]
    pub fn quant_max(&self) -> i32 {
        self.quant_max
    }
}

/// Everything the page needs, behind one object.
#[wasm_bindgen]
pub struct Engine {
    gguf: &'static Gguf<'static>,
    model: Model<'static>,
    tokenizer: Tokenizer<'static>,
    ws: Workspace,
    cache: KvCache,
    sampler: Sampler,
    config: ModelConfig,
    /// Bytes of a partially-decoded UTF-8 sequence, carried between steps.
    pending: Vec<u8>,
    prefill_tokens: u32,
    prefill_ms: f64,
    decode_tokens: u32,
    decode_ms: f64,
    /// Latched on the first decode step after a reset and never overwritten:
    /// the point is the *first* token, so later steps must not move it.
    first_token_ms: f64,
    /// Durations of the most recent decode steps, as a ring buffer.
    recent_ms: [f64; ROLLING_WINDOW],
    recent_at: usize,
    recent_len: usize,
    weight_bytes: f64,
}

/// How many recent tokens the live throughput figure averages over.
///
/// Sixteen is about a second of generation at the speeds this engine runs at:
/// long enough that one slow step does not make the number jump, short enough
/// that a real change shows up while you are still looking at it.
const ROLLING_WINDOW: usize = 16;

#[wasm_bindgen]
impl Engine {
    /// Parse a GGUF and build everything. `max_seq` bounds the KV cache and the
    /// RoPE table, both of which are allocated here and never resized.
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: Vec<u8>, max_seq: usize) -> Result<Engine, JsError> {
        // `Vec<u8>` rather than `&[u8]`: wasm-bindgen copies a slice argument
        // into wasm memory and then `.to_vec()` would copy it *again*. At 469 MB
        // that second copy is the difference between loading and an
        // out-of-memory. Taking the Vec by value moves it straight through.
        //
        // See the module docs: leaked so the borrows can be 'static.
        let bytes: &'static [u8] = alloc::boxed::Box::leak(model_bytes.into_boxed_slice());
        let gguf: &'static Gguf<'static> = alloc::boxed::Box::leak(alloc::boxed::Box::new(
            Gguf::parse(bytes).map_err(|e| JsError::new(&e.to_string()))?,
        ));

        let config = gguf.config().map_err(|e| JsError::new(&e.to_string()))?;
        let tokenizer = Tokenizer::from_gguf(gguf).map_err(|e| JsError::new(&e.to_string()))?;
        let weights = ModelWeights::from_gguf(gguf).map_err(|e| JsError::new(&e.to_string()))?;
        let weight_bytes = weights.byte_len() as f64;

        let max_seq = max_seq.clamp(16, config.context_length);
        let model = Model::new(weights, max_seq, RopeKind::SplitHalf);
        let mut ws = Workspace::new(&config, max_seq);
        // `core` has no clock of its own; hand it the host's.
        ws.set_clock(now_ms);
        let cache = KvCache::new(&config, max_seq);
        let sampler = Sampler::new(SamplerConfig::default(), config.vocab_size);

        Ok(Engine {
            gguf,
            model,
            tokenizer,
            ws,
            cache,
            sampler,
            config,
            pending: Vec::new(),
            prefill_tokens: 0,
            prefill_ms: 0.0,
            decode_tokens: 0,
            decode_ms: 0.0,
            first_token_ms: 0.0,
            recent_ms: [0.0; ROLLING_WINDOW],
            recent_at: 0,
            recent_len: 0,
            weight_bytes,
        })
    }

    /// A one-line description of the loaded model.
    #[wasm_bindgen(getter)]
    pub fn description(&self) -> String {
        alloc::format!(
            "{} · {} layers · d={} · {} heads / {} kv · vocab {}",
            self.config.architecture,
            self.config.block_count,
            self.config.embedding_length,
            self.config.head_count,
            self.config.head_count_kv,
            self.config.vocab_size
        )
    }

    #[wasm_bindgen(getter, js_name = contextLength)]
    pub fn context_length(&self) -> u32 {
        self.config.context_length as u32
    }

    #[wasm_bindgen(getter, js_name = maxSeq)]
    pub fn max_seq(&self) -> u32 {
        self.cache.max_seq() as u32
    }

    #[wasm_bindgen(getter, js_name = nLayers)]
    pub fn n_layers(&self) -> u32 {
        self.config.block_count as u32
    }

    #[wasm_bindgen(getter, js_name = nHeads)]
    pub fn n_heads(&self) -> u32 {
        self.config.head_count as u32
    }

    /// Choose the prefill kernel. Both paths give bit-identical results; this
    /// exists so the two can be compared on the machine that will run them.
    #[wasm_bindgen(js_name = setBatchedPrefill)]
    pub fn set_batched_prefill(&mut self, on: bool) {
        self.model.set_batched_prefill(on);
    }

    /// Record per-head attention weights during each decode step.
    ///
    /// Off by default: it copies `n_layers * n_heads * (pos+1)` floats per token,
    /// which is cheap but not free, and only the heatmap wants it.
    #[wasm_bindgen(js_name = setCaptureAttention)]
    pub fn set_capture_attention(&mut self, on: bool) {
        self.ws.set_capture_attention(on);
    }

    /// One head's attention distribution over the positions so far.
    ///
    /// Copied, not a view: it is at most `max_seq` floats, and a copy cannot go
    /// stale the way [`Self::logits_view`] can.
    #[wasm_bindgen(js_name = attention)]
    pub fn attention(&self, layer: u32, head: u32) -> Vec<f32> {
        let len = self.cache.len();
        if layer as usize >= self.config.block_count || head as usize >= self.config.head_count {
            return Vec::new();
        }
        self.ws
            .attention(layer as usize, head as usize, len)
            .to_vec()
    }

    // ------------------------------------------ quantisation explorer ----

    /// Every tensor in the file, in file order.
    #[wasm_bindgen(js_name = tensorList)]
    pub fn tensor_list(&self) -> Vec<TensorSummary> {
        self.gguf
            .tensors
            .iter()
            .map(|t| {
                let elems = t.n_elements().max(1);
                TensorSummary {
                    name: t.name.to_string(),
                    format: t.ggml_type.name().to_string(),
                    // GGUF's dims[0] is the contiguous axis, so it is the row
                    // *length*; presenting it the other way round would suggest
                    // every matrix in the model is transposed.
                    rows: t.n_rows() as u32,
                    cols: t.row_len() as u32,
                    blocks: (elems / t.ggml_type.block_elems() as u64) as u32,
                    bytes: t.byte_size as f64,
                    bits_per_weight: t.byte_size as f64 * 8.0 / elems as f64,
                    inspectable: quant::is_supported(t.ggml_type) && t.ggml_type.block_elems() > 1,
                }
            })
            .collect()
    }

    /// Take one block of one tensor apart.
    #[wasm_bindgen(js_name = inspectBlock)]
    pub fn inspect_block(&self, name: &str, block: u32) -> Result<BlockView, JsError> {
        let info = self
            .gguf
            .find_tensor(name)
            .ok_or_else(|| JsError::new(&alloc::format!("no tensor named {name:?}")))?;
        let ty = info.ggml_type;
        let bb = ty.block_bytes();
        let n_blocks = info.byte_size as usize / bb;
        if block as usize >= n_blocks {
            return Err(JsError::new(&alloc::format!(
                "block {block} out of range; {name} has {n_blocks}"
            )));
        }

        let data = self.gguf.tensor_data(info);
        let start = block as usize * bb;
        let dec = quant::decompose_block(ty, &data[start..start + bb])
            .map_err(|e| JsError::new(&e.to_string()))?;

        Ok(BlockView {
            values: dec.reconstruct(),
            quants: dec.quants,
            scales: dec.scales,
            mins: dec.mins,
            group: dec.group as u32,
            quant_min: dec.quant_range.0,
            quant_max: dec.quant_range.1,
        })
    }

    /// Every head's attention at once, laid out `[layer][head][max_seq]`.
    ///
    /// The heatmap needs all `n_layers * n_heads` distributions every token, and
    /// fetching them one call at a time would be 336 boundary crossings per
    /// frame. Same staleness caveat as [`Self::logits_view`].
    #[wasm_bindgen(js_name = attentionView)]
    pub fn attention_view(&self) -> js_sys::Float32Array {
        #[allow(unsafe_code)]
        unsafe {
            js_sys::Float32Array::view(self.ws.attention_all())
        }
    }

    pub fn tokenize(&self, text: &str, parse_special: bool) -> Vec<u32> {
        self.tokenizer.encode(
            text,
            EncodeOptions {
                parse_special,
                add_bos: false,
            },
        )
    }

    pub fn detokenize(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens, true)
    }

    /// Wrap a user message in Qwen2's chat format, including the default system
    /// prompt the model was tuned with.
    #[wasm_bindgen(js_name = chatPrompt)]
    pub fn chat_prompt(&self, user: &str) -> String {
        apply_chat_template(&[ChatMessage::user(user)], true)
    }

    #[wasm_bindgen(js_name = setSampling)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_sampling(
        &mut self,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        repetition_penalty: f32,
        repetition_window: u32,
        seed: f64,
    ) {
        // `seed` arrives as an f64 because JS numbers are doubles; anything above
        // 2^53 could not have survived the trip anyway, so clamp rather than
        // silently wrap.
        let seed = if seed.is_finite() && seed >= 0.0 {
            seed.min(9_007_199_254_740_991.0) as u64
        } else {
            0
        };
        self.sampler.set_config(SamplerConfig {
            temperature,
            top_k: top_k as usize,
            top_p,
            repetition_penalty,
            repetition_window: repetition_window as usize,
            seed,
        });
    }

    /// Process a prompt. Returns milliseconds spent.
    ///
    /// This is the blocking part -- one call for the whole prompt -- so a long
    /// prompt will stall the frame it runs in. Callers that care should run it
    /// in a worker; the decode loop is the part that had to be incremental.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<f64, JsError> {
        if tokens.is_empty() {
            return Err(JsError::new("prefill needs at least one token"));
        }
        if self.cache.len() + tokens.len() > self.cache.max_seq() {
            return Err(JsError::new("prompt does not fit in the KV cache"));
        }
        let t0 = now_ms();
        // Batched: one weight-row unpack serves a whole chunk of positions.
        self.model
            .prefill(tokens, &mut self.ws, &mut self.cache)
            .map_err(|e| JsError::new(&e.to_string()))?;
        for &t in tokens {
            self.sampler.accept(t);
        }
        let dt = now_ms() - t0;
        self.prefill_tokens += tokens.len() as u32;
        self.prefill_ms += dt;
        Ok(dt)
    }

    /// Produce exactly one token. Call this from a `requestAnimationFrame` or
    /// `setTimeout` loop -- never in a `while` loop.
    #[wasm_bindgen(js_name = decodeStep)]
    pub fn decode_step(&mut self) -> Result<DecodeResult, JsError> {
        if self.cache.is_empty() {
            return Err(JsError::new("call prefill() before decode_step()"));
        }
        let t0 = now_ms();

        let token = self.sampler.sample(self.model.logits(&self.ws));
        let is_eos = Some(token) == self.config.eos_token_id;
        self.sampler.accept(token);

        let text = if is_eos {
            String::new()
        } else {
            self.push_text(token)
        };

        // Feed the token back in, unless it was EOS or the cache is full.
        if !is_eos && self.cache.len() < self.cache.max_seq() {
            let pos = self.cache.len();
            self.model
                .forward_token(token, pos, &mut self.ws, &mut self.cache, None)
                .map_err(|e| JsError::new(&e.to_string()))?;
        }

        let ms = now_ms() - t0;
        self.decode_tokens += 1;
        self.decode_ms += ms;
        if self.decode_tokens == 1 {
            self.first_token_ms = self.prefill_ms + ms;
        }
        self.recent_ms[self.recent_at] = ms;
        self.recent_at = (self.recent_at + 1) % ROLLING_WINDOW;
        self.recent_len = (self.recent_len + 1).min(ROLLING_WINDOW);
        let t = self.ws.timings();
        Ok(DecodeResult {
            token,
            text,
            is_eos,
            ms,
            matmul_ms: t.matmul_ms,
            attention_ms: t.attention_ms,
            other_ms: t.other_ms,
        })
    }

    /// Decode one token's bytes, holding back any incomplete UTF-8 sequence.
    fn push_text(&mut self, token: u32) -> String {
        self.pending
            .extend_from_slice(&self.tokenizer.decode_bytes(&[token], true));
        match core::str::from_utf8(&self.pending) {
            Ok(s) => {
                let out = s.to_string();
                self.pending.clear();
                out
            }
            Err(e) => {
                let valid = e.valid_up_to();
                let out = core::str::from_utf8(&self.pending[..valid])
                    .unwrap_or("")
                    .to_string();
                match e.error_len() {
                    // Genuinely invalid, not merely incomplete: drop the bad
                    // bytes, or `pending` would never drain and every later
                    // token would be swallowed.
                    Some(bad) => {
                        self.pending.drain(..valid + bad);
                    }
                    None => {
                        self.pending.drain(..valid);
                    }
                }
                out
            }
        }
    }

    /// Start a new conversation. Allocates nothing.
    pub fn reset(&mut self) {
        self.cache.reset();
        self.sampler.reset();
        self.pending.clear();
        self.prefill_tokens = 0;
        self.prefill_ms = 0.0;
        self.decode_tokens = 0;
        self.decode_ms = 0.0;
        self.first_token_ms = 0.0;
        self.recent_ms = [0.0; ROLLING_WINDOW];
        self.recent_at = 0;
        self.recent_len = 0;
    }

    pub fn stats(&self) -> Stats {
        Stats {
            prefill_tokens: self.prefill_tokens,
            prefill_ms: self.prefill_ms,
            decode_tokens: self.decode_tokens,
            decode_ms: self.decode_ms,
            first_token_ms: self.first_token_ms,
            rolling_tps: {
                let n = self.recent_len;
                let sum: f64 = self.recent_ms[..n].iter().sum();
                if n == 0 || sum <= 0.0 {
                    0.0
                } else {
                    n as f64 * 1000.0 / sum
                }
            },
            cache_used: self.cache.len() as u32,
            cache_capacity: self.cache.max_seq() as u32,
            cache_bytes: self.cache.byte_len() as f64,
            workspace_bytes: self.ws.byte_len() as f64,
            weight_bytes: self.weight_bytes,
        }
    }

    /// A zero-copy view of the current logits, straight into wasm memory.
    ///
    /// # This view can go stale
    ///
    /// It aliases the wasm linear memory. **Any allocation that grows that memory
    /// detaches the underlying `ArrayBuffer` and leaves this view zero-length**,
    /// which is why the engine preallocates its workspace and KV cache up front:
    /// steady-state decoding does not grow memory, so a view taken between steps
    /// stays valid. Even so, do not cache it across calls -- re-acquire it each
    /// time you need it. Copy with `.slice()` if you need to keep the data.
    #[wasm_bindgen(js_name = logitsView)]
    pub fn logits_view(&self) -> js_sys::Float32Array {
        let logits = self.model.logits(&self.ws);
        // SAFETY-adjacent: `Float32Array::view` is unsafe precisely because of
        // the detachment hazard documented above. It is contained here, and the
        // returned value is not stored anywhere.
        #[allow(unsafe_code)]
        unsafe {
            js_sys::Float32Array::view(logits)
        }
    }
}
