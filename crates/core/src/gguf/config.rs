//! Pulling the architecture hyper-parameters out of the metadata bag.
//!
//! GGUF namespaces the architecture keys under `general.architecture`, so for a
//! Qwen2 model the layer count lives at `qwen2.block_count`, not `block_count`.
//! Everything here builds those keys dynamically; nothing hardcodes "qwen2", so
//! a Llama or Mistral GGUF parses through the same path.

use alloc::format;
use alloc::string::{String, ToString};

use super::error::GgufError;
use super::Gguf;

/// Everything the forward pass needs to know about the model's shape.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub architecture: String,
    pub name: Option<String>,

    /// Number of transformer layers (`n_layers`).
    pub block_count: usize,
    /// Model width (`d_model`).
    pub embedding_length: usize,
    /// SwiGLU intermediate width. Note this is the width of *each* of the gate
    /// and up projections, not their sum.
    pub feed_forward_length: usize,

    pub head_count: usize,
    /// KV heads. Equal to `head_count` for plain MHA; smaller under GQA.
    pub head_count_kv: usize,
    /// Per-head width. Usually `embedding_length / head_count`, but some models
    /// state it explicitly via `attention.key_length`, so prefer the stated value.
    pub head_dim: usize,

    pub context_length: usize,
    pub rms_norm_eps: f32,
    pub rope_freq_base: f32,
    /// How many of each head's dimensions get rotated. Absent means "all of them".
    pub rope_dimension_count: Option<usize>,

    pub vocab_size: usize,

    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
    pub add_bos_token: Option<bool>,
    /// `tokenizer.ggml.model`, e.g. "gpt2" for Qwen2's byte-level BPE.
    pub tokenizer_model: Option<String>,
}

impl ModelConfig {
    /// Query heads per KV head. The GQA mapping is `kv_head = q_head / this`.
    pub fn kv_group_size(&self) -> usize {
        self.head_count / self.head_count_kv
    }

    /// Total width of the Q projection.
    pub fn q_dim(&self) -> usize {
        self.head_count * self.head_dim
    }

    /// Total width of each of the K and V projections. Under GQA this is smaller
    /// than `q_dim`, which is the entire point of GQA and also the entire source
    /// of off-by-one head-mapping bugs.
    pub fn kv_dim(&self) -> usize {
        self.head_count_kv * self.head_dim
    }

    /// Bytes of f32 KV cache needed for `seq_len` positions, both K and V, all layers.
    pub fn kv_cache_bytes(&self, seq_len: usize) -> usize {
        2 * self.block_count
            * self.head_count_kv
            * seq_len
            * self.head_dim
            * core::mem::size_of::<f32>()
    }

    fn validate(&self) -> Result<(), GgufError> {
        if self.head_count == 0 || self.head_count_kv == 0 {
            return Err(GgufError::InconsistentConfig(
                "head_count must be non-zero".to_string(),
            ));
        }
        if !self.head_count.is_multiple_of(self.head_count_kv) {
            return Err(GgufError::InconsistentConfig(format!(
                "head_count {} is not divisible by head_count_kv {}: GQA grouping would be undefined",
                self.head_count, self.head_count_kv
            )));
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            // RoPE rotates pairs of dimensions, so an odd head_dim cannot work.
            return Err(GgufError::InconsistentConfig(format!(
                "head_dim {} must be even for RoPE",
                self.head_dim
            )));
        }
        if let Some(rd) = self.rope_dimension_count {
            if rd > self.head_dim || !rd.is_multiple_of(2) {
                return Err(GgufError::InconsistentConfig(format!(
                    "rope.dimension_count {} must be even and <= head_dim {}",
                    rd, self.head_dim
                )));
            }
        }
        if self.block_count == 0 || self.embedding_length == 0 || self.vocab_size == 0 {
            return Err(GgufError::InconsistentConfig(
                "block_count, embedding_length and vocab_size must all be non-zero".to_string(),
            ));
        }
        Ok(())
    }
}

impl<'a> Gguf<'a> {
    /// Extract the architecture config. Fails loudly on anything missing rather
    /// than substituting a plausible default, except where the GGUF spec itself
    /// defines a default (noted inline).
    pub fn config(&self) -> Result<ModelConfig, GgufError> {
        let architecture = self.require_str("general.architecture")?;
        let k = |suffix: &str| format!("{architecture}.{suffix}");

        let embedding_length = self.require_usize(&k("embedding_length"))?;
        let head_count = self.require_usize(&k("attention.head_count"))?;

        // Absent head_count_kv means the model is plain multi-head attention.
        let head_count_kv = self
            .get_usize(&k("attention.head_count_kv"))
            .unwrap_or(head_count);

        // Prefer the explicit per-head width; several architectures (and every
        // model where d_model != n_heads * head_dim) rely on it.
        let head_dim = self
            .get_usize(&k("attention.key_length"))
            .unwrap_or_else(|| embedding_length.checked_div(head_count).unwrap_or(0));

        // The vocab is authoritative when present: `{arch}.vocab_size` is
        // sometimes stale or padded relative to the actual token list.
        let vocab_size = self
            .get_array("tokenizer.ggml.tokens")
            .map(|a| a.len())
            .or_else(|| self.get_usize(&k("vocab_size")))
            .ok_or_else(|| GgufError::MissingKey("tokenizer.ggml.tokens".to_string()))?;

        let cfg = ModelConfig {
            architecture: architecture.to_string(),
            name: self.get_str("general.name").map(|s| s.to_string()),
            block_count: self.require_usize(&k("block_count"))?,
            embedding_length,
            feed_forward_length: self.require_usize(&k("feed_forward_length"))?,
            head_count,
            head_count_kv,
            head_dim,
            context_length: self.require_usize(&k("context_length"))?,
            // GGUF's documented default for the RMS epsilon key.
            rms_norm_eps: self
                .get_f32(&k("attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-5),
            // GGUF's documented default. Qwen2.5 overrides it to 1e6, and using
            // 1e4 by mistake yields fluent-but-wrong text, so this is worth
            // eyeballing in `gguf-dump` output for any new model.
            rope_freq_base: self.get_f32(&k("rope.freq_base")).unwrap_or(10_000.0),
            rope_dimension_count: self.get_usize(&k("rope.dimension_count")),
            vocab_size,
            bos_token_id: self.get_u32("tokenizer.ggml.bos_token_id"),
            eos_token_id: self.get_u32("tokenizer.ggml.eos_token_id"),
            pad_token_id: self.get_u32("tokenizer.ggml.padding_token_id"),
            unk_token_id: self.get_u32("tokenizer.ggml.unknown_token_id"),
            add_bos_token: self.get_bool("tokenizer.ggml.add_bos_token"),
            tokenizer_model: self.get_str("tokenizer.ggml.model").map(|s| s.to_string()),
        };
        cfg.validate()?;
        Ok(cfg)
    }
}
