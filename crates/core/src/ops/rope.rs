//! Rotary position embeddings.
//!
//! # The convention trap
//!
//! RoPE rotates each head's vector in `head_dim/2` independent 2-D planes. The
//! rotation angles are unambiguous. **Which pairs of components form a plane is
//! not**, and there are two conventions in the wild:
//!
//! ```text
//!   split-half ("NeoX"):  pairs are (x[i], x[i + d/2])   i in 0..d/2
//!   adjacent   ("NORM"):  pairs are (x[2i], x[2i + 1])   i in 0..d/2
//! ```
//!
//! Both are self-consistent, both preserve norms, both produce fluent text. Using
//! the wrong one gives a model whose sense of position is scrambled in a way that
//! degrades coherence over distance without ever failing an assertion. There is
//! no numerical property that distinguishes them -- only a reference comparison
//! does. Hence [`RopeKind`] rather than a hardcoded loop.
//!
//! # Which one Qwen2 needs -- UNVERIFIED
//!
//! HuggingFace's Qwen2 uses `rotate_half`, which is split-half. llama.cpp
//! dispatches Qwen2 to its NEOX rope type, which is also split-half. Those agree,
//! so [`RopeKind::SplitHalf`] is the default here.
//!
//! The reason this is still marked unverified: for *Llama* models,
//! `convert_hf_to_gguf.py` permutes the Q and K weight matrices at conversion
//! time specifically so that llama.cpp's *adjacent* rope reproduces HF's
//! split-half result. Whether the Qwen2 conversion path applies that permutation
//! changes which convention is correct **for weights read out of a GGUF**, as
//! opposed to weights read out of a safetensors checkpoint. Reading the converter
//! is not proof; the check is `tools/dump_reference_ops.py --model`, comparing
//! our layer-0 K projection after rope against the reference. Until that runs,
//! treat the default as a well-motivated guess.
//!
//! # Partial rotation
//!
//! Only the first `rope_dim` components of each head are rotated; anything above
//! that passes through untouched. Qwen2.5-0.5B has `head_dim == rope_dim == 64`,
//! so nothing is left over, but the general case is cheap to support and models
//! that set `rope.dimension_count < head_dim` do exist.

use alloc::vec;
use alloc::vec::Vec;

use crate::math::{cos, powf, sin};

/// Which components pair up to form each rotation plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeKind {
    /// `(x[i], x[i + d/2])`. HuggingFace `rotate_half`, llama.cpp `NEOX`.
    /// What Qwen2 is believed to want.
    SplitHalf,
    /// `(x[2i], x[2i+1])`. GPT-J style, llama.cpp `NORM`.
    Adjacent,
}

/// Precomputed cos/sin for every (position, plane) pair.
///
/// Built once at load time for the full context. For Qwen2.5-0.5B that is
/// 32768 positions x 32 planes x 2 floats = 8 MiB, which buys us two table
/// lookups per plane instead of a `sin` and a `cos` per plane per token per
/// layer. Worth it -- but it is also 8 MiB of the wasm memory budget, so the
/// engine will size the table to the KV cache's `max_seq`, not to
/// `context_length`.
#[derive(Debug, Clone)]
pub struct RopeTable {
    /// `[max_seq][n_planes]`, row-major.
    cos: Vec<f32>,
    sin: Vec<f32>,
    head_dim: usize,
    rope_dim: usize,
    max_seq: usize,
    kind: RopeKind,
}

impl RopeTable {
    /// `theta_i = freq_base^(-2i / rope_dim)`, angle at position `p` is `p * theta_i`.
    ///
    /// `freq_base` is 1e6 for Qwen2.5, not the 1e4 that most references default
    /// to. Getting it wrong does not crash; it just rescales the model's whole
    /// sense of distance.
    pub fn new(
        head_dim: usize,
        rope_dim: usize,
        max_seq: usize,
        freq_base: f32,
        kind: RopeKind,
    ) -> Self {
        assert!(
            rope_dim <= head_dim,
            "rope_dim {rope_dim} exceeds head_dim {head_dim}"
        );
        assert!(
            rope_dim.is_multiple_of(2),
            "rope_dim {rope_dim} must be even"
        );

        let n_planes = rope_dim / 2;
        let mut cos_t = vec![0.0f32; max_seq * n_planes];
        let mut sin_t = vec![0.0f32; max_seq * n_planes];

        for i in 0..n_planes {
            let theta = 1.0 / powf(freq_base, 2.0 * i as f32 / rope_dim as f32);
            for p in 0..max_seq {
                let angle = p as f32 * theta;
                cos_t[p * n_planes + i] = cos(angle);
                sin_t[p * n_planes + i] = sin(angle);
            }
        }

        RopeTable {
            cos: cos_t,
            sin: sin_t,
            head_dim,
            rope_dim,
            max_seq,
            kind,
        }
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn rope_dim(&self) -> usize {
        self.rope_dim
    }

    pub fn kind(&self) -> RopeKind {
        self.kind
    }

    pub fn n_planes(&self) -> usize {
        self.rope_dim / 2
    }

    /// Bytes held by the table, for the memory panel.
    pub fn byte_len(&self) -> usize {
        (self.cos.len() + self.sin.len()) * core::mem::size_of::<f32>()
    }

    /// Rotate one head's vector in place, at position `pos`.
    ///
    /// `x` must be exactly `head_dim` long: one head, not the concatenated
    /// projection. Applying rope across a head boundary is a classic way to
    /// produce output that is fluent and wrong, so the length is asserted rather
    /// than inferred.
    pub fn apply_head(&self, x: &mut [f32], pos: usize) {
        assert_eq!(
            x.len(),
            self.head_dim,
            "rope expects exactly one head's vector"
        );
        assert!(
            pos < self.max_seq,
            "position {pos} beyond table size {}",
            self.max_seq
        );

        let n = self.n_planes();
        let base = pos * n;
        for i in 0..n {
            let c = self.cos[base + i];
            let s = self.sin[base + i];
            // The only difference between the two conventions.
            let (a, b) = match self.kind {
                RopeKind::SplitHalf => (i, i + n),
                RopeKind::Adjacent => (2 * i, 2 * i + 1),
            };
            let x0 = x[a];
            let x1 = x[b];
            x[a] = x0 * c - x1 * s;
            x[b] = x0 * s + x1 * c;
        }
        // Components at or above rope_dim are left alone by construction.
    }

    /// Rotate a whole projection of `n_heads` contiguous heads at one position.
    ///
    /// This is what the attention block calls for Q and for K. Never for V --
    /// V carries no positional information and rotating it is silently wrong.
    pub fn apply_projection(&self, x: &mut [f32], n_heads: usize, pos: usize) {
        assert_eq!(
            x.len(),
            n_heads * self.head_dim,
            "projection must be n_heads * head_dim"
        );
        for h in 0..n_heads {
            self.apply_head(&mut x[h * self.head_dim..(h + 1) * self.head_dim], pos);
        }
    }
}
