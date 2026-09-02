//! The transformer itself: weights, KV cache, forward pass.
//!
//! # Architecture (Qwen2, a Llama-style decoder)
//!
//! ```text
//! x = embed(token)
//! per layer:
//!   x += attention(rmsnorm(x, attn_norm))
//!   x += ffn(rmsnorm(x, ffn_norm))
//! logits = output @ rmsnorm(x, output_norm)
//! ```
//!
//! Pre-norm (normalise *into* the block, add the raw block output to the
//! residual), RMSNorm not LayerNorm, SwiGLU not GELU, GQA not MHA, and RoPE
//! applied to Q and K only.
//!
//! # The Qwen2-specific bit
//!
//! **Q, K and V have biases; the output projection does not.** Llama has no
//! attention biases at all, and most from-scratch implementations follow Llama.
//! The real GGUF carries 72 bias tensors (3 per layer x 24 layers); dropping
//! them does not crash anything, it just quietly makes every attention score
//! wrong. See [`LayerWeights`].

#![deny(unsafe_code)]

mod forward;
mod weights;

#[cfg(all(test, feature = "std"))]
mod tests;

pub use forward::{KvCache, KvLayout, Model, OpTimings, Trace, Workspace};
pub use weights::{LayerWeights, ModelWeights, WeightError};
