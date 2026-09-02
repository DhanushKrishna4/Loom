//! `nano-infer-core` -- the parts of the engine that are pure computation.
//!
//! Everything numerical lives here, and *nothing* in here knows that WebAssembly
//! or a browser exists.  That is a deliberate constraint, not an accident: the
//! whole crate builds and tests natively, so a wrong number can be chased with a
//! normal debugger and a normal test runner instead of through devtools.
//!
//! # Build status
//!
//! This tree currently covers build steps 1-2:
//!
//! * [`gguf`]  -- GGUF v2/v3 container parsing and model-config extraction.
//! * [`math`]  -- the float math `core` does not have.
//! * [`tensor`] -- shapes, views, and the quantised weight matrix.
//! * [`ops`]   -- naive f32 forward-pass kernels.
//! * [`quant`] -- f16 conversion and dequantisation for F32/F16/Q8_0/Q4_0/Q4_K/Q6_K.
//!
//! Still to come, in order: tokenizer (4), the forward pass (5), fused
//! quantised matmul (6), KV cache (7), sampling (8), optimisation (9).
//!
//! # `no_std`
//!
//! The crate is `no_std` + `alloc` when the `std` feature is off.  We do allocate
//! (metadata, tensor tables, the KV cache), we just do not need a host OS.

#![cfg_attr(not(feature = "std"), no_std)]
// SIMD128 in step 9 will need `unsafe`, and it will be opted into per-module with
// an explicit `#[allow(unsafe_code)]` and a comment justifying it.  Until then the
// entire crate is safe Rust, including all of the bit-twiddling in `quant`.
#![deny(unsafe_code)]
#![warn(clippy::all)]

extern crate alloc;

pub mod gguf;
pub mod math;
pub mod model;
pub mod ops;
pub mod quant;
pub mod sample;
pub mod tensor;
pub mod tokenizer;

#[cfg(all(test, feature = "std"))]
mod testutil;

pub use gguf::{GgmlType, Gguf, GgufError, ModelConfig};
pub use quant::{f16_to_f32, f32_to_f16, QuantError};
pub use tensor::{QuantMatrix, Shape, TensorView, TensorViewMut};
