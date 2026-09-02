//! The forward-pass kernels.
//!
//! # Status: naive
//!
//! Every kernel here is the obvious loop. That is on purpose. These are the
//! versions that get validated against PyTorch first; step 9 replaces the hot
//! ones with cache-blocked, SIMD128 implementations and keeps [`reference`] as
//! the oracle they are tested against. Optimising before there is something
//! correct to compare against is how you end up with a fast wrong answer.
//!
//! # Why these take slices, not tensors
//!
//! See [`crate::tensor`]. Shape bookkeeping happens once, at the call site; the
//! inner loops see contiguous `&[f32]` and nothing else.
//!
//! # Matrix convention
//!
//! Every weight matmul in the model is "row times row", never "row times
//! column". GGUF stores a weight matrix with the *input* dimension contiguous,
//! so output neuron `r`'s weights are `w[r * cols .. (r+1) * cols]` and the dot
//! product against the activation vector reads two contiguous runs. No transpose
//! ever happens, in either direction. [`matmul_nt`] is the general form and
//! [`matvec`] is its batch-of-one specialisation, which is where ~95% of decode
//! time will go.

#![deny(unsafe_code)]

mod activation;
mod elementwise;
mod matmul;
mod norm;
mod rope;

#[cfg(any(test, feature = "reference"))]
pub mod reference;

#[cfg(all(test, feature = "std"))]
mod tests;

pub use activation::{silu, silu_inplace, softmax, swiglu};
pub use elementwise::{add_assign, dot, mul_assign, scale};
pub use matmul::{matmul_fused, matmul_nt, matvec, matvec_dequant, matvec_fused};
pub use norm::rmsnorm;
pub use rope::{RopeKind, RopeTable};
