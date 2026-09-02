//! Dequantisation: ggml block formats -> f32.
//!
//! # What this module is, and what it is not
//!
//! Everything here dequantises a whole row into an f32 buffer. That is the right
//! tool for tests, for tooling like `nano-infer dequant`, and for the f32
//! reference path -- and it is emphatically the *wrong* tool for inference.
//!
//! Dequantising Qwen2.5-0.5B's weights to f32 would turn a ~400 MB model into
//! ~2 GB of f32, which does not fit in wasm's practical memory budget and would
//! be memory-bandwidth suicide even if it did. The real decode path (step 6)
//! never materialises a dequantised weight tensor: it fuses dequantisation into
//! the matmul inner loop so a block is unpacked into registers, used, and
//! discarded. These functions are the correctness oracle that the fused kernels
//! will be tested against, not the kernels themselves.
//!
//! # Supported formats
//!
//! F32, F16, Q8_0, Q4_0, Q5_0, Q4_K, Q6_K.
//!
//! That is exactly what the real Qwen2.5-0.5B-Instruct Q4_K_M file needs, which
//! is not what its name suggests: 55% of it is Q5_0 and only 6% is Q4_K, because
//! k-quants require rows that are a multiple of 256 and this model's rows are 896
//! long. See [`dequant_block_q5_0`]. Anything else returns
//! [`QuantError::UnsupportedType`] rather than producing garbage.

#![deny(unsafe_code)]

mod f16;
mod inspect;
mod k_quants;
mod legacy;
mod vecdot;

pub mod fixtures;

#[cfg(all(test, feature = "std"))]
mod tests;

pub use f16::{f16_to_f32, f32_to_f16, read_f16_le};
pub use inspect::{decompose_block, BlockDecomp};
pub use k_quants::{dequant_block_q4_k, dequant_block_q6_k, pack_scales_min_k4, scale_min_k4};
pub use legacy::{dequant_block_q4_0, dequant_block_q5_0, dequant_block_q8_0, quantize_block_q8_0};
pub use vecdot::{
    row_dot_q4_0, row_dot_q4_k, row_dot_q5_0, row_dot_q6_k, row_dot_q8_0, row_dot_unpacked,
    unpack_row, ActivationQ8, UnpackedRow, ACT_BLOCK,
};

use crate::gguf::GgmlType;
use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantError {
    /// We know the block geometry (so offsets are still correct) but have not
    /// written a kernel for this format.
    UnsupportedType(GgmlType),
    /// `src` is not exactly the number of bytes implied by `dst.len()`.
    BadSourceLength {
        ty: GgmlType,
        got: usize,
        want: usize,
    },
    /// `dst.len()` is not a whole number of blocks.
    NotBlockAligned {
        ty: GgmlType,
        len: usize,
        block: usize,
    },
}

impl fmt::Display for QuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantError::UnsupportedType(t) => {
                write!(f, "dequantisation for {t} is not implemented")
            }
            QuantError::BadSourceLength { ty, got, want } => write!(
                f,
                "{ty}: source is {got} bytes, expected {want} for the requested element count"
            ),
            QuantError::NotBlockAligned { ty, len, block } => write!(
                f,
                "{ty}: {len} elements is not a whole number of {block}-element blocks"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for QuantError {}

/// True if [`dequantize_row`] can handle this type.
pub const fn is_supported(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::Q8_0
            | GgmlType::Q4_0
            | GgmlType::Q5_0
            | GgmlType::Q4_K
            | GgmlType::Q6_K
    )
}

/// Dequantise `dst.len()` elements from `src` into `dst`.
///
/// `src` must be exactly the packed bytes for that many elements -- no more, no
/// less -- which is what [`crate::gguf::Gguf::tensor_data`] returns.
pub fn dequantize_row(ty: GgmlType, src: &[u8], dst: &mut [f32]) -> Result<(), QuantError> {
    if !is_supported(ty) {
        return Err(QuantError::UnsupportedType(ty));
    }

    let block_elems = ty.block_elems();
    if !dst.len().is_multiple_of(block_elems) {
        return Err(QuantError::NotBlockAligned {
            ty,
            len: dst.len(),
            block: block_elems,
        });
    }
    let n_blocks = dst.len() / block_elems;
    let want = n_blocks * ty.block_bytes();
    if src.len() != want {
        return Err(QuantError::BadSourceLength {
            ty,
            got: src.len(),
            want,
        });
    }

    // `as_chunks` hands each kernel a `&[u8; N]` directly, so the per-block
    // bounds check disappears at compile time. The remainders are provably empty
    // -- the length checks above guarantee it -- which is why they are only
    // debug-asserted rather than handled.
    macro_rules! blocks {
        ($sb:literal, $eb:literal, $kernel:path) => {{
            let (src_blocks, src_rest) = src.as_chunks::<$sb>();
            let (dst_blocks, dst_rest) = dst.as_chunks_mut::<$eb>();
            debug_assert!(src_rest.is_empty() && dst_rest.is_empty());
            for (s, d) in src_blocks.iter().zip(dst_blocks) {
                $kernel(s, d);
            }
        }};
    }

    match ty {
        GgmlType::F32 => {
            let (words, rest) = src.as_chunks::<4>();
            debug_assert!(rest.is_empty());
            for (o, c) in dst.iter_mut().zip(words) {
                *o = f32::from_le_bytes(*c);
            }
        }
        GgmlType::F16 => {
            let (words, rest) = src.as_chunks::<2>();
            debug_assert!(rest.is_empty());
            for (o, c) in dst.iter_mut().zip(words) {
                *o = f16_to_f32(u16::from_le_bytes(*c));
            }
        }
        GgmlType::Q8_0 => blocks!(34, 32, dequant_block_q8_0),
        GgmlType::Q4_0 => blocks!(18, 32, dequant_block_q4_0),
        GgmlType::Q5_0 => blocks!(22, 32, dequant_block_q5_0),
        GgmlType::Q4_K => blocks!(144, 256, dequant_block_q4_k),
        GgmlType::Q6_K => blocks!(210, 256, dequant_block_q6_k),
        // `is_supported` already filtered these out.
        other => return Err(QuantError::UnsupportedType(other)),
    }
    Ok(())
}

/// Dequantise a single block. Convenience wrapper for tooling; the row form is
/// what callers normally want.
pub fn dequantize_block(ty: GgmlType, src: &[u8], dst: &mut [f32]) -> Result<(), QuantError> {
    if dst.len() != ty.block_elems() {
        return Err(QuantError::NotBlockAligned {
            ty,
            len: dst.len(),
            block: ty.block_elems(),
        });
    }
    dequantize_row(ty, src, dst)
}
