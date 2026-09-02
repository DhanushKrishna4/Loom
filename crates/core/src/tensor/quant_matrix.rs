//! A 2-D weight matrix, still in the file's quantisation format.
//!
//! # The dispatch decision
//!
//! This is where "enum dispatch at the boundary" becomes concrete. The
//! quantisation is a property of the *file*, not of the source, so the branch
//! has to exist somewhere. Putting it here -- one `match` per matmul, on a field
//! that is constant for the lifetime of the model -- means the branch is hoisted
//! entirely out of the loops underneath, and each format gets a monomorphic
//! kernel written for its own packing.
//!
//! A `QuantFormat` trait with a generic kernel would produce the identical inner
//! loop, because the dispatch still has to happen at exactly this point. What it
//! would add is a shared abstraction over formats that have almost nothing in
//! common: Q4_K and Q8_0 differ in block size, packing, unpack cost and SIMD
//! strategy, so every trait method would end up `#[inline(always)]` and
//! specialised per implementor anyway.
//!
//! Note the discriminant is [`GgmlType`] itself rather than a parallel enum --
//! there is no second source of truth to drift.

use crate::gguf::{GgmlType, TensorInfo};
use crate::quant::{self, QuantError};

/// `rows` rows of `cols` contiguous weights, in the file's own format.
///
/// GGUF stores weight matrices with the *input* dimension contiguous, so one row
/// here is one output neuron's full set of input weights. That is exactly the
/// access pattern `matvec` wants, which is why no transpose ever happens.
#[derive(Debug, Clone, Copy)]
pub struct QuantMatrix<'a> {
    ty: GgmlType,
    data: &'a [u8],
    rows: usize,
    cols: usize,
}

impl<'a> QuantMatrix<'a> {
    /// Wrap a GGUF tensor. `data` must be the tensor's bytes, as returned by
    /// [`crate::gguf::Gguf::tensor_data`] -- borrowed, never copied.
    pub fn from_tensor(info: &TensorInfo<'_>, data: &'a [u8]) -> Result<Self, QuantError> {
        let ty = info.ggml_type;
        if !quant::is_supported(ty) {
            return Err(QuantError::UnsupportedType(ty));
        }
        // GGUF's dims[0] is the contiguous axis, so it is the row length.
        let cols = info.row_len() as usize;
        let rows = info.n_rows() as usize;

        if !cols.is_multiple_of(ty.block_elems()) {
            return Err(QuantError::NotBlockAligned {
                ty,
                len: cols,
                block: ty.block_elems(),
            });
        }
        let want = rows * (cols / ty.block_elems()) * ty.block_bytes();
        if data.len() != want {
            return Err(QuantError::BadSourceLength {
                ty,
                got: data.len(),
                want,
            });
        }
        Ok(QuantMatrix {
            ty,
            data,
            rows,
            cols,
        })
    }

    /// Build one directly, for tests and for synthesised weights.
    pub fn new(ty: GgmlType, data: &'a [u8], rows: usize, cols: usize) -> Result<Self, QuantError> {
        if !quant::is_supported(ty) {
            return Err(QuantError::UnsupportedType(ty));
        }
        if !cols.is_multiple_of(ty.block_elems()) {
            return Err(QuantError::NotBlockAligned {
                ty,
                len: cols,
                block: ty.block_elems(),
            });
        }
        let want = rows * (cols / ty.block_elems()) * ty.block_bytes();
        if data.len() != want {
            return Err(QuantError::BadSourceLength {
                ty,
                got: data.len(),
                want,
            });
        }
        Ok(QuantMatrix {
            ty,
            data,
            rows,
            cols,
        })
    }

    pub fn ggml_type(&self) -> GgmlType {
        self.ty
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn bytes_per_row(&self) -> usize {
        (self.cols / self.ty.block_elems()) * self.ty.block_bytes()
    }

    /// The raw bytes of one row, still quantised.
    ///
    /// This is what the fused kernels in step 6 will consume. They will walk it
    /// a block at a time, unpacking into registers, never into memory.
    pub fn row_bytes(&self, r: usize) -> &'a [u8] {
        let n = self.bytes_per_row();
        &self.data[r * n..(r + 1) * n]
    }

    /// Dequantise one row into `out`, which must be `cols` long.
    ///
    /// For the f32 reference path and for tooling. The decode path must not use
    /// this: materialising rows defeats the entire point of the fused kernels.
    pub fn dequant_row(&self, r: u32, out: &mut [f32]) -> Result<(), QuantError> {
        // The one dispatch, hoisted above every loop underneath it.
        quant::dequantize_row(self.ty, self.row_bytes(r as usize), out)
    }

    /// Total bytes held. Never a copy -- this is the size of the borrowed slice.
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }
}
