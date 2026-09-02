//! Shapes, views, and the quantised weight matrix.
//!
//! # Where the abstraction stops
//!
//! Tensors live at the *graph* layer: carving a big preallocated buffer into the
//! Q/K/V slices for a layer, addressing a head inside the KV cache, walking rows
//! of a weight matrix. Kernels take plain `&[f32]` and explicit dimensions.
//!
//! That split is deliberate. A kernel that accepts a strided N-D tensor has to
//! either handle arbitrary strides in its inner loop -- which is exactly the
//! indexing overhead we are trying to avoid -- or check contiguity and bail,
//! which is an abstraction that lies. Keeping tensors out of the kernels means
//! the hot loops see contiguous slices and nothing else, and the shape bookkeeping
//! happens once per op instead of once per element.
//!
//! So: [`Shape`] and [`TensorView`] for the model graph, slices for [`crate::ops`].

#![deny(unsafe_code)]

mod quant_matrix;

#[cfg(all(test, feature = "std"))]
mod tests;

pub use quant_matrix::QuantMatrix;

use crate::gguf::GGML_MAX_DIMS;

/// Maximum tensor rank, matching ggml.
pub const MAX_RANK: usize = GGML_MAX_DIMS;

/// A shape in **row-major order**: `dims[0]` is the outermost axis and
/// `dims[rank-1]` is the contiguous one.
///
/// Note this is the *reverse* of GGUF's on-disk convention, where `dims[0]` is
/// the contiguous axis. [`crate::gguf::TensorInfo::shape_row_major`] does the
/// flip, and doing it exactly once -- at the boundary, on the way in -- is what
/// keeps the rest of the engine from having to think about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    dims: [usize; MAX_RANK],
    rank: usize,
}

impl Shape {
    /// Panics if `dims` is empty or longer than [`MAX_RANK`]; shapes are built
    /// from parsed model metadata, so a bad one is a bug rather than bad input.
    pub fn new(dims: &[usize]) -> Self {
        assert!(!dims.is_empty(), "shape must have at least one dimension");
        assert!(
            dims.len() <= MAX_RANK,
            "shape rank {} exceeds {MAX_RANK}",
            dims.len()
        );
        let mut d = [1usize; MAX_RANK];
        d[..dims.len()].copy_from_slice(dims);
        Shape {
            dims: d,
            rank: dims.len(),
        }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims[..self.rank]
    }

    pub fn dim(&self, i: usize) -> usize {
        self.dims[i]
    }

    pub fn n_elements(&self) -> usize {
        self.dims[..self.rank].iter().product()
    }

    /// Row-major strides in elements. `strides[rank-1]` is always 1.
    pub fn strides(&self) -> [usize; MAX_RANK] {
        let mut s = [0usize; MAX_RANK];
        let mut acc = 1usize;
        for i in (0..self.rank).rev() {
            s[i] = acc;
            acc *= self.dims[i];
        }
        s
    }

    /// Flat offset of a multi-dimensional index.
    pub fn offset(&self, index: &[usize]) -> usize {
        debug_assert_eq!(index.len(), self.rank);
        let strides = self.strides();
        index.iter().zip(&strides).map(|(i, s)| i * s).sum()
    }
}

/// An immutable contiguous view over f32 data with a shape attached.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    data: &'a [f32],
    shape: Shape,
}

impl<'a> TensorView<'a> {
    /// Panics if `data` does not match `shape` exactly. Views are constructed
    /// from buffers this crate allocated, so a mismatch is a bug.
    pub fn new(data: &'a [f32], shape: Shape) -> Self {
        assert_eq!(
            data.len(),
            shape.n_elements(),
            "data length {} does not match shape {:?}",
            data.len(),
            shape.dims()
        );
        TensorView { data, shape }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn as_slice(&self) -> &'a [f32] {
        self.data
    }

    /// One slice along the outermost axis, keeping the remaining shape.
    pub fn row(&self, i: usize) -> TensorView<'a> {
        let stride = self.shape.strides()[0];
        let sub = Shape::new(if self.shape.rank == 1 {
            &[1]
        } else {
            &self.shape.dims[1..self.shape.rank]
        });
        TensorView::new(&self.data[i * stride..(i + 1) * stride], sub)
    }
}

/// The mutable counterpart. Separate type rather than a flag, so a kernel that
/// writes cannot be handed a read-only view by accident.
#[derive(Debug)]
pub struct TensorViewMut<'a> {
    data: &'a mut [f32],
    shape: Shape,
}

impl<'a> TensorViewMut<'a> {
    pub fn new(data: &'a mut [f32], shape: Shape) -> Self {
        assert_eq!(
            data.len(),
            shape.n_elements(),
            "data length {} does not match shape {:?}",
            data.len(),
            shape.dims()
        );
        TensorViewMut { data, shape }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn as_slice(&self) -> &[f32] {
        self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.data
    }

    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        let stride = self.shape.strides()[0];
        &mut self.data[i * stride..(i + 1) * stride]
    }

    pub fn as_view(&self) -> TensorView<'_> {
        TensorView::new(self.data, self.shape)
    }
}
