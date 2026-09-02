//! GGUF v2/v3 container parsing.
//!
//! # File layout
//!
//! ```text
//! magic "GGUF"            4 bytes
//! version                 u32          (2 or 3; identical layout little-endian)
//! tensor_count            u64
//! metadata_kv_count       u64
//! metadata_kv[]           key:string, type:u32, value
//! tensor_info[]           name:string, n_dims:u32, dims:u64[n_dims], type:u32, offset:u64
//! <padding to `general.alignment`, measured from the start of the FILE>
//! tensor data             blob; each tensor's `offset` is relative to here
//! ```
//!
//! # What this module deliberately does not do
//!
//! It never copies tensor data.  [`Gguf::tensor_data`] hands back a subslice of
//! the caller's buffer.  A 0.5B Q4_K model is ~400 MB; holding a second copy is
//! the fastest way to hit the wasm memory ceiling, so the borrow is load-bearing,
//! not a micro-optimisation.

#![deny(unsafe_code)]

mod config;
mod error;
mod ggml_type;
mod reader;
mod value;

// Test-only helpers. Gated on `std` too so that
// `cargo test --no-default-features` still compiles (it just runs nothing).
#[cfg(all(test, feature = "std"))]
mod builder;
/// The in-memory GGUF writer, for tests in other modules of this crate.
#[cfg(all(test, feature = "std"))]
pub(crate) mod tests_support {
    pub(crate) use super::builder::Builder;
}
#[cfg(all(test, feature = "std"))]
mod tests;

pub use config::ModelConfig;
pub use error::GgufError;
pub use ggml_type::{GgmlType, QK_K};
pub use value::{Array, MetaType, Value};

use alloc::string::ToString;
use alloc::vec::Vec;
use reader::Reader;

/// ggml tensors are at most 4-dimensional.
pub const GGML_MAX_DIMS: usize = 4;

/// Default tensor-data alignment when `general.alignment` is absent.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// Description of one tensor, with its data still in the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo<'a> {
    pub name: &'a str,
    /// Dimensions in **ggml order**: `dims[0]` is the fastest-varying axis.
    ///
    /// This is the reverse of how PyTorch prints shapes, and it is a standing
    /// trap.  A Qwen2 `attn_q.weight` that torch calls `[896, 896]` (out, in)
    /// appears here as `dims = [896, 896]` meaning "896 rows of 896 contiguous
    /// elements", i.e. row length first.  For a non-square matrix like
    /// `ffn_gate.weight` torch says `[4864, 896]` (out, in) and ggml stores
    /// `dims = [896, 4864]`.  Use [`Self::shape_row_major`] when comparing
    /// against reference tensors.
    pub dims: [u64; GGML_MAX_DIMS],
    pub n_dims: u32,
    pub ggml_type: GgmlType,
    /// Byte offset from the start of the tensor-data section (not the file).
    pub offset: u64,
    /// Size on disk, derived from `dims` and `ggml_type`.  Validated at parse.
    pub byte_size: u64,
}

impl<'a> TensorInfo<'a> {
    pub fn n_elements(&self) -> u64 {
        self.dims[..self.n_dims as usize].iter().product()
    }

    /// Dimensions reordered outermost-first, matching how PyTorch/numpy print
    /// shapes.  `[896, 4864]` in ggml order becomes `[4864, 896]` here.
    pub fn shape_row_major(&self) -> Vec<u64> {
        self.dims[..self.n_dims as usize]
            .iter()
            .rev()
            .copied()
            .collect()
    }

    /// Row length in elements, i.e. the contiguous run: `dims[0]`.
    pub fn row_len(&self) -> u64 {
        self.dims[0]
    }

    /// Number of rows, i.e. everything above the innermost axis.
    pub fn n_rows(&self) -> u64 {
        self.dims[1..self.n_dims as usize].iter().product()
    }
}

/// A parsed GGUF file, borrowing the buffer it was parsed from.
#[derive(Debug)]
pub struct Gguf<'a> {
    file: &'a [u8],
    pub version: u32,
    pub alignment: u64,
    /// Absolute byte offset in `file` where the tensor-data section begins.
    pub tensor_data_offset: usize,
    /// Metadata in file order.  A `Vec` rather than a map: there are ~30 keys, a
    /// linear scan beats a hash, and it keeps the crate free of `hashbrown`.
    pub metadata: Vec<(&'a str, Value<'a>)>,
    pub tensors: Vec<TensorInfo<'a>>,
}

impl<'a> Gguf<'a> {
    /// Parse the header, metadata and tensor table.  Tensor *data* is only
    /// bounds-checked, never touched.
    pub fn parse(file: &'a [u8]) -> Result<Self, GgufError> {
        let mut r = Reader::new(file);

        let magic = r.take(4)?;
        if magic != b"GGUF" {
            let mut m = [0u8; 4];
            m.copy_from_slice(magic);
            return Err(GgufError::BadMagic(m));
        }

        // v1 used u32 counts and is long dead; v2 and v3 are byte-identical for
        // little-endian files (v3 only added the big-endian variant).
        let version = r.read_u32()?;
        if !(2..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = r.read_len("tensor")?;
        let kv_count = r.read_len("metadata kv")?;

        let mut metadata = Vec::with_capacity(kv_count);
        for _ in 0..kv_count {
            let key = r.read_str()?;
            let ty = MetaType::from_u32(r.read_u32()?)?;
            let val = Value::read(&mut r, ty, 0)?;
            metadata.push((key, val));
        }

        // `general.alignment` governs the padding before the data section, so it
        // has to be resolved from the metadata we just read, before we can know
        // where tensor data starts.
        let alignment = match metadata.iter().find(|(k, _)| *k == "general.alignment") {
            Some((_, v)) => v.as_u64().ok_or_else(|| GgufError::WrongType {
                key: "general.alignment".to_string(),
                wanted: "integer",
                found: v.type_name(),
            })?,
            None => DEFAULT_ALIGNMENT,
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(GgufError::BadAlignment(alignment));
        }

        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            tensors.push(read_tensor_info(&mut r)?);
        }

        // Padding is measured from the start of the file, not from the end of
        // the tensor table.
        let unpadded = r.pos() as u64;
        let tensor_data_offset = unpadded.next_multiple_of(alignment);
        if tensor_data_offset > file.len() as u64 {
            return Err(GgufError::UnexpectedEof {
                at: unpadded as usize,
                needed: (tensor_data_offset - unpadded) as usize,
                remaining: file.len() - unpadded as usize,
            });
        }
        let tensor_data_offset = tensor_data_offset as usize;
        let section_len = (file.len() - tensor_data_offset) as u64;

        // Validate every tensor's range now, so `tensor_data()` can be infallible
        // and the hot path never re-checks.
        for t in &tensors {
            if t.offset % alignment != 0 {
                return Err(GgufError::MisalignedTensor {
                    name: t.name.to_string(),
                    offset: t.offset,
                    alignment,
                });
            }
            let end = t
                .offset
                .checked_add(t.byte_size)
                .ok_or_else(|| GgufError::SizeOverflow {
                    name: t.name.to_string(),
                })?;
            if end > section_len {
                return Err(GgufError::TensorOutOfBounds {
                    name: t.name.to_string(),
                    offset: t.offset,
                    size: t.byte_size,
                    section_len,
                });
            }
        }

        Ok(Gguf {
            file,
            version,
            alignment,
            tensor_data_offset,
            metadata,
            tensors,
        })
    }

    /// The raw bytes of a tensor, borrowed from the input buffer. No copy.
    ///
    /// Infallible because [`Gguf::parse`] already proved the range is inside the
    /// data section; a panic here would be a parser bug, not bad input.
    pub fn tensor_data(&self, info: &TensorInfo<'a>) -> &'a [u8] {
        let start = self.tensor_data_offset + info.offset as usize;
        &self.file[start..start + info.byte_size as usize]
    }

    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo<'a>> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Total bytes of tensor data described by the table.
    pub fn tensor_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.byte_size).sum()
    }

    pub fn get(&self, key: &str) -> Option<&Value<'a>> {
        self.metadata
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| v)
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key)?.as_u64()
    }

    pub fn get_usize(&self, key: &str) -> Option<usize> {
        usize::try_from(self.get_u64(key)?).ok()
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        u32::try_from(self.get_u64(key)?).ok()
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.get(key)?.as_f32()
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key)?.as_bool()
    }

    pub fn get_str(&self, key: &str) -> Option<&'a str> {
        self.get(key)?.as_str()
    }

    pub fn get_array(&self, key: &str) -> Option<&Array<'a>> {
        self.get(key)?.as_array()
    }

    /// As [`Self::get`], but an absent key is an error rather than `None`.
    pub fn require(&self, key: &str) -> Result<&Value<'a>, GgufError> {
        self.get(key)
            .ok_or_else(|| GgufError::MissingKey(key.to_string()))
    }

    pub fn require_u64(&self, key: &str) -> Result<u64, GgufError> {
        let v = self.require(key)?;
        v.as_u64().ok_or_else(|| GgufError::WrongType {
            key: key.to_string(),
            wanted: "integer",
            found: v.type_name(),
        })
    }

    pub fn require_usize(&self, key: &str) -> Result<usize, GgufError> {
        let v = self.require_u64(key)?;
        usize::try_from(v).map_err(|_| GgufError::WrongType {
            key: key.to_string(),
            wanted: "integer fitting in usize",
            found: "u64",
        })
    }

    pub fn require_f32(&self, key: &str) -> Result<f32, GgufError> {
        let v = self.require(key)?;
        v.as_f32().ok_or_else(|| GgufError::WrongType {
            key: key.to_string(),
            wanted: "float",
            found: v.type_name(),
        })
    }

    pub fn require_str(&self, key: &str) -> Result<&'a str, GgufError> {
        let v = self.require(key)?;
        v.as_str().ok_or_else(|| GgufError::WrongType {
            key: key.to_string(),
            wanted: "string",
            found: v.type_name(),
        })
    }
}

fn read_tensor_info<'a>(r: &mut Reader<'a>) -> Result<TensorInfo<'a>, GgufError> {
    let name = r.read_str()?;
    let n_dims = r.read_u32()?;
    if n_dims as usize > GGML_MAX_DIMS {
        return Err(GgufError::TooManyDims {
            name: name.to_string(),
            n_dims,
        });
    }
    // Unused axes are 1, so `n_elements` is just the product of all four.
    let mut dims = [1u64; GGML_MAX_DIMS];
    for d in dims.iter_mut().take(n_dims as usize) {
        *d = r.read_u64()?;
    }
    let ggml_type = GgmlType::from_u32(r.read_u32()?)?;
    let offset = r.read_u64()?;

    let n_elements = dims[..n_dims as usize]
        .iter()
        .try_fold(1u64, |a, &b| a.checked_mul(b))
        .ok_or_else(|| GgufError::SizeOverflow {
            name: name.to_string(),
        })?;

    let byte_size = ggml_type.bytes_for(n_elements).ok_or_else(|| {
        if n_elements % ggml_type.block_elems() as u64 != 0 {
            GgufError::NotBlockAligned {
                name: name.to_string(),
                elements: n_elements,
                block: ggml_type.block_elems(),
            }
        } else {
            GgufError::SizeOverflow {
                name: name.to_string(),
            }
        }
    })?;

    Ok(TensorInfo {
        name,
        dims,
        n_dims,
        ggml_type,
        offset,
        byte_size,
    })
}
