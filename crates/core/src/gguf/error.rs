//! Errors produced while parsing a GGUF container.
//!
//! These are all *file* errors, not programmer errors: the byte buffer arrives
//! over the network from a CDN and may be truncated, corrupt, or hostile, so
//! every one of these is a normal `Err` rather than a panic.

use alloc::string::String;
use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufError {
    /// Ran off the end of the buffer.  Usually a truncated download.
    UnexpectedEof {
        at: usize,
        needed: usize,
        remaining: usize,
    },
    /// First four bytes were not `GGUF`.
    BadMagic([u8; 4]),
    /// We handle v2 and v3.  v1 used 32-bit lengths everywhere and is extinct.
    UnsupportedVersion(u32),
    /// Metadata value type tag outside 0..=12.
    UnknownValueType(u32),
    /// `ggml_type` tag we do not have a block layout for.  We refuse rather than
    /// guess, because guessing a block size silently corrupts every offset after it.
    UnknownGgmlType(u32),
    /// A length-prefixed string was not valid UTF-8.
    InvalidUtf8 { at: usize },
    /// A count/length in the header is larger than the entire remaining file.
    /// Checked *before* allocating, so a corrupt u64 cannot make us reserve 16 EiB.
    ImplausibleCount {
        what: &'static str,
        count: u64,
        remaining: usize,
    },
    /// Arrays of arrays of arrays... bounded so a crafted file cannot blow the stack.
    ArrayNestingTooDeep,
    /// `general.alignment` must be a non-zero power of two.
    BadAlignment(u64),
    /// GGML tensors are at most 4-dimensional.
    TooManyDims { name: String, n_dims: u32 },
    /// Tensor's data range does not fit inside the tensor-data section.
    TensorOutOfBounds {
        name: String,
        offset: u64,
        size: u64,
        section_len: u64,
    },
    /// Tensor offsets are relative to the (aligned) start of the data section and
    /// must themselves be aligned.
    MisalignedTensor {
        name: String,
        offset: u64,
        alignment: u64,
    },
    /// A quantised tensor's element count is not a whole number of blocks.
    NotBlockAligned {
        name: String,
        elements: u64,
        block: usize,
    },
    /// Arithmetic on file-supplied sizes overflowed.  Only reachable on hostile input.
    SizeOverflow { name: String },
    /// Required metadata key absent.
    MissingKey(String),
    /// Key present but holding a type we cannot coerce to what the caller wanted.
    WrongType {
        key: String,
        wanted: &'static str,
        found: &'static str,
    },
    /// Config values that contradict each other (e.g. head_count not divisible by
    /// head_count_kv, which would make GQA head mapping nonsense).
    InconsistentConfig(String),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use GgufError::*;
        match self {
            UnexpectedEof { at, needed, remaining } => write!(
                f,
                "unexpected end of file at byte {at}: needed {needed} more bytes, {remaining} remain \
                 (truncated download?)"
            ),
            BadMagic(m) => write!(f, "bad magic {m:02x?}: expected \"GGUF\""),
            UnsupportedVersion(v) => write!(f, "unsupported GGUF version {v} (this parser handles 2 and 3)"),
            UnknownValueType(t) => write!(f, "unknown metadata value type tag {t}"),
            UnknownGgmlType(t) => write!(f, "unknown ggml type id {t}: refusing to guess its block layout"),
            InvalidUtf8 { at } => write!(f, "invalid UTF-8 in string at byte {at}"),
            ImplausibleCount { what, count, remaining } => write!(
                f,
                "{what} count {count} cannot fit in the {remaining} remaining bytes; file is corrupt"
            ),
            ArrayNestingTooDeep => write!(f, "metadata array nesting exceeded the depth limit"),
            BadAlignment(a) => write!(f, "general.alignment = {a}: must be a non-zero power of two"),
            TooManyDims { name, n_dims } => write!(f, "tensor {name:?} has {n_dims} dims; ggml allows at most 4"),
            TensorOutOfBounds { name, offset, size, section_len } => write!(
                f,
                "tensor {name:?} spans [{offset}, {}) but the data section is only {section_len} bytes",
                offset + size
            ),
            MisalignedTensor { name, offset, alignment } => {
                write!(f, "tensor {name:?} offset {offset} is not a multiple of alignment {alignment}")
            }
            NotBlockAligned { name, elements, block } => write!(
                f,
                "tensor {name:?} has {elements} elements, not a whole number of {block}-element blocks"
            ),
            SizeOverflow { name } => write!(f, "size computation for tensor {name:?} overflowed"),
            MissingKey(k) => write!(f, "required metadata key {k:?} is missing"),
            WrongType { key, wanted, found } => {
                write!(f, "metadata key {key:?} is {found}, wanted {wanted}")
            }
            InconsistentConfig(m) => write!(f, "inconsistent model config: {m}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for GgufError {}
