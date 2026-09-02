//! Reference blocks pulled out of a real GGUF file.
//!
//! Hand-built blocks (see `tests.rs`) prove the bit arithmetic matches the spec
//! as I read it. They cannot prove I read the spec the same way ggml did. The
//! only thing that proves that is a block taken from a file llama.cpp produced,
//! dequantised by llama.cpp's own code, compared against ours.
//!
//! `tools/dump_gguf_blocks.py` does exactly that: it reads a real GGUF, hands
//! the raw block bytes to gguf-py's `dequantize()`, and regenerates
//! `fixtures_generated.rs`. Until it is run the table below is empty and the
//! fixture test passes vacuously -- deliberately, so the suite stays green on a
//! machine with no model file, while still failing loudly the moment real
//! fixtures exist and disagree with us.
//!
//! Run:
//! ```text
//! python3 tools/dump_gguf_blocks.py models/qwen2.5-0.5b-instruct-q4_k_m.gguf
//! ```

/// One block of real quantised data plus the reference implementation's output.
#[derive(Debug, Clone, Copy)]
pub struct BlockFixture {
    /// ggml type id, matching [`crate::gguf::GgmlType`].
    pub ggml_type: u32,
    /// Tensor it came from, for error messages.
    pub tensor: &'static str,
    /// Which block within that tensor.
    pub block_index: usize,
    /// The raw on-disk bytes of exactly one block.
    pub raw: &'static [u8],
    /// What the reference implementation produced for those bytes.
    pub expected: &'static [f32],
}

include!("fixtures_generated.rs");
