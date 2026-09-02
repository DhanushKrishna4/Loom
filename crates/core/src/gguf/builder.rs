//! A minimal GGUF *writer*, for tests only.
//!
//! We need real GGUF bytes to test the parser, and the real model is a 400 MB
//! download that must never enter the repo. So the parser tests build files in
//! memory instead. This writer is deliberately dumb and independent of the
//! reader: it does not share a single line of the parsing code, so a shared
//! misconception cannot make a test pass.

use alloc::string::String;
use alloc::vec::Vec;

use super::ggml_type::GgmlType;
use super::value::MetaType;

pub(crate) struct Builder {
    kv: Vec<u8>,
    kv_count: u64,
    ti: Vec<u8>,
    t_count: u64,
    data: Vec<u8>,
    alignment: u64,
    version: u32,
    magic: [u8; 4],
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

impl Builder {
    pub fn new() -> Self {
        Self {
            kv: Vec::new(),
            kv_count: 0,
            ti: Vec::new(),
            t_count: 0,
            data: Vec::new(),
            alignment: 32,
            version: 3,
            magic: *b"GGUF",
        }
    }

    pub fn version(mut self, v: u32) -> Self {
        self.version = v;
        self
    }

    pub fn magic(mut self, m: [u8; 4]) -> Self {
        self.magic = m;
        self
    }

    /// Sets the padding used by the writer. Emit `general.alignment` separately
    /// if you also want the parser to see it.
    pub fn alignment(mut self, a: u64) -> Self {
        self.alignment = a;
        self
    }

    fn kv_header(&mut self, key: &str, ty: MetaType) {
        put_str(&mut self.kv, key);
        self.kv.extend_from_slice(&(ty as u32).to_le_bytes());
        self.kv_count += 1;
    }

    pub fn u32(mut self, key: &str, v: u32) -> Self {
        self.kv_header(key, MetaType::U32);
        self.kv.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn i32(mut self, key: &str, v: i32) -> Self {
        self.kv_header(key, MetaType::I32);
        self.kv.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn u64(mut self, key: &str, v: u64) -> Self {
        self.kv_header(key, MetaType::U64);
        self.kv.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn u8(mut self, key: &str, v: u8) -> Self {
        self.kv_header(key, MetaType::U8);
        self.kv.push(v);
        self
    }

    pub fn i16(mut self, key: &str, v: i16) -> Self {
        self.kv_header(key, MetaType::I16);
        self.kv.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn f32(mut self, key: &str, v: f32) -> Self {
        self.kv_header(key, MetaType::F32);
        self.kv.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn f64(mut self, key: &str, v: f64) -> Self {
        self.kv_header(key, MetaType::F64);
        self.kv.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn bool(mut self, key: &str, v: bool) -> Self {
        self.kv_header(key, MetaType::Bool);
        self.kv.push(v as u8);
        self
    }

    pub fn string(mut self, key: &str, v: &str) -> Self {
        self.kv_header(key, MetaType::String);
        put_str(&mut self.kv, v);
        self
    }

    /// Writes a raw (already-encoded) string value, so tests can inject invalid UTF-8.
    pub fn raw_string(mut self, key: &str, bytes: &[u8]) -> Self {
        self.kv_header(key, MetaType::String);
        self.kv
            .extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        self.kv.extend_from_slice(bytes);
        self
    }

    fn array_header(&mut self, key: &str, elem: MetaType, n: usize) {
        self.kv_header(key, MetaType::Array);
        self.kv.extend_from_slice(&(elem as u32).to_le_bytes());
        self.kv.extend_from_slice(&(n as u64).to_le_bytes());
    }

    pub fn str_array(mut self, key: &str, vs: &[&str]) -> Self {
        self.array_header(key, MetaType::String, vs.len());
        for s in vs {
            put_str(&mut self.kv, s);
        }
        self
    }

    pub fn f32_array(mut self, key: &str, vs: &[f32]) -> Self {
        self.array_header(key, MetaType::F32, vs.len());
        for v in vs {
            self.kv.extend_from_slice(&v.to_le_bytes());
        }
        self
    }

    pub fn i32_array(mut self, key: &str, vs: &[i32]) -> Self {
        self.array_header(key, MetaType::I32, vs.len());
        for v in vs {
            self.kv.extend_from_slice(&v.to_le_bytes());
        }
        self
    }

    /// An array header claiming `claimed` elements while writing none, for
    /// testing the allocation guard.
    pub fn lying_array(mut self, key: &str, elem: MetaType, claimed: u64) -> Self {
        self.kv_header(key, MetaType::Array);
        self.kv.extend_from_slice(&(elem as u32).to_le_bytes());
        self.kv.extend_from_slice(&claimed.to_le_bytes());
        self
    }

    /// Appends a tensor, padding the data section so its offset stays aligned.
    pub fn tensor(mut self, name: &str, dims: &[u64], ty: GgmlType, bytes: &[u8]) -> Self {
        while !(self.data.len() as u64).is_multiple_of(self.alignment) {
            self.data.push(0);
        }
        let offset = self.data.len() as u64;
        self.push_tensor_info(name, dims, ty, offset);
        self.data.extend_from_slice(bytes);
        self
    }

    /// Declares a tensor at an arbitrary offset without writing data, for
    /// testing the bounds and alignment checks.
    pub fn tensor_info_only(mut self, name: &str, dims: &[u64], ty: GgmlType, offset: u64) -> Self {
        self.push_tensor_info(name, dims, ty, offset);
        self
    }

    fn push_tensor_info(&mut self, name: &str, dims: &[u64], ty: GgmlType, offset: u64) {
        put_str(&mut self.ti, name);
        self.ti
            .extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            self.ti.extend_from_slice(&d.to_le_bytes());
        }
        self.ti.extend_from_slice(&(ty as u32).to_le_bytes());
        self.ti.extend_from_slice(&offset.to_le_bytes());
        self.t_count += 1;
    }

    /// Raw tensor-info bytes, so tests can emit a bogus n_dims or type tag.
    pub fn raw_tensor_info(
        mut self,
        name: &str,
        n_dims: u32,
        dims: &[u64],
        type_tag: u32,
        offset: u64,
    ) -> Self {
        put_str(&mut self.ti, name);
        self.ti.extend_from_slice(&n_dims.to_le_bytes());
        for d in dims {
            self.ti.extend_from_slice(&d.to_le_bytes());
        }
        self.ti.extend_from_slice(&type_tag.to_le_bytes());
        self.ti.extend_from_slice(&offset.to_le_bytes());
        self.t_count += 1;
        self
    }

    pub fn build(self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.magic);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&self.t_count.to_le_bytes());
        out.extend_from_slice(&self.kv_count.to_le_bytes());
        out.extend_from_slice(&self.kv);
        out.extend_from_slice(&self.ti);
        while !(out.len() as u64).is_multiple_of(self.alignment) {
            out.push(0);
        }
        out.extend_from_slice(&self.data);
        out
    }
}

/// The metadata of Qwen2.5-0.5B-Instruct, with the vocab arrays shrunk to
/// `vocab` entries. Numbers are the published config for that model.
pub(crate) fn qwen2_5_0_5b(vocab: &[&str]) -> Builder {
    Builder::new()
        .string("general.architecture", "qwen2")
        .string("general.name", "Qwen2.5-0.5B-Instruct")
        .u32("general.file_type", 15) // MOSTLY_Q4_K_M
        .u32("general.quantization_version", 2)
        .u32("qwen2.block_count", 24)
        .u32("qwen2.context_length", 32768)
        .u32("qwen2.embedding_length", 896)
        .u32("qwen2.feed_forward_length", 4864)
        .u32("qwen2.attention.head_count", 14)
        .u32("qwen2.attention.head_count_kv", 2)
        .f32("qwen2.attention.layer_norm_rms_epsilon", 1e-6)
        .f32("qwen2.rope.freq_base", 1_000_000.0)
        // The real file has NO `rope.dimension_count`, so the default --
        // rotate every head dimension -- is what actually applies.
        .string("tokenizer.ggml.model", "gpt2")
        .string("tokenizer.ggml.pre", "qwen2")
        .str_array("tokenizer.ggml.tokens", vocab)
        .u32("tokenizer.ggml.bos_token_id", 151643)
        .u32("tokenizer.ggml.eos_token_id", 151645)
        .u32("tokenizer.ggml.padding_token_id", 151643)
        .bool("tokenizer.ggml.add_bos_token", false)
}

/// Helper for tests that need a vocab of a particular size without caring about
/// its contents. Leaks are irrelevant: this only runs under `cargo test`.
pub(crate) fn dummy_vocab(n: usize) -> Vec<String> {
    use alloc::format;
    (0..n).map(|i| format!("tok{i}")).collect()
}
