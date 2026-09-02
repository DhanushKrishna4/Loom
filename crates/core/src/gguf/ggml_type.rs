//! The `ggml_type` enum and its block geometry.
//!
//! Every quantised format in ggml is a *block* format: `block_elems` logical f32
//! values are stored in `block_bytes` bytes, and the whole tensor is a dense
//! array of such blocks in row-major element order.  Getting `block_bytes` wrong
//! for even one type silently shifts every tensor offset after it, so the table
//! below is transcribed directly from the `block_*` struct definitions in
//! `ggml-common.h` with the arithmetic spelled out in comments.
//!
//! `QK_K` (the k-quant super-block size) is 256 throughout; llama.cpp can be
//! built with `QK_K = 64` but no published GGUF uses it.

use super::error::GgufError;

/// ggml's k-quant super-block size.
pub const QK_K: usize = 256;

#[allow(non_camel_case_types)] // keep ggml's own spelling: Q4_K, not Q4K
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // 4 and 5 were Q4_2 / Q4_3 and were removed from ggml; the ids are never reused.
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Result<Self, GgufError> {
        use GgmlType::*;
        Ok(match v {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2_K,
            11 => Q3_K,
            12 => Q4_K,
            13 => Q5_K,
            14 => Q6_K,
            15 => Q8_K,
            16 => IQ2_XXS,
            17 => IQ2_XS,
            18 => IQ3_XXS,
            19 => IQ1_S,
            20 => IQ4_NL,
            21 => IQ3_S,
            22 => IQ2_S,
            23 => IQ4_XS,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            29 => IQ1_M,
            30 => BF16,
            34 => TQ1_0,
            35 => TQ2_0,
            other => return Err(GgufError::UnknownGgmlType(other)),
        })
    }

    /// Number of logical elements per stored block.
    pub const fn block_elems(self) -> usize {
        use GgmlType::*;
        match self {
            F32 | F16 | BF16 | F64 | I8 | I16 | I32 | I64 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 | IQ4_NL => 32,
            _ => QK_K,
        }
    }

    /// Bytes occupied by one block on disk.
    pub const fn block_bytes(self) -> usize {
        use GgmlType::*;
        match self {
            F32 => 4,
            F16 => 2,
            BF16 => 2,
            F64 => 8,
            I8 => 1,
            I16 => 2,
            I32 => 4,
            I64 => 8,

            // { f16 d; u8 qs[16] }
            Q4_0 => 18,
            // { f16 d; f16 m; u8 qs[16] }
            Q4_1 => 20,
            // { f16 d; u8 qh[4]; u8 qs[16] }
            Q5_0 => 22,
            // { f16 d; f16 m; u8 qh[4]; u8 qs[16] }
            Q5_1 => 24,
            // { f16 d; i8 qs[32] }
            Q8_0 => 34,
            // { f32 d; f32 s; i8 qs[32] }
            Q8_1 => 36,

            // { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin }
            Q2_K => 84,
            // { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d }
            Q3_K => 110,
            // { f16 d; f16 dmin; u8 scales[12]; u8 qs[128] }
            Q4_K => 144,
            // { f16 d; f16 dmin; u8 scales[12]; u8 qh[32]; u8 qs[128] }
            Q5_K => 176,
            // { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d }
            Q6_K => 210,
            // { f32 d; i8 qs[256]; i16 bsums[16] }
            Q8_K => 292,

            IQ2_XXS => 66, // { f16 d; u16 qs[32] }
            IQ2_XS => 74,  // { f16 d; u16 qs[32]; u8 scales[8] }
            IQ2_S => 82,   // { f16 d; u8 qs[64]; u8 qh[8]; u8 scales[8] }
            IQ3_XXS => 98, // { f16 d; u8 qs[96] }
            IQ3_S => 110,  // { f16 d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4] }
            IQ1_S => 50,   // { f16 d; u8 qs[32]; u16 qh[8] }
            IQ1_M => 56,   // { u8 qs[32]; u8 qh[16]; u8 scales[8] }
            IQ4_NL => 18,  // { f16 d; u8 qs[16] }
            IQ4_XS => 136, // { f16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128] }
            TQ1_0 => 54,   // { u8 qs[48]; u8 qh[4]; f16 d }
            TQ2_0 => 66,   // { u8 qs[64]; f16 d }
        }
    }

    /// ggml's own name for the type, as it appears in llama.cpp logs.
    #[rustfmt::skip] // the alignment is the point: this is a lookup table
    pub const fn name(self) -> &'static str {
        use GgmlType::*;
        match self {
            F32 => "f32", F16 => "f16", BF16 => "bf16", F64 => "f64",
            I8 => "i8", I16 => "i16", I32 => "i32", I64 => "i64",
            Q4_0 => "q4_0", Q4_1 => "q4_1", Q5_0 => "q5_0", Q5_1 => "q5_1",
            Q8_0 => "q8_0", Q8_1 => "q8_1",
            Q2_K => "q2_K", Q3_K => "q3_K", Q4_K => "q4_K", Q5_K => "q5_K",
            Q6_K => "q6_K", Q8_K => "q8_K",
            IQ2_XXS => "iq2_xxs", IQ2_XS => "iq2_xs", IQ2_S => "iq2_s",
            IQ3_XXS => "iq3_xxs", IQ3_S => "iq3_s",
            IQ1_S => "iq1_s", IQ1_M => "iq1_m",
            IQ4_NL => "iq4_nl", IQ4_XS => "iq4_xs",
            TQ1_0 => "tq1_0", TQ2_0 => "tq2_0",
        }
    }

    /// True for everything that is not a plain float/int array.
    pub const fn is_quantized(self) -> bool {
        self.block_elems() > 1
    }

    /// Bytes needed to store `n_elements` of this type, or `None` on overflow or
    /// if `n_elements` is not a whole number of blocks.
    pub fn bytes_for(self, n_elements: u64) -> Option<u64> {
        let be = self.block_elems() as u64;
        if !n_elements.is_multiple_of(be) {
            return None;
        }
        (n_elements / be).checked_mul(self.block_bytes() as u64)
    }
}

impl core::fmt::Display for GgmlType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}
