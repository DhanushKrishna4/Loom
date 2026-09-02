//! Fused quantised dot products: the decode hot path.
//!
//! # The idea
//!
//! `matvec_dequant` unpacks a weight row into a scratch buffer and then does an
//! f32 dot product over it. That writes ~3.5 KB to memory and reads it straight
//! back, per row, per matmul — and at batch size 1 there is no reuse to amortise
//! it against, so the memory traffic *is* the cost.
//!
//! Instead: quantise the activation vector to 8-bit **once per matmul**, then
//! make the inner loop an integer dot product between two quantised blocks, with
//! the scales applied once at the end of each block. The weights are unpacked
//! into registers, used, and dropped. Nothing dequantised ever reaches memory.
//!
//! # Why the reconstruction form decides the kernel
//!
//! For the 32-element formats the reconstruction is *linear* — `value = d * q`,
//! with `q` already signed once the bias is folded in — so the whole block
//! collapses to one integer dot times `d_w * d_a`.
//!
//! Q4_K is *affine*: `value = d * scale * q - dmin * min`. The constant term does
//! not vanish, so each sub-block also needs the plain sum of the activation
//! quants:
//!
//! ```text
//! sum_i (d*sc*q_i - dmin*m) * a_i
//!   = d_a * ( d*sc * sum_i(q_i * qa_i)  -  dmin*m * sum_i(qa_i) )
//!                    ^^^^ integer dot        ^^^^ integer sum, precomputed
//! ```
//!
//! That second term is the entire reason [`ActivationQ8`] carries per-block sums.
//! ggml solves the same problem by giving its activation format (`Q8_1`, `Q8_K`)
//! exactly the same extra field.
//!
//! Q6_K is linear again but its scales are per *16* elements, so the accumulator
//! is split sixteen ways instead of one.
//!
//! # Status
//!
//! Scalar. The i8 multiply-accumulate is what SIMD128 wants (`i16x8.extmul`,
//! `i32x4.dot_i16x8_s`), and that is step 9. This version exists to be correct
//! and to be the thing the SIMD version is measured against.

use alloc::vec;
use alloc::vec::Vec;

use super::f16::read_f16_le;
use super::k_quants::scale_min_k4;
use crate::gguf::GgmlType;
use crate::math::round_half_away_from_zero;

/// Elements per activation block. Matches every 32-element weight format, and
/// divides the 256-element k-quant super-blocks exactly eight ways.
pub const ACT_BLOCK: usize = 32;

/// An activation vector quantised to signed 8-bit, one scale per 32 elements.
///
/// Allocated once and refilled per matmul: at ~9 matmuls per layer this would
/// otherwise be the busiest allocation site in the engine.
#[derive(Debug, Clone)]
pub struct ActivationQ8 {
    /// Per-block scale, kept in f32 rather than f16: it is recomputed every
    /// matmul, so there is nothing to be gained from shrinking it and a little
    /// accuracy to lose.
    scales: Vec<f32>,
    qs: Vec<i8>,
    /// Per-block sum of `qs`, for the affine formats. See the module docs.
    sums: Vec<i32>,
    len: usize,
}

impl ActivationQ8 {
    /// Capacity for `n` elements. `n` must be a multiple of [`ACT_BLOCK`].
    pub fn new(n: usize) -> Self {
        assert!(
            n.is_multiple_of(ACT_BLOCK),
            "activation length {n} must be a multiple of {ACT_BLOCK}"
        );
        let nb = n / ACT_BLOCK;
        ActivationQ8 {
            scales: vec![0.0; nb],
            qs: vec![0; n],
            sums: vec![0; nb],
            len: n,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn n_blocks(&self) -> usize {
        self.len / ACT_BLOCK
    }

    /// Reconstruct the quantised activation as f32.
    ///
    /// For tests: it lets a fused kernel be compared against
    /// "dequantise the weights and dot against the *same* quantised
    /// activation", which isolates the kernel's arithmetic from the accuracy
    /// cost of quantising the activation at all.
    pub fn dequantize(&self, out: &mut [f32]) {
        assert_eq!(out.len(), self.len);
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.scales[i / ACT_BLOCK] * self.qs[i] as f32;
        }
    }

    pub fn byte_len(&self) -> usize {
        self.scales.len() * 4 + self.qs.len() + self.sums.len() * 4
    }

    /// Quantise `x` in place. `x.len()` may be shorter than the capacity, as
    /// long as it is a whole number of blocks.
    pub fn quantize(&mut self, x: &[f32]) {
        assert!(
            x.len() <= self.qs.len(),
            "activation is longer than the buffer"
        );
        assert!(
            x.len().is_multiple_of(ACT_BLOCK),
            "activation must be block-aligned"
        );
        self.len = x.len();

        for (b, chunk) in x.as_chunks::<ACT_BLOCK>().0.iter().enumerate() {
            let amax = chunk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            // 127 rather than 128 keeps the scale symmetric, so an all-zero
            // block dequantises to exact zeros.
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            self.scales[b] = d;

            let mut sum = 0i32;
            let out = &mut self.qs[b * ACT_BLOCK..(b + 1) * ACT_BLOCK];
            for (o, &v) in out.iter_mut().zip(chunk) {
                let q = round_half_away_from_zero(v * id).clamp(-127.0, 127.0) as i32;
                *o = q as i8;
                sum += q;
            }
            self.sums[b] = sum;
        }
    }

    #[inline]
    fn block(&self, b: usize) -> &[i8] {
        &self.qs[b * ACT_BLOCK..(b + 1) * ACT_BLOCK]
    }
}

/// `sum(a[i] * b[i])` over 32 signed bytes, in i32.
///
/// The products fit comfortably: 127*127*32 is about 516k, far inside i32.
/// SIMD128 build: two 16-byte lanes through `extmul` + `extadd_pairwise`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn idot32(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), 32);
    debug_assert_eq!(b.len(), 32);
    idot32_simd128(a, b)
}

/// `sum(a[i] * b[i])` over 16 signed bytes.
///
/// Q6_K carries a scale per 16 weights, and two consecutive groups cannot be
/// merged into one 32-wide dot because their scales differ — so the narrower
/// width earns its own kernel rather than being emulated with masking.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
#[inline]
fn idot16(a: &[i8], b: &[i8]) -> i32 {
    use core::arch::wasm32::*;
    debug_assert_eq!(a.len(), 16);
    debug_assert_eq!(b.len(), 16);
    // SAFETY: both slices are exactly 16 bytes, so one unaligned 16-byte load
    // from each is in bounds. `v128_load` has no alignment requirement.
    unsafe {
        let x = v128_load(a.as_ptr() as *const v128);
        let y = v128_load(b.as_ptr() as *const v128);
        let lo = i16x8_extmul_low_i8x16(x, y);
        let hi = i16x8_extmul_high_i8x16(x, y);
        let acc = i32x4_add(
            i32x4_extadd_pairwise_i16x8(lo),
            i32x4_extadd_pairwise_i16x8(hi),
        );
        i32x4_extract_lane::<0>(acc)
            + i32x4_extract_lane::<1>(acc)
            + i32x4_extract_lane::<2>(acc)
            + i32x4_extract_lane::<3>(acc)
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline(always)]
fn idot16(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), 16);
    debug_assert_eq!(b.len(), 16);
    let mut acc = 0i32;
    for i in 0..16 {
        acc += a[i] as i32 * b[i] as i32;
    }
    acc
}

/// Everywhere else: the plain loop, which LLVM autovectorises well on its own.
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline(always)]
fn idot32(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), 32);
    debug_assert_eq!(b.len(), 32);
    idot32_scalar(a, b)
}

/// The portable fallback. Not compiled when the SIMD128 path is available, so
/// there is no dead code in the shipping wasm build.
///
/// `inline(always)`, not `inline`: the native build was declining the hint, and
/// the resulting call boundary stopped LLVM eliminating the slice bounds checks
/// — worth a 2.2x regression in batched prefill on aarch64.
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline(always)]
fn idot32_scalar(a: &[i8], b: &[i8]) -> i32 {
    let mut acc = 0i32;
    for i in 0..32 {
        acc += a[i] as i32 * b[i] as i32;
    }
    acc
}

/// Hand-written SIMD128 for the same dot product.
///
/// `i8 * i8` fits in `i16` (127*127 = 16129), so `extmul` gives exact products
/// without widening first, and `extadd_pairwise` folds them into `i32` lanes
/// where the running sum cannot overflow.
///
/// This is the one place in the crate that uses `unsafe`. The whole crate is
/// otherwise `#![deny(unsafe_code)]`, and this is an explicit, justified
/// exception rather than a general relaxation.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
#[inline]
fn idot32_simd128(a: &[i8], b: &[i8]) -> i32 {
    use core::arch::wasm32::*;

    // SAFETY: both callers pass exactly 32 bytes (debug-asserted above and
    // guaranteed by the fixed-size blocks every caller slices), so the two
    // 16-byte loads at offsets 0 and 16 are in bounds. `v128_load` is unaligned.
    unsafe {
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let mut acc = i32x4_splat(0);
        for off in [0usize, 16] {
            let x = v128_load(ap.add(off) as *const v128);
            let y = v128_load(bp.add(off) as *const v128);
            let lo = i16x8_extmul_low_i8x16(x, y);
            let hi = i16x8_extmul_high_i8x16(x, y);
            acc = i32x4_add(acc, i32x4_extadd_pairwise_i16x8(lo));
            acc = i32x4_add(acc, i32x4_extadd_pairwise_i16x8(hi));
        }
        i32x4_extract_lane::<0>(acc)
            + i32x4_extract_lane::<1>(acc)
            + i32x4_extract_lane::<2>(acc)
            + i32x4_extract_lane::<3>(acc)
    }
}

/// One Q8_0 weight row against a quantised activation.
///
/// `value = d_w * q_w`, so the block reduces to `d_w * d_a * idot`.
pub fn row_dot_q8_0(row: &[u8], a: &ActivationQ8) -> f32 {
    let mut acc = 0.0f32;
    for (b, blk) in row.as_chunks::<34>().0.iter().enumerate() {
        let d = read_f16_le(blk, 0);
        let qw = &blk[2..34];
        let qa = a.block(b);
        // Deliberately NOT routed through the shared SIMD `idot32`.
        //
        // Doing so needs the stored bytes reinterpreted as `i8`, which without
        // `unsafe` means a 32-byte copy -- and that measured *slower*
        // (0.128 ns/element against 0.122) than leaving this loop for LLVM to
        // autovectorise. Q8_0 is the only format whose quants are already 8-bit
        // and need no unpacking, so it is the only one where the copy is not
        // amortised against real work. See the README's table.
        let mut dot = 0i32;
        for i in 0..32 {
            // `qs` is signed; widening it as u8 would flip every negative weight.
            dot += (qw[i] as i8) as i32 * qa[i] as i32;
        }
        acc += d * a.scales[b] * dot as f32;
    }
    acc
}

/// Bit-to-byte expansion table: `BIT_TO_HIGH[b][k]` is `0x10` when bit `k` of
/// `b` is set, and 0 otherwise.
///
/// Q5_0's fifth bits arrive as a 32-bit mask, one bit per element, which has to
/// become one *byte* per element before it can be OR-ed into the nibbles.
/// Extracting them one at a time is 32 dependent shift-and-mask pairs that
/// nothing can vectorise. Four table lookups produce the same 32 bytes, and the
/// loop that consumes them is then uniform enough for LLVM to widen.
///
/// 2 KiB of table, and llama.cpp uses the same trick for its NEON path.
static BIT_TO_HIGH: [[u8; 8]; 256] = build_bit_table();

const fn build_bit_table() -> [[u8; 8]; 256] {
    let mut t = [[0u8; 8]; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut k = 0usize;
        while k < 8 {
            t[b][k] = if (b >> k) & 1 == 1 { 0x10 } else { 0 };
            k += 1;
        }
        b += 1;
    }
    t
}

/// One Q5_0 weight row. `value = d_w * (q - 16)`, with the fifth bit in `qh`.
pub fn row_dot_q5_0(row: &[u8], a: &ActivationQ8) -> f32 {
    let mut acc = 0.0f32;
    let mut w = [0i8; 32];
    for (b, blk) in row.as_chunks::<22>().0.iter().enumerate() {
        let d = read_f16_le(blk, 0);
        let qs = &blk[6..22];

        // Bit i of qh is element i's fifth bit, running straight through 0..32 --
        // it does not follow the low-half/high-half split that qs uses. So
        // elements 0..15 take their high bits from qh bytes 0 and 1, and
        // elements 16..31 from bytes 2 and 3.
        let hi_lo0 = &BIT_TO_HIGH[blk[2] as usize];
        let hi_lo1 = &BIT_TO_HIGH[blk[3] as usize];
        let hi_hi0 = &BIT_TO_HIGH[blk[4] as usize];
        let hi_hi1 = &BIT_TO_HIGH[blk[5] as usize];

        for j in 0..8 {
            w[j] = (((qs[j] & 0x0f) | hi_lo0[j]) as i32 - 16) as i8;
            w[j + 8] = (((qs[j + 8] & 0x0f) | hi_lo1[j]) as i32 - 16) as i8;
            w[j + 16] = (((qs[j] >> 4) | hi_hi0[j]) as i32 - 16) as i8;
            w[j + 24] = (((qs[j + 8] >> 4) | hi_hi1[j]) as i32 - 16) as i8;
        }
        acc += d * a.scales[b] * idot32(&w, a.block(b)) as f32;
    }
    acc
}

/// One Q4_0 weight row. `value = d_w * (nibble - 8)`.
pub fn row_dot_q4_0(row: &[u8], a: &ActivationQ8) -> f32 {
    let mut acc = 0.0f32;
    let mut w = [0i8; 32];
    for (b, blk) in row.as_chunks::<18>().0.iter().enumerate() {
        let d = read_f16_le(blk, 0);
        let qs = &blk[2..18];
        for j in 0..16 {
            w[j] = ((qs[j] & 0x0f) as i32 - 8) as i8;
            w[j + 16] = ((qs[j] >> 4) as i32 - 8) as i8;
        }
        acc += d * a.scales[b] * idot32(&w, a.block(b)) as f32;
    }
    acc
}

/// One Q4_K weight row.
///
/// The affine case: each 32-element sub-block contributes
/// `d_a * (d*sc*idot - dmin*m*sum)`. Sub-blocks line up one-to-one with
/// activation blocks, which is why [`ACT_BLOCK`] is 32.
pub fn row_dot_q4_k(row: &[u8], a: &ActivationQ8) -> f32 {
    let mut acc = 0.0f32;
    let mut w = [0i8; 32];

    for (sb, blk) in row.as_chunks::<144>().0.iter().enumerate() {
        let d = read_f16_le(blk, 0);
        let dmin = read_f16_le(blk, 2);
        let scales: &[u8; 12] = blk[4..16].try_into().expect("12 bytes");
        let qs = &blk[16..144];

        for s in 0..8 {
            let (sc, m) = scale_min_k4(s, scales);
            // Sub-blocks 2c and 2c+1 share the 32 bytes at qs[c*32..]: the even
            // one takes low nibbles, the odd one high nibbles.
            let base = (s / 2) * 32;
            if s % 2 == 0 {
                for l in 0..32 {
                    w[l] = (qs[base + l] & 0x0f) as i8;
                }
            } else {
                for l in 0..32 {
                    w[l] = (qs[base + l] >> 4) as i8;
                }
            }

            let ab = sb * 8 + s;
            let dot = idot32(&w, a.block(ab)) as f32;
            // The `- dmin*m*sum` term is the whole reason `sums` exists.
            acc += a.scales[ab] * (d * sc as f32 * dot - dmin * m as f32 * a.sums[ab] as f32);
        }
    }
    acc
}

/// One Q6_K weight row.
///
/// Linear (`value = d * sc * (q - 32)`) but with a scale per *16* elements, so
/// the accumulator splits sixteen ways per super-block instead of eight.
pub fn row_dot_q6_k(row: &[u8], a: &ActivationQ8) -> f32 {
    let mut acc = 0.0f32;

    for (sb, blk) in row.as_chunks::<210>().0.iter().enumerate() {
        let d = read_f16_le(blk, 208);
        let ql_all = &blk[0..128];
        let qh_all = &blk[128..192];
        let sc_all = &blk[192..208];

        // One accumulator per 16-element scale group.
        let mut group = [0i32; 16];

        let qa_all = &a.qs[sb * 256..(sb + 1) * 256];

        for n in 0..2 {
            let ql = &ql_all[n * 64..n * 64 + 64];
            let qh = &qh_all[n * 32..n * 32 + 32];

            // Split `l` at 16 so every scale-group index is loop-invariant.
            //
            // Written as one loop over 0..32, the accumulator index is `e / 16`
            // with `e` derived from `l`, and LLVM cannot vectorise a scatter into
            // an array at a computed index -- this kernel was the one format that
            // got *no* benefit from enabling SIMD128. Splitting the range makes
            // all four indices constant within each half, leaving four
            // independent reductions over 16 elements, which vectorises.
            for half in 0..2 {
                let g = n * 8 + half;
                let (mut a1, mut a2, mut a3, mut a4) = (0i32, 0i32, 0i32, 0i32);

                for l in half * 16..half * 16 + 16 {
                    let h = qh[l];
                    // The four elements this iteration produces are 32 apart.
                    let q1 = ((ql[l] & 0x0f) | ((h & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[l + 32] & 0x0f) | (((h >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[l] >> 4) | (((h >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((h >> 6) & 3) << 4)) as i32 - 32;

                    let e = n * 128 + l;
                    a1 += q1 * qa_all[e] as i32;
                    a2 += q2 * qa_all[e + 32] as i32;
                    a3 += q3 * qa_all[e + 64] as i32;
                    a4 += q4 * qa_all[e + 96] as i32;
                }

                group[g] += a1;
                group[g + 2] += a2;
                group[g + 4] += a3;
                group[g + 6] += a4;
            }
        }

        for (g, &acc_g) in group.iter().enumerate() {
            // Group g spans elements [16g, 16g+16), inside activation block g/2.
            let sc = sc_all[g] as i8 as f32;
            acc += d * sc * a.scales[sb * 8 + g / 2] * acc_g as f32;
        }
    }
    acc
}

// ============================================ unpack once, dot many times ==

/// One weight row unpacked into quants plus per-group scale data.
///
/// # Why this exists alongside the fused kernels
///
/// At batch size 1 the fused `row_dot_*` kernels win: they unpack a block into
/// registers, use it once, and never write it to memory. During *prefill* there
/// are many activation vectors sharing one weight row, and unpacking that row
/// once to serve all of them turns the unpack from a per-token cost into a
/// per-prompt one.
///
/// # Bit-identical, deliberately
///
/// [`row_dot_unpacked`] reproduces each fused kernel's floating-point
/// association exactly — including that the linear formats fold as
/// `(scale_w * scale_a) * dot` while Q4_K folds as
/// `scale_a * (scale_w * dot - min_w * sum)`. Float multiplication is not
/// associative, so "same maths" is not enough; getting this wrong would make a
/// batched prefill disagree with an incremental decode in the last mantissa bits
/// and quietly break the equivalence test that guards the KV cache.
#[derive(Debug, Clone)]
pub struct UnpackedRow {
    quants: Vec<i8>,
    /// Weight scale per group, with the super-block scale already folded in.
    scales: Vec<f32>,
    /// Weight offset per group. Only Q4_K is affine; the rest are all zero.
    mins: Vec<f32>,
    /// Elements per scale group: 32 for most formats, 16 for Q6_K.
    group: usize,
    /// True when `mins` carries meaning, which also implies `group == ACT_BLOCK`.
    affine: bool,
    len: usize,
}

impl UnpackedRow {
    /// Capacity for a row of `cols` elements. Reused across rows.
    pub fn new(cols: usize) -> Self {
        UnpackedRow {
            quants: vec![0; cols],
            // 16-element groups are the finest any supported format uses.
            scales: vec![0.0; cols / 16 + 1],
            mins: vec![0.0; cols / 16 + 1],
            group: ACT_BLOCK,
            affine: false,
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn byte_len(&self) -> usize {
        self.quants.len() + (self.scales.len() + self.mins.len()) * 4
    }
}

/// Unpack one weight row into `out`. `row` must be exactly the row's bytes.
pub fn unpack_row(ty: GgmlType, row: &[u8], out: &mut UnpackedRow) {
    let bb = ty.block_bytes();
    let be = ty.block_elems();
    let n_blocks = row.len() / bb;
    out.len = n_blocks * be;
    debug_assert!(out.quants.len() >= out.len);

    match ty {
        GgmlType::Q8_0 => {
            out.group = 32;
            out.affine = false;
            for (b, blk) in row.as_chunks::<34>().0.iter().enumerate() {
                out.scales[b] = read_f16_le(blk, 0);
                out.mins[b] = 0.0;
                for i in 0..32 {
                    out.quants[b * 32 + i] = blk[2 + i] as i8;
                }
            }
        }
        GgmlType::Q5_0 => {
            out.group = 32;
            out.affine = false;
            for (b, blk) in row.as_chunks::<22>().0.iter().enumerate() {
                out.scales[b] = read_f16_le(blk, 0);
                out.mins[b] = 0.0;
                let qs = &blk[6..22];
                let (h0, h1) = (&BIT_TO_HIGH[blk[2] as usize], &BIT_TO_HIGH[blk[3] as usize]);
                let (h2, h3) = (&BIT_TO_HIGH[blk[4] as usize], &BIT_TO_HIGH[blk[5] as usize]);
                let q = &mut out.quants[b * 32..b * 32 + 32];
                for j in 0..8 {
                    q[j] = (((qs[j] & 0x0f) | h0[j]) as i32 - 16) as i8;
                    q[j + 8] = (((qs[j + 8] & 0x0f) | h1[j]) as i32 - 16) as i8;
                    q[j + 16] = (((qs[j] >> 4) | h2[j]) as i32 - 16) as i8;
                    q[j + 24] = (((qs[j + 8] >> 4) | h3[j]) as i32 - 16) as i8;
                }
            }
        }
        GgmlType::Q4_0 => {
            out.group = 32;
            out.affine = false;
            for (b, blk) in row.as_chunks::<18>().0.iter().enumerate() {
                out.scales[b] = read_f16_le(blk, 0);
                out.mins[b] = 0.0;
                let qs = &blk[2..18];
                let q = &mut out.quants[b * 32..b * 32 + 32];
                for j in 0..16 {
                    q[j] = ((qs[j] & 0x0f) as i32 - 8) as i8;
                    q[j + 16] = ((qs[j] >> 4) as i32 - 8) as i8;
                }
            }
        }
        GgmlType::Q4_K => {
            out.group = 32;
            out.affine = true;
            for (sb, blk) in row.as_chunks::<144>().0.iter().enumerate() {
                let d = read_f16_le(blk, 0);
                let dmin = read_f16_le(blk, 2);
                let scales: &[u8; 12] = blk[4..16].try_into().expect("12 bytes");
                let qs = &blk[16..144];
                for s in 0..8 {
                    let (sc, m) = scale_min_k4(s, scales);
                    let g = sb * 8 + s;
                    // Fold exactly as the fused kernel does, so the products
                    // that follow round the same way.
                    out.scales[g] = d * sc as f32;
                    out.mins[g] = dmin * m as f32;
                    let base = (s / 2) * 32;
                    let q = &mut out.quants[g * 32..g * 32 + 32];
                    if s % 2 == 0 {
                        for l in 0..32 {
                            q[l] = (qs[base + l] & 0x0f) as i8;
                        }
                    } else {
                        for l in 0..32 {
                            q[l] = (qs[base + l] >> 4) as i8;
                        }
                    }
                }
            }
        }
        GgmlType::Q6_K => {
            // Sixteen-element groups: Q6_K carries a scale per 16 weights.
            out.group = 16;
            out.affine = false;
            for (sb, blk) in row.as_chunks::<210>().0.iter().enumerate() {
                let d = read_f16_le(blk, 208);
                let ql_all = &blk[0..128];
                let qh_all = &blk[128..192];
                let sc_all = &blk[192..208];
                for (g, &sc) in sc_all.iter().enumerate() {
                    out.scales[sb * 16 + g] = d * (sc as i8) as f32;
                    out.mins[sb * 16 + g] = 0.0;
                }
                let q = &mut out.quants[sb * 256..sb * 256 + 256];
                for n in 0..2 {
                    let ql = &ql_all[n * 64..n * 64 + 64];
                    let qh = &qh_all[n * 32..n * 32 + 32];
                    for l in 0..32 {
                        let h = qh[l];
                        let e = n * 128 + l;
                        q[e] = (((ql[l] & 0x0f) | ((h & 3) << 4)) as i32 - 32) as i8;
                        q[e + 32] =
                            (((ql[l + 32] & 0x0f) | (((h >> 2) & 3) << 4)) as i32 - 32) as i8;
                        q[e + 64] = (((ql[l] >> 4) | (((h >> 4) & 3) << 4)) as i32 - 32) as i8;
                        q[e + 96] = (((ql[l + 32] >> 4) | (((h >> 6) & 3) << 4)) as i32 - 32) as i8;
                    }
                }
            }
        }
        _ => {
            out.len = 0;
        }
    }
}

/// Dot an unpacked weight row with a quantised activation vector.
///
/// The two association shapes below are not interchangeable — see the note on
/// [`UnpackedRow`]. `(scale_w * scale_a) * dot` and
/// `scale_a * (scale_w * dot - ...)` differ in the last bits, and matching the
/// fused kernels exactly is what lets prefill and decode agree bit for bit.
pub fn row_dot_unpacked(row: &UnpackedRow, a: &ActivationQ8) -> f32 {
    let g = row.group;
    let n_groups = row.len / g;
    let mut acc = 0.0f32;

    // Integer accumulation, so the SIMD kernels' pairwise order gives exactly
    // what a sequential loop would. Bit-identity survives the widening.
    if row.affine {
        // Q4_K. Groups line up one-to-one with activation blocks.
        debug_assert_eq!(g, ACT_BLOCK);
        for gi in 0..n_groups {
            let dot = idot32(&row.quants[gi * g..gi * g + g], a.block(gi));
            acc += a.scales[gi] * (row.scales[gi] * dot as f32 - row.mins[gi] * a.sums[gi] as f32);
        }
    } else if g == ACT_BLOCK {
        for gi in 0..n_groups {
            let start = gi * g;
            let dot = idot32(&row.quants[start..start + g], a.block(gi));
            acc += (row.scales[gi] * a.scales[gi]) * dot as f32;
        }
    } else {
        // Q6_K: two 16-element groups per activation block.
        debug_assert_eq!(g, 16);
        for gi in 0..n_groups {
            let start = gi * g;
            let ablock = start / ACT_BLOCK;
            let dot = idot16(&row.quants[start..start + g], &a.qs[start..start + g]);
            acc += (row.scales[gi] * a.scales[ablock]) * dot as f32;
        }
    }
    acc
}
