//! The two "legacy" 32-element block formats: Q4_0 and Q8_0.
//!
//! These are implemented first because they are simple enough to be obviously
//! correct, which makes them the baseline that Q4_K gets checked against.

use super::f16::read_f16_le;

/// `block_q4_0 { f16 d; u8 qs[16]; }` -- 32 weights in 18 bytes (4.5 bits each).
///
/// # The nibble order is not what you would guess
///
/// "Two 4-bit values per byte" is true but says nothing about *which* two. ggml
/// does **not** store elements 0 and 1 in byte 0. It splits the block in half:
///
/// ```text
///   qs[j] low nibble  -> element j          (j = 0..15)
///   qs[j] high nibble -> element j + 16
/// ```
///
/// The reason is SIMD: one 16-byte load yields two 16-lane vectors after a mask
/// and a shift, with no interleave. Reading it as (0,1),(2,3),... instead
/// produces a permutation of the right values -- every weight present, all in
/// the wrong places -- which is exactly the kind of bug that still generates
/// fluent text.
///
/// Values are stored biased by 8: `value[i] = d * (nibble - 8)`, so the 4-bit
/// range 0..15 maps onto -8..7.
#[inline]
pub fn dequant_block_q4_0(src: &[u8; 18], dst: &mut [f32; 32]) {
    let d = read_f16_le(src, 0);
    let qs = &src[2..18];
    for j in 0..16 {
        let b = qs[j];
        dst[j] = ((b & 0x0f) as i32 - 8) as f32 * d;
        dst[j + 16] = ((b >> 4) as i32 - 8) as f32 * d;
    }
}

/// `block_q5_0 { f16 d; u8 qh[4]; u8 qs[16]; }` -- 32 weights in 22 bytes (5.5 bits each).
///
/// # Why this format matters more than the name of the file suggests
///
/// A "Q4_K_M" Qwen2.5-0.5B is 55% Q5_0 by size, not Q4_K. k-quants need the row
/// length to be a multiple of their 256-element super-block, and this model's
/// rows are 896 long (896 / 256 = 3.5). Only `ffn_down`, whose rows are 4864
/// (= 19 x 256), can hold a k-quant; every other matrix falls back to a 32-element
/// block format. So Q5_0 is not an optional extra here -- without it the model
/// does not load at all.
///
/// Q5_0 is Q4_0 plus one bit. The low four bits of each weight sit in `qs` with
/// the same low-half/high-half split as Q4_0; the fifth bit lives in `qh`, a
/// 32-bit little-endian mask where **bit `i` is the high bit of element `i`**,
/// running straight through 0..32 rather than following the nibble split. Values
/// are biased by 16, giving the range -16..15.
#[inline]
pub fn dequant_block_q5_0(src: &[u8; 22], dst: &mut [f32; 32]) {
    let d = read_f16_le(src, 0);
    let qh = u32::from_le_bytes([src[2], src[3], src[4], src[5]]);
    let qs = &src[6..22];

    for j in 0..16 {
        // Bit j is element j's fifth bit; bit j+16 is element (j+16)'s.
        let h0 = ((qh >> j) & 1) as u8;
        let h1 = ((qh >> (j + 16)) & 1) as u8;
        let b = qs[j];
        let q0 = ((b & 0x0f) | (h0 << 4)) as i32 - 16;
        let q1 = ((b >> 4) | (h1 << 4)) as i32 - 16;
        dst[j] = q0 as f32 * d;
        dst[j + 16] = q1 as f32 * d;
    }
}

/// `block_q8_0 { f16 d; i8 qs[32]; }` -- 32 weights in 34 bytes (8.5 bits each).
///
/// No bias and no packing: the only subtlety is that `qs` is *signed*, so the
/// bytes must be reinterpreted as `i8` rather than widened as `u8`.
#[inline]
pub fn dequant_block_q8_0(src: &[u8; 34], dst: &mut [f32; 32]) {
    let d = read_f16_le(src, 0);
    let qs = &src[2..34];
    for j in 0..32 {
        dst[j] = (qs[j] as i8) as f32 * d;
    }
}

/// Quantise 32 f32 values into a Q8_0 block.
///
/// This is the mirror of [`dequant_block_q8_0`] and follows ggml's
/// `quantize_row_q8_0_ref`: a single symmetric scale per block, `d = max|x|/127`,
/// with round-half-away-from-zero.
///
/// It exists now for the round-trip property test, but it is also the exact
/// routine the fused quantised matmul will need in step 6: the activation vector
/// gets quantised to Q8 once per matmul so the inner loop can be integer dot
/// products.
pub fn quantize_block_q8_0(src: &[f32; 32], dst: &mut [u8; 34]) {
    let amax = src.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    // 127, not 128: the negative end is clamped to match, keeping the scale
    // symmetric so that dequantising a zero block gives exact zeros.
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    // Round-trip the scale through f16 *before* quantising against it. The block
    // stores an f16, so quantising against the f32 scale would bake in an error
    // that dequantisation cannot undo.
    let d_stored = super::f16::f32_to_f16(d);
    let d_eff = super::f16::f16_to_f32(d_stored);
    let id = if d_eff != 0.0 { 1.0 / d_eff } else { id };

    dst[0..2].copy_from_slice(&d_stored.to_le_bytes());
    for j in 0..32 {
        let q =
            crate::math::round_half_away_from_zero(src[j] * id).clamp(-127.0, 127.0) as i32 as i8;
        dst[2 + j] = q as u8;
    }
}
