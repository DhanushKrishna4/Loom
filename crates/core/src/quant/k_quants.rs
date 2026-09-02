//! The k-quant super-block formats: Q4_K and Q6_K.
//!
//! A k-quant super-block covers `QK_K = 256` weights and splits them into
//! sub-blocks with their own scales, so a single outlier weight only ruins the
//! precision of its own sub-block instead of the whole 256. The per-sub-block
//! scales are themselves quantised against a super-block scale, which is where
//! all the bit-packing comes from.

use super::f16::read_f16_le;

// ---------------------------------------------------------------- Q4_K ------

/// Unpack the `j`-th 6-bit scale/min pair from a Q4_K `scales[12]` array.
///
/// # The packing
///
/// A Q4_K super-block has 8 sub-blocks, each needing a 6-bit scale *and* a 6-bit
/// min. That is 8 * 6 * 2 = 96 bits = 12 bytes exactly, with no slack -- which
/// is why the layout is awkward. ggml splits the eight into a low group (0..3)
/// that gets whole bytes and a high group (4..7) that gets scavenged bits:
///
/// ```text
///   byte 0..3   bits 0-5 : scale[0..3]        bits 6-7 : scale[4..7] bits 4-5
///   byte 4..7   bits 0-5 : min[0..3]          bits 6-7 : min[4..7]   bits 4-5
///   byte 8..11  bits 0-3 : scale[4..7] bits 0-3
///               bits 4-7 : min[4..7]   bits 0-3
/// ```
///
/// So a high-group value is assembled from its low nibble in bytes 8..11 and its
/// top two bits stolen from the spare bits of bytes 0..7. This is the single
/// most likely place in the whole engine to have a silent correctness bug: get
/// it wrong and every weight is scaled by a plausible-but-wrong factor, which
/// degrades output quality without ever failing an assertion. Hence
/// [`pack_scales_min_k4`] and the isolated tests next to it.
///
/// Transcribed from `get_scale_min_k4` in ggml's `ggml-quants.c`.
#[inline]
pub fn scale_min_k4(j: usize, scales: &[u8; 12]) -> (u8, u8) {
    debug_assert!(j < 8);
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        // j is 4..=7, so j+4 indexes 8..=11 and j-4 indexes 0..=3.
        let d = (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Inverse of [`scale_min_k4`]: pack 8 six-bit scales and 8 six-bit mins into 12
/// bytes. Only used to build test blocks (we never quantise weights ourselves),
/// but having both directions makes the packing testable in isolation.
///
/// Input values above 63 are truncated to 6 bits, matching what a quantiser
/// would have to do.
pub fn pack_scales_min_k4(scales: &[u8; 8], mins: &[u8; 8]) -> [u8; 12] {
    let mut out = [0u8; 12];
    for j in 0..4 {
        out[j] = (scales[j] & 63) | (((scales[j + 4] & 63) >> 4) << 6);
        out[j + 4] = (mins[j] & 63) | (((mins[j + 4] & 63) >> 4) << 6);
        out[j + 8] = (scales[j + 4] & 0x0f) | ((mins[j + 4] & 0x0f) << 4);
    }
    out
}

/// `block_q4_K { f16 d; f16 dmin; u8 scales[12]; u8 qs[128]; }`
/// -- 256 weights in 144 bytes (4.5 bits each).
///
/// Reconstruction, per sub-block `s`:
///
/// ```text
///   value = d * scale[s] * q  -  dmin * min[s]
/// ```
///
/// Note the *asymmetric* form: unlike Q4_0 there is no fixed -8 bias. Each
/// sub-block carries its own min, so the 16 representable levels can sit
/// anywhere on the number line. Subtracting rather than adding the min matches
/// ggml, and flipping that sign is another silent-wrongness bug.
///
/// Nibble order within a sub-block pair follows the same split as Q4_0: the 32
/// bytes shared by sub-blocks `2c` and `2c+1` give their low nibbles to the
/// first and their high nibbles to the second.
#[inline]
pub fn dequant_block_q4_k(src: &[u8; 144], dst: &mut [f32; 256]) {
    let d = read_f16_le(src, 0);
    let dmin = read_f16_le(src, 2);
    let scales: &[u8; 12] = src[4..16].try_into().expect("12 bytes");
    let qs = &src[16..144];

    let mut out = 0usize; // element index
    let mut qi = 0usize; // byte index into qs
    let mut is = 0usize; // sub-block index

    // Four passes of 64 elements: each consumes 32 bytes and two sub-blocks.
    while out < 256 {
        let (sc1, m1) = scale_min_k4(is, scales);
        let (sc2, m2) = scale_min_k4(is + 1, scales);
        let d1 = d * sc1 as f32;
        let min1 = dmin * m1 as f32;
        let d2 = d * sc2 as f32;
        let min2 = dmin * m2 as f32;

        for l in 0..32 {
            let b = qs[qi + l];
            dst[out + l] = d1 * (b & 0x0f) as f32 - min1;
            dst[out + 32 + l] = d2 * (b >> 4) as f32 - min2;
        }
        out += 64;
        qi += 32;
        is += 2;
    }
}

// ---------------------------------------------------------------- Q6_K ------

/// `block_q6_K { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d; }`
/// -- 256 weights in 210 bytes (6.5625 bits each).
///
/// Note the super-block scale is at the **end** here, not the start as in Q4_K.
///
/// Each weight is 6 bits, assembled from a 4-bit low part in `ql` and a 2-bit
/// high part in `qh`, then biased by -32 to give the signed range -32..31.
/// Scales are plain signed bytes, one per 16 weights, so a super-block has 16 of
/// them and each 128-element half uses 8.
///
/// The traversal is the awkward part: one pass over `l = 0..32` emits four
/// weights that are 32 elements apart, drawing all four 2-bit high parts from
/// the *same* `qh[l]` byte at shifts 0/2/4/6. Q6_K only shows up as the type of
/// a handful of tensors in a Q4_K_M mix (typically the output/token-embedding
/// matrix), but those are exactly the tensors that decide the final logits.
#[inline]
pub fn dequant_block_q6_k(src: &[u8; 210], dst: &mut [f32; 256]) {
    let ql_all = &src[0..128];
    let qh_all = &src[128..192];
    let sc_all = &src[192..208];
    let d = read_f16_le(src, 208);

    // Two halves of 128 elements each.
    for n in 0..2 {
        let ql = &ql_all[n * 64..n * 64 + 64];
        let qh = &qh_all[n * 32..n * 32 + 32];
        let sc = &sc_all[n * 8..n * 8 + 8];
        let y = &mut dst[n * 128..n * 128 + 128];

        for l in 0..32 {
            let is = l / 16; // which of the two 16-weight scale groups
            let h = qh[l];
            let q1 = ((ql[l] & 0x0f) | ((h & 0x03) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0x0f) | (((h >> 2) & 0x03) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((h >> 4) & 0x03) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((h >> 6) & 0x03) << 4)) as i32 - 32;

            // `sc` bytes are signed; widening them as u8 would turn every
            // negative scale into a large positive one.
            y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
            y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
        }
    }
}
