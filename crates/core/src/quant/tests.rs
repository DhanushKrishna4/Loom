//! Numeric tests for the dequantisation kernels.
//!
//! Three layers, in increasing order of how much they prove:
//!
//! 1. **Hand-computed blocks.** Byte patterns chosen so the expected f32 output
//!    can be worked out on paper, with the arithmetic written into the comment.
//!    These catch a wrong shift or a swapped nibble.
//! 2. **Independent re-implementations.** Each kernel is checked against a second
//!    version written with a different traversal. Same formula, different loop
//!    structure, so a transposition bug shows up while genuine formula changes
//!    do not produce spurious failures.
//! 3. **Real fixtures.** Blocks from an actual GGUF, dequantised by gguf-py.
//!    Empty until `tools/dump_gguf_blocks.py` is run; see `fixtures.rs`.
//!
//! Layer 1 proves the code matches the spec as I read it. Only layer 3 proves I
//! read the spec the way ggml wrote it.

// The reference implementations below index by element number on purpose:
// they mirror ggml's C, and rewriting them as iterator chains would make the
// index arithmetic -- which is the thing under test -- harder to check by eye.
#![allow(clippy::needless_range_loop)]

use super::fixtures::FIXTURES;
use super::*;
use crate::gguf::GgmlType;
use proptest::prelude::*;

// ============================================================== f16 <-> f32 ==

#[test]
fn f16_known_values() {
    // (bits, expected f32) -- the corner cases, not a sample of the middle.
    let cases: &[(u16, f32)] = &[
        (0x0000, 0.0),
        (0x8000, -0.0),
        (0x3c00, 1.0),
        (0xbc00, -1.0),
        (0x4000, 2.0),
        (0x3800, 0.5),
        (0xc500, -5.0),
        (0x3555, 0.33325195), // nearest half to 1/3
        (0x7bff, 65504.0),    // largest finite half
        (0xfbff, -65504.0),
        (0x0400, 6.1035156e-5), // smallest normal, 2^-14
        (0x03ff, 6.097555e-5),  // largest subnormal, 1023 * 2^-24
        (0x0001, 5.9604645e-8), // smallest subnormal, 2^-24
        (0x8001, -5.9604645e-8),
        (0x0200, 3.0517578e-5), // mid subnormal, 512 * 2^-24
    ];
    for &(bits, want) in cases {
        let got = f16_to_f32(bits);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "f16 {bits:#06x}: got {got:e}, want {want:e}"
        );
    }
}

#[test]
fn f16_infinities_and_nans() {
    assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
    assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
    assert!(f16_to_f32(0x7e00).is_nan()); // quiet NaN
    assert!(f16_to_f32(0x7c01).is_nan()); // signalling NaN must NOT become inf
    assert!(f16_to_f32(0xfe00).is_nan());

    assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
    assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xfc00);
    assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
}

#[test]
fn f16_round_trip_is_exact_for_all_65536_patterns() {
    // Exhaustive, because it is cheap and because a subnormal bug is invisible
    // in any sample that does not deliberately include subnormals.
    let mut subnormals = 0usize;
    for h in 0..=u16::MAX {
        let f = f16_to_f32(h);
        if f.is_nan() {
            // NaN payloads are not required to survive; only NaN-ness is.
            assert_eq!((h >> 10) & 0x1f, 0x1f);
            assert_ne!(h & 0x3ff, 0);
            continue;
        }
        if (h >> 10) & 0x1f == 0 && h & 0x3ff != 0 {
            subnormals += 1;
        }
        assert_eq!(
            f32_to_f16(f),
            h,
            "{h:#06x} -> {f:e} -> {:#06x}",
            f32_to_f16(f)
        );
    }
    // 1023 non-zero mantissas, two signs.
    assert_eq!(subnormals, 2046);
}

#[test]
fn f32_to_f16_rounds_half_to_even_and_saturates() {
    assert_eq!(f32_to_f16(70000.0), 0x7c00, "overflow becomes inf");
    // 65520 is exactly halfway between 65504 and 65536; ties-to-even picks
    // 65536, which is out of range, so IEEE gives infinity.
    assert_eq!(f32_to_f16(65520.0), 0x7c00);
    assert_eq!(f32_to_f16(65504.0), 0x7bff);
    assert_eq!(f32_to_f16(1e-10), 0x0000, "underflow becomes zero");
    assert_eq!(f32_to_f16(-1e-10), 0x8000, "...keeping the sign");
    // 2^-25 is exactly half of the smallest subnormal; ties-to-even -> zero.
    assert_eq!(f32_to_f16(2.9802322e-8), 0x0000);
    // Just above it rounds up to the smallest subnormal.
    assert_eq!(f32_to_f16(3.0e-8), 0x0001);
}

proptest! {
    /// Widening then narrowing is lossless; narrowing then widening is within
    /// half an ulp of f16, i.e. a relative error of 2^-11.
    #[test]
    fn f32_to_f16_relative_error_is_bounded(x in -60000.0f32..60000.0) {
        prop_assume!(x.abs() > 1e-4); // stay out of the subnormal range
        let back = f16_to_f32(f32_to_f16(x));
        let rel = ((back - x) / x).abs();
        prop_assert!(rel <= 2f32.powi(-11), "x={x} back={back} rel={rel}");
    }
}

// ==================================================================== Q4_0 ==

#[test]
fn q4_0_hand_computed_block() {
    // d = 2.0. qs[j] packs element j in the low nibble and element j+16 in the
    // high nibble, so with low = j and high = 15-j:
    //   dst[j]      = 2 * (j - 8)          -> -16, -14, ... 14
    //   dst[j + 16] = 2 * ((15 - j) - 8)   ->  14,  12, ... -16
    let mut src = [0u8; 18];
    src[0..2].copy_from_slice(&f32_to_f16(2.0).to_le_bytes());
    for j in 0..16u8 {
        src[2 + j as usize] = j | ((15 - j) << 4);
    }

    let mut dst = [0f32; 32];
    dequant_block_q4_0(&src, &mut dst);

    for j in 0..16 {
        assert_eq!(dst[j], 2.0 * (j as f32 - 8.0), "low nibble, element {j}");
        assert_eq!(
            dst[j + 16],
            2.0 * (7.0 - j as f32),
            "high nibble, element {}",
            j + 16
        );
    }
    assert_eq!(dst[0], -16.0);
    assert_eq!(dst[15], 14.0);
    assert_eq!(dst[16], 14.0);
    assert_eq!(dst[31], -16.0);
}

#[test]
fn q4_0_nibble_halves_are_not_interleaved() {
    // A block where only element 1 is non-minimal. If the layout were
    // (byte0 -> elements 0,1) instead of (low half, high half), this value would
    // land at index 16 rather than index 1.
    let mut src = [0u8; 18];
    src[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
    src[2..18].fill(0x00); // every nibble 0 -> value -8
    src[3] = 0x0f; // qs[1] low nibble = 15 -> element 1

    let mut dst = [0f32; 32];
    dequant_block_q4_0(&src, &mut dst);
    assert_eq!(dst[1], 7.0);
    assert_eq!(dst[17], -8.0);
    assert_eq!(dst.iter().filter(|v| **v != -8.0).count(), 1);
}

// ==================================================================== Q8_0 ==

#[test]
fn q8_0_hand_computed_block() {
    // d = 0.5, qs[i] = i - 64 as a signed byte, so dst[i] = 0.5 * (i - 64):
    // -32.0, -31.5, ... -16.5. Exercises the signed reinterpretation: read as
    // unsigned, dst[0] would be 96.0 instead of -32.0.
    let mut src = [0u8; 34];
    src[0..2].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
    for i in 0..32 {
        src[2 + i] = (i as i32 - 64) as i8 as u8;
    }

    let mut dst = [0f32; 32];
    dequant_block_q8_0(&src, &mut dst);

    assert_eq!(dst[0], -32.0);
    assert_eq!(dst[31], -16.5);
    for i in 0..32 {
        assert_eq!(dst[i], 0.5 * (i as f32 - 64.0), "element {i}");
    }
}

#[test]
fn q8_0_quantise_dequantise_is_exact_for_representable_values() {
    // Values that are exact multiples of the block scale must survive intact.
    let mut src = [0f32; 32];
    for (i, v) in src.iter_mut().enumerate() {
        *v = (i as f32 - 16.0) * 0.25; // amax = 4.0, d = 4/127
    }
    let mut blk = [0u8; 34];
    quantize_block_q8_0(&src, &mut blk);
    let mut back = [0f32; 32];
    dequant_block_q8_0(&blk, &mut back);

    let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
    for i in 0..32 {
        assert!(
            (back[i] - src[i]).abs() <= 0.5 * d,
            "element {i}: {} vs {} (d = {d})",
            back[i],
            src[i]
        );
    }
}

#[test]
fn q8_0_all_zeros_stays_all_zeros() {
    // The degenerate block: a division by a zero scale must not produce NaN.
    let src = [0f32; 32];
    let mut blk = [0u8; 34];
    quantize_block_q8_0(&src, &mut blk);
    let mut back = [0f32; 32];
    dequant_block_q8_0(&blk, &mut back);
    assert!(back.iter().all(|v| *v == 0.0), "{back:?}");
}

#[test]
fn q8_0_rounds_small_values_to_zero() {
    // Regression: an "add 0.5 and truncate" rounding helper returns 1 instead of
    // 0 for inputs just below a .5 boundary, because the addition crosses a
    // binade and loses the deciding bit. That would give every near-zero weight
    // in a block a spurious +/-1 quant -- invisible in aggregate, permanent in
    // effect. See math::round_half_away_from_zero.
    let mut src = [0f32; 32];
    src[0] = 100.0; // sets the block scale: d = 100/127
    let d = 100.0 / 127.0;
    // Just under half a quantisation step: must round to 0, not to +/-1.
    src[1] = d * 0.49999997;
    src[2] = -d * 0.49999997;
    // Just over: must round away from zero.
    src[3] = d * 0.51;
    src[4] = -d * 0.51;

    let mut blk = [0u8; 34];
    quantize_block_q8_0(&src, &mut blk);
    let q = |i: usize| blk[2 + i] as i8;

    assert_eq!(q(1), 0, "just under half a step must quantise to zero");
    assert_eq!(q(2), 0);
    assert_eq!(q(3), 1, "just over half a step must round away from zero");
    assert_eq!(q(4), -1);
    // And the value that set the scale must come back at full magnitude.
    assert_eq!(q(0), 127);
}

// ==================================================================== Q5_0 ==

#[test]
fn q5_0_hand_computed_block() {
    // d = 1.0, qh = 0 (no fifth bits), qs[j] = j | ((15-j) << 4).
    //   dst[j]      = (j - 16)
    //   dst[j + 16] = ((15 - j) - 16) = -1 - j
    let mut src = [0u8; 22];
    src[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
    for j in 0..16u8 {
        src[6 + j as usize] = j | ((15 - j) << 4);
    }
    let mut dst = [0f32; 32];
    dequant_block_q5_0(&src, &mut dst);
    for j in 0..16 {
        assert_eq!(dst[j], j as f32 - 16.0, "low nibble, element {j}");
        assert_eq!(
            dst[j + 16],
            -1.0 - j as f32,
            "high nibble, element {}",
            j + 16
        );
    }
}

#[test]
fn q5_0_fifth_bit_comes_from_qh_bit_i() {
    // qh bit i is element i's fifth bit, running 0..32 straight through -- it does
    // NOT follow the low-half/high-half split that qs uses. Setting bits 0 and 16
    // must lift exactly elements 0 and 16, by exactly 16 each.
    let mut base = [0u8; 22];
    base[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
    base[6..22].fill(0x00); // every nibble 0 -> every value -16

    let mut plain = [0f32; 32];
    dequant_block_q5_0(&base, &mut plain);
    assert!(plain.iter().all(|v| *v == -16.0));

    let mut src = base;
    src[2..6].copy_from_slice(&0x0001_0001u32.to_le_bytes()); // bits 0 and 16
    let mut dst = [0f32; 32];
    dequant_block_q5_0(&src, &mut dst);
    for i in 0..32 {
        let want = if i == 0 || i == 16 { 0.0 } else { -16.0 };
        assert_eq!(dst[i], want, "element {i}");
    }

    // All bits set: every element gains exactly 16.
    let mut src = base;
    src[2..6].copy_from_slice(&u32::MAX.to_le_bytes());
    let mut dst = [0f32; 32];
    dequant_block_q5_0(&src, &mut dst);
    assert!(dst.iter().all(|v| *v == 0.0), "first few: {:?}", &dst[..4]);
}

#[test]
fn q5_0_covers_the_full_five_bit_range() {
    // Values must span -16..15, not Q4_0's -8..7.
    let mut src = [0u8; 22];
    src[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
    src[6] = 0x00; // element 0 low nibble 0
    src[7] = 0x0f; // element 1 low nibble 15
    src[2..6].copy_from_slice(&0x0000_0002u32.to_le_bytes()); // fifth bit on element 1
    let mut dst = [0f32; 32];
    dequant_block_q5_0(&src, &mut dst);
    assert_eq!(dst[0], -16.0, "minimum representable");
    assert_eq!(dst[1], 15.0, "maximum representable");
}

// ==================================================================== Q4_K ==

/// The 12 scale bytes used by the hand-computed tests below, chosen so that
/// every high-group value needs bits from two different bytes.
const Q4K_SCALE_BYTES: [u8; 12] = [
    0xc1, 0x42, 0x83, 0x04, // scales[0..3] in bits 0-5, scales[4..7] top bits in 6-7
    0xe0, 0x50, 0x88, 0x06, // mins[0..3]   in bits 0-5, mins[4..7]   top bits in 6-7
    0x12, 0x34, 0x56, 0x78, // low nibbles of scales[4..7] / mins[4..7]
];

#[test]
fn q4_k_six_bit_scales_unpack_correctly() {
    // Worked by hand from the packing rule. Low group (j < 4) is a plain mask:
    //   scale[0] = 0xc1 & 63 = 1     min[0] = 0xe0 & 63 = 32
    //   scale[1] = 0x42 & 63 = 2     min[1] = 0x50 & 63 = 16
    //   scale[2] = 0x83 & 63 = 3     min[2] = 0x88 & 63 = 8
    //   scale[3] = 0x04 & 63 = 4     min[3] = 0x06 & 63 = 6
    // High group (j >= 4) glues a low nibble from bytes 8..11 to two bits
    // scavenged from bytes 0..7:
    //   scale[4] = (0x12 & 0xf) | ((0xc1 >> 6) << 4) = 2  | 48 = 50
    //   min[4]   = (0x12 >> 4)  | ((0xe0 >> 6) << 4) = 1  | 48 = 49
    //   scale[5] = (0x34 & 0xf) | ((0x42 >> 6) << 4) = 4  | 16 = 20
    //   min[5]   = (0x34 >> 4)  | ((0x50 >> 6) << 4) = 3  | 16 = 19
    //   scale[6] = (0x56 & 0xf) | ((0x83 >> 6) << 4) = 6  | 32 = 38
    //   min[6]   = (0x56 >> 4)  | ((0x88 >> 6) << 4) = 5  | 32 = 37
    //   scale[7] = (0x78 & 0xf) | ((0x04 >> 6) << 4) = 8  |  0 = 8
    //   min[7]   = (0x78 >> 4)  | ((0x06 >> 6) << 4) = 7  |  0 = 7
    let want_scales = [1u8, 2, 3, 4, 50, 20, 38, 8];
    let want_mins = [32u8, 16, 8, 6, 49, 19, 37, 7];

    for j in 0..8 {
        let (sc, m) = scale_min_k4(j, &Q4K_SCALE_BYTES);
        assert_eq!(sc, want_scales[j], "scale[{j}]");
        assert_eq!(m, want_mins[j], "min[{j}]");
        assert!(sc < 64 && m < 64, "6-bit values must not exceed 63");
    }
}

#[test]
fn q4_k_scale_packing_is_a_bijection() {
    // 8 scales + 8 mins at 6 bits each is 96 bits, and the container is 96 bits,
    // so pack and unpack must be exact inverses in both directions -- there is
    // nowhere for a lost bit to hide.
    let want_scales = [1u8, 2, 3, 4, 50, 20, 38, 8];
    let want_mins = [32u8, 16, 8, 6, 49, 19, 37, 7];
    assert_eq!(
        pack_scales_min_k4(&want_scales, &want_mins),
        Q4K_SCALE_BYTES
    );

    // Unpack -> repack over the whole byte space (sampled deterministically).
    let mut state = 0x2545_f491_4f6c_dd1du64;
    for _ in 0..5000 {
        let mut bytes = [0u8; 12];
        for b in bytes.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *b = state as u8;
        }
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for j in 0..8 {
            let (s, m) = scale_min_k4(j, &bytes);
            scales[j] = s;
            mins[j] = m;
        }
        assert_eq!(
            pack_scales_min_k4(&scales, &mins),
            bytes,
            "from {bytes:02x?}"
        );
    }
}

/// Build the hand-computed Q4_K block: d = 1.0, dmin = 0.5, the scale bytes
/// above, and qs[i] = i.
fn q4k_hand_block() -> [u8; 144] {
    let mut src = [0u8; 144];
    src[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
    src[2..4].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
    src[4..16].copy_from_slice(&Q4K_SCALE_BYTES);
    for i in 0..128 {
        src[16 + i] = i as u8;
    }
    src
}

#[test]
fn q4_k_hand_computed_block() {
    let mut dst = [0f32; 256];
    dequant_block_q4_k(&q4k_hand_block(), &mut dst);

    // value = d * scale[s] * q - dmin * min[s], with d = 1, dmin = 0.5.
    // Sub-block s covers elements s*32..s*32+32. The 32 bytes qs[c*32..] are
    // shared: sub-block 2c takes their low nibbles, 2c+1 their high nibbles.
    //
    //  idx   s  scale min  qs byte      nibble   value
    //    0   0    1    32  qs[0]=0x00   lo  0    1*1*0  - 0.5*32 = -16.0
    //    1   0    1    32  qs[1]=0x01   lo  1    1*1*1  - 16     = -15.0
    //   31   0    1    32  qs[31]=0x1f  lo 15    1*15   - 16     =  -1.0
    //   32   1    2    16  qs[0]=0x00   hi  0    2*0    - 8      =  -8.0
    //   63   1    2    16  qs[31]=0x1f  hi  1    2*1    - 8      =  -6.0
    //   64   2    3     8  qs[32]=0x20  lo  0    3*0    - 4      =  -4.0
    //   65   2    3     8  qs[33]=0x21  lo  1    3*1    - 4      =  -1.0
    //  128   4   50    49  qs[64]=0x40  lo  0    50*0   - 24.5   = -24.5
    //  129   4   50    49  qs[65]=0x41  lo  1    50*1   - 24.5   =  25.5
    //  160   5   20    19  qs[64]=0x40  hi  4    20*4   - 9.5    =  70.5
    //  192   6   38    37  qs[96]=0x60  lo  0    38*0   - 18.5   = -18.5
    //  224   7    8     7  qs[96]=0x60  hi  6    8*6    - 3.5    =  44.5
    //  255   7    8     7  qs[127]=0x7f hi  7    8*7    - 3.5    =  52.5
    let expect: &[(usize, f32)] = &[
        (0, -16.0),
        (1, -15.0),
        (31, -1.0),
        (32, -8.0),
        (63, -6.0),
        (64, -4.0),
        (65, -1.0),
        (128, -24.5),
        (129, 25.5),
        (160, 70.5),
        (192, -18.5),
        (224, 44.5),
        (255, 52.5),
    ];
    for &(i, want) in expect {
        assert_eq!(dst[i], want, "element {i}");
    }
}

/// Independent re-implementation of Q4_K: walks the 8 sub-blocks directly
/// instead of the paired 64-element passes the real kernel uses. Same formula,
/// different traversal, so it catches an index or nibble-half mix-up.
fn q4k_reference(src: &[u8; 144]) -> [f32; 256] {
    let d = f16_to_f32(u16::from_le_bytes([src[0], src[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([src[2], src[3]]));
    let scales: [u8; 12] = src[4..16].try_into().unwrap();
    let qs = &src[16..144];

    let mut out = [0f32; 256];
    for s in 0..8 {
        let (sc, m) = scale_min_k4(s, &scales);
        let ds = d * sc as f32;
        let ms = dmin * m as f32;
        for e in 0..32 {
            let byte = qs[(s / 2) * 32 + e];
            let q = if s % 2 == 0 { byte & 0x0f } else { byte >> 4 };
            out[s * 32 + e] = ds * q as f32 - ms;
        }
    }
    out
}

#[test]
fn q4_k_matches_the_independent_traversal() {
    let mut dst = [0f32; 256];
    dequant_block_q4_k(&q4k_hand_block(), &mut dst);
    assert_eq!(dst, q4k_reference(&q4k_hand_block()));
}

#[test]
fn q4_k_min_is_subtracted_not_added() {
    // All quants zero, so every value is purely -dmin * min[s]. If the sign were
    // flipped these would all come out positive.
    let mut src = [0u8; 144];
    src[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
    src[2..4].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
    src[4..16].copy_from_slice(&pack_scales_min_k4(&[7; 8], &[1, 2, 3, 4, 5, 6, 7, 8]));

    let mut dst = [0f32; 256];
    dequant_block_q4_k(&src, &mut dst);
    for s in 0..8 {
        assert_eq!(dst[s * 32], -((s + 1) as f32), "sub-block {s}");
    }
}

// ==================================================================== Q6_K ==

/// d = 1.0, ql[i] = i, qh[i] = i, scales[i] = i - 8.
fn q6k_hand_block() -> [u8; 210] {
    let mut src = [0u8; 210];
    for i in 0..128 {
        src[i] = i as u8;
    }
    for i in 0..64 {
        src[128 + i] = i as u8;
    }
    for i in 0..16 {
        src[192 + i] = (i as i32 - 8) as i8 as u8;
    }
    src[208..210].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
    src
}

#[test]
fn q6_k_hand_computed_block() {
    let mut dst = [0f32; 256];
    dequant_block_q6_k(&q6k_hand_block(), &mut dst);

    // Each pass over l = 0..32 emits four weights 32 apart, taking their 2-bit
    // high parts from the same qh[l] byte at shifts 0/2/4/6.
    //
    //  idx   half l  is  ql            qh          q            scale  value
    //    0     0   0  0  ql[0]=0x00    qh[0]=0     0|0   -32=-32  s[0]=-8   256
    //   32     0   0  0  ql[32]=0x20   qh[0]>>2=0  0|0   -32=-32  s[2]=-6   192
    //   64     0   0  0  ql[0]>>4=0    qh[0]>>4=0  0|0   -32=-32  s[4]=-4   128
    //   96     0   0  0  ql[32]>>4=2   qh[0]>>6=0  2|0   -32=-30  s[6]=-2    60
    //   17     0  17  1  ql[17]=0x11   qh[17]&3=1  1|16  -32=-15  s[1]=-7   105
    //   49     0  17  1  ql[49]=0x31   qh[17]>>2=0 1|0   -32=-31  s[3]=-5   155
    //   81     0  17  1  ql[17]>>4=1   qh[17]>>4=1 1|16  -32=-15  s[5]=-3    45
    //  113     0  17  1  ql[49]>>4=3   qh[17]>>6=0 3|0   -32=-29  s[7]=-1    29
    //  128     1   0  0  ql[64]=0x40   qh[32]&3=0  0|0   -32=-32  s[8]= 0     0
    //  159     1  31  1  ql[95]=0x5f   qh[63]&3=3  15|48 -32= 31  s[9]= 1    31
    //  160     1   0  0  ql[96]=0x60   qh[32]>>2=0 0|0   -32=-32  s[10]=2   -64
    //  192     1   0  0  ql[64]>>4=4   qh[32]>>4=2 4|32  -32=  4  s[12]=4    16
    //  224     1   0  0  ql[96]>>4=6   qh[32]>>6=0 6|0   -32=-26  s[14]=6  -156
    //  255     1  31  1  ql[127]>>4=7  qh[63]>>6=0 7|0   -32=-25  s[15]=7  -175
    let expect: &[(usize, f32)] = &[
        (0, 256.0),
        (32, 192.0),
        (64, 128.0),
        (96, 60.0),
        (17, 105.0),
        (49, 155.0),
        (81, 45.0),
        (113, 29.0),
        (128, 0.0),
        (159, 31.0),
        (160, -64.0),
        (192, 16.0),
        (224, -156.0),
        (255, -175.0),
    ];
    for &(i, want) in expect {
        assert_eq!(dst[i], want, "element {i}");
    }
}

#[test]
fn q6_k_scales_are_signed() {
    // Every quant is 0 (i.e. -32 after the bias) and every scale is -1, so a
    // correct implementation gives +32 everywhere. Reading the scale byte as
    // unsigned would give 255 * -32 = -8160.
    let mut src = [0u8; 210];
    for i in 0..16 {
        src[192 + i] = (-1i8) as u8;
    }
    src[208..210].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());

    let mut dst = [0f32; 256];
    dequant_block_q6_k(&src, &mut dst);
    assert!(dst.iter().all(|v| *v == 32.0), "first few: {:?}", &dst[..8]);
}

/// Independent re-implementation of Q6_K, indexing every element directly from
/// its element number rather than walking the four-at-a-time pattern.
fn q6k_reference(src: &[u8; 210]) -> [f32; 256] {
    let d = f16_to_f32(u16::from_le_bytes([src[208], src[209]]));
    let mut out = [0f32; 256];
    for i in 0..256 {
        let half = i / 128; // which 128-element half
        let j = i % 128; // position within the half
        let group = j / 32; // 0..3: which of the four interleaved streams
        let l = j % 32;

        let ql_base = half * 64;
        let ql_byte = src[ql_base + (group % 2) * 32 + l];
        let low = if group < 2 {
            ql_byte & 0x0f
        } else {
            ql_byte >> 4
        };
        let qh_byte = src[128 + half * 32 + l];
        let high = (qh_byte >> (2 * group)) & 0x03;
        let q = (low | (high << 4)) as i32 - 32;

        let sc = src[192 + half * 8 + group * 2 + l / 16] as i8;
        out[i] = d * sc as f32 * q as f32;
    }
    out
}

#[test]
fn q6_k_matches_the_independent_traversal() {
    let mut dst = [0f32; 256];
    dequant_block_q6_k(&q6k_hand_block(), &mut dst);
    assert_eq!(dst, q6k_reference(&q6k_hand_block()));
}

// ================================================================ row API ==

#[test]
fn dequantize_row_handles_plain_float_types() {
    let vals = [1.0f32, -2.5, 0.0, 1e10];
    let mut src = Vec::new();
    for v in vals {
        src.extend_from_slice(&v.to_le_bytes());
    }
    let mut dst = [0f32; 4];
    dequantize_row(GgmlType::F32, &src, &mut dst).unwrap();
    assert_eq!(dst, vals);

    let mut src = Vec::new();
    for v in [1.0f32, -2.5, 0.0, 0.5] {
        src.extend_from_slice(&f32_to_f16(v).to_le_bytes());
    }
    let mut dst = [0f32; 4];
    dequantize_row(GgmlType::F16, &src, &mut dst).unwrap();
    assert_eq!(dst, [1.0, -2.5, 0.0, 0.5]);
}

#[test]
fn dequantize_row_equals_per_block_dequant() {
    // Three Q4_K super-blocks back to back: the row loop must not lose its place
    // between blocks.
    let one = q4k_hand_block();
    let mut src = Vec::new();
    for k in 0..3u8 {
        let mut b = one;
        b[16] = b[16].wrapping_add(k); // make the blocks distinguishable
        src.extend_from_slice(&b);
    }

    let mut row = vec![0f32; 768];
    dequantize_row(GgmlType::Q4_K, &src, &mut row).unwrap();

    for k in 0..3usize {
        let mut b = one;
        b[16] = b[16].wrapping_add(k as u8);
        let mut block = [0f32; 256];
        dequant_block_q4_k(&b, &mut block);
        assert_eq!(&row[k * 256..(k + 1) * 256], &block[..], "block {k}");
    }
}

#[test]
fn dequantize_row_rejects_mismatched_lengths() {
    let src = [0u8; 144];
    let mut dst = [0f32; 256];

    assert!(dequantize_row(GgmlType::Q4_K, &src, &mut dst).is_ok());
    assert_eq!(
        dequantize_row(GgmlType::Q4_K, &src[..143], &mut dst),
        Err(QuantError::BadSourceLength {
            ty: GgmlType::Q4_K,
            got: 143,
            want: 144
        })
    );
    let mut short = [0f32; 255];
    assert_eq!(
        dequantize_row(GgmlType::Q4_K, &src, &mut short),
        Err(QuantError::NotBlockAligned {
            ty: GgmlType::Q4_K,
            len: 255,
            block: 256
        })
    );
}

#[test]
fn unsupported_types_fail_loudly() {
    // We can still compute their sizes (so offsets stay right) but must never
    // pretend to decode them.
    for ty in [
        GgmlType::Q5_K,
        GgmlType::Q2_K,
        GgmlType::Q3_K,
        GgmlType::IQ4_XS,
        GgmlType::BF16,
    ] {
        assert!(!is_supported(ty), "{ty}");
        let src = vec![0u8; ty.block_bytes()];
        let mut dst = vec![0f32; ty.block_elems()];
        assert_eq!(
            dequantize_row(ty, &src, &mut dst),
            Err(QuantError::UnsupportedType(ty))
        );
    }
    for ty in [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::Q4_0,
        GgmlType::Q5_0,
        GgmlType::Q8_0,
        GgmlType::Q4_K,
        GgmlType::Q6_K,
    ] {
        assert!(is_supported(ty), "{ty}");
    }
}

// ================================================================ property ==

fn bytes(n: usize) -> impl Strategy<Value = Vec<u8>> {
    proptest::collection::vec(any::<u8>(), n)
}

proptest! {
    /// Q8_0 is a symmetric per-block scale, so the reconstruction error is
    /// bounded by half a quantisation step -- no matter what the inputs are.
    #[test]
    fn q8_0_round_trip_error_is_bounded(xs in proptest::collection::vec(-1000.0f32..1000.0, 32)) {
        let src: [f32; 32] = xs.try_into().unwrap();
        let mut blk = [0u8; 34];
        quantize_block_q8_0(&src, &mut blk);
        let mut back = [0f32; 32];
        dequant_block_q8_0(&blk, &mut back);

        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let amax = src.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        for i in 0..32 {
            let err = (back[i] - src[i]).abs();
            // Half a step, plus slack for the block scale itself being an f16.
            prop_assert!(
                err <= 0.5 * d + 1e-6 * amax,
                "element {i}: |{} - {}| = {err} > {}", back[i], src[i], 0.5 * d
            );
        }
    }

    /// The real kernel and the independently-written traversal must agree
    /// exactly (not approximately) on arbitrary block contents.
    #[test]
    fn q4_k_agrees_with_reference_on_random_blocks(
        d in -8.0f32..8.0,
        dmin in -8.0f32..8.0,
        scales in bytes(12),
        qs in bytes(128),
    ) {
        let mut src = [0u8; 144];
        src[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        src[2..4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());
        src[4..16].copy_from_slice(&scales);
        src[16..144].copy_from_slice(&qs);

        let mut got = [0f32; 256];
        dequant_block_q4_k(&src, &mut got);
        prop_assert_eq!(got, q4k_reference(&src));
        prop_assert!(got.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn q6_k_agrees_with_reference_on_random_blocks(
        d in -8.0f32..8.0,
        body in bytes(208),
    ) {
        let mut src = [0u8; 210];
        src[0..208].copy_from_slice(&body);
        src[208..210].copy_from_slice(&f32_to_f16(d).to_le_bytes());

        let mut got = [0f32; 256];
        dequant_block_q6_k(&src, &mut got);
        prop_assert_eq!(got, q6k_reference(&src));
        prop_assert!(got.iter().all(|v| v.is_finite()));
    }

    /// Every Q4_K output must be exactly one of the 16 levels its sub-block can
    /// represent. Catches a stray offset that would land values between levels.
    #[test]
    fn q4_k_values_land_on_representable_levels(scales in bytes(12), qs in bytes(128)) {
        let mut src = [0u8; 144];
        src[0..2].copy_from_slice(&f32_to_f16(0.25).to_le_bytes());
        src[2..4].copy_from_slice(&f32_to_f16(0.125).to_le_bytes());
        src[4..16].copy_from_slice(&scales);
        src[16..144].copy_from_slice(&qs);
        let sc12: [u8; 12] = scales.try_into().unwrap();

        let mut got = [0f32; 256];
        dequant_block_q4_k(&src, &mut got);

        for s in 0..8 {
            let (sc, m) = scale_min_k4(s, &sc12);
            let levels: Vec<f32> = (0..16).map(|q| 0.25 * sc as f32 * q as f32 - 0.125 * m as f32).collect();
            for e in 0..32 {
                let v = got[s * 32 + e];
                prop_assert!(levels.contains(&v), "sub-block {s} element {e}: {v} not a level");
            }
        }
    }

    /// Q4_0's dequantised values are always an exact multiple of the block scale
    /// in the range [-8d, 7d].
    #[test]
    fn q4_0_values_are_in_range(d in -4.0f32..4.0, qs in bytes(16)) {
        let mut src = [0u8; 18];
        src[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        src[2..18].copy_from_slice(&qs);
        let de = f16_to_f32(u16::from_le_bytes([src[0], src[1]]));

        let mut got = [0f32; 32];
        dequant_block_q4_0(&src, &mut got);
        for (i, v) in got.iter().enumerate() {
            let levels: Vec<f32> = (0..16).map(|q| (q as f32 - 8.0) * de).collect();
            prop_assert!(levels.contains(v), "element {i}: {v}");
        }
    }
}

// ================================================================ fixtures ==

#[test]
fn matches_reference_implementation_on_real_blocks() {
    if FIXTURES.is_empty() {
        // Nothing to check yet. This is the honest state of the test rather than
        // a silent pass: hand-built blocks prove we match the spec as written,
        // and only these fixtures prove we match what llama.cpp actually emits.
        eprintln!(
            "note: no real-model fixtures compiled in; run\n  \
             python3 tools/dump_gguf_blocks.py <model.gguf>\n  \
             to generate them from a real GGUF"
        );
        return;
    }

    let mut worst = 0.0f32;
    let mut per_type: Vec<(GgmlType, usize)> = Vec::new();

    for fx in FIXTURES {
        let ty = GgmlType::from_u32(fx.ggml_type).expect("fixture has a known type");
        match per_type.iter_mut().find(|(t, _)| *t == ty) {
            Some(e) => e.1 += 1,
            None => per_type.push((ty, 1)),
        }
        assert_eq!(
            fx.raw.len(),
            ty.block_bytes(),
            "{} fixture raw length",
            fx.tensor
        );
        assert_eq!(
            fx.expected.len(),
            ty.block_elems(),
            "{} fixture expected length",
            fx.tensor
        );

        let mut got = vec![0f32; ty.block_elems()];
        dequantize_row(ty, fx.raw, &mut got).unwrap();

        for (i, (&g, &w)) in got.iter().zip(fx.expected).enumerate() {
            // gguf-py dequantises in f32 with the same operation order, so this
            // should be exact; the epsilon only absorbs a different fused-multiply
            // decision in numpy.
            let tol = 1e-6 * w.abs().max(1.0);
            worst = worst.max((g - w).abs());
            assert!(
                (g - w).abs() <= tol,
                "{} block {} element {}: got {g}, reference {w} (from {})",
                fx.tensor,
                fx.block_index,
                i,
                super::fixtures::SOURCE_MODEL
            );
        }
    }

    per_type.sort();
    std::eprintln!(
        "  checked {} real blocks from {} ({}), worst abs error {worst:e}",
        FIXTURES.len(),
        super::fixtures::SOURCE_MODEL,
        per_type
            .iter()
            .map(|(t, n)| alloc::format!("{n}x {t}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    assert!(
        worst == 0.0,
        "expected bit-exact agreement with gguf-py, got {worst:e}"
    );
}

// ================================================== fused quantised dots ==

mod fused {
    use super::super::*;
    use crate::ops::dot;

    /// Deterministic pseudo-random bytes.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed | 1)
        }
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn bytes(&mut self, n: usize) -> Vec<u8> {
            (0..n).map(|_| self.next() as u8).collect()
        }
        fn floats(&mut self, n: usize) -> Vec<f32> {
            (0..n)
                .map(|_| ((self.next() >> 40) as f32 / 4_194_304.0) - 2.0)
                .collect()
        }
    }

    /// Build a weight row of `n_blocks` blocks with plausible scales.
    fn weight_row(ty: GgmlType, n_blocks: usize, rng: &mut Rng) -> Vec<u8> {
        let mut row = rng.bytes(n_blocks * ty.block_bytes());
        // Random bytes make wild f16 scales (inf, NaN). Overwrite the scale
        // fields with small finite values so the comparison is meaningful.
        for b in 0..n_blocks {
            let o = b * ty.block_bytes();
            let d = f32_to_f16(0.02).to_le_bytes();
            match ty {
                GgmlType::Q8_0 | GgmlType::Q5_0 | GgmlType::Q4_0 => {
                    row[o..o + 2].copy_from_slice(&d);
                }
                GgmlType::Q4_K => {
                    row[o..o + 2].copy_from_slice(&d);
                    row[o + 2..o + 4].copy_from_slice(&f32_to_f16(0.01).to_le_bytes());
                }
                GgmlType::Q6_K => {
                    row[o + 208..o + 210].copy_from_slice(&d);
                    // Scales are i8; keep them in a sane range.
                    for i in 0..16 {
                        row[o + 192 + i] = ((row[o + 192 + i] as i8) / 4) as u8;
                    }
                }
                _ => unreachable!(),
            }
        }
        row
    }

    fn kernel(ty: GgmlType) -> fn(&[u8], &ActivationQ8) -> f32 {
        match ty {
            GgmlType::Q8_0 => row_dot_q8_0,
            GgmlType::Q5_0 => row_dot_q5_0,
            GgmlType::Q4_0 => row_dot_q4_0,
            GgmlType::Q4_K => row_dot_q4_k,
            GgmlType::Q6_K => row_dot_q6_k,
            _ => unreachable!(),
        }
    }

    /// The kernel must compute exactly what "dequantise the weights, then dot
    /// against the same quantised activation" computes.
    ///
    /// This isolates the fused arithmetic. Comparing against the *unquantised*
    /// activation instead would fold in the accuracy cost of Q8 activations and
    /// hide a real kernel bug inside it -- that cost is measured separately
    /// below.
    #[test]
    fn kernels_match_dequantise_then_dot() {
        let mut rng = Rng::new(0xFEED);
        for ty in [
            GgmlType::Q8_0,
            GgmlType::Q5_0,
            GgmlType::Q4_0,
            GgmlType::Q4_K,
            GgmlType::Q6_K,
        ] {
            let n_blocks = 4;
            let n = n_blocks * ty.block_elems();
            let row = weight_row(ty, n_blocks, &mut rng);

            let mut w = vec![0.0f32; n];
            dequantize_row(ty, &row, &mut w).unwrap();

            let x = rng.floats(n);
            let mut act = ActivationQ8::new(n);
            act.quantize(&x);
            let mut xq = vec![0.0f32; n];
            act.dequantize(&mut xq);

            let got = kernel(ty)(&row, &act);
            let want = dot(&w, &xq);

            // Both compute the same value; they differ only in accumulation
            // order (exact i32 against sequential f32), so the tolerance is
            // rounding, not approximation.
            let tol = 1e-4 * want.abs().max(1e-2);
            assert!(
                (got - want).abs() <= tol,
                "{ty}: fused {got} vs dequantised {want} (diff {:e})",
                (got - want).abs()
            );
        }
    }

    /// The unpack-once path must agree with the fused kernels **bit for bit**.
    ///
    /// Not "within tolerance". Prefill uses the unpacked path and decode uses
    /// the fused one, and the KV cache equivalence test asserts that a batched
    /// prefill and an incremental decode produce identical logits. Float
    /// multiplication is not associative, so the two kernels have to fold their
    /// scales in exactly the same order — `(scale_w * scale_a) * dot` for the
    /// linear formats, `scale_a * (scale_w * dot - min_w * sum)` for Q4_K. This
    /// test is what stops that from drifting.
    #[test]
    fn unpacked_path_is_bit_identical_to_the_fused_kernels() {
        let mut rng = Rng::new(0xBEEF);
        for ty in [
            GgmlType::Q8_0,
            GgmlType::Q5_0,
            GgmlType::Q4_0,
            GgmlType::Q4_K,
            GgmlType::Q6_K,
        ] {
            // Several blocks, so cross-block accumulation order is covered too.
            let n_blocks = 7;
            let n = n_blocks * ty.block_elems();
            let row = weight_row(ty, n_blocks, &mut rng);

            let mut act = ActivationQ8::new(n);
            act.quantize(&rng.floats(n));

            let fused = kernel(ty)(&row, &act);

            let mut unpacked = UnpackedRow::new(n);
            unpack_row(ty, &row, &mut unpacked);
            assert_eq!(unpacked.len(), n, "{ty}: unpacked the wrong length");
            let batched = row_dot_unpacked(&unpacked, &act);

            assert_eq!(
                fused.to_bits(),
                batched.to_bits(),
                "{ty}: fused {fused} vs unpacked {batched} — not bit-identical"
            );
        }
    }

    /// Reusing one buffer across rows must not leak state between them.
    #[test]
    fn unpacked_buffer_is_safe_to_reuse() {
        let mut rng = Rng::new(0x5EED);
        let n = 256;
        let mut buf = UnpackedRow::new(n);
        let mut act = ActivationQ8::new(n);
        act.quantize(&rng.floats(n));

        // A wide format first, then a narrower one into the same buffer.
        let big = weight_row(GgmlType::Q6_K, 1, &mut rng);
        unpack_row(GgmlType::Q6_K, &big, &mut buf);
        let _ = row_dot_unpacked(&buf, &act);

        let small = weight_row(GgmlType::Q8_0, 8, &mut rng);
        unpack_row(GgmlType::Q8_0, &small, &mut buf);
        assert_eq!(
            row_dot_unpacked(&buf, &act).to_bits(),
            row_dot_q8_0(&small, &act).to_bits(),
            "stale state from the previous unpack leaked through"
        );
    }

    /// What quantising the activation actually costs, per format.
    ///
    /// This is the real accuracy question, and it is a documented trade rather
    /// than a bug: 8-bit activations are what make the integer inner loop
    /// possible in the first place.
    #[test]
    fn activation_quantisation_error_is_small() {
        let mut rng = Rng::new(0xC0DE);
        for ty in [
            GgmlType::Q8_0,
            GgmlType::Q5_0,
            GgmlType::Q4_0,
            GgmlType::Q4_K,
            GgmlType::Q6_K,
        ] {
            // 896 elements, the real d_model, so the averaging is realistic.
            let n_blocks = 896 / ty.block_elems();
            let n = n_blocks * ty.block_elems();
            let row = weight_row(ty, n_blocks, &mut rng);
            let mut w = vec![0.0f32; n];
            dequantize_row(ty, &row, &mut w).unwrap();

            let x = rng.floats(n);
            let mut act = ActivationQ8::new(n);
            act.quantize(&x);

            let fused = kernel(ty)(&row, &act);
            let exact = dot(&w, &x);
            let rel = (fused - exact).abs() / exact.abs().max(1e-3);
            std::eprintln!("  {ty:<6} n={n:<4} fused vs exact-f32: rel {rel:.3e}");
            assert!(
                rel < 5e-2,
                "{ty}: activation quantisation cost {rel:e} is too high"
            );
        }
    }

    #[test]
    fn activation_quantisation_round_trips_within_half_a_step() {
        let mut rng = Rng::new(7);
        let x = rng.floats(256);
        let mut act = ActivationQ8::new(256);
        act.quantize(&x);
        let mut back = vec![0.0f32; 256];
        act.dequantize(&mut back);
        for b in 0..8 {
            let chunk = &x[b * 32..(b + 1) * 32];
            let amax = chunk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let step = amax / 127.0;
            for i in 0..32 {
                let j = b * 32 + i;
                assert!(
                    (back[j] - x[j]).abs() <= 0.5 * step + 1e-6,
                    "element {j}: {} vs {}",
                    back[j],
                    x[j]
                );
            }
        }
    }

    #[test]
    fn all_zero_activation_gives_zero() {
        // A zero block has a zero scale; nothing may divide by it.
        let mut act = ActivationQ8::new(64);
        act.quantize(&[0.0; 64]);
        let mut rng = Rng::new(3);
        for ty in [GgmlType::Q8_0, GgmlType::Q5_0, GgmlType::Q4_0] {
            let row = weight_row(ty, 2, &mut rng);
            let v = kernel(ty)(&row, &act);
            assert_eq!(v, 0.0, "{ty} against a zero activation");
        }
    }

    #[test]
    fn unsupported_format_is_rejected_not_guessed() {
        let data = vec![0u8; 176];
        let m = crate::tensor::QuantMatrix::new(GgmlType::Q5_K, &data, 1, 256);
        // Q5_K is not implemented anywhere, so it never reaches a kernel.
        assert!(m.is_err());
    }
}

// ================================================= block decomposition ==

mod decompose {
    use super::super::*;
    use crate::gguf::GgmlType;

    /// The decomposition must reproduce the kernel exactly, for every format.
    ///
    /// `decompose_block` is a second, independent expression of each format's
    /// packing -- written for a UI rather than for speed -- so agreeing with
    /// `dequantize_row` bit for bit is a real cross-check of both.
    #[test]
    fn reconstructs_exactly_what_dequantise_produces() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut byte = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        };

        for ty in [
            GgmlType::Q8_0,
            GgmlType::Q5_0,
            GgmlType::Q4_0,
            GgmlType::Q4_K,
            GgmlType::Q6_K,
        ] {
            let mut src: Vec<u8> = (0..ty.block_bytes()).map(|_| byte()).collect();
            // Keep the f16 scales finite; random bit patterns are full of NaN.
            let d = f32_to_f16(0.02).to_le_bytes();
            match ty {
                GgmlType::Q8_0 | GgmlType::Q5_0 | GgmlType::Q4_0 => src[0..2].copy_from_slice(&d),
                GgmlType::Q4_K => {
                    src[0..2].copy_from_slice(&d);
                    src[2..4].copy_from_slice(&f32_to_f16(0.01).to_le_bytes());
                }
                GgmlType::Q6_K => src[208..210].copy_from_slice(&d),
                _ => {}
            }

            let mut want = vec![0.0f32; ty.block_elems()];
            dequantize_row(ty, &src, &mut want).unwrap();

            let dec = decompose_block(ty, &src).unwrap();
            assert_eq!(dec.quants.len(), ty.block_elems(), "{ty} quant count");
            assert_eq!(
                dec.scales.len(),
                ty.block_elems() / dec.group,
                "{ty} scale count"
            );
            assert_eq!(
                dec.reconstruct(),
                want,
                "{ty}: decomposition disagrees with the kernel"
            );

            // Every quant must sit inside the range the format can represent.
            let (lo, hi) = dec.quant_range;
            for (i, &q) in dec.quants.iter().enumerate() {
                assert!(
                    q >= lo && q <= hi,
                    "{ty} element {i}: {q} outside {lo}..={hi}"
                );
            }
            // Only Q4_K is affine; everything else must have a zero offset.
            if ty != GgmlType::Q4_K {
                assert!(dec.mins.iter().all(|m| *m == 0.0), "{ty} should be linear");
            }
        }
    }

    #[test]
    fn rejects_a_wrong_sized_block() {
        assert!(decompose_block(GgmlType::Q4_K, &[0u8; 143]).is_err());
        assert!(decompose_block(GgmlType::Q5_K, &[0u8; 176]).is_err());
    }
}
