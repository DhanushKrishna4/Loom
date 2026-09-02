//! Taking a block apart into the pieces it is actually made of.
//!
//! Every format this engine supports reconstructs a weight the same way:
//!
//! ```text
//! value[i] = scale[group(i)] * quant[i] - min[group(i)]
//! ```
//!
//! What differs is how wide a group is, how the quant is unpacked, and whether
//! the offset exists at all. [`decompose_block`] normalises all of that into one
//! shape, which makes the bit-packing legible instead of invisible -- the whole
//! point of the quantisation explorer.
//!
//! It is also the shape a *batched* fused kernel wants: unpack a weight row once
//! into quants plus scales, then reuse it across many activation vectors. That
//! refactor is still outstanding, and this is the half of it that now exists and
//! is tested.

use alloc::vec;
use alloc::vec::Vec;

use super::f16::read_f16_le;
use super::k_quants::scale_min_k4;
use super::QuantError;
use crate::gguf::GgmlType;

/// One block, taken apart.
///
/// `values[i] == scales[i / group] * quants[i] as f32 - mins[i / group]`, exactly.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockDecomp {
    /// The stored integer for each element, *after* any bias is folded in — so
    /// Q4_0's raw nibble 0..15 appears here as -8..7, which is the number the
    /// arithmetic actually uses.
    pub quants: Vec<i32>,
    /// Multiplier per scale group.
    pub scales: Vec<f32>,
    /// Offset per scale group, subtracted. Zero for every linear format; only
    /// Q4_K, which is affine, has non-zero entries.
    pub mins: Vec<f32>,
    /// Elements covered by one scale. 32 for the legacy formats and Q4_K's
    /// sub-blocks, 16 for Q6_K.
    pub group: usize,
    /// Smallest and largest integer the format can store, for drawing a scale.
    pub quant_range: (i32, i32),
}

impl BlockDecomp {
    /// Reconstruct the block. Must equal [`super::dequantize_row`] exactly.
    pub fn reconstruct(&self) -> Vec<f32> {
        self.quants
            .iter()
            .enumerate()
            .map(|(i, &q)| {
                let g = i / self.group;
                self.scales[g] * q as f32 - self.mins[g]
            })
            .collect()
    }

    /// Gap between adjacent representable values in a group.
    ///
    /// This is the quantisation resolution: any weight is stored to within half
    /// of this, and it is the honest answer to "what did quantising cost here".
    pub fn step(&self, group: usize) -> f32 {
        self.scales[group].abs()
    }
}

/// Decompose one block of `ty` into quants, scales and offsets.
pub fn decompose_block(ty: GgmlType, src: &[u8]) -> Result<BlockDecomp, QuantError> {
    if src.len() != ty.block_bytes() {
        return Err(QuantError::BadSourceLength {
            ty,
            got: src.len(),
            want: ty.block_bytes(),
        });
    }

    Ok(match ty {
        GgmlType::Q8_0 => {
            let d = read_f16_le(src, 0);
            BlockDecomp {
                quants: src[2..34].iter().map(|&b| (b as i8) as i32).collect(),
                scales: vec![d],
                mins: vec![0.0],
                group: 32,
                quant_range: (-128, 127),
            }
        }
        GgmlType::Q4_0 => {
            let d = read_f16_le(src, 0);
            let qs = &src[2..18];
            let mut quants = vec![0i32; 32];
            for j in 0..16 {
                // Low nibbles are the first half of the block, high nibbles the
                // second -- not an interleave.
                quants[j] = (qs[j] & 0x0f) as i32 - 8;
                quants[j + 16] = (qs[j] >> 4) as i32 - 8;
            }
            BlockDecomp {
                quants,
                scales: vec![d],
                mins: vec![0.0],
                group: 32,
                quant_range: (-8, 7),
            }
        }
        GgmlType::Q5_0 => {
            let d = read_f16_le(src, 0);
            let qh = u32::from_le_bytes([src[2], src[3], src[4], src[5]]);
            let qs = &src[6..22];
            let mut quants = vec![0i32; 32];
            for j in 0..16 {
                // Bit i of qh is element i's fifth bit, straight through 0..32.
                let h0 = ((qh >> j) & 1) as u8;
                let h1 = ((qh >> (j + 16)) & 1) as u8;
                quants[j] = ((qs[j] & 0x0f) | (h0 << 4)) as i32 - 16;
                quants[j + 16] = ((qs[j] >> 4) | (h1 << 4)) as i32 - 16;
            }
            BlockDecomp {
                quants,
                scales: vec![d],
                mins: vec![0.0],
                group: 32,
                quant_range: (-16, 15),
            }
        }
        GgmlType::Q4_K => {
            let d = read_f16_le(src, 0);
            let dmin = read_f16_le(src, 2);
            let scales_raw: &[u8; 12] = src[4..16].try_into().expect("12 bytes");
            let qs = &src[16..144];

            let mut quants = vec![0i32; 256];
            let mut scales = vec![0.0f32; 8];
            let mut mins = vec![0.0f32; 8];
            for s in 0..8 {
                let (sc, m) = scale_min_k4(s, scales_raw);
                scales[s] = d * sc as f32;
                mins[s] = dmin * m as f32;
                let base = (s / 2) * 32;
                for l in 0..32 {
                    // Sub-blocks 2c and 2c+1 share 32 bytes: even takes low
                    // nibbles, odd takes high.
                    quants[s * 32 + l] = if s % 2 == 0 {
                        (qs[base + l] & 0x0f) as i32
                    } else {
                        (qs[base + l] >> 4) as i32
                    };
                }
            }
            // Unsigned: Q4_K is the affine format, so the offset carries the sign.
            BlockDecomp {
                quants,
                scales,
                mins,
                group: 32,
                quant_range: (0, 15),
            }
        }
        GgmlType::Q6_K => {
            let d = read_f16_le(src, 208);
            let ql_all = &src[0..128];
            let qh_all = &src[128..192];
            let sc_all = &src[192..208];

            let mut quants = vec![0i32; 256];
            let mut scales = vec![0.0f32; 16];
            for (g, s) in scales.iter_mut().enumerate() {
                *s = d * (sc_all[g] as i8) as f32;
            }
            for n in 0..2 {
                let ql = &ql_all[n * 64..n * 64 + 64];
                let qh = &qh_all[n * 32..n * 32 + 32];
                for l in 0..32 {
                    let h = qh[l];
                    let e = n * 128 + l;
                    quants[e] = ((ql[l] & 0x0f) | ((h & 3) << 4)) as i32 - 32;
                    quants[e + 32] = ((ql[l + 32] & 0x0f) | (((h >> 2) & 3) << 4)) as i32 - 32;
                    quants[e + 64] = ((ql[l] >> 4) | (((h >> 4) & 3) << 4)) as i32 - 32;
                    quants[e + 96] = ((ql[l + 32] >> 4) | (((h >> 6) & 3) << 4)) as i32 - 32;
                }
            }
            BlockDecomp {
                quants,
                scales,
                mins: vec![0.0; 16],
                group: 16,
                quant_range: (-32, 31),
            }
        }
        other => return Err(QuantError::UnsupportedType(other)),
    })
}
