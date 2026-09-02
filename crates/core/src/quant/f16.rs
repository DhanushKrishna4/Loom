//! IEEE-754 binary16 <-> binary32, by hand.
//!
//! Every ggml block format stores its scale as an f16, so this is the very first
//! thing in the numeric stack and everything downstream inherits its mistakes.
//! It is also the classic place to be *almost* right: the normal path is four
//! lines and obviously correct, and then subnormals quietly produce values that
//! are wrong by a factor of two and output that still looks like English.
//!
//! Layout reminder:
//!
//! ```text
//!            sign  exponent  mantissa   bias   subnormal value
//!   binary16   1      5         10       15    mant * 2^-24
//!   binary32   1      8         23      127    mant * 2^-149
//! ```
//!
//! The four cases that matter, in the order the code handles them:
//!
//! * `exp == 0, mant == 0`  -> signed zero
//! * `exp == 0, mant != 0`  -> subnormal; must be *renormalised* into f32, since
//!   every f16 subnormal is a perfectly ordinary normal f32
//! * `exp == 31`            -> inf (mant 0) or NaN (mant != 0); the payload is
//!   shifted up so a signalling NaN does not turn into an infinity
//! * otherwise              -> normal; rebias by 127-15 = 112 and shift the
//!   mantissa left by 23-10 = 13

/// f16 bit pattern -> f32. Exact for every one of the 65536 inputs.
#[inline]
pub const fn f16_to_f32(h: u16) -> f32 {
    let sign = (h as u32 & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;

    let bits = if exp == 0 {
        if mant == 0 {
            // +/-0
            sign
        } else {
            // Subnormal: the value is mant * 2^-24, which f32 represents as a
            // normal number. Shift the mantissa up until its leading 1 sits at
            // bit 10, then account for the shift in the exponent.
            //
            // `mant` is in 1..=1023, so `leading_zeros()` on the u32 is 22..=31
            // and `shift` lands in 0..=9... except for mant == 1, where the
            // leading bit must travel all 10 places. Concretely:
            //   mant = 1    -> shift 10, exponent 103 -> 2^-24  (smallest f16)
            //   mant = 1023 -> shift  1, exponent 112 -> ~6.1e-5 (largest subnormal)
            let shift = mant.leading_zeros() - 21;
            let m = (mant << shift) & 0x03ff; // drop the now-implicit leading 1
            let e = 113 - shift; // 127 - 15 + 1 - shift
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        // inf or NaN. Shifting the mantissa left by 13 keeps a NaN a NaN
        // (non-zero payload stays non-zero) and keeps an inf an inf.
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        // Normal. 127 - 15 = 112.
        sign | ((exp + 112) << 23) | (mant << 13)
    };

    f32::from_bits(bits)
}

/// f32 -> f16 bit pattern, round-to-nearest-even (the IEEE default, and what
/// ggml's quantiser uses when it writes block scales).
///
/// Needed for building test fixtures and, later, for quantising activations.
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let raw_exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x007f_ffff;

    if raw_exp == 0xff {
        // inf or NaN. Force a non-zero payload for NaN so it cannot collapse
        // into infinity when the low mantissa bits are the only ones set.
        return if mant == 0 {
            sign | 0x7c00
        } else {
            sign | 0x7c00 | ((mant >> 13) as u16) | 0x0200
        };
    }

    // Exponent in f16's frame of reference.
    let exp = raw_exp - 127 + 15;

    if exp >= 0x1f {
        // Overflows f16's range; IEEE round-to-nearest gives infinity.
        return sign | 0x7c00;
    }

    if exp <= 0 {
        // Result is subnormal in f16 (or underflows to zero).
        //
        // v = m * 2^(e-23) with m = mant | implicit 1, and an f16 subnormal is
        // h * 2^-24, so h = m >> (14 - exp). Anything past exp < -10 shifts the
        // whole mantissa out; ties-to-even then rounds it to zero.
        if exp < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - exp) as u32; // 14..=24
        let h = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        // Round half to even. If this carries h up to 0x400 the bit pattern is
        // exactly the smallest normal, which is the correct answer.
        let h = if rem > halfway || (rem == halfway && (h & 1) == 1) {
            h + 1
        } else {
            h
        };
        return sign | h as u16;
    }

    // Normal. Drop 13 mantissa bits, rounding half to even. A carry out of the
    // mantissa flows into the exponent field for free, and a carry out of the
    // exponent lands exactly on the inf pattern -- both correct.
    let h = ((exp as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    let h = if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h + 1
    } else {
        h
    };
    sign | h as u16
}

/// Read a little-endian f16 from the first two bytes of `b` and widen it.
///
/// Every block format starts with (or ends with) a scale in this form, so this
/// is the single place that knows the byte order of a ggml scale.
#[inline]
pub fn read_f16_le(b: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[at], b[at + 1]]))
}
