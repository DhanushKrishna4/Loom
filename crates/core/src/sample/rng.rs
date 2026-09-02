//! A small, seedable PRNG.
//!
//! PCG32 rather than xorshift: same size, same speed, but it passes statistical
//! test suites that xorshift fails, and here the RNG *is* the reproducibility
//! guarantee. A seed goes in the URL so a specific generation can be shared, so
//! the sequence has to be identical on every machine, in every build, forever.
//!
//! Everything below is fixed-width integer arithmetic with explicit wrapping, so
//! native and wasm produce bit-identical streams.

/// PCG-XSH-RR 64/32, the variant from the reference implementation.
#[derive(Debug, Clone)]
pub struct Pcg32 {
    state: u64,
    /// Stream selector. Always odd, which is what makes the sequence full-period.
    inc: u64,
}

const MULT: u64 = 6_364_136_223_846_793_005;

impl Pcg32 {
    /// Seed the generator. The same seed always yields the same sequence.
    pub fn new(seed: u64) -> Self {
        Self::with_stream(seed, 0xda3e_39cb_94b9_5bdb)
    }

    pub fn with_stream(seed: u64, stream: u64) -> Self {
        let mut r = Pcg32 {
            state: 0,
            inc: (stream << 1) | 1,
        };
        // The reference seeding routine: step, add the seed, step again.
        r.next_u32();
        r.state = r.state.wrapping_add(seed);
        r.next_u32();
        r
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(MULT).wrapping_add(self.inc);
        // Output function: xorshift the high bits down, then rotate by the top
        // five. The rotation is what defeats the lattice structure a plain LCG
        // has.
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in `[0, 1)`.
    ///
    /// Built from 24 bits so every value is exactly representable in f32 and the
    /// result can never round up to 1.0 -- which would walk off the end of a
    /// cumulative-probability scan.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}
