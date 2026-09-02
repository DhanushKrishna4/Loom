//! A tiny wasm module for verifying and timing the kernels **on wasm**.
//!
//! SIMD128 is a wasm feature. Benchmarking the scalar path on an aarch64 laptop
//! says nothing about whether a `v128` rewrite helps in a browser, and shipping
//! an unverified SIMD kernel because it "looks right" is exactly the class of
//! silent-wrongness this project keeps trying to avoid.
//!
//! So: plain `extern "C"` exports, no wasm-bindgen, no JS glue. Node loads the
//! `.wasm` directly and calls in with integers. Everything the benchmark needs
//! is built inside the module from a fixed seed, so nothing crosses the boundary
//! but numbers.
//!
//! Run it with `node tools/wasm_bench.js`.

use nano_infer_core::gguf::GgmlType;
use nano_infer_core::quant::{self, ActivationQ8};

/// Deterministic filler, so every run measures identical data.
struct Rng(u64);

impl Rng {
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

fn ty_of(kind: u32) -> GgmlType {
    match kind {
        0 => GgmlType::Q8_0,
        1 => GgmlType::Q5_0,
        2 => GgmlType::Q4_0,
        3 => GgmlType::Q4_K,
        _ => GgmlType::Q6_K,
    }
}

/// A weight row with scales forced into a sane range, matching the native tests.
fn weight_row(ty: GgmlType, n_blocks: usize, rng: &mut Rng) -> Vec<u8> {
    let mut row = rng.bytes(n_blocks * ty.block_bytes());
    let d = quant::f32_to_f16(0.02).to_le_bytes();
    for b in 0..n_blocks {
        let o = b * ty.block_bytes();
        match ty {
            GgmlType::Q8_0 | GgmlType::Q5_0 | GgmlType::Q4_0 => {
                row[o..o + 2].copy_from_slice(&d);
            }
            GgmlType::Q4_K => {
                row[o..o + 2].copy_from_slice(&d);
                row[o + 2..o + 4].copy_from_slice(&quant::f32_to_f16(0.01).to_le_bytes());
            }
            GgmlType::Q6_K => {
                row[o + 208..o + 210].copy_from_slice(&d);
                for i in 0..16 {
                    row[o + 192 + i] = ((row[o + 192 + i] as i8) / 4) as u8;
                }
            }
            _ => {}
        }
    }
    row
}

fn kernel(ty: GgmlType) -> fn(&[u8], &ActivationQ8) -> f32 {
    match ty {
        GgmlType::Q8_0 => quant::row_dot_q8_0,
        GgmlType::Q5_0 => quant::row_dot_q5_0,
        GgmlType::Q4_0 => quant::row_dot_q4_0,
        GgmlType::Q4_K => quant::row_dot_q4_k,
        _ => quant::row_dot_q6_k,
    }
}

fn setup(kind: u32, n_elems: usize) -> (GgmlType, Vec<u8>, ActivationQ8) {
    let ty = ty_of(kind);
    let n_blocks = n_elems / ty.block_elems();
    let n = n_blocks * ty.block_elems();
    let mut rng = Rng(0xFEED_1234_5678_9ABC);
    let row = weight_row(ty, n_blocks, &mut rng);
    let x = rng.floats(n);
    let mut act = ActivationQ8::new(n);
    act.quantize(&x);
    (ty, row, act)
}

/// Check every kernel against dequantise-then-dot, **on wasm**.
///
/// Returns a bitmask of failures, one bit per format, so a SIMD path that is
/// right on x86 and wrong on wasm cannot slip through. 0 means all passed.
#[no_mangle]
pub extern "C" fn self_test() -> u32 {
    let mut failures = 0u32;
    for kind in 0..5u32 {
        let (ty, row, act) = setup(kind, 896);
        let n = act.len();

        let mut w = vec![0.0f32; n];
        if quant::dequantize_row(ty, &row, &mut w).is_err() {
            failures |= 1 << kind;
            continue;
        }
        let mut xq = vec![0.0f32; n];
        act.dequantize(&mut xq);

        let got = kernel(ty)(&row, &act);
        let want: f32 = w.iter().zip(&xq).map(|(a, b)| a * b).sum();
        let tol = 1e-3 * want.abs().max(1e-2);
        if !(got - want).abs().le(&tol) {
            failures |= 1 << kind;
        }
    }
    failures
}

/// Time `iters` row dot products. Returns a sink so nothing is optimised away.
#[no_mangle]
pub extern "C" fn bench_row_dot(kind: u32, n_elems: u32, iters: u32) -> f32 {
    let (ty, row, act) = setup(kind, n_elems as usize);
    let f = kernel(ty);
    let mut sink = 0.0f32;
    for _ in 0..iters {
        sink += f(&row, &act);
    }
    sink
}

/// Time the unpack-once path used by batched prefill.
///
/// `reuse` is how many activation vectors share one unpack — 1 models decode,
/// 32 models a prefill chunk. The interesting number is how the per-dot cost
/// falls as the unpack is amortised.
#[no_mangle]
pub extern "C" fn bench_row_dot_unpacked(kind: u32, n_elems: u32, reuse: u32, iters: u32) -> f32 {
    let (ty, row, _) = setup(kind, n_elems as usize);
    let n = n_elems as usize;

    // `reuse` DISTINCT activations, not one reused. An earlier version of this
    // benchmark dotted the same vector every time, so it sat in L1 and reported
    // batching as a 2x win -- while the real path cycles a different activation
    // per token and streams all of them past every weight row. Measuring the
    // access pattern you do not have is worse than not measuring.
    let mut rng = Rng(0xA5A5_1234_5678_9ABC);
    let acts: Vec<ActivationQ8> = (0..reuse.max(1) as usize)
        .map(|_| {
            let mut a = ActivationQ8::new(n);
            a.quantize(&rng.floats(n));
            a
        })
        .collect();

    let mut unpacked = quant::UnpackedRow::new(n);
    let mut sink = 0.0f32;
    for _ in 0..iters {
        quant::unpack_row(ty, &row, &mut unpacked);
        for a in &acts {
            sink += quant::row_dot_unpacked(&unpacked, a);
        }
    }
    sink
}

/// Time activation quantisation, which runs once per matmul.
#[no_mangle]
pub extern "C" fn bench_quantize(n_elems: u32, iters: u32) -> f32 {
    let mut rng = Rng(0x1234_5678_9ABC_DEF0);
    let x = rng.floats(n_elems as usize);
    let mut act = ActivationQ8::new(n_elems as usize);
    let mut sink = 0.0f32;
    for _ in 0..iters {
        act.quantize(&x);
        sink += act.len() as f32;
    }
    sink
}

/// Reports whether this module was compiled with SIMD128 enabled, so the
/// benchmark cannot silently report scalar numbers as a SIMD result.
#[no_mangle]
pub extern "C" fn has_simd128() -> u32 {
    u32::from(cfg!(target_feature = "simd128"))
}
