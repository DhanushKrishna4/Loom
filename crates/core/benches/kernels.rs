//! Native benchmarks for the quantised kernels.
//!
//! # What this is and is not for
//!
//! These run on the host, and the host is not the target. SIMD128 is a wasm
//! feature, so the number that decides whether a kernel rewrite ships comes from
//! `tools/wasm_bench.js` running the real thing under a wasm engine. What
//! criterion is good for is the *native* dev loop: it catches an algorithmic
//! regression in seconds, without a wasm rebuild, and its statistics are far
//! better than a stopwatch.
//!
//! Run with `cargo bench -p nano-infer-core`.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use nano_infer_core::gguf::GgmlType;
use nano_infer_core::quant::{self, ActivationQ8};
use nano_infer_core::tensor::QuantMatrix;

/// d_model for Qwen2.5-0.5B: one real weight row.
const N: usize = 896;

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

/// A weight row whose scales are forced into a plausible range; random f16 bit
/// patterns would otherwise be full of infinities and denormals, and denormals
/// in particular would make this a benchmark of the FPU's slow path.
fn weight_row(ty: GgmlType, n_blocks: usize, rng: &mut Rng) -> Vec<u8> {
    let mut row = rng.bytes(n_blocks * ty.block_bytes());
    let d = quant::f32_to_f16(0.02).to_le_bytes();
    for b in 0..n_blocks {
        let o = b * ty.block_bytes();
        match ty {
            GgmlType::Q8_0 | GgmlType::Q5_0 | GgmlType::Q4_0 => row[o..o + 2].copy_from_slice(&d),
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

fn row_dots(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_dot");
    group.throughput(Throughput::Elements(N as u64));

    for ty in [
        GgmlType::Q8_0,
        GgmlType::Q5_0,
        GgmlType::Q4_0,
        GgmlType::Q4_K,
        GgmlType::Q6_K,
    ] {
        let n_blocks = N / ty.block_elems();
        let n = n_blocks * ty.block_elems();
        let mut rng = Rng(0xFEED_1234_5678_9ABC);
        let row = weight_row(ty, n_blocks, &mut rng);
        let mut act = ActivationQ8::new(n);
        act.quantize(&rng.floats(n));
        let f = kernel(ty);

        group.bench_function(ty.name(), |b| {
            b.iter(|| f(black_box(&row), black_box(&act)))
        });
    }
    group.finish();
}

/// The fused path against the dequantise-then-dot oracle it replaced. This is
/// the comparison that justified step 6 existing at all.
fn fused_vs_oracle(c: &mut Criterion) {
    let ty = GgmlType::Q5_0; // 55% of the real model's weights
    let rows = 64;
    let mut rng = Rng(0x1234);
    let n_blocks = N / ty.block_elems();
    let data = {
        let mut v = Vec::new();
        for _ in 0..rows {
            v.extend_from_slice(&weight_row(ty, n_blocks, &mut rng));
        }
        v
    };
    let w = QuantMatrix::new(ty, &data, rows, N).unwrap();
    let x = rng.floats(N);
    let mut act = ActivationQ8::new(N);
    act.quantize(&x);
    let mut out = vec![0.0f32; rows];
    let mut scratch = vec![0.0f32; N];

    let mut group = c.benchmark_group("matvec");
    group.throughput(Throughput::Elements((rows * N) as u64));
    group.bench_function("fused", |b| {
        b.iter(|| nano_infer_core::ops::matvec_fused(black_box(&w), black_box(&act), &mut out))
    });
    group.bench_function("dequant_oracle", |b| {
        b.iter(|| {
            nano_infer_core::ops::matvec_dequant(
                black_box(&w),
                black_box(&x),
                &mut out,
                &mut scratch,
            )
        })
    });
    group.finish();
}

/// Runs once per matmul rather than once per row, so it is amortised over ~900
/// to ~4900 rows. Benchmarked to confirm it stays negligible.
fn activation_quantise(c: &mut Criterion) {
    let mut rng = Rng(7);
    let x = rng.floats(N);
    let mut act = ActivationQ8::new(N);
    c.bench_function("activation_quantise", |b| {
        b.iter(|| act.quantize(black_box(&x)))
    });
}

criterion_group!(benches, row_dots, fused_vs_oracle, activation_quantise);
criterion_main!(benches);
