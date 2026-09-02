#!/usr/bin/env node
// Verify and time the quantised kernels ON WASM.
//
// SIMD128 is a wasm feature: benchmarking the scalar path natively says nothing
// about whether a v128 rewrite helps in a browser. This loads the .wasm built
// from crates/wasmbench directly -- no wasm-bindgen, no glue, just integers in
// and floats out -- and reports both correctness and ns/element.
//
//   cargo build -p nano-infer-wasmbench --target wasm32-unknown-unknown --release
//   node tools/wasm_bench.js [path-to-wasm]

// ESM, because the project's package.json declares "type": "module" for Vite.
// Adding that field is what broke this file's `require` calls -- caught by the
// same CI step that runs this harness.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const FORMATS = ['q8_0', 'q5_0', 'q4_0', 'q4_K', 'q6_K'];
const N_ELEMS = 896;   // Qwen2.5-0.5B's d_model: one real weight row
const ITERS = 20000;

function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

async function main() {
  const wasmPath = process.argv[2] ||
    path.join(__dirname, '..', 'target', 'wasm32-unknown-unknown', 'release',
              'nano_infer_wasmbench.wasm');
  if (!fs.existsSync(wasmPath)) {
    console.error(`not found: ${wasmPath}\nbuild it first:\n  cargo build -p nano-infer-wasmbench --target wasm32-unknown-unknown --release`);
    process.exit(1);
  }

  const bytes = fs.readFileSync(wasmPath);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const ex = instance.exports;

  const simd = ex.has_simd128() === 1;
  console.log(`module:  ${path.basename(wasmPath)} (${(bytes.length / 1024).toFixed(1)} KiB)`);
  console.log(`simd128: ${simd ? 'ENABLED' : 'disabled'}`);
  console.log(`node:    ${process.version}\n`);

  // --- correctness first. A fast wrong kernel is worthless. ---------------
  const failures = ex.self_test();
  if (failures !== 0) {
    const bad = FORMATS.filter((_, i) => failures & (1 << i));
    console.error(`SELF TEST FAILED for: ${bad.join(', ')} (mask ${failures})`);
    process.exit(1);
  }
  console.log(`self test: all ${FORMATS.length} kernels match dequantise-then-dot\n`);

  // --- timing -------------------------------------------------------------
  console.log(`row dot, ${N_ELEMS} elements, ${ITERS} iterations, median of 5:`);
  console.log(`${'format'.padEnd(8)} ${'total'.padStart(10)} ${'per row'.padStart(12)} ${'per elem'.padStart(12)}`);

  const results = {};
  for (let k = 0; k < FORMATS.length; k++) {
    // Warm up properly. 200 iterations was not enough for the first format
    // measured: the JIT was still tiering up and reported ~2x its steady-state
    // cost, which reads exactly like a regression in an unrelated kernel.
    ex.bench_row_dot(k, N_ELEMS, ITERS);
    const runs = [];
    for (let t = 0; t < 5; t++) {
      const t0 = performance.now();
      const sink = ex.bench_row_dot(k, N_ELEMS, ITERS);
      const dt = performance.now() - t0;
      if (!Number.isFinite(sink)) throw new Error(`${FORMATS[k]} produced ${sink}`);
      runs.push(dt);
    }
    const ms = median(runs);
    const perRow = (ms * 1e6) / ITERS;         // ns
    const perElem = perRow / N_ELEMS;          // ns
    results[FORMATS[k]] = perElem;
    console.log(
      `${FORMATS[k].padEnd(8)} ${ms.toFixed(1).padStart(8)} ms ` +
      `${perRow.toFixed(0).padStart(9)} ns ${perElem.toFixed(3).padStart(9)} ns`
    );
  }

  // The unpack-once path against the fused one, at decode (reuse 1) and prefill
  // (reuse 32) batch sizes. This is the measurement that decides whether batched
  // prefill is worth its extra code.
  if (ex.bench_row_dot_unpacked) {
    console.log(`\nunpack-once path, ns per dot (fused shown for comparison):`);
    console.log(`${'format'.padEnd(8)} ${'fused'.padStart(10)} ${'reuse=1'.padStart(10)} ${'reuse=32'.padStart(10)}`);
    for (let k = 0; k < FORMATS.length; k++) {
      const time = (reuse, iters) => {
        ex.bench_row_dot_unpacked(k, N_ELEMS, reuse, Math.max(1, iters / 4));
        const runs = [];
        for (let t = 0; t < 5; t++) {
          const t0 = performance.now();
          ex.bench_row_dot_unpacked(k, N_ELEMS, reuse, iters);
          runs.push((performance.now() - t0) * 1e6 / (iters * reuse));
        }
        return median(runs);
      };
      const r1 = time(1, 4000);
      const r32 = time(32, 200);
      console.log(
        `${FORMATS[k].padEnd(8)} ${(results[FORMATS[k]] * N_ELEMS).toFixed(0).padStart(7)} ns ` +
        `${r1.toFixed(0).padStart(7)} ns ${r32.toFixed(0).padStart(7)} ns`
      );
    }
  }

  // Activation quantisation runs once per matmul, not once per row, so it is
  // amortised over ~900-4900 rows. Reported to confirm it stays negligible.
  ex.bench_quantize(N_ELEMS, 200);
  const q0 = performance.now();
  ex.bench_quantize(N_ELEMS, ITERS);
  const qms = performance.now() - q0;
  console.log(`\nactivation quantise: ${((qms * 1e6) / ITERS).toFixed(0)} ns per ${N_ELEMS}-element vector`);

  // A decode step for Qwen2.5-0.5B: 24 layers x (q,k,v,o,gate,up,down) plus the
  // unembedding, weighted by each tensor's actual format.
  // The real file's mix: everything with a 896-long row falls back to a
  // 32-element format, and only ffn_down (rows of 4864) can hold a k-quant --
  // 12 layers Q4_K, 12 layers Q6_K.
  const perLayerCommon =
    896 * 896 * results['q5_0'] +      // attn_q
    128 * 896 * results['q5_0'] +      // attn_k
    128 * 896 * results['q8_0'] +      // attn_v
    896 * 896 * results['q5_0'] +      // attn_output
    4864 * 896 * results['q5_0'] +     // ffn_gate
    4864 * 896 * results['q5_0'];      // ffn_up
  const ffnDown = 896 * 4864;
  const total =
    24 * perLayerCommon +
    12 * ffnDown * results['q4_K'] +
    12 * ffnDown * results['q6_K'] +
    151936 * 896 * results['q8_0'];    // unembedding
  console.log(`\nprojected decode step (matmuls only): ${(total / 1e6).toFixed(0)} ms ` +
              `-> ${(1000 / (total / 1e6)).toFixed(2)} tok/s`);
  console.log('(single-threaded, ignoring attention and sampling)');

  fs.writeFileSync(
    path.join(__dirname, 'reference', 'wasm-bench.json'),
    JSON.stringify({ simd, node: process.version, nsPerElement: results }, null, 2)
  );
}

main().catch((e) => { console.error(e); process.exit(1); });
