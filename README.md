# Loom

An LLM inference engine written from scratch in Rust, compiled to WebAssembly,
running a small language model entirely in the browser. No backend, no ML
crates, no `ndarray` — the forward pass is the project.

**Status: all 15 build steps complete. It runs in a browser, a revisit is instant,
you can watch it think, and you can see what quantisation actually did to the
weights.**

```
$ tools/build_web.sh          # wasm-pack -> tsc -> vite -> dist/
$ python3 tools/serve.py      # serves the repo root, so /models is reachable
```

Load a GGUF, type a prompt, watch it stream. **18.1 tok/s in-browser**, and the
same prompt produces byte-identical text to the native CLI — which is the payoff
for using libm on both targets rather than the host's intrinsics.

The 469 MB model is cached in IndexedDB, so the second visit is **0.6 s from
cache** instead of a re-download.

```
$ nano-infer generate model.gguf "The capital of France is" --n 12
 Paris. It is the largest city in Europe and the second
```

Dequantisation is **bit-exact against ggml**, the ops match PyTorch, the
tokenizer produces **identical token IDs to HuggingFace**, and the full forward
pass agrees with PyTorch **layer by layer** on the same weights.

## Quick start

```bash
cargo test                       # 159 tests in ~1s (127 need no model file)
cargo run -p nano-infer-cli -- help
```

Against a real model:

```bash
cargo run -p nano-infer-cli -- gguf-dump model.gguf --all
cargo run -p nano-infer-cli -- dequant model.gguf blk.0.ffn_gate.weight --blocks 2
cargo run -p nano-infer-cli -- tokenize model.gguf "Hello, world! 2024 don't 🙂"
cargo run -p nano-infer-cli -- tokenize model.gguf "Hi" --chat
cargo run --release -p nano-infer-cli -- generate model.gguf "Once upon a time" --n 32
cargo run --release -p nano-infer-cli -- generate model.gguf "Hi" --chat \
    --temp 0.8 --top-k 40 --top-p 0.95 --repeat-penalty 1.1 --seed 42
```

Without one, generate a synthetic stand-in with the same architecture metadata:

```bash
python3 tools/make_tiny_gguf.py /tmp/tiny.gguf
cargo run -p nano-infer-cli -- gguf-dump /tmp/tiny.gguf --all
```

## Layout

```
crates/core/   pure compute, no_std + alloc, one dependency (libm)
  gguf/        container parsing, metadata, model config
  quant/       f16 conversion, dequantisation kernels
  math/        the float math core lacks: sqrt, exp, sin, cos, powf, round
  tensor/      shapes, views, and the quantised weight matrix
  ops/         naive f32 kernels + an f64 oracle
  tokenizer/   byte-level BPE, pre-tokenizer, chat template
  model/       weights, KV cache, forward pass
  sample/      greedy, temperature, top-k, top-p, penalties, seeded PCG32
crates/wasm/   the wasm-bindgen boundary: Engine, DecodeResult, Stats
crates/wasmbench/  correctness + timing harness that runs ON wasm
web/           a page that loads a model and streams tokens
  app.ts       load, generate, and drive the panels
  idb.ts       IndexedDB model cache, no dependencies
  viz.ts       attention heatmap, KV cache, perf panel, quant explorer — all canvas
  bench.ts     side-by-side against transformers.js
crates/cli/    native driver — debug numerics here, never in a browser
crates/wasm/   wasm-bindgen boundary, marshalling only
tools/         python: fixture generation, synthetic model generation
```

`core` has exactly **one dependency**, `libm` (dev-only: `proptest`) — see
Decisions below. It builds `no_std` and for `wasm32-unknown-unknown`; both are
checked on every change.

## What's implemented

### GGUF (step 1)

Full v2/v3 parser: header, all 13 metadata value types including nested arrays,
tensor table, alignment-padded data section. Config extraction resolves the
architecture-namespaced keys (`qwen2.block_count`, not `block_count`) so it works
on any Llama-family GGUF, not just Qwen.

Tensor data is **never copied** — `tensor_data()` returns a subslice of the input
buffer, and a test asserts the returned pointer lies inside it.

Hostile-input handling matters here because the model arrives from a CDN: every
file-supplied `u64` count is bounds-checked against remaining bytes *before* it
reaches a `Vec::with_capacity` or gets narrowed to `usize` (which is 32-bit on
wasm). `rejects_truncation_at_every_prefix` parses all ~2400 strict prefixes of a
valid file and requires every one to return `Err` rather than panic.

### Quantisation (step 2)

| format | block | bytes | bits/weight | share of the real model |
|--------|-------|-------|-------------|------------------------|
| Q5_0 | 32 | 22 | 5.5 | **54.9%** |
| Q8_0 | 32 | 34 | 8.5 | 30.1% |
| Q6_K | 256 | 210 | 6.5625 | 8.8% |
| Q4_K | 256 | 144 | 4.5 | 6.1% |
| F32 | 1 | 4 | 32 | 0.1% |
| Q4_0 / F16 | 32 / 1 | 18 / 2 | 4.5 / 16 | not present (baselines) |

Unsupported types still have correct block geometry (so tensor offsets stay
right) but return `QuantError::UnsupportedType` rather than producing garbage.

That share column is the surprise, and it is worth internalising: **a "Q4_K_M"
Qwen2.5-0.5B is 55% Q5_0 and only 6% Q4_K.** k-quants need rows that are a
multiple of their 256-element super-block, and this model's rows are 896 long
(896 / 256 = 3.5). Only `ffn_down`, with rows of 4864 (= 19 × 256), can hold one
— that is the 12 Q4_K + 12 Q6_K tensors, one per layer. Everything else falls
back to a 32-element format. Q5_0 was not in the original format list and is not
optional: without it the model does not load.

f16↔f32 is implemented by hand including subnormals, and verified exhaustively
over all 65536 bit patterns.

**These are not the inference kernels.** Dequantising Qwen2.5-0.5B to f32 would
turn ~400 MB into ~2 GB. The decode path (step 6) fuses dequantisation into the
matmul inner loop; these row functions are the oracle those kernels get tested
against.

### Tensor and ops (step 3)

`matmul_nt`, `matvec`, `rmsnorm`, `softmax`, `silu`, `swiglu`, `rope`, plus the
elementwise helpers the residual stream needs.

Every kernel is the obvious loop, deliberately. These are the versions that get
validated first; step 9 replaces the hot ones and keeps `ops::reference` as the
oracle. All matmuls are "row times row" — GGUF stores weights with the input
dimension contiguous, so no transpose is ever materialised, in either direction.
`matvec` is a separate entry point from day one because at batch size 1 it is
memory-bandwidth bound rather than compute bound and wants a different
optimisation strategy entirely.

Accuracy against PyTorch, worst case over the reference tensors:

| op | worst abs | worst rel |
|----|-----------|-----------|
| rmsnorm | 2.9e-6 | 7.4e-7 |
| softmax | 1.8e-7 | 1.2e-6 |
| silu | 1.9e-6 | 5.3e-7 |
| swiglu | 1.9e-6 | 2.3e-7 |
| matmul | 4.8e-6 | 8.5e-6 |
| rope (both conventions) | 6.0e-7 | 6.5e-6 |

That is ~70x f32 epsilon at worst, against a 1e-3 budget. The numbers are printed
by `cargo test -- --nocapture` and exist to be the yardstick when step 9 starts
reordering summations: the question then is not whether the results move (they
will) but whether they move further from PyTorch than this.


### Tokenizer (step 4)

Byte-level BPE, built from the vocab in the GGUF metadata. Four parts, each of
which can be silently wrong on its own:

- **Byte-to-unicode mapping.** GPT-2's bijection from all 256 bytes onto
  printable codepoints. This is why a GGUF vocab is full of tokens like `Ġthe` —
  that is `" the"`, with byte 0x20 written as U+0120.
- **Pre-tokenizer.** Qwen2's pattern, hand-written, because merges may never
  cross a pre-token boundary and `regex` is not an allowed dependency. Unicode
  classification comes from generated range tables (~6.6 KB).
- **Merge loop.** Lowest-ranked adjacent pair first, merging every
  non-overlapping occurrence in one pass — which is what the reference does, and
  it matters: merging one at a time and re-scanning can pick a different pair
  in between.
- **Special tokens and chat template.** Qwen2's `<|im_start|>` format, including
  the default system prompt the model was tuned with.

No `HashMap` anywhere — `core` has none and `hashbrown` is not an allowed
dependency, so string→id and pair→merge are sorted arrays with a binary search,
built once at load.

```
$ nano-infer tokenize model.gguf "Hello, world! 2024 don't 🙂"
  0    9707  "Hello"      4     220  " "        8      19  "4"
  1      11  ","          5      17  "2"        9    1513  " don"
  2    1879  " world"     6      15  "0"       10     944  "'t"
  3       0  "!"          7      17  "2"       11   27484  " 🙂"
```

Digits split one at a time, the contraction stays whole, and the space attaches
to the following word — all three are decided by the pre-tokenizer, and all three
match HuggingFace exactly.

### Forward pass (step 5)

Pre-norm decoder: RMSNorm in, raw block output added to the residual. RMSNorm not
LayerNorm, SwiGLU not GELU, GQA not MHA, RoPE on Q and K only. Positions are
processed one at a time, appending to the KV cache, so prefill and decode are the
same code path — step 7 splits them.

**Qwen2 has QKV biases and Llama does not.** Loading them is not optional: with
them dropped nothing crashes, every attention score is just quietly wrong. The
loader also asserts each matrix's `(rows, cols)`, because GGUF's axis order is
reversed from torch's and a transposed weight produces plausible garbage rather
than an error.

Weights are never materialised as f32. `matvec_dequant` unpacks one row into a
scratch buffer, uses it, and discards it: 463 MB of Q5_0 would be 2 GB as f32,
which does not fit in wasm's budget and would be bandwidth suicide anyway. It is
still the *unfused* form — step 6 keeps the unpacked weights in registers, and
this function becomes the oracle that kernel is tested against.

That unfused path runs at ~2.4 tok/s natively. It is the correctness path, and
it is still available behind `--unfused` as the oracle for step 6's kernels.

### Fused quantised matmul (step 6)

The spec calls this the single most important performance decision in the
project, and it is: at batch size 1 there is no reuse to amortise memory traffic
against, so the traffic *is* the cost.

The activation vector is quantised to 8-bit **once per matmul**, then the inner
loop is an integer dot product between two quantised blocks with the scales
applied once per block. Weights are unpacked into registers, used, and dropped —
nothing dequantised ever reaches memory. Q, K and V share one quantisation
(they read the same normed vector), as do gate and up, so four matmuls per layer
cost two quantisations.

**The reconstruction form decides the kernel.** For the 32-element formats
(Q8_0, Q5_0, Q4_0) the reconstruction is linear once the bias is folded into a
signed quant, so a block collapses to one integer dot times `d_w * d_a`. Q4_K is
*affine* — `d*scale*q - dmin*min` — and the constant term does not vanish:

```
sum_i (d*sc*q_i - dmin*m) * a_i
  = d_a * ( d*sc * sum_i(q_i * qa_i)  -  dmin*m * sum_i(qa_i) )
                   ^^^^ integer dot        ^^^^ integer sum, precomputed
```

That second term is why `ActivationQ8` carries a per-block sum. ggml solves the
same problem by putting the same extra field in its activation formats (`Q8_1`,
`Q8_K`). Q6_K is linear again but scales per *16* elements, so its accumulator
splits sixteen ways per super-block instead of one.

| path | decode | prefill |
|------|--------|---------|
| `matvec_dequant` (oracle) | 2.41 tok/s | 2.35 tok/s |
| fused quantised | **9.22 tok/s** | 9.00 tok/s |

3.8x, with byte-identical generated text. Still scalar — the i8 multiply-
accumulate is exactly what SIMD128's `i32x4.dot_i16x8_s` wants, and that is
step 9.

### KV cache (step 7)

Preallocated at load and never resized: a reallocation mid-generation would
invalidate every `Float32Array` view JS holds into wasm memory, which is the
whole reason for the "preallocate everything" rule.

**The layout benchmark.** The spec says to pick by measurement rather than
intuition, so both layouts are implemented and selectable:

```
[layer][position][kv_head][head_dim]   PositionMajor  contiguous write, strided read
[layer][kv_head][position][head_dim]   HeadMajor      scattered write, contiguous read
```

Both reduce to `base + t * stride`, so the layout branch is hoisted out of the
attention loop and the two are compared on memory behaviour rather than on
differing index arithmetic.

Isolated attention gather, one layer, all heads, min of 5 trials:

| history | PositionMajor | HeadMajor | delta |
|---------|--------------|-----------|-------|
| 256 | 76.5 µs | 76.0 µs | +0.6% |
| 1024 | 306.1 µs | 303.3 µs | +0.9% |
| 4096 | 1241.6 µs | 1227.7 µs | +1.1% |

**HeadMajor wins by about 1%, and that is the whole story — my intuition said it
should be much more.** The reason contiguity barely helps: `head_dim` is 64
floats, so one head's slice is 256 bytes, already four whole cache lines. The
"strided" read touches four lines out of every eight at a perfectly regular
stride, which the hardware prefetcher handles without waste. Contiguity only
pays when the runs are short enough to share cache lines with data you do not
want — it would matter far more for a model with a small `head_dim` or many
KV heads.

End to end the difference disappears entirely into noise, because attention is
roughly 5% of the memory traffic per decode step: the cache read is ~25 MB at
1024 positions against 463 MB of weights. `bench-kv` prints both views; the
end-to-end one is the reason the default is chosen on a 1% margin rather than
being treated as important.

Default: `HeadMajor`, on that 1%, since the scattered write costs nothing
measurable at two KV heads.

### Sampling (step 8)

Greedy, temperature, top-k, top-p, and a repetition penalty, drawn with a seeded
PCG32. Filters run in the order HuggingFace and llama.cpp both use — penalty,
temperature, top-k, top-p, softmax, draw — and the order is not cosmetic:
temperature *before* top-p changes which tokens fall inside the nucleus, because
it changes the probabilities the cumulative sum runs over.

Two details worth stating, because both are easy to get backwards:

- **The repetition penalty divides positive logits and multiplies negative
  ones** (CTRL's formulation, which HuggingFace adopted). Naively dividing both
  would *promote* a negative logit rather than suppress it. There is a test for
  exactly that.
- **PCG32, not xorshift.** Same size and speed, but it passes test suites
  xorshift fails — and here the RNG *is* the reproducibility guarantee, since a
  seed in a URL has to reproduce a generation on someone else's machine.

  That last clause was, for a while, a claim the page did not honour: the CLI
  took `--seed`, the engine was properly seeded, and the web UI had a seed box,
  but nothing ever put the value in the URL, so a generation could not actually
  be shared. `?seed=<n>` is now read on load and written back with
  `replaceState` from `applySampling()` — the one place the seed reaches the
  engine, so the URL cannot drift from what is running.

```
$ ... --temp 0.7 --top-k 40 --seed 3 --repeat-penalty 1.0
 Red, Blue, and Green. A quick brown fox jumps over the lazy dog.

$ ... --temp 0.7 --top-k 40 --seed 3 --repeat-penalty 1.15
 Red, Blue and Green.
 Sure! Here are the colors I've chosen:
```

top-k uses `select_nth_unstable` rather than a full sort. top-p needs its
survivors ordered, so it sorts them — with `top_k` set first (the usual case)
that is a sort of `k` elements, not of 151936.

### Optimisation (step 9)

Measured in a real wasm engine, not natively. SIMD128 is a wasm feature, so
benchmarking the scalar path on an aarch64 laptop says nothing about whether a
`v128` rewrite helps in a browser — and `tools/wasm_bench.js` refuses to report a
SIMD result without calling `has_simd128()` first, so a mis-set flag cannot
quietly produce scalar numbers under a SIMD heading.

That probe is narrower than it looks, though: `has_simd128()` returns
`cfg!(target_feature = "simd128")`, which reports how the compiler was *invoked*,
not what came out the other end. The spec's own warning was to check the
disassembly rather than assume, so `tools/build_web.sh` now does exactly that —
`wasm-dis` over the shipped module, counting real `v128` instructions:

```
    simd128: 4716 v128 instructions emitted (floor 500)
```

The assumption turned out to be correct, which is the ordinary outcome and not
a reason to have left it unchecked. The floor exists because losing SIMD is
silent: a dropped `RUSTFLAGS` or a `.cargo/config.toml` that stopped being read
still builds, still returns correct answers, and is roughly 3x slower.

ns per element, one 896-element weight row, median of 5, with a projected decode
step recomputed for every stage from the same formula and the real format mix:

| stage | q8_0 | q5_0 | q4_0 | q4_K | q6_K | ms | tok/s |
|-------|------|------|------|------|------|-----|-------|
| baseline: scalar, no simd128 | 0.468 | 0.691 | 0.297 | 0.357 | 0.560 | 286.0 | 3.50 |
| `+ -C target-feature=+simd128` | 0.122 | 0.272 | 0.122 | 0.234 | 0.563 | 126.8 | 7.89 |
| + q5_0 bit-expansion table | 0.124 | 0.149 | 0.123 | 0.236 | 0.563 | 96.3 | 10.38 |
| + q6_K loop split | 0.122 | 0.148 | 0.123 | 0.236 | 0.161 | 74.8 | 13.37 |
| + explicit SIMD128 `idot32` | 0.120 | 0.096 | 0.075 | 0.080 | 0.160 | **53.3** | **18.77** |
| ~~q8_0 through the shared `idot32`~~ | 0.128 | — | — | — | — | 54.4 | 18.39 |

**5.4x overall.** What each step taught:

- **The compiler flag was the biggest single win, at zero code cost.** Enabling
  `simd128` more than doubled throughput before a line of SIMD was written — LLVM
  autovectorises the integer dot loops once allowed to. It now lives in
  `.cargo/config.toml` so it cannot be forgotten.
- **It did nothing at all for Q6_K** (0.560 -> 0.563). That kernel scatters into
  `group[e / 16]`, and LLVM will not vectorise a store to a computed index.
  Splitting the loop at 16 makes every accumulator index loop-invariant, leaving
  four independent reductions: 3.5x from a change that moves no arithmetic.
- **Q5_0 needed a table, not intrinsics.** Its fifth bits arrive as a 32-bit
  mask, and extracting them one at a time is 32 dependent shift-and-mask pairs
  nothing can widen. A 2 KiB bit-to-byte table replaces them with four lookups,
  and the loop that consumes them vectorises: 1.8x.
- **Hand-written intrinsics still beat the autovectoriser**, 1.5x on Q5_0 and
  2.9x on Q4_K. That is the one place in the crate that uses `unsafe` — an
  explicit exception to `#![deny(unsafe_code)]` for two in-bounds 16-byte loads.
- **One change was reverted.** Routing Q8_0 through the shared SIMD `idot32`
  needs its bytes reinterpreted as `i8`, which without `unsafe` means a 32-byte
  copy — and that measured *slower* (0.128 against 0.122). Q8_0 is the only format
  whose quants need no unpacking, so it is the only one where the copy is not
  amortised against real work.

All of it is integer-exact: reordering an integer sum cannot change it, and the
golden generation test produces byte-identical token ids before and after. The
speed cost no accuracy at all.

Native criterion benches live in `cargo bench -p nano-infer-core`. They are for
the dev loop — they catch an algorithmic regression in seconds without a wasm
rebuild — not for deciding what ships. The comparison they do settle:

```
matvec/fused           3.84 us
matvec/dequant_oracle 47.77 us     12.4x
```

### The wasm boundary and a page (step 10)

`Engine::new / tokenize / prefill / decodeStep / reset / setSampling / stats`,
and a page that drives them.

**There is deliberately no `generate()` that loops internally.** wasm runs on the
thread that called it, so a loop inside the module freezes the tab for the whole
generation — no repaint, no input, no Stop button. JS calls `decodeStep()` once
per turn of the event loop and stays in charge.

Three things the browser taught that native testing could not:

- **`requestAnimationFrame` was the wrong scheduler.** The obvious choice, and it
  does not fire in a hidden tab — switching away mid-generation stopped it dead,
  permanently. Caught by driving the page in a browser whose tab reported
  `visibilityState: "hidden"`: it never produced a token. `setTimeout(fn, 0)`
  yields just as well, still repaints between tokens, and keeps running hidden.
- **A background tab is throttled to ~2.6 tok/s**, against 18.1 in a tight loop.
  That is timer throttling, not the engine — measured by calling `decodeStep()`
  straight from the console, which gave 55.2 ms per token against the wasm
  benchmark's 54 ms projection. A Web Worker is the real fix.
- **`Vec<u8>` not `&[u8]` for the model bytes.** wasm-bindgen copies a slice
  argument into wasm memory, and `.to_vec()` would then copy it *again*. At
  469 MB that second copy is the difference between loading and running out of
  memory.

The engine is self-referential — the parsed model borrows the file buffer — so the
buffer is leaked into `&'static [u8]` at load rather than reaching for `unsafe`
or a self-ref crate. Stated plainly in the module docs: **loading a second model
leaks the first.** For a page that loads one model and keeps it that is the right
trade; `reset()` is for a new conversation and allocates nothing.

`logitsView()` hands back a `Float32Array` aliasing wasm memory with no copy, and
its doc comment spells out the hazard: any allocation that grows linear memory
detaches the buffer and leaves the view zero-length. That is exactly why the
workspace and KV cache are preallocated — steady-state decoding does not grow
memory, so a view taken between steps stays valid.

```
web/pkg/nano_infer_wasm_bg.wasm: 250858 bytes raw, 86817 gzipped
```

87 KB gzipped against a 500 KB budget, and that is before `wasm-opt`.

### Model caching (step 11)

`web/idb.ts`, about 100 lines and no dependencies.

**IndexedDB, not localStorage or the Cache API.** localStorage caps at ~5–10 MB
and is synchronous — three orders of magnitude short of a 469 MB model. The Cache
API would work but is keyed on Request/Response and re-runs the fetch pipeline;
IndexedDB stores the bytes and nothing else.

Models are stored as **Blobs, not ArrayBuffers**: a Blob can stay backed by disk
until it is read, so writing one does not need a second 469 MB copy in the JS
heap. The page also calls `navigator.storage.persist()` before storing — without
it a cache that size is "best effort" and can be evicted under disk pressure,
which silently turns an instant revisit back into a download.

| path | time |
|------|------|
| first visit: download + cache + parse | 469 MB transfer |
| revisit: read from IndexedDB + parse | **0.6 s** |

On a revisit the URL field is preselected to the most recently cached model, so
loading is one click and no download. Quota refusals are handled as a normal
outcome rather than an error: the model still loads, it just will not persist.

**One bug worth recording.** The first version's transaction helper resolved to
the `IDBRequest` object whenever `request.result` was `undefined` — which is what
a `get` returns on a miss. `IDBRequest` is truthy, so **every cache miss looked
like a hit**, and the caller then read `.bytes` off a request object. It surfaced
on the very first load against an empty database, which is the one case a
half-tested cache is guaranteed to hit and the one most easily skipped.

### Visualisations (step 12)

All canvas, no DOM. A 24x14 attention grid is 336 cells redrawn every token; as
DOM nodes that is 336 style recalculations per token competing with the decode
loop for the same thread.

**Attention heatmap.** Each cell is one head, rows are layers, shaded by how
*sharply* that head is attending rather than how much — every distribution sums
to 1, so magnitude carries no information and concentration is the whole signal.
Click a cell for that head's full distribution over the context, labelled with
the token it is looking at.

The brightness mapping is linear, and that is a measurement rather than a
default: sampled across all 336 heads on a real prompt, concentration already
spans the range (min 0.06, quartiles 0.29 / 0.48 / 0.75, max 1.00). An earlier
gamma of 0.55 washed the entire grid out by correcting a signal that needed no
correction.

**KV cache.** Positions used against allocated, with the allocation in MiB and a
tick every 256 positions so the scale is read rather than inferred.

**Perf panel.** A stacked bar fed from the engine's *own* per-op timings, not a
stopwatch around the call — so "overhead" genuinely means sampling plus
detokenisation plus the boundary crossing, and not "everything I forgot to
instrument". A representative step:

```
matmul 129.9 ms · attention 0.9 ms · elementwise 2.0 ms · overhead 0.6 ms
```

97.4% in the projections, which is what every optimisation decision in step 9
was premised on and had never actually been measured end to end until now.

Two pieces of plumbing this needed:

- **`core` has no clock.** It is `no_std` and knows nothing about the host, so
  the host supplies one: `Workspace::set_clock(fn() -> f64)`. Absent, `tick()`
  returns 0.0 and every bucket comes out zero, so instrumentation costs nothing
  when nobody asked for it. The wasm side caches the `performance.now` lookup,
  because the forward pass calls it ~150 times per token and four `Reflect`
  lookups per call would be instrumentation that changes what it measures.
- **Attention capture writes into a preallocated buffer** (`n_layers * n_heads *
  max_seq` floats, ~1.4 MiB), allocated unconditionally at load. Allocating it
  lazily would grow wasm linear memory mid-generation, which detaches every
  `Float32Array` the page holds. 1.4 MiB is a cheap price for never doing that.

### Quantisation explorer (step 13)

Pick any of the 291 tensors, pick a block, and see what is actually stored.

Every format this engine supports reconstructs a weight the same way:

```
value[i] = scale[group(i)] * quant[i] - min[group(i)]
```

What differs is how wide a group is, how the quant is unpacked, and whether the
offset exists at all. `quant::decompose_block` normalises all of that into one
shape — which is what makes the bit-packing legible instead of invisible, and is
the whole reason Q4_K was the scariest part of this project.

The explorer shows the stored integers as a grid, the reconstructed values
plotted against **the grid of levels they can occupy** (every weight lands
exactly on a line — that spacing *is* what quantisation cost), and a per-group
table:

```
q4_K · block 0 of 17024 · 256 weights in 144 bytes · 4.50 bits/weight · 8 groups of 32

group   scale       min         step      max error vs peak
0       1.279e-3    1.020e-2    1.28e-3   6.27%
1       1.389e-3    1.036e-2    1.39e-3   6.70%
2       1.147e-3    5.593e-3    1.15e-3   5.48%
```

That last column is the honest answer to "what did quantising cost here": each
weight is stored to within half a step, so ~6% of its sub-block's largest weight.
The `min` column only appears for Q4_K, because it is the only affine format —
every other one has a zero offset, and showing an empty column would imply
otherwise.

`decompose_block` is not only for the UI. It is exactly the shape a **batched**
fused kernel wants — unpack a row once into quants plus scales, then reuse it
across many activation vectors — so the half of that still-outstanding refactor
that can be written and tested independently now exists and is tested.

### Benchmark against transformers.js (step 14)

`web/bench.html` runs a fixed prompt through both engines: same model family,
same prompt, greedy decoding on both, one thread each.

Measured on a **loaded** 8 GB machine (see the caveat below):

| | load | prefill | decode | tok/s |
|---|---|---|---|---|
| Loom | 1.0 s | 0.45 s | 2.08 s | 11.5 – 17.8 |
| transformers.js (ONNX q4f16) | 44.3 s | 1.74 s | 15.55 s | 1.48 |

**Loom is roughly 8–12x faster at decoding here.** The range is honest: the
host was swapping, and the same build measured anywhere from 4.2 to 17.8 tok/s in
the browser across the session. The stable reference is the Node harness on the
same wasm: 16.2 / 18.5 / 17.2 tok/s across three consecutive runs.

Where the gap comes from, most likely first:

- **Weight format meets the platform.** q4f16 dequantises into fp16 and
  multiplies in float; wasm has no native fp16 arithmetic, so that is emulated.
  Loom quantises activations to int8 and keeps the inner loop integer,
  which maps onto `i16x8.extmul` and `i32x4.extadd_pairwise` — instructions wasm
  actually has. A platform-fit difference, not evidence of better code.
- **Single-threaded is deployment-accurate, not a handicap imposed for effect.**
  ONNX Runtime's threaded wasm needs SharedArrayBuffer, which needs
  cross-origin isolation, which needs COOP/COEP headers — and GitHub Pages
  cannot set them. Neither engine gets threads where this ships. transformers.js
  would be several times faster with four threads somewhere that can.
- **Prefill.** ORT batches the whole prompt into one matmul; Loom still
  loops one position at a time. This is the one place Loom is genuinely
  behind, and the fix is designed and unbuilt.
- **Generality.** ORT executes an arbitrary graph. Loom is 24 hardcoded
  Qwen2 layers with a fused kernel per format. Specialising is worth a lot, and
  it is also why this engine runs exactly one architecture.

Not identical quantisations — no two runtimes share one. q4f16 is the closest
ONNX build by size (483 MB against 469 MB) and bits per weight.

**The harness defends itself**, because this one nearly published a wrong number
twice:

- It counts the other engine's tokens with `token_callback_function`, not
  `callback_function`. The latter fires per decoded *text chunk*: an earlier
  version counted ~13 callbacks for ~20 tokens and reported transformers.js as
  1.5x slower than it was. Counting the other side's tokens wrong is the easiest
  way to publish a flattering benchmark.
- It runs Loom's decode **twice and reports the spread**, refusing to
  headline a ratio when the passes disagree by more than 20%. A single timing
  cannot tell you whether it measured the engine or the machine — this page read
  15.21 and then 4.20 tok/s with no code change, on a host that had gone 6 GB
  into swap.
- Results persist across reloads, because the two runtimes cannot share a tab on
  8 GB: 469 MB of GGUF plus 461 MB of ONNX graph pushes the machine into paging,
  and then the second number measures swap. Run one, reload, run the other.

### Deployment (step 15)

`.github/workflows/deploy.yml`: test → build → deploy to Pages, with the tests
gating the deploy. A build that fails its numerics must not reach production.

```
wasm       246029 bytes raw, 93201 gzipped  (budget 512000, 18% used)
dist/      304 KB total, hashed and minified
```

**The size budget is enforced, not aspirational.** `tools/build_web.sh` fails the
build if the gzipped wasm exceeds 500 KB. A target nobody checks is a target that
quietly stops holding.

**wasm-opt is verified, not trusted.** It rewrites the module, and the shipped
artifact is the optimised one — so CI re-runs the kernel self-test *after*
optimisation, on the opt'd binary. Locally the optimised build produces the exact
golden token sequence, byte-identical to the native CLI:

```
[12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 304, 279, 1879]
```

#### The frontend stack, and a deviation that cost something

The build is **wasm-pack → tsc → vite → dist/**, all through
`tools/build_web.sh` so CI and local runs cannot diverge.

An earlier version of this project shipped wasm-bindgen directly, no bundler, and
plain JavaScript. Two of those were argued for; the third — dropping TypeScript —
was never decided, it just followed from the other two. That chain is worth
naming, because each step made the next look reasonable: no npm dependencies, so
no bundler; no bundler, so no TS tooling; no TS tooling, so no types.

It cost two real bugs. `Float32Array.prototype.map` returns a *typed array*, so
mapping scales to HTML strings coerced every result back to a number and rendered
`NaNNaNNaN`. Under `tsc` that is:

```
error TS2322: Type 'string' is not assignable to type 'number'.
```

The IndexedDB bug — reading `.bytes` off an `IDBRequest` because a cache miss
returned the request object — is now impossible by construction: `tx<T>()` is
typed to resolve `T | undefined`, so a miss has to be handled.

Vite's `base` is `'./'` rather than `/repo-name/`: every asset reference is
already relative, and hardcoding the repo name would tie the build to one
repository and break local preview.

#### What the first real CI run cost

Live at <https://dhanushkrishna4.github.io/Loom/>. The
workflow needed three rounds, and the last one is the interesting one.

**1. Typecheck ran before the thing it typechecks existed.** `web/pkg` is
generated by wasm-pack and gitignored, so in a fresh checkout there are no
Engine types; every import failed and the rest of the errors cascaded from
that. The typecheck now lives in the build job, after wasm-pack. It never
failed locally because `web/pkg` had been sitting there since the first build.

**2. The kernel self-test passed and then died writing its own output.**
`tools/reference/` holds generated dumps and is gitignored, so it exists on a
machine that has run the dump scripts and nowhere else. Reproduced in a clean
clone before fixing, and fixed with one `mkdirSync`.

**3. Ubuntu ships binaryen 108, and 108 miscompiles this module.** The site
deployed green and then threw `RangeError: WebAssembly.Table.grow(): failed to
grow table by 4` on load, before a single token. The module declares two
tables:

| index | type | limits |
|---|---|---|
| 0 | funcref | min 110, **max 110** |
| 1 | externref | min 1024, no max |

and the glue initialises itself with `wasm.__wbindgen_externrefs.grow(4)`. In
the local build that export points at table 1. In the CI build it pointed at
table 0 — capped, so the grow could never succeed. Same source, same
wasm-bindgen; the only difference was `apt-get install binaryen` giving 108
(2022) where local dev has 132.

What makes this one worth writing down is why nothing caught it. The damage is
structural, not behavioural: the rewritten module still validates, still
instantiates under Node, still passes the kernel self-test, and still fits the
size budget. Every gate in the pipeline was green. Only a browser actually
executing the glue could see it, and until this push nothing ever had.

So the fix is two things, not one. Pinning binaryen to 132 fixes this instance.
`tools/check_wasm_tables.js` fixes the class: it parses the shipped binary,
resolves `__wbindgen_externrefs` through the table index space, and fails the
build unless it lands on an externref table with no maximum. It was written
against both binaries — it passes the good one and rejects the exact artifact
that was serving at the time.

Verified end to end on the deployed site afterwards: 468.6 MiB fetched from the
HuggingFace CDN (CORS and `content-length` both fine from `github.io`), parsed
and ready in 1.3 s, cached in IndexedDB, generating text with the attention
map, op timings and quantisation explorer all live.

### Batched prefill

Prefill and decode are genuinely different problems. At batch size 1 there is no
reuse: the model is ~463 MB and every token needs all of it. During prefill many
tokens are available at once, so a weight row can be unpacked **once per chunk**
instead of once per token.

Both paths produce **bit-identical** results, and that took deliberate care
rather than luck. Float multiplication is not associative, so the batched kernel
folds its scales in exactly the same order as the fused one — `(scale_w *
scale_a) * dot` for the linear formats, `scale_a * (scale_w * dot - min_w * sum)`
for Q4_K, which is the only affine one. There is a test asserting the two agree
to the bit, and another asserting a batched prefill and an incremental decode
produce identical logits across a chunk boundary.

**Which path wins depends on the target, so it is chosen per target and both are
measured:**

| | per-token prefill | batched prefill |
|---|---|---|
| wasm (112 tokens, browser, best of 3) | 17.0 tok/s | **43.3 tok/s** |
| native aarch64 (112 tokens) | **21.6 tok/s** | 11.0 tok/s |

2.55x on the platform this ships to, 2x *slower* on the one it is developed on.
The default follows `cfg!(target_arch = "wasm32")`, and `--batched-prefill` on
the CLI overrides it.

Two measurement mistakes on the way to that table, both worth recording:

- The first microbenchmark reused **one** activation vector across the batch, so
  it sat in L1 and reported a clean 2x for every format. The real path cycles a
  different activation per token and streams all of them past every weight row.
  Measuring an access pattern you do not have is worse than not measuring.
- The native regression looked like an inlining problem, so `inline(always)` went
  on the scalar fallbacks. It changed nothing. Chunk size from 4 to 32 changed
  almost nothing either. The honest conclusion is that the native fused kernels
  are simply good enough that the buffer round-trip does not pay — which is why
  the default is per-target rather than universal.

## Testing

107 tests, none of which need a model file.

**Ops** are checked in four layers: analytic values worked out by hand; an **f64
oracle** in `ops::reference` (f64, not a second f32 copy, so it measures real
numerical error rather than just catching transcription slips); proptest
invariants; and **reference tensors from PyTorch**.

```bash
python3 tools/dump_reference_ops.py     # needs torch; writes tools/reference/
cargo test -p nano-infer-core matches_numpy_reference -- --nocapture
```

That script cross-checks its own numpy transcription against torch and refuses to
write a reference the two disagree on. They currently agree to f64 precision
(~1e-16), and the RoPE reference calls HuggingFace's own `apply_rotary_pos_emb`
rather than re-deriving it — re-deriving the thing you are trying to verify is
not verification.

**Quantisation** has the same shape, and its top layer is now populated. 24
blocks pulled from the real Qwen2.5-0.5B file — 4 each of Q4_K, Q5_0, Q6_K, Q8_0
and 8 of F32 — dequantised by gguf-py and compared against ours:

```
checked 24 real blocks (8x f32, 4x q5_0, 4x q8_0, 4x q4_K, 4x q6_K),
worst abs error 0e0
```

**Bit-exact**, not merely within tolerance, so the test asserts exactly that.
Q4_K's 6-bit packed scales — the thing most likely to be silently wrong — are
verified against ggml's own output on real weights. Flipping a single bit in one
fixture makes the test fail, which was checked rather than assumed.

Regenerate with:

```bash
pip install gguf numpy
python3 tools/dump_gguf_blocks.py models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

The fixtures are committed (they are small); the model is not.

**The tokenizer** is checked against HuggingFace's own tokenizer for
`Qwen/Qwen2.5-0.5B-Instruct`, on a corpus built to be awkward: multiple leading
spaces, NBSP and zero-width space, digits, contractions in both cases, CJK,
Arabic, Hebrew, Devanagari, ZWJ emoji sequences, flag pairs, code, URLs, Windows
paths, and injected `<|im_*|>` markers.

```bash
pip install transformers
python3 tools/dump_reference_tokens.py     # downloads tokenizer.json (~7 MB)
cargo test -p nano-infer-core tokenizer -- --nocapture
#   80 corpus cases match HuggingFace exactly
```

Exact ID sequences, not round-trips. A wrong pre-tokenizer split, an off-by-one
merge rank, or a mishandled contraction all round-trip perfectly while producing
different IDs — and different IDs mean a different forward pass.

**The forward pass** is checked against PyTorch layer by layer, running the
*same weights* — dequantised out of the same GGUF — so a disagreement is our
arithmetic rather than the quantiser's. RoPE and the GQA head expansion come from
transformers' own `apply_rotary_pos_emb` and `repeat_kv`, so the two things most
likely to be wrong are not re-derived in the reference.

```bash
python3 tools/dump_reference_activations.py models/....gguf
cargo test --release -p nano-infer-core -- --ignored --nocapture
```

```
hidden.0       worst abs 0.000e0   worst rel 0.000e0     <- embedding, bit-exact
hidden.1       worst abs 9.298e-6  worst rel 7.415e-4
...
hidden.24      worst abs 3.815e-5  worst rel 4.349e-4
final_norm     worst abs 1.564e-4  worst rel 4.351e-4
logits         worst abs 5.132e-5  worst rel 1.179e-2
argmax agrees: token 12095
```

Absolute error stays around 1e-5 through all 24 layers — f32 accumulation order
against torch's BLAS, not a defect — and the predicted token agrees exactly.
That test runs the **unfused** path on purpose: PyTorch computes in f32 against
the same weights, so it is apples-to-apples and can hold a tight tolerance.

**The fused path is checked separately**, because it quantises activations to 8
bits and the numbers legitimately move. Worst element-wise relative error is the
wrong metric there — a hidden state is full of near-zero components, and dividing
by them turns a 1e-3 absolute difference into a relative error of 5. Relative L2
and cosine similarity answer the question that matters:

```
layer              rel L2       cosine
hidden.1         7.143e-3     0.999978
hidden.12        1.596e-2     0.999873
hidden.24        2.881e-2     0.999590
final_norm       2.514e-2     0.999689
argmax: unfused 12095, fused 12095
logits cosine (fused vs unfused): 0.999747
```

Error grows with depth, the direction does not drift, and the prediction is
unchanged. The kernels themselves are also checked against "dequantise the
weights, then dot against the *same* quantised activation", which isolates the
fused arithmetic from the cost of quantising at all — those agree to f32
rounding.

**The KV cache** has three equivalence tests, all bit-exact:

- both layouts produce **identical** logits — they hold the same numbers in a
  different order, so any difference is an indexing bug, and an indexing bug here
  reads as a model that has merely got worse;
- incremental decode equals a single pass over the same tokens — the property the
  cache exists to provide, and where position tracking and `len` bookkeeping fail
  if they are wrong;
- `reset()` returns the cache to a genuinely empty state, so the next
  conversation cannot attend to the previous one's tokens.

These six tests are `#[ignore]`d: they run real forward passes and take ~50 s in
a debug build against 1 s for everything else, so they are a verification you ask
for rather than a cost on every save:

```bash
cargo test --release -p nano-infer-core -- --ignored --nocapture
```

**The wasm build has its own self-test.** `crates/wasmbench` exports a
`self_test()` that checks every kernel against dequantise-then-dot *inside the
wasm module*, returning a per-format bitmask so a SIMD path that is correct
natively and wrong on wasm cannot slip through. `tools/wasm_bench.js` runs it
before it will print a single timing number.

```bash
cargo build -p nano-infer-wasmbench --target wasm32-unknown-unknown --release
node tools/wasm_bench.js
```

**The block decomposition** is checked against the kernels bit for bit, for all
five formats: `decompose_block(...).reconstruct()` must equal `dequantize_row`
exactly. It is a second, independent expression of each format's packing —
written for a UI rather than for speed — so exact agreement cross-checks both.
Every quant is also asserted to sit inside the range its format can represent,
and only Q4_K is allowed a non-zero offset.

**The captured attention** is checked as a real probability distribution: every
head's weights sum to 1 within 1e-4, all lie in [0,1], and at position 0 every
head attends to itself with weight exactly 1. A heatmap draws a plausible picture
whatever is in the buffer, so this is the only thing separating a visualisation
that is informative from one that is decorative.

**Time to first token** is reported separately from prefill, because they are
not the same number: the first decode step is paid before any character appears.
It is latched on that first step and never updated, so it stays the *first*
token's latency rather than drifting into an average. Its doc comment had come
adrift onto `cacheUsed` — which meant the generated `.d.ts` documented the KV
cache position count as "Time to first token" — so the getter it described was
written to match the comment.

**The per-op timings** are asserted to be zero without a clock and to put matmul
as the largest bucket with one. That second assertion caught a real bug: the FFN
timing block sat *outside* the layer loop, so gate/up/down — the largest matmuls
in the model — were being charged to "elementwise", which reported 45 ms against
matmul's 15 ms. A perf panel that confidently points at the wrong thing is worse
than no perf panel.

**Sampling** has 17 tests that need no model at all: the RNG's reproducibility
and uniformity, each filter collapsing to greedy at its degenerate setting
(`top_k = 1`, `top_p -> 0`, `temperature = 0`), a hand-computed nucleus cut on
logits chosen to give probabilities of exactly 0.5/0.25/0.125/0.125, and an
empirical check that 60000 draws match `softmax(logits)` to within 1%.

**Golden generation** is the cheapest regression guard here: a fixed prompt,
greedy decoding, and an exact 16-token sequence. Any change that perturbs the
numerics moves those ids and says so in one line, instead of leaving you to
notice the prose got slightly worse. A companion test asserts a seeded sampled
run reproduces exactly and that a different seed diverges.

A weaker cross-check also runs: `tools/make_tiny_gguf.py` is an independent
GGUF *writer* in Python. Python writes, Rust reads, the config round-trips.

## Traps addressed so far

- **Q4_K's 6-bit packed scales.** Eight scales and eight mins at 6 bits each is
  96 bits in a 96-bit container, so the high group is assembled from a nibble in
  bytes 8–11 plus two bits scavenged from the spare bits of bytes 0–7. Tested in
  isolation before anything used it, and now **verified bit-exact against ggml**
  on real blocks. Closed.
- **Q5_0's fifth bit.** Lives in a separate 32-bit mask where bit `i` belongs to
  element `i`, running straight through 0..32 — it does *not* follow the
  low-half/high-half split that the low four bits use. Pinned by a test that sets
  bits 0 and 16 and checks exactly elements 0 and 16 move.
- **Q4_0/Q4_K nibble order.** ggml does *not* store elements 0,1 in byte 0. Low
  nibbles are the first half of the block, high nibbles the second. Reading it
  the obvious way yields a permutation — every weight present, all misplaced.
- **f16 subnormals.** Renormalised properly, not flushed to zero.
- **Q6_K signed scales.** `i8`, not `u8`; a dedicated test would catch the
  difference between `+32` and `-8160`.
- **GQA head mapping.** `head_count % head_count_kv != 0` is rejected at config
  time rather than producing plausible garbage later.
- **ggml dimension order.** `dims[0]` is the *contiguous* axis, the reverse of
  how torch prints shapes. `shape_row_major()` exists for comparing against
  reference tensors, and `QuantMatrix` has a test asserting a GGUF `[64, 3]`
  becomes 3 rows of 64 rather than the transpose.
- **RMSNorm is not LayerNorm.** No mean subtraction, no bias. Pinned by a test
  that feeds in a constant vector: LayerNorm maps it to zeros, RMSNorm returns it
  unchanged.
- **Softmax overflow.** Attention scores reach the tens and `exp(89)` is inf,
  giving inf/inf = NaN that ends the generation. Max-subtracted, with a test at
  scores of ±1000 and one for a fully-masked row.
- **RoPE pairing convention.** See below — the one trap not yet fully closed.
- **`\p{N}` matches one digit for Qwen2**, not the `\p{N}{1,3}` of GPT-4 and
  Llama 3. `"2024"` is four tokens, not two.
- **`\s+(?!\S)` is what attaches a space to a word.** Greedy `\s+` takes the
  whole run, the lookahead fails, and backtracking gives back exactly one
  character — which is why `" hello"` is one pre-token but `"  hello"` is two.
- **Special tokens are a security boundary.** `EncodeOptions::parse_special` is
  off by default. With it on, a user who types `<|im_end|>` into a message can
  close the assistant's turn and impersonate the system. Chat *scaffolding* is
  encoded with it on; untrusted message bodies are not. Pinned by a test.

## What the real model file taught us

Downloading Qwen2.5-0.5B-Instruct-Q4_K_M (469 MiB) and pointing the parser at it
produced four findings that no synthetic fixture would have:

1. **Every config value matched.** 24 layers, d_model 896, ffn 4864, 14 heads /
   2 KV heads, eps 1e-6, rope base 1e6, ctx 32768, vocab 151936 — exactly what
   the step-1 test predicted.
2. **Q5_0 is 55% of the model** and was not in the format list. See above.
   Implemented, and verified bit-exact.
3. **Qwen2 has QKV biases.** 72 bias tensors — `attn_q.bias`, `attn_k.bias`,
   `attn_v.bias`, one set per layer. Llama has none, and the spec's attention
   description does not mention them. The forward pass in step 5 must add them
   to the Q, K and V projections (but not to `attn_output`, which has no bias).
4. **Two metadata assumptions were wrong.** The real file has *no*
   `rope.dimension_count` (so all 64 head dimensions rotate, via the default),
   and it *does* set `bos_token_id` = 151643 with `add_bos_token = false`. The
   synthetic fixture now mirrors the real file rather than my guess at it.


## The RoPE convention — closed

RoPE rotates each head in `head_dim/2` planes, and which components pair up has
two conventions in the wild, both self-consistent, both norm-preserving, both
producing fluent text:

```
split-half ("NeoX"):  (x[i], x[i + d/2])
adjacent   ("NORM"):  (x[2i], x[2i + 1])
```

All three legs are now verified:

1. **HuggingFace Qwen2 uses split-half** — read from `rotate_half` in the
   installed transformers source.
2. **Our `SplitHalf` matches HuggingFace** — numerically, against
   `apply_rotary_pos_emb` itself, to 6.5e-6 relative.
3. **GGUF weights want split-half too** — so the converter does *not* permute
   Q/K for Qwen2 the way it does for Llama. Measured after layer 0 on real
   weights:

   | convention | worst relative error |
   |------------|---------------------|
   | `SplitHalf` | 7.4e-4 |
   | `Adjacent` | 5.3e+1 |

A factor of 71,000. Visible end to end, too: `--rope-adjacent` turns
`" Paris. It is the largest city in Europe"` into `" 100000000"` — which is
exactly the failure mode to fear, since it is only obvious once you look.

`RopeKind` stays an enum: the next model may want the other one, and there is
still no numerical property that distinguishes them without a reference.

## Decisions taken (and why)

- **Byte-slice kernels, not `#[repr(C)]` structs.** Blocks are parsed from `&[u8]`
  rather than transmuted. No `unsafe`, no alignment assumptions, and it stays
  correct when the fused kernels start reading blocks straight out of the mapped
  file at arbitrary offsets. The whole crate is `#![deny(unsafe_code)]` until
  SIMD needs it in step 9.
- **Eager metadata, borrowed strings.** Token strings are `&'a str` into the file
  buffer (no copies); numeric arrays are `Vec` because a misaligned `&[f32]` is
  UB. For Qwen2.5-0.5B that's ~6 MB against a ~400 MB model.
- **`no_std` from the start.** Paid off twice: it surfaced the missing float math
  at step 2 rather than at wasm-build time, and the replacement helper I reached
  for turned out to be subtly wrong (see below), which a test caught because the
  module existed to be tested.
- **`roundf`, not the add-half-and-truncate trick.** `(x + 0.5) as i32` is the
  classic fast rounding idiom and it is wrong: for `0.49999997` the addition
  crosses a binade, lands exactly halfway between two representable f32s, and
  ties-to-even rounds it up to `1.0`. That would have quantised every near-zero
  activation in a Q8_0 block to +/-1. Pinned by tests in both `math` and `quant`.
- **Validate tensor ranges once, at parse.** `tensor_data()` is then infallible
  and the hot path never re-checks.

## Decisions taken for step 3

**Float math: `libm`, unconditionally, on every target.**

`core` has no `sqrt`, `exp` or `sin` — they live in `std` because they lower to
libm calls. `libm` is rust-lang's port of MUSL's, and it is the same code
`compiler_builtins` already vendors for wasm targets, so it is what we would link
regardless. It is `core`'s only dependency.

The non-obvious half is using it *even when `std` is available*, rather than
falling back to the host intrinsics. Host libm and Rust's libm agree to within an
ulp, not exactly — and an ulp is enough to decide a near-tie in an argmax or a
top-p cut. Two promises depend on that not happening: the golden generation test
only validates the wasm build if wasm does the same arithmetic as the native
build it was checked against, and a shared seed in a URL has to reproduce the
same tokens for whoever opens it. One implementation everywhere, no `cfg`.

The cost is a software `sqrt` instead of the hardware instruction. rmsnorm calls
it 48 times per token against millions of cycles of matmul — under 0.1%. If step
9's profiles disagree, that is the moment to revisit, with a benchmark in hand.

**Tensor storage: enum dispatch at the boundary, hand-written kernels per format.**

I recommended generics initially and was comparing the wrong two things. The
dispatch has to happen *somewhere* — the quantisation is a property of the file,
not of the source — so the question was never "branch or no branch". It is only
where the branch sits, and in both designs it hoists out of the loop:

```rust
match weights {                      // once per matmul, not per element
    QuantTensor::Q4_K(d) => matvec_q4_k(d, x, out),
    QuantTensor::Q8_0(d) => matvec_q8_0(d, x, out),
    ...
}
```

The inner loop is identical either way. What differs is the code you write around
it, and the shared code a `QuantFormat` trait would buy is largely fictional: the
Q4_K kernel and the Q8_0 kernel differ in block size, packing, unpacking cost and
SIMD strategy. A trait would end up as `#[inline(always)]` methods that every
implementor specialises anyway — the abstraction's shape without its benefit.

So: enum at the layer boundary, monomorphic kernels underneath. Same hot loop,
bounded code size and compile times, and it matches how the formats actually
differ. This is also what llama.cpp settled on, for the same reason.

## Next

Everything planned is built. What is left:

- **The first deploy.** The workflow has never run — it needs `git init`, a
  remote, and Pages set to "GitHub Actions". CI syntax that has not executed is a
  guess however carefully written, so expect a round of fixes.
- **A Web Worker.** A backgrounded tab throttles `setTimeout` and generation
  drops to ~2.6 tok/s. Workers are not throttled the same way, and it would also
  move the 469 MB parse off the main thread.
- **Batched prefill on native.** It is 2x slower there and disabled by default.
  Worth a look with a profiler rather than more guessing — the two hypotheses
  tried (inlining, cache-sized chunks) were both wrong.
- **Threading, if it ever becomes possible.** Single-threaded by design because
  GitHub Pages cannot set COOP/COEP. The KV cache and workspace are already
  per-generation state rather than globals, so the shape is there.

## Constraints (unchanged)

No ML/inference crates. No `ndarray`/`nalgebra` in the compute path. Allowed:
`wasm-bindgen`, `js-sys`, `web-sys`, `serde`/`serde_json` (config only),
`getrandom`, `console_error_panic_hook`; dev-only `criterion`, `proptest`.
Target `wasm32-unknown-unknown` + SIMD128, single-threaded, designed so threading
can be added without a rewrite. Model weights are never committed.
