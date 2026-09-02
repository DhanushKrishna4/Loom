//! `nano-infer` -- the native driver.
//!
//! This binary exists so that no numerical bug ever has to be diagnosed through
//! a browser. Everything the wasm build does, this does first, against the same
//! `nano-infer-core`.
//!
//! Argument parsing is hand-rolled: the dependency list for this project is
//! deliberately tiny, and `clap` would be the largest thing in it.

use std::process::ExitCode;

use nano_infer_core::gguf::{Array, Gguf, TensorInfo, Value};
use nano_infer_core::model::{KvCache, KvLayout, Model, ModelWeights, Workspace};
use nano_infer_core::ops::RopeKind;
use nano_infer_core::quant;
use nano_infer_core::sample::{Sampler, SamplerConfig};
use nano_infer_core::tokenizer::{apply_chat_template, ChatMessage, EncodeOptions, Tokenizer};
use nano_infer_core::GgmlType;

const USAGE: &str = "\
nano-infer -- native driver for the nano-infer engine

USAGE:
    nano-infer <COMMAND> [ARGS]

COMMANDS:
    gguf-dump <file.gguf> [--tensors] [--metadata] [--all]
        Parse a GGUF file and print its header, model config and a summary of
        its tensors. Use this to check a real model against the parser.

          --tensors    list every tensor (default: a per-type summary only)
          --metadata   print every metadata key (default: config-relevant only)
          --all        both of the above

    tokenize <file.gguf> <text> [--special] [--chat]
        Encode text and print the tokens with their ids and strings, then
        verify the round trip.

          --special   recognise <|im_start|> etc. in the input
          --chat      wrap the text in Qwen2's chat template first

    generate <file.gguf> <prompt> [--n N] [--chat] [--raw] [--rope-adjacent]
        Greedy generation. This is the correctness path, not a fast one: it
        dequantises each weight row on the fly. Build with --release.

          --n N            tokens to generate (default 32)
          --chat           wrap the prompt in Qwen2's chat template
          --raw            no chat template (default)
          --rope-adjacent  use the adjacent RoPE pairing instead of split-half
          --unfused        use the dequantise-then-dot reference path
          --batched-prefill  unpack each weight row once per chunk (wasm default)
          --temp T         temperature (0 = greedy, the default)
          --top-k K        keep the K highest-scoring tokens
          --top-p P        nucleus threshold
          --repeat-penalty R   penalise recently-seen tokens
          --repeat-window N    how far back the penalty looks (default 64)
          --seed S         RNG seed, so a generation can be reproduced
          --ids            also print the raw token ids

    bench-kv <file.gguf> [--seq N] [--n N]
        Benchmark the two KV cache layouts against each other, end to end.
        This is the measurement the layout choice is based on.

    dequant <file.gguf> <tensor-name> [--blocks N]
        Dequantise the first N blocks (default 4) of a tensor and print the
        values plus summary statistics. Sanity-checks a real quantised tensor.

    help
        Print this message.
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("help");

    let result = match cmd {
        "gguf-dump" => cmd_gguf_dump(&args[1..]),
        "tokenize" => cmd_tokenize(&args[1..]),
        "generate" => cmd_generate(&args[1..]),
        "bench-kv" => cmd_bench_kv(&args[1..]),
        "dequant" => cmd_dequant(&args[1..]),
        "help" | "-h" | "--help" => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        other => Err(format!("unknown command {other:?}\n\n{USAGE}")),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

// ------------------------------------------------------------ gguf-dump ----

fn cmd_gguf_dump(args: &[String]) -> Result<(), String> {
    let path = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .ok_or("gguf-dump needs a path to a .gguf file")?;
    let all = args.iter().any(|a| a == "--all");
    let show_tensors = all || args.iter().any(|a| a == "--tensors");
    let show_metadata = all || args.iter().any(|a| a == "--metadata");

    let bytes = std::fs::read(path).map_err(|e| format!("reading {path}: {e}"))?;
    let g = Gguf::parse(&bytes).map_err(|e| format!("parsing {path}: {e}"))?;

    println!("file                {path}");
    println!("size                {}", human_bytes(bytes.len() as u64));
    println!("gguf version        {}", g.version);
    println!("alignment           {}", g.alignment);
    println!("metadata keys       {}", g.metadata.len());
    println!("tensors             {}", g.tensors.len());
    println!(
        "data section        offset {} ({}), {} of tensor data",
        g.tensor_data_offset,
        human_bytes(g.tensor_data_offset as u64),
        human_bytes(g.tensor_bytes())
    );

    println!("\n--- model config ---");
    match g.config() {
        Ok(c) => {
            let row = |k: &str, v: String| println!("  {k:<24}{v}");
            row("architecture", c.architecture.clone());
            row("name", c.name.clone().unwrap_or_else(|| "-".into()));
            row("block_count", c.block_count.to_string());
            row("embedding_length", c.embedding_length.to_string());
            row("feed_forward_length", c.feed_forward_length.to_string());
            row("head_count", c.head_count.to_string());
            row("head_count_kv", c.head_count_kv.to_string());
            row(
                "head_dim",
                format!("{}  (kv group size {})", c.head_dim, c.kv_group_size()),
            );
            row("context_length", c.context_length.to_string());
            row("rms_norm_eps", format!("{:e}", c.rms_norm_eps));
            row("rope_freq_base", format!("{}", c.rope_freq_base));
            row(
                "rope_dim_count",
                c.rope_dimension_count
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| format!("- (all {} dims)", c.head_dim)),
            );
            row("vocab_size", c.vocab_size.to_string());
            row(
                "tokenizer",
                c.tokenizer_model.clone().unwrap_or_else(|| "-".into()),
            );
            row(
                "special tokens",
                format!(
                    "bos {:?}  eos {:?}  pad {:?}  unk {:?}",
                    c.bos_token_id, c.eos_token_id, c.pad_token_id, c.unk_token_id
                ),
            );
            row("q_dim / kv_dim", format!("{} / {}", c.q_dim(), c.kv_dim()));
            println!("\n  f32 KV cache size:");
            for n in [512usize, 2048, 4096, 8192] {
                if n <= c.context_length {
                    println!(
                        "    {n:>6} positions   {}",
                        human_bytes(c.kv_cache_bytes(n) as u64)
                    );
                }
            }
            println!(
                "    {:>6} positions   {}   <- full context",
                c.context_length,
                human_bytes(c.kv_cache_bytes(c.context_length) as u64)
            );
        }
        Err(e) => println!("  <could not extract: {e}>"),
    }

    println!("\n--- tensor types ---");
    let mut kinds: Vec<(GgmlType, usize, u64, u64)> = Vec::new();
    for t in &g.tensors {
        match kinds.iter_mut().find(|k| k.0 == t.ggml_type) {
            Some(k) => {
                k.1 += 1;
                k.2 += t.byte_size;
                k.3 += t.n_elements();
            }
            None => kinds.push((t.ggml_type, 1, t.byte_size, t.n_elements())),
        }
    }
    kinds.sort_by_key(|k| std::cmp::Reverse(k.2));
    let total = g.tensor_bytes().max(1);
    for (ty, count, bytes_, elems) in &kinds {
        println!(
            "  {:<8} {:>4} tensors  {:>12}  {:>5.1}%  {:.2} bits/weight{}",
            ty.name(),
            count,
            human_bytes(*bytes_),
            100.0 * *bytes_ as f64 / total as f64,
            *bytes_ as f64 * 8.0 / *elems as f64,
            if quant::is_supported(*ty) {
                ""
            } else {
                "   [dequant NOT implemented]"
            },
        );
    }

    if show_metadata {
        println!("\n--- metadata ---");
        for (k, v) in &g.metadata {
            println!("  {k:<44}{}", summarize(v));
        }
    }

    if show_tensors {
        println!("\n--- tensors ---");
        for t in &g.tensors {
            println!(
                "  {:<40} {:<7} {:<22} {:>12}  @ {}",
                t.name,
                t.ggml_type.name(),
                format!("{:?}", t.shape_row_major()),
                human_bytes(t.byte_size),
                t.offset
            );
        }
    }

    Ok(())
}

// -------------------------------------------------------------- tokenize ----

fn cmd_tokenize(args: &[String]) -> Result<(), String> {
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let path = positional.first().ok_or("tokenize needs a .gguf path")?;
    let text = positional.get(1).ok_or("tokenize needs some text")?;
    let parse_special = args.iter().any(|a| a == "--special") || args.iter().any(|a| a == "--chat");
    let chat = args.iter().any(|a| a == "--chat");

    let bytes = std::fs::read(path.as_str()).map_err(|e| format!("reading {path}: {e}"))?;
    let g = Gguf::parse(&bytes).map_err(|e| format!("parsing {path}: {e}"))?;
    let tk = Tokenizer::from_gguf(&g).map_err(|e| e.to_string())?;

    let rendered;
    let input: &str = if chat {
        rendered = apply_chat_template(&[ChatMessage::user(text)], true);
        &rendered
    } else {
        text
    };

    println!(
        "vocab {} tokens, {} merges",
        tk.vocab().len(),
        tk.vocab().n_merges()
    );
    if chat {
        println!("\n--- rendered template ---\n{input}");
    }

    let ids = tk.encode(
        input,
        EncodeOptions {
            parse_special,
            add_bos: false,
        },
    );

    println!("\n{} tokens:", ids.len());
    for (i, id) in ids.iter().enumerate() {
        // Show the decoded bytes, not the byte-mapped form, so spaces and
        // newlines read as themselves rather than as U+0120 / U+010A.
        let piece = tk.decode(&[*id], false);
        println!("  {i:>4}  {id:>6}  {piece:?}");
    }

    let back = tk.decode(&ids, false);
    println!(
        "\nround trip: {}",
        if back == input { "exact" } else { "MISMATCH" }
    );
    if back != input {
        println!("  got  {back:?}");
        println!("  want {input:?}");
        return Err("tokenizer round trip failed".into());
    }
    Ok(())
}

// -------------------------------------------------------------- generate ----

fn cmd_generate(args: &[String]) -> Result<(), String> {
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let path = positional.first().ok_or("generate needs a .gguf path")?;
    let prompt = positional.get(1).ok_or("generate needs a prompt")?;
    let n_gen: usize = match args.iter().position(|a| a == "--n") {
        Some(i) => args
            .get(i + 1)
            .and_then(|v| v.parse().ok())
            .ok_or("--n needs a number")?,
        None => 32,
    };
    let chat = args.iter().any(|a| a == "--chat");
    let fused = !args.iter().any(|a| a == "--unfused");
    let fnum = |flag: &str, default: f32| -> f32 {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let unum = |flag: &str, default: u64| -> u64 {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let sampling = SamplerConfig {
        temperature: fnum("--temp", 0.0),
        top_k: unum("--top-k", 0) as usize,
        top_p: fnum("--top-p", 1.0),
        repetition_penalty: fnum("--repeat-penalty", 1.0),
        repetition_window: unum("--repeat-window", 64) as usize,
        seed: unum("--seed", 0),
    };
    let rope_kind = if args.iter().any(|a| a == "--rope-adjacent") {
        RopeKind::Adjacent
    } else {
        RopeKind::SplitHalf
    };

    let t0 = std::time::Instant::now();
    let bytes = std::fs::read(path.as_str()).map_err(|e| format!("reading {path}: {e}"))?;
    let g = Gguf::parse(&bytes).map_err(|e| format!("parsing {path}: {e}"))?;
    let tk = Tokenizer::from_gguf(&g).map_err(|e| e.to_string())?;
    let weights = ModelWeights::from_gguf(&g).map_err(|e| e.to_string())?;

    let cfg = weights.config.clone();
    println!(
        "{} | {} layers, d={} | {} weights bytes | attn bias: {}",
        cfg.architecture,
        cfg.block_count,
        cfg.embedding_length,
        human_bytes(weights.byte_len() as u64),
        if weights.has_attn_bias() { "yes" } else { "no" },
    );
    print!("formats:");
    for (t, n) in weights.formats() {
        print!(" {}x{}", n, t.name());
    }
    println!(
        "\nrope: {rope_kind:?} | matmul: {}",
        if fused {
            "fused quantised"
        } else {
            "dequantise-then-dot"
        }
    );
    if sampling.temperature <= 0.0 && sampling.repetition_penalty == 1.0 {
        println!("sampling: greedy (deterministic)");
    } else {
        println!(
            "sampling: temp {} top-k {} top-p {} repeat {}x/{} seed {}",
            sampling.temperature,
            sampling.top_k,
            sampling.top_p,
            sampling.repetition_penalty,
            sampling.repetition_window,
            sampling.seed
        );
    }

    let rendered;
    let input: &str = if chat {
        rendered = apply_chat_template(&[ChatMessage::user(prompt)], true);
        &rendered
    } else {
        prompt
    };
    let tokens = tk.encode(
        input,
        EncodeOptions {
            parse_special: chat,
            add_bos: false,
        },
    );

    let max_seq = (tokens.len() + n_gen + 8).next_power_of_two().max(64);
    let mut model = Model::new(weights, max_seq, rope_kind);
    model.set_fused(fused);
    model.set_batched_prefill(args.iter().any(|a| a == "--batched-prefill"));
    let mut ws = Workspace::new(&cfg, max_seq);
    let mut cache = KvCache::new(&cfg, max_seq);
    println!(
        "kv cache {} | workspace {} | load {:.1}s",
        human_bytes(cache.byte_len() as u64),
        human_bytes(ws.byte_len() as u64),
        t0.elapsed().as_secs_f64(),
    );

    println!("\n--- prompt ({} tokens) ---\n{input}", tokens.len());
    println!("\n--- generated ---");

    let t1 = std::time::Instant::now();
    // Batched: one weight-row unpack serves a whole chunk of positions.
    model
        .prefill(&tokens, &mut ws, &mut cache)
        .map_err(|e| e.to_string())?;
    let prefill = t1.elapsed();

    let mut sampler = Sampler::new(sampling, cfg.vocab_size);
    // Seed the repetition penalty with the prompt: a token the user just wrote
    // is exactly as "recently seen" as one the model just produced.
    for &t in &tokens {
        sampler.accept(t);
    }

    let mut out_ids = Vec::new();
    let t2 = std::time::Instant::now();
    let mut next = sampler.sample(model.logits(&ws));
    for _ in 0..n_gen {
        if Some(next) == cfg.eos_token_id {
            break;
        }
        out_ids.push(next);
        sampler.accept(next);
        if cache.len() >= max_seq {
            break;
        }
        model
            .forward_token(next, cache.len(), &mut ws, &mut cache, None)
            .map_err(|e| e.to_string())?;
        next = sampler.sample(model.logits(&ws));
    }
    let decode = t2.elapsed();

    println!("{}", tk.decode(&out_ids, true));
    if args.iter().any(|a| a == "--ids") {
        println!("\nprompt ids:    {tokens:?}");
        println!("generated ids: {out_ids:?}");
    }
    println!(
        "\nprefill {} tok in {:.2}s ({:.2} tok/s) | decode {} tok in {:.2}s ({:.2} tok/s)",
        tokens.len(),
        prefill.as_secs_f64(),
        tokens.len() as f64 / prefill.as_secs_f64(),
        out_ids.len(),
        decode.as_secs_f64(),
        out_ids.len() as f64 / decode.as_secs_f64().max(1e-9),
    );
    Ok(())
}

// -------------------------------------------------------------- bench-kv ----

/// Time real decode steps under each KV layout.
///
/// End to end rather than a microbenchmark of the gather: the question is which
/// layout makes the *engine* faster, and a microbenchmark would happily report a
/// win on a loop that is 3% of the runtime.
fn cmd_bench_kv(args: &[String]) -> Result<(), String> {
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let path = positional.first().ok_or("bench-kv needs a .gguf path")?;
    let num = |flag: &str, default: usize| -> usize {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let n_steps = num("--n", 24);

    let bytes = std::fs::read(path.as_str()).map_err(|e| format!("reading {path}: {e}"))?;
    let g = Gguf::parse(&bytes).map_err(|e| format!("parsing {path}: {e}"))?;
    let weights = ModelWeights::from_gguf(&g).map_err(|e| e.to_string())?;
    let cfg = weights.config.clone();

    // Prefill lengths chosen to show whether the gap grows with history, which
    // is the whole question: reads scale with sequence length, writes do not.
    let seq_points = [64usize, 256, 1024];
    let max_seq = seq_points.iter().max().copied().unwrap() + n_steps + 8;
    let model = Model::new(weights, max_seq, RopeKind::SplitHalf);

    println!(
        "kv cache {} at max_seq {} | {} decode steps per point\n",
        human_bytes(KvCache::new(&cfg, max_seq).byte_len() as u64),
        max_seq,
        n_steps
    );
    println!(
        "{:>8}  {:>16}  {:>16}  {:>8}",
        "history", "PositionMajor", "HeadMajor", "delta"
    );

    let micro_only = args.iter().any(|a| a == "--micro");
    for &seq in seq_points.iter().filter(|_| !micro_only) {
        let mut timings = Vec::new();
        for layout in [KvLayout::PositionMajor, KvLayout::HeadMajor] {
            let mut ws = Workspace::new(&cfg, max_seq);
            let mut cache = KvCache::with_layout(&cfg, max_seq, layout);

            // Fill the history with a fixed token so both layouts see identical
            // work, then time only the decode steps that follow.
            for p in 0..seq {
                model
                    .forward_token(9707, p, &mut ws, &mut cache, None)
                    .map_err(|e| e.to_string())?;
            }
            let t0 = std::time::Instant::now();
            for i in 0..n_steps {
                model
                    .forward_token(9707, seq + i, &mut ws, &mut cache, None)
                    .map_err(|e| e.to_string())?;
            }
            timings.push(t0.elapsed().as_secs_f64() / n_steps as f64);
        }
        let (pm, hm) = (timings[0], timings[1]);
        println!(
            "{seq:>8}  {:>13.2} ms  {:>13.2} ms  {:>7.1}%",
            pm * 1e3,
            hm * 1e3,
            (pm - hm) / pm * 100.0
        );
    }
    println!("\n(delta > 0 means HeadMajor is faster)");

    // The end-to-end numbers above are dominated by the weight matmuls, so a
    // real layout difference can hide inside their noise. This isolates the
    // attention gather itself: same loops, same data, nothing else.
    println!("\n--- attention gather only, one layer, all heads ---");
    println!(
        "{:>8}  {:>16}  {:>16}  {:>8}",
        "history", "PositionMajor", "HeadMajor", "delta"
    );

    let head_dim = cfg.head_dim;
    let n_heads = cfg.head_count;
    let group = cfg.kv_group_size();
    for &seq in &[256usize, 1024, 4096] {
        let mut timings = Vec::new();
        for layout in [KvLayout::PositionMajor, KvLayout::HeadMajor] {
            // One layer's worth is enough and keeps the allocation sane.
            let mut one = cfg.clone();
            one.block_count = 1;
            let mut cache = KvCache::with_layout(&one, seq, layout);
            let k = vec![0.5f32; cfg.kv_dim()];
            for p in 0..seq {
                cache.fill_for_bench(0, p, &k, &k);
            }
            let q = vec![0.25f32; cfg.q_dim()];
            let mut att = vec![0.0f32; seq];
            let mut out = vec![0.0f32; head_dim];

            let reps = (2_000_000 / seq).max(4);
            // Min over trials, not mean: we are after the machine's best case,
            // and a single slow trial is the scheduler's fault rather than the
            // layout's.
            let mut best = f64::INFINITY;
            let mut sink = 0.0f32;
            for _trial in 0..5 {
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    for h in 0..n_heads {
                        let kv_head = h / group;
                        let qh = &q[h * head_dim..(h + 1) * head_dim];
                        let (kbuf, kbase, kstride) = cache.k_span(0, kv_head);
                        for (t, a) in att.iter_mut().enumerate() {
                            let o = kbase + t * kstride;
                            *a = nano_infer_core::ops::dot(qh, &kbuf[o..o + head_dim]);
                        }
                        out.fill(0.0);
                        let (vbuf, vbase, vstride) = cache.v_span(0, kv_head);
                        for (t, &pw) in att.iter().enumerate() {
                            let o = vbase + t * vstride;
                            for (d, vi) in out.iter_mut().zip(&vbuf[o..o + head_dim]) {
                                *d += pw * vi;
                            }
                        }
                        sink += out[0];
                    }
                }
                best = best.min(t0.elapsed().as_secs_f64() / reps as f64);
            }
            std::hint::black_box(sink);
            timings.push(best);
        }
        let (pm, hm) = (timings[0], timings[1]);
        println!(
            "{seq:>8}  {:>13.1} us  {:>13.1} us  {:>7.1}%",
            pm * 1e6,
            hm * 1e6,
            (pm - hm) / pm * 100.0
        );
    }
    Ok(())
}

// --------------------------------------------------------------- dequant ----

fn cmd_dequant(args: &[String]) -> Result<(), String> {
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let path = positional.first().ok_or("dequant needs a .gguf path")?;
    let name = positional.get(1).ok_or("dequant needs a tensor name")?;
    let n_blocks: usize = match args.iter().position(|a| a == "--blocks") {
        Some(i) => args
            .get(i + 1)
            .and_then(|v| v.parse().ok())
            .ok_or("--blocks needs a number")?,
        None => 4,
    };

    let bytes = std::fs::read(path.as_str()).map_err(|e| format!("reading {path}: {e}"))?;
    let g = Gguf::parse(&bytes).map_err(|e| format!("parsing {path}: {e}"))?;
    let t: &TensorInfo = g
        .find_tensor(name)
        .ok_or_else(|| format!("no tensor named {name:?} (try `gguf-dump --tensors`)"))?;

    let ty = t.ggml_type;
    let n_blocks = n_blocks.min((t.n_elements() / ty.block_elems() as u64) as usize);
    let n_elems = n_blocks * ty.block_elems();

    println!(
        "{}  {}  shape {:?}  {} blocks of {} elements",
        t.name,
        ty.name(),
        t.shape_row_major(),
        t.n_elements() / ty.block_elems() as u64,
        ty.block_elems()
    );

    let src = &g.tensor_data(t)[..n_blocks * ty.block_bytes()];
    let mut out = vec![0f32; n_elems];
    quant::dequantize_row(ty, src, &mut out).map_err(|e| e.to_string())?;

    for b in 0..n_blocks {
        let blk = &out[b * ty.block_elems()..(b + 1) * ty.block_elems()];
        println!("\nblock {b}:");
        for (i, chunk) in blk.chunks(8).enumerate().take(4) {
            let vals: Vec<String> = chunk.iter().map(|v| format!("{v:>10.5}")).collect();
            println!("  [{:>3}] {}", i * 8, vals.join(" "));
        }
        if blk.len() > 32 {
            println!("  ... ({} more)", blk.len() - 32);
        }
    }

    let min = out.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = out.iter().sum::<f32>() / out.len() as f32;
    let rms = (out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32).sqrt();
    let nonfinite = out.iter().filter(|v| !v.is_finite()).count();
    println!("\nstats over {n_elems} values:");
    println!("  min {min:.6}  max {max:.6}  mean {mean:.6}  rms {rms:.6}");
    println!("  non-finite: {nonfinite}");
    if nonfinite > 0 {
        return Err("dequantisation produced non-finite values".into());
    }
    Ok(())
}

// ---------------------------------------------------------------- helpers ----

/// One-line rendering of a metadata value, truncating the big vocab arrays that
/// would otherwise bury everything else.
fn summarize(v: &Value) -> String {
    match v {
        Value::Str(s) => {
            if s.len() > 60 {
                format!(
                    "{:?}... ({} bytes)",
                    &s[..s.floor_char_boundary(57)],
                    s.len()
                )
            } else {
                format!("{s:?}")
            }
        }
        Value::Array(a) => {
            let head = match a {
                Array::Str(v) => v
                    .iter()
                    .take(4)
                    .map(|s| format!("{s:?}"))
                    .collect::<Vec<_>>(),
                Array::F32(v) => v.iter().take(4).map(|x| format!("{x}")).collect(),
                Array::I32(v) => v.iter().take(4).map(|x| format!("{x}")).collect(),
                Array::U32(v) => v.iter().take(4).map(|x| format!("{x}")).collect(),
                _ => vec!["...".into()],
            };
            format!(
                "[{}; {}] {}{}",
                a.elem_type_name(),
                a.len(),
                head.join(", "),
                if a.len() > 4 { ", ..." } else { "" }
            )
        }
        Value::U8(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::F32(x) => x.to_string(),
        Value::F64(x) => x.to_string(),
        Value::Bool(x) => x.to_string(),
    }
}

fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}
