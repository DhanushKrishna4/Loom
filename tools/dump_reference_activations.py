#!/usr/bin/env python3
"""Run the model in PyTorch and dump every intermediate activation.

This is the layer-by-layer check the whole project's testing strategy rests on.
A bug in layer 3 is obvious here and effectively undiagnosable from final logits.

The reference uses **the same weights** as the Rust engine -- dequantised out of
the same GGUF with gguf-py -- rather than the HuggingFace bf16 checkpoint. That
makes the comparison exact rather than "within quantisation noise": any
difference is our arithmetic, not the quantiser's. RoPE and the GQA head
expansion come from transformers' own `apply_rotary_pos_emb` and `repeat_kv`,
so the two things most likely to be wrong are not re-derived here.

Weights are dequantised one layer at a time and freed, so peak memory stays
around 200 MB instead of the 2 GB an f32 copy of the whole model would need.

    pip install gguf numpy torch transformers
    python3 tools/dump_reference_activations.py models/....gguf

Writes tools/reference/activations/ (gitignored).
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType
from gguf.quants import dequantize

OUT = Path(__file__).resolve().parent / "reference" / "activations"

# Must match tools/dump_reference_tokens.py and the Rust test.
PROMPT_TOKENS = [785, 6722, 315, 9625, 374]  # "The capital of France is"


def load_reader(path: Path):
    r = GGUFReader(path)
    by_name = {t.name: t for t in r.tensors}
    meta = {}
    for k, f in r.fields.items():
        try:
            meta[k] = f.contents()
        except Exception:
            pass
    return r, by_name, meta


def deq(tensor) -> np.ndarray:
    """Dequantise a GGUF tensor to a float32 array of shape (rows, cols)."""
    qtype = GGMLQuantizationType(tensor.tensor_type)
    raw = np.asarray(tensor.data).reshape(-1)
    flat = np.asarray(dequantize(raw, qtype), dtype=np.float32).reshape(-1)
    # GGUF shape is [cols, rows, ...] -- fastest axis first.
    dims = [int(d) for d in tensor.shape if int(d) > 0]
    cols = dims[0]
    rows = int(np.prod(dims[1:])) if len(dims) > 1 else 1
    return flat[: rows * cols].reshape(rows, cols)


def t(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def rms_norm(x, w, eps):
    # Qwen2RMSNorm, verbatim: variance in f32, rsqrt, then scale.
    var = x.pow(2).mean(-1, keepdim=True)
    return w * (x * torch.rsqrt(var + eps))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--layers", type=int, default=None, help="stop after N layers")
    args = ap.parse_args()

    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    OUT.mkdir(parents=True, exist_ok=True)
    reader, tensors, meta = load_reader(args.model)

    def m(key, default=None):
        v = meta.get(key, default)
        if isinstance(v, (list, tuple)) and len(v) == 1:
            v = v[0]
        return v

    n_layers = int(m("qwen2.block_count"))
    d_model = int(m("qwen2.embedding_length"))
    n_heads = int(m("qwen2.attention.head_count"))
    n_kv = int(m("qwen2.attention.head_count_kv"))
    eps = float(m("qwen2.attention.layer_norm_rms_epsilon"))
    base = float(m("qwen2.rope.freq_base"))
    head_dim = d_model // n_heads
    n_rep = n_heads // n_kv
    if args.layers:
        n_layers = min(n_layers, args.layers)

    print(f"qwen2: {n_layers} layers, d={d_model}, heads={n_heads}/{n_kv}, "
          f"head_dim={head_dim}, eps={eps}, rope_base={base}")

    ids = PROMPT_TOKENS
    T = len(ids)

    # --- embedding: only the rows we need -----------------------------------
    embd = tensors["token_embd.weight"]
    emb_full = deq(embd)
    h = t(emb_full[ids]).unsqueeze(0)  # [1, T, d]
    del emb_full
    gc.collect()
    np.save(OUT / "hidden.0.npy", h[0].numpy())

    # --- rope tables, built the HuggingFace way ------------------------------
    pos = torch.arange(T, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)   # the duplication rotate_half expects
    cos, sin = emb.cos()[None], emb.sin()[None]

    causal = torch.full((T, T), float("-inf")).triu(1)[None, None]

    for li in range(n_layers):
        p = f"blk.{li}."
        w = {k: t(deq(tensors[p + k])) for k in [
            "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
            "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
        ]}
        nrm = {k: t(deq(tensors[p + k]).reshape(-1)) for k in ["attn_norm.weight", "ffn_norm.weight"]}
        bias = {}
        for k in ["attn_q.bias", "attn_k.bias", "attn_v.bias"]:
            if p + k in tensors:
                bias[k] = t(deq(tensors[p + k]).reshape(-1))

        residual = h
        x = rms_norm(h, nrm["attn_norm.weight"], eps)
        np.save(OUT / f"l{li}.attn_norm.npy", x[0].numpy())

        q = x @ w["attn_q.weight"].T
        k = x @ w["attn_k.weight"].T
        v = x @ w["attn_v.weight"].T
        # Qwen2's attention biases. Llama has none; omitting them here would
        # make this "reference" agree with a wrong implementation.
        if "attn_q.bias" in bias:
            q = q + bias["attn_q.bias"]
            k = k + bias["attn_k.bias"]
            v = v + bias["attn_v.bias"]

        q = q.view(1, T, n_heads, head_dim).transpose(1, 2)
        k = k.view(1, T, n_kv, head_dim).transpose(1, 2)
        v = v.view(1, T, n_kv, head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)   # transformers' own
        k = repeat_kv(k, n_rep)                        # transformers' own GQA map
        v = repeat_kv(v, n_rep)

        scores = (q @ k.transpose(-1, -2)) / (head_dim ** 0.5) + causal
        probs = torch.softmax(scores, dim=-1)
        o = (probs @ v).transpose(1, 2).reshape(1, T, n_heads * head_dim)
        o = o @ w["attn_output.weight"].T              # no bias on the output proj
        h = residual + o

        residual = h
        x = rms_norm(h, nrm["ffn_norm.weight"], eps)
        np.save(OUT / f"l{li}.ffn_norm.npy", x[0].numpy())

        gate = x @ w["ffn_gate.weight"].T
        up = x @ w["ffn_up.weight"].T
        h = residual + (torch.nn.functional.silu(gate) * up) @ w["ffn_down.weight"].T

        np.save(OUT / f"hidden.{li+1}.npy", h[0].numpy())
        print(f"  layer {li:>2}  |h| rms {h.pow(2).mean().sqrt():.4f}")

        del w, nrm, bias
        gc.collect()

    # --- final norm and logits ----------------------------------------------
    final = rms_norm(h, t(deq(tensors["output_norm.weight"]).reshape(-1)), eps)
    np.save(OUT / "final_norm.npy", final[0].numpy())

    # Unembedding in chunks: 151936 x 896 as f32 is 544 MB in one piece.
    out_t = tensors["output.weight"]
    W = deq(out_t)
    logits = (final[0] @ t(W).T).numpy()
    del W
    gc.collect()
    np.save(OUT / "logits.npy", logits)

    last = logits[-1]
    top = np.argsort(-last)[:5]
    print(f"\ntop-5 next tokens for position {T-1}: "
          + ", ".join(f"{i}({last[i]:.3f})" for i in top))
    print(f"wrote {len(list(OUT.glob('*.npy')))} activation files to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
