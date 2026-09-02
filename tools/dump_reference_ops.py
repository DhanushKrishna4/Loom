#!/usr/bin/env python3
"""Generate reference tensors for the f32 ops, from PyTorch / HuggingFace.

Every op in `crates/core/src/ops` is checked against these by the Rust test
`ops::tests::matches_numpy_reference_tensors`. Hand-computed values prove the
kernels match the formula as I read it; these prove they match what the actual
reference implementation computes.

The RoPE reference is the important one. It calls HuggingFace's own
`apply_rotary_pos_emb` for the split-half convention rather than re-deriving it,
because the pairing convention is precisely the thing that cannot be verified by
re-deriving it the same way twice.

    python3 tools/dump_reference_ops.py            # needs torch
    python3 tools/dump_reference_ops.py --numpy    # numpy-only fallback

Writes to tools/reference/ops/ (gitignored). Then:

    cargo test -p nano-infer-core matches_numpy_reference
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# These MUST match the constants at the top of crates/core/src/ops/tests.rs.
RMSNORM_EPS = 1e-6
ROPE_FREQ_BASE = 1_000_000.0
ROPE_HEAD_DIM = 64
ROPE_N_POS = 16

OUT = Path(__file__).resolve().parent / "reference" / "ops"


def save(name: str, arr) -> None:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    np.save(OUT / f"{name}.npy", a)
    print(f"  {name:<20} {str(a.shape):<12} {a.dtype}")


# --------------------------------------------------------------- numpy ref ----
# Computed in float64 and cast down, so the saved reference is as close to exact
# as the format allows and the Rust f32 kernel is measured against it.

def np_rmsnorm(x, w, eps):
    x = x.astype(np.float64)
    var = (x**2).mean(-1, keepdims=True)
    return (x / np.sqrt(var + eps)) * w.astype(np.float64)


def np_softmax(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(-1, keepdims=True)


def np_silu(x):
    x = x.astype(np.float64)
    # exp(-x) overflows for very negative x; x/inf = -0.0 is the correct limit,
    # so the overflow is expected rather than a problem worth warning about.
    with np.errstate(over="ignore"):
        return x / (1.0 + np.exp(-x))


def np_rope(x, positions, head_dim, base, split_half):
    """x is [n_pos, head_dim]; rotates row p by position p."""
    x = x.astype(np.float64)
    n = head_dim // 2
    i = np.arange(n, dtype=np.float64)
    inv_freq = 1.0 / (base ** (2.0 * i / head_dim))
    angles = positions.astype(np.float64)[:, None] * inv_freq[None, :]
    c, s = np.cos(angles), np.sin(angles)

    out = x.copy()
    if split_half:
        a, b = x[:, :n], x[:, n:]
        out[:, :n] = a * c - b * s
        out[:, n:] = a * s + b * c
    else:
        a, b = x[:, 0::2], x[:, 1::2]
        out[:, 0::2] = a * c - b * s
        out[:, 1::2] = a * s + b * c
    return out


# ---------------------------------------------------------------- torch ref ----

def torch_refs(data):
    """Recompute everything with torch/HF and return it, or None if unavailable."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return None

    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float64))
    out = {}

    x = t(data["rmsnorm_x"])
    w = t(data["rmsnorm_w"])
    var = x.pow(2).mean(-1, keepdim=True)
    out["rmsnorm_out"] = (x * torch.rsqrt(var + RMSNORM_EPS) * w).numpy()

    out["softmax_out"] = torch.softmax(t(data["softmax_x"]), dim=-1).numpy()
    out["silu_out"] = F.silu(t(data["silu_x"])).numpy()
    out["swiglu_out"] = (F.silu(t(data["swiglu_gate"])) * t(data["swiglu_up"])).numpy()
    out["matmul_out"] = (t(data["matmul_lhs"]) @ t(data["matmul_rhs"]).T).numpy()

    # RoPE via HuggingFace's own code path, not a re-derivation.
    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    except ImportError:
        print("  note: transformers unavailable; rope reference is numpy-only")
        return out

    hd, n_pos = ROPE_HEAD_DIM, ROPE_N_POS
    pos = torch.arange(n_pos, dtype=torch.float64)
    inv_freq = 1.0 / (ROPE_FREQ_BASE ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    freqs = torch.outer(pos, inv_freq)          # [n_pos, hd/2]
    emb = torch.cat((freqs, freqs), dim=-1)     # [n_pos, hd]  -- the duplication
    cos, sin = emb.cos()[None], emb.sin()[None] # [1, n_pos, hd]

    q = t(data["rope_x"])[None, None]           # [1, 1, n_pos, hd]
    q_embed, _ = apply_rotary_pos_emb(q, q, cos, sin, unsqueeze_dim=1)
    out["rope_split_half"] = q_embed[0, 0].numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--numpy", action="store_true", help="skip the torch cross-check")
    ap.add_argument("--seed", type=int, default=20240817)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Shapes are Qwen2.5-0.5B's where it matters (d_model 896) and small
    # elsewhere; the Rust side reads them out of the .npy headers.
    data = {
        "rmsnorm_x": rng.standard_normal((7, 896)),
        "rmsnorm_w": rng.standard_normal(896) * 0.5 + 1.0,
        "softmax_x": rng.standard_normal((5, 128)) * 8.0,
        "silu_x": np.concatenate([
            rng.standard_normal(1000) * 5.0,
            # Pin the saturating ends explicitly: these are where a naive
            # sigmoid produces inf/inf.
            np.array([-100.0, -90.0, -88.0, 0.0, 88.0, 90.0, 100.0, -1e30, 1e30,
                      -0.0, 1e-8, -1e-8, 1.0, -1.0, 0.5, 20.0, -20.0, 7.5,
                      -7.5, 3.0, -3.0, 12.0, -12.0, 0.001]),
        ]),
        "swiglu_gate": rng.standard_normal((4, 256)) * 2.0,
        "swiglu_up": rng.standard_normal((4, 256)) * 2.0,
        "matmul_lhs": rng.standard_normal((7, 64)),
        "matmul_rhs": rng.standard_normal((32, 64)),
        "rope_x": rng.standard_normal((ROPE_N_POS, ROPE_HEAD_DIM)),
    }

    positions = np.arange(ROPE_N_POS)
    computed = {
        "rmsnorm_out": np_rmsnorm(data["rmsnorm_x"], data["rmsnorm_w"], RMSNORM_EPS),
        "softmax_out": np_softmax(data["softmax_x"]),
        "silu_out": np_silu(data["silu_x"]),
        "swiglu_out": np_silu(data["swiglu_gate"]) * data["swiglu_up"].astype(np.float64),
        "matmul_out": data["matmul_lhs"].astype(np.float64) @ data["matmul_rhs"].astype(np.float64).T,
        "rope_split_half": np_rope(data["rope_x"], positions, ROPE_HEAD_DIM, ROPE_FREQ_BASE, True),
        "rope_adjacent": np_rope(data["rope_x"], positions, ROPE_HEAD_DIM, ROPE_FREQ_BASE, False),
    }

    if not args.numpy:
        tr = torch_refs(data)
        if tr is None:
            print("note: torch unavailable, using numpy references only")
        else:
            # Cross-check: if my numpy transcription disagrees with torch/HF, the
            # numpy version is the one that is wrong, and I want to know now
            # rather than after the Rust tests start failing against it.
            print("cross-checking numpy against torch/HF:")
            worst_all = 0.0
            for k, v in tr.items():
                a = np.asarray(computed[k], dtype=np.float64)
                b = np.asarray(v, dtype=np.float64)
                denom = np.maximum(np.abs(b), 1e-8)
                worst = float(np.max(np.abs(a - b) / denom))
                worst_all = max(worst_all, worst)
                flag = "ok" if worst < 1e-10 else ("close" if worst < 1e-6 else "MISMATCH")
                print(f"  {k:<20} worst rel {worst:.3e}  {flag}")
                if worst >= 1e-6:
                    print(f"\nerror: numpy and torch disagree on {k}; refusing to write "
                          f"a reference I do not trust", file=sys.stderr)
                    return 1
                # Prefer torch's value where we have one: it is the reference.
                computed[k] = v
            if worst_all < 1e-10:
                print("  -> numpy and torch/HF agree to f64 precision")

    print(f"\nwriting to {OUT}:")
    for k, v in data.items():
        save(k, v)
    for k, v in computed.items():
        save(k, v)

    print(f"\nconstants baked in (must match ops/tests.rs):")
    print(f"  RMSNORM_EPS      {RMSNORM_EPS}")
    print(f"  ROPE_FREQ_BASE   {ROPE_FREQ_BASE}")
    print(f"  ROPE_HEAD_DIM    {ROPE_HEAD_DIM}")
    print("\nnow run:  cargo test -p nano-infer-core matches_numpy_reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
