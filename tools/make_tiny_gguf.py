#!/usr/bin/env python3
"""Write a small synthetic GGUF file with Qwen2-shaped metadata.

The real model is a ~400 MB download that must never enter the repo, so this
produces a few-hundred-KB stand-in with the same architecture keys and a handful
of tensors in each supported quantisation.  It is enough to exercise
`nano-infer gguf-dump` and `nano-infer dequant` end to end.

It is also a deliberately independent writer: nothing here shares code with the
Rust parser, so if the two agree on a file, they agree about the format.

    python3 tools/make_tiny_gguf.py /tmp/tiny.gguf
    cargo run -p nano-infer-cli -- gguf-dump /tmp/tiny.gguf --all
"""

from __future__ import annotations

import random
import struct
import sys
from pathlib import Path

U32, F32, BOOL, STRING, ARRAY = 4, 6, 7, 8, 9
ALIGNMENT = 32


def s(text: str) -> bytes:
    b = text.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


class Writer:
    def __init__(self):
        self.kv = bytearray()
        self.n_kv = 0
        self.ti = bytearray()
        self.n_t = 0
        self.data = bytearray()

    def _key(self, key: str, tag: int):
        self.kv += s(key) + struct.pack("<I", tag)
        self.n_kv += 1

    def u32(self, key, v):
        self._key(key, U32); self.kv += struct.pack("<I", v); return self

    def f32(self, key, v):
        self._key(key, F32); self.kv += struct.pack("<f", v); return self

    def boolean(self, key, v):
        self._key(key, BOOL); self.kv += struct.pack("<B", 1 if v else 0); return self

    def string(self, key, v):
        self._key(key, STRING); self.kv += s(v); return self

    def str_array(self, key, vs):
        self._key(key, ARRAY)
        self.kv += struct.pack("<IQ", STRING, len(vs))
        for v in vs:
            self.kv += s(v)
        return self

    def tensor(self, name, dims, type_id, payload: bytes):
        while len(self.data) % ALIGNMENT:
            self.data += b"\x00"
        offset = len(self.data)
        self.ti += s(name) + struct.pack("<I", len(dims))
        self.ti += struct.pack(f"<{len(dims)}Q", *dims)
        self.ti += struct.pack("<IQ", type_id, offset)
        self.n_t += 1
        self.data += payload
        return self

    def build(self) -> bytes:
        out = bytearray(b"GGUF")
        out += struct.pack("<IQQ", 3, self.n_t, self.n_kv)
        out += self.kv + self.ti
        while len(out) % ALIGNMENT:
            out += b"\x00"
        return bytes(out + self.data)


def f32_to_f16_bits(x: float) -> int:
    import struct as _s
    return _s.unpack("<H", _s.pack("<e", x))[0]


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tiny.gguf")
    rng = random.Random(0xC0FFEE)

    # A scaled-down Qwen2: same head geometry ratios, 2 layers instead of 24.
    n_layers, d_model, n_heads, n_kv_heads, d_ff = 2, 896, 14, 2, 4864
    vocab = [f"tok{i}" for i in range(512)]

    w = (Writer()
         .string("general.architecture", "qwen2")
         .string("general.name", "tiny-qwen2-synthetic")
         .u32("general.file_type", 15)
         .u32("general.quantization_version", 2)
         .u32("qwen2.block_count", n_layers)
         .u32("qwen2.context_length", 32768)
         .u32("qwen2.embedding_length", d_model)
         .u32("qwen2.feed_forward_length", d_ff)
         .u32("qwen2.attention.head_count", n_heads)
         .u32("qwen2.attention.head_count_kv", n_kv_heads)
         .f32("qwen2.attention.layer_norm_rms_epsilon", 1e-6)
         .f32("qwen2.rope.freq_base", 1_000_000.0)
         .u32("qwen2.rope.dimension_count", 64)
         .string("tokenizer.ggml.model", "gpt2")
         .string("tokenizer.ggml.pre", "qwen2")
         .str_array("tokenizer.ggml.tokens", vocab)
         .u32("tokenizer.ggml.eos_token_id", 151645)
         .u32("tokenizer.ggml.padding_token_id", 151643)
         .boolean("tokenizer.ggml.add_bos_token", False)
         .string("tokenizer.chat_template",
                 "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"))

    def rand_bytes(n):
        return bytes(rng.randrange(256) for _ in range(n))

    def q4k(n_blocks):
        # d and dmin kept small and positive so dequantised values look plausible.
        b = bytearray()
        for _ in range(n_blocks):
            b += struct.pack("<HH", f32_to_f16_bits(0.01), f32_to_f16_bits(0.005))
            b += rand_bytes(12) + rand_bytes(128)
        return bytes(b)

    def q6k(n_blocks):
        b = bytearray()
        for _ in range(n_blocks):
            b += rand_bytes(128) + rand_bytes(64)
            b += bytes((rng.randrange(-8, 8) & 0xFF) for _ in range(16))
            b += struct.pack("<H", f32_to_f16_bits(0.002))
        return bytes(b)

    def q8_0(n_blocks):
        b = bytearray()
        for _ in range(n_blocks):
            b += struct.pack("<H", f32_to_f16_bits(0.01)) + rand_bytes(32)
        return bytes(b)

    def q4_0(n_blocks):
        b = bytearray()
        for _ in range(n_blocks):
            b += struct.pack("<H", f32_to_f16_bits(0.02)) + rand_bytes(16)
        return bytes(b)

    def f32_data(n):
        return struct.pack(f"<{n}f", *[rng.uniform(0.5, 1.5) for _ in range(n)])

    d_head = d_model // n_heads
    kv_dim = n_kv_heads * d_head

    # Only a couple of rows per matrix -- the shapes are honest about the ggml
    # axis order, the sizes are not, which keeps the file small.
    w.tensor("token_embd.weight", [d_model, 512], 14, q6k(d_model * 512 // 256))
    for i in range(n_layers):
        w.tensor(f"blk.{i}.attn_norm.weight", [d_model], 0, f32_data(d_model))
        w.tensor(f"blk.{i}.attn_q.weight", [d_model, 256], 12, q4k(d_model * 256 // 256))
        w.tensor(f"blk.{i}.attn_k.weight", [d_model, kv_dim], 12, q4k(d_model * kv_dim // 256))
        w.tensor(f"blk.{i}.attn_v.weight", [d_model, kv_dim], 8, q8_0(d_model * kv_dim // 32))
        w.tensor(f"blk.{i}.attn_output.weight", [256, d_model], 12, q4k(256 * d_model // 256))
        w.tensor(f"blk.{i}.ffn_norm.weight", [d_model], 0, f32_data(d_model))
        w.tensor(f"blk.{i}.ffn_gate.weight", [d_model, 256], 2, q4_0(d_model * 256 // 32))
        w.tensor(f"blk.{i}.ffn_up.weight", [d_model, 256], 12, q4k(d_model * 256 // 256))
        w.tensor(f"blk.{i}.ffn_down.weight", [256, d_model], 12, q4k(256 * d_model // 256))
    w.tensor("output_norm.weight", [d_model], 0, f32_data(d_model))
    w.tensor("output.weight", [d_model, 512], 14, q6k(d_model * 512 // 256))

    blob = w.build()
    out.write_bytes(blob)
    print(f"wrote {out} ({len(blob) / 1024:.1f} KiB, {w.n_t} tensors, {w.n_kv} metadata keys)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
