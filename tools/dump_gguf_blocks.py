#!/usr/bin/env python3
"""Extract real quantised blocks from a GGUF file and turn them into Rust test fixtures.

Hand-built test blocks prove our kernels match the format spec as we read it.
They cannot prove we read it the same way ggml did.  This script closes that gap:
it pulls raw blocks out of a file llama.cpp actually produced, dequantises them
with gguf-py's own reference implementation, and writes the pair into
`crates/core/src/quant/fixtures_generated.rs`, where the Rust test suite picks
them up automatically.

Usage:
    python3 tools/dump_gguf_blocks.py models/qwen2.5-0.5b-instruct-q4_k_m.gguf
    python3 tools/dump_gguf_blocks.py model.gguf --tensor blk.0.ffn_gate.weight
    python3 tools/dump_gguf_blocks.py model.gguf --blocks 8 --list

Requires (for fixture generation only):
    pip install gguf numpy

Without those, `--list` still works: the GGUF parsing here is pure stdlib and is
deliberately a second, independent implementation of the same format, so a
disagreement between it and the Rust parser is itself informative.
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------- format ----

# Metadata value type tags.
(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64) = range(13)

_FIXED = {
    U8: ("<B", 1), I8: ("<b", 1),
    U16: ("<H", 2), I16: ("<h", 2),
    U32: ("<I", 4), I32: ("<i", 4),
    U64: ("<Q", 8), I64: ("<q", 8),
    F32: ("<f", 4), F64: ("<d", 8),
    BOOL: ("<B", 1),
}

# (name, block_elems, block_bytes) keyed by ggml type id.  Mirrors the Rust
# table in crates/core/src/gguf/ggml_type.rs -- keep the two in sync.
GGML_TYPES = {
    0: ("f32", 1, 4), 1: ("f16", 1, 2),
    2: ("q4_0", 32, 18), 3: ("q4_1", 32, 20),
    6: ("q5_0", 32, 22), 7: ("q5_1", 32, 24),
    8: ("q8_0", 32, 34), 9: ("q8_1", 32, 36),
    10: ("q2_K", 256, 84), 11: ("q3_K", 256, 110), 12: ("q4_K", 256, 144),
    13: ("q5_K", 256, 176), 14: ("q6_K", 256, 210), 15: ("q8_K", 256, 292),
    16: ("iq2_xxs", 256, 66), 17: ("iq2_xs", 256, 74), 18: ("iq3_xxs", 256, 98),
    19: ("iq1_s", 256, 50), 20: ("iq4_nl", 32, 18), 21: ("iq3_s", 256, 110),
    22: ("iq2_s", 256, 82), 23: ("iq4_xs", 256, 136),
    24: ("i8", 1, 1), 25: ("i16", 1, 2), 26: ("i32", 1, 4), 27: ("i64", 1, 8),
    28: ("f64", 1, 8), 29: ("iq1_m", 256, 56), 30: ("bf16", 1, 2),
    34: ("tq1_0", 256, 54), 35: ("tq2_0", 256, 66),
}

# The formats nano-infer implements. Fixtures for anything else would only
# document a gap, so we skip them.
# 0 f32, 1 f16, 2 q4_0, 6 q5_0, 8 q8_0, 12 q4_K, 14 q6_K.
SUPPORTED = {0, 1, 2, 6, 8, 12, 14}


@dataclass
class Tensor:
    name: str
    dims: list[int]
    type_id: int
    offset: int

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def type_name(self) -> str:
        return GGML_TYPES[self.type_id][0]

    @property
    def n_blocks(self) -> int:
        return self.n_elements // GGML_TYPES[self.type_id][1]


class Cursor:
    """Bounds-checked reader over an open file."""

    def __init__(self, f):
        self.f = f

    def read(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError(f"wanted {n} bytes at {self.f.tell()}, got {len(b)}")
        return b

    def scalar(self, tag: int):
        fmt, size = _FIXED[tag]
        v = struct.unpack(fmt, self.read(size))[0]
        return bool(v) if tag == BOOL else v

    def string(self) -> str:
        (n,) = struct.unpack("<Q", self.read(8))
        return self.read(n).decode("utf-8", errors="replace")

    def value(self, tag: int, depth: int = 0):
        if tag == STRING:
            return self.string()
        if tag == ARRAY:
            if depth > 4:
                raise ValueError("array nesting too deep")
            (elem,) = struct.unpack("<I", self.read(4))
            (n,) = struct.unpack("<Q", self.read(8))
            return [self.value(elem, depth + 1) for _ in range(n)]
        return self.scalar(tag)


def parse_gguf(path: Path):
    """Returns (metadata dict, tensors, data_offset, alignment)."""
    f = open(path, "rb")
    c = Cursor(f)

    magic = c.read(4)
    if magic != b"GGUF":
        raise ValueError(f"not a GGUF file: magic is {magic!r}")
    version, = struct.unpack("<I", c.read(4))
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}")
    tensor_count, kv_count = struct.unpack("<QQ", c.read(16))

    meta = {}
    for _ in range(kv_count):
        key = c.string()
        (tag,) = struct.unpack("<I", c.read(4))
        meta[key] = c.value(tag)

    alignment = int(meta.get("general.alignment", 32))

    tensors = []
    for _ in range(tensor_count):
        name = c.string()
        (n_dims,) = struct.unpack("<I", c.read(4))
        dims = list(struct.unpack(f"<{n_dims}Q", c.read(8 * n_dims)))
        (type_id,) = struct.unpack("<I", c.read(4))
        (offset,) = struct.unpack("<Q", c.read(8))
        if type_id not in GGML_TYPES:
            raise ValueError(f"unknown ggml type {type_id} for tensor {name!r}")
        tensors.append(Tensor(name, dims, type_id, offset))

    pos = f.tell()
    data_offset = (pos + alignment - 1) // alignment * alignment
    return f, meta, tensors, data_offset, alignment, version


# --------------------------------------------------------------- fixtures ----

def reference_dequantize(raw: bytes, type_id: int):
    """Dequantise with gguf-py, the reference implementation."""
    import numpy as np
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize

    arr = np.frombuffer(raw, dtype=np.uint8)
    out = dequantize(arr, GGMLQuantizationType(type_id))
    return np.asarray(out, dtype=np.float32).reshape(-1)


def rust_literal(values) -> str:
    """Shortest literals that still round-trip through f32.

    `repr()` on a Python float prints the f64 shortest form, which for an f32
    value is ~17 digits -- correct, but clippy flags every one of them as
    excessive precision, and 24 blocks of 256 values is a lot of noise. numpy's
    float32 str() gives the shortest decimal that round-trips to the same f32.
    The Rust test asserts bit-exact agreement, so a bad round-trip here fails
    loudly rather than silently loosening the check.
    """
    import numpy as np

    out = []
    for v in values:
        f = np.float32(v)
        if np.isnan(f):
            out.append("f32::NAN")
        elif np.isinf(f):
            out.append("f32::INFINITY" if f > 0 else "f32::NEG_INFINITY")
        else:
            t = str(f)
            # Rust needs a decimal point or an exponent to read this as a float.
            if "." not in t and "e" not in t and "E" not in t:
                t += ".0"
            out.append(f"{t}f32")
    return ", ".join(out)


def emit(fixtures, model_name: str, out_path: Path) -> None:
    lines = [
        "// @generated by tools/dump_gguf_blocks.py -- do not edit by hand.",
        "//",
        f"// Source model: {model_name}",
        "// Expected values come from gguf-py's reference dequantize().",
        "",
        "/// Blocks extracted from a real model, with reference dequantised values.",
        "pub static FIXTURES: &[BlockFixture] = &[",
    ]
    for fx in fixtures:
        raw = ", ".join(f"0x{b:02x}" for b in fx["raw"])
        lines += [
            "    BlockFixture {",
            f"        ggml_type: {fx['type_id']},",
            f"        tensor: {fx['tensor']!r},".replace("'", '"'),
            f"        block_index: {fx['block_index']},",
            f"        raw: &[{raw}],",
            f"        expected: &[{rust_literal(fx['expected'])}],",
            "    },",
        ]
    lines += [
        "];",
        "",
        "/// Which file these came from, for the test's failure message.",
        f'pub static SOURCE_MODEL: &str = "{model_name}";',
        "",
    ]
    out_path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("--tensor", action="append", default=[],
                    help="tensor to sample (repeatable). Default: one per quantised type.")
    ap.add_argument("--blocks", type=int, default=3, help="blocks per tensor (default 3)")
    ap.add_argument("--list", action="store_true", help="just list tensors and exit")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "crates/core/src/quant/fixtures_generated.rs")
    args = ap.parse_args()

    f, meta, tensors, data_offset, alignment, version = parse_gguf(args.model)
    arch = meta.get("general.architecture", "?")
    print(f"{args.model.name}: GGUF v{version}, arch {arch}, "
          f"{len(tensors)} tensors, {len(meta)} metadata keys, "
          f"alignment {alignment}, data at {data_offset}")

    if args.list:
        for t in tensors:
            print(f"  {t.name:<44} {t.type_name:<8} {t.dims}  {t.n_blocks} blocks @ {t.offset}")
        return 0

    # Default selection: the first tensor of each supported quantised type, plus
    # one from the middle of the network -- early tensors are often atypical.
    if args.tensor:
        chosen = [t for t in tensors if t.name in args.tensor]
        missing = set(args.tensor) - {t.name for t in chosen}
        if missing:
            print(f"error: no such tensor(s): {sorted(missing)}", file=sys.stderr)
            return 1
    else:
        chosen, seen = [], set()
        for t in tensors:
            if t.type_id in SUPPORTED and t.type_id not in seen and t.n_blocks > 0:
                seen.add(t.type_id)
                chosen.append(t)
        mid = [t for t in tensors if t.name.startswith("blk.12.") and t.type_id in SUPPORTED]
        chosen += mid[:1]

    try:
        import numpy  # noqa: F401
        from gguf.quants import dequantize  # noqa: F401
    except ImportError:
        print("\nerror: fixture generation needs the reference implementation:\n"
              "    pip install gguf numpy\n"
              "(--list works without it)", file=sys.stderr)
        return 1

    fixtures = []
    for t in chosen:
        _, be, bb = GGML_TYPES[t.type_id]
        n = min(args.blocks, t.n_blocks)
        print(f"  sampling {n} block(s) of {t.name} ({t.type_name})")
        for i in range(n):
            # Spread samples across the tensor: block 0 is often unrepresentative
            # (all-zero padding rows show up at the start of some tensors).
            bi = i * max(1, t.n_blocks // max(n, 1))
            f.seek(data_offset + t.offset + bi * bb)
            raw = f.read(bb)
            expected = reference_dequantize(raw, t.type_id)
            assert len(expected) == be, f"{t.name}: got {len(expected)} values, expected {be}"
            fixtures.append({
                "type_id": t.type_id,
                "tensor": t.name,
                "block_index": bi,
                "raw": raw,
                "expected": expected.tolist(),
            })

    emit(fixtures, args.model.name, args.out)
    print(f"\nwrote {len(fixtures)} fixtures to {args.out}")
    print("now run:  cargo test -p nano-infer-core matches_reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
