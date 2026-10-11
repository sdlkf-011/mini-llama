#!/usr/bin/env python3
"""Export deterministic random F32 weights for inference and numerical tests.

This is an untrained ASCII demo, not a language-quality model. It writes the
loader's model.json/model.bin plus a complete vocab.json using only stdlib.
"""

import argparse
import json
import math
from pathlib import Path
import random
import struct

from generate_vocab_json import write_vocab
from upgrade_model_json import normalize_manifest


def export_model(output, seed=42, dim=32, hidden_dim=86, n_layers=2,
                 n_heads=4, n_kv_heads=4, max_seq_len=128, force=False):
    manifest = normalize_manifest(dict(
        config=dict(vocab_size=128, dim=dim, hidden_dim=hidden_dim,
                    n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads,
                    max_seq_len=max_seq_len),
        tokenizer=dict(type="json_vocab", path="vocab.json")))
    output = Path(output)
    for name in ("model.json", "model.bin", "vocab.json"):
        if (output / name).exists() and not force:
            raise ValueError(f"output already exists: {output / name}; use --force")
    output.mkdir(parents=True, exist_ok=True)
    generator = random.Random(seed)
    with (output / "model.bin").open("wb") as stream:
        for tensor in manifest["tensors"]:
            count = math.prod(tensor["shape"])
            norm = tensor["name"].endswith("norm")
            # Bounded chunks avoid allocating an entire large tensor in Python.
            for begin in range(0, count, 4096):
                values = [1.0 if norm else generator.gauss(0, 0.02)
                          for _ in range(min(4096, count - begin))]
                stream.write(struct.pack(f"<{len(values)}f", *values))
    write_vocab(output / "vocab.json")
    (output / "model.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                       encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", "--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    for name, default in (("dim", 32), ("hidden-dim", 86), ("n-layers", 2),
                          ("n-heads", 4), ("n-kv-heads", 4), ("max-seq-len", 128)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        manifest = export_model(args.output_dir, args.seed, args.dim, args.hidden_dim,
                                args.n_layers, args.n_heads, args.n_kv_heads,
                                args.max_seq_len, args.force)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(f"Exported {len(manifest['tensors'])} F32 tensors to {args.output_dir}")


if __name__ == "__main__":
    main()
