#!/usr/bin/env python3
"""Upgrade flat F32 model configs to the JSON+BIN manifest used by the loader.

Inferred weights follow embedding, each layer's attention/FFN, final norm,
and lm_head order. Existing offsets are preserved; weights are never rewritten.
"""

import argparse
import copy
import json
import math
from pathlib import Path



def tensor_shapes(config):
    dim, hidden = config["dim"], config["hidden_dim"]
    kv_dim = config["n_kv_heads"] * (dim // config["n_heads"])
    shapes = [("token_embedding", [config["vocab_size"], dim])]
    for layer in range(config["n_layers"]):
        prefix = f"layers.{layer}."
        shapes.extend((prefix + name, shape) for name, shape in (
            ("attention_norm", [dim]), ("wq", [dim, dim]),
            ("wk", [kv_dim, dim]), ("wv", [kv_dim, dim]), ("wo", [dim, dim]),
            ("ffn_norm", [dim]), ("w_gate", [hidden, dim]),
            ("w_up", [hidden, dim]), ("w_down", [dim, hidden])))
    shapes.extend((("final_norm", [dim]), ("lm_head", [config["vocab_size"], dim])))
    return shapes


def normalize_manifest(source):
    """Fill legacy defaults and validate the loader's complete F32 layout."""
    if not isinstance(source, dict):
        raise ValueError("model configuration must be an object")
    config = copy.deepcopy(source.get("config", source))
    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    config = {key: config[key] for key in (
        "vocab_size", "dim", "hidden_dim", "n_layers", "n_heads", "n_kv_heads",
        "max_seq_len", "rope_theta", "rms_norm_eps") if key in config}
    config.setdefault("n_kv_heads", config.get("n_heads"))
    config.setdefault("max_seq_len", 128)
    config.setdefault("rope_theta", 10000.0)
    config.setdefault("rms_norm_eps", 1e-5)
    for key in ("vocab_size", "dim", "hidden_dim", "n_layers", "n_heads",
                "n_kv_heads", "max_seq_len"):
        value = config.get(key)
        if type(value) is not int or not 0 < value <= 2147483647:
            raise ValueError(f"{key} must be a positive int32")
    if (config["dim"] % config["n_heads"]
            or config["n_heads"] % config["n_kv_heads"]
            or (config["dim"] // config["n_heads"]) % 2):
        raise ValueError("invalid Attention/GQA/RoPE dimensions")
    for key in ("rope_theta", "rms_norm_eps"):
        value = config[key]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not 0 < value <= 3.402823466e38):
            raise ValueError(f"{key} must be a positive finite F32 value")
    tokenizer = copy.deepcopy(source.get("tokenizer", {"type": "ascii"}))
    if not isinstance(tokenizer, dict) or tokenizer.get("type", "ascii") not in (
            "ascii", "json_vocab"):
        raise ValueError("tokenizer must use ascii or json_vocab")
    tokenizer.setdefault("type", "ascii")
    for key, default in (("bos_id", 1), ("eos_id", 2), ("unk_id", 0)):
        value = tokenizer.setdefault(key, default)
        if type(value) is not int or not 0 <= value < config["vocab_size"]:
            raise ValueError(f"{key} must be inside the vocabulary")
    if (tokenizer["type"] == "json_vocab" and
            (not isinstance(tokenizer.get("path"), str) or not tokenizer["path"])):
        raise ValueError("json_vocab tokenizer requires a path")
    tensors = copy.deepcopy(source.get("tensors"))
    shapes = dict(tensor_shapes(config))
    if tensors is None:
        tensors, offset = [], 0
        for name, shape in shapes.items():
            size = math.prod(shape) * 4
            tensors.append(dict(name=name, shape=shape, dtype="float32",
                                offset=offset, byte_size=size))
            offset += size
    if not isinstance(tensors, list):
        raise ValueError("tensors must be an array")
    names, ranges = set(), []
    for tensor in tensors:
        if not isinstance(tensor, dict):
            raise ValueError("each tensor must be an object")
        name = tensor.get("name")
        shape = tensor.get("shape")
        if (not isinstance(name, str) or name not in shapes or name in names
                or not isinstance(shape, list) or any(type(d) is not int for d in shape)
                or shape != shapes[name]):
            raise ValueError(f"invalid/duplicate tensor name or shape: {name}")
        names.add(name)
        tensor.setdefault("dtype", "float32")
        tensor.setdefault("byte_size", math.prod(shapes[name]) * 4)
        offset, size = tensor.get("offset"), tensor["byte_size"]
        if (tensor["dtype"] != "float32" or type(offset) is not int or offset < 0
                or type(size) is not int or size != math.prod(shapes[name]) * 4):
            raise ValueError(f"invalid F32 storage: {name}")
        ranges.append((offset, offset + size))
    if names != shapes.keys():
        raise ValueError(f"missing tensors: {sorted(shapes.keys() - names)}")
    ranges.sort()
    if any(first[1] > second[0] for first, second in zip(ranges, ranges[1:])):
        raise ValueError("tensor byte ranges overlap")
    return dict(config=config, tokenizer=tokenizer, tensors=tensors)


def check_weights(manifest, weights):
    expected = max(t["offset"] + t["byte_size"] for t in manifest["tensors"])
    if Path(weights).stat().st_size != expected:
        raise ValueError(f"weights must contain exactly {expected} bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, help="default: model.upgraded.json beside input")
    parser.add_argument("--weights", type=Path, help="also validate the model.bin length")
    args = parser.parse_args()
    try:
        manifest = normalize_manifest(json.loads(args.input.read_text(encoding="utf-8")))
        if args.weights:
            check_weights(manifest, args.weights)
        output = args.output or args.input.with_name(args.input.stem + ".upgraded.json")
        if output.exists():
            raise ValueError(f"output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        parser.exit(1, f"Upgrade failed: {error}\n")
    print(output)


if __name__ == "__main__":
    main()
