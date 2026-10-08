#!/usr/bin/env python3
"""Independent NumPy F32 forward for the tiny model.json/model.bin format.

Linear weights are row-major [out, in]. A model owns its [layer, pos,
kv_head, head_dim] caches; reuse it for every prompt and decode token.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


F32 = np.float32


@dataclass
class ReferenceModel:
    config: dict
    weights: dict
    keys: np.ndarray
    values: np.ndarray
    next_pos: int = 0


def load_model(model_dir: Path, config_path=None, weights_path=None) -> ReferenceModel:
    """Read F32 weights by the manifest's byte offsets, without C++ bindings."""
    model_dir = Path(model_dir)
    manifest = json.loads(Path(config_path or model_dir / "model.json").read_text())
    config = dict(manifest["config"])
    config["head_dim"] = config["dim"] // config["n_heads"]
    if (config["dim"] % config["n_heads"]
            or config["n_heads"] % config["n_kv_heads"]
            or config["head_dim"] % 2):
        raise ValueError("invalid head dimensions")
    weights = {}
    with Path(weights_path or model_dir / "model.bin").open("rb") as stream:
        for tensor in manifest["tensors"]:
            if tensor["dtype"] != "float32":
                raise ValueError(f"reference only supports F32: {tensor['name']}")
            count = int(np.prod(tensor["shape"]))
            if tensor["byte_size"] != count * 4:
                raise ValueError(f"invalid byte_size: {tensor['name']}")
            stream.seek(tensor["offset"])
            raw = stream.read(tensor["byte_size"])
            if len(raw) != tensor["byte_size"]:
                raise ValueError(f"truncated weight: {tensor['name']}")
            weights[tensor["name"]] = np.frombuffer(raw, dtype="<f4").astype(
                F32, copy=True
            ).reshape(tensor["shape"])
    cache_shape = (config["n_layers"], config["max_seq_len"],
                   config["n_kv_heads"], config["head_dim"])
    return ReferenceModel(config, weights, np.zeros(cache_shape, dtype=F32),
                          np.zeros(cache_shape, dtype=F32))


def rms_norm(x, weight, eps):
    # Accumulate in input order, as the scalar C++ RMSNorm does.
    mean_square = np.add.accumulate(x * x, dtype=F32)[-1] / F32(x.size)
    scale = F32(1) / np.sqrt(mean_square + F32(eps))
    return (x * scale) * weight


def rope(x, pos, theta):
    """Normal/Llama RoPE: adjacent pairs, with the absolute token position."""
    head_dim = x.shape[1]
    exponent = np.arange(0, head_dim, 2, dtype=F32) / F32(head_dim)
    frequency = F32(1) / np.power(F32(theta), exponent)
    angle = F32(pos) * frequency
    cosine, sine = np.cos(angle), np.sin(angle)
    even, odd = x[:, 0::2].copy(), x[:, 1::2].copy()
    x[:, 0::2] = even * cosine - odd * sine
    x[:, 1::2] = even * sine + odd * cosine
    return x


def softmax(scores):
    exp = np.exp(scores - np.max(scores))
    return exp / np.add.accumulate(exp, dtype=F32)[-1]


def forward_token(model: ReferenceModel, pos: int, token_id: int) -> np.ndarray:
    """Forward one token at pos, retaining this model's KV cache for later calls."""
    c, w = model.config, model.weights
    if not 0 <= token_id < c["vocab_size"]:
        raise ValueError("token_id outside vocabulary")
    if not 0 <= pos < c["max_seq_len"] or pos != model.next_pos:
        raise ValueError("positions must be contiguous, starting at zero")
    x = w["token_embedding"][token_id].copy()
    head_dim = c["head_dim"]
    scale = F32(1) / np.sqrt(F32(head_dim))
    group_size = c["n_heads"] // c["n_kv_heads"]

    for layer in range(c["n_layers"]):
        prefix = f"layers.{layer}."

        def linear(name, value):
            return w[prefix + name] @ value

        normalized = rms_norm(x, w[prefix + "attention_norm"], c["rms_norm_eps"])
        q = linear("wq", normalized).reshape(c["n_heads"], head_dim)
        k = linear("wk", normalized).reshape(c["n_kv_heads"], head_dim)
        v = linear("wv", normalized).reshape(c["n_kv_heads"], head_dim)
        q = rope(q, pos, c["rope_theta"])
        k = rope(k, pos, c["rope_theta"])
        model.keys[layer, pos] = k
        model.values[layer, pos] = v

        attention = np.empty((c["n_heads"], head_dim), dtype=F32)
        for q_head in range(c["n_heads"]):
            kv_head = q_head // group_size
            keys = model.keys[layer, :pos + 1, kv_head]
            values = model.values[layer, :pos + 1, kv_head]
            scores = np.add.accumulate(keys * q[q_head], axis=1, dtype=F32)[:, -1]
            probabilities = softmax(scores * scale)
            attention[q_head] = np.add.accumulate(
                probabilities[:, None] * values, axis=0, dtype=F32
            )[-1]
        x = x + linear("wo", attention.reshape(-1))
        normalized = rms_norm(x, w[prefix + "ffn_norm"], c["rms_norm_eps"])
        gate, up = linear("w_gate", normalized), linear("w_up", normalized)
        activated = (gate / (F32(1) + np.exp(-gate))) * up
        x = x + linear("w_down", activated)

    normalized = rms_norm(x, w["final_norm"], c["rms_norm_eps"])
    model.next_pos = pos + 1
    return np.asarray(w["lm_head"] @ normalized, dtype=F32)
