#!/usr/bin/env python3
"""Export tokenizer metadata embedded in a GGUF file.

The generated vocab.json, merges.txt and special_tokens.json can be loaded by
mini-llama's BpeTokenizer.  tokenizer_metadata.json and chat_template.txt are
also written to make the original GGUF metadata easy to inspect.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any, BinaryIO


GGUF_TYPES = {
    0: ("B", 1),   # uint8
    1: ("b", 1),   # int8
    2: ("H", 2),   # uint16
    3: ("h", 2),   # int16
    4: ("I", 4),   # uint32
    5: ("i", 4),   # int32
    6: ("f", 4),   # float32
    7: ("?", 1),   # bool
    10: ("Q", 8),  # uint64
    11: ("q", 8),  # int64
    12: ("d", 8),  # float64
}
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
MAX_COLLECTION_SIZE = 20_000_000
MAX_STRING_SIZE = 256 * 1024 * 1024


class GgufError(RuntimeError):
    """Raised when a GGUF file is truncated or structurally invalid."""


class GgufReader:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream

    def read_exact(self, size: int) -> bytes:
        data = self.stream.read(size)
        if len(data) != size:
            raise GgufError("unexpected end of GGUF file")
        return data

    def scalar(self, value_type: int) -> Any:
        try:
            format_code, size = GGUF_TYPES[value_type]
        except KeyError as error:
            raise GgufError(f"unsupported GGUF scalar type: {value_type}") from error
        return struct.unpack("<" + format_code, self.read_exact(size))[0]

    def uint32(self) -> int:
        return int(self.scalar(4))

    def uint64(self) -> int:
        return int(self.scalar(10))

    def string(self) -> str:
        size = self.uint64()
        if size > MAX_STRING_SIZE:
            raise GgufError(f"GGUF string is too large: {size} bytes")
        try:
            return self.read_exact(size).decode("utf-8")
        except UnicodeDecodeError as error:
            raise GgufError("GGUF metadata contains invalid UTF-8") from error

    def value(self, value_type: int) -> Any:
        if value_type in GGUF_TYPES:
            return self.scalar(value_type)
        if value_type == GGUF_TYPE_STRING:
            return self.string()
        if value_type == GGUF_TYPE_ARRAY:
            element_type = self.uint32()
            count = self.uint64()
            if count > MAX_COLLECTION_SIZE:
                raise GgufError(f"GGUF array is too large: {count} elements")
            return [self.value(element_type) for _ in range(count)]
        raise GgufError(f"unsupported GGUF value type: {value_type}")


def read_metadata(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as stream:
        reader = GgufReader(stream)
        if reader.read_exact(4) != b"GGUF":
            raise GgufError("invalid GGUF magic")
        version = reader.uint32()
        if version not in (2, 3):
            raise GgufError(f"unsupported GGUF version: {version}")
        _tensor_count = reader.uint64()
        metadata_count = reader.uint64()
        if metadata_count > MAX_COLLECTION_SIZE:
            raise GgufError(f"too many metadata entries: {metadata_count}")

        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = reader.string()
            if key in metadata:
                raise GgufError(f"duplicate metadata key: {key}")
            metadata[key] = reader.value(reader.uint32())
        return version, metadata


def tokenizer_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items()
            if key.startswith("tokenizer.")}


def build_special_tokens(metadata: dict[str, Any], tokens: list[str]) -> dict[str, int]:
    special: dict[str, int] = {}
    id_keys = {
        "tokenizer.ggml.bos_token_id": "bos_id",
        "tokenizer.ggml.eos_token_id": "eos_id",
        "tokenizer.ggml.unknown_token_id": "unk_id",
        "tokenizer.ggml.unk_token_id": "unk_id",
        "tokenizer.ggml.padding_token_id": "pad_id",
        "tokenizer.ggml.separator_token_id": "sep_id",
    }
    for metadata_key, output_key in id_keys.items():
        value = metadata.get(metadata_key)
        if isinstance(value, int) and not isinstance(value, bool):
            special[output_key] = value

    token_types = metadata.get("tokenizer.ggml.token_type")
    if not isinstance(token_types, list) or len(token_types) != len(tokens):
        token_types = [None] * len(tokens)
    for token_id, (token, token_type) in enumerate(zip(tokens, token_types)):
        looks_special = (
            (token.startswith("<|") and token.endswith("|>"))
            or token in {"<unk>", "<s>", "</s>", "<bos>", "<eos>"}
        )
        if token_type in (2, 3, 4) or looks_special:
            special[token] = token_id
    return special


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=False)
        output.write("\n")


def export_tokenizer(gguf_path: Path, output_dir: Path) -> list[Path]:
    version, metadata = read_metadata(gguf_path)
    tokens = metadata.get("tokenizer.ggml.tokens")
    if not isinstance(tokens, list) or not tokens or not all(
            isinstance(token, str) for token in tokens):
        raise GgufError("GGUF does not contain tokenizer.ggml.tokens")
    if len(set(tokens)) != len(tokens):
        raise GgufError("tokenizer.ggml.tokens contains duplicate strings")
    merges = metadata.get("tokenizer.ggml.merges", [])
    if not isinstance(merges, list) or not all(isinstance(item, str) for item in merges):
        raise GgufError("tokenizer.ggml.merges is not a string array")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "tokenizer_metadata.json"
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"
    special_path = output_dir / "special_tokens.json"

    inspectable = {"gguf_version": version, **tokenizer_metadata(metadata)}
    write_json(metadata_path, inspectable)
    write_json(vocab_path, {token: token_id for token_id, token in enumerate(tokens)})
    with merges_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("#version: 0.2\n")
        for merge in merges:
            output.write(merge)
            output.write("\n")
    write_json(special_path, build_special_tokens(metadata, tokens))

    written = [metadata_path, vocab_path, merges_path, special_path]
    chat_template = metadata.get("tokenizer.chat_template")
    if isinstance(chat_template, str):
        template_path = output_dir / "chat_template.txt"
        with template_path.open("w", encoding="utf-8", newline="\n") as output:
            output.write(chat_template)
            if chat_template and not chat_template.endswith("\n"):
                output.write("\n")
        written.append(template_path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export tokenizer metadata and BPE files from a GGUF model.")
    parser.add_argument("gguf", type=Path, help="input .gguf file")
    parser.add_argument(
        "output_dir", nargs="?", type=Path,
        help="destination directory (default: <model-stem>-tokenizer)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gguf_path: Path = args.gguf
    output_dir: Path = args.output_dir or gguf_path.with_name(
        gguf_path.stem + "-tokenizer")
    try:
        written = export_tokenizer(gguf_path, output_dir)
    except (OSError, GgufError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
