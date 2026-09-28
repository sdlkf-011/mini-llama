#!/usr/bin/env python3
"""Generate a tiny deterministic GGUF v3 fixture for local tests and CI."""

import argparse
import pathlib
import struct


ALIGNMENT = 32


def string(value: str) -> bytes:
    """Encode a GGUF string as a uint64 length followed by UTF-8 bytes."""
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def metadata(key: str, value_type: int, value: bytes) -> bytes:
    """Encode one GGUF metadata key/value record."""
    return string(key) + struct.pack("<I", value_type) + value


def align(length: int, alignment: int) -> int:
    """Return the first aligned offset at or after length."""
    return (length + alignment - 1) // alignment * alignment


def build_fixture() -> bytes:
    """Build a small GGUF containing metadata and one four-value F32 tensor."""
    metadata_records = b"".join(
        [
            metadata("general.architecture", 8, string("mini-llama")),
            metadata("general.name", 8, string("tiny-test")),
            metadata("general.alignment", 4, struct.pack("<I", ALIGNMENT)),
            metadata("test.int", 5, struct.pack("<i", -7)),
            metadata("test.float", 6, struct.pack("<f", 1.5)),
            metadata(
                "test.strings",
                9,
                struct.pack("<IQ", 8, 2) + string("one") + string("two"),
            ),
            metadata(
                "test.ints",
                9,
                struct.pack("<IQ", 5, 3) + struct.pack("<iii", 1, -2, 3),
            ),
        ]
    )
    tensor_info = (
        string("test.weight")
        + struct.pack("<I", 2)
        + struct.pack("<QQ", 2, 2)
        + struct.pack("<IQ", 0, 0)
    )
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 7)
    body = header + metadata_records + tensor_info
    data_offset = align(len(body), ALIGNMENT)
    tensor_data = struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0)
    return body + bytes(data_offset - len(body)) + tensor_data


def main() -> None:
    """Parse the output path and write the generated fixture atomically."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(build_fixture())


if __name__ == "__main__":
    main()
