#!/usr/bin/env python3
"""Generate the complete 128-ID ASCII JSON vocabulary used by demo models."""

import argparse
import json
from pathlib import Path


def ascii_vocab():
    names = ("<unk>", "<bos>", "<eos>")
    return [dict(id=index, content=names[index] if index < 3 else chr(index),
                 special=index < 3) for index in range(128)]


def write_vocab(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ascii_vocab(), ensure_ascii=True, indent=2) + "\n",
                      encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        parser.error(f"output already exists: {args.output}; use --force to replace")
    write_vocab(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
