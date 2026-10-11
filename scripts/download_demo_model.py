#!/usr/bin/env python3
"""Download a GGUF URL or a JSON+BIN demo bundle into a local model directory.

Use --url for one GGUF (optionally --sha256), or --base-url for model.json,
model.bin and vocab.json. Sources are explicit; no model is chosen implicitly.
Downloads are staged and validated before replacing destination files.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.parse
import urllib.request

from upgrade_model_json import check_weights, normalize_manifest


ROOT = Path(__file__).resolve().parents[1]


def download(url, destination, timeout):
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=timeout) as source, destination.open("wb") as target:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            target.write(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--url", help="URL of a GGUF model file")
    sources.add_argument("--base-url", help="directory URL containing the JSON+BIN bundle")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/demo")
    parser.add_argument("--sha256", help="expected SHA-256 for --url")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0 or not args.timeout < float("inf"):
        parser.error("timeout must be positive and finite")
    if args.sha256 and (not args.url or len(args.sha256) != 64
                        or any(c not in "0123456789abcdefABCDEF" for c in args.sha256)):
        parser.error("--sha256 requires --url and a 64-digit hexadecimal digest")
    try:
        if args.url:
            name = Path(urllib.parse.unquote(urllib.parse.urlsplit(args.url).path)).name
            if not name.endswith(".gguf"):
                raise ValueError("--url must name a .gguf file")
            urls = {name: args.url}
        else:
            urls = {name: args.base_url.rstrip("/") + "/" + name
                    for name in ("model.json", "model.bin", "vocab.json")}
        for name in urls:
            if (args.output_dir / name).exists() and not args.force:
                raise ValueError(f"output already exists: {args.output_dir / name}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".download-", dir=args.output_dir) as temporary:
            staging = Path(temporary)
            for name, url in urls.items():
                digest = download(url, staging / name, args.timeout)
                if args.sha256 and digest != args.sha256.lower():
                    raise ValueError("SHA-256 mismatch")
            if args.url:
                with (staging / name).open("rb") as stream:
                    if stream.read(4) != b"GGUF":
                        raise ValueError("download does not have a GGUF header")
            else:
                manifest = normalize_manifest(json.loads((staging / "model.json").read_text()))
                check_weights(manifest, staging / "model.bin")
                if (manifest["tokenizer"]["type"] == "json_vocab"
                        and manifest["tokenizer"]["path"] != "vocab.json"):
                    raise ValueError("bundle tokenizer must reference vocab.json")
                if not isinstance(json.loads((staging / "vocab.json").read_text()), list):
                    raise ValueError("vocab.json must contain the JSON tokenizer's entry array")
                (staging / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
            for name in urls:
                (staging / name).replace(args.output_dir / name)
        print(f"Downloaded {len(urls)} file(s) to {args.output_dir}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Download failed: {error}\n")


if __name__ == "__main__":
    main()
