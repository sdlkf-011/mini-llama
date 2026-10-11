#!/usr/bin/env python3
"""Run C++ dump -> NumPy reference -> strict logits comparison using tiny weights."""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

try:
    import numpy as np
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    print("NOT RUN: golden-logits requires NumPy", file=sys.stderr)
    raise SystemExit(77)

from compare_logs import compare_logits, read_tokens, step_files
from make_golden import encode_prompt


ROOT = Path(__file__).resolve().parents[1]


def run(command):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command}\n"
                           f"{result.stdout}\n{result.stderr}")
    return result.stdout


def write_ascii_vocab(path):
    # Existing tiny/vocab.json is sparse. Supply the complete ASCII mapping for
    # this test through the existing JSON tokenizer override, including BOS=1.
    entries = [{"id": i, "content": name, "special": True}
               for i, name in enumerate(("<unk>", "<bos>", "<eos>"))]
    entries += [{"id": i, "content": chr(i), "special": False} for i in range(128)]
    path.write_text(json.dumps(entries))


def write_variant(directory, mode):
    """Reuse tiny weights; alter only test metadata or selected fixture tensors."""
    directory.mkdir()
    manifest = json.loads((ROOT / "models/tiny/model.json").read_text())
    data = bytearray((ROOT / "models/tiny/model.bin").read_bytes())
    if mode == "gqa":
        manifest["config"]["n_kv_heads"] = 2
        # Non-default parameters make ignored RMS eps/RoPE theta detectable.
        manifest["config"]["rms_norm_eps"] = 0.003
        manifest["config"]["rope_theta"] = 1234.0
        for tensor in manifest["tensors"]:
            if tensor["name"].endswith((".wk", ".wv")):
                tensor["shape"][0] //= 2
                tensor["byte_size"] //= 2
    elif mode in ("tie", "eos"):
        for tensor in manifest["tensors"]:
            name = tensor["name"]
            if name == "lm_head" or (mode == "eos" and len(tensor["shape"]) == 2):
                weight = np.zeros(tensor["shape"], dtype="<f4")
                if mode == "eos" and name == "token_embedding":
                    weight.fill(1)
                if mode == "eos" and name == "lm_head":
                    weight[2].fill(1)
                start = tensor["offset"]
                data[start:start + tensor["byte_size"]] = weight.tobytes()
    else:
        raise ValueError(f"unknown fixture mode: {mode}")
    (directory / "model.json").write_text(json.dumps(manifest))
    (directory / "model.bin").write_bytes(data)
    return directory


def check_comparator(directory):
    """Ensure bad dumps cannot silently pass the regression gate."""
    directory.mkdir()
    cpp = directory / "logits_step0.bin"
    py = directory / "python_logits_step0.bin"
    cpp_tokens = directory / "generation_tokens.txt"
    py_tokens = directory / "python_generation_tokens.txt"

    def reset():
        np.array([0.001, 1.0], dtype=np.float32).tofile(cpp)
        np.array([0.001, 1.0], dtype=np.float32).tofile(py)
        cpp_tokens.write_text("1\n")
        py_tokens.write_text("1\n")

    def expect_failure(label, corrupt):
        reset()
        corrupt()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                compare_logits(directory, directory)
        except ValueError:
            return
        raise RuntimeError(f"comparator accepted {label}")

    expect_failure("missing step", py.unlink)
    expect_failure("absolute error", lambda: np.array([0.001, 1.01], dtype=np.float32).tofile(cpp))
    expect_failure("relative error", lambda: np.array([0.001001, 1], dtype=np.float32).tofile(cpp))
    expect_failure("non-finite logits", lambda: np.array([np.nan, 1], dtype=np.float32).tofile(cpp))
    expect_failure("truncated binary", lambda: cpp.write_bytes(b"bad"))
    expect_failure("generation mismatch", lambda: cpp_tokens.write_text("0\n"))

    def top1_mismatch():
        np.array([0, 1e-6], dtype=np.float32).tofile(cpp)
        np.array([1e-6, 0], dtype=np.float32).tofile(py)

    expect_failure("top-1 mismatch within tolerance", top1_mismatch)
    reset()
    # Near-zero reference values are excluded only from relative error.
    np.array([0, 1], dtype=np.float32).tofile(py)
    np.array([1e-5, 1], dtype=np.float32).tofile(cpp)
    with contextlib.redirect_stdout(io.StringIO()):
        compare_logits(directory, directory)
    reset()
    np.array([0.002, 1], dtype=np.float32).tofile(py)
    np.array([0.0020005, 1], dtype=np.float32).tofile(cpp)
    # Confirm the CLI exposes failures as a nonzero exit status too.
    result = subprocess.run([sys.executable, str(ROOT / "scripts/compare_logs.py"),
                             "--cpp-dir", str(directory)], capture_output=True, timeout=30)
    if result.returncode == 0:
        raise RuntimeError("comparator CLI returned zero for mismatched logits")
    print("PASS: comparator rejects corrupt dumps, numerical errors and token mismatches")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path,
                        default=Path(os.environ.get("MINI_LLAMA_BIN", ROOT / "build/mini-llama")))
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise RuntimeError(f"build the CLI first: {binary}")

    with tempfile.TemporaryDirectory(prefix="mini-llama-golden-") as temporary:
        directory = Path(temporary)
        vocab = directory / "ascii-vocab.json"
        write_ascii_vocab(vocab)
        tiny = ROOT / "models/tiny"
        gqa = write_variant(directory / "gqa", "gqa")
        tie = write_variant(directory / "tie", "tie")
        eos = write_variant(directory / "eos", "eos")
        cases = [
            ("tiny", tiny, "hello", 8, None),
            ("gqa", gqa, "Hi 42!", 8, None),
            ("single-generated", tiny, "", 1, None),
            ("prefill-only", tiny, "ASCII ~\t", 0, []),
            ("greedy-tie", tie, "hi", 4, [0, 0, 0, 0]),
            ("early-eos", eos, "hi", 4, [2]),
        ]
        for name, model, prompt, budget, expected_tokens in cases:
            print(f"CASE: {name}", flush=True)
            cpp_dir, py_dir = directory / name / "cpp", directory / name / "python"
            common = ["--prompt", prompt, "--n-predict", str(budget)]
            run([str(binary), "generate", "--model", str(model), "--tokenizer", str(vocab),
                 *common, "--temperature", "0", "--top-k", "0", "--threads", "1",
                 "--dump-logits", str(cpp_dir)])
            print(run([sys.executable, str(ROOT / "scripts/make_golden.py"),
                       "--model", str(model), "--output-dir", str(py_dir), *common]), end="")
            print(run([sys.executable, str(ROOT / "scripts/compare_logs.py"),
                       "--cpp-dir", str(cpp_dir), "--python-dir", str(py_dir)]), end="")
            generated = read_tokens(cpp_dir / "generation_tokens.txt")
            if expected_tokens is not None and generated != expected_tokens:
                raise RuntimeError(f"{name}: unexpected generated tokens: {generated}")
            if len(generated) != budget and (not generated or generated[-1] != 2):
                raise RuntimeError(f"{name}: generation stopped before budget without EOS")
            expected_steps = len(encode_prompt(prompt)) + max(len(generated) - 1, 0)
            for dump, prefix in ((cpp_dir, "logits_step"), (py_dir, "python_logits_step")):
                if sorted(step_files(dump, prefix)) != list(range(expected_steps)):
                    raise RuntimeError(f"{name}: expected {expected_steps} logits steps")
            # Preserve equivalence with the normal batch-prefill generation path.
            normal = run([str(binary), "generate", "--model", str(model),
                          "--tokenizer", str(vocab), *common, "--threads", "1"])
            expected_line = "generated tokens: " + str(generated)
            if expected_line not in normal.splitlines():
                raise RuntimeError(f"{name}: dump and normal generation differ")
        check_comparator(directory / "comparator")
    print("RESULT: ALL PASSED")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"RESULT: FAILED\n{error}", file=sys.stderr)
        sys.exit(1)
