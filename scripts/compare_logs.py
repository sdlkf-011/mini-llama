#!/usr/bin/env python3
"""Compare C++ and NumPy logits dumps with strict absolute/relative tolerances."""

import argparse
from pathlib import Path
import re
import sys

import numpy as np


MAX_ERROR = 1e-4
RELATIVE_FLOOR = 1e-3


def step_files(directory, prefix):
    files = {}
    for path in Path(directory).glob(f"{prefix}*.bin"):
        match = re.fullmatch(re.escape(prefix) + r"(0|[1-9][0-9]*)\.bin", path.name)
        if not match:
            raise ValueError(f"invalid step filename: {path.name}")
        files[int(match.group(1))] = path
    return files


def read_logits(path):
    raw = path.read_bytes()
    if not raw or len(raw) % 4:
        raise ValueError(f"invalid float32 binary size: {path}")
    logits = np.frombuffer(raw, dtype=np.float32)
    if not np.all(np.isfinite(logits)):
        raise ValueError(f"non-finite logits: {path}")
    return logits


def read_tokens(path):
    return [int(value) for value in path.read_text().split()]


def compare_logits(cpp_dir, python_dir):
    cpp_files = step_files(cpp_dir, "logits_step")
    py_files = step_files(python_dir, "python_logits_step")
    if not cpp_files or cpp_files.keys() != py_files.keys():
        raise ValueError(f"step sets differ or empty: C++={sorted(cpp_files)}, "
                         f"Python={sorted(py_files)}")
    if sorted(cpp_files) != list(range(len(cpp_files))):
        raise ValueError("step numbering must be contiguous from zero")
    worst_abs = worst_rel = 0.0
    vocab_size = None
    failures = []
    for step in sorted(cpp_files):
        cpp, py = read_logits(cpp_files[step]), read_logits(py_files[step])
        if cpp.shape != py.shape or (vocab_size is not None and cpp.size != vocab_size):
            raise ValueError(f"step {step}: inconsistent vocabulary shape")
        vocab_size = cpp.size
        # Float64 here makes error measurement independent of F32 subtraction rounding.
        error = np.abs(cpp.astype(np.float64) - py.astype(np.float64))
        max_abs = float(np.max(error))
        mask = np.abs(py) >= RELATIVE_FLOOR
        max_rel = float(np.max(error[mask] / np.abs(py[mask]))) if np.any(mask) else 0.0
        cpp_top, py_top = int(np.argmax(cpp)), int(np.argmax(py))
        print(f"step {step}: max_abs_error={max_abs:.8g}, "
              f"max_rel_error={max_rel:.8g}, top1={cpp_top}/{py_top}")
        worst_abs, worst_rel = max(worst_abs, max_abs), max(worst_rel, max_rel)
        if max_abs > MAX_ERROR or max_rel > MAX_ERROR or cpp_top != py_top:
            failures.append(f"step {step}: tolerance or top-1 mismatch")

    cpp_tokens = read_tokens(Path(cpp_dir) / "generation_tokens.txt")
    py_tokens = read_tokens(Path(python_dir) / "python_generation_tokens.txt")
    if cpp_tokens != py_tokens:
        failures.append(f"generation tokens differ: C++={cpp_tokens}, Python={py_tokens}")
    if any(token < 0 or token >= vocab_size for token in cpp_tokens + py_tokens):
        failures.append("generation token outside vocabulary")
    if failures:
        raise ValueError("\n".join(failures))
    print(f"PASS: {len(cpp_files)} steps, max_abs_error={worst_abs:.8g}, "
          f"max_rel_error={worst_rel:.8g}, generation tokens match")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-dir", type=Path, required=True)
    parser.add_argument("--python-dir", type=Path)
    args = parser.parse_args()
    try:
        compare_logits(args.cpp_dir, args.python_dir or args.cpp_dir)
    except (ValueError, OSError) as error:
        print(f"RESULT: FAILED\n{error}", file=sys.stderr)
        return 1
    print("RESULT: ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
