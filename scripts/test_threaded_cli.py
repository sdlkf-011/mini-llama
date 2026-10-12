#!/usr/bin/env python3
"""Check CPU thread configuration and inference consistency using the tiny model."""

import argparse
import array
import math
import pathlib
import re
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_cli(binary: pathlib.Path, *arguments: str) -> str:
    result = subprocess.run(
        [str(binary), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    require(
        result.returncode == 0,
        f"CLI failed ({result.returncode}): {' '.join(arguments)}\n{result.stderr}",
    )
    return result.stdout


def check_threads(output: str, expected: int) -> None:
    match = re.search(r"^threads: (\d+)$", output, re.MULTILINE)
    require(match is not None, "missing threads output")
    require(int(match.group(1)) == expected, f"expected threads: {expected}")


def compare_logits(first: pathlib.Path, second: pathlib.Path) -> int:
    names = sorted(path.name for path in first.glob("logits_step*.bin"))
    require(bool(names), "no logits were exported")
    require(
        names == sorted(path.name for path in second.glob("logits_step*.bin")),
        "single-thread and multithread logits steps differ",
    )
    for name in names:
        left, right = array.array("f"), array.array("f")
        left.frombytes((first / name).read_bytes())
        right.frombytes((second / name).read_bytes())
        require(len(left) == len(right) == 128, f"invalid tiny-model logits: {name}")
        for index, (actual, expected) in enumerate(zip(right, left)):
            require(
                math.isfinite(actual)
                and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6),
                f"logits mismatch: {name}, index {index}: {actual} vs {expected}",
            )
    return len(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, default=ROOT / "build/mini-llama")
    binary = parser.parse_args().binary.resolve()
    require(binary.is_file(), f"build the CLI first: {binary}")
    common = [
        str(ROOT / "models/tiny"),
        "--tokenizer", str(ROOT / "models/tiny/vocab.json"),
        "--prompt", "hello",
        "--n-predict", "4",
        "--seed", "42",
    ]

    for command in ("generate", "bench"):
        require("--threads" in run_cli(binary, command, "--help"),
                f"{command} help is missing --threads")

    with tempfile.TemporaryDirectory(prefix="mini-llama-threaded-cli-") as directory:
        dump_dirs = [pathlib.Path(directory) / f"threads-{count}" for count in (1, 4)]
        generated_lines = []
        bench_tokens = []
        for threads, dump_dir in zip((1, 4), dump_dirs):
            output = run_cli(binary, "generate", *common, "--threads", str(threads),
                             "--dump-logits", str(dump_dir))
            check_threads(output, threads)
            tokens = re.search(r"^generated tokens: \[.*\]$", output, re.MULTILINE)
            require(tokens is not None, "missing generated token IDs")
            generated_lines.append(tokens.group(0))
            bench_output = run_cli(binary, "bench", *common, "--threads", str(threads),
                                   "--warmup", "0", "--repeat", "1", "--verbose")
            check_threads(bench_output, threads)
            for label in ("[verbose] prefill", "tokens/s (total):", "tokens/s (Decode):"):
                require(label in bench_output, f"bench is missing {label}")
            tokens = re.findall(r"^\[verbose\] Decode step \d+: token=(\d+)$",
                                bench_output, re.MULTILINE)
            require(len(tokens) == 4, "bench did not execute all generation steps")
            bench_tokens.append(tokens)

        require(generated_lines[0] == generated_lines[1], "generated token IDs differ")
        require(
            (dump_dirs[0] / "generation_tokens.txt").read_text()
            == (dump_dirs[1] / "generation_tokens.txt").read_text(),
            "exported generated token IDs differ",
        )
        steps = compare_logits(*dump_dirs)
        require(bench_tokens[0] == bench_tokens[1], "benchmark token IDs differ")

    automatic = run_cli(binary, "bench", *common, "--threads", "0")
    match = re.search(r"^threads: (\d+)$", automatic, re.MULTILINE)
    require(match is not None and int(match.group(1)) > 0,
            "--threads 0 did not resolve to a positive thread count")
    for command in ("generate", "bench"):
        for value in ("-1", "abc", "2147483648"):
            result = subprocess.run([str(binary), command, "--threads", value],
                                    cwd=ROOT, capture_output=True, text=True, timeout=30)
            require(result.returncode != 0 and "--threads" in result.stderr,
                    f"{command} accepted invalid --threads {value}")

    print(f"PASS: --threads 1/4/0, {steps} logits steps, generated IDs and bench output")


if __name__ == "__main__":
    main()
