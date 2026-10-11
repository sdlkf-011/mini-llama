#!/usr/bin/env python3
"""CPU/CUDA benchmark runner. Unmeasured values are null, never synthetic zeroes.

Only the standard library is required. Run from any directory; model paths are
relative to the repository, build/output paths are relative to the caller.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import subprocess
import tempfile
import textwrap
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
TARGETED = ("cuda_quant_q4|cuda_forward_q4|cuda_forward_quant|"
            "cuda_forward_device_resident|CudaForwardAttentionTest|cuda-chat-smoke")
METRICS = {
    "prompt_tokens": ("prompt tokens",),
    "generated_tokens": ("generated tokens",),
    "decode_tokens": ("decode tokens",),
    "prefill_ms": ("prefill time (ms)", "prefill"),
    "decode_ms": ("decode time (ms)", "decode"),
    "total_ms": ("total time (ms)", "total"),
    "tokens_per_sec_total": ("tokens/s (total)", "throughput"),
    "tokens_per_sec_decode": ("tokens/s (decode)", "decode throughput"),
}
COUNTERS = {
    "linear_calls": "cuda linear calls",
    "activation_calls": "cuda activation calls",
    "attention_calls": "cuda attention calls",
    "cpu_attention_fallback_calls": "cpu attention fallback calls",
    "kv_write_bytes": "cuda kv write bytes",
    "kv_read_bytes": "cuda kv read bytes",
    "host_to_device_copies": "host->device copies",
    "device_to_host_copies": "device->host copies",
}
WEIGHTS = ("f32", "q8_0", "q4_0", "q4_1")


def not_run(reason, command=None):
    return dict(status="NOT_RUN", reason=reason, command=command or [],
                returncode=None, duration_s=None, stdout="", stderr="")


def run_command(command, timeout=30, input_text=None):
    command = list(map(str, command))
    start = time.monotonic()
    result = dict(command=command, returncode=None, stdout="", stderr="", reason="")
    try:
        with subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, errors="replace", start_new_session=os.name == "posix") as proc:
            try:
                stdout, stderr = proc.communicate(input=input_text, timeout=timeout)
                result.update(status="PASSED" if proc.returncode == 0 else "FAILED")
                if proc.returncode:
                    result["reason"] = f"command exited with {proc.returncode}"
            except subprocess.TimeoutExpired:
                # CTest owns child test processes. Kill this private process
                # group so a timed-out suite cannot leak tests into benchmarks.
                if os.name == "posix":
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    proc.kill()
                stdout, stderr = proc.communicate()
                result.update(status="TIMEOUT", reason=f"timeout after {timeout}s")
            result.update(returncode=proc.returncode, stdout=stdout, stderr=stderr)
    except OSError as exc:
        result.update(status="FAILED", reason=str(exc))
    result["duration_s"] = time.monotonic() - start
    return result


def collect_environment(build_dir):
    binary = build_dir / "mini-llama"
    cache_path = build_dir / "CMakeCache.txt"
    cache = {}
    if cache_path.is_file():
        for line in cache_path.read_text(errors="replace").splitlines():
            match = re.match(r"([^:#]+):[^=]+=(.*)", line)
            if match:
                cache[match[1]] = match[2]
    commands = {
        "git_commit": ["git", "rev-parse", "HEAD"],
        "git_branch": ["git", "branch", "--show-current"],
        "git_status": ["git", "status", "--short"],
        "cmake": ["cmake", "--version"],
        "cxx_compiler": [cache.get("CMAKE_CXX_COMPILER", "c++"), "--version"],
        "nvidia_smi": ["nvidia-smi"],
        "nvcc": [cache.get("CMAKE_CUDA_COMPILER", "nvcc"), "--version"],
        "cuda_runtime": [binary, "--cuda-info"],
    }
    probes = {name: run_command(cmd) for name, cmd in commands.items()}
    environment = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                       repo=str(ROOT), build_dir=str(build_dir), binary=str(binary),
                       os=platform.platform(), python=platform.python_version(),
                       build_type=cache.get("CMAKE_BUILD_TYPE", "unavailable"),
                       cuda_build=cache.get("MINI_LLAMA_CUDA", "unavailable"),
                       probes=probes)
    for name, result in probes.items():
        environment[name] = result["stdout"].strip() if result["status"] == "PASSED" else "unavailable"
    runtime = probes["cuda_runtime"]
    count = re.search(r"^CUDA GPU count: (\d+)$", runtime["stdout"], re.M)
    environment["cuda_available"] = bool(runtime["status"] == "PASSED" and
                                         "CUDA backend built: yes" in runtime["stdout"] and
                                         count and int(count[1]) > 0)
    return environment


def make_cases(q4_model=None):
    # chat retains native Q8_0; avoid converting/copying the full real model.
    groups = [("tiny", "models/tiny", "default", None),
              ("chat-q8_0", "models/chat", "q8_0", None),
              ("tiny-q4_0", "models/tiny", "q4_0", "q4_0")]
    if q4_model:
        groups.append(("real-q4", str(q4_model), "q4_native", None))
    return [dict(name=f"{group}-{backend}", group=group, model=model,
                 backend=backend, quant=quant, quant_override=override)
            for group, model, quant, override in groups for backend in ("cpu", "cuda")]


def parse_benchmark_stdout(stdout):
    """Parse current and historical labels; require metrics and validate units.

    Counters describe cumulative prefill + decode work. Old logs without a Total
    label derive it from prefill + decode and identify this in derived_metrics.
    Missing or malformed values raise ValueError instead of silently becoming 0.
    """
    metrics, weights, counters = {}, {}, {}
    weight_scalars = {}
    for line in stdout.splitlines():
        line = line.strip().lower()
        if ":" not in line:
            continue
        label, value = (part.strip() for part in line.split(":", 1))
        for key, aliases in METRICS.items():
            if label in aliases:
                suffix = "ms" if key.endswith("_ms") and label in ("prefill", "decode", "total") else ""
                if label in ("throughput", "decode throughput"):
                    suffix = "tokens/s"
                pattern = r"(\d+)" if key.endswith("_tokens") else r"([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+\-]?\d+)?)"
                match = re.fullmatch(pattern + (r"\s+" + re.escape(suffix) if suffix else ""), value)
                if not match:
                    raise ValueError(f"invalid metric {label}: {value}")
                number = int(match[1]) if key.endswith("_tokens") else float(match[1])
                if not math.isfinite(number) or number < 0:
                    raise ValueError(f"nonfinite/negative metric {label}")
                if key in metrics and metrics[key] != number:
                    raise ValueError(f"conflicting metric {label}")
                metrics[key] = number
        weight_match = re.fullmatch(r"cuda (f32|q8_0|q4_0|q4_1) linear weights", label)
        if weight_match:
            match = re.fullmatch(r"(\d+)/(\d+)", value)
            if not match:
                raise ValueError(f"invalid weight coverage: {line}")
            weights[weight_match[1]] = dict(uploaded=int(match[1]), expected=int(match[2]))
        if label in ("uploaded weights", "gpu memory used"):
            pattern = r"(\d+)" if label == "uploaded weights" else r"(\d+) bytes(?: \([^\n]+\))?"
            match = re.fullmatch(pattern, value)
            if not match:
                raise ValueError(f"invalid counter: {line}")
            weight_scalars[label.replace(" ", "_")] = int(match[1])
        for key, counter_label in COUNTERS.items():
            if label == counter_label:
                copies = key.endswith("copies")
                match = re.fullmatch(r"(\d+)(?: \((\d+) bytes\))" if copies else r"(\d+)", value)
                if not match:
                    raise ValueError(f"invalid counter: {line}")
                if key in counters:
                    raise ValueError(f"duplicate CUDA counter: {label}")
                counters[key] = int(match[1])
                if copies:
                    counters[key.replace("copies", "bytes")] = int(match[2])
    derived = []
    if "total_ms" not in metrics and all(k in metrics for k in ("prefill_ms", "decode_ms")):
        metrics["total_ms"] = metrics["prefill_ms"] + metrics["decode_ms"]
        derived.append("total_ms")
    missing = set(METRICS) - metrics.keys()
    if missing:
        raise ValueError("missing metrics: " + ", ".join(sorted(missing)))
    if metrics["prompt_tokens"] <= 0 or metrics["decode_tokens"] != max(metrics["generated_tokens"] - 1, 0):
        raise ValueError("invalid prompt/generated/decode token counts")
    if abs(metrics["total_ms"] - metrics["prefill_ms"] - metrics["decode_ms"]) > 0.002:
        raise ValueError("total time disagrees with prefill + decode")
    for count, duration, throughput in (("generated_tokens", "total_ms", "tokens_per_sec_total"),
                                        ("decode_tokens", "decode_ms", "tokens_per_sec_decode")):
        if metrics[count] and (metrics[duration] <= 0 or metrics[throughput] <= 0):
            raise ValueError(f"invalid duration/throughput for {count}")
        # Account for the CLI's independently rounded millisecond and rate fields.
        if metrics[duration] > 0.002:
            rate = metrics[count] * 1000 / metrics[duration]
            tolerance = 0.001 + rate * 0.001 / metrics[duration]
            if abs(rate - metrics[throughput]) > tolerance:
                raise ValueError(f"inconsistent throughput for {count}")
        if not metrics[count] and metrics[throughput] != 0:
            raise ValueError(f"nonzero throughput without {count}")
    return dict(metrics=metrics, weights=weights, cuda_counters={"total": counters} if counters else {},
                **weight_scalars, derived_metrics=derived)


def parse_ctest_junit(xml):
    root = ET.fromstring(xml)
    tests = []
    for node in root.iter("testcase"):
        output = node.findtext("system-out", "")
        if node.find("failure") is not None or node.find("error") is not None:
            status = "FAILED"
        elif node.find("skipped") is not None or "[  SKIPPED ]" in output:
            # GTest exits 0 for GTEST_SKIP; CTest may label that as passed.
            status = "NOT_RUN"
        else:
            status = "PASSED"
        reason = ""
        if status == "NOT_RUN":
            match = re.search(r"Skipped\r?\n([^\n]+)|skipped ['\"]([^\n]+)['\"]", output)
            reason = next((part for part in match.groups() if part), "") if match else "test skipped; see raw output"
        applicable = not (status == "NOT_RUN" and re.search(
            r"requires CPU-only build|CPU stub.*(?:CUDA=OFF|MINI_LLAMA_CUDA=OFF)|CPU stub only", output))
        tests.append(dict(name=node.get("name"), status=status, applicable=applicable, reason=reason,
                          duration_s=float(node.get("time", "0")), output=output))
    return dict(total=len(tests), passed=sum(t["status"] == "PASSED" for t in tests),
                failed=sum(t["status"] == "FAILED" for t in tests),
                not_run=sum(t["status"] == "NOT_RUN" for t in tests),
                required_not_run=sum(t["status"] == "NOT_RUN" and t["applicable"] for t in tests), tests=tests)


def run_ctests(build_dir, skip_tests=False, skip_cuda=False, cuda_available=False, timeout=1800):
    results = []
    for name, extra in (("cuda-targeted", ["-R", TARGETED]), ("full", [])):
        command = ["ctest", "--test-dir", str(build_dir), *extra, "--output-on-failure"]
        reason = "--skip-tests" if skip_tests else ""
        if name == "cuda-targeted" and not reason and (skip_cuda or not cuda_available):
            reason = "--skip-cuda" if skip_cuda else "CUDA Runtime/GPU unavailable"
        if reason:
            result = not_run(reason, command)
            result.update(total=None, passed=None, failed=None, not_run=None, tests=[])
        else:
            print(f"Running {name} CTest", flush=True)
            with tempfile.TemporaryDirectory(prefix="mini-llama-ctest-") as directory:
                junit = Path(directory) / "results.xml"
                result = run_command([*command, "--output-junit", junit], timeout)
                result.update(total=None, passed=None, failed=None, not_run=None, tests=[])
                try:
                    xml = junit.read_text()
                    result["junit_xml"] = xml
                    result.update(parse_ctest_junit(xml))
                    if result["status"] == "PASSED":
                        if result["total"] == 0 or result["failed"]:
                            result.update(status="FAILED", reason="no tests executed or failing JUnit results")
                        elif name == "cuda-targeted" and (result["required_not_run"] or not result["passed"]):
                            result.update(status="NOT_RUN", reason="targeted CUDA tests were skipped")
                except (OSError, ET.ParseError, ValueError) as exc:
                    if result["status"] == "PASSED":
                        result.update(status="FAILED", reason=f"CTest results unavailable: {exc}")
        result["name"] = name
        results.append(result)
    return results


def correctness_block(tests, backend, skip_tests):
    if skip_tests:
        return ""
    if any(test["status"] in ("FAILED", "TIMEOUT") for test in tests):
        return "correctness CTest failed; benchmarks blocked"
    if backend == "cuda" and any(test.get("required_not_run", test["not_run"]) or test["status"] != "PASSED"
                                 for test in tests if test["name"] == "cuda-targeted"):
        return "CUDA correctness tests incomplete; benchmarks blocked"
    full = next(test for test in tests if test["name"] == "full")
    if full["status"] != "PASSED" or not full["passed"]:
        return "full correctness CTest incomplete; benchmarks blocked"
    return ""


def validate_cuda(parsed, quant):
    if not all(key in parsed for key in ("uploaded_weights", "gpu_memory_used")):
        raise ValueError("missing CUDA upload/memory counters")
    if not parsed["uploaded_weights"] or not parsed["gpu_memory_used"]:
        raise ValueError("no CUDA weights uploaded")
    weights = parsed["weights"]
    if set(weights) != set(WEIGHTS):
        raise ValueError("missing CUDA Linear weight type counters")
    if any(w["uploaded"] != w["expected"] for w in weights.values()):
        raise ValueError("incomplete CUDA Linear weight coverage")
    if quant in ("q8_0", "q4_0") and weights[quant]["expected"] == 0:
        raise ValueError(f"no {quant} CUDA Linear weights")
    if quant == "q4_native" and not (weights["q4_0"]["expected"] + weights["q4_1"]["expected"]):
        raise ValueError("real Q4 model has no Q4 Linear weights")
    linears = sum(w["expected"] for w in weights.values())
    if quant == "q4_0" and weights["q4_0"]["expected"] != linears:
        raise ValueError("Q4_0 override did not cover every Linear weight")
    if linears < 8 or (linears - 1) % 7:
        raise ValueError("unexpected Linear topology")
    layers = (linears - 1) // 7
    counters = parsed["cuda_counters"].get("total", {})
    metrics = parsed["metrics"]
    forwards = metrics["prompt_tokens"] + metrics["decode_tokens"]
    required = set(COUNTERS) | {"host_to_device_bytes", "device_to_host_bytes"}
    if required - counters.keys():
        raise ValueError("missing cumulative CUDA counters")
    if counters["linear_calls"] != linears * forwards or counters["attention_calls"] != layers * forwards:
        raise ValueError("unexpected cumulative Linear/Attention calls")
    if counters["cpu_attention_fallback_calls"] or counters["host_to_device_copies"] or counters["host_to_device_bytes"]:
        raise ValueError("CPU attention fallback or H2D transfer detected")
    if counters["device_to_host_copies"] != forwards:
        raise ValueError("unexpected cumulative logits D2H copies")
    if any(counters[key] <= 0 for key in
           ("activation_calls", "kv_write_bytes", "kv_read_bytes", "device_to_host_bytes")):
        raise ValueError("missing activation/KV/logits work")
    # KV writes/logits downloads scale with forwards; attention reads sum the
    # history lengths 1..forwards. Precise sizes are checked by CUDA CTests.
    for key in ("kv_write_bytes", "device_to_host_bytes", "kv_read_bytes"):
        work = forwards * (forwards + 1) // 2 if key == "kv_read_bytes" else forwards
        unit, remainder = divmod(counters[key], work)
        if remainder or unit == 0:
            raise ValueError(f"inconsistent per-token/position CUDA bytes: {key}")


def run_benchmarks(cases, config, environment, tests):
    results = []
    inspections = {}
    for case in cases:
        command = [environment["binary"], "bench", case["model"], "--backend", case["backend"],
                   "-p", config["prompt"], "-n", str(config["n_predict"]), "--seed", str(config["seed"]),
                   "--threads", str(config["threads"])]
        if case["quant_override"]:
            command += ["--quant", case["quant_override"]]
        reason = correctness_block(tests, case["backend"], config["skip_tests"])
        if case["backend"] == "cuda" and (config["skip_cuda"] or not environment["cuda_available"]):
            reason = "--skip-cuda" if config["skip_cuda"] else "CUDA Runtime/GPU unavailable"
        if reason:
            result = not_run(reason, command)
        else:
            print(f"Running {case['name']}", flush=True)
            # Verify native quantization before attaching a Q8/Q4 case label.
            if case["quant"] in ("q8_0", "q4_native"):
                if case["model"] not in inspections:
                    inspections[case["model"]] = run_command(
                        [environment["binary"], "inspect", case["model"]], config["timeout"])
                inspection = inspections[case["model"]]
                formats = re.findall(
                    r"^tensor blk\.\d+\.(?:attn_[qkv]|attn_output|ffn_(?:gate|up|down))\.weight .* type=(\w+)",
                    inspection["stdout"], re.M)
                valid = (bool(formats) and all(q == "Q8_0" for q in formats)
                         if case["quant"] == "q8_0" else any(q in ("Q4_0", "Q4_1") for q in formats))
                if inspection["status"] != "PASSED" or not valid:
                    result = dict(inspection)
                    if inspection["status"] != "PASSED":
                        result["reason"] = "model inspection failed: " + inspection["reason"]
                    else:
                        result.update(status="FAILED", reason="native model quantization check failed")
                else:
                    result = run_command(command, config["timeout"])
                result["model_inspection"] = inspection
            else:
                result = run_command(command, config["timeout"])
        result.update(case, metrics=None, weights=None, cuda_counters=None,
                      speedup_decode=None, counter_validation="NOT_RUN")
        if result["status"] == "PASSED":
            try:
                if "status=error" in result["stdout"] + result["stderr"]:
                    raise ValueError("CLI reported status=error")
                parsed = parse_benchmark_stdout(result["stdout"])
                result.update(parsed)
                if parsed["metrics"]["generated_tokens"] > config["n_predict"]:
                    raise ValueError("generated token count exceeds request")
                if case["backend"] == "cuda":
                    result["counter_validation"] = "FAILED"
                    validate_cuda(parsed, case["quant"])
                    result["counter_validation"] = "PASSED"
            except ValueError as exc:
                result.update(status="FAILED", reason=str(exc))
        results.append(result)
    # Never compare failed cases or different effective token workloads.
    for cuda in (r for r in results if r["backend"] == "cuda"):
        cpu = next(r for r in results if r["group"] == cuda["group"] and r["backend"] == "cpu")
        cuda["comparison_reason"] = "CPU/CUDA pair not both PASSED"
        if cpu["status"] == cuda["status"] == "PASSED":
            if any(cpu["metrics"][key] != cuda["metrics"][key] for key in
                   ("prompt_tokens", "generated_tokens", "decode_tokens")):
                cuda["comparison_reason"] = "effective token workloads differ"
            elif cpu["metrics"]["tokens_per_sec_decode"] > 0:
                cuda["speedup_decode"] = (cuda["metrics"]["tokens_per_sec_decode"] /
                                          cpu["metrics"]["tokens_per_sec_decode"])
                cuda["comparison_reason"] = ""
            else:
                cuda["comparison_reason"] = "no decode workload"
    return results


def run_cuda_smokes(config, environment, tests):
    results = []
    for command_name in ("generate", "run"):
        command = [environment["binary"], command_name, "models/chat", "--backend", "cuda",
                   "-p", config["prompt"], "-n", "8", "--seed", str(config["seed"]),
                   "--threads", str(config["threads"])]
        reason = correctness_block(tests, "cuda", config["skip_tests"])
        if config["skip_cuda"] or not environment["cuda_available"]:
            reason = "--skip-cuda" if config["skip_cuda"] else "CUDA Runtime/GPU unavailable"
        if reason:
            result = not_run(reason, command)
        else:
            print(f"Running chat CUDA {command_name} smoke", flush=True)
            result = run_command(command, config["timeout"], input_text="/exit\n")
            logs = result["stdout"] + result["stderr"]
            # run handles per-turn exceptions and can return 0 after a failed turn.
            if result["status"] == "PASSED" and ("status=error" in logs or not re.search(
                    rf"mode={command_name} status=ok .*generated_tokens=[1-9]\d*", logs)):
                result.update(status="FAILED", reason="missing successful generation trace or failed chat turn")
        result["name"] = f"chat-cuda-{command_name}"
        results.append(result)
    return results


def display(value):
    if value is None:
        return "unavailable"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def cell(value):
    return display(value).replace("|", "\\|").replace("\n", "; ") or "—"


def format_markdown(markdown):
    """Wrap prose and space table delimiters; preserve logs and wide tables.

    Only blocks whose original layout requires an exception receive scoped
    lint directives. No rules are disabled for the surrounding document.
    """
    lines = markdown.splitlines()
    output = []
    index = 0
    while index < len(lines):
        line = lines[index]
        fence = re.match(r"^(\s*)(`{3,}|~{3,})", line)
        if fence or line.startswith("|"):
            block = [line]
            index += 1
            if fence:
                while index < len(lines):
                    block.append(lines[index])
                    index += 1
                    if block[-1].strip() == fence[2]:
                        break
            else:
                while index < len(lines) and lines[index].startswith("|"):
                    block.append(lines[index])
                    index += 1
                block = ["| " + " | ".join(part.strip() for part in row.strip("|").split("|")) + " |"
                         if re.fullmatch(r"\|[ :\-|]+\|", row) else row for row in block]
            rules = []
            if any(len(row) > 80 for row in block):
                rules.append("MD013")
            if fence and any("\t" in row for row in block):
                rules.append("MD010")
            if fence and any(row.endswith((" ", "\t")) for row in block):
                rules.append("MD009")
            indent = fence[1] if fence else ""
            if rules:
                output.append(indent + "<!-- markdownlint-disable " + " ".join(rules) + " -->")
            output.extend(block)
            if rules:
                output.append(indent + "<!-- markdownlint-enable " + " ".join(rules) + " -->")
            continue
        line = line.rstrip()
        if len(line) > 80:
            prefix = re.match(r"^\s*(?:[-*+]\s+)?", line)[0]
            output.extend(textwrap.wrap(line, width=80, subsequent_indent=" " * len(prefix),
                                        break_long_words=False, break_on_hyphens=False))
        else:
            output.append(line)
        index += 1
    return "\n".join(output).rstrip() + "\n"


def raw_log(result):
    content = shlex.join(result["command"]) + "\n" + result.get("stdout", "")
    if result.get("stderr"):
        content += "\n[stderr]\n" + result["stderr"]
    # Fence length must exceed any backtick sequence in the original log.
    fence = "`" * max(3, 1 + max((len(s) for s in re.findall(r"`+", content)), default=0))
    return f"{fence}text\n{content}\n{fence}\n"


def build_markdown(report):
    lines = ["# mini-llama CPU/CUDA Benchmark", "", f"Run status: **{report['status']}**", "",
             "NOT_RUN means no measurement. Missing metrics are unavailable; JSON uses null and CSV uses empty cells.",
             "", "## Environment", ""]
    for name, value in report["environment"].items():
        if name != "probes":
            if "\n" in display(value):
                lines += [f"- {name}:", "", "  ```text"]
                lines += ["  " + line if line else "" for line in display(value).splitlines()]
                lines += ["  ```", ""]
            else:
                lines.append(f"- {name}: {cell(value)}")
    lines += ["", "## Configuration and timing", "", "```json",
              json.dumps(report["config"], indent=2), "```", "",
              "Prefill measures prompt batch construction and ForwardBatch; Decode measures subsequent single-token forwards plus CPU sampling. "
              "The first sampled token belongs to generated_tokens, but not decode_tokens. Total = Prefill + Decode; "
              "model loading, weight upload, tokenization, context allocation and first-token sampling are excluded. "
              "Total throughput = generated_tokens / Total; Decode throughput = decode_tokens / Decode (seconds). "
              "Each case is one fresh process with greedy sampling, no warm-up or repetitions.", "",
              "## CTest correctness gate", "",
              "| Suite | Status | Total | Passed | Failed | NOT_RUN | Wall seconds | Reason |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for test in report["tests"]:
        lines.append("| " + " | ".join(cell(test.get(k)) for k in
                     ("name", "status", "total", "passed", "failed", "not_run", "duration_s", "reason")) + " |")
    skipped = [(suite["name"], test) for suite in report["tests"]
               for test in suite["tests"] if test["status"] == "NOT_RUN"]
    if skipped:
        lines += ["", "Individual tests not run:", "",
                  "| Suite | Test | Status | Reason |", "|---|---|---|---|"]
        for suite, test in skipped:
            lines.append("| " + " | ".join(map(cell, (suite, test["name"], test["status"], test["reason"]))) + " |")
    lines += ["", "Full-suite totals include skipped tests. GTest skips are counted as NOT_RUN even when CTest returns zero. "
              "CPU-only stub skips in a CUDA build are inapplicable and do not block the CUDA gate; "
              "skipped GPU tests do block the targeted gate. "
              "--skip-tests explicitly disables the gate; those runs are unverified performance samples.", "",
              "## Benchmark matrix and measured performance", "",
              "| Case | Model | Backend | Quant | Status | Prompt | Generated | Decode tokens | Prefill ms | Decode ms | Total ms | Total tok/s | Decode tok/s | CUDA/CPU Decode speedup | Reason |",
              "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for case in report["benchmarks"]:
        values = [case[k] for k in ("name", "model", "backend", "quant", "status")]
        values += [(case["metrics"] or {}).get(k) for k in METRICS]
        values += [case["speedup_decode"], case["reason"] or case.get("comparison_reason", "")]
        lines.append("| " + " | ".join(map(cell, values)) + " |")
    lines += ["", "## CUDA runtime counters", "",
              "Uploaded weights and gpu memory used describe model weights only (not process peak VRAM, KV or activations). "
              "The CLI resets model runtime counters after upload. Counters cover Prefill + Decode cumulatively; "
              "total is sampled before quantization-error reference forwards. Counters cover instrumented forward calls, "
              "not all CUDA Runtime API operations; allocations and D2D copies are excluded. "
              "Zero cumulative H2D implies zero Decode H2D; nonzero totals cannot locate the phase. "
              "D2H logits for CPU sampling are expected once per prompt/decode forward.", ""]
    for case in report["benchmarks"]:
        if case["backend"] != "cuda":
            continue
        lines += [f"### {case['name']}: {case['status']}", "",
                  f"Counter validation: {case['counter_validation']}; {case['reason']}", "", "```json",
                  json.dumps({k: case.get(k) for k in ("uploaded_weights", "gpu_memory_used", "weights", "cuda_counters")}, indent=2), "```", ""]
    lines += ["## Remote functional checks", ""]
    for result in report["smokes"]:
        lines.append(f"- {result['name']}: {result['status']} — {result['reason']}")
    lines += ["", "Deployment commands, acceptance criteria and NOT_RUN result template: "
              "[remote CUDA validation](../../docs/cuda_benchmark_validation.md).", "",
              "## Raw outputs and failures", ""]
    all_results = list(report["environment"]["probes"].items())
    inspected = set()
    for case in report["benchmarks"]:
        if "model_inspection" in case and case["model"] not in inspected:
            all_results.append(("inspect " + case["model"], case["model_inspection"]))
            inspected.add(case["model"])
    all_results += [(r["name"], r) for r in report["tests"] + report["benchmarks"] + report["smokes"]]
    for name, result in all_results:
        status = f"Status: {result['status']}"
        if result["reason"]:
            status += "; " + result["reason"]
        lines += [f"### {name}", "", status, "", raw_log(result)]
    return format_markdown("\n".join(lines))


def write_json(path, report):
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, report):
    fields = ["name", "model", "backend", "quant", "quant_override", "status", "reason",
              "prompt", "n_predict", "seed", "threads", *METRICS, "speedup_decode",
              "duration_s", "counter_validation", "uploaded_weights", "gpu_memory_used"]
    fields += [f"{quant}_{key}" for quant in WEIGHTS for key in ("uploaded", "expected")]
    runtime_keys = list(COUNTERS) + ["host_to_device_bytes", "device_to_host_bytes"]
    fields += [f"cuda_{phase}_{key}" for phase in ("total",) for key in runtime_keys]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fields)
        writer.writeheader()
        for case in report["benchmarks"]:
            row = {key: case.get(key) for key in fields}
            row.update({key: report["config"][key] for key in ("prompt", "n_predict", "seed", "threads")})
            row.update(case["metrics"] or {})
            for quant, weights in (case["weights"] or {}).items():
                row.update({f"{quant}_{key}": value for key, value in weights.items()})
            for phase, counters in (case["cuda_counters"] or {}).items():
                row.update({f"cuda_{phase}_{key}": value for key, value in counters.items()})
            writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-cuda")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/benchmarks/cuda-benchmark")
    parser.add_argument("--q4-model", type=Path)
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--fast", action="store_true", help="4 generated tokens instead of 32; preserves the full matrix")
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4, help="fixed CPU thread configuration for every case")
    parser.add_argument("--n-predict", type=int)
    parser.add_argument("--timeout", type=float, default=900, help="per benchmark/smoke timeout, seconds")
    parser.add_argument("--test-timeout", type=float, default=1800, help="per CTest suite timeout, seconds")
    args = parser.parse_args(argv)
    n_predict = args.n_predict if args.n_predict is not None else 4 if args.fast else 32
    if n_predict < 0 or not 0 <= args.seed <= 4294967295 or args.threads < 0:
        parser.error("invalid token budget, unsigned seed or thread count")
    if not all(math.isfinite(t) and t > 0 for t in (args.timeout, args.test_timeout)):
        parser.error("timeouts must be finite and positive")
    build_dir = args.build_dir.resolve()
    out = args.out.resolve()
    if out.suffix.lower() in (".md", ".json", ".csv"):
        out = out.with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)
    config = dict(prompt=args.prompt, n_predict=n_predict, seed=args.seed, threads=args.threads,
                  skip_cuda=args.skip_cuda, skip_tests=args.skip_tests, fast=args.fast,
                  timeout=args.timeout, test_timeout=args.test_timeout,
                  q4_model=str(args.q4_model) if args.q4_model else None)
    environment = collect_environment(build_dir)
    tests = run_ctests(build_dir, args.skip_tests, args.skip_cuda,
                       environment["cuda_available"], args.test_timeout)
    cases = run_benchmarks(make_cases(args.q4_model), config, environment, tests)
    smokes = run_cuda_smokes(config, environment, tests)
    failed = any(r["status"] in ("FAILED", "TIMEOUT") for r in tests + cases + smokes)
    status = "FAILED" if failed else "PARTIAL" if any(r["status"] == "NOT_RUN" for r in tests + cases + smokes) else "PASSED"
    report = dict(schema_version=1, status=status, environment=environment, config=config,
                  tests=tests, benchmarks=cases, smokes=smokes)
    paths = {extension: Path(str(out) + "." + extension) for extension in ("md", "json", "csv")}
    paths["md"].write_text(build_markdown(report))
    write_json(paths["json"], report)
    write_csv(paths["csv"], report)
    for path in paths.values():
        print(f"Wrote {path}")
    print(f"Run status: {status}")
    # Explicit skips allow a successful local smoke invocation; unexpected GPU
    # unavailability/incomplete correctness gates must not look like a CI pass.
    incomplete = not args.skip_cuda and any(r["status"] == "NOT_RUN" for r in cases + smokes)
    return 1 if failed else 2 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
