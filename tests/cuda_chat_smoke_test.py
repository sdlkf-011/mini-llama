#!/usr/bin/env python3
"""Real-model CUDA generation smoke; exit 77 means prerequisites unavailable.

The model/tokenizer is selected by the CLI, never downloaded by this test.
CPU and CUDA prefill logits use the same input; later CUDA logits are checked
for finiteness without assuming sampled autoregressive histories remain equal.
"""

import math
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("MINI_LLAMA_BIN", ROOT / "build-cuda/mini-llama")).resolve()
MODEL = Path(os.environ.get("MINI_LLAMA_CHAT_MODEL", ROOT / "models/chat")).resolve()


def invoke(*args):
    return subprocess.run([str(BIN), *map(str, args)], cwd=ROOT,
                          capture_output=True, text=True, timeout=240)


class CudaChatSmokeTest(unittest.TestCase):
    def success(self, *args):
        result = invoke(*args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("status=error", result.stdout + result.stderr)
        return result

    def number(self, output, label):
        match = re.search(r"^\s*" + re.escape(label) + r": (\d+)$",
                          output, re.MULTILINE)
        self.assertIsNotNone(match, f"missing {label}:\n{output}")
        return int(match.group(1))

    def setUp(self):
        if not BIN.is_file():
            self.fail(f"mini-llama binary does not exist: {BIN}")
        diagnostics = invoke("--cuda-info")
        if "CUDA backend built: no" in diagnostics.stdout:
            self.skipTest("NOT RUN / PENDING GPU VALIDATION: CUDA was not built")
        self.assertIn("CUDA backend built: yes", diagnostics.stdout)
        count = re.search(r"^CUDA GPU count: (\d+)$", diagnostics.stdout, re.MULTILINE)
        if diagnostics.returncode or not count or int(count.group(1)) == 0:
            self.skipTest("NOT RUN / PENDING GPU VALIDATION: GPU unavailable: " +
                          diagnostics.stdout + diagnostics.stderr)
        if not MODEL.exists() or (MODEL.is_dir() and
                                 not any(MODEL.glob("*.gguf")) and
                                 not (MODEL / "model.bin").is_file()):
            self.skipTest(f"NOT RUN: test model unavailable: {MODEL}")
        # Present but malformed weights/tokenizer/config must fail, not skip.
        self.metadata = self.success("inspect", MODEL).stdout

    def test_generate_and_prefill_cpu_baseline(self):
        vocab = self.number(self.metadata, "vocab_size")
        layers = self.number(self.metadata, "n_layers")
        heads = self.number(self.metadata, "n_heads")
        kv_heads = self.number(self.metadata, "n_kv_heads")
        head_dim = self.number(self.metadata, "head_dim")
        with tempfile.TemporaryDirectory() as directory:
            cpu_dir, cuda_dir = (Path(directory) / name for name in ("cpu", "cuda"))
            self.success("generate", "--model", MODEL, "--backend", "cpu",
                         "-p", "hello", "-n", "1", "--seed", "1",
                         "--dump-logits", cpu_dir)
            result = self.success("generate", "--model", MODEL, "--backend", "cuda",
                                  "-p", "hello", "-n", "8", "--seed", "1",
                                  "--dump-logits", cuda_dir)
            self.assertIn("backend: cuda", result.stdout)
            self.assertIn('generated text: "', result.stdout)
            self.assertNotIn("\x00", result.stdout)
            match = re.search(r"^generated tokens: \[([^\]]*)\]$",
                              result.stdout, re.MULTILINE)
            self.assertIsNotNone(match, result.stdout)
            tokens = [int(token.strip()) for token in match.group(1).split(",")]
            self.assertTrue(1 <= len(tokens) <= 8, tokens)
            self.assertTrue(all(0 <= token < vocab for token in tokens), tokens)
            trace = re.search(r"^trace summary: .* mode=generate status=ok .*"
                              r"prefill_tokens=(\d+) decode_tokens=(\d+) "
                              r"generated_tokens=(\d+)", result.stderr, re.MULTILINE)
            self.assertIsNotNone(trace, result.stderr)
            prefill, decode, generated = map(int, trace.groups())
            self.assertEqual(generated, len(tokens))
            self.assertEqual(decode, generated - 1)
            self.assertGreater(prefill, 0)
            forwards = prefill + decode
            self.assertEqual(self.number(result.stdout, "cuda Linear calls"),
                             (7 * layers + 1) * forwards)
            self.assertGreater(self.number(result.stdout, "cuda activation calls"), 0)
            self.assertEqual(self.number(result.stdout, "cuda attention calls"),
                             layers * forwards)
            self.assertEqual(self.number(result.stdout, "cpu attention fallback calls"), 0)
            for label, copies, size in (("host->device copies", 0, 0),
                                        ("device->host copies", forwards,
                                         forwards * vocab * 4)):
                self.assertIn(f"{label}: {copies} ({size} bytes)", result.stdout)
            self.assertEqual(self.number(result.stdout, "cuda kv write bytes"),
                             2 * layers * kv_heads * head_dim * 4 * forwards)
            self.assertEqual(self.number(result.stdout, "cuda kv read bytes"),
                             2 * layers * heads * head_dim * 4 *
                             forwards * (forwards + 1) // 2)
            weights = re.findall(r"^cuda (f32|q8_0|q4_0|q4_1) Linear weights: "
                                 r"(\d+)/(\d+)$", result.stdout, re.MULTILINE)
            self.assertTrue(weights, result.stdout)
            self.assertTrue(all(uploaded == expected for _, uploaded, expected in weights))
            self.assertEqual(sum(int(uploaded) for _, uploaded, _ in weights), 7 * layers + 1)
            self.assertNotIn("weights missing:", result.stdout)
            cpu_files = list(cpu_dir.glob("logits_step*.bin"))
            cuda_files = list(cuda_dir.glob("logits_step*.bin"))
            self.assertEqual(len(cpu_files), prefill)
            self.assertEqual(len(cuda_files), forwards)
            for path in cuda_files:
                data = path.read_bytes()
                self.assertEqual(len(data), vocab * 4, path.name)
                actual = struct.unpack(f"={vocab}f", data)
                self.assertTrue(all(math.isfinite(x) for x in actual), path.name)
                cpu_path = cpu_dir / path.name
                if cpu_path.is_file():
                    cpu_data = cpu_path.read_bytes()
                    self.assertEqual(len(cpu_data), len(data), path.name)
                    expected = struct.unpack(f"={vocab}f", cpu_data)
                    for i, (cpu, cuda) in enumerate(zip(expected, actual)):
                        abs_err = abs(cuda - cpu)
                        rel_err = abs_err / max(1.0, abs(cpu))
                        self.assertTrue(math.isfinite(cpu) and
                                        (abs_err <= 2e-3 or rel_err <= 2e-3),
                                        f"{path.name} logits index={i} CPU expected={cpu} "
                                        f"CUDA actual={cuda} abs_error={abs_err} "
                                        f"relative_error={rel_err} abs_tol=0.002 rel_tol=0.002")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CudaChatSmokeTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(1 if not result.wasSuccessful() else 77 if result.skipped else 0)
