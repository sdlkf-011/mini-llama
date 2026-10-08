#!/usr/bin/env python3
"""Backend CLI smoke tests using only the Python standard library.

CUDA execution tests are explicitly skipped without a CUDA build/usable device.
On a CUDA build with a visible GPU, initialization/inference failures fail tests.
CUDA Runtime and CUDA inference integration are checked separately.
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
MODEL = ROOT / "models/tiny"
BIN = Path(os.environ.get("MINI_LLAMA_BIN", ROOT / "build/mini-llama")).resolve()


def invoke(*args, text="", env=None):
    environment = os.environ.copy()
    if env:
        environment.update(env)
    return subprocess.run([str(BIN), *map(str, args)], cwd=ROOT, input=text,
                          capture_output=True, text=True, timeout=30, env=environment)


class BackendCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BIN.is_file():
            raise RuntimeError(f"mini-llama binary does not exist: {BIN}")
        cls.diagnostics = invoke("--cuda-info")
        output = cls.diagnostics.stdout
        if "CUDA backend built: yes" in output:
            cls.cuda_built = True
        elif "CUDA backend built: no" in output:
            cls.cuda_built = False
        else:
            raise AssertionError(f"missing build status in --cuda-info: {cls.diagnostics}")
        count = re.search(r"^CUDA GPU count: (\d+)$", output, re.MULTILINE)
        cls.gpu_count = int(count.group(1)) if count else 0

    def success(self, *args, **kwargs):
        result = invoke(*args, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def failure(self, *args, contains, **kwargs):
        result = invoke(*args, **kwargs)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(contains, result.stderr)
        self.assertNotIn("status=ok", result.stderr)
        self.assertNotIn("generated tokens:", result.stdout)
        return result

    def require_gpu(self):
        if not self.cuda_built:
            self.skipTest("CUDA backend was not built; GPU inference was not executed")
        if self.diagnostics.returncode or self.gpu_count == 0:
            self.skipTest("CUDA device unavailable; GPU inference was not executed: "
                          + (self.diagnostics.stderr.strip() or "GPU count is zero"))
        probe = invoke("generate", "--model", "missing-model", "--backend", "cuda")
        if "CUDA inference is not integrated yet" in probe.stderr:
            self.skipTest("CUDA Runtime is available; CUDA inference was not integrated "
                          "or executed")

    def check_cpu(self, result):
        self.assertIn("backend: cpu\ncompute: cpu\n", result.stdout)
        self.assertNotIn("CUDA Linear calls:", result.stdout)
        self.assertNotIn("GPU 0:", result.stderr)

    def check_gpu(self, result):
        self.assertIn("backend: cuda\n", result.stdout)
        compute = re.search(r"^compute: (.+)$", result.stdout, re.MULTILINE)
        self.assertIsNotNone(compute, result.stdout)
        self.assertIn("cuda", compute.group(1).lower())
        self.assertIn("backend=cuda", result.stdout + result.stderr)

    def test_default_cpu_without_visible_devices(self):
        result = self.success("generate", "--n-predict", "4", "--threads", "1",
                              env={"CUDA_VISIBLE_DEVICES": "-1"})
        self.check_cpu(result)

    def test_generate_cpu_with_original_resource_arguments(self):
        result = self.success("generate", "--model", MODEL / "model.bin",
                              "--config", MODEL / "model.json", "--tokenizer",
                              MODEL / "vocab.json", "--prompt", "hello", "--backend",
                              "cpu", "--n-predict", "4", "--threads", "1")
        self.check_cpu(result)
        self.assertIn("generated tokens:", result.stdout)

    def test_bench_cpu(self):
        result = self.success("bench", MODEL, "--backend", "cpu", "-n", "4",
                              "--threads", "1", "--verbose")
        self.check_cpu(result)
        self.assertIn("tokens/s (Decode):", result.stdout)
        self.assertIn("decode tokens: 3", result.stdout)

    def test_run_cpu_and_terminal_compatibility(self):
        result = self.success("run", MODEL, "--backend", "cpu", "--threads", "1",
                              "-n", "2", text="hello\n/params\n/clear\n/exit\n")
        self.check_cpu(result)
        self.assertIn("Session cleared.", result.stdout)
        self.assertIn("Bye.", result.stdout)

    def test_backend_errors_precede_model_loading_for_every_command(self):
        for command in ("generate", "run", "bench", "inspect"):
            for arguments, diagnostic in (
                (("--backend", "metal"), "invalid --backend"),
                (("--device", "-1"), "invalid --device"),
                (("--device", "abc"), "invalid --device"),
                (("--device", "2147483648"), "invalid --device"),
                (("--device", "999999999999999999999999999999"), "invalid --device"),
                (("--device", "1x"), "invalid --device"),
                (("--device", "+1"), "invalid --device"),
                (("--device", " 1"), "invalid --device"),
                (("--backend", "cpu", "--device", "0"), "--device can only be used"),
                (("--device", "0", "--backend", "cpu"), "--device can only be used"),
                (("--device", "0"), "--device can only be used"),
                (("--backend",), "missing value for --backend"),
                (("--device",), "missing value for --device"),
                (("--backend", "--device", "0"), "missing value for --backend"),
                (("--device", "--backend", "cuda"), "missing value for --device"),
                (("--backend", "cpu", "--backend", "cuda"), "duplicate option"),
                (("--backend", "cpu", "--backend", "cpu"), "duplicate option"),
                (("--device", "0", "--device", "1"), "duplicate option"),
                (("--device", "0", "--device", "0"), "duplicate option"),
            ):
                with self.subTest(command=command, arguments=arguments):
                    result = self.failure(command, "--model", "missing-model", *arguments,
                                          contains=diagnostic)
                    self.assertNotIn("trace summary:", result.stderr)

    def test_inspect_backend_is_metadata_only(self):
        # No weights, vocabulary, driver, or even a built CUDA backend are needed.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "model.json").write_bytes((MODEL / "model.json").read_bytes())
            for backend in ("cpu", "cuda"):
                args = ["inspect", path, "--backend", backend]
                if backend == "cuda":
                    args += ["--device", "2147483647"]
                result = self.success(*args, env={"CUDA_VISIBLE_DEVICES": "-1"})
                self.assertIn("compute: none (metadata only;", result.stdout)
                self.assertIn("vocab_size:", result.stdout)
                self.assertNotIn("GPU ", result.stdout + result.stderr)

    def test_cuda_info(self):
        result = self.diagnostics
        if not self.cuda_built:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("CUDA info failed: CUDA backend was not built.", result.stderr)
            self.assertIn("MINI_LLAMA_CUDA=ON", result.stderr)
        elif result.returncode:
            self.assertIn("CUDA info failed:", result.stderr)
        else:
            self.assertIn("CUDA GPU count:", result.stdout)
            if self.gpu_count:
                self.assertIn("Driver Version (CUDA API):", result.stdout)
                self.assertIn("CUDA Runtime Version:", result.stdout)
            for device in range(self.gpu_count):
                self.assertRegex(result.stdout, rf"GPU {device}: .+; Compute Capability: "
                                 r"\d+\.\d+; memory bytes: [1-9]\d*")
        self.failure("--cuda-info", "--backend", "cpu", contains="--cuda-info")

    def test_cuda_request_in_cpu_build(self):
        if self.cuda_built:
            self.skipTest("requires a CPU-only build")
        for command in ("generate", "run", "bench"):
            result = self.failure(command, "--model", "missing-model", "--backend",
                                  "cuda", contains="MINI_LLAMA_CUDA=ON")
            self.assertNotIn("trace summary:", result.stderr)

    def test_cuda_build_with_no_visible_device(self):
        if not self.cuda_built:
            self.skipTest("requires a CUDA build; device-unavailable path was not executed")
        for command in ("generate", "run", "bench"):
            result = self.failure(command, "--model", "missing-model", "--backend",
                                  "cuda", contains="",
                                  env={"CUDA_VISIBLE_DEVICES": "-1"})
            self.assertTrue(result.stderr.strip(), "missing CUDA failure reason")
            self.assertNotIn("trace summary:", result.stderr)

    def test_original_commands_and_aliases(self):
        default = self.success("generate", MODEL, "-p", "hello", "-n", "4", "--threads", "1")
        legacy = self.success("--config", MODEL / "model.json", "--weights",
                              MODEL / "model.bin", "--vocab", MODEL / "vocab.json",
                              "-p", "hello", "--max-tokens", "4", "--threads", "1")
        self.assertEqual(re.search(r"generated tokens: (.+)", default.stdout).group(1),
                         re.search(r"generated tokens: (.+)", legacy.stdout).group(1))
        self.success("generate", "--model", MODEL / "model.gguf", "-n", "0",
                     "--threads", "1")
        self.assertIn("GGUF version=", self.success("inspect-gguf", MODEL / "model.gguf").stdout)
        for command in ("generate", "run", "bench", "inspect"):
            self.assertIn("--backend", self.success(command, "--help").stdout)

    def test_cuda_generate_logits_match_cpu(self):
        self.require_gpu()
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / backend for backend in ("cpu", "cuda")]
            for backend, path in zip(("cpu", "cuda"), paths):
                args = ["generate", MODEL, "--backend", backend, "--threads", "1",
                        "-p", "hello", "-n", "4", "--dump-logits", path]
                if backend == "cuda":
                    # Device-before-backend must also work.
                    args[2:2] = ["--device", "0"]
                result = self.success(*args)
                if backend == "cuda":
                    self.check_gpu(result)
            cpu_files = sorted(paths[0].glob("logits_step*.bin"))
            self.assertGreater(len(cpu_files), 1, "must execute prefill and decode")
            self.assertEqual([p.name for p in cpu_files],
                             sorted(p.name for p in paths[1].glob("logits_step*.bin")))
            for cpu_file in cpu_files:
                cpu_data = cpu_file.read_bytes()
                gpu_data = (paths[1] / cpu_file.name).read_bytes()
                self.assertEqual(len(cpu_data), len(gpu_data))
                cpu = struct.unpack(f"<{len(cpu_data) // 4}f", cpu_data)
                gpu = struct.unpack(f"<{len(gpu_data) // 4}f", gpu_data)
                for expected, actual in zip(cpu, gpu):
                    self.assertTrue(math.isfinite(actual))
                    self.assertLessEqual(abs(actual - expected), 2e-4 + 2e-4 * abs(expected))
            self.assertEqual((paths[0] / "generation_tokens.txt").read_text(),
                             (paths[1] / "generation_tokens.txt").read_text())

    def test_cuda_bench_and_run(self):
        self.require_gpu()
        self.check_gpu(self.success("bench", MODEL, "--backend", "cuda", "--device", "0",
                                    "-n", "4", "--threads", "1"))
        self.check_gpu(self.success("run", MODEL, "--backend", "cuda", "--device", "0",
                                    "-n", "2", "--threads", "1", "--prompt", "hello",
                                    text="/clear\nhello\n/exit\n"))

    def test_cuda_invalid_selected_device(self):
        self.require_gpu()
        self.failure("generate", "--backend", "cuda", "--device", "2147483647",
                     contains="out of range")


if __name__ == "__main__":
    unittest.main(verbosity=2)
