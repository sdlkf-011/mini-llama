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
import sys
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
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
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

    def check_chat(self, result, turns):
        # 聊天允许单轮失败后继续交互，退出码为零不能证明每轮推理成功。
        self.assertNotIn("Error:", result.stdout)
        self.assertNotIn("status=error", result.stdout + result.stderr)
        summaries = re.findall(r"^trace summary: .* mode=run status=ok .*"
                               r"generated_tokens=([0-9]+)",
                               result.stdout, re.MULTILINE)
        self.assertEqual(len(summaries), turns, result.stdout)
        self.assertTrue(all(int(count) > 0 for count in summaries), result.stdout)

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

    def test_bench_default_budget_matches_help(self):
        help_output = self.success("bench", "--help").stdout
        self.assertIn("default: 16; 0 = prefill only", help_output)
        bench = self.success("bench", MODEL, "--threads", "1")
        self.assertIn("generated tokens: 16\n", bench.stdout)
        self.assertIn("decode tokens: 15\n", bench.stdout)

    def test_run_cpu_and_terminal_compatibility(self):
        result = self.success("run", MODEL, "--backend", "cpu", "--threads", "1",
                              "-n", "2", text="hello\n/params\n/clear\n/exit\n")
        self.check_cpu(result)
        self.check_chat(result, 1)
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
        for command in ("generate", "run", "bench"):
            self.assertIn("--quant", self.success(command, "--help").stdout)

    def check_resident_counters(self, result):
        # CLI 按指定输出口径累计 Prefill + Decode，与 trace 中真实 Forward 次数对照。
        summary = re.search(r"trace summary: .* prefill_tokens=([0-9]+) "
                            r"decode_tokens=([0-9]+)", result.stderr)
        self.assertIsNotNone(summary, result.stderr)
        forwards = int(summary.group(1)) + int(summary.group(2))
        fields = {label: int(re.search(re.escape(label) + r": ([0-9]+)",
                                       result.stdout).group(1))
                  for label in ("cuda Linear calls", "cuda activation calls",
                                "cuda attention calls", "cpu attention fallback calls")}
        self.assertEqual(fields["cuda Linear calls"], forwards * 15)
        self.assertEqual(fields["cuda activation calls"], forwards * 15)
        self.assertEqual(fields["cuda attention calls"], forwards * 2)
        self.assertEqual(fields["cpu attention fallback calls"], 0)
        self.assertIn("host->device copies: 0 (0 bytes)", result.stdout)
        self.assertIn(f"device->host copies: {forwards} ({forwards * 512} bytes)",
                      result.stdout)

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
                    self.check_resident_counters(result)
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
                for index, (expected, actual) in enumerate(zip(cpu, gpu)):
                    abs_err = abs(actual - expected)
                    rel_err = abs_err / max(1.0, abs(expected))
                    diagnostic = (f"{cpu_file.name} logits index={index} CPU expected={expected} "
                                  f"CUDA actual={actual} abs_error={abs_err} relative_error={rel_err} "
                                  "threshold=2e-4 + 2e-4 * abs(CPU)")
                    self.assertTrue(math.isfinite(expected) and math.isfinite(actual), diagnostic)
                    self.assertLessEqual(abs_err, 2e-4 + 2e-4 * abs(expected), diagnostic)
            self.assertEqual((paths[0] / "generation_tokens.txt").read_text(),
                             (paths[1] / "generation_tokens.txt").read_text())

    def test_cuda_bench_and_run(self):
        self.require_gpu()
        bench = self.success("bench", MODEL, "--backend", "cuda", "--device", "0",
                             "-n", "4", "--threads", "1")
        self.check_gpu(bench)
        self.check_resident_counters(bench)
        chat = self.success("run", MODEL, "--backend", "cuda", "--device", "0",
                            "-n", "2", "--threads", "1", "--prompt", "hello",
                            text="again\n/clear\nhello\n/exit\n")
        self.check_gpu(chat)
        self.check_chat(chat, 3)
        self.assertIn("Session cleared.", chat.stdout)
        self.assertNotRegex(chat.stdout, r"cpu attention fallback calls: [1-9]")

    def test_quant_overrides_generate_and_bench(self):
        for quant in ("q8_0", "q4_0"):
            self.success("generate", MODEL, "--quant", quant, "-n", "2", "--seed", "1")
            bench = self.success("bench", MODEL, "--quant", quant, "-n", "2", "--seed", "1")
            self.assertIn("logits error vs model-native:", bench.stdout)
            chat = self.success("run", MODEL, "--quant", quant, "-n", "2",
                                "--seed", "1", text="hello\n/exit\n")
            self.check_cpu(chat)
            self.check_chat(chat, 1)
        self.failure("generate", "--model", "missing-model", "--quant", "unknown",
                     contains="unsupported quant type")

    def test_cuda_quant_overrides_generate_and_bench(self):
        self.require_gpu()
        for quant in ("q8_0", "q4_0"):
            result = self.success("generate", MODEL, "--backend", "cuda",
                                  "--quant", quant, "-n", "2", "--seed", "1")
            self.check_resident_counters(result)
            self.assertIn(f"cuda {quant} Linear weights: 15/15", result.stdout)
            self.assertIn("cuda f32 Linear weights: 0/0", result.stdout)
            bench = self.success("bench", MODEL, "--backend", "cuda",
                                 "--quant", quant, "-n", "2", "--seed", "1")
            self.check_gpu(bench)
            self.check_resident_counters(bench)
            self.assertIn("logits error vs model-native:", bench.stdout)
            chat = self.success("run", MODEL, "--backend", "cuda",
                                "--quant", quant, "-n", "2", "--seed", "1",
                                text="hello\n/exit\n")
            self.check_gpu(chat)
            self.check_chat(chat, 1)

    def test_cuda_invalid_selected_device(self):
        self.require_gpu()
        self.failure("generate", "--backend", "cuda", "--device", "2147483647",
                     contains="out of range")


@unittest.skipUnless(os.name == "posix", "CUDA ASan entry point requires Bash")
class CudaAsanScriptTest(unittest.TestCase):
    """Exercise the shell acceptance gate without a CUDA SDK or real GPU."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="mini-llama-asan-gate-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.tools = self.directory / "tools"
        self.build = self.directory / "build"
        self.tools.mkdir()
        self.build.mkdir()
        self.write_program(self.tools / "cmake", "raise SystemExit(0)")
        self.write_program(self.build / "mini-llama", '''
import os
print("CUDA backend built: yes")
count = os.environ.get("TEST_GPU_COUNT", "1")
if count != "missing":
    print("CUDA GPU count: " + count)
raise SystemExit(int(os.environ.get("TEST_INFO_EXIT", "0")))
''')
        self.write_program(self.tools / "ctest", '''
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
Path(os.environ["TEST_CTEST_CALLED"]).touch()
names = [
    "cuda_runtime.CudaRuntimeGpuTest.RawHostDeviceAndDeviceDeviceCopiesRoundTrip",
    "CudaTensorGpuTest.RoundTripAndMovesPreserveTensorState",
    "cuda_matmul.NonSquareRowMajorMapping",
    "cuda_linear.HostVectorAndBatchWithOptionalBias",
    "cuda_ops.CudaOpsGpu.HostAndDeviceOpsMatchCpu",
    "cuda_quant_q8_0.LargeBatchAllEntrypointsMatchCpu",
    "cuda_quant_q4.LargeBatchQ40AllEntrypointsMatchCpu",
    "cuda_kv_cache.CudaKvCacheGpuTest.LayoutHostDeviceReadAndClearMatchCpu",
    "cuda_attention.CudaAttentionGpuTest.HostAndDeviceGqaMatchIndependentReference",
    "cuda_forward.CudaForwardTest.SingleAndMultipleTokensDownloadOnly128Logits",
    "cuda_forward.CudaForwardAttentionTest.CoversAllLayersGqaAndOptionalBias",
    "cuda_forward_device_resident.CudaForwardBufferTest.FlatAndHeadShapedDeviceWritesMatch",
    "cuda_forward_linear.CudaForwardLinearTest.ResidentF32ProjectionsMatchCpuPrefillAndDecode",
    "cuda_forward_linear.cuda_forward_quant.Q80ProjectionsRemainResident",
    "cuda_forward_linear.cuda_forward_q4.Q40ProjectionsRemainResident",
    "cuda_runtime.CudaRuntimeGpuTest.BuffersRetainTheirDeviceAcrossOtherDeviceOperations",
    "cuda_quant_q8_0_stub.AllEntrypointsRejectCpuOnlyBuild",
    "cuda-chat-smoke",
]
mode = os.environ.get("TEST_REPORT_MODE", "passed")
if "--show-only=json-v1" in sys.argv:
    print(json.dumps({"tests": [{"name": name} for name in names]}))
    raise SystemExit(0)
if "--output-junit" in sys.argv and mode != "absent":
    output = Path(sys.argv[sys.argv.index("--output-junit") + 1])
    if mode == "malformed":
        output.write_text("not XML")
    else:
        suite = ET.Element("testsuite")
        for i, name in enumerate(names):
            if mode == "missing" and i == 4 or mode == "missing-case" and i == 15:
                continue
            case = ET.SubElement(suite, "testcase", name=name, status="run")
            if (mode == "all-skipped" or i >= 16
                    or i == 15 and mode != "multi-passed"
                    or mode == "one-skipped" and i == 3):
                case.set("status", "notrun")
                ET.SubElement(case, "skipped")
        ET.ElementTree(suite).write(output)
print("100% tests passed, 0 tests failed")
raise SystemExit(int(os.environ.get("TEST_CTEST_EXIT", "0")))
''')

    def write_program(self, path, body):
        path.write_text(f"#!{sys.executable}\n" + body + "\n")
        path.chmod(0o755)

    def run_gate(self, **overrides):
        env = {**os.environ, "PATH": str(self.tools) + os.pathsep + os.environ["PATH"],
               "TEST_CTEST_CALLED": str(self.directory / "ctest-called"), **overrides}
        return subprocess.run(["bash", str(ROOT / "scripts/test_cuda_asan.sh"),
                               "--build-dir", str(self.build)], env=env,
                              capture_output=True, text=True, timeout=30)

    # CUDA 已编译但没有可见设备时，必须在 CTest 之前返回未执行状态。
    def test_zero_devices_is_not_validation(self):
        result = self.run_gate(TEST_GPU_COUNT="0", CUDA_VISIBLE_DEVICES="-1")
        self.assertEqual(result.returncode, 77, result.stdout + result.stderr)
        self.assertFalse((self.directory / "ctest-called").exists())

    # 诊断缺少设备数或命令失败时，不得继续验收测试。
    def test_bad_device_diagnostics_fail(self):
        for options in ({"TEST_GPU_COUNT": "missing"}, {"TEST_INFO_EXIT": "3"}):
            with self.subTest(options=options):
                result = self.run_gate(**options)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.directory / "ctest-called").exists())

    # CTest 退出零但全部或部分必需 GPU 用例被跳过时，仍返回未执行状态。
    def test_required_gpu_skips_are_not_validation(self):
        for mode in ("all-skipped", "one-skipped"):
            with self.subTest(mode=mode):
                result = self.run_gate(TEST_REPORT_MODE=mode)
                self.assertEqual(result.returncode, 77, result.stdout + result.stderr)

    # 未生成报告、报告损坏或缺失必需 GPU 模块都不能得到验收成功。
    def test_incomplete_reports_fail(self):
        for mode in ("absent", "malformed", "missing"):
            with self.subTest(mode=mode):
                result = self.run_gate(TEST_REPORT_MODE=mode)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotEqual(result.returncode, 77)

    # 单 GPU 可以跳过多设备用例、CPU stub 和依赖外部模型的聊天冒烟测试。
    def test_single_gpu_execution_passes(self):
        result = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    # 可见两张设备时，多设备用例也必须实际执行。
    def test_multi_gpu_skip_is_not_validation(self):
        result = self.run_gate(TEST_GPU_COUNT="2")
        self.assertEqual(result.returncode, 77, result.stdout + result.stderr)

    # 两张可见设备下，实际执行多设备用例才能通过完整验证。
    def test_multi_gpu_execution_passes(self):
        result = self.run_gate(TEST_GPU_COUNT="2", TEST_REPORT_MODE="multi-passed")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    # 同一 fixture 的其他用例通过，也不能掩盖必需多设备用例的结果缺失。
    def test_missing_individual_gpu_result_fails(self):
        result = self.run_gate(TEST_GPU_COUNT="2", TEST_REPORT_MODE="missing-case")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    # CTest 自身失败时保留非零退出码，不以报告校验掩盖失败。
    def test_ctest_failure_is_preserved(self):
        result = self.run_gate(TEST_CTEST_EXIT="8")
        self.assertEqual(result.returncode, 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
