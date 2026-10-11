#!/usr/bin/env python3
"""CPU MVP checks: metadata-only inspect, deterministic generation and chat."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("MINI_LLAMA_BIN", ROOT / "build/mini-llama"))
MODEL = ROOT / "models/tiny"


class MvpTest(unittest.TestCase):
    def invoke(self, *args, text="", status=0):
        result = subprocess.run([str(BIN), *map(str, args)], input=text,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, status, result.stdout + result.stderr)
        if status == 0:
            self.assertNotIn("status=error", result.stdout + result.stderr)
        return result

    # 帮助和清单检查不需要读取权重或启动推理。
    def test_help_and_metadata_only_inspect(self):
        self.assertIn("generate", self.invoke("--help").stdout)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "model.json").write_bytes((MODEL / "model.json").read_bytes())
            output = self.invoke("inspect", directory).stdout
            self.assertIn("vocab_size:", output)
            self.assertIn("metadata only", output)

    # 贪心生成结果在相同 seed 和 prompt 下稳定，不能越过生成预算。
    def test_generation_is_reproducible(self):
        args = ("generate", MODEL, "--backend", "cpu", "-p", "hello", "-n", "4",
                "--seed", "42", "--threads", "1")
        first, second = self.invoke(*args), self.invoke(*args)
        tokens = re.search(r"^generated tokens: \[([^\]]*)\]$", first.stdout, re.MULTILINE)
        self.assertIsNotNone(tokens, first.stdout)
        values = [int(value) for value in tokens.group(1).split(",") if value.strip()]
        self.assertTrue(1 <= len(values) <= 4)
        self.assertIn(tokens.group(0), second.stdout.splitlines())
        self.assertIn("backend: cpu", first.stdout)

    # prefill-only 导出一份完整词表 logits，但不执行采样或 decode。
    def test_prefill_only_dump(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = self.invoke("generate", MODEL, "-p", "hello", "-n", "0",
                                 "--dump-logits", temporary)
            self.assertIn("generated tokens: []", output.stdout)
            dumps = list(Path(temporary).glob("logits_step*.bin"))
            self.assertGreater(len(dumps), 0)
            self.assertTrue(all(path.stat().st_size == 128 * 4 for path in dumps))
            self.assertEqual((Path(temporary) / "generation_tokens.txt").read_text(), "\n")

    # 多轮聊天、统计、清空和退出必须由终端处理，不把命令送入模型。
    def test_chat_commands_and_turn_traces(self):
        output = self.invoke("run", MODEL, "-n", "1",
                             text="hello\n/stats\n/clear\n/stats\n/exit\n").stdout
        self.assertIn("mode=run status=ok", output)
        self.assertIn("Messages: 2", output)
        self.assertIn("Session cleared.", output)
        self.assertIn("Messages: 0", output)
        self.assertIn("Bye.", output)

    # 参数错误应在尝试加载不存在的模型之前报告。
    def test_invalid_arguments_fail_before_loading(self):
        result = self.invoke("generate", "missing-model", "-n", "-1", status=1)
        self.assertNotIn("generated tokens:", result.stdout)
        self.assertNotIn("trace summary", result.stderr)


def main():
    global BIN, MODEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=BIN)
    parser.add_argument("--model", type=Path, default=MODEL,
                        help="128-token JSON+BIN ASCII demo model")
    args = parser.parse_args()
    BIN, MODEL = args.binary.resolve(), args.model.resolve()
    if not BIN.is_file():
        parser.error(f"build the CLI first: {BIN}")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MvpTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
