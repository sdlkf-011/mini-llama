#!/usr/bin/env python3
"""Run the existing CPU/CUDA Q8_0 and Q4_0 CLI regressions as a focused suite.

Covers generate/bench/run, error reporting, uploaded-weight counters and CUDA
execution. GPU cases skip explicitly when CUDA or a device is unavailable.
"""

import argparse
from pathlib import Path
import unittest

import test_backend_cli as backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=backend.BIN)
    args = parser.parse_args()
    backend.BIN = args.binary.resolve()
    if not backend.BIN.is_file():
        parser.error(f"build the CLI first: {backend.BIN}")
    suite = unittest.TestSuite(backend.BackendCliTest(name) for name in (
        "test_quant_overrides_generate_and_bench",
        "test_cuda_quant_overrides_generate_and_bench"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
