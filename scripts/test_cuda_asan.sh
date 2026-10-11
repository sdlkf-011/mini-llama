#!/usr/bin/env bash
# 测试目的：以 AddressSanitizer/UBSan 检查 CUDA 测试的主机侧内存；
# 无可见 GPU 或必需 GPU 用例未执行时返回 77；
# --device-memcheck 另用 compute-sanitizer 检查设备内存，缺少工具时返回 77。
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: test_cuda_asan.sh [--build-dir PATH] [--jobs N] [--device-memcheck] [-- CMake options]
Builds CUDA with host AddressSanitizer/UBSan, then runs CUDA CTest cases.
Defaults: build-asan-cuda, 4 build jobs. CMake options can select your SDK/compiler.
ASAN_OPTIONS and UBSAN_OPTIONS may be supplied by the caller.
Exit 77 means GPU validation did not run; CI must treat it as skipped, not passed.
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
build_dir="$project_dir/build-asan-cuda"
build_jobs=4
device_memcheck=0
cmake_options=()
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --build-dir|--jobs)
      if (($# < 2)); then usage >&2; exit 2; fi
      if [[ "$1" == --build-dir ]]; then build_dir="$2"; else build_jobs="$2"; fi
      shift 2 ;;
    --device-memcheck) device_memcheck=1; shift ;;
    --) shift; cmake_options=("$@"); break ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
  printf '%s\n' '--jobs must be a positive integer' >&2; exit 2
fi
if ((device_memcheck)) && ! command -v compute-sanitizer >/dev/null 2>&1; then
  printf '%s\n' 'NOT RUN: compute-sanitizer is unavailable' >&2; exit 77
fi
cmake -S "$project_dir" -B "$build_dir" \
  -DMINI_LLAMA_CUDA=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug \
  '-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer' \
  '-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined' \
  "${cmake_options[@]}"
cmake --build "$build_dir" -j "$build_jobs"
export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=1:halt_on_error=1}"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:-halt_on_error=1:print_stacktrace=1}"
# --cuda-info can succeed with zero devices, so validate its actual diagnostics.
cuda_info="$("$build_dir/mini-llama" --cuda-info)"
printf '%s\n' "$cuda_info"
gpu_count="$(python3 - "$cuda_info" <<'PY'
import re
import sys

info = sys.argv[1]
counts = re.findall(r"^CUDA GPU count: ([0-9]+)$", info, re.MULTILINE)
if "CUDA backend built: yes" not in info.splitlines() or len(counts) != 1:
    sys.exit("GPU validation failed: incomplete CUDA device diagnostics")
count = int(counts[0])
if count == 0:
    print("NOT RUN: no visible CUDA GPU; GPU validation was not performed", file=sys.stderr)
    sys.exit(77)
print(count)
PY
)"

reports="$(mktemp -d)"
trap 'rm -rf -- "$reports"' EXIT
ctest --test-dir "$build_dir" --show-only=json-v1 -R 'cuda|Cuda' > "$reports/tests.json"
ctest --test-dir "$build_dir" --output-on-failure -j 1 -R 'cuda|Cuda' \
  --output-junit "$reports/results.xml"
python3 - "$gpu_count" "$reports/tests.json" "$reports/results.xml" <<'PY'
import json
import sys
import xml.etree.ElementTree as ET

# These fixtures exercise the runtime, tensors, compute kernels and full forward.
# Build/stub contracts and the external-model chat smoke are optional here.
families = (
    "cuda_runtime.CudaRuntimeGpuTest.", "CudaTensorGpuTest.",
    "cuda_matmul.", "cuda_linear.", "cuda_ops.CudaOpsGpu.",
    "cuda_quant_q8_0.", "cuda_quant_q4.",
    "cuda_kv_cache.CudaKvCacheGpuTest.", "cuda_attention.CudaAttentionGpuTest.",
    "cuda_forward.CudaForwardTest.", "cuda_forward.CudaForwardAttentionTest.",
    "cuda_forward_device_resident.CudaForwardBufferTest.",
    "cuda_forward_linear.CudaForwardLinearTest.",
    "cuda_forward_linear.cuda_forward_quant.", "cuda_forward_linear.cuda_forward_q4.",
)
multi_device = ("cuda_runtime.CudaRuntimeGpuTest."
                "BuffersRetainTheirDeviceAcrossOtherDeviceOperations")
try:
    with open(sys.argv[2], encoding="utf-8") as stream:
        names = {test["name"] for test in json.load(stream)["tests"]}
    missing_families = [family for family in families
                        if not any(name.startswith(family) for name in names)]
    if missing_families:
        raise ValueError("missing required GPU fixtures: " + ", ".join(missing_families))
    required = {name for name in names if name.startswith(families)}
    if int(sys.argv[1]) < 2:
        required.discard(multi_device)
    cases = {}
    for case in ET.parse(sys.argv[3]).getroot().iter("testcase"):
        name = case.get("name")
        if name in cases:
            raise ValueError("duplicate result: " + str(name))
        cases[name] = case
    if required - cases.keys():
        raise ValueError("missing GPU results: " + ", ".join(sorted(required - cases.keys())))
    skipped = []
    for name in sorted(required):
        case = cases[name]
        if case.find("failure") is not None or case.find("error") is not None:
            raise ValueError("GPU test failed: " + name)
        if case.find("skipped") is not None or case.get("status") in ("notrun", "disabled"):
            skipped.append(name)
    if skipped:
        print("NOT RUN: required GPU tests did not execute:\n  " + "\n  ".join(skipped),
              file=sys.stderr)
        sys.exit(77)
except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as error:
    sys.exit("GPU validation failed: " + str(error))
print(f"GPU validation passed: {len(required)} required tests executed")
PY
if ((device_memcheck)); then
  compute-sanitizer --tool memcheck --error-exitcode 1 \
    "$build_dir/mini-llama" generate --model "$project_dir/models/tiny" \
    --backend cuda --prompt hello --n-predict 4 --seed 1
fi
