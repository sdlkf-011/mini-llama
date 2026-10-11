#!/usr/bin/env bash
# 测试目的：以 AddressSanitizer/UBSan 检查 CUDA 测试的主机侧内存；
# --device-memcheck 另用 compute-sanitizer 检查设备内存，缺少工具时返回 77。
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: test_cuda_asan.sh [--build-dir PATH] [--jobs N] [--device-memcheck] [-- CMake options]
Builds CUDA with host AddressSanitizer/UBSan, then runs CUDA CTest cases.
Defaults: build-asan-cuda, 4 build jobs. CMake options can select your SDK/compiler.
ASAN_OPTIONS and UBSAN_OPTIONS may be supplied by the caller.
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
# Fail if the requested GPU is unavailable; a wholly skipped suite is not validation.
"$build_dir/mini-llama" --cuda-info
ctest --test-dir "$build_dir" --output-on-failure -j 1 -R 'cuda|Cuda'
if ((device_memcheck)); then
  compute-sanitizer --tool memcheck --error-exitcode 1 \
    "$build_dir/mini-llama" generate --model "$project_dir/models/tiny" \
    --backend cuda --prompt hello --n-predict 4 --seed 1
fi
