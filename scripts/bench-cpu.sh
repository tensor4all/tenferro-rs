#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
DEFAULT_THREADS="1,2,4"
THREADS_CSV="${TENFERRO_BENCH_THREADS_LIST:-$DEFAULT_THREADS}"
KIND="all"
DOWNLOAD_LIBTORCH=1
CRITERION_ARGS=()

DEFAULT_LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip"
LIBTORCH_URL="${LIBTORCH_URL:-$DEFAULT_LIBTORCH_URL}"

usage() {
  cat <<'EOF'
Usage: bash scripts/bench-cpu.sh [options] [-- criterion-args...]

Runs CPU benchmarks with thread counts pinned to 1, 2, and 4 by default.

Options:
  --kind tenferro|torch-cpp|all   Which benchmark to run (default: all)
  --threads LIST                  Comma-separated thread counts (default: 1,2,4)
  --no-download-libtorch          Require an existing LIBTORCH_DIR instead of downloading
  --help                          Show this help text

Environment:
  TENFERRO_BENCH_THREADS_LIST     Default for --threads
  TENFERRO_BENCH_DEPS_DIR         Repo-local dependency cache directory
  TENFERRO_MAIN_REPO_DIR          Main worktree used for default dependency cache
  LIBTORCH_DIR                    Existing LibTorch directory containing share/cmake/Torch
  LIBTORCH_URL                    LibTorch ZIP URL to download when LIBTORCH_DIR is absent

LibTorch cache policy:
  By default, downloads go under the main worktree's third_party/libtorch
  directory, not under linked git worktrees. This avoids one LibTorch copy per
  temporary worktree.

Examples:
  bash scripts/bench-cpu.sh
  bash scripts/bench-cpu.sh --kind tenferro --threads 1,4 -- --sample-size 10
  LIBTORCH_DIR=/opt/libtorch bash scripts/bench-cpu.sh --kind torch-cpp
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kind)
      KIND="${2:?missing value for --kind}"
      shift 2
      ;;
    --threads)
      THREADS_CSV="${2:?missing value for --threads}"
      shift 2
      ;;
    --no-download-libtorch)
      DOWNLOAD_LIBTORCH=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      CRITERION_ARGS=("$@")
      break
      ;;
    *)
      printf 'unknown argument: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$KIND" in
  tenferro|torch-cpp|all) ;;
  *)
    printf 'invalid --kind: %s\n' "$KIND" >&2
    exit 2
    ;;
esac

main_worktree() {
  git -C "$ROOT_DIR" worktree list --porcelain | awk '
    NR == 1 && $1 == "worktree" {
      print $2
      exit
    }
  '
}

MAIN_REPO_DIR="${TENFERRO_MAIN_REPO_DIR:-$(main_worktree)}"
if [[ -z "$MAIN_REPO_DIR" ]]; then
  MAIN_REPO_DIR="$ROOT_DIR"
fi

DEPS_DIR="${TENFERRO_BENCH_DEPS_DIR:-$MAIN_REPO_DIR/third_party/libtorch}"
LIBTORCH_DIR="${LIBTORCH_DIR:-$DEPS_DIR/libtorch}"

split_threads() {
  local csv="$1"
  local old_ifs="$IFS"
  IFS=','
  read -ra values <<< "$csv"
  IFS="$old_ifs"
  for value in "${values[@]}"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
      printf 'invalid thread count: %s\n' "$value" >&2
      exit 2
    fi
    printf '%s\n' "$value"
  done
}

pin_threads() {
  local threads="$1"
  export TENFERRO_BENCH_THREADS="$threads"
  export RAYON_NUM_THREADS="$threads"
  export OMP_NUM_THREADS="$threads"
  export OPENBLAS_NUM_THREADS="$threads"
  export MKL_NUM_THREADS="$threads"
  export VECLIB_MAXIMUM_THREADS="$threads"
  export NUMEXPR_NUM_THREADS="$threads"
}

ensure_libtorch() {
  if [[ -f "$LIBTORCH_DIR/share/cmake/Torch/TorchConfig.cmake" ]]; then
    return
  fi

  if [[ "$DOWNLOAD_LIBTORCH" -eq 0 ]]; then
    cat >&2 <<EOF
LibTorch was not found at:
  $LIBTORCH_DIR

Set LIBTORCH_DIR to an existing LibTorch install, or omit --no-download-libtorch.
EOF
    exit 1
  fi

  mkdir -p "$DEPS_DIR/archives"
  local archive="$DEPS_DIR/archives/$(basename "${LIBTORCH_URL%%\?*}")"
  if [[ ! -f "$archive" ]]; then
    printf 'Downloading LibTorch to %s\n' "$archive" >&2
    curl --fail --location --output "$archive" "$LIBTORCH_URL"
  fi

  local tmp_dir="$DEPS_DIR/.extract-$$"
  rm -rf "$tmp_dir"
  mkdir -p "$tmp_dir"
  unzip -q "$archive" -d "$tmp_dir"
  rm -rf "$LIBTORCH_DIR"
  mv "$tmp_dir/libtorch" "$LIBTORCH_DIR"
  rm -rf "$tmp_dir"
}

run_tenferro() {
  local threads="$1"
  pin_threads "$threads"
  printf '==> tenferro Criterion CPU benchmarks, threads=%s\n' "$threads" >&2
  cargo bench -p tenferro --bench cpu_bench -- "${CRITERION_ARGS[@]}"
}

build_torch_cpp() {
  ensure_libtorch
  local build_dir="$ROOT_DIR/target/torch-cpp-bench"
  cmake -S "$ROOT_DIR/benchmarks/torch-cpp" \
    -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR"
  cmake --build "$build_dir" --config Release --parallel
}

run_torch_cpp() {
  local threads="$1"
  pin_threads "$threads"
  build_torch_cpp
  export LD_LIBRARY_PATH="$LIBTORCH_DIR/lib:${LD_LIBRARY_PATH:-}"
  printf '==> LibTorch C++ CPU baseline, threads=%s\n' "$threads" >&2
  "$ROOT_DIR/target/torch-cpp-bench/torch_cpu_bench" "$threads"
}

mapfile -t THREADS < <(split_threads "$THREADS_CSV")

for threads in "${THREADS[@]}"; do
  case "$KIND" in
    tenferro)
      run_tenferro "$threads"
      ;;
    torch-cpp)
      run_torch_cpp "$threads"
      ;;
    all)
      run_tenferro "$threads"
      run_torch_cpp "$threads"
      ;;
  esac
done
