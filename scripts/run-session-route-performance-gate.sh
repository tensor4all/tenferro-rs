#!/usr/bin/env bash
set -euo pipefail

# Session-route unification performance campaign (issue #1926 / umbrella #1929).
#
# This is a separate campaign from the execution-engine "Unification"
# terminal gate owned by scripts/run-unification-performance-gate.sh. That
# campaign pins its own baseline commit and harness identity, so this one must
# not reuse it with a different pin. The protocol shape (one-thread backend,
# CPU affinity, recorded manifest, Criterion relative-change comparison,
# three alternating pairs) is shared.
#
# Purpose: measure the coexisting CPU execution routes BEFORE the route/API
# unification deletes the one-shot spelling, so that the surviving routes can
# be compared as a matched pair afterwards. The `oneshot/*` cases and
# `session_chain/*/one_shot` are before-only references: they cannot be
# reproduced after #1926 removes the spelling.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

BASELINE_COMMIT="${TENFERRO_SESSION_ROUTE_BASELINE_COMMIT:-25da8d431}"
MODE="dry-run"
LABEL="diagnostic"
CPU="${TENFERRO_SESSION_ROUTE_BENCH_CPU:-0}"
WARM_UP_TIME="${TENFERRO_SESSION_ROUTE_WARM_UP_TIME:-2}"
MEASUREMENT_TIME="${TENFERRO_SESSION_ROUTE_MEASUREMENT_TIME:-5}"
SAMPLE_SIZE="${TENFERRO_SESSION_ROUTE_SAMPLE_SIZE:-100}"
OUTPUT_DIR="${TENFERRO_SESSION_ROUTE_OUTPUT_DIR:-$ROOT_DIR/target/session-route-performance-gate}"
CRITERION_EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash scripts/run-session-route-performance-gate.sh [options] [-- criterion-args...]

Runs, or dry-runs, the session-route unification performance campaign.

Options:
  --mode dry-run|run          Print commands or execute them (default: dry-run)
  --label NAME               Result label: baseline, candidate, diagnostic
  --cpu N                    CPU id used with taskset when available (default: 0)
  --output-dir DIR           Directory for logs and manifest
  --warm-up-time SECONDS     Criterion warm-up time (default: 2)
  --measurement-time SECONDS Criterion measurement time (default: 5)
  --sample-size N            Criterion sample size (default: 100)
  --help                     Show this help text

Environment:
  TENFERRO_SESSION_ROUTE_BASELINE_COMMIT  Pinned pre-unification main commit
  TENFERRO_SESSION_ROUTE_BENCH_CPU        Default for --cpu
  TENFERRO_SESSION_ROUTE_OUTPUT_DIR       Default for --output-dir

The harness identity is the commit that adds this script and the
`session_route_*` resolution below. If the harness source changes after
baseline numbers are collected, baseline collection must restart.

Baseline runs apply this harness source to the pinned baseline code without
candidate implementation changes.
EOF
}

die() {
  printf '%s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:?--mode requires dry-run or run}"; shift 2 ;;
    --label) LABEL="${2:?--label requires a value}"; shift 2 ;;
    --cpu) CPU="${2:?--cpu requires an integer}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?--output-dir requires a path}"; shift 2 ;;
    --warm-up-time) WARM_UP_TIME="${2:?}"; shift 2 ;;
    --measurement-time) MEASUREMENT_TIME="${2:?}"; shift 2 ;;
    --sample-size) SAMPLE_SIZE="${2:?}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; CRITERION_EXTRA_ARGS=("$@"); break ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$MODE" in dry-run|run) ;; *) die "--mode must be dry-run or run" ;; esac
case "$LABEL" in baseline|candidate|diagnostic) ;; *) die "--label must be baseline, candidate, or diagnostic" ;; esac
[[ "$CPU" =~ ^[0-9]+$ ]] || die "--cpu must be a non-negative integer"
[[ "$SAMPLE_SIZE" =~ ^[1-9][0-9]*$ ]] || die "--sample-size must be a positive integer"

# package|bench|features|source|description
BENCHMARKS=(
  "tenferro-cpu|route_matrix||crates/tenferro-cpu/benches/route_matrix.rs|coexisting CPU entry routes on the same logical op: oneshot vs session vs execution scope, per-op and marginal-in-entry, plus throughput sizes"
  "tenferro-runtime|session_chain||crates/tenferro-runtime/benches/session_chain.rs|10-op chain through one session vs one-shot-per-op vs one execution scope; phase-1 mixed chain"
  "tenferro-runtime|elementwise_fusion||crates/tenferro-runtime/benches/elementwise_fusion.rs|compiled graph steady-state execution"
  "tenferro-ad|eager_dispatch_baseline||crates/tenferro-ad/benches/eager_dispatch_baseline.rs|eager small-op dispatch, including indexed slice"
  "tenferro-ad|eager_backward_shape_churn||crates/tenferro-ad/benches/eager_backward_shape_churn.rs|eager backward under shape churn"
  "tenferro-linalg|linalg_vjp_gate|autodiff|crates/tenferro-linalg/benches/linalg_vjp_gate.rs|extension-bearing linalg VJP through the CPU session downcast path"
  "tenferro-gpu|route_matrix_gpu|cuda|crates/tenferro-gpu/benches/route_matrix_gpu.rs|coexisting CUDA entry routes on one contraction, with enqueue and synchronized completion reported separately"
)

# CUDA runtime for the GPU targets. CUBECL_DEBUG_LOG=0 is required: without it
# CubeCL prints the generated CUDA source for every JIT-compiled kernel.
CUDA_ROOT="${TENFERRO_CUDA_ROOT:-/usr/local/cuda-12.6}"
CUTENSOR_LIB_DIR="${TENFERRO_CUTENSOR_LIB_DIR:-/usr/lib/x86_64-linux-gnu/libcutensor/12}"

criterion_args=(
  --warm-up-time "$WARM_UP_TIME"
  --measurement-time "$MEASUREMENT_TIME"
  --sample-size "$SAMPLE_SIZE"
)
if ((${#CRITERION_EXTRA_ARGS[@]} > 0)); then
  criterion_args+=("${CRITERION_EXTRA_ARGS[@]}")
fi

run_prefix=()
if command -v taskset >/dev/null 2>&1; then
  run_prefix=(taskset -c "$CPU")
fi

# One logical CPU thread for every provider and runtime. See REPOSITORY_RULES /
# AGENTS: never use the machine default as the overhead baseline.
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TENFERRO_BENCH_THREADS=1
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"

mkdir -p "$OUTPUT_DIR"
manifest="$OUTPUT_DIR/${LABEL}-manifest.txt"

{
  printf 'campaign=session-route-unification\n'
  printf 'label=%s\n' "$LABEL"
  printf 'mode=%s\n' "$MODE"
  printf 'repo=%s\n' "$ROOT_DIR"
  printf 'head=%s\n' "$(git -C "$ROOT_DIR" rev-parse HEAD)"
  printf 'baseline_commit=%s\n' "$BASELINE_COMMIT"
  printf 'cpu=%s\n' "$CPU"
  printf 'taskset_available=%s\n' "$(command -v taskset >/dev/null 2>&1 && printf yes || printf no)"
  printf 'warm_up_time=%s\n' "$WARM_UP_TIME"
  printf 'measurement_time=%s\n' "$MEASUREMENT_TIME"
  printf 'sample_size=%s\n' "$SAMPLE_SIZE"
  printf 'cargo_build_jobs=%s\n' "$CARGO_BUILD_JOBS"
  printf 'thread_env=RAYON=1 OMP=1 OPENBLAS=1 MKL=1 VECLIB=1 NUMEXPR=1 TENFERRO_BENCH_THREADS=1\n'
  printf 'loadavg=%s\n' "$(cut -d' ' -f1-3 /proc/loadavg)"
  printf 'nproc=%s\n' "$(nproc)"
} >"$manifest"

printf 'Session-route unification performance campaign (%s)\n' "$LABEL"
printf '  mode:            %s\n' "$MODE"
printf '  baseline commit: %s\n' "$BASELINE_COMMIT"
printf '  output dir:      %s\n' "$OUTPUT_DIR"
printf '  criterion args:  %s\n' "${criterion_args[*]}"

missing=0
for entry in "${BENCHMARKS[@]}"; do
  IFS='|' read -r package bench features path description <<<"$entry"
  printf '\n== %s / %s ==\n' "$package" "$bench"
  printf 'description: %s\n' "$description"
  printf 'features:    %s\n' "${features:-<default>}"
  if [[ ! -f "$ROOT_DIR/$path" ]]; then
    printf 'status:      missing\n'
    missing=1
    continue
  fi
  printf 'status:      present\n'

  build_cmd=(cargo bench -p "$package")
  run_cmd=("${run_prefix[@]}" cargo bench -p "$package")
  bench_features="$features"
  if [[ "$package" == "tenferro-linalg" && "$bench" == "linalg_vjp_gate" ]]; then
    # The semantic-extension AD API generation is required by this bench.
    bench_features="autodiff,__bench_unification_semantic_ad_api"
  fi
  if [[ -n "$bench_features" ]]; then
    build_cmd+=(--features "$bench_features")
    run_cmd+=(--features "$bench_features")
  fi
  build_cmd+=(--bench "$bench" --no-run)
  run_cmd+=(--bench "$bench" -- "${criterion_args[@]}")

  run_env=()
  if [[ ",$bench_features," == *",cuda,"* ]]; then
    printf 'cuda_root:   %s\n' "$CUDA_ROOT"
    printf 'cutensor:    %s\n' "$CUTENSOR_LIB_DIR"
    printf 'gpu_target=%s cuda_root=%s cutensor=%s\n' \
      "$bench" "$CUDA_ROOT" "$CUTENSOR_LIB_DIR" >>"$manifest"
    run_env=(
      CUBECL_DEBUG_LOG=0
      "CUDA_PATH=$CUDA_ROOT"
      "LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$CUTENSOR_LIB_DIR:${LD_LIBRARY_PATH:-}"
    )
  fi

  printf 'build:       %s\n' "${build_cmd[*]}"
  printf 'run:         %s\n' "${run_env[*]:-} ${run_cmd[*]}"

  if [[ "$MODE" == "run" ]]; then
    (
      cd "$ROOT_DIR"
      env "${run_env[@]}" "${build_cmd[@]}"
    ) 2>&1 | tee "$OUTPUT_DIR/${LABEL}-${package}-${bench}-build.log"
    (
      cd "$ROOT_DIR"
      env "${run_env[@]}" "${run_cmd[@]}"
    ) 2>&1 | tee "$OUTPUT_DIR/${LABEL}-${package}-${bench}-run.log"
  fi
done

if [[ "$missing" -ne 0 ]]; then
  die "one or more benchmark targets are missing"
fi
