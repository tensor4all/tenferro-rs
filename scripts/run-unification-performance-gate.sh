#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

BASELINE_COMMIT="${TENFERRO_UNIFICATION_BASELINE_COMMIT:-c6418eecfe2d38ca09d6e6386760fcb23982691e}"
MODE="dry-run"
LABEL="diagnostic"
ALLOW_MISSING=0
CPU="${TENFERRO_UNIFICATION_BENCH_CPU:-0}"
WARM_UP_TIME="${TENFERRO_UNIFICATION_WARM_UP_TIME:-2}"
MEASUREMENT_TIME="${TENFERRO_UNIFICATION_MEASUREMENT_TIME:-5}"
SAMPLE_SIZE="${TENFERRO_UNIFICATION_SAMPLE_SIZE:-100}"
OUTPUT_DIR="${TENFERRO_UNIFICATION_OUTPUT_DIR:-$ROOT_DIR/target/unification-performance-gate}"
CRITERION_EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash scripts/run-unification-performance-gate.sh [options] [-- criterion-args...]

Runs, or dry-runs, the predeclared Unification terminal performance suite.

Options:
  --mode dry-run|run          Print commands or execute them (default: dry-run)
  --label NAME                Result label: baseline, candidate, diagnostic
  --cpu N                     CPU id used with taskset when available (default: 0)
  --output-dir DIR            Directory for logs and manifest
  --allow-missing             Skip benchmark targets not yet introduced
  --warm-up-time SECONDS      Criterion warm-up time (default: 2)
  --measurement-time SECONDS  Criterion measurement time (default: 5)
  --sample-size N             Criterion sample size (default: 100)
  --help                      Show this help text

Environment:
  TENFERRO_UNIFICATION_BASELINE_COMMIT  Pinned pre-migration main commit
  TENFERRO_UNIFICATION_BENCH_CPU        Default for --cpu
  TENFERRO_UNIFICATION_OUTPUT_DIR       Default for --output-dir

The benchmark harness identity is the integration-branch source that contains
this script and the named bench targets. For baseline runs, apply that harness
source to the pinned baseline code without candidate implementation changes.
EOF
}

die() {
  printf '%s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || die "--mode requires dry-run or run"
      MODE="$2"
      shift 2
      ;;
    --label)
      [[ $# -ge 2 ]] || die "--label requires a value"
      LABEL="$2"
      shift 2
      ;;
    --cpu)
      [[ $# -ge 2 ]] || die "--cpu requires an integer"
      CPU="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || die "--output-dir requires a path"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --allow-missing)
      ALLOW_MISSING=1
      shift
      ;;
    --warm-up-time)
      [[ $# -ge 2 ]] || die "--warm-up-time requires seconds"
      WARM_UP_TIME="$2"
      shift 2
      ;;
    --measurement-time)
      [[ $# -ge 2 ]] || die "--measurement-time requires seconds"
      MEASUREMENT_TIME="$2"
      shift 2
      ;;
    --sample-size)
      [[ $# -ge 2 ]] || die "--sample-size requires a count"
      SAMPLE_SIZE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      CRITERION_EXTRA_ARGS=("$@")
      break
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

case "$MODE" in
  dry-run|run) ;;
  *) die "--mode must be dry-run or run" ;;
esac

case "$LABEL" in
  baseline|candidate|diagnostic) ;;
  *) die "--label must be baseline, candidate, or diagnostic" ;;
esac

[[ "$CPU" =~ ^[0-9]+$ ]] || die "--cpu must be a non-negative integer"
[[ "$SAMPLE_SIZE" =~ ^[1-9][0-9]*$ ]] || die "--sample-size must be a positive integer"

BENCHMARKS=(
  "tenferro-ad|eager_dispatch_baseline||crates/tenferro-ad/benches/eager_dispatch_baseline.rs|eager small-op dispatch, including indexed slice"
  "tenferro-runtime|elementwise_fusion||crates/tenferro-runtime/benches/elementwise_fusion.rs|compiled graph steady-state execution"
  "tenferro-einsum|changing_shape_prepare|autodiff|crates/tenferro-einsum/benches/changing_shape_prepare.rs|changing-shape einsum prepare throughput"
  "tenferro-ad|eager_backward_shape_churn||crates/tenferro-ad/benches/eager_backward_shape_churn.rs|eager backward under shape churn"
  "tenferro-linalg|linalg_vjp_gate|autodiff|crates/tenferro-linalg/benches/linalg_vjp_gate.rs|extension-bearing linalg VJP"
)

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
  printf 'allow_missing=%s\n' "$ALLOW_MISSING"
} >"$manifest"

printf 'Unification performance gate (%s)\n' "$LABEL"
printf '  mode:             %s\n' "$MODE"
printf '  baseline commit:  %s\n' "$BASELINE_COMMIT"
printf '  output dir:       %s\n' "$OUTPUT_DIR"
printf '  criterion args:   %s\n' "${criterion_args[*]}"
printf '  thread env:       RAYON=1 OMP=1 OPENBLAS=1 MKL=1 VECLIB=1 NUMEXPR=1\n'
printf '  build jobs:       CARGO_BUILD_JOBS=%s\n' "$CARGO_BUILD_JOBS"

missing=0
for entry in "${BENCHMARKS[@]}"; do
  IFS='|' read -r package bench features path description <<<"$entry"
  printf '\n== %s / %s ==\n' "$package" "$bench"
  printf 'description: %s\n' "$description"
  if [[ -n "$features" ]]; then
    printf 'features:    %s\n' "$features"
  else
    printf 'features:    <default>\n'
  fi
  printf 'source:      %s\n' "$path"
  if [[ ! -f "$ROOT_DIR/$path" ]]; then
    printf 'status:      missing\n'
    missing=1
    if [[ "$ALLOW_MISSING" -eq 0 ]]; then
      continue
    fi
    printf 'action:      skipped due to --allow-missing\n'
    continue
  fi

  printf 'status:      present\n'
  bench_features="$features"
  bench_api="default"
  if [[ "$package" == "tenferro-linalg" && "$bench" == "linalg_vjp_gate" ]]; then
    if grep -q "with_semantic_extension_rules" "$ROOT_DIR/crates/tenferro-ad/src/context.rs" \
      && grep -q "semantic_ad_rules" "$ROOT_DIR/crates/tenferro-linalg/src/lib.rs"; then
      bench_api="semantic-ad"
      if [[ -n "$bench_features" ]]; then
        bench_features+=",__bench_unification_semantic_ad_api"
      else
        bench_features="__bench_unification_semantic_ad_api"
      fi
    else
      bench_api="legacy-extension-ad"
    fi
    printf 'ad api:      %s\n' "$bench_api"
    printf 'linalg_vjp_gate_ad_api=%s\n' "$bench_api" >>"$manifest"
  fi
  build_log="$OUTPUT_DIR/${LABEL}-${package}-${bench}-build.log"
  run_log="$OUTPUT_DIR/${LABEL}-${package}-${bench}-run.log"
  build_cmd=(cargo bench -p "$package")
  run_cmd=("${run_prefix[@]}" cargo bench -p "$package")

  if [[ "$package" == "tenferro-runtime" && "$bench" == "elementwise_fusion" ]]; then
    runtime_api="run-compiled"
    printf 'runtime api: %s\n' "$runtime_api"
    printf 'elementwise_fusion_runtime_api=%s\n' "$runtime_api" >>"$manifest"
  fi

  if [[ -n "$bench_features" ]]; then
    build_cmd+=(--features "$bench_features")
    run_cmd+=(--features "$bench_features")
  fi
  build_cmd+=(--bench "$bench" --no-run)
  run_cmd+=(--bench "$bench" -- "${criterion_args[@]}")

  printf 'build:       %s\n' "${build_cmd[*]}"
  printf 'run:         %s\n' "${run_cmd[*]}"

  if [[ "$MODE" == "run" ]]; then
    (
      cd "$ROOT_DIR"
      "${build_cmd[@]}"
    ) 2>&1 | tee "$build_log"
    (
      cd "$ROOT_DIR"
      "${run_cmd[@]}"
    ) 2>&1 | tee "$run_log"
  fi
done

if [[ "$missing" -ne 0 && "$ALLOW_MISSING" -eq 0 ]]; then
  die "one or more benchmark targets are missing; rerun with --allow-missing for a partial dry run"
fi
