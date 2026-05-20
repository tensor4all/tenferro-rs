#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
DEFAULT_JULIA_PROJECT="$ROOT_DIR/benchmarks/julia"
JULIA_PROJECT_DIR="${JULIA_PROJECT_DIR:-$DEFAULT_JULIA_PROJECT}"

KIND="both"
WARM_UP_TIME="${WARM_UP_TIME:-0.5}"
MEASUREMENT_TIME="${MEASUREMENT_TIME:-1}"
SAMPLE_SIZE="${SAMPLE_SIZE:-10}"
CRITERION_EXTRA_ARGS=()
JULIA_EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash scripts/bench-pairwise-contraction.sh [options] [-- criterion-args...]

Runs tenferro and ITensors pairwise-contraction benchmarks with all known
thread pools pinned to one thread.

Options:
  --kind rust|julia|both         Which benchmark to run (default: both)
  --warm-up-time SECONDS         Warm-up time (default: 0.5)
  --measurement-time SECONDS     Measurement time (default: 1)
  --sample-size N                Rust Criterion sample size and Julia min samples (default: 10)
  --julia-project DIR            Julia environment with ITensors (default: benchmarks/julia)
  --julia-arg ARG                Extra argument forwarded to the Julia benchmark; repeatable
  --help                         Show this help text

Environment overrides:
  WARM_UP_TIME                   Same as --warm-up-time
  MEASUREMENT_TIME               Same as --measurement-time
  SAMPLE_SIZE                    Same as --sample-size
  JULIA_PROJECT_DIR              Same as --julia-project

Examples:
  bash scripts/bench-pairwise-contraction.sh
  bash scripts/bench-pairwise-contraction.sh --kind rust -- pairwise_contraction/c64/one_thread/normal_per_call
  bash scripts/bench-pairwise-contraction.sh --kind julia --julia-arg --chis --julia-arg 4,8,16
EOF
}

log() {
  printf '%s\n' "$*"
}

die() {
  log "$*" >&2
  exit 1
}

run_one_thread() {
  log "+ $*"
  RAYON_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    JULIA_NUM_THREADS=1 \
    "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kind)
      [[ $# -ge 2 ]] || die "--kind requires rust, julia, or both"
      KIND="$2"
      shift 2
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
    --julia-project)
      [[ $# -ge 2 ]] || die "--julia-project requires a directory"
      JULIA_PROJECT_DIR="$2"
      shift 2
      ;;
    --julia-arg)
      [[ $# -ge 2 ]] || die "--julia-arg requires a value"
      JULIA_EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    --)
      shift
      CRITERION_EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

case "$KIND" in
  rust | julia | both) ;;
  *) die "--kind must be rust, julia, or both" ;;
esac

cd "$ROOT_DIR"

log "Pairwise contraction benchmark"
log "  kind:             $KIND"
log "  warm-up time:     $WARM_UP_TIME"
log "  measurement time: $MEASUREMENT_TIME"
log "  sample size:      $SAMPLE_SIZE"
log "  threads:          RAYON=1 VECLIB=1 OPENBLAS=1 OMP=1 MKL=1 JULIA=1"

if [[ "$KIND" == "rust" || "$KIND" == "both" ]]; then
  log ""
  log "== Rust tenferro-tensor =="
  criterion_args=(
    --warm-up-time "$WARM_UP_TIME"
    --measurement-time "$MEASUREMENT_TIME"
    --sample-size "$SAMPLE_SIZE"
  )
  if ((${#CRITERION_EXTRA_ARGS[@]} > 0)); then
    criterion_args+=("${CRITERION_EXTRA_ARGS[@]}")
  fi
  run_one_thread \
    cargo bench -p tenferro-tensor --bench pairwise_contraction -- \
    "${criterion_args[@]}"
fi

if [[ "$KIND" == "julia" || "$KIND" == "both" ]]; then
  log ""
  log "== Julia ITensors =="
  [[ -f "$JULIA_PROJECT_DIR/Project.toml" ]] || die "Julia project not found: $JULIA_PROJECT_DIR"
  run_one_thread \
    julia --startup-file=no --project="$JULIA_PROJECT_DIR" \
    "$ROOT_DIR/scripts/bench-itensors-pairwise-contraction.jl" \
    --warm-up-time "$WARM_UP_TIME" \
    --measurement-time "$MEASUREMENT_TIME" \
    --min-samples "$SAMPLE_SIZE" \
    --blas-threads 1 \
    "${JULIA_EXTRA_ARGS[@]}"
fi
