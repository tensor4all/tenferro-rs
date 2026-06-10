#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

BACKEND="both"
KIND="runtime"
BLAS_FEATURE="${BLAS_FEATURE:-blas-accelerate}"
WARM_UP_TIME="${WARM_UP_TIME:-1}"
MEASUREMENT_TIME="${MEASUREMENT_TIME:-2}"
SAMPLE_SIZE="${SAMPLE_SIZE:-10}"
CRITERION_EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash scripts/bench-mps-inner-product.sh [options] [-- criterion-args...]

Runs the tenferro MPS inner-product benchmark with one-thread settings pinned
for Rust, BLAS, Accelerate, OpenMP, and MKL.

Options:
  --kind runtime|eager|compile|both|all
                                Runtime sweep, eager sweep, compile-only sweep,
                                runtime+compile, or all three (default: runtime)
  --backend faer|blas|both       Which backend configuration to run (default: both)
  --blas-feature FEATURE         BLAS provider feature for the blas run (default: blas-accelerate)
  --warm-up-time SECONDS         Criterion warm-up time (default: 1)
  --measurement-time SECONDS     Criterion measurement time (default: 2)
  --sample-size N                Criterion sample size (default: 10)
  --help                         Show this help text

Environment overrides:
  BLAS_FEATURE                   Same as --blas-feature
  WARM_UP_TIME                   Same as --warm-up-time
  MEASUREMENT_TIME               Same as --measurement-time
  SAMPLE_SIZE                    Same as --sample-size

Examples:
  bash scripts/bench-mps-inner-product.sh
  bash scripts/bench-mps-inner-product.sh --kind eager
  bash scripts/bench-mps-inner-product.sh --kind compile
  bash scripts/bench-mps-inner-product.sh --backend faer --measurement-time 5
  bash scripts/bench-mps-inner-product.sh --backend blas -- --save-baseline blas
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
    "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      [[ $# -ge 2 ]] || die "--backend requires faer, blas, or both"
      BACKEND="$2"
      shift 2
      ;;
    --kind)
      [[ $# -ge 2 ]] || die "--kind requires runtime, eager, compile, both, or all"
      KIND="$2"
      shift 2
      ;;
    --blas-feature)
      [[ $# -ge 2 ]] || die "--blas-feature requires a Cargo feature"
      BLAS_FEATURE="$2"
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

case "$BACKEND" in
  faer | blas | both) ;;
  *) die "--backend must be faer, blas, or both" ;;
esac
case "$KIND" in
  runtime | eager | compile | both | all) ;;
  *) die "--kind must be runtime, eager, compile, both, or all" ;;
esac

cd "$ROOT_DIR"

criterion_args=(
  --warm-up-time "$WARM_UP_TIME"
  --measurement-time "$MEASUREMENT_TIME"
  --sample-size "$SAMPLE_SIZE"
)
if ((${#CRITERION_EXTRA_ARGS[@]} > 0)); then
  criterion_args+=("${CRITERION_EXTRA_ARGS[@]}")
fi

log "MPS inner-product benchmark"
log "  kind:             $KIND"
log "  backend:          $BACKEND"
log "  BLAS feature:     $BLAS_FEATURE"
log "  warm-up time:     $WARM_UP_TIME"
log "  measurement time: $MEASUREMENT_TIME"
log "  sample size:      $SAMPLE_SIZE"
log "  threads:          RAYON=1 VECLIB=1 OPENBLAS=1 OMP=1 MKL=1"

if [[ "$KIND" == "runtime" || "$KIND" == "both" || "$KIND" == "all" ]]; then
  if [[ "$BACKEND" == "faer" || "$BACKEND" == "both" ]]; then
    log ""
    log "== faer/default =="
    run_one_thread \
      cargo bench -p tenferro-einsum --bench mps_inner_product -- \
      "${criterion_args[@]}"
  fi

  if [[ "$BACKEND" == "blas" || "$BACKEND" == "both" ]]; then
    log ""
    log "== BLAS (${BLAS_FEATURE}) =="
    run_one_thread \
      cargo bench -p tenferro-einsum --no-default-features --features "$BLAS_FEATURE" \
      --bench mps_inner_product -- \
      "${criterion_args[@]}"
  fi
fi

if [[ "$KIND" == "eager" || "$KIND" == "all" ]]; then
  if [[ "$BACKEND" == "faer" || "$BACKEND" == "both" ]]; then
    log ""
    log "== eager faer/default =="
    run_one_thread \
      cargo bench -p tenferro-einsum --bench mps_inner_product_eager -- \
      "${criterion_args[@]}"
  fi

  if [[ "$BACKEND" == "blas" || "$BACKEND" == "both" ]]; then
    log ""
    log "== eager BLAS (${BLAS_FEATURE}) =="
    run_one_thread \
      cargo bench -p tenferro-einsum --no-default-features --features "$BLAS_FEATURE" \
      --bench mps_inner_product_eager -- \
      "${criterion_args[@]}"
  fi
fi

if [[ "$KIND" == "compile" || "$KIND" == "both" || "$KIND" == "all" ]]; then
  log ""
  log "== compile-only =="
  run_one_thread \
    cargo bench -p tenferro-einsum --bench mps_inner_product_compile -- \
    "${criterion_args[@]}"
fi
