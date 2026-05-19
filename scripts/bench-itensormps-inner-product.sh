#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
DEFAULT_JULIA_PROJECT="$ROOT_DIR/../tensor4all-rs/benchmarks/julia"
JULIA_PROJECT_DIR="${JULIA_PROJECT_DIR:-$DEFAULT_JULIA_PROJECT}"

if [[ ! -f "$JULIA_PROJECT_DIR/Project.toml" ]]; then
  cat >&2 <<EOF
Julia project not found: $JULIA_PROJECT_DIR

Set JULIA_PROJECT_DIR to a Julia environment that has ITensors and ITensorMPS.
EOF
  exit 1
fi

RAYON_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  JULIA_NUM_THREADS=1 \
  julia --startup-file=no --project="$JULIA_PROJECT_DIR" \
  "$ROOT_DIR/scripts/bench-itensormps-inner-product.jl" "$@"
