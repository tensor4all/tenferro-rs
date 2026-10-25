#!/usr/bin/env bash
# Fails when a scheduler (rayon) path appears in the public API of tenferro-cpu.
# Uses `cargo public-api` (cargo install cargo-public-api; needs a nightly
# toolchain for rustdoc JSON).
set -euo pipefail
cd "$(dirname "$0")/.."
api=$(cargo public-api -p tenferro-cpu)
if grep -n 'rayon::' <<<"$api"; then
    echo "error: rayon appears in the public API of tenferro-cpu" >&2
    exit 1
fi
echo "ok: no rayon in the public API of tenferro-cpu"
