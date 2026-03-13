#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
OUT_DIR="${1:-$ROOT_DIR/target/docs-site}"

rm -rf "$OUT_DIR"

if ! command -v quarto >/dev/null 2>&1; then
  echo "quarto is required to build the docs site" >&2
  exit 1
fi

quarto render "$ROOT_DIR/docs"
python3 "$ROOT_DIR/scripts/check_docs_site.py" --site-root "$OUT_DIR"
