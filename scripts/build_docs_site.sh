#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
OUT_DIR="${1:-$ROOT_DIR/target/docs-site}"
RUSTDOC_DIR="$OUT_DIR/rustdoc"
DOC_ROOT="$ROOT_DIR/target/doc"

rm -rf "$OUT_DIR"
mkdir -p "$RUSTDOC_DIR"

echo "[1/4] Building rustdoc"
rm -rf "$DOC_ROOT"
cargo doc --workspace --no-deps

echo "[2/4] Copying rustdoc output"
cp -a "$DOC_ROOT/." "$RUSTDOC_DIR/"

echo "[3/4] Generating optional dependency graph"
if [[ -x "$ROOT_DIR/scripts/gen_dep_graph.py" || -f "$ROOT_DIR/scripts/gen_dep_graph.py" ]]; then
  if command -v dot >/dev/null 2>&1; then
    python3 "$ROOT_DIR/scripts/gen_dep_graph.py" --root-dir "$ROOT_DIR" | dot -Tsvg > "$RUSTDOC_DIR/dep_graph.svg"
  else
    echo "  Warning: graphviz (dot) not found; dependency graph skipped."
    if [[ "${CI:-}" == "true" ]]; then
      echo "  graphviz is required in CI to render the dependency graph."
      exit 1
    fi
  fi
else
  echo "  No scripts/gen_dep_graph.py found; skipping dependency graph."
fi

echo "[4/4] Rendering Quarto docs site"
if [[ -f "$ROOT_DIR/docs/_quarto.yml" ]]; then
  if command -v quarto >/dev/null 2>&1; then
    quarto render "$ROOT_DIR/docs"
  else
    echo "  Warning: quarto not found; docs site skipped."
    if [[ "${CI:-}" == "true" ]]; then
      echo "  quarto is required in CI to render the docs site."
      exit 1
    fi
  fi
else
  echo "  No docs/_quarto.yml found; skipping docs site render."
fi

python3 "$ROOT_DIR/scripts/check-docs-site.py" --root-dir "$ROOT_DIR"

echo "Done: $OUT_DIR"
