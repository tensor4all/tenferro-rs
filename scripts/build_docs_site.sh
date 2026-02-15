#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
OUT_DIR="${1:-$ROOT_DIR/target/docs-site}"
DESIGN_DIR="$OUT_DIR/design"
API_DIR="$OUT_DIR/api"

rm -rf "$OUT_DIR"
mkdir -p "$DESIGN_DIR" "$API_DIR"

echo "[1/4] Building rustdoc"
cargo doc --workspace --no-deps

echo "[2/4] Copying rustdoc output"
cp -a "$ROOT_DIR/target/doc/." "$API_DIR/"

echo "[3/4] Converting docs/design markdown to HTML"
for md in "$ROOT_DIR"/docs/design/*.md; do
  base="$(basename "$md" .md)"
  html="$DESIGN_DIR/$base.html"

  if command -v pandoc >/dev/null 2>&1; then
    pandoc "$md" -f gfm -t html5 -o "$html" --standalone
  else
    {
      printf '<!doctype html><html><head><meta charset="utf-8"><title>%s</title></head><body><pre>\n' "$base"
      sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g' "$md"
      printf '\n</pre></body></html>\n'
    } >"$html"
  fi
done

echo "[4/4] Copying site top page"
cp "$ROOT_DIR/docs/site_index.html" "$OUT_DIR/index.html"

echo "Done: $OUT_DIR"
