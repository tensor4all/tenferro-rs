#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
OUT_DIR="${1:-$ROOT_DIR/target/docs-site}"
API_DIR="$OUT_DIR/api"

rm -rf "$OUT_DIR"
mkdir -p "$API_DIR"

echo "[1/4] Building rustdoc"
rm -rf "$ROOT_DIR/target/doc"
cargo doc --workspace --no-deps

echo "[2/4] Copying rustdoc output"
cp -a "$ROOT_DIR/target/doc/." "$API_DIR/"

echo "[3/4] Generating dependency graph and API index"
if command -v dot >/dev/null 2>&1; then
  python3 "$ROOT_DIR/scripts/gen_dep_graph.py" --root-dir "$ROOT_DIR" \
    | dot -Tsvg > "$API_DIR/dep_graph.svg"
else
  echo "  Warning: graphviz (dot) not found; dependency graph SVG skipped."
  if [ "${CI:-}" = "true" ]; then
    echo "  graphviz is required in CI to generate the dependency graph."
    exit 1
  fi
fi

# Convert api_index.md to HTML fragment
OVERVIEW_HTML=""
API_INDEX_MD="$ROOT_DIR/docs/api_index.md"
if [ -f "$API_INDEX_MD" ]; then
  if command -v pandoc >/dev/null 2>&1; then
    OVERVIEW_HTML=$(pandoc "$API_INDEX_MD" -f gfm -t html5)
  else
    if [ "${CI:-}" = "true" ]; then
      echo "pandoc is required in CI to render docs/api_index.md."
      exit 1
    fi
    echo "  Warning: pandoc not found; overview section will be plain text."
    OVERVIEW_HTML="<pre>$(sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g' "$API_INDEX_MD")</pre>"
  fi
fi

{
  cat <<'HEADER'
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>tenferro-rs API docs</title>
    <style>
      body { font-family: "IBM Plex Sans", "Segoe UI", sans-serif; margin: 2rem auto; max-width: 56rem; padding: 0 1rem; color: #14231f; }
      h1 { margin-bottom: 0.5rem; }
      h2 { margin-top: 2rem; color: #2e3d38; }
      h3 { color: #2e3d38; }
      p { color: #3a4a45; line-height: 1.6; }
      ul { line-height: 1.8; }
      a { color: #0c7a5a; text-decoration: none; }
      a:hover { text-decoration: underline; }
      code { font-family: "IBM Plex Mono", "SFMono-Regular", monospace; }
      pre { background: #f4f7f5; padding: 1rem; border-radius: 6px; overflow-x: auto; }
      .dep-graph { margin: 1.5rem 0; text-align: center; }
      .dep-graph img { max-width: 100%; height: auto; }
    </style>
  </head>
  <body>
HEADER

  # Insert overview from Markdown
  if [ -n "$OVERVIEW_HTML" ]; then
    echo "$OVERVIEW_HTML"
  fi

  cat <<'FOOTER'
  </body>
</html>
FOOTER
} >"$API_DIR/index.html"

echo "[4/4] Generating site top page redirect"
cat <<'REDIRECT' >"$OUT_DIR/index.html"
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="refresh" content="0; url=api/index.html" />
    <title>Redirecting to API docs</title>
  </head>
  <body>
    <p>Redirecting to <a href="api/index.html">API documentation</a>.</p>
  </body>
</html>
REDIRECT

echo "Done: $OUT_DIR"
