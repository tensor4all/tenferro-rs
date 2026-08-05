#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$ROOT_DIR/scripts/lib/python.sh"
SITE_DIR="${1:-$ROOT_DIR/target/docs-site}"
PORT="${PORT:-8000}"

if [ ! -f "$SITE_DIR/index.html" ]; then
  echo "docs site not found: $SITE_DIR"
  echo "run ./scripts/build_docs_site.sh first"
  exit 1
fi

echo "serving $SITE_DIR at http://127.0.0.1:$PORT"
cd "$SITE_DIR"
run_python -m http.server "$PORT"
