#!/usr/bin/env bash
set -euo pipefail

# Make `python3` resolve to a 3.11+ interpreter (issue #1606).
# shellcheck source=scripts/lib/python.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib/python.sh"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SITE_DIR="${1:-$ROOT_DIR/target/docs-site}"
PORT="${PORT:-8000}"

if [ ! -f "$SITE_DIR/index.html" ]; then
  echo "docs site not found: $SITE_DIR"
  echo "run ./scripts/build_docs_site.sh first"
  exit 1
fi

echo "serving $SITE_DIR at http://127.0.0.1:$PORT"
cd "$SITE_DIR"
python3 -m http.server "$PORT"
