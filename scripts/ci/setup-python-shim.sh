#!/usr/bin/env bash
# Put the uv-managed interpreter on PATH for the rest of a CI job (issue #1606).
#
# Workflow steps invoke `python3 …` directly rather than going through the
# shell entry points, so sourcing `scripts/lib/python.sh` in one step would be
# lost by the next. Appending the shim directory to $GITHUB_PATH makes every
# later step in the job use the interpreter pinned by `.python-version`, which
# is what gives local runs and CI the same version.
set -euo pipefail

# shellcheck source=scripts/lib/python.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd -P)/python.sh"

if [ -z "${TENFERRO_PYTHON_SHIM_DIR:-}" ]; then
  echo "no shim installed; PATH already provides the pinned interpreter" >&2
  exit 0
fi

if [ -n "${GITHUB_PATH:-}" ]; then
  echo "$TENFERRO_PYTHON_SHIM_DIR" >>"$GITHUB_PATH"
fi
echo "python3 -> $(python3 -c 'import sys; print(sys.version.split()[0])')"
