# shellcheck shell=bash
#
# Resolve a Python interpreter new enough for this repository's scripts.
#
# The tooling needs Python 3.11 or newer — `enum.StrEnum` in
# `scripts/ci/change_policy.py`, `tomllib` in the doc/API consistency checks —
# but the shell entry points used to invoke a bare `python3`. On macOS that is
# 3.9, so `check-pr-fast.sh` died inside `change_policy.py` before running a
# single check, while CI stayed green on its runner default (3.12). See #1606.
#
# Resolution order, first hit wins:
#   1. $PYTHON            — an explicit contributor override
#   2. python3.13/12/11   — a suitable interpreter already on PATH
#   3. python3            — when it is itself 3.11+
#   4. uv                 — `uv run` with a managed 3.12
#
# `uv` is a FALLBACK, not a prerequisite: CI keeps using its default
# interpreter, contributors with a modern `python3` are unaffected, and a
# machine with neither gets an error naming both remedies rather than an
# AttributeError from deep inside an unrelated module.
#
# Usage:
#   . "$(dirname "${BASH_SOURCE[0]}")/lib/python.sh"
#   py scripts/some_check.py --flag
#   py - <<'PY'      # heredocs work: `py` forwards every argument
#   ...
#   PY

TENFERRO_MIN_PYTHON="3.11"

# Populated by `tenferro_resolve_python`; an array because the uv fallback is a
# multi-word command.
TENFERRO_PYTHON=()

tenferro_python_is_new_enough() {
  "$@" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' \
    >/dev/null 2>&1
}

tenferro_resolve_python() {
  if [ "${#TENFERRO_PYTHON[@]}" -gt 0 ]; then
    return 0
  fi

  if [ -n "${PYTHON:-}" ]; then
    # Honour the override even if it is a multi-word command.
    # shellcheck disable=SC2206
    local override=($PYTHON)
    if tenferro_python_is_new_enough "${override[@]}"; then
      TENFERRO_PYTHON=("${override[@]}")
      return 0
    fi
    echo "PYTHON=$PYTHON is older than $TENFERRO_MIN_PYTHON" >&2
    return 1
  fi

  local candidate
  for candidate in python3.13 python3.12 python3.11; do
    if command -v "$candidate" >/dev/null 2>&1 &&
      tenferro_python_is_new_enough "$candidate"; then
      TENFERRO_PYTHON=("$candidate")
      return 0
    fi
  done

  if command -v python3 >/dev/null 2>&1 && tenferro_python_is_new_enough python3; then
    TENFERRO_PYTHON=(python3)
    return 0
  fi

  # `--no-project` keeps uv from trying to sync a project that does not exist
  # here; the managed interpreter is downloaded once and cached.
  if command -v uv >/dev/null 2>&1; then
    local uv_python=(uv run --no-project --python 3.12 python)
    if tenferro_python_is_new_enough "${uv_python[@]}"; then
      TENFERRO_PYTHON=("${uv_python[@]}")
      return 0
    fi
  fi

  {
    echo "No Python $TENFERRO_MIN_PYTHON+ interpreter found."
    echo "This repository's scripts need it for enum.StrEnum and tomllib."
    echo "Fix by either:"
    echo "  - installing one, e.g. 'brew install python@3.12', or"
    echo "  - installing uv (https://docs.astral.sh/uv/), which this script"
    echo "    will then use automatically, or"
    echo "  - pointing PYTHON at a suitable interpreter."
  } >&2
  return 1
}

# Run a Python script with the resolved interpreter.
py() {
  tenferro_resolve_python || exit 1
  "${TENFERRO_PYTHON[@]}" "$@"
}
