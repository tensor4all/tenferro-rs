#!/usr/bin/env bash

_PYTHON_RESOLVER_READY=0
_PYTHON_RESOLVER_COMMAND=()
_PYTHON_RESOLVER_SOURCE_DIR="$(pwd -P)"

_python_resolver_path() {
  local candidate="$1"
  local resolved
  local candidate_dir
  local candidate_name

  if [[ "$candidate" == */* ]]; then
    if [[ "$candidate" != /* ]]; then
      candidate="$_PYTHON_RESOLVER_SOURCE_DIR/$candidate"
    fi
    candidate_dir="${candidate%/*}"
    candidate_name="${candidate##*/}"
    [[ -n "$candidate_dir" ]] || candidate_dir="/"
    candidate_dir="$(cd -P "$candidate_dir" 2>/dev/null && pwd -P)" || return 1
    resolved="$candidate_dir/$candidate_name"
    [[ -x "$resolved" ]] || return 1
    printf '%s\n' "$resolved"
    return 0
  fi

  resolved="$(type -P "$candidate" 2>/dev/null || true)"
  [[ -n "$resolved" && -x "$resolved" ]] || return 1
  printf '%s\n' "$resolved"
}

_python_resolver_version_ok() {
  local executable="$1"

  "$executable" -c \
    'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' \
    >/dev/null 2>&1
}

_python_resolver_error() {
  printf '%s\n' "$*" >&2
  return 1
}

_python_resolver_select() {
  local candidate
  local resolved

  if [[ "${PYTHON+x}" == x ]]; then
    candidate="$PYTHON"
    if [[ -z "$candidate" ]] || ! resolved="$(_python_resolver_path "$candidate")" || \
      ! _python_resolver_version_ok "$resolved"; then
      _python_resolver_error \
        "error: \$PYTHON='$candidate' is invalid, not executable, or older than Python 3.11; refusing fallback"
      return 1
    fi
    _PYTHON_RESOLVER_COMMAND=("$resolved")
    _PYTHON_RESOLVER_READY=1
    return 0
  fi

  for candidate in python3.13 python3.12 python3.11 python3; do
    if resolved="$(_python_resolver_path "$candidate")" && \
      _python_resolver_version_ok "$resolved"; then
      _PYTHON_RESOLVER_COMMAND=("$resolved")
      _PYTHON_RESOLVER_READY=1
      return 0
    fi
  done

  if resolved="$(_python_resolver_path uv)"; then
    _PYTHON_RESOLVER_COMMAND=("$resolved" run --no-project --python 3.12 python)
    _PYTHON_RESOLVER_READY=1
    return 0
  fi

  _python_resolver_error \
    'error: Python 3.11+ is required; set literal $PYTHON to an executable, install Python 3.11+, or install uv for the optional uv fallback'
}

run_python() {
  if [[ "$_PYTHON_RESOLVER_READY" -eq 0 ]]; then
    _python_resolver_select || return
  fi
  "${_PYTHON_RESOLVER_COMMAND[@]}" "$@"
}
