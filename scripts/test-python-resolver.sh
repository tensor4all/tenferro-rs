#!/usr/bin/env bash
# Tests for scripts/lib/python.sh (issue #1606).
#
# Each case builds a PATH containing only the interpreters it wants visible, so
# the resolution order is exercised rather than whatever this machine happens
# to have installed.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
LIB="$ROOT_DIR/scripts/lib/python.sh"
STUB_DIR="$(mktemp -d)"
BASE_PATH="/usr/bin:/bin:/usr/sbin:/sbin"
failures=0

cleanup() { rm -rf "$STUB_DIR"; }
trap cleanup EXIT

pass() { printf 'ok   %s\n' "$1"; }
fail() {
  printf 'FAIL %s\n     %s\n' "$1" "$2" >&2
  failures=$((failures + 1))
}

# Resolve in a subshell with a controlled PATH; echo the resolved command.
resolve_with() {
  env -i HOME="$HOME" PATH="$1" ${2:+PYTHON="$2"} bash -c "
    set -u
    . '$LIB'
    tenferro_resolve_python || exit 1
    printf '%s' \"\${TENFERRO_PYTHON[*]}\"
  " 2>/dev/null
}

resolved_version() {
  env -i HOME="$HOME" PATH="$1" ${2:+PYTHON="$2"} bash -c "
    set -u
    . '$LIB'
    py -c 'import sys; print(\"%d.%d\" % sys.version_info[:2])'
  " 2>/dev/null
}

link_stub() {
  local name="$1" target="$2" dir="$STUB_DIR/$3"
  mkdir -p "$dir"
  ln -sf "$target" "$dir/$name"
}

modern_python="$(command -v python3.13 || command -v python3.12 || command -v python3.11 || true)"
uv_bin="$(command -v uv || true)"
system_python3="$(PATH="$BASE_PATH" command -v python3 || true)"

# --- 1. an explicit $PYTHON override wins ---------------------------------
if [ -n "$modern_python" ]; then
  got="$(resolve_with "$BASE_PATH" "$modern_python")"
  if [ "$got" = "$modern_python" ]; then
    pass "PYTHON override is honoured"
  else
    fail "PYTHON override is honoured" "resolved '$got', wanted '$modern_python'"
  fi

  # --- 2. a versioned interpreter on PATH ---------------------------------
  link_stub "$(basename "$modern_python")" "$modern_python" versioned
  got="$(resolve_with "$STUB_DIR/versioned:$BASE_PATH")"
  if [ "$got" = "$(basename "$modern_python")" ]; then
    pass "pythonX.Y on PATH is preferred"
  else
    fail "pythonX.Y on PATH is preferred" "resolved '$got'"
  fi

  got="$(resolved_version "$STUB_DIR/versioned:$BASE_PATH")"
  case "$got" in
    3.1[1-9] | 3.[2-9][0-9] | [4-9].*) pass "resolved interpreter is 3.11+ ($got)" ;;
    *) fail "resolved interpreter is 3.11+" "reported '$got'" ;;
  esac
else
  printf 'skip python3.11+ cases: no versioned interpreter on PATH\n'
fi

# --- 3. uv fallback when no suitable interpreter exists --------------------
if [ -n "$uv_bin" ] && [ -n "$system_python3" ]; then
  link_stub uv "$uv_bin" uvonly
  sys_major_minor="$("$system_python3" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  case "$sys_major_minor" in
    3.9 | 3.10 | 3.[0-8] | 2.*)
      got="$(resolve_with "$STUB_DIR/uvonly:$BASE_PATH")"
      case "$got" in
        uv\ run*) pass "uv fallback is used when python3 is too old" ;;
        *) fail "uv fallback is used when python3 is too old" "resolved '$got'" ;;
      esac
      got="$(resolved_version "$STUB_DIR/uvonly:$BASE_PATH")"
      case "$got" in
        3.1[1-9] | 3.[2-9][0-9]) pass "uv fallback yields 3.11+ ($got)" ;;
        *) fail "uv fallback yields 3.11+" "reported '$got'" ;;
      esac
      ;;
    *)
      printf 'skip uv fallback: system python3 is already %s\n' "$sys_major_minor"
      ;;
  esac
else
  printf 'skip uv fallback: uv or python3 unavailable\n'
fi

# --- 4. neither: fail with a message naming the requirement ----------------
if [ -n "$system_python3" ]; then
  sys_major_minor="$("$system_python3" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  case "$sys_major_minor" in
    3.9 | 3.10 | 3.[0-8] | 2.*)
      # BASE_PATH has the too-old python3 and no uv.
      message="$(env -i HOME="$HOME" PATH="$BASE_PATH" bash -c "
        set -u
        . '$LIB'
        tenferro_resolve_python
      " 2>&1)"
      status=$?
      if [ "$status" -ne 0 ] && printf '%s' "$message" | grep -q "3.11+"; then
        pass "missing interpreter fails with an explanatory message"
      else
        fail "missing interpreter fails with an explanatory message" \
          "status=$status message=$message"
      fi
      ;;
    *)
      printf 'skip missing-interpreter case: system python3 is %s\n' "$sys_major_minor"
      ;;
  esac
fi

if [ "$failures" -ne 0 ]; then
  printf '%d test(s) failed\n' "$failures" >&2
  exit 1
fi
printf 'all python resolver tests passed\n'
