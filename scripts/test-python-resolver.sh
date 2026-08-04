#!/usr/bin/env bash
# Tests for scripts/lib/python.sh (issue #1606).
#
# Each case builds a PATH exposing only the tools it wants visible, so the
# behaviour under test is exercised rather than whatever this machine happens
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

# Source the lib with a controlled environment and run `script` afterwards.
in_env() {
  local path="$1" python_override="$2" script="$3"
  env -i HOME="$HOME" PATH="$path" ${python_override:+PYTHON="$python_override"} \
    bash -c ". '$LIB'
$script" 2>&1
}

link_stub() {
  local name="$1" target="$2" dir="$STUB_DIR/$3"
  mkdir -p "$dir"
  ln -sf "$target" "$dir/$name"
}

uv_bin="$(command -v uv || true)"
modern_python="$(command -v python3.13 || command -v python3.12 || command -v python3.11 || true)"
pinned="$(tr -d '[:space:]' <"$ROOT_DIR/.python-version")"

# --- 1. uv supplies the pinned interpreter --------------------------------
if [ -n "$uv_bin" ]; then
  link_stub uv "$uv_bin" uvonly
  got="$(in_env "$STUB_DIR/uvonly:$BASE_PATH" "" \
    'python3 -c "import sys; print(\"%d.%d\" % sys.version_info[:2])"')"
  if [ "$got" = "$pinned" ]; then
    pass "python3 resolves to the pinned interpreter ($pinned)"
  else
    fail "python3 resolves to the pinned interpreter" "got '$got', wanted '$pinned'"
  fi

  # The property the PATH shim exists for: a CHILD process sees it too, which
  # is what `scripts/ci/run_profile.py`'s `shell=True` commands depend on.
  got="$(in_env "$STUB_DIR/uvonly:$BASE_PATH" "" \
    'python3 -c "import subprocess,sys; print(subprocess.run([\"bash\",\"-c\",\"python3 -c \\\"import sys;print(sys.version_info[0],sys.version_info[1])\\\"\"],capture_output=True,text=True).stdout.strip())"')"
  case "$got" in
    "${pinned%%.*} ${pinned#*.}") pass "the shim survives into child processes" ;;
    *) fail "the shim survives into child processes" "child reported '$got'" ;;
  esac
else
  printf 'skip uv cases: uv is not installed\n'
fi

# --- 2. an explicit $PYTHON override wins ---------------------------------
if [ -n "$modern_python" ]; then
  got="$(in_env "$BASE_PATH" "$modern_python" \
    'python3 -c "import sys; print(sys.executable)"')"
  # The shim forwards to the override, so `sys.executable` is that interpreter.
  if [ "$got" = "$modern_python" ]; then
    pass "PYTHON override is honoured"
  else
    fail "PYTHON override is honoured" "sys.executable='$got', wanted '$modern_python'"
  fi
else
  printf 'skip PYTHON override case: no versioned interpreter on PATH\n'
fi

# --- 3. an override that is too old is rejected loudly --------------------
system_python3="$(PATH="$BASE_PATH" command -v python3 || true)"
if [ -n "$system_python3" ]; then
  sys_version="$("$system_python3" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  case "$sys_version" in
    3.9 | 3.10 | 3.[0-8] | 2.*)
      message="$(in_env "$BASE_PATH" "$system_python3" 'true')"
      if printf '%s' "$message" | grep -q "older than"; then
        pass "a too-old PYTHON override is rejected"
      else
        fail "a too-old PYTHON override is rejected" "message='$message'"
      fi

      # --- 4. no uv and no override: name uv in the error ------------------
      message="$(in_env "$BASE_PATH" "" 'true')"
      if printf '%s' "$message" | grep -q "uv is required"; then
        pass "missing uv fails with an explanatory message"
      else
        fail "missing uv fails with an explanatory message" "message='$message'"
      fi
      ;;
    *)
      printf 'skip too-old cases: system python3 is %s\n' "$sys_version"
      ;;
  esac
fi

if [ "$failures" -ne 0 ]; then
  printf '%d test(s) failed\n' "$failures" >&2
  exit 1
fi
printf 'all python resolver tests passed\n'
