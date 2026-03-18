#!/usr/bin/env bash
set -euo pipefail

# Exit codes:
#   0  up-to-date
#   10 update available or lockfile missing
#   11 unable to check upstream state
#   12 managed files differ from the recorded lockfile hashes

QUIET=0
LOCK_PATH="ai/agent-assets.lock"
MANIFEST_PATH="ai/manifest.json"
UPSTREAM_MANIFEST_FILE=""
UPSTREAM_REVISION=""
SOURCE_REPO=""
SOURCE_REF=""
SOURCE_MANIFEST_PATH="ai/manifest.json"

usage() {
  cat <<'EOF'
Usage: bash scripts/check-agent-assets.sh [options]

Options:
  --quiet                         Print nothing when up-to-date
  --lock PATH                     Path to the local lockfile
  --manifest PATH                 Path to the local manifest
  --upstream-manifest-file PATH   Read the upstream manifest from a local file
  --upstream-revision SHA         Override the upstream bundle revision
  --source-repo OWNER/REPO        Override the upstream GitHub repository
  --source-ref REF                Override the upstream Git ref
  --source-manifest-path PATH     Override the upstream manifest path
  --help                          Show this help text
EOF
}

log() {
  printf '%s\n' "$*"
}

log_quiet_ok() {
  if [[ "$QUIET" -eq 0 ]]; then
    log "$*"
  fi
}

managed_asset_requires_execute_bit() {
  local asset_path="$1"
  [[ "$asset_path" == *.sh ]]
}

json_get() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

path, key = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
for part in key.split("."):
    data = data[part]
if isinstance(data, (dict, list)):
    raise SystemExit(f"key {key} does not resolve to a scalar")
print(data)
PY
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
    return
  fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
    return
  fi
  python3 - "$1" <<'PY'
import hashlib
import sys

path = sys.argv[1]
digest = hashlib.sha256()
with open(path, "rb") as handle:
    for chunk in iter(lambda: handle.read(65536), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

emit_lock_assets() {
  python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = json.load(handle)
for asset_id, info in sorted(data.get("assets", {}).items()):
    print(f"{asset_id}\t{info['path']}\t{info['sha256']}")
PY
}

revisions_match() {
  local lhs="$1"
  local rhs="$2"

  if [[ -z "$lhs" || -z "$rhs" ]]; then
    return 1
  fi

  [[ "$lhs" == "$rhs" || "$lhs" == "$rhs"* || "$rhs" == "$lhs"* ]]
}

fetch_remote_file() {
  gh api "repos/${SOURCE_REPO}/contents/$1?ref=${SOURCE_REF}" --jq .content \
    | python3 -c 'import base64, sys; sys.stdout.buffer.write(base64.b64decode(sys.stdin.read()))'
}

load_source_defaults() {
  if [[ -n "$SOURCE_REPO" && -n "$SOURCE_REF" ]]; then
    return
  fi
  if [[ -f "$MANIFEST_PATH" ]]; then
    if [[ -z "$SOURCE_REPO" ]]; then
      SOURCE_REPO="$(json_get "$MANIFEST_PATH" "source_repo")"
    fi
    if [[ -z "$SOURCE_REF" ]]; then
      SOURCE_REF="$(json_get "$MANIFEST_PATH" "source_ref")"
    fi
  fi
  if [[ -z "$SOURCE_REPO" ]]; then
    SOURCE_REPO="tensor4all/template-rs"
  fi
  if [[ -z "$SOURCE_REF" ]]; then
    SOURCE_REF="main"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quiet)
      QUIET=1
      shift
      ;;
    --lock)
      LOCK_PATH="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST_PATH="$2"
      shift 2
      ;;
    --upstream-manifest-file)
      UPSTREAM_MANIFEST_FILE="$2"
      shift 2
      ;;
    --upstream-revision)
      UPSTREAM_REVISION="$2"
      shift 2
      ;;
    --source-repo)
      SOURCE_REPO="$2"
      shift 2
      ;;
    --source-ref)
      SOURCE_REF="$2"
      shift 2
      ;;
    --source-manifest-path)
      SOURCE_MANIFEST_PATH="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      log "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

load_source_defaults

if [[ -n "$UPSTREAM_MANIFEST_FILE" ]]; then
  if [[ ! -f "$UPSTREAM_MANIFEST_FILE" ]]; then
    log "unable-to-check: upstream manifest file not found: $UPSTREAM_MANIFEST_FILE"
    exit 11
  fi
  if [[ -z "$UPSTREAM_REVISION" ]]; then
    if python3 - "$UPSTREAM_MANIFEST_FILE" >/dev/null 2>&1 <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = json.load(handle)
raise SystemExit(0 if "bundle_revision" in data else 1)
PY
    then
      UPSTREAM_REVISION="$(json_get "$UPSTREAM_MANIFEST_FILE" "bundle_revision")"
    else
      UPSTREAM_REVISION="local-file"
    fi
  fi
else
  if ! gh auth status >/dev/null 2>&1; then
    log "unable-to-check: gh authentication is unavailable"
    exit 11
  fi
  if [[ -z "$UPSTREAM_REVISION" ]]; then
    if ! UPSTREAM_REVISION="$(gh api "repos/${SOURCE_REPO}/commits/${SOURCE_REF}" --jq .sha 2>/dev/null)"; then
      log "unable-to-check: failed to resolve ${SOURCE_REPO}@${SOURCE_REF}"
      exit 11
    fi
  fi
  if ! fetch_remote_file "$SOURCE_MANIFEST_PATH" >/dev/null 2>&1; then
    log "unable-to-check: failed to fetch ${SOURCE_MANIFEST_PATH} from ${SOURCE_REPO}@${SOURCE_REF}"
    exit 11
  fi
fi

if [[ ! -f "$LOCK_PATH" ]]; then
  log "update-available: managed lockfile missing at $LOCK_PATH"
  exit 10
fi

LOCAL_REVISION="$(json_get "$LOCK_PATH" "bundle_revision")"
if ! revisions_match "$LOCAL_REVISION" "$UPSTREAM_REVISION"; then
  log "update-available: local=${LOCAL_REVISION} upstream=${UPSTREAM_REVISION}"
  exit 10
fi

MODIFIED=0
while IFS=$'\t' read -r _asset_id asset_path expected_hash; do
  if [[ ! -f "$asset_path" ]]; then
    log "managed-file-missing: $asset_path"
    MODIFIED=1
    continue
  fi
  actual_hash="$(sha256_file "$asset_path")"
  if [[ "$actual_hash" != "$expected_hash" ]]; then
    log "managed-file-modified: $asset_path"
    MODIFIED=1
  fi
  if managed_asset_requires_execute_bit "$asset_path" && [[ ! -x "$asset_path" ]]; then
    log "managed-file-not-executable: $asset_path"
    MODIFIED=1
  fi
done < <(emit_lock_assets "$LOCK_PATH")

if [[ "$MODIFIED" -ne 0 ]]; then
  exit 12
fi

log_quiet_ok "up-to-date: ${UPSTREAM_REVISION}"
