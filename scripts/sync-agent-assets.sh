#!/usr/bin/env bash
set -euo pipefail

FORCE=0
LOCK_PATH="ai/agent-assets.lock"
MANIFEST_PATH="ai/manifest.json"
UPSTREAM_MANIFEST_FILE=""
UPSTREAM_REVISION=""
SOURCE_REPO=""
SOURCE_REF=""
SOURCE_MANIFEST_PATH="ai/manifest.json"
SOURCE_ROOT=""

usage() {
  cat <<'EOF'
Usage: bash scripts/sync-agent-assets.sh [options]

Options:
  --force                         Overwrite locally modified managed files
  --lock PATH                     Path to the local lockfile
  --manifest PATH                 Path to the local manifest
  --upstream-manifest-file PATH   Read the upstream manifest from a local file
  --upstream-revision SHA         Override the upstream bundle revision
  --source-repo OWNER/REPO        Override the upstream GitHub repository
  --source-ref REF                Override the upstream Git ref
  --source-manifest-path PATH     Override the upstream manifest path
  --source-root PATH              Root directory for local-file sync mode
  --help                          Show this help text
EOF
}

log() {
  printf '%s\n' "$*"
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

emit_manifest_assets() {
  python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = json.load(handle)
for asset in data["managed_assets"]:
    print(f"{asset['id']}\t{asset['source']}\t{asset['target']}")
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

write_lock() {
  python3 - "$LOCK_PATH" "$SOURCE_REPO" "$SOURCE_REF" "$UPSTREAM_REVISION" "$1" <<'PY'
import json
import sys

lock_path, source_repo, source_ref, revision, manifest_path = sys.argv[1:]

assets = {}
with open(manifest_path, "r", encoding="utf-8") as handle:
    for line in handle:
        asset_id, target_path, asset_hash = line.rstrip("\n").split("\t")
        assets[asset_id] = {"path": target_path, "sha256": asset_hash}

data = {
    "source_repo": source_repo,
    "source_ref": source_ref,
    "bundle_revision": revision,
    "synced_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "assets": assets,
}
with open(lock_path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
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
    --source-root)
      SOURCE_ROOT="$2"
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

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
UPSTREAM_MANIFEST_TMP="$TMP_DIR/upstream-manifest.json"
LOCK_RECORDS="$TMP_DIR/lock-assets.tsv"

if [[ -n "$UPSTREAM_MANIFEST_FILE" ]]; then
  if [[ ! -f "$UPSTREAM_MANIFEST_FILE" ]]; then
    log "upstream manifest file not found: $UPSTREAM_MANIFEST_FILE"
    exit 1
  fi
  cp "$UPSTREAM_MANIFEST_FILE" "$UPSTREAM_MANIFEST_TMP"
  if [[ -z "$SOURCE_ROOT" ]]; then
    SOURCE_ROOT="$(cd "$(dirname "$UPSTREAM_MANIFEST_FILE")/.." && pwd)"
  fi
  if [[ -z "$UPSTREAM_REVISION" ]]; then
    if python3 - "$UPSTREAM_MANIFEST_TMP" >/dev/null 2>&1 <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = json.load(handle)
raise SystemExit(0 if "bundle_revision" in data else 1)
PY
    then
      UPSTREAM_REVISION="$(json_get "$UPSTREAM_MANIFEST_TMP" "bundle_revision")"
    else
      UPSTREAM_REVISION="local-file"
    fi
  fi
else
  if ! gh auth status >/dev/null 2>&1; then
    log "gh authentication is unavailable"
    exit 1
  fi
  fetch_remote_file "$SOURCE_MANIFEST_PATH" >"$UPSTREAM_MANIFEST_TMP"
  if [[ -z "$UPSTREAM_REVISION" ]]; then
    UPSTREAM_REVISION="$(gh api "repos/${SOURCE_REPO}/commits/${SOURCE_REF}" --jq .sha)"
  fi
fi

if [[ -f "$LOCK_PATH" && "$FORCE" -eq 0 ]]; then
  while IFS=$'\t' read -r _asset_id asset_path expected_hash; do
    if [[ ! -f "$asset_path" ]]; then
      continue
    fi
    actual_hash="$(sha256_file "$asset_path")"
    if [[ "$actual_hash" != "$expected_hash" ]]; then
      log "refusing to overwrite locally modified managed file: $asset_path"
      log "rerun with --force if you want to replace managed assets"
      exit 1
    fi
  done < <(emit_lock_assets "$LOCK_PATH")
fi

: >"$LOCK_RECORDS"
while IFS=$'\t' read -r asset_id asset_source asset_target; do
  mkdir -p "$(dirname "$asset_target")"
  tmp_asset="$TMP_DIR/$asset_id"
  if [[ -n "$UPSTREAM_MANIFEST_FILE" ]]; then
    cp "$SOURCE_ROOT/$asset_source" "$tmp_asset"
  else
    fetch_remote_file "$asset_source" >"$tmp_asset"
  fi
  mv "$tmp_asset" "$asset_target"
  asset_hash="$(sha256_file "$asset_target")"
  printf '%s\t%s\t%s\n' "$asset_id" "$asset_target" "$asset_hash" >>"$LOCK_RECORDS"
done < <(emit_manifest_assets "$UPSTREAM_MANIFEST_TMP")

write_lock "$LOCK_RECORDS"
log "synced-agent-assets: ${SOURCE_REPO}@${SOURCE_REF} (${UPSTREAM_REVISION})"
