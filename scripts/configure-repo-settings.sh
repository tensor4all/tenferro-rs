#!/usr/bin/env bash
set -euo pipefail

REPO=""
SETTINGS_PATH="ai/repo-settings.json"

usage() {
  cat <<'EOF'
Usage: bash scripts/configure-repo-settings.sh [options]

Options:
  --repo OWNER/REPO     Repository to configure (defaults to the current gh repo)
  --settings PATH       Repository settings JSON (default: ai/repo-settings.json)
  --help                Show this help text
EOF
}

log() {
  printf '%s\n' "$*"
}

resolve_settings_path() {
  if [[ "$SETTINGS_PATH" == "ai/repo-settings.json" && -f "ai/repo-settings.local.json" ]]; then
    printf '%s\n' "ai/repo-settings.local.json"
    return
  fi
  if [[ -f "$SETTINGS_PATH" ]]; then
    printf '%s\n' "$SETTINGS_PATH"
    return
  fi
  if [[ "$SETTINGS_PATH" == "ai/repo-settings.json" && -f "ai/vendor/template-rs/repo-settings.json" ]]; then
    printf '%s\n' "ai/vendor/template-rs/repo-settings.json"
    return
  fi
  printf '%s\n' "$SETTINGS_PATH"
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

build_branch_protection_payload() {
  python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    settings = json.load(handle)

payload = {
    "required_status_checks": {
        "strict": settings["required_status_checks"]["strict"],
        "contexts": settings["required_status_checks"]["contexts"],
    },
    "enforce_admins": False,
    "required_pull_request_reviews": None,
    "restrictions": None,
    "required_linear_history": False,
    "allow_force_pushes": False,
    "allow_deletions": False,
    "block_creations": False,
    "required_conversation_resolution": True,
    "lock_branch": False,
    "allow_fork_syncing": False,
}
print(json.dumps(payload))
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO="$2"
      shift 2
      ;;
    --settings)
      SETTINGS_PATH="$2"
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

SETTINGS_PATH="$(resolve_settings_path)"

if [[ ! -f "$SETTINGS_PATH" ]]; then
  log "settings file not found: $SETTINGS_PATH"
  exit 1
fi

if [[ -z "$REPO" ]]; then
  REPO="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"
fi

DEFAULT_BRANCH="$(json_get "$SETTINGS_PATH" "default_branch")"
ALLOW_AUTO_MERGE="$(json_get "$SETTINGS_PATH" "allow_auto_merge")"
DELETE_BRANCH_ON_MERGE="$(json_get "$SETTINGS_PATH" "delete_branch_on_merge")"

repo_edit_args=()
if [[ "$ALLOW_AUTO_MERGE" == "True" ]]; then
  repo_edit_args+=(--enable-auto-merge)
else
  repo_edit_args+=(--enable-auto-merge=false)
fi
if [[ "$DELETE_BRANCH_ON_MERGE" == "True" ]]; then
  repo_edit_args+=(--delete-branch-on-merge)
else
  repo_edit_args+=(--delete-branch-on-merge=false)
fi

gh repo edit "$REPO" "${repo_edit_args[@]}"
build_branch_protection_payload "$SETTINGS_PATH" \
  | gh api -X PUT "repos/$REPO/branches/$DEFAULT_BRANCH/protection" \
      -H "Accept: application/vnd.github+json" \
      --input -

bash scripts/check-repo-settings.sh --repo "$REPO" --settings "$SETTINGS_PATH"
