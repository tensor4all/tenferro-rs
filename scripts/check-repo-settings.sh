#!/usr/bin/env bash
set -euo pipefail

REPO=""
SETTINGS_PATH="ai/repo-settings.json"
QUIET=0

usage() {
  cat <<'EOF'
Usage: bash scripts/check-repo-settings.sh [options]

Options:
  --repo OWNER/REPO     Repository to inspect (defaults to the current gh repo)
  --settings PATH       Repository settings JSON (default: ai/repo-settings.json)
  --quiet               Suppress the success message
  --help                Show this help text
EOF
}

log() {
  printf '%s\n' "$*"
}

gh_api_optional() {
  local output
  if output="$(gh api "$@" 2>/dev/null)"; then
    printf '%s\n' "$output"
  fi
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

json_get_optional() {
  python3 - "$1" "$2" "$3" <<'PY'
import json
import sys

path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
for part in key.split("."):
    if not isinstance(data, dict) or part not in data:
        print(default)
        raise SystemExit(0)
    data = data[part]
if isinstance(data, (dict, list)):
    raise SystemExit(f"key {key} does not resolve to a scalar")
print(data)
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
    --quiet)
      QUIET=1
      shift
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
EXPECTED_AUTO_MERGE="$(json_get "$SETTINGS_PATH" "allow_auto_merge")"
EXPECTED_DELETE_BRANCH="$(json_get "$SETTINGS_PATH" "delete_branch_on_merge")"
EXPECTED_STRICT="$(json_get "$SETTINGS_PATH" "required_status_checks.strict")"
EXPECTED_PAGES_ENABLED="$(json_get_optional "$SETTINGS_PATH" "pages.enabled" "False")"
EXPECTED_PAGES_BUILD_TYPE="$(json_get_optional "$SETTINGS_PATH" "pages.build_type" "")"

REPO_JSON="$(gh api "repos/$REPO")"
PROTECTION_JSON="$(gh_api_optional "repos/$REPO/branches/$DEFAULT_BRANCH/protection")"
PAGES_JSON="$(gh_api_optional "repos/$REPO/pages")"

python3 - "$SETTINGS_PATH" "$REPO_JSON" "$PROTECTION_JSON" "$PAGES_JSON" "$EXPECTED_AUTO_MERGE" "$EXPECTED_DELETE_BRANCH" "$EXPECTED_STRICT" "$EXPECTED_PAGES_ENABLED" "$EXPECTED_PAGES_BUILD_TYPE" <<'PY'
import json
import sys

(
    settings_path,
    repo_json,
    protection_json,
    pages_json,
    expected_auto_merge,
    expected_delete_branch,
    expected_strict,
    expected_pages_enabled,
    expected_pages_build_type,
) = sys.argv[1:]

with open(settings_path, "r", encoding="utf-8") as handle:
    settings = json.load(handle)

repo = json.loads(repo_json)
protection = json.loads(protection_json) if protection_json.strip() else None
pages = json.loads(pages_json) if pages_json.strip() else None

errors = []
if repo.get("allow_auto_merge") != (expected_auto_merge == "True"):
    errors.append("allow_auto_merge mismatch")
if repo.get("delete_branch_on_merge") != (expected_delete_branch == "True"):
    errors.append("delete_branch_on_merge mismatch")
if protection is None:
    errors.append("default branch is not protected")
else:
    actual_checks = protection.get("required_status_checks") or {}
    actual_contexts = set(actual_checks.get("contexts") or [])
    expected_contexts = set(settings["required_status_checks"]["contexts"])
    if actual_checks.get("strict") != (expected_strict == "True"):
        errors.append("required_status_checks.strict mismatch")
    if actual_contexts != expected_contexts:
        missing = sorted(expected_contexts - actual_contexts)
        extra = sorted(actual_contexts - expected_contexts)
        if missing:
            errors.append("missing required status checks: " + ", ".join(missing))
        if extra:
            errors.append("unexpected protected checks: " + ", ".join(extra))

if expected_pages_enabled == "True":
    if pages is None:
        errors.append("pages is not enabled")
    elif expected_pages_build_type and pages.get("build_type") != expected_pages_build_type:
        errors.append(
            "pages.build_type mismatch: "
            f"expected {expected_pages_build_type}, got {pages.get('build_type')!r}"
        )
if errors:
    for error in errors:
        print(error)
    raise SystemExit(1)
PY

if [[ "$QUIET" -eq 0 ]]; then
  log "repo-settings-ok: $REPO"
fi
