#!/usr/bin/env bash
set -euo pipefail

BASE_BRANCH="main"
TITLE=""
BODY_FILE=""
ALLOW_STALE=0
AUTO_MERGE=1
DRAFT=0
AI_TOOL_NAME=""
AI_TOOL_URL=""

usage() {
  cat <<'EOF'
Usage: bash scripts/create-pr.sh [options]

Options:
  --base BRANCH          Base branch for the pull request (default: main)
  --title TITLE          Pull request title (defaults to the latest commit subject)
  --body-file PATH       Markdown body file to pass to gh pr create
  --allow-stale          Allow stale or unverified agent assets
  --no-auto-merge        Do not enable auto-merge after PR creation
  --draft                Create the PR as a draft
  --ai-tool-name NAME    Attribution display name, for example "Claude Code"
  --ai-tool-url URL      Attribution URL paired with --ai-tool-name
  --help                 Show this help text
EOF
}

log() {
  printf '%s\n' "$*"
}

require_clean_tree() {
  if [[ -n "$(git status --short)" ]]; then
    log "working tree is not clean"
    exit 1
  fi
}

ensure_body_file() {
  if [[ -n "$BODY_FILE" ]]; then
    return
  fi

  BODY_FILE="$(mktemp)"
  trap 'rm -f "$BODY_FILE"' EXIT
  {
    printf '## Summary\n\n'
    git log --format='- %s' "${BASE_BRANCH}..HEAD" 2>/dev/null || true
    printf '\n## Verification\n\n'
    printf -- '- `cargo fmt --all --check`\n'
    printf -- '- `cargo test --workspace --release`\n'
    printf -- '- `cargo llvm-cov --workspace --json --output-path coverage.json`\n'
    printf -- '- `python3 scripts/check-coverage.py coverage.json`\n'
    printf -- '- `cargo doc --workspace --no-deps`\n'
    printf -- '- `python3 scripts/check-docs-site.py`\n'
    printf '\n## Documentation\n\n'
    printf -- '- Reviewed `README.md`, `docs/design/**`, `docs/api_index.md`, and public rustdoc for consistency.\n'
  } >"$BODY_FILE"
}

append_ai_attribution() {
  if [[ -z "$AI_TOOL_NAME" || -z "$AI_TOOL_URL" ]]; then
    return
  fi
  if grep -qi 'Generated with \[' "$BODY_FILE"; then
    return
  fi
  {
    printf '\nGenerated with [%s](%s)\n' "$AI_TOOL_NAME" "$AI_TOOL_URL"
  } >>"$BODY_FILE"
}

run_required_checks() {
  cargo fmt --all --check
  cargo test --workspace --release
  cargo llvm-cov --workspace --json --output-path coverage.json
  python3 scripts/check-coverage.py coverage.json
  cargo doc --workspace --no-deps
  python3 scripts/check-docs-site.py
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)
      BASE_BRANCH="$2"
      shift 2
      ;;
    --title)
      TITLE="$2"
      shift 2
      ;;
    --body-file)
      BODY_FILE="$2"
      shift 2
      ;;
    --allow-stale)
      ALLOW_STALE=1
      shift
      ;;
    --no-auto-merge)
      AUTO_MERGE=0
      shift
      ;;
    --draft)
      DRAFT=1
      shift
      ;;
    --ai-tool-name)
      AI_TOOL_NAME="$2"
      shift 2
      ;;
    --ai-tool-url)
      AI_TOOL_URL="$2"
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

current_branch="$(git branch --show-current)"
if [[ -z "$current_branch" ]]; then
  log "not on a named branch"
  exit 1
fi
if [[ "$current_branch" == "main" || "$current_branch" == "master" ]]; then
  log "refusing to create a PR from ${current_branch}"
  exit 1
fi

require_clean_tree

set +e
bash scripts/check-agent-assets.sh --quiet
asset_status=$?
set -e
if [[ "$asset_status" -ne 0 && "$ALLOW_STALE" -eq 0 ]]; then
  log "agent assets are stale or could not be verified; rerun with --allow-stale to continue"
  exit 1
fi

bash scripts/check-repo-settings.sh --quiet

log "docs gate: review README.md, docs/design/**, docs/api_index.md, and public rustdoc before continuing"
run_required_checks

if [[ -z "$TITLE" ]]; then
  TITLE="$(git log -1 --format=%s)"
fi

ensure_body_file
append_ai_attribution

if git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' >/dev/null 2>&1; then
  git push
else
  git push -u origin "$current_branch"
fi

create_args=(pr create --base "$BASE_BRANCH" --title "$TITLE" --body-file "$BODY_FILE")
if [[ "$DRAFT" -eq 1 ]]; then
  create_args+=(--draft)
fi

pr_url="$(gh "${create_args[@]}")"
log "$pr_url"

if [[ "$AUTO_MERGE" -eq 1 ]]; then
  gh pr merge --auto --squash --delete-branch
fi
