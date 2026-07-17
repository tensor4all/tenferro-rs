#!/usr/bin/env bash
set -euo pipefail

BASE_BRANCH="main"
TITLE=""
BODY_FILE=""
AUTO_MERGE=1
DRAFT=0
AI_TOOL_NAME=""
AI_TOOL_URL=""
FOCUSED_TESTS=()

usage() {
  cat <<'EOF'
Usage: bash scripts/create-pr.sh [options]

Options:
  --base BRANCH          Base branch for the pull request (default: main)
  --title TITLE          Pull request title (defaults to the latest commit subject)
  --body-file PATH       Markdown body file to pass to gh pr create
  --test COMMAND         Focused local verification command; repeatable
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
    printf -- '- Focused local verification:\n'
    if [[ "${#FOCUSED_TESTS[@]}" -eq 0 ]]; then
      printf -- '  - `bash scripts/check-pr-fast.sh` (documentation-only path)\n'
    else
      for command in "${FOCUSED_TESTS[@]}"; do
        printf -- '  - `%s`\n' "$command"
      done
    fi
    printf -- '- `python3 scripts/repository-rules-review.py --base origin/%s --head HEAD`\n' "$BASE_BRANCH"
    printf '\n## Documentation\n\n'
    printf -- '- Reviewed `README.md`, `docs/design/**`, `docs/api/index.md`, and public rustdoc for consistency.\n'
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
  fast_gate_args=(
    --base "origin/${BASE_BRANCH}"
    --coverage-reviewed
  )
  for command in "${FOCUSED_TESTS[@]}"; do
    fast_gate_args+=(--test "$command")
  done
  bash scripts/check-pr-fast.sh "${fast_gate_args[@]}"
  python3 scripts/repository-rules-review.py \
    --base "origin/${BASE_BRANCH}" \
    --head HEAD \
    --output-json /tmp/repository-rules-review.json
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
    --test)
      [[ $# -ge 2 ]] || {
        log "--test requires a command"
        exit 1
      }
      FOCUSED_TESTS+=("$2")
      shift 2
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

bash scripts/check-repo-settings.sh --quiet

log "local PR gate: fast debug checks plus repository-rules review"
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
  gh pr merge --auto --squash --delete-branch "$pr_url"
fi

log "monitoring required PR checks every 30 seconds"
bash scripts/monitor-pr-checks.sh "$pr_url" --interval 30
