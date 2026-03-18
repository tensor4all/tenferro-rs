#!/usr/bin/env bash
set -euo pipefail

PR_REF=""
INTERVAL=30
REPO=""

usage() {
  cat <<'EOF'
Usage: bash scripts/monitor-pr-checks.sh [pr-number|pr-url|branch] [options]

Options:
  --interval SECONDS      Poll interval in seconds (default: 30)
  --repo OWNER/REPO       Repository override for gh commands
  --help                  Show this help text
EOF
}

log() {
  printf '%s\n' "$*"
}

gh_pr_checks() {
  local args=(pr checks --required --json name,state,link,bucket,workflow)
  if [[ -n "$PR_REF" ]]; then
    args+=("$PR_REF")
  fi
  if [[ -n "$REPO" ]]; then
    args+=(-R "$REPO")
  fi
  gh "${args[@]}"
}

classify_checks() {
  python3 -c '
import json
import re
import sys

checks = json.load(sys.stdin)
if not isinstance(checks, list):
    raise SystemExit("expected a JSON list from gh pr checks")

if not checks:
    print("status\tpending")
    print("summary\tno required checks reported yet")
    raise SystemExit(0)

failures = []
pending = []

for item in checks:
    name = item.get("name") or "<unnamed>"
    workflow = item.get("workflow") or ""
    bucket = item.get("bucket") or ""
    link = item.get("link") or ""
    state = item.get("state") or ""

    if bucket in {"fail", "cancel"}:
        failures.append((name, workflow, link, state))
        continue
    if bucket in {"pass", "skipping"}:
        continue
    pending.append((name, workflow, link, state, bucket))

if failures:
    print("status\tfail")
    for name, workflow, link, state in failures:
        run_id = ""
        match = re.search(r"/runs/([0-9]+)", link or "")
        if match:
            run_id = match.group(1)
        print("failure\t{}\t{}\t{}\t{}\t{}".format(name, workflow, link, state, run_id))
    raise SystemExit(0)

if pending:
    print("status\tpending")
    print("summary\tpending checks: " + ", ".join(name for name, *_ in pending))
    raise SystemExit(0)

print("status\tpass")
print("summary\tall required checks passed")
'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval)
      INTERVAL="$2"
      shift 2
      ;;
    --repo)
      REPO="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      if [[ -n "$PR_REF" ]]; then
        log "unexpected argument: $1"
        usage
        exit 1
      fi
      PR_REF="$1"
      shift
      ;;
  esac
done

while true; do
  checks_json="$(gh_pr_checks)"
  parsed_output="$(printf '%s' "$checks_json" | classify_checks)"

  status=""
  summary=""
  while IFS=$'\t' read -r record_type field1 field2 field3 field4 field5; do
    case "$record_type" in
      status)
        status="$field1"
        ;;
      summary)
        summary="$field1"
        ;;
      failure)
        log "failed check: $field1"
        if [[ -n "$field2" ]]; then
          log "workflow: $field2"
        fi
        if [[ -n "$field3" ]]; then
          log "details: $field3"
        fi
        if [[ -n "$field5" ]]; then
          log "inspect logs: gh run view $field5 --log-failed"
        else
          if [[ -n "$PR_REF" ]]; then
            log "inspect checks: gh pr checks $PR_REF --required"
          else
            log "inspect checks: gh pr checks --required"
          fi
        fi
        ;;
      *)
        ;;
    esac
  done <<<"$parsed_output"

  case "$status" in
    pass)
      log "${summary:-all required checks passed}"
      exit 0
      ;;
    fail)
      if [[ -n "$PR_REF" ]]; then
        log "after fixing the failure locally, push and rerun: bash scripts/monitor-pr-checks.sh $PR_REF --interval $INTERVAL"
      else
        log "after fixing the failure locally, push and rerun: bash scripts/monitor-pr-checks.sh --interval $INTERVAL"
      fi
      exit 1
      ;;
    pending)
      log "${summary:-waiting for required checks}"
      sleep "$INTERVAL"
      ;;
    *)
      log "unexpected monitor status: ${status:-<empty>}"
      exit 1
      ;;
  esac
done
