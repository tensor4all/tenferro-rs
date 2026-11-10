#!/usr/bin/env bash
# Report and clear the oracle nightly failure issue.
#
# The workflow hands the job results in and decides nothing itself: this script
# owns the issue body, the deduplication and the recovery decision.
#
# Inputs (environment):
#   RUN_URL        URL of the workflow run
#   HEAD_SHA       commit the run used
#   RUN_ORACLE     recent-commits output: "true" when the replay was requested
#   RECENT_RESULT  recent-commits job result
#   REPLAY_RESULT  oracle-replay job result
#   ORACLE_LOG     replay log captured by the oracle-replay job (may be absent)
set -euo pipefail

ISSUE_TITLE="oracle-replay-nightly is failing"
ISSUE_LABEL="nightly-oracle-failure"
ISSUE_SEARCH="oracle-replay-nightly is failing in:title"
DB_ROOT="third_party/tensor-ad-oracles"
TORCH_PIN="${DB_ROOT}/pyproject.toml"
# Records are named <op>_<dtype>_<family>_<number>; the replay error strings carry
# the failing record's case_id verbatim. Replay workers run every record and the
# runner returns the first error it collects, so the log holds the first failure
# rather than an inventory.
CASE_ID_PATTERN='[a-z][a-z0-9_]*_(f32|f64|c32|c64|c128)_[a-z0-9_]+_[0-9]{3}'
CASE_ID_LIMIT=20
TITLE_FILTER='[.[] | select(.title == $title)][0].number // empty'

log() { printf '%s\n' "$*"; }

# The vendored database carries no upstream revision pin, so name what is
# actually pinned instead of inventing a revision.
oracle_db_line() {
  local ref commit tree torch_line torch_version
  ref="${HEAD_SHA:-HEAD}"
  commit="$(git rev-parse --short=12 "${ref}" 2>/dev/null || echo unknown)"
  tree="$(git rev-parse "${ref}:${DB_ROOT}" 2>/dev/null || echo unknown)"
  torch_line="$(grep -n '"torch==' "${TORCH_PIN}" 2>/dev/null | head -n 1 | cut -d: -f1 || true)"
  torch_version="$(sed -n 's/^[[:space:]]*"torch==\([^"]*\)".*$/\1/p' "${TORCH_PIN}" 2>/dev/null | head -n 1 || true)"
  printf 'Oracle DB: `%s` as vendored at commit `%s` (tree `%s`); no upstream revision pin is recorded in this repository; the vendored generator requires `torch==%s` (`%s:%s`).\n' \
    "${DB_ROOT}" "${commit}" "${tree}" "${torch_version:-unknown}" \
    "${TORCH_PIN}" "${torch_line:-unknown}"
}

extract_case_ids() {
  local log_file="${1:-}"
  [ -n "${log_file}" ] && [ -f "${log_file}" ] || return 0
  grep -oE "${CASE_ID_PATTERN}" "${log_file}" | sort -u || true
}

case_id_section() {
  local case_ids="${1:-}"
  local count shown
  count="$(printf '%s' "${case_ids}" | grep -c . || true)"
  printf '### Failing case IDs\n\n'
  if [ "${count}" -eq 0 ]; then
    # Say only what the log shows: a missing artifact, an empty log and an
    # unrecognized log all look alike from here.
    printf 'No case ID could be extracted (`%s` is missing, empty, or has no\n' "${ORACLE_LOG:-<unset>}"
    printf 'recognized `<op>_<dtype>_<family>_<n>` record ID). Triage the failing jobs above.\n'
    return 0
  fi
  printf 'The replay runner reports the first error it collects, so start with:\n\n'
  shown="$(printf '%s\n' "${case_ids}" | head -n "${CASE_ID_LIMIT}")"
  printf '%s\n' "${shown}" | sed 's/^/- `/; s/$/`/'
  if [ "${count}" -gt "${CASE_ID_LIMIT}" ]; then
    printf '\n%d further case IDs are in the run log.\n' "$((count - CASE_ID_LIMIT))"
  fi
}

# Never fatal: a missing run id or a failed API call must still leave an issue.
failed_jobs() {
  if [ -z "${GITHUB_RUN_ID:-}" ]; then
    printf '%s\n' '- (run id unavailable)'
    return 0
  fi
  gh run view "${GITHUB_RUN_ID}" --json jobs \
    --jq '.jobs[] | select(.conclusion == "failure") | "- \(.name)"' \
    || printf '%s\n' '- (failed job list unavailable)'
}

failure_body() {
  local case_ids
  case_ids="$(extract_case_ids "${ORACLE_LOG:-}")"
  printf 'The scheduled `oracle-replay-nightly` run failed.\n\n'
  printf -- '- Run: %s\n' "${RUN_URL}"
  printf -- '- Commit: `%s`\n' "${HEAD_SHA}"
  printf -- '- Gate: recent-commits=%s, run_oracle=%s, oracle-replay=%s\n' \
    "${RECENT_RESULT:-unknown}" "${RUN_ORACLE:-unknown}" "${REPLAY_RESULT:-unknown}"
  printf -- '- Failed jobs:\n'
  failed_jobs | sed 's/^/  /'
  printf '\n'
  case_id_section "${case_ids}"
  printf '\n%s\n' "$(oracle_db_line)"
}

recovery_body() {
  printf 'Oracle replay recovered.\n\n'
  printf -- '- Run: %s\n' "${RUN_URL}"
  printf -- '- Commit: `%s`\n' "${HEAD_SHA}"
  printf -- '- Gate: recent-commits=%s, run_oracle=%s, oracle-replay=%s\n\n' \
    "${RECENT_RESULT:-unknown}" "${RUN_ORACLE:-unknown}" "${REPLAY_RESULT:-unknown}"
  printf 'The nightly replay ran and passed, so this issue is closed.\n\n'
  printf '%s\n' "$(oracle_db_line)"
}

# The label scopes the search and is the failure marker; the exact title keeps it
# unambiguous. `gh issue list --label` fails when the label does not exist yet
# (the first failure ever): that one error means "absent" and the title search in
# the next step still finds an issue filed before the label existed. Any other
# lookup failure aborts instead of reporting a second issue.
find_open_issue() {
  local labelled err number
  err="$(mktemp)"
  if ! labelled="$(gh issue list --state open --label "${ISSUE_LABEL}" --limit 50 \
    --json number,title 2>"${err}")"; then
    if grep -qiE 'not found|unknown label' "${err}"; then
      labelled=""
    else
      printf 'labelled issue lookup failed: %s\n' "$(cat "${err}")" >&2
      rm -f "${err}"
      return 1
    fi
  fi
  rm -f "${err}"
  number="$(printf '%s' "${labelled}" \
    | jq -r --arg title "${ISSUE_TITLE}" "${TITLE_FILTER}" 2>/dev/null || true)"
  if [ -n "${number}" ]; then
    printf '%s\n' "${number}"
    return 0
  fi
  # One issue under the label is the failure issue even if its title was edited;
  # several are ambiguous and only the exact title may match.
  number="$(printf '%s' "${labelled}" \
    | jq -r 'if length == 1 then .[0].number else empty end' 2>/dev/null || true)"
  if [ -n "${number}" ]; then
    printf '%s\n' "${number}"
    return 0
  fi
  gh issue list --state open --search "${ISSUE_SEARCH}" --limit 50 --json number,title \
    | jq -r --arg title "${ISSUE_TITLE}" "${TITLE_FILTER}"
}

# `gh issue create --label` and `gh issue edit --add-label` both fail when the
# label is absent, so it is ensured before either is used.
ensure_label() {
  gh label create "${ISSUE_LABEL}" --force \
    --description "Oracle nightly replay is failing" --color B60205 >/dev/null
}

should_report() {
  [ "${RECENT_RESULT:-}" = failure ] || [ "${REPLAY_RESULT:-}" = failure ]
}

# Recovery needs the replay itself to have run and passed. A skipped run
# (run_oracle=false), a cancelled run and any run whose replay was not requested
# are not recovery.
should_recover() {
  [ "${RUN_ORACLE:-}" = true ] && [ "${REPLAY_RESULT:-}" = success ]
}

report() {
  local number
  number="$(find_open_issue)"
  ensure_label
  if [ -z "${number}" ]; then
    gh issue create --title "${ISSUE_TITLE}" --label "${ISSUE_LABEL}" \
      --body "$(failure_body)"
    return 0
  fi
  gh issue comment "${number}" --body "$(failure_body)"
  gh issue edit "${number}" --add-label "${ISSUE_LABEL}" >/dev/null
}

recover() {
  local number
  number="$(find_open_issue)"
  if [ -z "${number}" ]; then
    log "no open failure issue to close"
    return 0
  fi
  gh issue comment "${number}" --body "$(recovery_body)"
  gh issue close "${number}"
  log "closed #${number} after a successful replay"
}

if should_recover; then
  recover
elif should_report; then
  report
else
  log "nothing to report: recent-commits=${RECENT_RESULT:-none} oracle-replay=${REPLAY_RESULT:-none} run_oracle=${RUN_ORACLE:-none}"
fi
