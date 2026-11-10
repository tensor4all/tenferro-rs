#!/usr/bin/env bash
# Report and clear the oracle nightly failure issue.
#
# The workflow hands the job results in and decides nothing itself: this script
# owns the issue body, the tracker selection and the recovery decision.
#
# Inputs (environment):
#   RUN_URL        URL of the workflow run
#   HEAD_SHA       commit the run used
#   RUN_ORACLE     recent-commits output: "true" when the replay was requested
#   RECENT_RESULT  recent-commits job result
#   REPLAY_RESULT  oracle-replay job result
#   ORACLE_LOG     replay log captured by the oracle-replay job (may be absent)
# The checkout must contain the tracker's recorded failure commit, so the
# workflow checks out full history; the ancestry test needs it.
set -euo pipefail

ISSUE_TITLE="oracle-replay-nightly is failing"
ISSUE_LABEL="nightly-oracle-failure"
ISSUE_SEARCH="oracle-replay-nightly is failing in:title"
TITLE_FILTER='[.[] | select(.title == $title)][0].number // empty'
# One page of `gh issue list` results. Hitting it means the search cannot be
# trusted to contain a tracker, so the script refuses to act instead of
# creating a duplicate.
PAGE_LIMIT=1000
DB_ROOT="third_party/tensor-ad-oracles"
TORCH_PIN="${DB_ROOT}/pyproject.toml"
# Records are named <op>_<dtype>_<family>_<number>; the replay error strings carry
# the failing record's case_id verbatim. The lookarounds keep a longer token
# (`…_0012`), a `__`-prefixed op name and a path component (`…/cache_f64_x_001`)
# from being reported as a case ID. The replay runner reports the first error it
# collects, so this is the first failing record rather than an inventory. No
# `__`-prefixed op is in the replay support table, so those report as "no case ID"
# rather than being guessed at.
CASE_ID_PATTERN='(?<![A-Za-z0-9_/])[A-Za-z][A-Za-z0-9_]*_(f32|f64|c64|c128)_[A-Za-z0-9_]+_[0-9]{3}(?![A-Za-z0-9_])'
CASE_ID_LIMIT=20

TRACKER_NUMBER=""
TRACKER_LABELLED=0
AMBIGUOUS=""

log() { printf '%s\n' "$*"; }

die_ambiguous() {
  printf 'ambiguous nightly failure trackers: %s; refusing to %s\n' "${AMBIGUOUS}" "$1" >&2
  exit 1
}

# The vendored database carries no upstream revision pin, so name what is
# actually pinned instead of inventing a revision.
oracle_db_line() {
  local commit tree torch_line torch_version
  commit="$(git rev-parse --short=12 "${HEAD_SHA:-HEAD}" 2>/dev/null || echo unknown)"
  tree="$(git rev-parse "${HEAD_SHA:-HEAD}:${DB_ROOT}" 2>/dev/null || echo unknown)"
  torch_line="$(grep -n '"torch==' -- "${TORCH_PIN}" 2>/dev/null | head -n 1 | cut -d: -f1 || true)"
  torch_version="$(sed -n 's/^[[:space:]]*"torch==\([^"]*\)".*$/\1/p' "${TORCH_PIN}" 2>/dev/null | head -n 1 || true)"
  printf 'Oracle DB: `%s` as vendored at commit `%s` (tree `%s`); no upstream revision pin is recorded in this repository; the vendored generator requires `torch==%s` (`%s:%s`).\n' \
    "${DB_ROOT}" "${commit}" "${tree}" "${torch_version:-unknown}" \
    "${TORCH_PIN}" "${torch_line:-unknown}"
}

extract_case_ids() {
  local log_file="${1:-}"
  [ -n "${log_file}" ] && [ -f "${log_file}" ] || return 0
  grep -oP -- "${CASE_ID_PATTERN}" "${log_file}" 2>/dev/null | sort -u || true
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
  # `- Commit:` is also the marker recovery reads back to relate its evidence to
  # the currently tracked failure, so keep the line in this exact shape.
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

stale_recovery_body() {
  printf 'Recovery evidence is older than the tracked failure, so this issue stays open.\n\n'
  printf -- '- Recovering run: %s\n' "${RUN_URL}"
  printf -- '- Recovering commit: `%s`\n' "${HEAD_SHA}"
  printf -- '- Tracked failure commit: `%s`\n\n' "$1"
  printf 'The nightly replay passed on `%s`, which is not the tracked failure commit or a\n' "${HEAD_SHA}"
  printf 'descendant of it, so it does not show that the tracked failure is fixed.\n'
}

# `gh issue list --label` fails when the label does not exist yet (the first
# failure ever). That one error means "absent", and the title search covers it.
# Anything else (auth, HTTP, rate limit) must not look like "no tracker".
list_labelled() {
  local err
  err="$(mktemp)"
  if ! gh issue list --state open --label "${ISSUE_LABEL}" --limit "${PAGE_LIMIT}" \
    --json number,title 2>"${err}"; then
    if grep -qiE 'not found|unknown label' "${err}"; then
      rm -f "${err}"
      printf '[]\n'
      return 0
    fi
    printf 'labelled issue lookup failed: %s\n' "$(cat "${err}")" >&2
    rm -f "${err}"
    return 1
  fi
  rm -f "${err}"
}

count_of() { printf '%s' "${1:-[]}" | jq -r 'length' 2>/dev/null || echo 0; }

# Sets TRACKER_NUMBER (empty when there is none), TRACKER_LABELLED and AMBIGUOUS.
# "Trackers exist but none is identifiable" is ambiguous, never "no tracker":
# the caller must not create a third issue in that state.
resolve_tracker() {
  local labelled exact count
  TRACKER_NUMBER=""
  TRACKER_LABELLED=0
  AMBIGUOUS=""
  labelled="$(list_labelled)"
  exact="$(printf '%s' "${labelled}" | jq -r --arg title "${ISSUE_TITLE}" "${TITLE_FILTER}" 2>/dev/null || true)"
  if [ -n "${exact}" ]; then
    TRACKER_NUMBER="${exact}"
    TRACKER_LABELLED=1
    return 0
  fi
  count="$(count_of "${labelled}")"
  if [ "${count}" -eq 1 ]; then
    # One labelled issue is the tracker even when its title was edited.
    TRACKER_NUMBER="$(printf '%s' "${labelled}" | jq -r '.[0].number')"
    TRACKER_LABELLED=1
    return 0
  fi
  if [ "${count}" -gt 1 ]; then
    AMBIGUOUS="$(printf '%s' "${labelled}" | jq -r '.[].number' | sed 's/^/#/' | paste -sd' ' -)"
    return 0
  fi
  labelled="$(gh issue list --state open --search "${ISSUE_SEARCH}" --limit "${PAGE_LIMIT}" \
    --json number,title)"
  exact="$(printf '%s' "${labelled}" | jq -r --arg title "${ISSUE_TITLE}" "${TITLE_FILTER}" 2>/dev/null || true)"
  if [ -n "${exact}" ]; then
    TRACKER_NUMBER="${exact}"
    return 0
  fi
  if [ "$(count_of "${labelled}")" -ge "${PAGE_LIMIT}" ]; then
    AMBIGUOUS="the title search returned a full page of ${PAGE_LIMIT} issues without an exact match"
  fi
}

ensure_label() {
  gh label create "${ISSUE_LABEL}" --force \
    --description "Oracle nightly replay is failing" --color B60205 >/dev/null
}

# The newest failure the tracker records: the issue body holds the first report
# and every later one is a comment, so the last marker wins. `--paginate` keeps a
# long tracker from hiding its newest failure behind a page boundary.
tracker_failure_sha() {
  local number="$1"
  {
    gh api "repos/${GH_REPO}/issues/${number}" --jq '.body'
    gh api --paginate "repos/${GH_REPO}/issues/${number}/comments" --jq '.[].body'
  } 2>/dev/null \
    | sed -n 's/^- Commit: `\([0-9a-f]\{7,40\}\)`$/\1/p' \
    | tail -n 1
}

# Recovery evidence must not be older than the failure it would close: the
# recovering commit has to be the tracked failure commit or a descendant.
recovery_covers() {
  local failure_sha="${1:-}"
  [ -n "${failure_sha}" ] || return 1
  [ "${failure_sha}" = "${HEAD_SHA}" ] && return 0
  git cat-file -e "${failure_sha}^{commit}" 2>/dev/null || return 1
  git merge-base --is-ancestor "${failure_sha}" "${HEAD_SHA}" 2>/dev/null
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
  resolve_tracker
  [ -n "${AMBIGUOUS}" ] && die_ambiguous "file a failure"
  if [ -z "${TRACKER_NUMBER}" ]; then
    ensure_label
    gh issue create --title "${ISSUE_TITLE}" --label "${ISSUE_LABEL}" \
      --body "$(failure_body)"
    return 0
  fi
  local body
  body="$(failure_body)"
  gh issue comment "${TRACKER_NUMBER}" --body "${body}"
  if [ "${TRACKER_LABELLED}" -eq 0 ]; then
    # A tracker filed before the label existed adopts it, once.
    ensure_label
    gh issue edit "${TRACKER_NUMBER}" --add-label "${ISSUE_LABEL}" >/dev/null
  fi
}

recover() {
  resolve_tracker
  [ -n "${AMBIGUOUS}" ] && die_ambiguous "close a failure"
  if [ -z "${TRACKER_NUMBER}" ]; then
    log "no open failure issue to close"
    return 0
  fi
  local failure_sha
  failure_sha="$(tracker_failure_sha "${TRACKER_NUMBER}")"
  if ! recovery_covers "${failure_sha}"; then
    gh issue comment "${TRACKER_NUMBER}" --body "$(stale_recovery_body "${failure_sha:-unknown}")"
    log "recovery evidence does not cover the tracked failure (${failure_sha:-unknown}); issue left open"
    return 0
  fi
  gh issue comment "${TRACKER_NUMBER}" --body "$(recovery_body)"
  gh issue close "${TRACKER_NUMBER}"
  log "closed #${TRACKER_NUMBER} after a successful replay"
}

if should_recover; then
  recover
elif should_report; then
  report
else
  log "nothing to report: recent-commits=${RECENT_RESULT:-none} oracle-replay=${REPLAY_RESULT:-none} run_oracle=${RUN_ORACLE:-none}"
fi
