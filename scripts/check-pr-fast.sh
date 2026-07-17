#!/usr/bin/env bash
set -euo pipefail

BASE_REF="origin/main"
FETCH=1
DOC_SNIPPETS="auto"
COVERAGE_REVIEWED=0
FOCUSED_TESTS=()
CI_PROFILES=()
CI_PROFILE_DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash scripts/check-pr-fast.sh [options]

Runs the lightweight local checks that are useful before opening or updating a
PR. Full workspace tests, rustdoc, and coverage gates are left to CI.

Options:
  --base REF                 Base ref for branch freshness and diff checks (default: origin/main)
  --no-fetch                 Do not fetch origin before checking BASE_REF
  --coverage-reviewed        Confirm changed code was manually/agent reviewed for test coverage
  --test COMMAND             Run one focused verification command; repeatable
  --ci-profile NAME          Run one shared CI profile; repeatable
  --ci-profile-dry-run       Print selected CI profile commands without running them
  --doc-snippets             Always run docs snippet sync checks
  --skip-doc-snippets        Never run docs snippet sync checks
  --help                     Show this help text

Example:
  bash scripts/check-pr-fast.sh \
    --coverage-reviewed \
    --test 'cargo test -p tenferro eager_matmul_gradients_match_finite_difference'
EOF
}

log() {
  printf '%s\n' "$*"
}

run() {
  log "+ $*"
  "$@"
}

die() {
  log "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)
      [[ $# -ge 2 ]] || die "--base requires a ref"
      BASE_REF="$2"
      shift 2
      ;;
    --no-fetch)
      FETCH=0
      shift
      ;;
    --coverage-reviewed)
      COVERAGE_REVIEWED=1
      shift
      ;;
    --test)
      [[ $# -ge 2 ]] || die "--test requires a command"
      FOCUSED_TESTS+=("$2")
      shift 2
      ;;
    --ci-profile)
      [[ $# -ge 2 ]] || die "--ci-profile requires a name"
      CI_PROFILES+=("$2")
      shift 2
      ;;
    --ci-profile-dry-run)
      CI_PROFILE_DRY_RUN=1
      shift
      ;;
    --doc-snippets)
      DOC_SNIPPETS="always"
      shift
      ;;
    --skip-doc-snippets)
      DOC_SNIPPETS="never"
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ "$FETCH" -eq 1 && "$BASE_REF" == origin/* ]]; then
  run git fetch origin
fi

git rev-parse --verify "${BASE_REF}^{commit}" >/dev/null ||
  die "base ref does not resolve to a commit: ${BASE_REF}"

branch="$(git branch --show-current)"
if [[ -z "$branch" ]]; then
  die "not on a named branch"
fi

if ! git merge-base --is-ancestor "$BASE_REF" HEAD; then
  die "HEAD is not based on ${BASE_REF}; rebase/merge latest ${BASE_REF} before PR checks"
fi

base_short="$(git rev-parse --short "$BASE_REF")"
head_short="$(git rev-parse --short HEAD)"
log "branch: ${branch}"
log "base:   ${BASE_REF} (${base_short})"
log "head:   ${head_short}"

changed_files=()
while IFS= read -r path; do
  changed_files+=("$path")
done < <(
  {
    git diff --name-only "${BASE_REF}...HEAD"
    git diff --cached --name-only
    git diff --name-only
    git ls-files --others --exclude-standard
  } | awk 'NF' | sort -u
)

untracked_files=()
while IFS= read -r path; do
  untracked_files+=("$path")
done < <(git ls-files --others --exclude-standard | awk 'NF' | sort -u)

if [[ "${#changed_files[@]}" -eq 0 ]]; then
  log "changed files: none"
else
  log "changed files:"
  printf '  %s\n' "${changed_files[@]}"
fi

policy_args=(python3 scripts/ci/change_policy.py)
for path in "${changed_files[@]}"; do
  policy_args+=(--path "$path")
done
policy_json="$("${policy_args[@]}")"
mapfile -t policy_fields < <(
  python3 -c \
    'import json, sys; data = json.load(sys.stdin); print(data["classification"]); print(data["reason"])' \
    <<<"${policy_json}"
)
change_class="${policy_fields[0]}"
change_reason="${policy_fields[1]}"
log "classification: ${change_class}"
log "classification reason: ${change_reason}"

run git diff --check "${BASE_REF}...HEAD"
run git diff --cached --check
run git diff --check
if [[ "${#untracked_files[@]}" -gt 0 ]]; then
  log "note: untracked files are listed above, but git diff --check only covers tracked/staged content"
  log "      stage or commit new files before relying on this whitespace check"
fi
if [[ "${change_class}" == "code" ]]; then
  run cargo fmt --all --check
else
  log "cargo fmt: skipped for ${change_class} changes"
fi

run_doc_snippets=0
case "$DOC_SNIPPETS" in
  always)
    run_doc_snippets=1
    ;;
  never)
    run_doc_snippets=0
    ;;
  auto)
    for path in "${changed_files[@]}"; do
      if [[ "$path" == docs/* || "$path" == README.md || "$path" == *.md || "$path" == *.qmd ]]; then
        run_doc_snippets=1
        break
      fi
    done
    ;;
esac

if [[ "$run_doc_snippets" -eq 1 ]]; then
  run python3 scripts/check-doc-snippets.py --root-dir . --check
else
  log "docs snippets: skipped"
fi

if [[ "${change_class}" != "docs-only" && "${#FOCUSED_TESTS[@]}" -eq 0 && "${#CI_PROFILES[@]}" -eq 0 ]]; then
  die "focused verification command required for ${change_class} changes; pass --test COMMAND or --ci-profile NAME"
fi

for command in "${FOCUSED_TESTS[@]}"; do
  log "+ ${command}"
  bash -lc "$command"
done

if [[ "${#CI_PROFILES[@]}" -gt 0 ]]; then
  profile_args=("${CI_PROFILES[@]}")
  if [[ "$CI_PROFILE_DRY_RUN" -eq 1 ]]; then
    profile_args=(--dry-run "${profile_args[@]}")
  fi
  python3 scripts/ci/run_profile.py "${profile_args[@]}"
elif [[ "$CI_PROFILE_DRY_RUN" -eq 1 ]]; then
  die "--ci-profile-dry-run requires at least one --ci-profile"
fi

if [[ "${change_class}" == "code" && "$COVERAGE_REVIEWED" -ne 1 ]]; then
  cat >&2 <<'EOF'
coverage review not confirmed

Before using this as a PR-ready local check, review changed code for:
- new branches and error paths
- dtype, rank, shape, zero-size, and device cases
- AD/JVP/VJP paths when touched
- focused tests or an explicit reason to leave the remaining coverage to CI

Rerun with --coverage-reviewed after that review.
EOF
  exit 1
fi

log "fast PR checks passed"
