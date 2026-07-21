# Issue #1439 repository-role authorization

## Context

PR #1438 showed that trusted RunPod authorization rejected a collaborator
assigned the `maintain` repository role. GitHub returned the legacy
`permission` field as `write` and the actual role in `role_name`. The same
field mismatch existed in both maintainer-only review-label checks.

The bug-fix scope gate was applied at `upstream/main` commit `85855e27` after
reviewing `CONTRIBUTING.md`, `AGENTS.md`, `REPOSITORY_RULES.md`, the bug-fix
workflow, issue #1439, and the existing RunPod and review-bot trust boundaries.

## Decision

- Read required `role_name` values for RunPod and review-label authorization.
- Preserve the existing `admin|maintain` allowlist and every actor/author
  check.
- Fail closed when `role_name` is missing or unknown.

Falling back to legacy `permission` was rejected because it cannot distinguish
`maintain` from `write`. GPU classification, same-repository and pinned-SHA
checks, manual author checks, and the 90-minute wrapper remain unchanged.

## Verification

- `python3 -m unittest scripts.ci.tests.test_workflow_contracts` (28 tests)
- `python3 scripts/ci/run_profile.py ci-config` (154 tests and actionlint 1.7.7)
- `bash scripts/check-pr-fast.sh --base upstream/main --no-fetch --ci-profile ci-config`
- `cargo fmt --all --check`
- committed-head repository-rules review: pass, no findings

## Remaining risk

A trusted exact-SHA RunPod recovery must confirm the live GitHub authorization
response and GPU workflow before merge.
