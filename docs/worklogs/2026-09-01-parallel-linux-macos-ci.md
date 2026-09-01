# Parallel Linux and macOS workspace CI

Date: 2026-09-01
Issue: #1748

## Summary

Start the required macOS workspace lane after change classification, in
parallel with the Linux workspace and extension lanes. Previously macOS waited
for the Linux aggregate gate, extending pull-request wall-clock time.

## Context read

- `.github/workflows/ci-pr-workspace-tests.yml`
- `scripts/ci/change_policy.py`
- `scripts/ci/run_profile.py`
- `scripts/ci/tests/test_workflow_contracts.py`
- `docs/design/change-aware-ci.md`
- `docs/worklogs/2026-08-24-macos-ci-restoration.md`
- GitHub's current Actions billing and hosted-runner documentation

## Decisions

- Depend on `changes`, not `ci-gate`, so native macOS work begins as soon as
  classification succeeds.
- Keep the stable `macOS workspace tests` required-check name and preserve the
  explicit Ubuntu no-op for changes that do not require native validation.
- Fail closed when classification fails without allocating a macOS runner.
- Keep the shared `workspace-faer` command profile unchanged; command and
  feature coverage are outside this ordering-only change.
- Keep Linux and macOS failures independent and visible. The downstream RunPod
  workflow still waits for the workspace workflow to complete, preserving the
  paid GPU allocation boundary.

## Alternatives

- Retain Linux-first ordering to avoid unnecessary macOS work. This was
  rejected because the repository is public, `macos-15` is a standard hosted
  runner, and faster cross-platform feedback is more useful than suppressing
  work on revisions that may fail on Linux.
- Remove macOS doctests while changing the ordering. This was deferred because
  command coverage is separate from the dependency-graph change requested by
  #1748.

## Verification

- `python3.11 -m unittest scripts.ci.tests.test_workflow_contracts scripts.ci.tests.test_change_policy`: pass (50 tests)
- `python3.11 -m unittest discover -s scripts/ci/tests -v`: pass (198 tests)
- `actionlint .github/workflows/ci-pr-workspace-tests.yml .github/workflows/runpod-gpu-test.yml`: pass
- `bash scripts/check-pr-fast.sh --test '<focused CI contract tests>'`: pass
- `python3 scripts/ci/run_profile.py ci-config`: attempted, but the unrelated
  release-publish and storage-ownership helper fixtures fail under this macOS
  checkout's local command environment; the directly affected CI test suite
  and `actionlint` pass independently as listed above.
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD
  --dry-run --llm-skipped-reason "local deterministic review"`: pass; external
  LLM review intentionally skipped

## Residual risks

Linux failures no longer prevent an already-selected macOS job from consuming
runner capacity. Both jobs are cancellation-aware at the workflow level, and
the reduced PR latency is the intentional tradeoff.
