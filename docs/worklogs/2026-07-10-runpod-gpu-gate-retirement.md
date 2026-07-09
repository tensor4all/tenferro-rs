# RunPod GPU gate retirement

## Summary

Issue #1341 retires the legacy `CI_gpu.yml` workflow from automatic PR
execution and makes the trusted RunPod workflow own the PR GPU gate.

## Context read

- Issue #1341 acceptance criteria.
- Shared tensor4all common repository and docs/tests rules.
- `REPOSITORY_RULES.md`, especially CI cost discipline.
- `.github/workflows/CI_gpu.yml`.
- `.github/workflows/runpod-gpu-test.yml`.

## Decisions

- Keep `CI_gpu.yml` as a manual fallback instead of deleting it immediately, so
  maintainers can still compare the org larger GPU runner with RunPod.
- Rename the legacy workflow's final check to `CI GPU legacy manual gate` so it
  no longer owns the branch-protection-required `CI GPU gate` name.
- Add the required `CI GPU gate` check to the RunPod workflow after
  authorization, non-GPU pre-gating, archive build, pod startup, GPU tests, and
  cleanup all succeed.
- Preserve the existing trusted-base `workflow_run` model and pinned merge-SHA
  checkout behavior from PR #1340.

## Deferred

- Deleting `CI_gpu.yml` entirely remains a maintainer decision after RunPod has
  enough production history.
- Branch protection should continue requiring the same check name,
  `CI GPU gate`, now emitted by the RunPod workflow.

## Verification

No local CI was run in this editing pass. The intended verification is a manual
dispatch of `runpod-gpu-test.yml` against a pinned PR merge SHA, followed by a
maintainer same-repo PR update confirming that only RunPod automatically emits
`CI GPU gate`.
