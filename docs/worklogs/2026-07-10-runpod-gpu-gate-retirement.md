# RunPod GPU gate retirement

## Summary

Issue #1341 retires the legacy `CI_gpu.yml` workflow from automatic same-repo
PR execution and makes the trusted RunPod workflow own the maintainer PR GPU
gate. Fork PRs keep the legacy pull_request path because base-repo Checks API
writes cannot attach a required check to fork commits.

## Context read

- Issue #1341 acceptance criteria.
- Shared tensor4all common repository and docs/tests rules.
- `REPOSITORY_RULES.md`, especially CI cost discipline.
- `.github/workflows/CI_gpu.yml`.
- `.github/workflows/runpod-gpu-test.yml`.

## Decisions

- Keep `CI_gpu.yml` as the PR-attached `CI GPU gate` wrapper. Same-repo PRs do
  not run legacy GPU work there; they wait for the trusted RunPod check. Fork
  PRs still use the legacy `ubuntu-gpu` path because the RunPod workflow cannot
  attach a base-repo Checks API result to fork commits.
- Keep a separate `CI GPU legacy manual gate` for manual fallback runs.
- Add the required `CI GPU gate` check to the RunPod workflow after
  authorization, non-GPU pre-gating, archive build, pod startup, GPU tests, and
  cleanup all succeed. Because `workflow_run` jobs are attached to the default
  branch commit, the final GitHub-hosted gate publishes a Checks API result to
  the PR head SHA instead of relying on the workflow job's own check run.
- Keep `checks: write` scoped to the final GitHub-hosted gate job only; the
  self-hosted RunPod job remains read-only.
- Keep a manual `ci_gpu_gate_head_sha` input so workflow-change PRs can run the
  PR branch workflow against a pinned test ref and still publish the required
  gate to the PR head before the trusted workflow has landed on `main`.
- Restrict the final gate job to manual dispatch and PR-sourced
  `workflow_run` events. Push-sourced `CI PR workspace tests` runs on `main`
  should not create a failing RunPod workflow result when there is no PR gate to
  publish.
- Make the legacy `CI_gpu.yml` manual fallback check out the requested
  `tenferro_ref` in both archive and GPU-runner jobs so PJRT validation uses
  the same ref as the archived CUDA tests. Pull request runs use the PR event
  ref.
- Keep `CI_gpu.yml` automatic only for fork PRs. Same-repo PRs must use RunPod,
  while fork PRs cannot receive the RunPod-published base-repo check on the
  fork head SHA.
- Preserve the existing trusted-base `workflow_run` model and pinned merge-SHA
  checkout behavior from PR #1340.

## Deferred

- Deleting `CI_gpu.yml` entirely requires a separate fork-PR gate design, not
  just more RunPod production history.
- Branch protection should continue requiring the same check name,
  `CI GPU gate`, emitted by RunPod for same-repo maintainer PRs and by
  `CI_gpu.yml` for fork PRs.

## Verification

No local CI was run in this editing pass. The intended verification is a manual
dispatch of `runpod-gpu-test.yml` against a pinned PR merge SHA, followed by a
maintainer same-repo PR update confirming that only RunPod automatically emits
`CI GPU gate` for same-repo PRs. A future fork PR should confirm the legacy
`CI_gpu.yml` path still emits `CI GPU gate`.
