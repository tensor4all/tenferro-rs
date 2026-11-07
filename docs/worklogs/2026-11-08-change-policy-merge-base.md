# Change policy reads the merge-base diff

## Decisions

- **Classify from the fork point, not from the two endpoint trees.** `--base`
  carries the base branch tip (`github.event.pull_request.base.sha`), not the
  commit the pull request forked from. `git diff base head` (two-dot) therefore
  reported every path that landed on the base branch after the fork, as if the
  pull request had removed it. When one of those late landings was in
  `_GPU_DEVICE_CONTROL_FILES`, a pull request that never touched it reserved a
  paid pod: a prose-only change classified as `code`/`ci-only` with
  `run_gpu: true` instead of `docs-only` with `run_gpu: false`. The hunk reader
  in `_docs_only_rust_paths` already used `base...head`; the path reader was
  the inconsistent one.
- **Keep the `--base`/`--head` contract.** Only the comparison base moves to
  the merge base. Callers that pass an explicit `--path` list (the GitHub API
  file list, `check-pr-fast.sh`) keep their meaning, and `--path` still wins.
- **Fail closed when the fork point is unknown.** A shallow clone without the
  common ancestor, an unknown revision, or an unusable git keeps the full
  policy with `run_gpu: true` and an explicit reason. Falling back to the
  two-tree diff was rejected: that comparison both invents entries and hides
  real ones, so it is not a conservative default for a decision that spends
  money.

## Verification conclusions and constraints

- Red before the fix: the new `StaleBaseTests` cases fail with the late landing
  in the diff (`'ci-only' != 'docs-only'`, `'code' != 'docs-only'`). Reverting
  the comparison to two-dot fails four of the five new tests.
- The hosted callers that read paths from the endpoints (`ci.yml`,
  `ci-pr-workspace-tests.yml`) use `fetch-depth: 0`, so the merge base is
  available. The RunPod workflows pass `--path` and are unaffected.
- Shallow history is the residual limit: a checkout that fetches both endpoints
  without their common ancestor now keeps the paid lane instead of guessing.
  Widening the fetch is a workflow change, not a policy change.
