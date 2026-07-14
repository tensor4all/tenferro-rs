# Issue #1379 change-aware CI and RunPod recovery

## Summary

Issue #1379 consolidates CI latency and RunPod recovery improvements observed
during the NUMA implementation and its documentation follow-up. This change
centralizes local/hosted command profiles, makes pull-request lanes
change-aware with conservative fallbacks, and hardens the trusted RunPod
control plane against schema drift and deterministic request failures.

## Context read

- Issue #1379 and the timing/recovery evidence from PR #1376.
- Shared tensor4all repository and docs/tests rules.
- `AGENTS.md`, `CONTRIBUTING.md`, and `REPOSITORY_RULES.md`, especially CI cost
  discipline, documentation, design-record, and PR verification requirements.
- `.github/workflows/ci.yml`, `ci-pr-workspace-tests.yml`, `CI_gpu.yml`, and
  `runpod-gpu-test.yml`.
- `scripts/check-pr-fast.sh` and existing docs-site, coverage, and repository
  review commands.
- RunPod's live `POST /pods` OpenAPI schema on 2026-07-14 and 2026-07-15.
- RunPod's published Pod prices on 2026-07-15.
- The prior RunPod retirement work log, including its now-superseded raw
  workflow-branch recovery path.

## Decisions

- Keep one standard-library Python policy layer under `scripts/ci/`; workflows
  orchestrate it rather than reproducing path regexes or command lists.
- Treat empty and unknown diffs as code. Docs-only and CI-only are explicit
  allowlists, and required check names complete as auditable no-ops.
- Force the comprehensive non-GPU matrix on pushes to `main`. Keep paid GPU
  work on code PRs and RunPod control-plane changes, after cheaper gates.
- Validate repository GPU IDs against the live request schema before archive
  setup. The 2026-07-15 live schema accepted all 19 configured IDs across the
  cost-preferred, premium, and A100 tiers.
- Retry only 408, 429, 5xx, and transport failures. Bound retries by count and
  deadline; treat other 4xx responses and malformed success responses as
  permanent.
- Treat an explicit machine-capacity error separately: move immediately to the
  next price tier rather than sleeping and resubmitting the same candidates.
  Allow one same-tier retry for unrelated transient failures, cap all creation
  work at 60 seconds, and exclude GPUs above the reviewed A100 SXM tier.
- Publish the selected tier and provider GPU ID to both ordinary logs and the
  GitHub job summary, then show them next to the machine's `nvidia-smi` output.
- Bound every HTTP request by the remaining global deadline, reject a missing
  or out-of-tier assigned GPU, and pass provider output to shell through the
  step environment rather than interpolating it into shell source.
- Exclude ref/SHA identity from the CUDA archive cache key and include all
  content/configuration inputs that affect the archive.
- Replace raw PR ref/SHA recovery with a PR-number command that dispatches only
  trusted `main`. The hosted authorization job owns state, repository,
  permission, changed-file, and head-stability checks.
- Keep secrets on trusted GitHub-hosted jobs. The external RunPod runner stays
  read-only and never receives RunPod or GitHub App credentials.
- Publish the durable policy in `docs/design/change-aware-ci.md`; keep the
  detailed accepted design and execution plan as development records.

## Alternatives rejected

- Inline workflow regexes and duplicated Cargo commands drift between local
  and hosted execution.
- A permissive “non-Rust means docs” rule can skip validation for unknown new
  paths.
- Retrying every non-2xx response repeats deterministic schema/auth failures
  and wastes paid-runner latency.
- A cache key containing a branch or SHA prevents reuse of identical content.
- Dispatching a secret-bearing workflow from a PR branch makes untrusted
  workflow/helper content part of the control plane.
- An external classification service adds availability and ownership without
  improving the repository-local trust model.

## Verification so far

- Baseline `cargo test --workspace --release --quiet` passed before changes.
- CI helper unit/source-contract tests passed through each TDD cycle.
- actionlint 1.7.7 passed all workflows after registering the organization
  `ubuntu-gpu` label and removing an obsolete `rust-cache` input.
- The shared faer and BLAS workspace profiles each passed 2,150 nextest tests
  plus workspace doctests.
- Provider-injection release tests passed: 3 tests in 5m11s for the cold build.
- The extension profile passed tropical and sparse release tests and doctests,
  plus the KdV sample all-target release check.
- The docs profile passed its source-contract checks, all four executable guide
  dependency snippets, the 78-page Quarto render, and rendered-site validation.
- The coverage profile passed all 159 included source files at their configured
  thresholds.
- The committed-head repository-rules review passed with no findings.
- The live RunPod OpenAPI schema accepted all 19 configured GPU IDs.
- Automatic run 29334073902 failed after five submissions of the same request:
  RunPod returned HTTP 500 with “This machine does not have the resources to
  deploy your pod” each time. Archive creation and cleanup behaved correctly;
  this evidence motivated immediate price-tier failover.
- Attempt 2 of run 29334073902 then recovered with an NVIDIA GeForce RTX 4090.
  The CUDA archive tests and OpenXLA PJRT end-to-end tests passed, the success
  marker was emitted, and pod `37vcua0n4ayryn` was deleted successfully. This
  confirms the existing trusted execution path while the new tier behavior
  still awaits a post-merge run from `main`.
- Trusted PR recovery dry run targets only
  `runpod-gpu-test.yml --ref main -f pr_number=1379`.

The committed-head repository review and trusted post-merge SECURE GPU run are
recorded before closing issue #1379. The live run must happen after the new
secret-bearing workflow has landed on `main`; pre-merge execution of the PR
branch workflow is intentionally prohibited.

## Residual risks

- RunPod may change its live schema or capacity between preflight and pod
  creation; schema mismatch fails early, while transient capacity remains a
  bounded retry.
- actionlint validates workflow structure and shell contracts but cannot model
  GitHub/RunPod external state.
- The first cache population still pays full CUDA archive cost; content reuse
  improves subsequent equivalent runs.
