# Change-aware CI and trusted RunPod recovery

This document defines how tenferro selects validation work without weakening
required checks, and how paid RunPod validation stays behind a trusted control
plane. The executable sources of truth are `scripts/ci/` and the workflows
under `.github/workflows/`.

## Change policy

Pull requests have one primary class and independent lane flags:

- **code** includes Rust, manifests, build configuration, unknown paths, and
  empty diffs. It uses the full CPU, extension, docs, CI-configuration, and GPU
  policy.
- **docs-only** contains only rendered documentation or repository prose. It
  runs documentation validation and skips compiled-code lanes.
- **CI-only** contains only known workflow and CI-helper paths. It runs helper
  tests and actionlint. RunPod workflow, request, recovery, or classification
  changes additionally require the GPU gate.

Mixed docs and CI changes are CI-only with both lightweight flags enabled.
Unknown paths always fall back to code. Pushes to `main` force the
comprehensive non-GPU matrix; they do not add a second paid GPU run after the
pull-request gate.

Required job names do not disappear when work is unnecessary. Each required
job either runs its selected profile or publishes an explicit successful
no-op. A classification failure is a validation failure, never a cheap
fallback.

## Shared command profiles

`scripts/ci/run_profile.py` owns immutable profiles for workspace-faer,
workspace-blas, provider injection, extensions, documentation, coverage, and
CI configuration. Local and hosted execution call the same profile names.
`full` is composition rather than another command list, so repeated profiles
execute once.

## RunPod trust boundary

The RunPod workflow is triggered from trusted `main`, never from
`pull_request_target`. Before any archive build or pod allocation it:

1. authorizes the actor and rejects fork PRs;
2. resolves an immutable same-repository PR revision and rechecks head
   stability;
3. obtains the complete changed-file list through the GitHub API and classifies
   it with the helper from trusted `main`;
4. validates configured GPU IDs against RunPod's live `POST /pods` OpenAPI
   request schema.

Only trusted GitHub-hosted jobs receive the RunPod API key or GitHub App
credentials. The self-hosted pod never receives those credentials. The final
GitHub-hosted job publishes `CI GPU gate` to the authorized PR head SHA. A
docs-only or unrelated CI-only skip is successful only when the trusted
classifier says GPU validation is unnecessary.

Pod creation treats HTTP 408, 429, 5xx responses, and transport failures as
retryable. Other 4xx responses are permanent. An explicit RunPod machine
capacity error does not retry the same candidate set: the client moves without
sleeping from the cost-preferred tier to the premium tier and finally the A100
tier. Automatic selection excludes H100-class GPUs; the reviewed ceiling is
A100 SXM (listed at USD 1.49/hour when the tiers were reviewed on 2026-07-15).

Unrelated transient failures receive one short retry in the current tier. All
requests and sleeps share a 60-second deadline, honor numeric `Retry-After`,
and use bounded jittered backoff. Request diagnostics redact the JIT
configuration and startup command. The `Start RunPod org runner` job summary
records the selected price tier and provider GPU ID, and the GPU job prints the
same values next to `nvidia-smi` so the assigned machine remains auditable. A
successful response without an assigned GPU ID, or with an ID outside the
requested tier, is rejected before the external runner starts. The created pod
ID is still forwarded to the trusted startup-failure cleanup path so rejection
cannot leave a paid pod running.

The CUDA test archive key is content-addressed across source, manifests,
tests, lockfile, workflow, and RunPod configuration. It excludes branch, ref,
and commit identity, allowing equivalent automatic and recovery runs to reuse
the hosted cache while still uploading a per-run artifact for the external
runner.

## Recovery

Maintainers recover a PR by number with:

```bash
python3 scripts/ci/recover_runpod_pr.py PR_NUMBER --wait
```

The command has no workflow-ref option and always dispatches
`runpod-gpu-test.yml` at `main`. The trusted workflow verifies that the PR is
open, same-repository, authorized, and head-stable; it derives both the tested
revision and required-check target rather than accepting them from the caller.
Raw revision dispatch remains available for trusted post-merge validation, but
cannot be combined with PR-number recovery.
