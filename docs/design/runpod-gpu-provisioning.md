# RunPod GPU Provisioning: Cheapest Compatible Host

Status: active. Implements issue #1404 under the #1401 CI performance
umbrella. Companion code: `scripts/ci/runpod_pricing.py`,
`scripts/ci/cuda_smoke_test.py`, `scripts/ci/runpod_provision.py`; contract
tests in `scripts/ci/tests/`.

## Problem

GPU model metadata alone does not establish CUDA/PTX compatibility:
observed same-SKU RTX 4090 hosts carried different NVIDIA drivers, and only
some accepted CUDA 12.8-generated PTX. A fixed premium GPU choice also
overpays when cheaper reviewed cards are in stock and compatible.

## Candidate selection

The reviewed tier allowlist in `runpod_config.json` remains the eligibility
boundary — live data never adds a GPU type maintainers did not review.
`runpod_pricing.candidate_plan` builds the attempt order:

1. Query the public RunPod GraphQL `gpuTypes` endpoint (no credential) for
   Secure Cloud stock, VRAM, and hourly price of the eligible types.
2. Drop out-of-stock types, types without a Secure Cloud offer, and types
   below `min_vram_gb`.
3. Emit one single-GPU candidate per type, cheapest first, capped by
   `max_price_candidates`.
4. Always append the static reviewed tiers as the documented fallback, so a
   failed or stale pricing answer degrades to today's behavior instead of
   losing GPU coverage.

`allowedCudaVersions` metadata filtering stays on every create request as
the first-line filter; the runtime smoke proof below is the real
compatibility decision.

## Runtime compatibility proof (smoke test)

`cuda_smoke_test.py` runs on the pod, as root, inside the startup script
BEFORE the GitHub runner registers — so an incompatible host is rejected
before any dependency setup or test execution:

1. driver visibility and CUDA API version via `nvidia-smi`,
2. runtime tier selection mirroring the workflow (12.8 full / 12.4
   baseline),
3. minimal NVRTC install for the selected tier only,
4. NVRTC compilation of a tiny kernel for the device's compute capability,
5. PTX load through the driver, kernel launch, synchronize, and output
   readback,
6. VRAM check against `min_vram_gb`.

The script is embedded into the startup script by the trusted
start-runpod job from its own checkout (no pod-side network fetch), and
receives its parameters through non-secret pod environment variables. On
failure it exits nonzero, the container stops, and the pod never becomes a
runner.

## Bounded provision loop

`runpod_provision.py` runs in the trusted `start-runpod` job (all
credentials stay GitHub-hosted):

- mint one single-use JIT runner config per attempt under a fresh
  per-attempt label (`<prefix>-cN`): a JIT config cannot be replayed after
  an earlier candidate registered with it, and a shared label would let a
  stale online record from a rejected pod accept an unproven new pod;
  `run-gpu-tests` targets the accepted attempt's label;
- create one candidate pod at a time, cheapest first;
- watch two signals: the org runner registry (runner online = smoke proof
  passed) and the pod's container state via GraphQL (`desiredStatus` plus
  the `runtime` object — RunPod keeps `desiredStatus` at RUNNING after the
  container exits, so a null runtime after boot is the authoritative
  startup-failure signal);
- delete a rejected or timed-out pod immediately and move to the next
  candidate, reusing the same immutable per-run archive (#1403) — retries
  never compile Rust;
- stop after `max_provision_attempts` with an explicit exhaustion error;
- stop early after `max_consecutive_startup_failures` candidates failed to
  register a runner. Consecutive failures of that kind (created, never
  online) mean the provider is not delivering runners, and the remaining
  attempts would only add paid pod time; one outage otherwise creates and
  pays for every candidate without running a single test. `0` disables the
  early stop and restores the plain bounded ladder.

Capacity failures move to the next candidate without creating a pod, so
nothing was paid for and they do not count toward the early stop.
`startup_timeout_seconds` bounds each candidate's wait;
`startup_poll_seconds` is the poll cadence.

## Cost containment in the startup window

Every second between pod creation and runner registration is billed at the
GPU rate, and a rejected candidate pays it too, so the startup script keeps
only what registration and the smoke proof need:

- the driver/VRAM check, the CUDA smoke proof, and the runner bootstrap;
- the build toolchain, `git`, `jq`, and `zstd` are installed by the test
  job's first step, which runs after registration and only on the accepted
  pod. `zstd` still lands before the `actions/cache` restore step there,
  which is what keeps the cache version hash compatible with the
  zstd-equipped hosted publisher;
- the pinned actions-runner tarball is served from the pod's persistent
  volume (`/workspace/runpod-ci-cache`) when an earlier pod populated it,
  and its SHA-256 still decides whether the cached copy is usable. Every
  step of the cache path degrades to the normal download, so a missing,
  unwritable, or stale cache cannot fail startup.

## Observability

- Each attempt logs candidate name, GPU type, hourly price
  (`costPerHr`/`adjustedCostPerHr` from the pod record), outcome, rejection
  reason, and startup or wasted seconds with an estimated paid cost in
  dollars. Rejections also accumulate, so an accepted pod reports what the
  rejected candidates before it cost and an exhaustion error reports the
  total.
- The accepted pod's GPU, tier, price, startup time, and attempt count go
  to the job summary and `gpu_cost_per_hr` output; the pod-side "Check
  machine" step echoes them next to `nvidia-smi`.
- `cleanup-runpod` reads the pod record before deletion and logs paid time
  and estimated cost for the whole run.

## Local GPU validation instead of provisioning

When the provider cannot deliver a runner, the paid path otherwise fails after
spending pods, and every retry spends more. The intended substitute is a
maintainer decision recorded on the PR: the `gpu-validated-locally` label plus a
PR comment containing a line that starts with `Local GPU validation:`, naming the
GPU, the commit, the commands, and the observed result.

`authorize` reads the label and publishes `local_gpu_validation`. Passing that
decision to the paid workflow through `workflow_call` inputs did not work: a live
labelled dispatch still scheduled `start-runpod` with the boolean input
comparison and again after switching to a textual one, because job-level `if`
conditions in a called workflow do not see the caller's inputs. The same dispatch
showed `inputs` *is* populated inside steps (`Revalidate queued PR before
provisioning`, gated on `inputs.pr_number`, ran), so the decision is applied in a
step: `start-runpod` reads the PR's label, publishes `paid_path_skipped`, skips
the provisioning step, and `run-gpu-tests` refuses to wait for a runner that will
never exist. A labelled PR therefore creates no pod, and `ci-gpu-gate` publishes
success from the recorded evidence (verified end to end). The paid gate remains
the required path whenever a runner is available, and #1907's early stop bounds
the spend when the provider is down.

## Security invariants (unchanged)

JIT runner registration, maintainer/admin authorization, fork rejection,
read-only workflow permissions, and unconditional cleanup are preserved.
Secret isolation, precisely: the long-lived RunPod API key and GitHub App
token never reach the pod; the single-use, single-job JIT runner config is
the one credential that necessarily does, and the smoke child runs with it
stripped from its environment (`env -u RUNNER_JIT_CONFIG`). The pricing
query is unauthenticated. The new scripts are part of the GPU control
plane in `change_policy.py`, so changing them requires the GPU gate.

## Residual risks

- The GraphQL pricing endpoint is not part of the versioned REST contract;
  failures fall back to static tiers by design.
- Pod-status polling depends on RunPod reporting exited containers
  promptly; the per-candidate timeout bounds the damage.
- End-to-end behavior on paid hardware (smoke rejection, candidate
  failover, cost logging) can only be demonstrated in live CI runs.
- Pod logs are not exposed by any RunPod API; the `keep_failed_pods`
  dispatch input keeps rejected pods alive (billing!) so their console
  logs can be read in the RunPod dashboard when a smoke failure needs
  manual triage.
