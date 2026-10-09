# RunPod GPU Provisioning: Cheapest Compatible Host

Status: active. Implements issue #1404 under the #1401 CI performance
umbrella; the paid-cost controls below implement #2002. Companion code:
`scripts/ci/runpod_pricing.py`, `scripts/ci/cuda_smoke_test.py`,
`scripts/ci/runpod_provision.py`, `scripts/ci/gpu_gate_reuse.py`,
`scripts/ci/runpod_cost.py`, `scripts/ci/runner_pin_check.py`; contract tests
in `scripts/ci/tests/`.

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

## One paid pod per tested merge ref

`runpod-gpu-test.yml` subscribes to `workflow_run` `in_progress` so trusted
preparation starts early, and GitHub delivers that event several times per
upstream run (#2002 measured 3.0 RunPod runs per upstream run; 61% of the
successful-pod spend re-tested a merge ref that had already been paid for).
The exact delivery rule is not documented, so the fix does not depend on it:

- The parent workflow has a concurrency group per PR head
  (`runpod-gpu-prepare-<head sha>`, `cancel-in-progress: false`). The first
  delivery starts at once, so early-start latency is unchanged. Later
  deliveries for the same head wait; GitHub keeps only the newest waiting run,
  and it starts after the first run has finished, gate included. Manual
  dispatches use their own run id as the group.
- The gate records what it validated: the `CI GPU gate` check run carries
  `external_id = runpod-gpu-gate:v1:<tested ref>:<kind>`, where the tested ref
  is the pinned merge SHA (or the dispatched ref) and the kind is `paid`,
  `reused`, `failed`, `not-required`, `local`, or `skipped`.
- `Decide whether the paid path runs`, inside the global paid lock and right
  before any spend, asks `gpu_gate_reuse.py` for every gate on the PR head
  (`filter=all`). If the newest completed gate for exactly this tested ref is a
  `paid` or `reused` success, no pod is created and the gate publishes
  "passed (reused)" with a link to that check. A moved base is a different
  merge ref and is never reused; a newer failure for the ref re-enables the
  paid path; gates published before this marker existed are never reused.
- The lookup fails open toward validation: any API, parse, or script error
  answers "run", which is the previous behavior.

A failed paid gate is still retried by any later delivery for the same ref.
Re-run a failed gate with `gh run rerun <run id> --failed` (the failure
summary names it).

## Bounded setup on the pod

Before #2002 the setup steps of `run-gpu-tests` were bounded only by the
45-minute job timeout; one artifact download took 34 minutes and another pod
spent 25.8 minutes downloading plus 10.3 minutes restoring the CUDA runtime
before it was cancelled without a test result. Every step before
`Run CUDA tests from archive` has its own `timeout-minutes`; the aggregate
watchdog below is stricter than the sum of those individual limits. A normal
setup takes about 2.5 minutes; archive downloads allow one bounded retry:

- the cache restores abort a stalled segment after
  `SEGMENT_DOWNLOAD_TIMEOUT_MINS=2` and are non-blocking, because a test-archive
  miss falls back to its immutable artifact. Runtime dependency preparation
  and cache repair happen on the hosted prerequisite;
- the archive artifact download is retried once with the same bound.

A separate hosted setup watchdog bounds the accepted pod from its
`lastStartedAt` to the start of `Run CUDA tests from archive` to 900 seconds.
It retains the RunPod credential on the hosted runner, polls read-only job
progress, and confirms deletion if setup expires or the GPU job finishes
before reaching tests. Progress API failures do not extend the deadline;
unreadable pod start metadata triggers deletion and a visible failure.
A bounded inline deletion step also covers checkout/helper failures before
the guard can finish; an already-deleted pod is accepted idempotently.
Real test execution disarms this setup-only guard, so the complete numerical
suite retains its existing timeouts. Normal cleanup remains mandatory and
idempotent. Seeding the test archive onto persistent storage remains deferred.

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
  and estimated cost for the whole run (`runpod_cost.py`). RunPod's REST
  `lastStartedAt` is Go's time format (`2026-10-04 11:09:24.633 +0000 UTC`),
  not ISO-8601; the previous inline parser raised on it in every cleanup job
  that had a pod (#2002). Both forms are parsed, an unreadable record prints a
  `::warning::` with the raw value, and the report runs as a non-blocking step
  before the deletion step, so telemetry can never keep a pod alive.
- The provisioner runs with `PYTHONUNBUFFERED=1`, so each provision log line
  carries its event time rather than the time a block buffer was flushed.

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
success from the recorded evidence. Until #2002, `authorize` printed the label
decision to stdout instead of `GITHUB_OUTPUT`, so the gate never saw it and
passed labelled PRs without checking the evidence comment; the decision now
reaches the gate. The paid gate remains
the required path whenever a runner is available, and #1907's early stop bounds
the spend when the provider is down.

## Runner pin runbook

The pod registers a JIT runner from the `actions/runner` release pinned as
`RUNNER_VERSION` / `RUNNER_SHA256` in `runpod-gpu-execute.yml`. GitHub stops
queueing jobs to runners that fall too far behind; a rejected runner looks like
a pod that passes the CUDA smoke proof and never registers, and the provision
ladder pays for every candidate (#1921: 2.335.1 was rejected on 2026-09-24,
29 days after 2.337.0 and 66 days after 2.336.0 shipped).

`.github/workflows/runner-pin-check.yml` runs `runner_pin_check.py` daily (and on
PRs that touch the pin or the check). It needs no secret. It fails when the pin
is not a published stable release, when `RUNNER_SHA256` differs from the
`linux-x64` checksum in the release notes, when two or more newer releases
exist, or when a newer release is at least 14 days old; one younger newer
release only warns. The check is not a required PR check.

To bump the pin:

1. `gh api repos/actions/runner/releases/latest --jq '.tag_name, .body'` and
   take the `actions-runner-linux-x64-<version>.tar.gz` SHA-256 from the
   "SHA-256 Checksums" section.
2. Update `RUNNER_VERSION` and `RUNNER_SHA256` together in
   `runpod-gpu-execute.yml`.
3. `python3 scripts/ci/runner_pin_check.py` locally must print `verdict=ok`;
   the PR also runs the check.
4. Registration is proven by the next paid run: its `start-runpod` log shows
   `Runner ... online` for the accepted pod. The persistent-volume tarball
   cache is keyed by version and checksum, so the new tarball is downloaded
   once and cached again.

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
- The hosted setup watchdog also covers a queued GPU job, but its own hosted
  queue can delay enforcement. The 900-second deadline is not extended by
  that delay; deletion still requires working provider APIs and is bounded
  by polling and request/retry time once the guard is running. Permanent
  deletion failures remain visible and normal cleanup makes another attempt.
  Live healthy-start and stalled-setup validation must be recorded before
  claiming an end-to-end billing bound.
- The gate reuse and the parent concurrency group run from the default branch,
  so they first execute live on the first RunPod run after they merge.
