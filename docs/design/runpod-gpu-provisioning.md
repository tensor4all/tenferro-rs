# RunPod GPU Provisioning: Cheapest Compatible Host

Status: active. Implements issue #1404 under the #1401 CI performance
umbrella; the paid-cost controls below implement #2002. Companion code:
`scripts/ci/runpod_pricing.py`, `scripts/ci/cuda_smoke_test.py`,
`scripts/ci/runpod_provision.py`, `scripts/ci/gpu_gate_reuse.py`,
`scripts/ci/runpod_cost.py`; contract tests
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

1. GET the authenticated REST v2 `/catalog/gpus` endpoint with
   `include=AVAILABILITY`, `product=POD`, `cloud=SECURE`, GPU count and CUDA
   `minCudaVersion=12.6` for Secure Cloud stock, VRAM and the eligible types' list prices.
2. Drop out-of-stock types, types without a Secure Cloud offer, and types
   below `min_vram_gb`.
3. Emit one single-GPU candidate per type, cheapest first, capped by
   `max_price_candidates`.
4. Reserve a static candidate slot per reviewed tier inside the provision attempt
   budget, reducing the live-price prefix if necessary. Interleave remaining
   single-type static candidates by tier and deduplicate GPU IDs already priced.
   Each tier is reachable within the configured six attempts; the independent
   consecutive-startup-failure stop still limits paid failures.

Catalog queries and every create request use the same open-ended CUDA floor:
`min_cuda_version: "12.6"` in the configuration maps to catalog
`minCudaVersion=12.6` and Pod `gpu.minCudaVersion: "12.6"`. Exact-version
allowlists excluded compatible newer hosts, including CUDA 13.2 A100s observed
in the catalog. The lower bound admits future driver versions without changing
the configuration; the runtime smoke proof below remains the real
compatibility decision. See the RunPod [catalog filter](https://docs.runpod.io/api-reference-v2/catalog/list-gpu-types)
and [Pod creation](https://docs.runpod.io/api-reference-v2/pods/create-a-pod) contracts.

Runner readiness queries GitHub by the unique per-attempt runner name, then
checks its online status and matching label. The JIT minter uses the same value
for name and label. Never infer absence from the first page of an unfiltered
organization registry: organizations with more than 100 registrations can have
the accepted runner on a later page.

## Runtime compatibility proof (smoke test)

`cuda_smoke_test.py` runs on the pod, as root, inside the startup script
BEFORE the GitHub runner registers — so an incompatible host is rejected
before any dependency setup or test execution:

1. driver visibility and CUDA API version via `nvidia-smi`,
2. runtime tier selection mirroring the workflow (12.8 full / 12.6
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
  passed) and REST v2 Pod `status` plus `runtime`. Accept only RUNNING with
  runtime present and the fresh attempt's runner online. EXITED, ERROR and
  TERMINATED reject immediately; runtime loss after boot also rejects;
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
only what registration and the smoke proof need. The official GitHub
`actions/actions-runner:latest` image is resolved to its current OCI digest
on the hosted runner before provisioning. It supplies Python, download tools, the
runner and its native dependencies before the paid startup script begins.
The script uses the image's passwordless sudo to retain root execution,
downloads and verifies a pinned driver-compatible NVRTC package, proves a
CUDA kernel can run,
and only then registers the preinstalled runner. NVIDIA compute/utility
capabilities are passed explicitly because this is not a CUDA base image.

`cuda_smoke_test.py` owns the NVRTC 12.6/12.8 Debian package versions and
SHA-256 values from NVIDIA's official repository. These packages have no
package dependencies and are installed directly with `dpkg`, avoiding a
package-index refresh before registration. A download or checksum failure
stops startup before registration. When changing a supported runtime tier,
update this pin, verify native compilation and the full GPU proof, and keep
the staged execution SDK's runtime selection aligned.

The accepted job installs its remaining execution tools without recommended
packages. `zstd` lands before archive cache restore, retaining compatibility
with hosted cache writers. Readiness is polled every ten seconds; startup,
setup and lifetime deadlines and failed-candidate deletion remain bounded.
There is no pod-side runner archive download or persistent tarball cache.

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
`startedAt` to the start of `Run CUDA tests from archive` to 900 seconds.
It retains the RunPod credential on the hosted runner, polls attempt-specific job
progress, and confirms deletion if setup expires or the GPU job finishes
before reaching tests. Progress API failures do not extend the deadline;
unreadable pod start metadata triggers deletion and a visible failure.
A bounded inline deletion step also covers checkout/helper failures before
the guard can finish; an already-deleted pod is accepted idempotently.
Real test execution disarms the setup deadline, while the same hosted job
continues monitoring through GPU job completion with a 3600-second total
accepted-pod lifetime backstop. It stops work when the authorized PR head moves
or the PR closes, confirms pod deletion, then cancels the obsolete workflow.
The provisioner checks the same PR condition before each candidate and while
waiting for runner registration. Unknown PR/API state never means obsolete;
existing deadlines remain effective. A base-branch update alone does not abort
an already-running immutable test. Manual revision validations (`pr_number=0`)
retain their pinned workload. The complete numerical suite keeps its existing
timeouts, and normal cleanup remains mandatory and idempotent.
Debug retention applies to genuine startup failures; a pod made obsolete by
PR closure/head movement is deleted even when `keep_failed_pods` was requested.

## Orphan recovery

`runpod-reap.yml` runs independently of the GPU workflow every 15 minutes on
trusted main. It considers only pods tagged by the provisioner with repository,
workflow owner, run id, and attempt, whose name agrees with that identity.
Untagged/foreign pods and explicitly recorded startup failures retained for
inspection are excluded. `keep_failed_pods` alone does not exempt a pod: the
provisioner records actual retained IDs in a small `runpod-debug-retained`
artifact, whose name includes the run, attempt and comma-separated pod IDs.
The reaper checks this record only for otherwise eligible debug-dispatch pods;
accepted pods and failed obsolete deletions remain reclaimable. If the retention
record cannot be uploaded or is later removed, the pod is reclaimable. An API
failure reading retention records leaves the decision unknown for that sweep. The
owning GitHub run must match the repository and GPU workflow path. A candidate
is reclaimed if its run completed at least five minutes ago, its attempt was
superseded, or its active pod exceeds the two-hour lifetime backstop. A v2
Pod stuck before startup uses `createdAt` for orphan age when `startedAt` is
null; this fallback is never used for cost estimates. API errors
leave the pod untouched and fail visibly for the next scheduled retry.

Deletion is bounded and idempotent; cancellation follows confirmed deletion and
never intentionally targets a newer attempt. A manual dispatch defaults to
dry-run (`execute=false`). Decision artifacts retain pod/run identity, deletion
time, and available cost estimates, excluding pod environment/credentials.
Schedule and hosted-runner queue delays, GitHub outages, and persistent provider
DELETE failures mean this is best-effort recovery, not an independent billing
guarantee. Legacy untagged pods need the existing manual cleanup workflow.
Seeding the test archive onto persistent storage remains deferred.

## Observability

- Each attempt logs candidate name, GPU type, hourly price
  (`cost` from the v2 pod record), outcome, rejection
  reason, and startup or wasted seconds with an estimated paid cost in
  dollars. Rejections also accumulate, so an accepted pod reports what the
  rejected candidates before it cost and an exhaustion error reports the
  total.
- The accepted pod's GPU, tier, price, startup time, and attempt count go
  to the job summary and `gpu_cost_per_hr` output; the pod-side "Check
  machine" step echoes them next to `nvidia-smi`.
- `cleanup-runpod` captures the pod record with a five-second bound, deletes
  the pod, and reports its paid window by stage (`runpod_workflow_cost.py`).
  v2 uses ISO-8601 `startedAt` and `cost`; historic saved v1 records with
  Go-format `lastStartedAt` and `costPerHr` remain readable. Unknown or null
  start times are never replaced with `createdAt`. An unreadable record prints a
  `::warning::`, and reporting runs as a non-blocking step after deletion.
  Early watchdog deletion logs the cost from its already-held pod metadata
  before requesting workflow cancellation, preserving interrupted-run evidence.
- The provisioner runs with `PYTHONUNBUFFERED=1`, so each provision log line
  carries its event time rather than the time a block buffer was flushed.

For read-only cross-run analysis, run:

```bash
python3 -m scripts.ci.runpod_cost_history \
  --since 2026-10-06 --until 2026-10-09 \
  --directory /tmp/tenferro-runpod-history
```

The command collects every workflow attempt in the inclusive UTC date range,
downloads logs and only the small cost artifacts, and writes JSON/Markdown
summaries outside the repository. Omit the dates to regenerate a report from
the saved evidence without network access. It reports repeated automatic refs
separately from manual confirmations, reused gates, rejected candidates,
per-GPU spend including failures, and stage totals where available. Missing
logs/prices/deletion evidence remain explicit unknowns. Older/rejected-pod log
windows are estimates and can miss provider latency; these are not invoices.
Independent reaper decision artifacts provide the separate recovery record when
normal workflow logs cannot confirm deletion. Compare GPU rankings only on
matched workloads/revisions; these historical totals alone do not select a new
GPU policy.

## Local GPU validation instead of provisioning

A maintainer with a local GPU can stand in for the pod: the PR carries the
`gpu-validated-locally` label, and `scripts/ci/local_gpu_gate.py` runs the GPU
lane (the CUDA and PJRT host and device partitions plus the CUDA tutorial, from
the same archives) at the commit that is pushed, posts the SHA-bound evidence as a
PR comment, and dispatches `local-gpu-gate.yml` on `main`. That workflow holds no
secret and runs no PR code; it re-verifies the dispatcher's role (admin or
maintain, like `check_permission`), the PR's state, origin, label and head SHA,
and the comment (author, PR, first-line marker for that SHA), reads the PR a
second time, and publishes `CI GPU gate` on the head with external_id
`local-gpu-gate:v1:<sha>`, which `gpu_gate_reuse.py` never treats as reusable.

The relay exists because branch protection pins `CI GPU gate` to the GitHub
Actions app: an account token can create neither a check run nor a status that
satisfies it.

For a labelled PR `authorize` publishes `local_gpu_validation`, every RunPod job
after it is skipped, and `ci-gpu-gate` publishes nothing (it re-reads the label,
which may have been added after `authorize`), so the RunPod workflow can never
override the local publication. `CI_gpu.yml`'s PR-attached gate accepts only the
`local-gpu-gate:v1:<head>` check for a labelled PR and waits up to 330 minutes for
it. Inside the paid workflow the label is still applied in a step: job-level `if`
conditions in a called workflow do not see the caller's inputs (a live labelled
dispatch scheduled `start-runpod` with both boolean and textual comparisons), so
`start-runpod` reads the label itself and publishes `paid_path_skipped`.

An earlier version accepted any `Local GPU validation:` comment by a maintainer
as the evidence, which bound it to no commit; it is replaced by the flow above.

## Runner image selection

`RUNPOD_IMAGE` follows the official `ghcr.io/actions/actions-runner:latest`
image. Before creating a Pod, the hosted provisioning job resolves `latest`
with `docker buildx imagetools inspect` and passes `latest@sha256:...` to
RunPod. All candidate attempts in that run use the same resolved image; the
next run resolves `latest` again. Registry lookup failures stop the hosted
job before any GPU allocation.

The selected image and digest appear in the provisioning log and job summary.
The accepted GPU job's setup log records the actual runner version. The
existing CUDA launch proof still completes before runner registration.
Runner releases no longer require manual version/digest updates or a
scheduled stale-pin check. This avoids repeating the old-runner outage in
#1921 while keeping the exact image used by each run traceable.

## Security invariants (unchanged)

JIT runner registration, maintainer/admin authorization, fork rejection,
read-only workflow permissions, and unconditional cleanup are preserved.
Secret isolation, precisely: the long-lived RunPod API key and GitHub App
token never reach the pod; the single-use, single-job JIT runner config is
the one credential that necessarily does, and the smoke child runs with it
stripped from its environment (`env -u RUNNER_JIT_CONFIG`). The catalog
request uses the RunPod key only on the hosted runner. The new scripts are part of the GPU control
plane in `change_policy.py`, so changing them requires the GPU gate.

## Residual risks

- Catalog list prices and availability are advisory; catalog failures fall
  back to static reviewed single-GPU candidates. The assigned Pod `cost` is
  the reporting rate, not a billing invoice.
- Pod-status polling depends on RunPod reporting exited containers
  promptly; the per-candidate timeout bounds the damage.
- End-to-end behavior on paid hardware (smoke rejection, candidate
  failover, cost logging) can only be demonstrated in live CI runs.
- v2 exposes a Pod log endpoint, but this migration does not add a log
  collector. The existing `keep_failed_pods` dispatch input still retains
  rejected pods for manual dashboard inspection and continues billing.
- The hosted setup watchdog also covers a queued GPU job, but its own hosted
  queue can delay enforcement. The 900-second deadline is not extended by
  that delay; deletion still requires working provider APIs and is bounded
  by polling and request/retry time once the guard is running. Permanent
  deletion failures remain visible and normal cleanup makes another attempt.
  Live healthy-start and stalled-setup validation must be recorded before
  claiming an end-to-end billing bound.
- The gate reuse and the parent concurrency group run from the default branch,
  so they first execute live on the first RunPod run after they merge.


## REST v2 migration and request policy

All live RunPod calls use `https://api.runpod.io/v2`, following the
[official migration guide](https://docs.runpod.io/api-reference-v2/migrate-from-v1)
and [OpenAPI schema](https://api.runpod.io/v2/openapi.json). Pod create/get/list/
delete, metadata, hosted lifecycle watchdog, orphan reaper and manual cleanup
are migrated together. Creation sends nested `gpu` and `mounts`, `image`, `disk`,
`entrypoint` and `cmd`. Assigned GPU comes from `gpu.id`. Listing reads `pods`
and bounded cursor pagination before making any reaper decisions. Ownership
environment and exact workflow-attempt checks remain unchanged.

GraphQL inventory and replacement are complete: former `gpuTypes` price/stock
reads now use the catalog, and former Pod `desiredStatus`/`runtime` queries use
`GET /v2/pods/{id}`. No GraphQL runtime dependency remains, so no later switch
is needed for the announced January 2027 retirement. The REST v1 November 15
retirement and October 27 rate-limit changes require no separate URL-only fix.

The normal startup cadence is ten seconds (at most six Pod reads/minute per
active provisioner, previously thirty). Each iteration also checks PR state;
the runner registry is read only after a live RUNNING Pod observation. Pricing
is one catalog read per provision invocation, with no tight retry loop. Existing
global paid-work concurrency serializes automatic GPU creation; manual cleanup,
the watcher and scheduled reaper can still overlap, as can other API-key users.
There is no distributed account-wide rate limiter. Configure fewer parallel
callers if account-wide limits require it; per-process cooldown is not a global
quota guarantee.

HTTP 408/429, transport failures and 5xx use bounded backoff. Retry-After accepts
seconds and HTTP dates and is a minimum, even above the 60-second backoff cap.
Catalog, creation, startup observations and rejected-Pod deletion share a
process-local cooldown across candidate changes. Unknown/malformed observations
back off and never mean container death or a healthy runner. Recovery restores
the normal cadence. Startup requests and waits fit the remaining 420-second
candidate deadline; a longer required pause ends that candidate without an early
retry. Create retains its 60-second deadline and attempt budget.

Shared cleanup uses at most five DELETE attempts within 180 seconds, with
20-second request timeouts capped by remaining budget. A 404 confirms deletion;
permanent errors stop retries. A lost response may be checked with GET only
after cooldown. Exhaustion fails visibly, writes the remaining Pod ID to the
job summary and stops new paid creation. Pre-delete cost metadata is a single
best-effort GET bounded to five seconds inside the same helper; its cooldown
also applies to DELETE, and missing data falls back to the saved startup
snapshot after deletion. Reporting never prevents deletion. Normal cleanup
fetches only trusted helper sources before deletion; if checkout fails, a
bounded v2 curl fallback remains independent of Python files and reports the
remaining Pod ID. Hosted queue/provider outages still prevent a strict billing
guarantee; the scheduled owned-Pod reaper supplies independent recovery.

API-shape, timeout, retry/cooldown, malformed observation, ownership, metadata
and workflow failure paths are verified with fixtures and local HTTP only.
No paid Pod is created to validate this migration. Published v2 schema/example
fixtures record their retrieval source and date in
`scripts/ci/tests/fixtures/runpod_v2_sources.json`.
