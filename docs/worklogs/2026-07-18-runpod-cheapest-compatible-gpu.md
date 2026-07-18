# RunPod Cheapest Compatible GPU Provisioning

## Session summary

Implemented issue #1404: order reviewed RunPod GPU candidates by live
Secure Cloud price, prove CUDA compatibility with an NVRTC → PTX → launch →
synchronize smoke test before runner registration, and retry the cheapest
next candidate with bounded attempts, deleting incompatible pods before any
dependency setup.

## Context read

- Issue #1404 and umbrella #1401 acceptance criteria
- `runpod_client.py` create/retry/tier structure and its transport-injection
  test style; `runpod_contract.py` tier validation
- `runpod-gpu-test.yml` start-runpod job: JIT config, startup script, wait
  loop, delete-on-failure, cleanup
- RunPod REST OpenAPI (no pricing endpoint; pod record carries
  `costPerHr`/`lastStartedAt`) and the public GraphQL `gpuTypes` query
  (verified live: stock, VRAM, secure price without authentication)
- The existing `Verify loaded NVRTC version` runtime check and the
  2026-07-18 CUDA driver filter worklog

## Chosen design

- `runpod_pricing.py`: unauthenticated GraphQL query ordering only the
  reviewed allowlist; stock/VRAM/secure-cloud filters; static tier fallback
  appended so pricing failures degrade to current behavior.
- `cuda_smoke_test.py`: driver parse → runtime tier selection (mirrors the
  workflow) → minimal NVRTC install → ctypes NVRTC compile → driver PTX
  load → launch → sync → readback → VRAM check. Runs in the pod startup
  script before `run.sh --jitconfig`; fetched at the trusted default-branch
  SHA; parameters via non-secret pod env.
- `runpod_provision.py`: trusted-side loop creating one candidate at a
  time, watching runner-online (accept) vs pod-exited (reject) signals,
  deleting rejected pods immediately, bounded by
  `max_provision_attempts`, logging GPU/price/reason/startup/estimated
  cost; replaces the workflow's shell wait loop.
- Cost observability: `gpu_cost_per_hr` output, job summary line, and a
  cleanup-time paid-cost estimate from the pod record.

## Rejected alternatives

- Trusting `nvidia-smi`/metadata alone: cannot prove PTX acceptance; the
  observed failures were driver-level despite identical GPU models.
- Running the smoke test as the first `run-gpu-tests` step: a failure there
  cannot retry another host automatically without re-dispatching the whole
  workflow; pre-registration gating keeps retry inside one trusted job.
- Passing the RunPod API key to the pod for self-reporting: violates the
  secret-isolation invariant; pod status polling gives the same signal.
- A fixed premium GPU model: overpays and still does not guarantee a
  compatible driver.

## External review follow-up

A Codex review of the initial diff requested changes; all findings were
addressed:

- Unverifiable-GPU pods (`AssignedGpuError`) are now published for the
  workflow safety net, deleted with confirmation, and skipped instead of
  leaking.
- Pod deletion is status-aware with bounded retries and a GET-404
  confirmation; an unconfirmed deletion raises `PodLeakError` and stops
  further pod creation.
- The accept path checks the pod is alive before trusting the runner
  registry, closing the stale-online-runner / dead-pod race.
- Live-priced candidates are capped at 3 with a 6-attempt budget so the 3
  static fallback tiers always remain reachable (contract-tested).
- The start-runpod job timeout (70 min) now contains the worst-case
  provision budget (contract-tested).
- The `stockStatus` literal string "None" is treated as out of stock.
- The secret-isolation claim was narrowed to the accurate invariant: only
  the single-use JIT runner config reaches the pod, and the smoke child
  runs with it stripped (`env -u RUNNER_JIT_CONFIG`).
- The test job's CUDA runtime discovery rejects partial trees (the
  smoke's NVRTC-only install) via a library-completeness check, and pod
  API transport errors (URLError/timeout) are treated as transient polls
  instead of aborting the provision loop.
- Each provision attempt mints its own single-use JIT config under a
  fresh per-attempt runner label, closing the stale-label acceptance race
  and the JIT-replay dead end after a registered candidate dies;
  run-gpu-tests targets the accepted attempt's label.
- zstd is installed on the pod before any cache restore so the
  actions/cache version hash matches the hosted publisher's entries
  (found by inspecting live run logs: exact-key "Cache not found").

## Verification

- `python3 -m unittest` over `scripts/ci/tests`: new pricing, smoke-logic,
  and provision suites plus updated workflow contract and change-policy
  tests (140 tests; the 6 pre-existing `LocalGateTests` failures are a
  local Python 3.9 environment artifact reproducing on `main`)
- `actionlint` on the workflow
- Live GraphQL pricing response shape verified against the real endpoint
- A paid RunPod run is still required to validate smoke rejection,
  candidate failover, and cost logging end-to-end

## Residual risks

- GraphQL pricing is outside the versioned REST contract (fallback covers
  outages)
- Pod exit detection latency depends on RunPod status reporting; bounded by
  the per-candidate startup timeout
- `min_vram_gb` is set to 8 to keep every reviewed tier card eligible;
  raising it requires reviewing the tier list
