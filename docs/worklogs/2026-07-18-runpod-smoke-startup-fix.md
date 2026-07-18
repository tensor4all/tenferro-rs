# RunPod Smoke Startup Fix After Live Verification

## Session summary

The first live verification dispatch of the #1404 provision loop failed:
all four created pods (RTX 2000 Ada, L4, A40, RTX 5090) timed out after
420 s without registering the runner, and the loop exhausted its bounded
budget (which itself worked as designed, including price ordering,
per-attempt JIT minting, immediate deletion, and cost logging). This
session diagnosed the failure from the run logs and fixed the three
implicated mechanisms.

## Context read

- Live run 29640572328 logs (provision attempts, timings, rejections)
- Previous-flow timing evidence: pod creation to runner online took 31 s
  (run 29638763285), so 420 s timeouts on every host indicate a startup
  script failure, not slow hosts
- RunPod API docs and community reports: pod logs are not exposed through
  any public API; `desiredStatus` stays RUNNING after the container exits;
  the GraphQL `runtime` object is the container-liveness signal

## Diagnosis

- The dead-pod detection polled REST `desiredStatus`, which never reflects
  container exit — every startup failure burned the full per-candidate
  timeout and was misreported as "timed out".
- The most likely common-mode startup failure is the pod-side
  `raw.githubusercontent.com` fetch of the smoke script (datacenter IPs
  are aggressively rate-limited for unauthenticated raw fetches); the URL,
  pod env, and script content were verified correct from the hosted side.

## Chosen design

- Embed `cuda_smoke_test.py` into the startup script from the trusted
  checkout (heredoc splice, verified byte-identical and py_compile-clean
  locally) — no pod-side network fetch remains for the proof.
- Detect container death through GraphQL `pod { desiredStatus runtime }`:
  once a runtime has been observed, a null runtime before runner
  registration is an immediate, authoritative failure signal; boot-phase
  null runtime (image pull) is tolerated.
- Add a `keep_failed_pods` workflow_dispatch input (default false) that
  keeps smoke-rejected pods alive for manual console-log triage in the
  RunPod dashboard — the only place pod logs exist.

## Rejected alternatives

- Fetching the smoke script with authentication: would put a credential on
  the pod.
- Serving smoke diagnostics over a pod port: expands the pod's network
  surface for a debug concern the dashboard already covers.
- Raising the startup timeout: the 31 s historical baseline shows timeouts
  were a symptom of broken failure detection, not slow startup.

## Verification

- 150 unit/contract tests pass (new: container-stop fast-fail, boot-grace,
  keep-mode, GraphQL state parsing; updated: embed and debug-input
  contracts)
- `actionlint`; local render of the startup-script composition with a
  byte-identical embedded script check
- A follow-up live dispatch after merge revalidates end-to-end

## Live-log confirmation and final fix

A `keep`-free re-dispatch after the embed/GraphQL fix reached the pods and
the RunPod console log gave the definitive root cause: on a driver-13.0 L4
host, `cuda-nvrtc-12-8` was installed correctly but the smoke loaded
NVRTC **11.8** — the bare `libnvrtc.so` from the pod image's CUDA 11.8
toolkit shadows the selected tier because it sits in the default linker
path. The version-consistency check then correctly failed the proof on
every otherwise-compatible host. Fixed by ordering NVRTC load candidates
most-specific-first (`/usr/local/cuda-X.Y/...`) and dropping the bare
`libnvrtc.so` name entirely (`nvrtc_library_candidates`, unit-tested).

## Residual risks

- If the live failure was not the raw fetch, the next dispatch will now
  fail fast with "container stopped" and `keep_failed_pods=true` exposes
  the console logs for direct diagnosis.
- The GraphQL runtime signal is outside the versioned REST contract;
  transient query failures degrade to the bounded timeout.
