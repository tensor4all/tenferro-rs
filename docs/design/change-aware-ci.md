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
  tests and actionlint. RunPod workflow, request, recovery, runtime preparation,
  CUDA-header validation, or classification changes additionally require the
  GPU gate. Hosted preparation still determines the payload consumed on the
  paid GPU, even when the helper itself runs without a GPU.

Mixed docs and CI changes are CI-only with both lightweight flags enabled.
Unknown paths always fall back to code. `docs/tutorial-code/` is executable
workspace code, not a docs-only exception. Shared tensor, runtime, AD, CPU,
and extension dependencies can affect GPU behavior, so source changes retain
the conservative full policy rather than a GPU-directory allowlist.
Pushes to `main` force the
comprehensive Linux and macOS matrix; they do not add a second paid RunPod GPU
run after the pull-request gate.

Required job names do not disappear when work is unnecessary. Each required
job either runs its selected profile or publishes an explicit successful
no-op. A classification failure is a validation failure, never a cheap
fallback.

## Shared command profiles

`scripts/ci/run_profile.py` owns immutable profiles for workspace-faer,
workspace-blas, macos-accelerate, provider injection, extensions, documentation, coverage, and
CI configuration. Local and hosted execution call the same profile names.
`full` is composition rather than another command list, so repeated profiles
execute once. Hosted Rust compile/test commands use workspace `[profile.ci]`
(`opt-level=0`, `debug=0`, `incremental=false`, `strip="symbols"`) rather than
`--release`; local `dev`/`test` profiles stay incremental. Non-incremental CI
avoids generating per-crate edit-loop state on ephemeral runners; it does not
disable Cargo's reuse of unchanged dependency artifacts.

On GitHub Actions, the profile runner logs the detected CPU count and explicitly
sets Cargo build jobs (CPU count, capped at 16), nextest test concurrency, and
libtest threads (CPU count). Explicit environment overrides are preserved;
local execution is unchanged. Test scheduling is separate from backend-internal
BLAS/Rayon threading. GPU archive execution keeps its independent serial policy.
Coverage uses `cargo llvm-cov nextest`, allowing tests from different binaries
to overlap while retaining the existing per-file coverage thresholds. It does
not replace Linux doctest validation or the external extension fixture.

## macOS support gate

`macOS workspace tests` is a required Apple Silicon execution gate for code
changes and `main` pushes. It runs `macos-accelerate` on `macos-15`, disabling
default features and selecting Accelerate rather than faer. Only the dedicated
Apple context, CPU/Metal FFT, and Apple shared Cholesky test targets are built;
the latter also checks Accelerate GEMM and Cholesky independently of Metal.
Selecting targets rather than runtime test-name filters avoids compiling the
full integration suites. Common workspace tests, doctests, and scalar codegen
checks remain in Linux lanes. The profile requires Metal initialization to
succeed (`TENFERRO_REQUIRE_METAL=1`); an unavailable device is not a passing
hardware test. After change classification, the job starts in parallel with
the selected Linux workspace and extension lanes. Linux and macOS remain
independent required checks, reducing pull-request wall-clock time while still
failing closed on either platform. The trusted RunPod workflow starts from the
workspace workflow's `in_progress` event; its hosted archive build overlaps
CPU/macOS/coverage/docs validation, while `pre-runpod-gate` waits only for
successful `rustfmt` and `clippy` before provisioning a paid pod. The other
checks remain independent merge blockers.

The change policy also runs this gate when its workflow, shared profile, or
classifier changes, so CI-only edits can validate the native lane. Other
CI-only and docs-only changes run an explicit successful no-op on
`ubuntu-latest` instead of allocating a macOS runner. A Linux gate failure also
does not cancel an already-running macOS lane; both results remain visible and
merge-blocking. The older Linux-hosted Apple cross-target type-check is removed
because the real macOS workspace run compiles and executes the same
target-gated code.

## Overlapping preparation without adding paid GPUs

The parent `runpod-gpu-test.yml` authorizes and prepares each request independently.
Its hosted preparation remains read-only and may overlap other PRs' paid runs.
After authorization, lint and archive success, it calls the trusted local reusable
`runpod-gpu-execute.yml`. That workflow holds one repository-wide concurrency
group from head revalidation through provisioning, execution and cleanup; it
uses `queue: max` so a third request cannot evict the second pending request.
No cancellation of an active paid lifecycle is introduced. The parent publishes
the required check only after the reusable workflow finishes, including cleanup.
A stale/closed/forked PR fails before provisioning, and manual refs are resolved
to immutable commits before preparation. Existing debug keep-pod behavior remains
an explicit maintainer-only exception.

Heavy doctest crates use edition 2024 to combine compatible examples, without
removing examples or either Linux backend lane. Edition-sensitive tests retain
individual compilation where needed. Scalar assembly inspection uses the same
`ci` profile as the tests, including the correct assembly output directory.

The hosted CUDA toolkit is restored by exact versioned key. Only the existing
main-only cache publisher saves it; PR builds never publish toolkit contents.
A miss still installs the same toolkit and all builds remain valid without cache.

## Device-independent and hardware test partitions

PR relevance and physical-device requirements are separate decisions. Hosted
preparation executes device-independent tests from the **same CUDA/PJRT-feature
archives** used on RunPod, on both fresh builds and cache hits. This execution
must succeed before provisioning. It does not substitute default-feature tests
for CUDA-feature coverage. The paid node executes only the audited single-GPU
partition, the CUDA tutorial, and the CUDA-plugin PJRT E2E cases.

`scripts/ci/{cuda,pjrt}_test_partition.tsv` records exact binary/test identities.
`gpu_test_partition.py` checks the real nextest inventory for exhaustive,
disjoint membership before selecting a lane. New, removed, or renamed tests
fail until their bodies and helper calls are audited and the inventory updated;
CUDA names, feature gates, and `#[ignore]` alone are not classifications. Host
execution also catches an existing host test acquiring a device dependency.
The archive is extracted once and its build metadata reused without compilation.
Nextest remains serial; hosted backend thread variables are explicitly one.

Single-GPU CUDA tests are ignored in ordinary runs and fail without a device
when explicitly selected. Required PJRT execution fails without its configured
plugin. The single-GPU workflow explicitly reports **NOT RUN**, never PASS, for
the two-device registration test; a separately provisioned two-GPU run is outside
this lane. Existing trybuild checks remain in ordinary hosted CI, the dedicated
A100 benchmark remains outside correctness CI, and the optional `run_hlo_module`
checks are reported not run because this workflow does not install that external
tool. These exclusions have separate inventory categories, not host/device
success results. No test bodies or numerical assertions are removed.

The legacy `CI_gpu.yml` fork/manual fallback retains its existing full archive
selection; this partition changes the trusted RunPod preparation/execution path.
Before trust adoption, a PR run using the old default-branch workflow validates
the test changes, not the new workflow wiring. Validate that wiring locally and
through a trusted post-merge dispatch, without executing a PR-controlled
secret-bearing workflow.

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

After confirmed pod deletion, the hosted cleanup job emits a best-effort JSON
artifact with the tested ref, archive/cuTENSOR/CUDA cache-hit observations and
estimated paid cost by stage. The paid window uses the pod's start timestamp
through confirmed deletion and its adjusted hourly price, with list-price
fallback. Startup/queue, test-job setup and execution, and cleanup/queue remain
visible; unassigned overhead reconciles the stage totals to the paid window.
Missing cache outputs remain unknown. Failed workloads retain their failure
conclusion. Reporting and artifact upload are bounded, nonblocking, and run
after deletion so they cannot extend this pod's paid lifetime. Rejected
provisioning attempts and storage charges are separate from this estimate.

The hosted lifecycle watcher retains the setup deadline through CUDA test start
and monitors PR obsolescence through GPU job completion. It deletes the pod
before cancelling a moved-head/closed PR run; immutable manual validations keep
running. An independent trusted-main reaper recovers explicitly tagged CI pods
after completed/superseded workflows or the two-hour lifetime backstop, while
respecting debug retention. See [RunPod lifecycle and orphan recovery](runpod-gpu-provisioning.md#orphan-recovery)
for ownership, API-failure behavior and read-only cost-history collection.

The CUDA/PJRT test archive key is content-addressed across source, manifests,
tests, lockfile, workflow, and RunPod configuration. It excludes branch, ref,
and commit identity, allowing equivalent automatic and recovery runs to reuse
the hosted cache while still uploading a per-run artifact for the external
runner. Hosted archive builds produce separate `cuda-tests.tar.zst` and
`pjrt-tests.tar.zst` nextest archives; the GPU node runs both from archive and
does not compile Rust (PJRT plugin wheels remain a runtime download). The
CUDA tutorial uses the workspace `ci` profile and is archived from `target/ci`,
so it does not create a separate release-profile rebuild.

On a pod archive-cache miss, the three compressed archives travel as five
balanced, uncompressed Actions artifacts, enabling download-artifact's five
concurrent artifact downloads. The trusted workflow embeds the transport shell
steps so recovery of older tested refs does not depend on new helper files.
The pod reconstructs the archive files and verifies their hosted SHA-256 sums
before testing. Download retains a four-minute bound and one bounded retry;
reconstruction has a one-minute bound. Transfer through verification is recorded
in the job summary. A separate single artifact remains available for hosted
cross-run reuse; the cache content key and restore-only ownership are unchanged.

The archive is compiled with cudarc's CUDA 12.8 binding set, while CubeCL JITs
PTX on the external runner. RunPod therefore accepts CUDA 12.6-or-newer hosts
and chooses NVRTC after reading the assigned host's driver API: NVRTC 12.6 for
the baseline tier and NVRTC 12.8 for hosts supporting CUDA 12.8 or newer. This
keeps PTX compatible with older drivers while retaining all hardware-supported
CubeCL features on the newer tier. Before tests, the runner logs both versions
and rejects runtimes below 12.6 or NVRTC newer than the driver.

Runtime tree caches contain CUDA JIT headers (including CRT and CCCL) and
shared runtime libraries. Soname links and the `include`/`lib64` aliases remain
links to one physical copy. Compiler binaries, static libraries and driver
stubs are excluded. The main-only publisher prepares both supported runtime
tiers; consumers restore the same versioned format and retain installation
on a miss. A tree is marked complete only after its library links resolve.

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

## Hosted GPU execution dependencies

The read-only runtime preparation workflow runs on a hosted Ubuntu 24.04
runner before the paid lifecycle, alongside test archive preparation. It
restores trusted cuTENSOR and minimal CUDA 12.6/12.8 caches, installs misses
on the hosted runner, verifies the JIT headers with real NVRTC, and prepares
immutable five-part artifacts for common tools/PJRT wheels and each SDK.
The trusted controller passes the authorized tested ref to the read-only
preparer, which checks out that revision and executes its payload/header
helpers. Thus the GPU gate consumes dependencies prepared by the proposed
helper code as well as test archives compiled from the same revision. No
provider secrets or cache-write credentials enter this job. Workflow
orchestration remains on trusted main; proposed workflow definitions are
checked by CI configuration tests/actionlint and verified after merge when
the controller changes. Shared cache publication remains owned by
ci-cache-publish on main.

The digest-pinned CUDA 12.6.3 runtime image retains the 12.6 driver floor.
Pre-registration NVRTC compile/load/launch validation remains mandatory.
After registration the GPU selects the driver-compatible SDK (12.8 for the
full capability tier), transfers only that SDK and the common payload,
verifies checksums and required libraries, and runs the complete archives.
The common payload retains the single-GPU `libcutensor.so*` ABI and its
symlinks, omitting static archives and the independent multi-GPU/MPI libraries.
The hosted cuTENSOR cache is preserved. Cargo/nextest and PJRT wheels are staged; no Rust toolchain installation or
PJRT package download occurs on the accepted GPU. Preparation failures block
allocation and fail the required GPU gate. SDK and cuTENSOR paid-side cache
hits are reported as unknown because those dependencies are now host-staged;
the test archive's restore-only cache remains unchanged.

## GPU cleanup and cost reporting

After GPU execution, the hosted cleanup job reads the soon-to-be-deleted Pod
record with a two-second connection timeout and a five-second total timeout.
A missing record or failed read does not prevent deletion. Mandatory DELETE
runs before any checkout or cost calculation; only confirmed deletion permits
the reporting checkout and stage-cost artifact. The stage report uses the
recorded deletion timestamp, so time spent reporting is not counted as paid
GPU time. Reporting remains best effort and does not change the GPU result.
