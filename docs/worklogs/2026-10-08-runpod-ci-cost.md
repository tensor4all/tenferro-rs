# RunPod full CI cost reduction (#2002)

## Decisions

- Record stage costs and cache hits from the trusted execution workflow after
  mandatory deletion. Preserve failed workload conclusions and unknown cache
  outputs; report an estimate rather than a provider invoice. Telemetry must
  never delay or block pod deletion.
- Investigate hosted runtime preparation, slimmer CUDA images and GPU selection
  using the complete frozen CUDA/PJRT/tutorial workload. Experimental benchmark
  workflows do not alter production until complete-workload evidence supports
  promotion. Keep failed pilots and their expense visible.

## Verification conclusions and constraints

- Original production observation 37783818267: A40 at $0.49/hour, 577.430 paid
  seconds, estimated $0.07859; all 285 CUDA cases, three PJRT cases and tutorial
  passed. This single observation motivates work; it does not prove savings.
- Native CUDA PyTorch image diagnostic 37788536318 passed the same full workload
  but cost $0.08658 for 636.124 seconds. Do not promote this larger image.
- Slim-image experiments are ongoing in tenferro-benchmark branch
  experiment/runpod-ci-cost. Paid GPU time is the primary measure; monetary cost is secondary.
- The focused CI helper gate passed; pagination, failure preservation, cache
  unknowns, exact stage reconciliation and nonblocking invalid metadata are
  covered. No Rust numerical or device-execution semantics are changed by the
  telemetry helper.

- Need measurement for the host SVD checker: production run 37783818267 spent
  83.584s and 83.900s in the tall/wide full-SVD-above-1024 cases, about 45%
  of its CUDA test duration. Their host oracle calculates both orientations of
  every Gram entry with per-element Complex64 iterator arithmetic. Threshold
  shape and all acceptance identities must remain; do not sample columns or
  remove the row-unitarity checks.
- Candidate direction: compute each unordered column pair once (Hermitian Gram
  symmetry), use contiguous column slices and scalar real/imaginary dot sums.
  Retain the existing norm tolerance and both U/Vt left/right identities.
  CPU tests compare with the complete original complex Gram reference and
  reject changed entries and nonfinite matrices. Full paid comparison still
  uses all 285 CUDA cases, three PJRT cases and tutorial, serial, profile ci.
  Baseline source is 3f10f960af05efa4fc5705aac3dafbac1ae2269e; freeze the
  committed candidate and matching workflow configuration before any candidate
  measurement. Three complete alternating pairs, median paid time >=20% reduction,
  every pair nonregressing and within-arm max/min <=1.5 remain the primary
  acceptance gates. No speedup is inferred from the static change alone.

- The extracted checker passed two CPU integration tests in the ordinary test
  profile and the same source module's tests compiled with rustc -O against the
  existing release num-complex dependency. This checks numerical parity in
  optimized code; it is not a runtime speed measurement or a full release GPU
  test. The code-change PR gate, including CI-parity clippy, formatting and
  293 CI helper tests, passed. Numerical library code and GPU kernels are
  unchanged; only test-oracle arithmetic and reporting are changed.

- Source diagnostic 37795634794 was cancelled during hosted compilation before
  any pod was created: review found the two added host-check tests also need
  explicit CUDA archive partition entries. Classify both as host so the
  exhaustive inventory remains closed and the GPU workload remains 285 cases.
  The next committed source supersedes 829a4b1a for diagnostics; no primary
  measurements have begun.

## Runtime cache payload reduction candidate

The timed-out CUDA 12.8 cache contained 1,150,165,089 compressed bytes;
restoration waited three minutes before falling back to about 75 seconds of
installation. Seed only JIT headers and shared runtime library families, keeping
soname links and include/lib64 aliases rather than dereferencing duplicate
trees. Compiler, static archives and driver stubs do not belong in the runtime
payload. Bump the restore/publish key to v7 so existing v6 payloads cannot mask
the new format. Both CUDA 12.6 and 12.8 publishers remain.

Actual-copy fixture tests verify headers and chained sonames resolve, only one
physical library copy remains, static/compiler/stub files are excluded, and
broken sonames cannot mark a tree complete. All 295 CI helper tests pass.
The GPU validation and prepared payload sizes below supersede the initial
fixture-only evidence; a cache-format speedup still requires measurement. A40, A6000 and 4090 diagnostic attempts were
out of stock before pod creation; no GPU charges arose from those attempts.

An official CUDA 12.8 package fixture exposed missing crt/mma.h when
compiling the CubeCL-style include set with NVRTC. Include the small cuda-crt
header package in both runtime tiers. With cudart, NVRTC and these headers,
the seeded tree passed real NVRTC compilation without a GPU. This validates
header completeness for the fixture, not full vendor-library GPU execution
or the final production payload size. The full local PR gate passed before
this package-list correction; recheck the affected helpers afterward.

For the official CUDA 12.8 NVRTC/cudart/CRT/CCCL/header fixture, the old
dereferencing copy used 503,706,320 logical bytes versus 239,597,400 for
the seeded tree. This demonstrates fixture payload reduction only; it excludes
cuBLAS/cuSOLVER/cuSPARSE and does not establish full cache size or paid-time
savings.

The equivalent official CUDA 12.6 package fixture also passed real NVRTC
compilation of the CubeCL-style include set after compact seeding, confirming
header completeness for both supported runtime tiers without a GPU. At this fixture stage vendor-library GPU execution was unverified; the
subsequent complete-workload diagnostic below supplies that evidence.

Inspecting the old-copy fixture showed its six NVRTC alias paths share
one inode (link count 6): cp -aL preserved those as hard links. Thus the
fixture size reduction is primarily removal of static archives, not proof
of duplicate physical shared-library storage. Retaining explicit symlink
aliases clarifies the tree but must not be credited with measured byte
savings from those already hard-linked libraries. Full payload measurement
is still required for cache-size or paid-time claims.

Production-compatible diagnostic 37809763815 passed 285 CUDA cases,
three PJRT cases and tutorial on A4000, $0.25/hour, with confirmed deletion
and no rejected pods: 520.432445 seconds, estimated $0.03614. CUDA runtime
12.8 v6 cache restored successfully, so compact seeding was NOT exercised.
SVD tall/wide cases took 8.732/8.917 seconds and CUDA nextest 261.700 seconds.
This is a diagnostic on different hardware from the frozen A40 confirmation;
it neither confirms 20% paid-time savings nor validates the compact GPU tree.

## Compact runtime GPU validation and paid-time confirmation

Hosted benchmark preparation 37814278025 used the production seed helpers
at f3b995657adbbcabc636c5377877205d920a35aa and compiled the complete
CubeCL header set with real NVRTC for both 12.6 and 12.8. Separately split
SDK artifacts total 656,567,385 and 1,074,447,802 compressed bytes. Those
are measured artifacts, not an asserted speedup over the old cache.

Benchmark diagnostic 37815639364 passed all 285 CUDA and three PJRT cases
plus tutorial using the compact SDK, actual NVRTC 12.8, digest-pinned
CUDA 12.6.3 runtime/Ubuntu 24.04 image, and A40 at $0.59/hour. The pod was
deleted; accepted paid time was 452.517209 seconds (~$0.07416). This
validates the compact SDK's GPU correctness. It is not a primary sample.

Previous complete paired campaigns were inconclusive after provisioning
failures (37807870697 and 37808032186); preserve their complete records,
including the initial pair's descriptive 35.65% shorter paid time, without
promoting that single pair. A new three-pair campaign fixes controller
dba171d742d8aff450ea9479706870ecd9456f46, A40, actual NVRTC 12.8, sources
3f10f960af05efa4fc5705aac3dafbac1ae2269e /
2605c46f45016a2472ebdf7ba75bba533bc6a5fd, and prepared runs
37814618112 / 37814278025. First baseline is 37816754595. Runtime
preparation is identical in both arms, so this isolates the SVD oracle;
it does not measure staging versus the original production workflow.
All six must pass the complete workload and deletion, median paid time
must reduce >=20%, every pair must be nonregressing, and each arm's
max/min must be <=1.5. Any failure invalidates the complete campaign; no
selective replacement or exclusions. Confirmation is currently pending.

Campaign 3's first complete pair passed every test and deletion: baseline
37816754595 paid 547.591566 seconds; candidate 37818003215 paid
390.447417 seconds (descriptive 28.6973% reduction). Both were A40 with
NVRTC 12.8 at $0.59/hour. This one pair is not promotion evidence;
second-pair candidate 37818887742 is running and full confirmation remains
pending. A transient API 404 immediately after dispatch was re-polled on
the same run handle, which was confirmed live; no replacement was created.

Campaign 3 is INCONCLUSIVE: candidate 37818887742 passed the complete
workload and deletion in 390.088442 paid seconds, but baseline 37819774823
failed creation with HTTP 500 / no instances available before any pod was
created (zero fee). The complete campaign stopped at that failure. All
three successful samples and the failed run are retained; no promotion
is inferred from the 28.70% first pair.

Production integration uses a separate hosted runtime preparation workflow,
so existing immutable test archive reuse and its content key stay intact.
The hosted workflow restores trusted caches and stages both minimal SDKs,
cuTENSOR, real Cargo/nextest and pinned PJRT wheels. The paid lifecycle
requires its successful output, selects one SDK, verifies and extracts the
five-part artifacts, and retains mandatory cleanup and full test counts.
The image is the GPU-validated digest-pinned CUDA 12.6.3 runtime on Ubuntu
24.04. Artifact packaging/reconstruction, selected-only installation and
corruption rejection are tested with tiny fixtures; a preparation failure
fails the required gate. Production workflow execution and final timing
confirmation are still pending.

The integrated local PR gate passed: 300 CI helper tests, both SVD host
oracle integration tests, formatting and CI-parity clippy for root and
standalone extensions. After the final readiness-output guard, affected
helper tests and actionlint passed again. Runtime identity is emitted only
after every upload succeeds, so failed preparation cannot expose a ready
prefix. The root header checker also compiled the official 12.6 and 12.8
package fixtures with real NVRTC. These checks do not replace the pending
full paid-time campaign or post-merge production GPU validation.

Campaign 4 freezes controller 65cdbb82473642bbf5d53b9d23bc2321858d25ce
with the same sources, GPU, SDK, workload and acceptance conditions. It
allows up to three create-only attempts for the exact no-instances HTTP
500 before any pod exists; any paid-pod failure still invalidates the
whole campaign. This addresses zero-cost capacity failures without
replacing failed paid samples. Baselines 37821964329, 37825246097 and
37826624024 passed the complete workload and deletion in 510.722024,
614.272320 and 622.580076 seconds. Candidates 37823114735 and
37824191820 passed in 465.357591 and 456.052088 seconds; final candidate
37828044819 passed in 469.472534 seconds, with complete workload and
confirmed deletion. The complete campaign passed: median paid time fell
from 614.272320 to 465.357591 seconds (24.2425%), all three pairs were
nonregressing, and both arms stayed within the declared max/min <=1.5
noise bound. This is evidence for the SVD oracle under identical staged
runtime conditions, not an isolated measurement of staging/image savings. Prepared common-plus-selected-SDK artifact sizes differ
by only 10,300 bytes between arms, so larger candidate payloads do not
explain the observed transfer-time difference.

Reproduction conditions: profile ci, one nextest test thread, 200 seconds
per case, 285 CUDA cases, three PJRT cases and tutorial; six fresh A40
pods at $0.59/hour in B,C,C,B,B,C order, CUDA compute/JIT caches cleared
before each test run, and actual NVRTC 12.8 verified. Logs retain hardware,
CPU affinity, per-case results and deletion timestamps. Source 3f10f960's
tree is identical to the original compiled baseline 20b04's tree
(5d87d309d93e7e47435e557771060ca817b2aec4); candidate source is 2605c46f.
The CI profile measures this CI workload, not release kernel performance.

| Pair | Baseline paid seconds | Candidate paid seconds |
|---|---:|---:|
| 1 | [510.722024](https://github.com/tensor4all/tenferro-benchmark/actions/runs/37821964329) | [465.357591](https://github.com/tensor4all/tenferro-benchmark/actions/runs/37823114735) |
| 2 | [614.272320](https://github.com/tensor4all/tenferro-benchmark/actions/runs/37825246097) | [456.052088](https://github.com/tensor4all/tenferro-benchmark/actions/runs/37824191820) |
| 3 | [622.580076](https://github.com/tensor4all/tenferro-benchmark/actions/runs/37826624024) | [469.472534](https://github.com/tensor4all/tenferro-benchmark/actions/runs/37828044819) |

All six passed and their pods were deleted. Confirmation applies to the
frozen comparison above. Post-merge production execution remains required
because its workflow integration is newer than the experimental controller.

PR review found that hosted preparation called both installers even after
restoring valid caches. Preparation now verifies the cuTENSOR shared library,
SDK completion marker and vendor libraries, and compiles the CubeCL headers
with real NVRTC before reusing a restored SDK. Only a missing or incomplete
tree is installed. Fixture coverage confirms a warm cache invokes no
installer and missing cuTENSOR, SDK marker, vendor library or header repairs
only the affected tree. This changes hosted preparation, not the frozen SVD
comparison or the paid workload.

## Post-merge runtime integration follow-up

PR #2043 merged as ac1d2b341be84d6dfce1989d1e5d935ff55dba60 after
all required checks passed. Manual production verification 37834194980
and main-triggered deliveries failed workflow startup before any job or
GPU allocation. Static review identified a permission ceiling mismatch:
the reusable cleanup requests actions: read for its post-deletion timing
report, while its caller grants only checks/contents read. A regression
contract rejects that original configuration. The prepared correction
explicitly grants checks/contents/actions read on the calling job; all
permissions stay read-only. A successful trusted-main run after this fix
is still required; the frozen 24.24% SVD comparison is unchanged.

The permission repair merged in PR #2045 as
485179bc995c97bf177fd18cef0af47e39cf2312. Production run
37855580181 passed hosted preparation but could not register the minimal-image
runner. Its startup omitted the verified runner's dependency installer, unlike
the measured experiment controller. The exact image and runner abort locally
without ICU; invoking the supplied installer makes Runner.Listener report
2.337.0 successfully. The failed production run was cancelled; pod
xyborcbk0z0uwr was deleted by provisioning, and 5sz9fl50lgu4yt by mandatory
cleanup (HTTP 204). The follow-up restores that installer before registration
and guards its order after checksum verification. Full production validation
remains pending this repair; no failed sample is included in the frozen
24.24% comparison.


## Follow-up transfer and setup controls

Runtime setup accounted for 41 of 636.124217 paid seconds in the complete
native-image A40 pilot, exceeding the predeclared 5% need threshold. The next
transfer candidate omits only cuTENSOR static archives and its independent
multi-GPU/MPI providers. The single-GPU shared library and symlinks remain
unchanged; the hosted source cache is preserved. With the frozen runtime
artifacts from run 37855580181, common compressed bytes fell from
1,567,514,494 to 1,196,745,934. Including the selected 12.8 SDK
(1,074,446,861 bytes), required runtime transfer is 14.0338% smaller and passes
the predeclared 5% byte gate. This is a byte-count result, not a paid-time
speedup; full GPU correctness validation passed in run 37861535693: all 285 CUDA
cases, three PJRT cases, and the tutorial passed with NVRTC 12.8, and the fresh
A40 pod was deleted. The observed 340.518493 paid seconds is a diagnostic
sample, not a paired timing claim.
The preserved shared library SHA-256 is
`224d65152fe5bc5d61e00d15081d6f78b16bc6961b1650579e0a82489b9dcdbc`.

The hosted setup watchdog uses a 900-second deadline from the accepted pod's
start timestamp. It confirms deletion on deadline or setup failure, disarms
only when the real CUDA test step starts, and leaves the mandatory cleanup
path independent. Local tests cover stalled setup, queued/skipped jobs,
progress outages, malformed start metadata, deletion retries/failures, and
healthy disarm. Live validation confirmed healthy disarm in run 37861535693. A deliberately
unassigned GPU job in run 37862165502 exercised the same guard with a shortened
120-second budget to limit experimental spend: confirmed HTTP 204 deletion
occurred at 120.942753 paid seconds, followed by successful mandatory cleanup.
This demonstrates inclusion of startup and queue time in the deadline; the
production 900-second value is covered by the workflow regression, not a
900-second paid drill. Hosted queue delay and permanent provider deletion
failures remain limitations.

A dependency-preloaded image was built and validated locally with the pinned
runner, actual NVRTC 12.6/12.8 JIT header compilation, and cuTENSOR loading.
Publication requires the shared human-only registry handoff. No image switch
is included here: a larger cold pull can regress paid time, so promotion needs
three complete fresh A40 baseline/candidate pairs, at least 10% median paid-time
reduction, no paired regression, and within-arm max/min <=1.5, with the full
numerical workload and confirmed deletion in every run.

The low-priority test audit reused all three successful candidate-run logs.
The slowest individual test had a 9.117-second median, 1.96% of median paid
time, below the 5% need threshold. Individual test optimization is deferred;
shared initialization/JIT cost was not isolated by that audit. No numerical
assertions, tolerances, or workload entries are removed.


Post-repair production validation at main `3ede60cc30e4b48fa6cef776d28c3d93aa7e48a0`
(run 37860294000) passed all 285 CUDA cases, three PJRT cases, and the tutorial,
with loaded NVRTC 12.8. The first RTX 2000 Ada pod registered in 76 seconds;
cleanup confirmed HTTP 204 deletion. The measured paid window was 409.115
seconds at $0.24/hour (estimated $0.0272743). Runtime setup was 107 seconds.
This validates the integrated production controller and both bootstrap repairs;
it is not a same-hardware comparison with the A40 confirmation campaign.

The custom-image publication route is held locally. An alternative experiment
uses the existing public NVIDIA CUDA 12.8.1 runtime image, reusing only files
that match the frozen SDK byte for byte and transferring every other file.
The assembled SDK passed real NVRTC/CubeCL JIT-header compilation locally.
The 12.8 SDK compressed transfer is 523,422,490 bytes rather than
1,074,446,861 bytes; this is not yet a paid-time result. Both image arms use
the same pruned common payload, test archives, and exact remaining SDK bytes.
This experiment is restricted to the 12.8 host tier; production's existing
12.6 floor must remain supported before any image change can be promoted.
