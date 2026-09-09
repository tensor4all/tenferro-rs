# CPU binary and FFT performance batch (#1765, #1766)

## Scope and decisions

Baseline: `a096f280ab0dd774f92f533aa0f38b663a4c94d3`.
One implementation PR, with coherent commits and non-squash merge.
The maintainer approved binary copy avoidance, FFT pooled output and lane
parallelism, ordinary eager-drop recycling, and consuming eager CPU c2c
in-place execution. The maintainer subsequently approved changes to host
storage reclamation and CPU pool synchronization when source inspection showed
that normal drop cannot currently return host storage to the pool.
The maintainer also explicitly approved retaining the managed/Apple shared
output copy boundary in this PR, while applying lane parallelism and staging
reuse. Direct managed writes, their synchronization contract, and Apple hardware
validation belong to a separate improvement.

Classification: #1766 is an existing-contract fix; #1765 lane parallelism and
output reuse are accepted performance work; consuming eager in-place and host
reclamation are explicitly approved contract changes. None is completed yet.

## Source evidence and implementation direction

- Runtime `tensor.rs` duplicates both same-shape inputs; `typed_tensor.rs`
  already has borrowed/owned input preparation. Share that implementation.
- FFT `cpu.rs` builds a fresh output Vec and visits lanes sequentially.
  Write the final pooled destination directly, retain full-overwrite safety,
  and execute lane jobs under the session's existing CPU context.
- `HostAllocation` owns storage but no recycler. `CpuEngine.resources` holds
  the pool under the execution-resource lock. A destructor must not reacquire
  that lock. Separate short-lived pool-state synchronization from execution
  admission; keep the existing bounded pool, not a second cache or queue.
- A host root may carry a weak typed recycler. Final destruction returns the
  allocation only if its original pool still exists. Explicit extraction must
  disarm recycling, preventing duplicate return. Retention remains owned by
  AllocationGroup, so AD/capture/aliases keep the allocation alive normally.
  Exact integration and error handling are still under investigation.
- Existing `EagerTensor::into_value` checks record/container ownership and
  extracts through AllocationGroup. Reuse it for consuming in-place c2c;
  additionally reject grad/capture participation. Do not infer write authority
  from root-reference counts. Preserve errors rather than silently copying.

## Verification protocol (declared before candidate measurements)

Use release, default cpu-faer, explicit backend thread counts, value-dependent
black_box inputs/outputs, and record effective thread counts and CPU affinity.
Baseline and candidate run as paired complete suites, five repetitions after
warmup; report medians and all samples plus bootstrap 95% intervals. No selective
case retries. Record hardware, load and affinity before/after. A repetition
spread exceeding 20% of the median or changed CPU affinity makes the complete
paired timing experiment inconclusive; rerun the full pair, not selected cases.

Cases: binary public versus raw session mul at C64 lengths 8, 32768 and 128^3
with 1T; FFT C64 three-axis transforms at 8^3, 32^3, 128^3 with 1T and 8T, with
per-axis timings. Separate normal eager result drop, explicit reclamation and
consuming in-place. Separate cold allocation from warmed reuse; allocation
counts/bytes and pointer reuse are checked outside timed runs, including outputs
kept alive. Include F32/F64 and C32/C64 correctness, norms, padding/truncation,
strided-axis traversal, AD/capture/alias and failure cases in focused tests.

Primary gates: eliminate the two input-sized copies for same-shape binary ops;
large 1T public mul within 10% of raw session mul; warm 128^3 8T FFT at least
1.5x faster than baseline 8T and at least 1.5x faster than candidate 1T.
Non-regression gate: no median slowdown over 10% for larger ordinary cases;
small cases allow 2 microseconds absolute overhead. Normal drop must demonstrate
actual original-pool reuse, not just successful pool allocation. In-place must
preserve the input allocation. Timings do not substitute for correctness.

Run focused tests and the required local check-pr-fast gate, deterministic
repository-rules review, changed-file coverage review, documentation checks and
release evidence before PR creation; hosted CI and approvals gate merging.

## Progress

Read workspace and repository rules, shared common/Rust rules, contribution
workflows, and the relevant runtime/FFT/eager/storage/pool entry points.
Implemented (not yet PR-ready): shared binary borrowed preparation; weak
host-root recycling and a short independently synchronized existing pool;
recycled uninitialized output handoff; host FFT pooled direct output and scoped
Rayon lanes with per-job scratch. Managed FFT now shares the same kernel and
recycles its staging allocation, but retains its copy-only provider boundary.
A direct mapping attempt failed the managed regression test: HostWriteGuard
exposes only copy_from_slice, and the provider root has no writable mapping.
Its allocation contract also allows uninitialized values, so exposing it as
initialized &mut[T] is not a valid shortcut. The design now records the approved
scope: retain this managed copy boundary.

Consuming eager fft_in_place/ifft_in_place and compact borrowed FFT inputs are
implemented. In-place uses the normal targeted extension registration/ingress
validation, rejects active recording, structurally extracts the original owner,
and returns unchanged rejected inputs (boxed on the error path). Execution
errors after ownership acquisition consume the input. Four existing autodiff
parity tests used the removed Tensor::clone API; they now explicitly duplicate
their independent reference inputs.

Fresh focused evidence after these changes: FFT integration targets passed
40 tests (2 CPU reuse, 1 eager allocation, 3 consuming in-place, 34 existing FFT
checks), and FFT autodiff all-target clippy passed with -D warnings. The eager
allocation test verifies zero input/output-sized allocations after warmup for
normal drop/reuse and consuming FFT. Internal CPU kernels passed 59 unit tests
and 25 doctests; tensor storage passed 43 focused tests. FFT doctests previously
passed 25 tests and need their affected final-state rerun. Complete paired
timing, remaining safety/coverage checks, documentation,
local PR gates and merge remain unfinished.

Passed focused checks:
- runtime `binary_allocations`: 1 test; restoring the two input duplicates
  intentionally failed this test (exit 101), and restoring the fix passed again.
- runtime integration `session_ops`: 24 tests.
- internal CPU kernels `buffer_pool`: passed.
- internal CPU kernels `pooled_uninit_output`: 6 tests, including final group
  ownership, drop during another checkout, explicit extraction, owner-first
  drop, limit changes and clear.
- FFT library: 18 tests.
- FFT integration `cpu_reuse`: 2 tests, proving same-session drop/reuse without
  reusing live results and exact 1T/8T agreement across every axis, all norms,
  padding and truncation.
These are partial checks, not the full local gate or completion evidence.

Baseline probe is `/tmp/tenferro-1765-1766-probe`, with preserved baseline binary
`/tmp/tenferro-1765-1766-baseline` and raw log
`/tmp/tenferro-1765-1766-baseline.log`. Host: Linux primerose, EPYC 7713P,
affinity 0-63, effective configured threads asserted as 1 and 8. Baseline only:
large C64 mul public ~62 ms versus raw ~22 ms at 1T. Small-case spread exceeds
the declared 20% validity gate, so these samples are diagnostic, not an accepted
paired performance result. The final complete pair remains required. The first
release build timed out; inspected the owned descendants, waited for its
compiler to terminate, then completed the build with RUSTC_WRAPPER unset and
-j4. No unrelated processes were terminated. External issue benchmarks remain
reported evidence, not reproduced ratios or copied implementations.

Candidate diagnostic probe completed in release mode. Its lockfile change only
adds the already-resolved rayon dependency to tenferro-fft. Large 1T public/raw
mul medians were 21.30/20.94 ms; 128^3 three-axis FFT medians were 42.75 ms (1T)
and 9.41 ms (8T). These are NOT accepted speedups: the full diagnostic experiment
is INCONCLUSIVE, with several spreads above 20% and small 8T FFT overhead
requiring further examination. Raw candidate samples are preserved in
/tmp/tenferro-1765-1766-candidate-diagnostic.log.

Before the next complete experiment, amend the inadequate single-call warmup
and five-call measurement windows: every case now warms for at least 100 ms
and measures whole batches for at least 50 ms. Rebuild BOTH baseline and
candidate from this identical probe source; do not compare the new timings
against old diagnostic samples. Keep all cases, five repetitions, confidence
intervals, primary/non-regression thresholds and the 20% spread validity gate
unchanged. Pin both processes to CPUs 0-7, separately asserting backend 1T/8T;
record environment and source identity alongside the complete pair. This is
measurement stabilization, not a relaxation of the acceptance criteria.

The complete stabilized baseline/candidate pair ran on CPUs 0-7, with identical
probe SHA256 54a3b2265bc8da4ceebb9835a921938c2ab26352de43874120b52759de26323f.
Baseline source is the detached original commit in /tmp/tenferro-1765-1766-baseline-src;
candidate is the current uncommitted implementation. Raw samples, environment,
and all case medians/spreads/bootstrap intervals are retained at
/tmp/tenferro-1765-1766-{baseline-stable.log,candidate-stable.log,stable-environment.log,stable-summary.json}.
Status remains INCONCLUSIVE: 32^3 axis-2 candidate spreads are 36.6% at 1T and
30.0% at 8T, violating the unchanged 20% full-experiment validity gate. Do not
promote favorable large-case ratios as accepted improvements. Also investigate
observed 1T 32^3 axes 0/1 median regressions (15.6%/12.8%) and 8^3 three-axis
1T overhead (2.096 us), which exceed the stated non-regression allowances.
Final-state FFT library tests passed 20 cases and FFT doctests passed 25 cases.
No PR or merge has been performed.

A second COMPLETE pair used unchanged binaries, probe, affinity and thresholds;
/tmp/tenferro-1765-1766-pair2-{baseline.log,candidate.log,environment.log,summary.json}
retain all samples and bootstrap intervals. This pair is also INCONCLUSIVE:
32^3 axis-2 spreads exceed 20% again. The 1T 32^3 axis-1 median regression also
recurs (95.059 -> 106.412 us, +11.9%). Small three-axis overhead is now within
its allowance (10.934 -> 12.519 us). Do not repeatedly rerun unchanged timing
in search of a pass. Inspect strided gather/scatter locality and pool-reused
buffer alignment as hypotheses; neither cache conflict nor host noise is yet
established as the cause. The next change must address a demonstrated kernel
cost and rerun correctness plus the complete performance pair.

Kernel follow-up: perf hardware counters are unavailable (perf_event_paranoid=4);
no shared system settings were changed. Comparing original source identified
identity normalization that the rewrite unnecessarily multiplied into every
scatter element. Restored the original scale!=1 guard and contiguous scratch
scaling. Full pair3 remained INCONCLUSIVE with the axis-1 regression unresolved.
Then removed per-lane runtime division/remainder by advancing lane bases within
validated blocks (division only at job entry). Full pair4 has no observed median
non-regression failures: 1T 32^3 axes 0/1 are 81.620->79.376 and 96.185->90.093 us.
However the WHOLE pair remains INCONCLUSIVE because 1T 32^3 axis-2 spread is
26.6%; do not declare the performance gate passed. All pair3/pair4 samples,
bootstrap intervals and environment are /tmp/tenferro-1765-1766-pair{3,4}-*;
the reusable analysis script is /tmp/tenferro-1765-1766-analyze.py.

Added an empty-job guard before initial lane-base computation and a numerical
9-lane/8-job test for partial and empty jobs. Latest FFT integration checks pass
41 tests and autodiff all-target clippy passes with -D warnings. Pair4 predates
this final empty-job guard, so final-state performance evidence remains due.

Public error-documentation audit found two insufficient # Errors descriptions
(execute_fft_read and assume_init_recycled); both now name the actual typed
failure categories. The focused audit over eight affected API/owner files
passes, and the FFT library passes 20 tests on the latest lane implementation.
Added a downstream backward regression: an untracked pooled FFT coefficient is
used by a tracked multiplication, then consumed by in-place FFT (or returned
unchanged if retained), and dropped before backward. The gradient still equals
the original coefficient. This focused saved-value test passes; it does not
substitute for the other required gates.

Added and passed focused nested FFT admission/recovery and partial-write failure
tests. Existing CPU policy rejects a nested execution scope (rather than
silently starting another pool); FFT preserves the existing panic boundary and
the backend remains usable afterwards. A partially initialized pooled output
is discarded on both ordinary error return and unwind, and its replacement
accounting is cancelled before the pool is reused.

Ran the shared CI clippy profile for the root workspace plus standalone tropical,
sparse and TBLIS manifests. It first found an autodiff-only MaybeUninit import
that was unconditional; feature-gated that import. The complete clippy profile
then passed. This is broader lint evidence, not completion of check-pr-fast,
coverage review, final performance, or hosted CI.

Full formatting profile passed. Worktree deterministic rules review identified
missing method examples for execute_fft_read and HostBufferRecycler::recycle.
Added executable examples; both focused doctests pass (the borrowed method is
fft_read, corrected after the initial example compile check). The deterministic
worktree review was rerun after correction. A committed-head review is still
required before PR creation; no independent LLM review was requested or run.

Expanded the isolated eager allocation regression to cold execution, a second
output while the first remains live, ordinary drop reuse, explicit reclaim and
consuming in-place. All checks pass with effective backend threads=1 and C64
shape 64x64 (65536-byte payload). The allocator also reports all allocation/
reallocation requests, not only payload-sized ones; these are debug-build
requested-byte totals, not retained bytes or release timing:

| Path | Payload-sized count / bytes | Total count / requested bytes |
|---|---:|---:|
| Cold eager FFT | 1 / 65536 | 83 / 102805 |
| Eager FFT with previous result live | 1 / 65536 | 38 / 79701 |
| Ordinary drop reuse | 0 / 0 | 38 / 14357 |
| Second ordinary drop reuse | 0 / 0 | 38 / 14357 |
| Explicit extraction and reclaim | 0 / 0 | 0 / 0 |
| FFT after explicit reclaim | 0 / 0 | 38 / 14357 |
| Consuming in-place FFT | 0 / 0 | 21 / 6473 |

The test checks distinct live output pointers and preserves the pointer across
ordinary return, explicit return, and consuming mutation. Metadata allocations
remain: this is not a claim of zero total allocations. Raw output is
/tmp/tenferro-1765-1766-allocation-paths.log.

The complete FFT package with autodiff passes. Ran focused package coverage with
cargo llvm-cov --lib --tests: report /tmp/tenferro-1765-1766-fft-coverage.json.
Initial line coverage includes lanes.rs 79.29%, eager_in_place.rs 81.08%, cpu.rs
78.49%, and lib.rs 88.16%; these are NOT a passed 90% per-file coverage gate.
Added private kernel boundary tests for mismatched input/output spans, checked
shape overflow and zero lanes, plus a retained-reshape-source mutation test.
Both focused tests and the rerun instrumented package pass. The numerical and
buffer-error paths execute, but exported line summaries remain below target;
coverage interpretation and remaining reachable ownership-failure branches
still require review rather than an unsupported coverage acknowledgement.

Inspected llvm-cov's annotated text and re-exported the same merged profile:
lanes.rs shows positive execution counts for every substantive branch, including
error exits and empty jobs, while its exported line summary still reports
79.29% (47/70 generic instantiations covered). The discrepancy is unresolved;
no threshold, exclusion, or coverage checker was changed. Annotated output is
/tmp/tenferro-1765-1766-coverage-text and refreshed JSON is
/tmp/tenferro-1765-1766-fft-coverage-refreshed.json.
Added meaningful coverage independent of that discrepancy: compact F32/F64
borrowed reads, full and one-sided spectra, C32/C64 borrowed inverse reads,
all axes and norms with asserted 1T/8T backends. Roundtrip maximum-error checks
pass with tolerances 1e-4 (f32) and 1e-10 (f64); log is
/tmp/tenferro-1765-1766-real-parallel-tests.log.

Further inspection identified unexecuted monomorphizations, including the
private boundary test's job closure (its original cases only returned early).
Added a valid-span control case and reran the instrumented package including
real-read tests. All tests pass; lib.rs coverage is now 90.97%, lanes.rs 81.43%,
while cpu.rs remains 78.49% and eager_in_place.rs 81.08%. Remaining file-level
coverage deficits are still open. This evidence does not establish a coverage
tool bug and no such claim or exemption is used.

Added an analytical impulse/DC reference test through the same generic kernel
entry for real, complex, consuming in-place and Hermitian inverse inputs, with
all three normalization conventions. Tests and instrumented package pass;
lanes.rs now measures 131/140 lines (93.57%) without changing coverage tooling
or thresholds. Other file deficits remain open.
Also added managed-provider failure regressions for allocation failure, wrong
dtype, foreign output domain and wrong output length. Each produces the typed
error and the original input remains numerically usable on a valid domain-bound
backend. Focused test passes; /tmp/tenferro-1765-1766-managed-errors.log records
it. This retains the approved managed copy boundary.

Separate alignment diagnostics (not the frozen comparison probe) show a strong
within-run association at 1T 32^3 axis 2: the same input and output offset +16
modulo 64 KiB produced 363/365 us; +32 produced 316/316 us; +48 produced 287 us.
Raw addresses and all cases are /tmp/tenferro-1765-1766-alignment.log. This
supports investigating memory locality, not labeling all variation host noise.
Before the next candidate comparison, group at most four adjacent strided lanes
in the existing per-job scratch, process their complete transforms together via
RustFFT, then scatter. Never cross a stride block or job boundary; gather the
whole group before any in-place writes. Contiguous lanes remain single. No
allocator padding, extra pool, threshold relaxation, or selected-case retry.
The full frozen baseline/candidate suite and correctness checks remain required.

The grouped-lane candidate passes the full FFT/autodiff package. Complete pair5
is INCONCLUSIVE: many spreads exceed 20%, larger raw mul regresses despite no
change to that kernel, and 1T 32^3 axes 0/1 exceed 10%. All raw cases and CIs are
/tmp/tenferro-1765-1766-pair5-*. No favorable subset is promoted.
A subsequent /proc/stat sample found CPUs 1,3,6 at 100% busy without our probe;
a second sample confirmed it. No other processes or shared settings were
changed. Before another COMPLETE pair, move BOTH executables to CPUs 40-47
(the sampled maximum utilization there was 0.7%). Keep the frozen probe,
repetitions, primary/non-regression limits and spread gate unchanged. Record
this affinity change as a new experiment; do not compare times across CPU sets.

Complete pair6 on CPUs 40-47 still fails non-regression for 1T 32^3 axes 0/1
and is INCONCLUSIVE due to an 8T 128^3 axis-0 spread above 20%. A bounded
follow-up replaced tile chunk iterators with checked-by-construction slices;
complete pair7 still fails those two serial cases (90.506->139.731 us and
102.829->149.143 us), with an additional spread failure at 8T 128^3 axis 1.
Both complete experiments, including unfavorable cases and CIs, remain in
/tmp/tenferro-1765-1766-pair{6,7}-*. The four-lane experiment is REJECTED, not
promoted: removed the tiling changes and restored the simpler incremental
single-lane kernel. Its full FFT/autodiff package passes again; see
/tmp/tenferro-1765-1766-restored-tests.log. Keep the analytical/boundary tests.
Alignment dependence remains a supported diagnostic observation, not a proven
microarchitectural cause or a solved performance gate.

Complete pair8 measures the RESTORED single-lane candidate on CPUs 40-47, not
the rejected tiled candidate. Larger serial 32^3 axes 0/1 again meet the
non-regression bound. The complete experiment remains INCONCLUSIVE: 1T 32^3
axis-2 spread is 20.1%, above the unchanged 20% threshold, and small 8T
three-axis FFT overhead is 2.286 us, above the 2 us allowance. No rounding,
case exclusion or mixing with tiled samples is used. All cases/CIs/environment
are /tmp/tenferro-1765-1766-pair8-*. Final performance acceptance is still open.

Ownership self-review traced production completion and both reclamation routes:
PooledUninitOutput::finish -> host root attachment -> final HostAllocation::drop
for automatic return, versus CpuBackend reclaim_typed -> into_host_vec (disarm)
-> pool_release for explicit return. Pool bins contain scalar Vecs, not tensor
owners, and the root holds only a Weak recycler, so the reviewed path has no
ownership cycle or second destructor return after extraction. This is a scoped
review, not the final whole-diff signoff. Added and passed a public integration
test transferring explicit reclamation to a different backend: the origin pool
stays empty, the destination reuses the exact pointer, and later drop returns
only to the destination. Log: /tmp/tenferro-1765-1766-cross-pool.log.
Runtime integration suite also passes 146 tests with four test workers; log:
/tmp/tenferro-1765-1766-runtime-integration.log.

Updated docs/guides/tenferro-fft.md to distinguish borrowed/non-destructive
operations from consuming host-only FFT, document final-owner pool return and
error ownership, and explicitly state that managed shared residency still uses
a CPU staging copy. Shipped compute-skill mirrors contain no FFT mutation advice
that conflicts with the addition, so no redundant mirror edits were made.
Doc snippet synchronization and git diff --check pass. The guide dependency
smoke build first hit an undersized 30-second deadline; no owned descendants
remained. Reran with RUSTC_WRAPPER unset, four build jobs and a 600-second
budget; the guide dependency smoke checks pass. Logs are
/tmp/tenferro-1765-1766-{doc-snippets,guide-dependencies}.log.

Added and passed CPU FFT capability tests for a valid saved spec paired with
wrong-dtype and wrong-shape owned tensors, and a wrong-dtype compact borrowed
view. Existing validation returns Error::Validation and leaves the fresh pool
empty. No production validation branches or test-only production hooks were
added. Log: /tmp/tenferro-1765-1766-spec-mismatch.log.

Fresh focused coverage after managed/spec tests: cpu.rs 83.98%, lanes.rs 93.57%,
lib.rs 90.97%, eager_in_place.rs 81.08%; not a full coverage-gate pass.
A subsequent lazy-transpose regression exposed a real missing precondition:
into_value structurally extracts ownership but preserves noncompact layouts.
The consuming FFT previously accepted that view instead of rejecting it before
the compact lane kernel. The test failed before the fix. Added actual input
compactness validation before ownership extraction, documented the compact
host contract, and verified unchanged returned values plus successful borrowed
FFT on the rejected view. Full FFT/autodiff package and all-target FFT clippy
pass after the correction. Logs: /tmp/tenferro-1765-1766-layout-fix-{tests,clippy}.log.
This test covers layout rejection, not IntoValueError::Extract; the earlier
assumption that noncompact extraction itself must fail was incorrect.

The repository-local gate now passes on the layout-corrected worktree:
`RUSTC_WRAPPER= CARGO_BUILD_JOBS=4 bash scripts/check-pr-fast.sh
--coverage-reviewed --test 'cargo test -p tenferro-fft --features autodiff
--test eager_in_place'`. The gate fetched origin/main (still a096f280), ran
its formatting/docs checks, CI-parity clippy including standalone extensions,
and all six consuming-FFT integration tests. `git diff --check` also passes.
Full log: /tmp/tenferro-1765-1766-local-gate.log. The coverage acknowledgement
means reviewed, not threshold compliance. Performance acceptance, remaining
coverage, final integrated review and committed-head rules check, PR, hosted
CI/approval and non-squash merge remain outstanding.

After the layout correction, focused coverage is cpu.rs 84.44% and
eager_in_place.rs 82.50% (layout-coverage.json/log under the same /tmp prefix).
Investigated a concrete small-FFT overhead hypothesis: with_linalg_pool might
re-enter the CPU executor on every transform. Source tracing shows an already
entered managed session reuses its context and only sets the native policy.
A standalone diagnostic, pinned to CPU 40 with backend and effective native
thread counts both asserted to be 1, measured five 100ms-warmed/50ms batches:
15.27–17.01ns per empty native scope versus 0.62–0.68ns for the empty control.
This does not justify bypassing required CPU policy to address microsecond-scale
FFT overhead. It is a 1T diagnostic, not an 8T or full performance-gate result.
Source: /tmp/tenferro-1765-1766-probe/src/bin/scope.rs; build and results:
/tmp/tenferro-1765-1766-scope-build.log and /tmp/tenferro-1765-1766-scope.log.
No production policy changes or repeated full comparison were made.

Follow-up ownership review found that compactness does not imply zero offset
or a full-root extent. A compact eager slice reproduced Execution(buffer-length
validation failure) because host_data_mut exposed the root rather than the
logical descriptor. Replaced that borrow with the existing with_host_write
prepared guard in in_place_typed, preserving descriptor offset and bounds;
no new ownership API or copy was introduced. Regression tests cover prefix
and offset slices, stable logical data pointer, exact FFT output, and unchanged
root elements outside the selected slice. The pre-fix test fails; the full FFT
package including allocation tests and focused bounds test pass afterward,
as does all-target FFT clippy. Logs: /tmp/tenferro-1765-1766-slice-before.log,
slice-after.log, slice-bounds.log and slice-clippy.log (same prefix for all).
The earlier local gate predates this correction; final-state checks remain due.

Release-mode verification after the descriptor-bounded borrow change passes:
seven consuming-FFT tests plus the allocation regression (RUSTC_WRAPPER unset,
four build jobs). Log: /tmp/tenferro-1765-1766-slice-release.log.
Captured updated allocation totals from the optimized allocation executable:
cold 83 calls/102805 bytes; second simultaneously live output 38/79701;
ordinary-drop reuse 38/14357 on both measured repetitions; explicit reclamation
0/0; post-reclamation FFT 38/14357; consuming FFT 22/6505. Only cold and second
live outputs allocate a 65536-byte input-sized buffer; all reuse and consuming
measurements allocate zero input-sized buffers. The prepared guard adds one
small allocation (32 bytes) to the previously recorded consuming totals;
this is not zero-total-allocation FFT. The fixture asserts effective backend
1T and live-pointer separation/reuse. Log:
/tmp/tenferro-1765-1766-slice-release-allocations.log.

Scoped binary diff review: dynamic operands remain TensorRead::Tensor, so the
existing default backend *_read hooks delegate to owned-tensor methods without
introducing a new view requirement. The shared broadcast planner/error mapper
is unchanged; reshapes/broadcasts still retain their owned temporaries through
execution. Existing session_ops tests cover full dtype-error payloads,
binary/ternary broadcasting and values, typed results and single-session entry;
the earlier successful run remains applicable (no production binary changes).

Filled an evidence gap in binary_allocations: retain the original large-buffer
regression criterion, but also count/report total allocation calls and requested
bytes with logging outside the measured interval. The updated 1T test passes.
For 4096 F64 elements, raw mul is 11 calls/32944 bytes, public mul 12/32952;
both allocate one payload-sized buffer (the output). Add/sub match those totals;
public div/rem/pow/min/max are 10/32936 versus raw 11/32944. Compare is public
10/4264 versus raw 9/4256 with no F64-payload-sized allocations. Clamp/select
are each 14/33000 on both paths. These are requested allocation bytes, not live
or retained memory, and do not claim zero metadata allocation. Log:
/tmp/tenferro-1765-1766-binary-allocation-totals.log.

Created the first coherent local commit, b7b53232615a63767e6a35924978a26b32ef254d
(perf(runtime): borrow binary operands through shared read preparation), containing
only the runtime manifest, two runtime implementation files and allocation test.
It was reviewed against the existing typed path and default backend read hooks.
Tests cited above ran in the integrated worktree, not a clean binary-only checkout.
FFT/recycler/docs changes remain uncommitted. Nothing has been pushed, no PR has
been opened, and this commit is not a claim that performance acceptance is complete.

Verified b7b53232615a63767e6a35924978a26b32ef254d independently in the clean detached
worktree /tmp/tenferro-1765-1766-binary-verify, without any uncommitted FFT or
recycler changes. The allocation regression and all 24 session_ops integration
tests pass; reported allocation counts/bytes match the integrated worktree.
The untracked Cargo.lock was newly resolved (notably zerocopy 0.8.57 rather
than the integrated lock's 0.8.56); retained it in that worktree. This is an
independent correctness/allocation check, not a controlled performance pair.
Logs: /tmp/tenferro-1765-1766-binary-isolated.log and
/tmp/tenferro-1765-1766-binary-isolated-session.log. Git status is clean.

Committed the reviewed storage/pool slice as
a2fc6e9493738eb5b69d29ad64bb2d8af902f19f (feat(cpu): recycle initialized host
outputs on final owner drop). Reused the clean detached verification worktree
at /tmp/tenferro-1765-1766-binary-verify, advancing it from b7b5323 to a2fc6e9.
Without uncommitted FFT integration, internal-cpu-kernels passes 60 unit tests
and 25 doctests; the tensor `storage::` filter passes 36 tests (other integration
tests are filtered, not claimed as run). Logs:
/tmp/tenferro-1765-1766-recycler-isolated.log and
/tmp/tenferro-1765-1766-storage-isolated.log. Both commits remain local;
FFT integration and docs remain uncommitted, with performance/coverage and
final submitted-state validation still outstanding.

Committed FFT integration as 0edb0d93f9e0197a7c2e272175bff3658944cde3. Review
covered recording/ingress checks before extraction, preserved structural error
ownership, descriptor-bounded mutation, disjoint lane writes and joining before
borrow release, and policy-derived thread counts. Added runnable doctests for
the new hidden public preparation/thread-count helpers; both pass (logs:
/tmp/tenferro-1765-1766-preparation-doctest.log and thread-count-doctest.log,
with the same prefix). Also documented propagation of factory/install errors.
Latest focused coverage after slice handling remains insufficient: cpu.rs
84.32%, eager_in_place.rs 82.50%, backend.rs 68.85%, eager_ext.rs 85.00%; lanes
93.57% and lib 90.97%. Evidence: /tmp/tenferro-1765-1766-slice-coverage.json/log.

The deterministic committed-head repository-rules review passes for 0edb0d9,
with the expected external-LLM-skipped warning (per repository policy).
Logs: /tmp/tenferro-1765-1766-committed-rules.{log,json}. This does not resolve
performance/coverage or validate a future head containing the pending docs.
All three code commits remain local; guide/design/worklog changes are not yet
committed, and no PR/push/merge has occurred.
