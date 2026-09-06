# Einsum component probe slice

Parent #1758; measurement infrastructure for #1760 and benchmark #95.
Baseline `0457a2ed0aeea21b14f4297f7f4731e09b3a0507`.
Design: [einsum-component-probes](../design/einsum-component-probes.md).

## Behavioral optimizer bypass evidence

#1761 requires an actual counter/seam, not source matching. Test-only counters
now sit at both general optimizer entries (omeco and self-greedy). The public
optimize and optimize_with_options calls, plus ConcreteEinsumPlan::prepare for a
binary contraction, leave both at zero. N-ary planning and an explicit fallback call are positive controls that
the counters are connected. Thread-local scalar counters isolate parallel tests;
no plan, tensor, validation fact or other reusable execution state is cached.
The instrumentation is absent from non-test builds.

A mutation check temporarily removed only the binary shortcut. The new test
failed with actual counts (1,0), expected (0,0). After restoring the shortcut,
all184 library tests passed with one ignored probe in debug, release and
coverage-instrumented profiles. Strict crate Clippy and rustfmt passed;
planning/tree.rs coverage is414/439 lines (94.31%). After adding the explicit
ConcreteEinsumPlan assertion, focused debug/release tests and strict Clippy passed
again; the unchanged source paths retain the full-suite and coverage evidence.
No timed probe was run.

The first release attempt timed out during a cold-dependency build. Its surviving
owned rustc (launched by the existing shared wrapper) was inspected and allowed
to finish; the shared service was not stopped. The successful retry used a longer
observed-build budget, eight jobs and a task-local empty RUSTC_WRAPPER override
so the attempt did not depend on that wrapper. No global setting was changed.

The counters do affect lib-test artifacts. Future component comparisons must
apply identical instrumentation to both revisions and obtain fresh receipts;
old frozen baseline/candidate artifacts and measurements are unchanged. Existing
performance data is not silently reattributed to this new test binary.

## Direct-work diagnostic and shared-owner candidate

Historical review/delegation notes below are not current workflow gates.
The integrated probe at `e120bbf` and benchmark harness `acf631f` completed all
30 stages in the CPU devcontainer with verified release provenance. The frozen
precision protocol used 5 independent processes, 7 measured samples per process,
3 warmups, a 10ms minimum aggregate, and CoV ≤ 0.10. Earlier incomplete pilots
remain archived and are not accepted comparisons. This is a component diagnostic,
not end-to-end baseline/candidate acceptance.

Combined binary preparation dominates parse/input-metadata costs, and explicit
pair construction avoids much of its cost. The selected candidate therefore
bypasses ordering search for two operands in the shared
`ContractionTree::optimize_with_options` owner. It keeps `options.validate()` and
uses existing `from_pairs` shape validation and step-plan construction. It does
not specialize rank, dtype, operation spelling or execution surface, add a cache,
or weaken prepared execution checks. Three-or-more-operand optimization is unchanged.

New label/shape-equivalence and rejection tests passed on the original path before
the shortcut. With the candidate, all 183 einsum unit tests passed, one probe was
ignored, and strict crate/test Clippy and rustfmt passed. Release/paired performance
validation, full public matrix coverage and final PR gates remain outstanding.

## Pre-implementation review

DeepSeek V4 Flash, design round1: **Correct-to-merge**, before implementation.
Reviewed concrete private preparation/spec/revalidation seams, planner, syntax,
module wiring and existing CountingAllocator patterns. Non-blocking guidance is
incorporated: environment configuration, panicking-closure guard test and complete
einsum unit-suite forwarding-transparency check for the test-only global allocator.
Implementation assigned to Luna; no implementation existed at design approval.

## Narrow rework

The test-only probe now requires `TENFERRO_PROBE_STAGE` for timed and
allocation invocations, filters contract output by `TENFERRO_PROBE_CASE`, and
rejects zero allocation iterations. Timing/allocation loops black-box the real
`Result`, check expected values or typed shape-mismatch outcomes, and emit a raw
invalid record (including under-duration) before returning failure. The fixed
pair probe uses preconstructed spec shapes with stack references rather than a
per-iteration `Vec`. Revalidation contracts explicitly name count, dtype, and
shape error categories. The allocation guard uses nonallocating TLS `try_with`,
rejects nesting before reset, and restores disabled state during unwind.
Correctness mode executes each fixed fixture and the alternating fixture through
the ordinary CPU einsum path, checking exact shape, dtype, and values.

## Parent verification and final contract corrections

Three invalid revalidation cases are now independently selectable:
`rank2-binary-f64-count-invalid`, `rank2-binary-f64-dtype-invalid`, and
`rank2-binary-f64-shape-invalid`. Their actual metadata and expected typed errors
match their exports; they are not merely labels for unit-only checks. Parent ran
the ignored correctness entry point for each and parsed its emitted JSON strictly.
All three passed without timing samples. Luna corrected three redundant match
guards; the exact crate clippy command with `-D warnings` then passed.

The branch was fast-forwarded to merged inventory revision
`6858475b7bd8156e8e78abe55c2a8958d6deca21`, preserving the probe edits; production
Rust source in that inventory merge was unchanged. A separately bounded 300-second
release build timed out (exit 124); parent found no remaining owned compilers.
A narrower release build of `tenferro-internal-cpu-kernels` alone also timed out
at 600 seconds (exit 124), with no remaining owned compilers. Further identical
retries are not justified on this host. Release readiness is not claimed. Earlier
allocation smoke output is not an accepted performance baseline.

## Full-diff review

DeepSeek V4 Flash reviewed all four changed files in snapshot
`48c6775ffedb89ec25c74044d68f1d89dd290ab91d556d1fc914b040caee2f1a`
(SHA-256 of the complete staged diff): source-code **Correct-to-merge**, with no
Critical/Important findings. Three non-blocking notes concerned unreachable partial
TLS initialization failure, an unreachable fixture dtype panic, and the allocation
iteration default. Allocation mode defaults to one iteration when unspecified;
this is a diagnostic convenience, not a calibrated performance result.
Parent then reran the full 181-unit-test suite and exact clippy command successfully.
This code verdict is not PR readiness: release smoke, local coverage/docs/rules
gates, integration, hosted CI and accepted measurement evidence remain pending.

## Constraints and remaining gates

No new public API/dependency, no production optimizer/cache change, no performance
measurement on the busy host. Combined preparation is not pure validation; caller
allocation accounting excludes worker/provider/native allocations. This slice does
not close #1760, #95 or #1758. Full-diff Flash review, local tests/lint/docs/gates,
source/contract integration, hosted CI and valid raw baseline evidence remain required.

## Verification

- `cargo test -p tenferro-einsum concrete::probes --lib`: 9 passed, 1 ignored.
- `cargo test -p tenferro-einsum --lib`: 181 passed, 1 ignored.
- Correctness-only ignored entrypoint passed for all five cases, including exact
  fixed and alternating fixture outputs; allocation mode passed a two-iteration
  alternating revalidation sample. Contract output was parsed with Python's
  standard-library JSON parser and case filtering returned only the selected case.
- `python3 scripts/ci/run_profile.py fmt` passed; `git diff --check` passed.
- Two 45-second release test-binary compile attempts timed out while compiling
  `tenferro-internal-cpu-kernels`; the owned rustc processes were terminated after
  inspection. No timed probe mode was run. Release smoke remains pending.
