# Crate-owned einsum component probes

Status: DeepSeek V4 Flash design round 1 Correct-to-merge; probe slice implemented, full-diff review pending.
Parent: #1758. Measurement infrastructure for #1760 and benchmark #95, not closure
of either issue. Baseline: `0457a2ed0aeea21b14f4297f7f4731e09b3a0507`.

## Outcome and boundaries

Add a small test-only component probe inside tenferro-einsum that invokes existing
private production helpers, without new public APIs, dependencies, production
instrumentation, optimizer changes or copied library implementation. This is the
first independently mergeable probe slice; tensor/CPU/session attribution and the
ordinary benchmark matrix remain required follow-up work, not claimed here.

Known owners: concrete.rs `input_specs`, `prepare_subscripts_internal`,
`ConcreteEinsumPlan::validate_inputs`; eager.rs `plan_subscripts`; planning/tree.rs
`ContractionTree::optimize_with_options` and `from_pairs`. The first collects owned
input dtype/shape metadata; validation of a compatible prepared spec is distinct
from full execution validation. Planning includes validation and step preparation.
Never label planning as pure validation or integer-label preparation as parsing.

## Test-only entry points

Use a module under `src/concrete/` referenced with `#[cfg(test)] mod ...` from
concrete.rs, so private input_specs and validate_inputs remain private. Do not
expose benchmark-only helpers or add a feature flag. Add ordinary correctness
unit tests and a separately ignored measurement entry point. Default unit tests
must never collect/report performance timings. The benchmark-owned driver invokes
the exact ignored entry point in an isolated release test process with one test
thread. Configure named case/mode and sampling via documented `TENFERRO_PROBE_*`
environment variables. Timed and allocation invocations require
`TENFERRO_PROBE_STAGE` and execute exactly one stage; `contract` accepts a
specific case or `all`. Record the exact binary/source identity and parse
explicitly prefixed machine-readable records. Do not create another standalone
suite orchestrator.

Stages:

1. `parse`: existing notation/subscript parser; preparation is excluded.
2. `input_metadata`: input_specs on preconstructed tensors; shape/dtype collection
   and destruction included, no validation claim.
3. `prepared_spec_revalidation`: validate_inputs with preconstructed expected and
   actual specs. Count/dtype/shape contract only; read/layout/storage/backend checks
   belong to execution and are explicitly excluded.
4. `prepare_combined`: existing prepare_subscripts_internal; includes shape-reference
   collection, input validation inside planner, contraction search and pair steps.
5. `fixed_pair_prepare_combined`: existing from_pairs for the two-operand control;
   validation and step compilation included; not a numerical candidate optimization.
6. `empty_control`: same loop/output-control scaffolding without component work.

No independent-median subtraction is an exact attribution. Further pure validation
and output-metadata probes must be added at their actual owning seams under #1760;
this slice does not pretend the combined prepare stage satisfies both separately.

## Bounded case contracts

Use rank2 as the baseline, selected rank4/8 cases, and operand-count1/2/4/8 for
applicable stages. Include F64 and C64 input metadata, fixed and alternating metadata,
and representative invalid count/shape/dtype revalidation. Tensor payloads are tiny
and preconstructed; shape-changing fixtures must not allocate huge dense buffers.
Parse/plan shape-only cases may vary extents to demonstrate that no payload is read.
Every nonempty control case has expected metadata, error category or plan structure
assertions; accompanying numerical checks outside timing compare ordinary execution
with known nontrivial values. Do not use rank-positive/finite-only smoke assertions.

Provide a deterministic case-contract export from the test-only probe (no timings),
with stable IDs, stage/phase, dtype/rank/input-count, fixed/changing metadata,
setup inclusions/exclusions, calls per workflow and accounting scope. Align this
export with the #1759/#95 contracts during integration; unintegrated IDs remain
explicitly pending. No duplicated operation registry is added.

## Timing and allocation diagnostics

One invocation handles a named case/mode; orchestration, independent repetitions,
idle-core selection, pinning and result directories stay in benchmark #95. The
probe refuses timed mode unless explicit configuration supplies iteration/sample
counts and minimum aggregate duration. Use std::time::Instant and black_box on
value-dependent results/metadata. Emit raw elapsed ns and iteration/sample IDs;
verify every aggregate meets its configured minimum. Calibration policy belongs
to the driver; under-duration output is invalid, never silently accepted.

Allocation mode is a separate invocation from timing. Reuse the established
CountingAllocator pattern from CPU/tensor allocation tests with a test-only System
forwarder. Account caller-thread allocation calls/requested bytes explicitly;
worker/provider/native allocations are NOT covered and must not be inferred.
The forwarding allocator and const-initialized thread-local counters must not
allocate recursively; disabled counting must be the default and a guard must restore
state on unwind. Forward alloc/alloc_zeroed/realloc/dealloc correctly with SAFETY
comments. Requested-byte traffic is not retained memory. No allocation diagnostics
are collected while measuring latency. No new dependency or helper crate is needed.

## Verification and acceptance for this probe slice

- Real helper calls, deterministic expected metadata/typed errors and numerical
  sanity checks; no helper copies, production changes or public API additions.
- Tests for zero/invalid configuration, known allocating/nonallocating controls,
  counter disable/unwind behavior (including a panicking closure), stage contract/export
  and under-duration rejection. The allocator covers the whole unit-test binary;
  rerun the default einsum unit suite to verify disabled forwarding transparency.
- Ordinary tests do not emit timing evidence; ignored measurement mode is explicit.
- Focused unit tests and release test-binary compile/correctness-only smoke pass;
  inspect that normal builds contain no test-only allocator/probe entry points.
- Repository formatting, focused coverage review/attestation, relevant doctests,
  CI-parity lint, `scripts/check-pr-fast.sh` and committed-head deterministic rules
  review pass, followed by Flash full-diff approval and required hosted CI.
- Record exact design/diff reviews and commands in docs/worklogs. Raw measured
  baseline evidence remains pending until #95 can run on observed idle resources.
  Do not run timed mode on the currently contended host.

## Rejected alternatives

A public prepare-stage helper solely for benchmarking expands unsupported surface.
A benchmark-owned copy of validation mismeasures real behavior. Adding production
caches/fast paths before attribution violates the parent need gate. A new generic
benchmark framework or dependency is unnecessary for six crate-owned probe stages.
