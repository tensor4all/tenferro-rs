# Issue 1471 Explicit Transfers

## Session Summary

Implemented split PR 1 from issue
[#1471](https://github.com/tensor4all/tenferro-rs/issues/1471): synchronous
production execution now consumes explicit scheduled transfer nodes and retains
tensor values by execution location. The transfer-provider registry remains
keyed by source and destination storage class.

## Context Read

- Workspace and repository `AGENTS.md`, workspace `CODING_RULES.md`, and
  `REPOSITORY_RULES.md`.
- Issue #1471, including the split proposal in comment `5115212328`.
- Shared Tensor4all repository, Rust, performance, documentation, and test
  rules.
- Runtime preparation, schedule generation, prepared execution, transfer
  registration, and existing integration-test paths after refreshing the local
  CodeGraph index.

## Decisions Made

- Added a crate-private execution location identity containing engine ID,
  runtime event-domain ID, and storage class. Event-domain identity is required
  because CUDA and WebGPU can share a storage-class identifier while remaining
  distinct execution locations.
- Kept provider lookup keyed by `(source storage class, destination storage
  class)`. `TransferRequest` now exposes the concrete engine and event-domain
  endpoints in addition to those storage classes.
- Generated `ScheduledTransfer` nodes whenever an instruction needs a value at
  a location where no copy exists. Storage-class equality alone does not avoid
  a transfer.
- Added an explicit execution-instruction index to every scheduled operation.
  Production execution no longer equates schedule-node position with
  instruction position.
- Stored each execution slot as zero or more location-tagged copies. A transfer
  adds a destination copy without removing the source; global last use clears
  every retained copy.
- Preserved the existing executor ABI by staging only the copy for the current
  operation into a temporary slot vector, then returning live inputs and
  outputs to the location-aware store.
- Added a public typed `TransferError::MissingProvider` source while preserving
  the existing runtime-state error category.
- Removed inline `ensure_slot_in_storage` transfer execution and the stale
  module-wide `dead_code` allowance from `schedule.rs`.

## TDD

- RED: the schedule test failed because execution locations, explicit
  instruction indices, and value-bearing transfer nodes did not exist.
- RED: integration tests failed because transfer requests lacked engine/event
  endpoints and no typed missing-provider error existed.
- GREEN: schedule generation emits one same-storage/different-location transfer
  for a split-use program and reuses the retained source copy for the later
  source-side operation.
- GREEN: `Runtime::run_compiled` executes scheduled transfers for cross-storage
  and same-storage location changes.
- GREEN: a split-use graph performs one forward and one reverse transfer,
  proving that the first transfer does not destroy the source copy.
- GREEN: an intentional transfer failure skips the downstream operation and
  drops a tracked source-location backend value. A separate downstream
  extension family records an explicit zero execution count after the failed
  transfer.
- GREEN: identical transfer-provider registration is idempotent, while a
  different provider at the same class-pair key returns the typed conflict.

## Specification Review Follow-up

The initial specification review returned `NOT_APPROVED`. The follow-up:

- places an independently counted extension operation downstream of the
  intentional transfer failure and asserts its execution count remains zero;
- replaces the preparation-path `expect()` used to resolve execution locations
  with `PrepareError::ResolvedEngineUnavailable`, propagated through both
  same-storage and cross-storage dispatch construction; and
- adds runnable `# Examples` to all public transfer endpoint accessors and
  `TransferError`.

## W7 Code-Quality Review Follow-up

- Added an engine-owned input-placement validator that explicitly declares
  which `(tensor placement, storage class)` pairs may enter an execution
  engine. Preparation resolves and caches one validated execution location per
  program input instead of assigning every input to the first operation.
- Schedule generation initializes each input slot at its resolved ingress
  location. Production execution applies the same per-input locations and
  revalidates the actual runtime tensor placement before tagging a value.
- Added an end-to-end two-location test whose first operation runs on a
  different engine from input ingress. The test records the emitted transfer's
  concrete endpoints and proves the provider executes before the consumer.
- Added `TransferProviderContractError` and validates provider output dtype,
  shape, destination placement/storage compatibility, and backing-buffer
  length before retaining a destination copy. Contract failures remain
  available through the full `Error::source` chain.
- Added faulty-provider coverage for each contract dimension and confirmed the
  downstream operation does not execute for rejected output.
- Confirmed the `// INVARIANT` markers introduced by historical commit
  `06147e4e` remain at both deferred `schedule.rs` dead-code allowances. The
  commit itself is represented through the squash-merged #1514 tree rather
  than direct ancestry.

## W7 Ingress And Reachability Follow-up

- Replaced the placement-only execution ingress assumption with an explicit
  three-part registration contract: preparation placement eligibility, actual
  runtime-input acceptance, and destination-buffer residency.
- Made execution-bridge registration without the complete ingress contract a
  typed `RuntimeConfigError::MissingInputIngressValidator` failure at
  registration and candidate validation.
- Added CPU, CUDA, and WebGPU contracts. GPU ingress requires the expected
  device kind and ordinal plus the backend family; WebGPU also requires the
  expected allocation domain.
- Validated transfer outputs against destination residency after dtype, shape,
  checked logical element count, buffer length, and placement validation.
  Metadata-only retagging of a foreign allocation now returns
  `TransferProviderContractError::DestinationResidencyMismatch`.
- Passed the registered direct storage-class reachability set into schedule
  construction. A transfer source is selected only from an available copy with
  a registered direct provider; no route now returns typed
  `PrepareError::MissingTransferProvider` before execution.
- Added fake allocation-domain buffers and a materializing two-location
  provider test. The provider allocates and copies into the destination domain,
  and the test asserts that source and destination allocation identities
  differ.
- Replaced the unchecked transfer shape product with the shared checked tensor
  shape-product helper and preserved its typed source through
  `TransferProviderContractError::LogicalElementCount`.

## Independent Review High Findings

- Validate every executor-produced output against the scheduled engine's
  resident-tensor contract before retaining or publishing the slot. A foreign
  allocation-domain output now fails with typed
  `EngineExecutionContractError::OutputResidencyMismatch`.
- Include backend family and allocation domain in input signatures so prepared
  cache roots cannot alias physically incompatible ingress schedules that have
  the same logical placement and layout.
- Select input ingress using both the physical input signature and direct
  transfer reachability to the input's first scheduled consumer. A candidate
  that accepts the input but cannot reach that consumer is skipped.
- Added regression coverage for a faulty executor, alternating same-placement
  inputs from two allocation domains across cache reuse, and choosing a
  route-capable ingress over an earlier registered dead end.

## Merged-Main Integration

- Reconstructed the follow-up on merged main `1cbee7a7` after #1514 and #1517
  by cherry-picking only historical commits `6e8ee624`, `1eef5ce2`, and
  `9f99bb3e`.
- `git range-diff` reports each reconstructed commit as patch-identical to its
  historical source. The squash-merged #1514 commits were not replayed.
- The three commits applied without conflicts. The #1517 static-indexing and
  buffer-pool changes remain independent of the runtime ingress/residency
  files in this follow-up.
- Repository-rules review found that the public runtime entrypoint rustdoc did
  not name the new no-ingress and missing-transfer-route preparation failures.
  Updated `run_compiled`, `prepare_compiled`, `submit`, and
  `run_compiled_values` to document those typed causes.

## Independent Review Follow-Up

- Moved compiled-graph preparation ahead of `thread::spawn` in
  `Runtime::submit`. Invalid physical ingress and missing transfer routes now
  return synchronously, while the worker receives the prepared artifact and
  retained owned inputs.
- Replaced first-semantic-consumer ingress selection with staged physical-use
  analysis. Preparation stages the semantic program once, records every direct
  input-slot consumer in instruction order, and selects an ingress whose
  retained copies can reach the complete use sequence.
- Physical-use analysis assigns compiler-synthesized instructions with no
  semantic operation index to the root execution location, matching schedule
  construction instead of silently omitting those consumers.
- Added `InputIngressContractError::ResidencyMismatch` as the typed source for
  `run_prepared` inputs whose logical metadata still matches but whose backend
  family or allocation domain is incompatible with the prepared ingress.
- Kept `TensorView` physical residency accessors public and documented in
  parity with `TensorRead`; the final repository-rules review identified the
  initial private visibility as unintended public-surface asymmetry.
- TDD regressions first reproduced both synchronous-submit failures, a split
  input whose second consumer was unreachable, a synthesized root instruction
  omitted by semantic-only selection, and message-only prepared-input
  rejection. Each focused test passed after its owning fix.

## Independent Review Changes Requested

- Replaced the raw join-handle submission with a runtime-owned in-flight state.
  The state owns the admitted prepared graph and retained snapshot/providers,
  owned inputs, and completion output/error. The worker captures only this
  state, so reconfiguration after successful admission cannot invalidate the
  run. Handle drop remains nonblocking and does not cancel execution.
- Added typed `SubmissionError::WorkerSpawn` handling around
  `thread::Builder::spawn`. Deterministic worker tests cover delayed start,
  spawn failure, blocked-worker handle drop, execution error, and panic
  completion; public integration coverage reconfigures immediately after
  `submit` and still observes the admitted result.
- Integrated operation placement with ingress and schedule reachability.
  Preparation tries each registered engine as the deterministic preferred
  anchor, falls back only where that engine lacks an operation capability,
  validates the complete input use graph, and constructs the physical schedule
  before accepting a placement. This is polynomial in engine and operation
  count rather than a Cartesian placement search. Route-specific failures
  continue to later engines and storage classes; if every route fails, the last typed
  `NoInputIngress`/`MissingTransferProvider` is preserved.
- Kept `ExecutionLocation` attached to terminal slots through output
  collection. Tensor and value materialization now use the executor retained
  for the output's physical engine rather than the semantic root executor.
  A foreign allocation-domain, non-contiguous lazy-read test verifies that the
  nonroot location and physical residency reach the materializer.
- TDD first reproduced the same-storage first-capable/dead-ingress failure, the
  later-storage variant, and location loss at the terminal lazy-read boundary.
  The focused tests passed after their owning changes.

## Exact-Commit Review Follow-Up

- Preserved the deterministic engine-anchor placement search as the polynomial
  fast path, then added a bounded Cartesian fallback only after every anchor
  fails with a route-specific typed error. The fallback has a hard 4,096-build
  budget and returns `DispatchSearchBudgetExceeded` instead of allowing an
  unsatisfiable graph to grow exponentially without bound. A hash set skips
  anchor vectors already covered by the fallback without quadratic duplicate
  scans.
- Enumerated every registered storage class for root and per-operation
  placement. Engine defaults remain globally preferred before non-default
  classes, and explicit storage constraints remain unchanged.
- Replaced the synthetic admission epoch test and the racy immediate-
  reconfigure integration test with a deterministic two-barrier test through
  `submit_with_spawner`, real graph preparation, and real scheduled output
  materialization. The replacement executor deliberately rejects
  materialization, proving admitted work retains and uses the prior executor.
- TDD RED reproduced both placement defects as typed `NoInputIngress`:
  the only valid two-operation combination was `B -> D`, and a single engine's
  only valid ingress was its non-default registered storage class.
- TDD GREEN covers the exact `B -> D` execution and transfer endpoints, the
  non-default storage execution, existing route retries, and typed exhaustion.
- Follow-up review found that GPU ingress trusted forgeable family metadata.
  CUDA and WebGPU buffers now retain their private source-device ordinal, and
  runtime ingress requires both the concrete executor-owned buffer type and
  matching source ordinal. Tests reject synthetic family matches and relabeled
  real buffers; backend-created CUDA and WebGPU tensors remain accepted.
- Renamed the synchronous transfer entry point to `transfer_blocking` and
  documented immediate destination readability as a provider requirement.
  Native stream/queue enqueue belongs to the event-domain driver contract and
  cannot be registered through this interface.
- Kept runtime ingress distinct from direct backend error handling. CPU runtime
  ingress rejects GPU residency so routing must use an explicit transfer
  provider; it does not replace the existing direct `CpuBackend` runtime-state
  error and its actionable "download to host" diagnostic.

## Deferred Scope

- Event-domain schedulers and device-native asynchronous completion.
- CUDA and WebGPU transfer adapters.
- Collectives, distributed tensors, and real multi-GPU execution.
- Any change to transfer-provider registry keying.

## Documentation Impact

The scheduling model remains crate-private. Public documentation changes cover
the additive `TransferRequest` endpoint accessors, `TransferError`,
`InputIngressContractError`, and the synchronous preparation guarantee on
`Runtime::submit`; no user guide or migration guide change is required.

## Verification

- `cargo test -p tenferro-runtime runtime::tests::schedule:: --lib`
- `cargo test -p tenferro-runtime --test integration runtime_execution::`
- `cargo test -p tenferro-runtime`: passed across library, integration, and
  doctest targets.
- `cargo test -p tenferro-runtime
  runtime::preparation::execution_location_tests::missing_resolved_engine_returns_typed_prepare_error
  --lib`
- `cargo fmt --all --check`: passed.
- `bash scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test -p
  tenferro-runtime --test integration runtime_execution::'`: passed, including
  workspace and extension clippy with warnings denied.
- The fast check was rerun after rebasing onto `origin/main` at `96c9e1c4`
  with both the runtime execution module and typed preparation error test; it
  passed.
- `python3 scripts/repository-rules-review.py --base origin/main --worktree`:
  passed. Follow-up unit tests are organized under `src/runtime/tests/`.
- Rebased the complete W7 follow-up onto `origin/main` at `035c02b0`, then
  reran `cargo test -p tenferro-runtime --lib` (336 passed), the 14 focused
  production runtime execution integration tests, CUDA and WebGPU ingress
  validator tests, and `cargo check -p tenferro-gpu --features cuda,webgpu`.
- `cargo test -p tenferro-runtime --doc` (367 passed) and
  `cargo test -p tenferro-tensor --doc` (307 passed), including the new public
  ingress, transfer-contract, and tensor-buffer identity examples.
- After the independent-review fixes, `cargo test -p tenferro-runtime` passed:
  336 unit tests, 93 integration tests, and 369 doctests.
- `cargo test -p tenferro-cpu runtime_adapter -j 48` passed 10 focused tests.
- `cargo test -p tenferro-gpu --features cuda,webgpu registration_ingress -j
  48` passed the CUDA and WebGPU ingress tests, and `cargo check -p
  tenferro-gpu --features cuda,webgpu -j 48` passed.
- On merged main `1cbee7a7`, `cargo test -p tenferro-runtime -j 48` passed 336
  unit tests, 93 integration tests, and 369 doctests.
- After the exact-commit review follow-up, `cargo test -p tenferro-runtime -j
  48` passed 341 unit tests, 102 integration tests, and 371 doctests.
- `python3 scripts/ci/run_profile.py fmt` and
  `python3 scripts/ci/run_profile.py clippy` passed, including all workspace
  targets and extension manifests with warnings denied.
- `python3 scripts/repository-rules-review.py --base HEAD --worktree --timeout
  120` passed with no findings after documenting the complete placement
  fallback's explicit complexity invariant.
- After rebasing the exact-commit review tree onto `origin/main` at
  `39e96af5`, `scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test
  -p tenferro-runtime -j 48'` passed. This includes workspace and extension
  fmt/clippy plus 341 unit tests, 102 integration tests, and 371 doctests.
- The final run used a branch-specific Cargo target. Reusing one target across
  divergent worktrees incorrectly reused a generated core-op macro artifact
  from the sum-of-squares branch; isolating the target restored source-consistent
  compilation without changing this branch.
- On the same integrated tree, `cargo test -p tenferro-cpu runtime_adapter -j
  48` passed 10 focused tests, `cargo check -p tenferro-gpu --features
  cuda,webgpu -j 48` passed, and the CUDA/WebGPU ingress test filter passed two
  tests.
- After the in-flight/placement/output review changes, `cargo test -p
  tenferro-runtime -j 48` passed 341 unit tests, 101 integration tests, and 371
  doctests. CPU adapter tests passed 10/10; the combined CUDA/WebGPU feature
  check and two ingress tests also passed.
- The final review repair ran the forged/owned/relabeled CUDA ingress tests on
  the local A100 (2/2 passed), the equivalent WebGPU tests (2/2 passed), the
  bounded-search unit test, and all 13 transfer-focused runtime tests.
