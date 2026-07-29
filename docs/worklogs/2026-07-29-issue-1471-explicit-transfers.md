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
- Confirmed commit `06147e4e` is in branch history and its adjacent
  `// INVARIANT` markers remain at both deferred `schedule.rs` dead-code
  allowances.

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

## Deferred Scope

- Submit/event schedulers and asynchronous completion.
- CUDA and WebGPU transfer adapters.
- Collectives, distributed tensors, and real multi-GPU execution.
- Any change to transfer-provider registry keying.

## Documentation Impact

The scheduling model remains crate-private. Public documentation changes are
limited to the additive `TransferRequest` endpoint accessors and
`TransferError`; no user guide or migration guide change is required.

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
- The fast PR check passed against the branch base `035c02b0`, including
  workspace and extension clippy with warnings denied and 17 focused runtime
  execution integration tests. The ordinary `origin/main` freshness check
  remains pending because this existing review branch has not yet been rebased.
- Deterministic repository-rules review of the uncommitted delta against
  `1eef5ce2` passed; external LLM review was intentionally skipped before PR
  creation.
