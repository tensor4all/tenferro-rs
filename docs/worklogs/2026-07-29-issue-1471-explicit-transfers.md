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
