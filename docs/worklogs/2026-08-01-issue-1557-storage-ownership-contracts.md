# Worklog: #1557 Storage Ownership Contracts Document

This worklog records the authoring of
`docs/design/storage-ownership-contracts.md`, the Phase 1 contract document
for the storage ownership redesign tracked by
[#1555](https://github.com/tensor4all/tenferro-rs/issues/1555) and owned by
[#1557](https://github.com/tensor4all/tenferro-rs/issues/1557).

## Scope

- Added `docs/design/storage-ownership-contracts.md`: the seven design gates
  (span access and retirement, `AllocationGroup`, submission, method
  distribution, raw handles and reclamation, documentation ownership, AD
  value retention) as signature sketches plus state-transition tables, each
  row answering the six review-checklist questions from #1557 (capability,
  borrow, synchronization, failure return, panic/drop state, reclamation
  legality).
- Added the document to the `docs/design/index.md` Core Design table.
- No code changes. The Phase 1 verification harness (trybuild, parity
  contract, inventories) is separate work under #1557.

## Context read

- #1555 consolidated body (invariants I1 through I10, maintainer synthesis,
  decomposition-check and evaluation comments).
- Phase issues #1556 through #1569, including the expanded gate 7 (AD value
  retention) and the #1568 AD provider-coverage section.
- Current implementation seams cited in the review threads:
  `crates/tenferro-ad/src/eager.rs` (`materialized()`, `GradSlot`,
  `Arc<TensorValue>`), `crates/tenferro-runtime/src/checkpoint.rs`,
  `crates/tenferro-gpu/src/webgpu/mod.rs` (`map_read`/`map_write` guard
  types), `crates/tenferro-tensor/src/types.rs`
  (`try_multi_slice_mut`, `TypedTensorViewMutPair`, `Placement`).

## Decisions

- One document, sections numbered G1 through G7 to match the #1555 gate list
  one to one, so "gate N" resolves to exactly one contract section.
- `UseLease` is specified as `'static` with provider pins rather than a
  borrow, because leases must move into runtime retirement records that
  outlive the submitting borrow. Capability enforcement stays on the
  acquisition methods (`&self` for reads, `&mut self` for writes); the lease
  itself carries no authority.
- Guard leak (`mem::forget`) is documented as sound but possibly
  liveness-degrading until owner drop, so the contract does not depend on
  `Drop` running for safety.
- AD handle types remain `Clone` explicitly: the non-`Clone` rule applies to
  owners and capabilities, and gate 7 defines handles as read-only
  descriptor references. This resolves the apparent conflict between
  "remove shallow `Clone`" (#1559) and existing `EagerTensor` cloning.
- Copy accounting has no retention reason variant at all; acceptance asserts
  every copy in an AD scenario carries one of the enumerated reasons. This
  encodes "retention performs no copy" structurally instead of comparing
  aggregate counts.

## Alternatives rejected

- Splitting the contracts into one document per gate: rejected because #1557
  requires one canonical path referenced by all later phases.
- Publishing a `timeline()` accessor in the trait sketch: rejected by the
  maintainer synthesis; access acquisition methods own that state.
- Specifying exact final public names: rejected; #1557 requires shapes and
  capability rules while names stay provisional until owning phases land.

## Verification

- `bash scripts/check-pr-fast.sh` (documentation-only mode).
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD`.
- Manual cross-check of every contract clause against the #1555 body and the
  phase issues it cites; no contradiction found at authoring time.

## Remaining risks

- The G6 command table names current scripts; a phase that renames tooling
  must update the table in the same PR (stated in the document).
- Signature sketches will drift slightly as owning phases land; the change
  control rule (update document plus tests in the same PR) is the guard.
- The AD `ValueGuard`/`Gradients` sketches are the least implementation-
  tested part of the contract; Phase 3 and Phase 9 may need to refine them
  through the documented change-control path.
