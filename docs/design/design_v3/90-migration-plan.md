# Design V3 Migration Plan

## Summary

The `v3` migration should be staged so that each step removes architectural
debt while preserving behavior.

The repository should not attempt a single cutover PR.

## Stage 0: Proposal And Review

Goals:

- review the `design_v3` proposal set
- supersede `#738`, `#740`, and `#741` — this document set is the source of
  truth for the direction they explored, and they are closed in favour of it
- separate clearly accepted direction from open questions

Deliverables:

- design approval on this proposal set
- a narrowed implementation plan with explicit sequencing
- closure comments on `#738` / `#740` / `#741` pointing readers to
  `docs/design/design_v3/`

## Stage 1: Value-Side Metadata Cleanup

Land the `20-shape-metadata.md` direction first.

Goals:

- expose shape and dtype queries at the builder and emitter boundary
- remove Category C shape snapshots from op variants incrementally

Why first:

- lowest architectural risk
- improves hashing and AD clarity immediately
- reduces coupling before any larger op-model decisions

**Prerequisite**: decide serialized graph compatibility policy. Removing
Category C fields changes op payload hashes, so any existing serialized graph
persistence must be versioned or explicitly declared non-stable before this
stage lands.

**Acceptance criteria**:

- oracle-replay baselines stay green after every sub-step
- no AD rule reads a Category C snapshot after the stage is complete
- step 2 (`lhs_rank`/`rhs_rank`) already landed in `#737`; remaining steps
  follow the order in `20-shape-metadata.md`

## Stage 2: Clarify The Mainline Graph Story

Goals:

- document one main traced op vocabulary as the architectural source of truth
- stop treating `SemiringOp` as co-equal to the traced AD substrate

Possible code moves:

- de-emphasize `SemiringOp` in docs and tests
- isolate semiring-only compile and execution paths as non-mainline
- trim trait surfaces whose only purpose is to keep semiring and standard graph
  paths looking symmetric

## Stage 3: Core AD Consolidation

Goals:

- finish moving AD rules to metadata queries rather than op-embedded input
  shapes
- keep `PrimitiveOp` and current differentiation flow
- flatten any remaining op vocabulary inconsistencies

This stage is still an architectural consolidation, not a runtime rewrite.

**Exit gate — `tenferro-algebra` fate**: before closing this stage, resolve
the binary choice described in `30-algebra-and-tropical.md`:

- Option A: delete the crate if `Semiring`/`HasAlgebra`/`Standard` have no
  remaining callers
- Option B: retain as eager-only convenience with all graph-facing API
  removed

This gate must be closed before Stage 4 begins so that the external tropical
crate is built against a stable boundary.

**Acceptance criteria**:

- oracle-replay baselines stay green
- no AD rule depends on algebra-generic graph ops

## Stage 4: Tropical Externalization — Phase 1 (Composition Only)

Goals:

- create the external `tenferro-ext-tropical` crate outside the core
  workspace members
- implement tropical surface operations as compositions of core primitives
  (e.g. max-plus via `BroadcastInDim + Add + ReduceMax`)
- ship `MaxPlus<T>`, `MinPlus<T>`, `MaxMul<T>` scalar newtypes with standard
  Rust arithmetic trait implementations for eager T-generic kernels

Why external:

- validates that the public `tenferro` facade is sufficient for realistic
  extension work without any core-workspace dependency
- keeps the core graph substrate free of algebra-parameterized types

**Acceptance criteria**:

- `tenferro-ext-tropical` compiles and its tests stay green using only
  public `tenferro` APIs
- no new core-workspace dependency is required for the composition path
- oracle-replay baselines stay green for any core AD rule touched during
  integration

## Stage 5: Define The `ExtensionOp` Contract (Document-Only)

This stage is deliverable-driven, not evidence-driven. Its goal is a written
specification of the extension substrate.

Goals:

- specify what it means for two `ExtensionOp` values to be equal
- specify the hashing protocol (stable family identifier + hashable payload)
- specify the `Clone` requirement and identity preservation
- specify the AD closure requirement: `linearize` and `transpose_rule` for
  an `ExtensionOp` must emit only core op values
- specify the serialization compatibility policy, including versioning of
  the family identifier

Deliverables:

- a new spec document (proposed file: `docs/spec/extension-op.md`) that
  normatively describes the contract
- cross-links from `40-extension-boundary.md` to the spec

**Acceptance criteria**:

- the spec answers every question currently listed in
  `40-extension-boundary.md` under "Why A Raw Trait Object Is Not Enough"
- the spec is signed off by project review before Stage 6 begins

## Stage 6: Implement The `ExtensionOp` Mechanism

Goals:

- add a `TensorOp::Extension(Arc<dyn ExtensionOp>)` variant (or an
  equivalently specified carrier) to the core op vocabulary
- wire the variant through the engine, compile path, eager emitter, and
  backend dispatch so that a registered `ExtensionOp` can be executed and
  differentiated
- provide a registration API that lets external crates supply
  `ExtensionOp` implementations without modifying the core

This stage implements exactly what Stage 5 specified; it does not relitigate
the contract.

**Acceptance criteria**:

- one or more in-repo smoke-test `ExtensionOp` examples execute end-to-end
  (primal and backward)
- oracle-replay baselines stay green for all existing core ops
- no core call site depends on the concrete `ExtensionOp` type

## Stage 7: Tropical Externalization — Phase 2 (Fused `ExtensionOp`)

Stage 7 delivers the first canonical external `ExtensionOp`, and doubles as
the contract test for Stages 5 and 6.

Goals:

- implement `FusedTropicalDotGeneral` in `tenferro-ext-tropical` as an
  `ExtensionOp`
- the primal path runs a fused tropical GEMM kernel
- the AD path is argmax-based: `linearize` records argmax indices; the
  backward rule uses `Gather` / `Scatter` on the core op vocabulary
- this resolves `#212` through the external crate, not through the core
  workspace

**Acceptance criteria**:

- the external crate registers `FusedTropicalDotGeneral` and runs its
  primal and backward successfully
- the same external crate exercises every requirement in the Stage 5
  contract — the stage effectively self-tests the `ExtensionOp` substrate
- oracle-replay baselines stay green for any shared code touched

## Stage 8 (Optional): Core-Owned Fused Primitives

This stage is gated on measured performance evidence. It is not on the
critical path.

Trigger conditions:

- composition-based or `ExtensionOp`-based implementations of a repeatedly
  needed pattern are measurably slow in production workloads
- the pattern is broad enough to justify a core op variant rather than an
  external `ExtensionOp`

Goals (when triggered):

- introduce a dedicated fused op variant in the core `TensorOp` enum
- the AD rule normally remains decomposition into core ops, unless
  profiling shows the decomposition itself is too expensive

No action is taken at Stage 8 unless evidence demands it.

## Open Questions To Resolve Before Implementation

1. `tenferro-algebra` A-vs-B decision — **scheduled: Stage 3 exit gate** (see
   `30-algebra-and-tropical.md`).
2. How much of the current `SemiringBackend` public API should remain during
   migration?
3. What is the exact metadata carrier type exposed to AD and lowering code?
4. Serialized graph compatibility policy — **scheduled: Stage 1
   prerequisite**.
5. `ExtensionOp` contract (identity, hashing, AD closure, serialization
   versioning) — **scheduled: Stage 5 deliverable**.
6. What performance threshold would justify a Stage 8 core-owned fused
   primitive?

## Recommended Order Of Engineering Work

The recommended order is:

1. doc alignment and issue closure (Stage 0)
2. shape metadata cleanup (Stage 1)
3. mainline graph story clarification (Stage 2)
4. core AD consolidation plus `tenferro-algebra` decision (Stage 3)
5. tropical externalization Phase 1 — composition only (Stage 4)
6. `ExtensionOp` contract specification (Stage 5)
7. `ExtensionOp` mechanism implementation (Stage 6)
8. tropical externalization Phase 2 — fused `ExtensionOp` (Stage 7)
9. optional core-owned fused primitives, evidence-gated (Stage 8)

This order commits to the generic extension mechanism as a planned stage
(Stages 5–7) rather than an open-ended deferred question, while still
allowing the optional fused-primitive path (Stage 8) to remain
evidence-gated.
