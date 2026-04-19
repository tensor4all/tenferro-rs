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

## Stage 4: Tropical Externalization

Goals:

- create the external tropical crate or externalized package layout
- implement tropical traced wrappers through composition
- validate that the public traced surface is sufficient without core changes

Optional eager work may happen before this stage if the scalar-newtype path is
useful and independent.

**Acceptance criteria**:

- the external tropical crate compiles and tests green using only public
  tenferro APIs
- no new core-workspace dependency is required for the composition path

## Stage 5: Measure Performance Gaps

Goals:

- benchmark decomposition-based tropical forward and backward paths
- identify whether any fused primitive is actually necessary

The default assumption should be that composition is good enough until data
shows otherwise.

## Stage 6: Revisit Fused Or Extension Mechanisms

Only if performance evidence demands it:

- introduce a core fused primitive for a specific hot path, or
- design a real extension substrate with explicit op identity

This stage should not begin until the identity, hashing, and AD-closure story
is documented in detail.

## Open Questions To Resolve Before Implementation

1. `tenferro-algebra` A-vs-B decision — **scheduled: Stage 3 exit gate** (see
   `30-algebra-and-tropical.md`).
2. How much of the current `SemiringBackend` public API should remain during
   migration?
3. What is the exact metadata carrier type exposed to AD and lowering code?
4. Serialized graph compatibility policy — **scheduled: Stage 1
   prerequisite**.
5. What performance threshold would justify a fused tropical primitive?

## Recommended Order Of Engineering Work

The recommended order is:

1. doc and issue alignment
2. shape metadata cleanup
3. traced AD consolidation
4. tropical composition in an external crate
5. performance evaluation
6. optional fused or extension work

This order keeps the repository on a path of normal evolution and avoids
locking in a generic extension design before the core graph substrate is clean.
