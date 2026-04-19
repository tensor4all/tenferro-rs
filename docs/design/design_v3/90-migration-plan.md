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
- `DotGeneralConfig` field cleanup in `#737` is a related but incomplete
  precedent; Stage 1 still removes `lhs_rank` / `rhs_rank` from
  `StdTensorOp::DotGeneral` itself
  (`tenferro-ops/src/std_tensor_op.rs:23-27`) and continues with the order
  in `20-shape-metadata.md`
- `TensorMeta` is total over symbolic inputs — no panic, no
  placeholder-to-zero collapse

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
- close symbolic-shape AD failure modes: fix the zero-tangent collapse at
  `tenferro/src/traced.rs:820-833`, re-enable the ignored tests in
  `tenferro/tests/symbolic_grad.rs`, and document the deferred
  zero-tangent policy
- add `Gather` and `Scatter` to the core AD dispatch. Today these variants
  are absent from `linearize_non_semiring` and `transpose_non_semiring`
  (`tenferro-ops/src/ad/mod.rs:18,161`), which means Stage 7's
  argmax-based tropical backward has no core op to emit. Closing this gap
  is a prerequisite for Stage 7

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
- `tenferro/tests/symbolic_grad.rs` runs with no `#[ignore]` attributes
  remaining
- the deferred zero-tangent semantics is captured as a design note
  (appended to `10-ad-model.md` or a new file under `docs/design/`)
- `Gather` and `Scatter` have `linearize` / `transpose_rule`
  implementations in the core AD dispatch, keeping the core op vocabulary
  closed under AD

## Stage 4: Tropical Externalization — Phase 1 (Composition Only)

Stage 4 creates the external `tenferro-ext-tropical` crate and proves the
public facade is sufficient for realistic composition-based extension work.

**Scope of "public tenferro APIs" in this stage**: the traced `TracedTensor`
API re-exported from the `tenferro` facade (`tenferro/src/lib.rs:39-48`).
Eager typed-tensor T-generic kernels via `SemiringBackend` are *not* part
of Stage 4; their fate is decided by the Stage 3 `tenferro-algebra` gate
and any follow-up eager work is scheduled after Stage 7.

### Stage 4a: Concrete-shape composition

Goals:

- create the external `tenferro-ext-tropical` crate outside the core
  workspace members
- implement tropical traced operations on concrete-shape inputs as
  compositions of core primitives (e.g. max-plus via
  `BroadcastInDim + Add + ReduceMax`)
- ship `MaxPlus<T>`, `MinPlus<T>`, `MaxMul<T>` scalar newtypes for eager
  T-generic kernels (eager-path integration deferred)

**Public-API prerequisites** (discovered during review): lift any core op
used by tropical composition that is currently internal onto the public
`TracedTensor` surface — e.g. `reduce_max` / `reduce_min` today live in
`tenferro/src/linalg_api.rs:713` as non-public helpers.

**Acceptance criteria**:

- `tenferro-ext-tropical` compiles and its concrete-shape tests stay
  green using only the traced `tenferro` facade
- no new core-workspace dependency is required for the composition path
- existing in-tree tropical tests (`tenferro/tests/tropical.rs`) have
  equivalent coverage in `tenferro-ext-tropical`, so Stage 6 can retire
  the in-tree path without losing test coverage
- oracle-replay baselines stay green for any core AD rule touched during
  integration

### Stage 4b: Symbolic-shape composition (contract test for Stage 3)

Stage 4b extends the Stage 4a composition surface so that the same
tropical wrappers work under symbolic-shape inputs, acting as the contract
test for Stage 3's symbolic-AD correctness work.

Goals:

- provide public traced APIs that accept symbolic shapes where Stage 4a
  requires concrete (e.g. a `broadcast_in_dim` variant taking `DimExpr`,
  or composition primitives that derive output shape from `TensorMeta`)
- extend the tropical composition wrappers to cover symbolic inputs

**Acceptance criteria**:

- symbolic-shape tropical wrappers run end-to-end in
  `tenferro-ext-tropical`, including backward
- no `#[ignore]` remains on related symbolic AD tests
- the added symbolic facade is source-compatible for Standard-scalar
  users (no type parameter leaks into existing methods)

## Stage 5: Define The `ExtensionOp` Contract (Document-Only)

This stage is deliverable-driven, not evidence-driven. Its goal is a
written specification of the extension substrate that is detailed enough
for Stage 6 to start without design re-interpretation.

Goals — the spec must cover all of the following:

- **Identity and hashing**: equality semantics, hashing protocol, stable
  family identifier, `Clone` and payload-hashing requirements
- **Arity and I/O shape**: `n_inputs` / `n_outputs` contract (aligning
  with `tenferro-ops/src/std_tensor_op.rs:389-457`)
- **Shape and dtype inference**: output shape / dtype derivation hook,
  covering the responsibility held today by `tenferro/src/shape_infer.rs`
- **Forward execution dispatch**: responsibility split between
  `tenferro/src/compiler.rs` (for compiled exec) and
  `tenferro/src/eager_exec.rs` / `tenferro/src/eager_emitter.rs` (for
  eager path)
- **Registration and lookup**: where the registry lives, how external
  crates register ops, how lookup failures and version mismatches are
  reported
- **AD API surface**: `linearize` and `transpose_rule` must emit only
  core op values and respect the `ShapeGuardContext` surface from
  `tenferro-ops/src/std_tensor_op.rs:521-548`
- **Serialization compatibility**: family identifier versioning,
  cross-process and cross-version guarantees, behavior when a consumer
  lacks a producer's extension family
- **Failure modes**: what happens if `eager_execute` errors, if a backend
  lacks implementation, if AD rules encounter an unregistered extension
- **Legacy-substrate retirement**: explicit policy for how the existing
  `SemiringOp` / `SemiringBackend` pipeline
  (`tenferro-ops/src/semiring_op.rs`,
  `tenferro/src/compiler.rs:101-171`) is retired at Stage 6; the spec
  must either subsume the semiring pipeline under `ExtensionOp` or
  document why it remains distinct

Deliverables:

- a new spec document (proposed file: `docs/spec/extension-op.md`) that
  normatively describes the contract
- cross-links from `40-extension-boundary.md` to the spec

**Acceptance criteria**:

- the spec answers every question currently listed in
  `40-extension-boundary.md` under "Why A Raw Trait Object Needs A Spec
  First"
- the spec is signed off by project review before Stage 6 begins
- the `SemiringOp` retirement path is normative, not aspirational

## Stage 6: Implement The `ExtensionOp` Mechanism

Goals:

- add an `Extension(Arc<dyn ExtensionOp>)` variant to the core op enum.
  The core enum is `StdTensorOp` today
  (`tenferro-ops/src/std_tensor_op.rs:16-17`); any rename is a separate
  later consolidation and is out of scope for this stage. The
  trait-object carrier is committed per `40-extension-boundary.md`;
  Stage 5 specifies the contract the carrier must satisfy, not the
  carrier shape
- wire the variant through the engine, compile path, eager emitter, and
  backend dispatch so that a registered `ExtensionOp` can be executed
  and differentiated
- provide a registration API that lets external crates supply
  `ExtensionOp` implementations without modifying the core
- retire the legacy in-tree `SemiringOp` / `SemiringBackend` pipeline per
  the policy normatively specified in Stage 5: delete
  `tenferro-ops/src/semiring_op.rs`, the semiring compile path in
  `tenferro/src/compiler.rs`, `SemiringBackend<Alg>` at
  `tenferro-tensor/src/backend.rs:566`, and the in-tree tropical tests
  whose coverage moved to `tenferro-ext-tropical` in Stage 4a. Removing
  `SemiringBackend` does *not* block eager tropical via scalar newtypes:
  `TypedTensor<T>` T-generic kernels remain and operate on `MaxPlus<T>`
  / `MinPlus<T>` through their standard Rust arithmetic trait impls

This stage implements exactly what Stage 5 specified; it does not
relitigate the contract.

**Acceptance criteria**:

- one or more in-repo smoke-test `ExtensionOp` examples execute
  end-to-end (primal and backward)
- oracle-replay baselines stay green for all existing core ops
- no core call site depends on the concrete `ExtensionOp` type
- the `SemiringOp` pipeline and its in-tree tropical tests are removed
- the Stage 3 `tenferro-algebra` A/B choice is reflected concretely in
  this stage's code deletions (Option A: crate fully removed here;
  Option B: only graph-facing API removed)

## Stage 7: Tropical Externalization — Phase 2 (Fused `ExtensionOp`)

Stage 7 delivers the first canonical external `ExtensionOp`, and doubles as
the contract test for Stages 5 and 6.

**Prerequisite**: `Gather` and `Scatter` AD rules must be in place in the
core AD dispatch. This is delivered as a Stage 3 goal; without it the
argmax-based backward cannot emit valid core-op cotangents.

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

1. `tenferro-algebra` A-vs-B decision — **scheduled: Stage 3 exit gate**
   (see `30-algebra-and-tropical.md`).
2. `SemiringBackend` public API fate during migration — **scheduled:
   Stage 5 spec, Stage 6 deletion**.
3. Exact metadata carrier type exposed to AD and lowering code — a
   `TensorMeta { dtype, shape: Vec<DimExpr> }` baseline is proposed in
   `20-shape-metadata.md`; final type name and placement are Stage 1
   deliverables.
4. Serialized graph compatibility policy — **scheduled: Stage 1
   prerequisite**.
5. `ExtensionOp` contract (identity, hashing, arity, shape inference,
   dispatch, registry, AD closure, serialization versioning, failure
   modes, legacy retirement) — **scheduled: Stage 5 deliverable**.
6. Symbolic-shape AD correctness (`tenferro/src/traced.rs:820-833` fix,
   `symbolic_grad.rs` re-enable, deferred zero-tangent policy) —
   **scheduled: Stage 3 goal and acceptance**.
7. What performance threshold would justify a Stage 8 core-owned fused
   primitive?

## Recommended Order Of Engineering Work

The recommended order is:

1. doc alignment and issue closure (Stage 0)
2. shape metadata cleanup (Stage 1)
3. mainline graph story clarification (Stage 2)
4. core AD consolidation, `tenferro-algebra` decision, and symbolic-shape
   AD correctness (Stage 3)
5. tropical externalization Phase 1 — concrete-shape composition first
   (Stage 4a), then symbolic-shape composition as the contract test for
   Stage 3's symbolic-AD work (Stage 4b)
6. `ExtensionOp` contract specification (Stage 5)
7. `ExtensionOp` mechanism implementation, including `SemiringOp` /
   `SemiringBackend` retirement (Stage 6)
8. tropical externalization Phase 2 — fused `ExtensionOp` (Stage 7)
9. optional core-owned fused primitives, evidence-gated (Stage 8)

This order commits to the generic extension mechanism as a planned stage
(Stages 5–7) rather than an open-ended deferred question, while still
allowing the optional fused-primitive path (Stage 8) to remain
evidence-gated.
