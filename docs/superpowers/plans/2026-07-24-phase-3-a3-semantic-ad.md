# Phase 3 A3 Semantic AD Migration Implementation Plan

**Goal:** Move extension automatic-differentiation ownership into
`tenferro-ad`, express every extension rule in semantic-program values, and
delete the old computegraph-keyed rule registry and execution compatibility
surface atomically before Phase 3 closes.

**Architecture:** `SemanticExtensionRuleSet`, owned by `AdContext`, stores
three object-safe rule roles keyed by extension family. Rules receive ordered
`ProgramValue` inputs, outputs, tangents/cotangents, active masks, residuals,
provenance, and an explicit `SemanticProgramBuilder`. The semantic transform
rejects effects before rule dispatch, validates all lengths and builder
ownership centrally, and tries extension semantic AD before any
extension-to-standard lowering. `AdValue::{Absent, Value(ProgramValue)}`
represents inactive values without synthetic graph keys.

**Tech Stack:** Rust, `tenferro-runtime::program`, `tenferro-ad`, extension
crates, traced/eager AD integration tests, Cargo tests/clippy/rustdoc.

**Completion status (2026-07-24):** Implemented through the Phase 3 closure
commit `e40bcd3c` and the subsequent completion audit. The audit also moved the
remaining AD-engine dispatch contract out of `ExtensionOp` ownership and added
a source-contract regression test.

---

### Task 1: Freeze the semantic extension AD contract

**Files:**
- Create: `crates/tenferro-ad/src/semantic_extension.rs`
- Modify: `crates/tenferro-ad/src/lib.rs`
- Modify: `crates/tenferro-ad/src/extension.rs`
- Test: `crates/tenferro-ad/tests/integration/semantic_extension.rs`

- [x] Add compile/runtime tests for object safety, ordered request fields,
  `AdValue`, active masks, residuals, provenance, and explicit builder use.
- [x] Add typed semantic AD errors for malformed family IDs, duplicate roles,
  missing roles, arity/mask/result mismatches, foreign builder values,
  effects, unsupported rules, and wrapped program-build failures.
- [x] Implement `SemanticLinearizeRule`, `SemanticLinearTransposeRule`, and
  `SemanticPrimalVjpRule` using only runtime program types and `ExtensionOp`.
- [x] Implement clone-on-write `SemanticExtensionRuleSet` with atomic merge,
  role-specific registration, lookup, and duplicate rejection.
- [x] Export the new contract from `tenferro-ad`, not
  `tenferro-internal-ops`.
- [x] Run focused integration tests, doctests, clippy, and public-error-doc
  checks; expected result is PASS.
- [ ] Commit the contract.

### Task 2: Make `AdContext` the sole semantic-rule owner

**Files:**
- Modify: `crates/tenferro-ad/src/context.rs`
- Modify: `crates/tenferro-ad/src/eager.rs`
- Modify: `crates/tenferro-ad/src/eager_builder.rs`
- Test: context/eager integration tests

- [ ] Replace `ExtensionRuleSet` fields and builder inputs with
  `SemanticExtensionRuleSet`.
- [ ] Preserve explicit context ownership and cache sharing; do not add a
  process-global fallback.
- [ ] Add tests proving atomic multi-set merge and independent cloned
  contexts.
- [ ] Run context, eager, cache, doctest, and clippy gates; expected result is
  PASS.
- [ ] Commit context ownership.

Progress: `AdContext` now owns and atomically merges
`SemanticExtensionRuleSet`; the legacy registry remains temporarily only
because the current traced transform and unmigrated families still consume it.

### Task 3: Transform extension operations semantically

**Files:**
- Modify/Create: semantic transform implementation under `crates/tenferro-ad`
- Modify: traced AD entry points and transform cache
- Test: semantic transform integration fixtures

- [ ] Add failing JVP/VJP tests for ordered inputs/outputs, inactive values,
  multi-output rules, residuals, active masks, aliases, guards, and provenance.
- [ ] Import primal dependency closures into a destination
  `SemanticProgramBuilder` and preserve ordered external inputs and bindings.
- [ ] Reject observable effects before dispatch and validate every rule return
  against output/input arity and destination-builder ownership.
- [ ] Attempt semantic extension rules before extension-to-standard lowering;
  keep differentiation-after-lowering only as an explicit typed fallback.
- [ ] Retain typed build/metadata sources across the AD error boundary.
- [ ] Run semantic AD, traced AD, eager AD, cache, and release gates; expected
  result is PASS.
- [ ] Commit semantic transformation.

### Task 4: Migrate every extension family

**Files:**
- Modify: FFT AD rules and tests
- Modify: serial einsum AD rules and tests
- Modify: all linalg AD rules and tests
- Modify: sparse/tropical extension rules and tests
- Modify: AD fixture extension rules

- [ ] Migrate FFT and declare complete semantic effects/aliases.
- [ ] Migrate serial einsum before editing its private execution staging
  owner.
- [ ] Migrate every linalg operation kind, including multi-output residual
  cases.
- [ ] Migrate sparse, tropical, and all in-tree fixtures.
- [ ] Run each owner crate's unit/integration/doctest/clippy gates after its
  migration; expected result is PASS except explicitly documented unrelated
  holds.
- [ ] Commit migrations in reviewable owner-crate groups.

Progress: FFT implements all three semantic roles and has extension-first JVP,
adjoint normalization, active/absent handling, and transform-length
truncate/pad coverage. The old FFT trait impl remains only until the traced
semantic transform switches over.

### Task 5: Delete the old AD and execution compatibility surfaces

**Files:**
- Modify: `crates/tenferro-internal-ops/src/ext_op.rs`
- Modify: `crates/tenferro-runtime/src/graph/executor.rs`
- Modify: remaining AD/runtime integration helpers
- Modify: serial einsum staging owner

- [ ] Delete `ExtensionLinearizeRule`, `ExtensionLinearTransposeRule`,
  `ExtensionPrimalVjpRule`, `ExtensionRuleSet`, their registry errors/roles,
  and all computegraph/tidu AD imports from extension-op ownership.
- [ ] Delete keyed `GraphExecutor::*_with_bindings` compatibility methods and
  the private `GraphProgramInput` staging metadata.
- [ ] Make raw `eval_exec_ir*` methods crate-private and migrate remaining AD
  reference tests to semantic/compiled execution.
- [ ] End the public `ExecProgram` construction/re-export surface, retaining
  only the single crate-private semantic-to-execution staging type and forward
  adapter allowed until Phase 5.
- [ ] Prove forbidden symbols are absent from current source and only the
  allowed private adapter remains.
- [ ] Run workspace debug/release tests, doctests, all-target clippy, formatting,
  GPU compile gates, and `git diff --check`; expected result is PASS.
- [ ] Apply the benchmark policy: a measured 5% change triggers remeasurement;
  only a reproducible slowdown of roughly 50% or more blocks progress.
- [ ] Commit the Phase 3 semantic AD checkpoint.

### Task 6: Replace graph-compiler einsum tracing atomically

**Files:**
- Modify: runtime trace context/compiler interface
- Modify: `crates/tenferro-einsum`
- Modify: dependent frontends and tests

- [ ] Add `TraceContextEinsumExt` methods accepting `TraceValue` and remove
  `GraphCompilerEinsumExt` without a compatibility shim.
- [ ] Route extension-first semantic compilation through the pure compiler
  boundary and its accepted cache key.
- [ ] Migrate all callers and docs in the same change.
- [ ] Run Phase 3 acceptance searches and full affected gates.
- [ ] Commit the Phase 3 closure.
