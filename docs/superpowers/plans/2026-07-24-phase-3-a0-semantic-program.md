# Phase 3 A0 Semantic Program Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and freeze the backend-neutral `tenferro_runtime::program`
artifact, bindings, structural identity, and validation-preserving transform
contract required by accepted issue #1449 through normative correction v2.4.

**Architecture:** A new public `program` module is self-contained inside
`tenferro-runtime` and temporarily coexists with the legacy graph/execution
types. Opaque builder-owned tokens index private SSA storage; immutable views
are allocation-free; tensor bindings freeze separately; imports and transforms
publish only after complete validation. `CoreSemanticOp` is a closed enum
matching the existing backend-neutral core vocabulary while excluding its
extension carrier; semantic inputs remain separate SSA declarations.

**Tech Stack:** Rust 2021, `Arc`, `thiserror`, `sha2`, existing
`tenferro-internal-ops` semantic types, unit/compile-fail/source-contract tests.

---

## File structure

- Create `crates/tenferro-runtime/src/program/mod.rs`: public re-exports and
  module-level contract.
- Create `crates/tenferro-runtime/src/program/value.rs`: nonces, opaque
  `ProgramValue`/`BindingKey`, bounded Debug.
- Create `crates/tenferro-runtime/src/program/metadata.rs`: input/value
  metadata, symbolic extents, typed effects, aliases, guards, placement
  constraints, provenance views.
- Create `crates/tenferro-runtime/src/program/op.rs`: public non-exhaustive
  closed `CoreSemanticOp` enum, private semantic operation storage, immutable
  operation views, and migration-only standard-op conversion.
- Create `crates/tenferro-runtime/src/program/bindings.rs`: immutable
  `ProgramBindings`, keys, iterators, bounded Debug.
- Create `crates/tenferro-runtime/src/program/builder.rs`: input/op/extension
  construction, binding, failure-atomic import, structural/binding freeze.
- Create `crates/tenferro-runtime/src/program/identity.rs`: normalized
  structural encoding, cached SHA-256 fingerprint, exact semantic equality.
- Create `crates/tenferro-runtime/src/program/transform.rs`: frozen artifacts,
  imports, transform context and object-safe transform trait.
- Create `crates/tenferro-runtime/src/program/error.rs`: typed build, finish,
  query, import, transform, and control-flow errors.
- Create `crates/tenferro-runtime/src/program/tests.rs`: public behavior,
  rollback, identity, opacity, object-safety, and scaling tests.
- Modify `crates/tenferro-internal-ops/src/ext_op.rs`: add explicit
  `Undeclared`/declared semantic effect and alias descriptors without a silent
  pure/fresh default.
- Modify `crates/tenferro-runtime/src/lib.rs`: expose only the accepted public
  program API.
- Modify `crates/tenferro-runtime/Cargo.toml`: add workspace `sha2`.

### Task 1: Opaque tokens and semantic metadata

- [x] Write failing unit tests asserting:

```rust
let mut left = SemanticProgramBuilder::new();
let mut right = SemanticProgramBuilder::new();
let foreign = left.input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))?;
assert_eq!(
    right.add_op(CoreSemanticOp::try_from(StdTensorOp::Neg)?, &[foreign]),
    Err(ProgramBuildError::ForeignValue)
);
assert_eq!(format!("{foreign:?}"), "ProgramValue(<opaque>)");
```

Also assert borrowed dtype/shape access, equality/order guard construction,
typed effect resources, and all four alias forms (`Fresh`, `ViewOf`,
`MustAlias`, `ExternalAlias`).

- [x] Run:

```bash
cargo test -p tenferro-runtime --lib program::tests::tokens_and_metadata
```

Expected: RED because `program` and its public types do not exist.

- [x] Implement owner nonces, opaque tokens, `ProgramInputSpec`,
`ProgramValueMetadata`, typed effects/aliases/guards/placement constraints,
bounded provenance, and manual Debug without raw IDs.

- [x] Re-run the focused test and require GREEN.

- [x] Commit:

```bash
git add crates/tenferro-runtime/src/program crates/tenferro-runtime/src/lib.rs
git commit -m "feat(runtime): define semantic program metadata"
```

### Task 2: Core and extension operation construction

- [x] Write failing tests for input arity, output metadata count, foreign input,
  invalid alias indices, extension arity, effect declaration, typed control
  flow rejection, and ordered borrowed `SemanticOperationView`.

```rust
let x = builder.input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))?;
let outputs = builder.add_op_with_metadata(
    CoreSemanticOp::try_from(StdTensorOp::Neg)?,
    &[x],
    [ProgramValueMetadata::new(DType::F64, [DimExpr::Const(2)])],
    OperationSemantics::pure_fresh(1),
)?;
assert_eq!(outputs.len(), 1);
```

- [x] Run the focused construction tests and observe the missing API failure.

- [x] Implement every current core primitive as a public
  `#[non_exhaustive] CoreSemanticOp` variant, private `SemanticOp`, operation
  storage, arity/metadata/guard/effect/alias validation, and allocation-free
  ordered operation views. A migration-only conversion from `StdTensorOp`
  rejects its extension carrier through a typed conversion error; extension
  operations enter only through `add_extension`. Add object-safe extension
  semantic declarations whose compatibility default is `Undeclared`; reject
  undeclared effects or aliases with typed errors instead of assuming
  pure/fresh.

- [x] Run construction plus existing runtime library tests.

- [x] Commit:

```bash
git add crates/tenferro-runtime/src/program
git commit -m "feat(runtime): build semantic SSA operations"
```

### Task 3: Separate tensor bindings and atomic finish

- [x] Write failing tests proving a tensor default is absent from semantic
  structure, binding lookup rejects foreign keys, duplicate binding is typed,
  unbound ordinary inputs remain valid, and a failed finish publishes no
  `SemanticProgram` or `ProgramBindings`.

```rust
let key = builder.bind_input(x, Arc::new(tensor.clone()))?;
let frozen = builder.finish(&[x])?;
assert_eq!(frozen.bindings.get(key), Some(&tensor));
assert!(frozen.program.semantic_eq(&same_structure_without_binding));
```

- [x] Run the binding tests and observe RED.

- [x] Implement private pending bindings, immutable sorted `Arc` storage,
  `ProgramBindings::{len,is_empty,get,iter}`, bounded Debug, structural
  validation into temporaries, and consuming atomic `finish`.

- [x] Run all program tests and existing graph compiler tests.

- [x] Commit:

```bash
git add crates/tenferro-runtime/src/program
git commit -m "feat(runtime): freeze semantic programs and bindings"
```

### Task 4: Failure-atomic import

- [x] Write failing import tests for empty roots, duplicate ordered roots,
  binding preservation, metadata/guards/effects/aliases/provenance roundtrip,
  foreign source roots, foreign source binding keys, and mid-import rollback.
  Record builder counts before the failing import and assert every observable
  count is unchanged afterward.

- [x] Run the import tests and observe RED.

- [x] Implement `ProgramImport`, `ImportedProgramValues::roots`, a private
  transaction containing value/binding/op/guard/provenance remaps, and a
  single commit after validation. No partial mutation may occur before commit.

- [x] Run import and full program tests.

- [x] Commit:

```bash
git add crates/tenferro-runtime/src/program
git commit -m "feat(runtime): import semantic programs atomically"
```

### Task 5: Cached structural fingerprint and exact equality

- [x] Add `sha2.workspace = true`, then write failing tests that independently
  built equal programs have equal fingerprints despite different nonces and
  bindings; semantic changes alter exact equality; diagnostic provenance and
  bindings do not; semantic guards/effects/aliases/constants do; and a forced
  fingerprint collision still performs exact equality before reuse.

- [x] Run the identity tests and observe RED.

- [x] Implement one canonical structural byte encoding with normalized input,
  value, and operation ordinals. Compute SHA-256 once during freeze and retain
  the exact normalized semantic representation used by `semantic_eq`.
  Extension encoding uses `family_id`, deterministic payload hashing, and
  `payload_eq`; it never hashes Debug output.

- [x] Add an allocation/counting test proving repeated fingerprint queries do
  not traverse operations or allocate, and `value_metadata` is O(1).

- [x] Run identity tests twice and the full runtime library suite.

- [x] Commit:

```bash
git add crates/tenferro-runtime/Cargo.toml crates/tenferro-runtime/src/program
git commit -m "feat(runtime): fingerprint semantic programs"
```

### Task 6: Binding-preserving semantic transforms

- [x] Write failing trait-object and external identity-transform tests:

```rust
fn assert_object_safe(_: Arc<dyn SemanticTransform>) {}

impl SemanticTransform for Identity {
    fn identity(&self) -> TransformIdentity { self.id }
    fn apply(
        &self,
        context: &mut SemanticTransformContext<'_>,
        input: &FrozenProgram,
    ) -> Result<Box<[ProgramValue]>, SemanticTransformError> {
        Ok(context
            .import_program(input, input.program.outputs())?
            .roots()
            .into())
    }
}
```

Cover ordered transforms, binding preservation, foreign returned roots,
failure rollback, and unchanged transform cache on failure.

- [x] Run transform tests and observe RED.

- [x] Implement opaque fixed-size transform identities, compiler-owned fresh
  builders, `SemanticTransformContext`, validated roots, atomic finish, and a
  collision bucket that retains and checks the exact input artifact.

- [x] Run program/transform tests and full runtime library tests.

- [x] Commit:

```bash
git add crates/tenferro-runtime/src/program
git commit -m "feat(runtime): add validated semantic transforms"
```

### Task 7: Public opacity, dependency, docs, and A0 checkpoint

- [x] Add source/compile-contract tests proving raw token fields and nonces are
  inaccessible, views have no mutable or source-ID escape, Debug is bounded,
  `SemanticTransform` is object-safe, and the `program` module imports no
  provider/resource/scheduler/AD modules.

- [x] Add runnable rustdoc examples for input, core op, extension op, import,
  binding iteration, query errors, identity transform, fingerprint, and
  control-flow rejection.

- [x] Run:

```bash
cargo fmt --all -- --check
cargo test -p tenferro-runtime --lib program
cargo test -p tenferro-runtime --lib
cargo test -p tenferro-runtime --test integration
cargo test -p tenferro-runtime --doc
cargo clippy -p tenferro-runtime --all-targets -- -D warnings
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
python3 scripts/check-public-error-docs.py
python3 scripts/test-doc-consistency.py
git diff --check
```

- [x] Update `docs/design/execution-engine-provider-architecture.md` only for
  implemented A0 details, add
  `docs/worklogs/2026-07-24-phase-3-a0-semantic-program.md`, and record
  remaining P3-A1–A3 migration boundaries without claiming Phase 3 complete.

- [x] Commit the verified A0 checkpoint:

```bash
git add crates/tenferro-runtime docs/design docs/worklogs
git add -f docs/superpowers/plans/2026-07-24-phase-3-a0-semantic-program.md
git commit -m "docs: record semantic program checkpoint"
```
