# Structured error model and fallible tensor API

**Date:** 2026-07-17
**Status:** Approved design, pending implementation plan

## Problem

Tenferro deliberately does not turn caller-controlled invalid input into a
panic at a public API boundary. Dynamic tensor operations therefore need a
fallible API when compatibility cannot be proved statically. The current code
implements that principle inconsistently:

- `tenferro-tensor-core`, `tenferro-tensor`, `tenferro-runtime`, and extension
  crates own overlapping error variants for shape, rank, axis, and dtype
  validation.
- Several crate-boundary conversions call `to_string()` and place the result in
  a generic `InvalidGraphBuild`, `ContractionError`, `InvalidConfig`, or
  `BackendFailure` variant. Callers can no longer distinguish a shape mismatch
  from an unsupported operation or a backend failure.
- Eager operations generally report concrete validation errors immediately,
  while traced operations may report the same failure under a stringly graph
  error or delay it until compilation or execution even when all relevant
  dimensions were already concrete.
- Eager `reshape` returns `Result`, while concrete traced `reshape` currently
  returns `TracedTensor` directly and defers element-count validation.
- `Add<Output = Result<_>>` is legal Rust and preserves fallibility, but it is
  unusual: `a + b + c` does not compose directly because `a + b` produces a
  `Result`. Current online examples do not consistently explain this.
- Public `Result` APIs do not consistently document their concrete failure
  conditions under `# Errors`.

The result is locally reasonable error handling without a stable, workspace-
wide semantic model.

## Design principles

1. Caller-controlled invalid input is a typed error, not a panic.
2. Internal invariant violations may panic only when the invariant and its
   proof boundary are local and documented. Otherwise they are typed internal
   errors.
3. The same semantic validation failure uses the same shared payload in eager,
   traced, backend, and extension paths.
4. Error kind and error phase are separate. `ShapeMismatch` describes what
   failed; graph build, compilation, or execution describes when it was
   detected.
5. Validation happens at the earliest phase with sufficient information.
6. Crate-specific errors remain owned by the crate that defines their
   semantics.
7. Error conversion preserves structured payloads and the `source()` chain.
   Strings are for display, logging, vendor/FFI input, or final serialization,
   not internal classification.
8. Explicit fallible methods are the canonical API. Fallible operators are
   optional notation layered on those methods.
9. Before 1.0, API cleanliness takes priority over compatibility. This change
   is a coordinated breaking migration with no deprecated compatibility layer.

## Goals

- Establish one shared representation for tensor validation failures.
- Let in-workspace and out-of-tree extension crates return the same structured
  shape, rank, axis, dtype, and argument errors.
- Preserve crate ownership of parsing, planning, numerical, backend, graph,
  and extension-specific failures.
- Make eager and traced behavior agree whenever both have the information
  needed to validate.
- Document immediate and deferred failure behavior accurately.
- Enforce `# Errors` and `# Panics` documentation across the workspace and
  separately built extension crates.
- Explain the composition limits and robustness rationale of fallible operator
  notation.

## Non-goals

- A single closed `Error` enum containing every error from every crate.
- Recovering from allocator out-of-memory behavior.
- Turning proven internal invariants into caller-recoverable validation errors.
- Erasing useful einsum, linalg, FFT, XLA, device, or vendor-specific details
  merely to fit a universal payload.
- Making dynamic-shape `AddAssign` fallible; Rust's trait cannot return an
  error.
- Introducing an infallible or panicking dynamic tensor operator for prettier
  notation.
- Preserving pre-change error variants or the current infallible traced
  `reshape` signature through deprecated shims.

## 1. Shared validation ownership

`tenferro-tensor-core` will own the shared tensor validation vocabulary. It is
already the lowest-level crate that owns `DType`, shape/layout metadata, and
most of the relevant validation errors. Using it avoids a new dependency cycle
and removes the current duplication between tensor-core and tensor.

The current broad `tenferro_tensor_core::Error` becomes a deliberately named
`ValidationError`. There is no deprecated `Error` alias: all workspace call
sites migrate in the same change.

Illustrative public shape:

```rust
#[non_exhaustive]
pub enum ValidationError {
    ShapeMismatch(ShapeMismatch),
    ShapeDataLengthMismatch {
        expected: usize,
        actual: usize,
    },
    RankMismatch {
        expected: usize,
        actual: usize,
    },
    AxisOutOfBounds {
        axis: usize,
        rank: usize,
    },
    DTypeMismatch {
        expected: DType,
        actual: DType,
    },
    InvalidArgument(InvalidArgument),
    // Existing layout-validation cases remain structured here.
}

#[non_exhaustive]
pub enum ShapeMismatch {
    IncompatibleShapes {
        lhs: ShapeVec,
        rhs: ShapeVec,
    },
    ExpectedActual {
        expected: ShapeVec,
        actual: ShapeVec,
    },
    ReshapeElementCount {
        from: usize,
        to: usize,
    },
    ContractedDimensions {
        lhs_axis: usize,
        lhs_size: usize,
        rhs_axis: usize,
        rhs_size: usize,
    },
}
```

The final variant inventory should preserve distinct machine-readable facts.
For example, reshape element-count mismatch and binary broadcast mismatch share
the broad kind `ShapeMismatch`, but should not be flattened into an ambiguous
`expected`/`actual` pair or a message string.

Shared payload types expose constructors and read-only accessors. They do not
carry runtime-specific context such as graph phase, backend name, or extension
family. Public enums are `#[non_exhaustive]` so downstream matching includes a
fallback without preventing extension crates from constructing known variants.

### Stable classification

`tenferro-tensor-core` also exposes coarse, copyable classifications used at
crate, dynamic-extension, and language-binding boundaries:

```rust
#[non_exhaustive]
pub enum ValidationKind {
    ShapeMismatch,
    RankMismatch,
    AxisOutOfBounds,
    DTypeMismatch,
    InvalidArgument,
}

#[non_exhaustive]
pub enum ErrorKind {
    Validation(ValidationKind),
    Unsupported,
    NumericalFailure,
    BackendFailure,
    Io,
    RuntimeState,
    Internal,
}

impl ValidationError {
    pub fn kind(&self) -> ValidationKind;
}
```

Callers use the detailed variant when they need fields and `kind()` when they
only need stable control-flow classification. Every public crate-local outer
error exposes `kind() -> ErrorKind`. This gives FFI, bindings, runtime
registries, and generic application code one classification surface without
forcing all crates into one concrete outer enum.

The classification is also a routing policy, not a fallback label:

- typed file, stream, serialization, or dynamic-library I/O sources use the
  `Io` category and remain available through `source()`;
- missing, uninitialized, poisoned, or otherwise invalid executor/cache/device
  state uses `RuntimeState`, preserving a typed source when one exists;
- `BackendFailure` is reserved for vendor/backend status text for which no
  typed source or more specific category exists.

An operation phase such as graph construction, compilation, or execution is
orthogonal to these categories. In particular, a runtime-state or I/O failure
must not be reclassified as a generic backend failure merely because it was
observed while a backend was executing.

## 2. Crate-local outer errors

Each crate retains an outer error that expresses its own domain. Shared
validation is wrapped as a typed source rather than copied or formatted.

Conceptually:

```rust
// tenferro-tensor
pub enum Error {
    Validation {
        op: &'static str,
        #[source]
        source: ValidationError,
    },
    Backend(BackendError),
    MissingValue { slot: usize },
    Internal(InternalError),
}

// tenferro-runtime
pub enum Error {
    Validation {
        op: &'static str,
        phase: ErrorPhase,
        #[source]
        source: ValidationError,
    },
    GraphBuild(GraphBuildError),
    Compile(CompileError),
    Execution(ExecutionError),
    Extension(ExtensionError),
    Internal(InternalError),
}

// tenferro-einsum
pub enum Error {
    Validation {
        op: &'static str,
        #[source]
        source: ValidationError,
    },
    InvalidSubscripts(InvalidSubscripts),
    Planning(PlanningError),
    Numerical(NumericalError),
}
```

These sketches define ownership, not mandatory variant spelling. The
implementation plan will inventory current variants and map each one to the
owning domain without changing this boundary.

Outer errors are not required to remain `Clone`, `Eq`, or `PartialEq` if doing
so would force a typed source to become a string. Tests should match stable
kinds and payload fields rather than require equality for opaque backend or
vendor sources.

### Extension boundary

Extension crates depend directly on `tenferro-tensor-core` when they construct
shared validation payloads. They keep a crate-local outer error for parsing,
planning, and numerical failures.

When an extension is called through its own public API, callers receive its
concrete outer error. When it crosses a type-erased runtime registry boundary:

- shared validation is promoted directly to the runtime/tensor validation
  wrapper;
- other extension errors retain the extension family, a stable broad kind,
  and a boxed typed `source` where the boundary requires erasure;
- callers may use the broad kind without downcasting and may downcast the
  source when extension-specific recovery is required.

The following conversions are prohibited inside the Rust workspace:

```rust
Error::ContractionError(source.to_string())
Error::InvalidGraphBuild { message: source.to_string() }
Error::backend_failure(op, source) // when `source` is already a typed error
```

A vendor API that supplies only an error code and message may still populate a
structured backend error containing that code and message; the design cannot
recover structure that the vendor did not provide.

## 3. Error phase and validation timing

Runtime validation errors carry a phase separate from their semantic payload:

```rust
#[non_exhaustive]
pub enum ErrorPhase {
    GraphBuild,
    Compile,
    Execution,
}
```

An eager tensor error does not need a graph phase; the direct API call is its
context. A runtime error exposes `phase()` where phase is meaningful.

The governing rule is:

> Validate at the earliest phase where the required facts are known, while
> preserving the same validation kind and payload shape at later phases.

Examples:

| Operation and input | Detection | Error |
|---|---|---|
| Eager binary op with incompatible concrete shapes | API call | `ShapeMismatch` |
| Traced binary op with incompatible constant dimensions | Graph build | `ShapeMismatch`, `GraphBuild` |
| Traced binary op with unresolved symbolic dimensions | Constraint recorded | No immediate error |
| Symbolic constraint violated by concrete bindings | Compile or execution, according to where bindings become available | `ShapeMismatch` with that phase |
| Concrete traced reshape with unequal element counts | Graph build | `ShapeMismatch::ReshapeElementCount` |
| Symbolic reshape whose product cannot yet be resolved | Constraint recorded | Same payload category when later disproved |

`InvalidGraphBuild { message: String }` is not a substitute for caller-facing
validation. It is replaced by structured graph-build errors for genuine graph
construction failures and shared validation wrappers for invalid tensor
relationships. Graph corruption that cannot be caused by valid public input is
an internal error or a locally proven invariant panic.

### Traced reshape

Traced reshape becomes fallible and agrees with eager reshape:

```rust
pub fn reshape(&self, shape: impl Into<...>) -> Result<TracedTensor>;
```

- Concrete input and output sizes are checked immediately.
- A partially or fully symbolic relationship records an element-count
  constraint and returns `Ok(TracedTensor)`.
- Compilation or execution reports the same structured shape mismatch if the
  eventual binding violates the constraint.
- The current direct-return signature is removed without a compatibility
  method.

## 4. Fallible methods and operators

Explicit methods are canonical:

```rust
let abc = a.add(&b)?.add(&c)?;
let reshaped = x.reshape([2, 3])?;
```

The existing method names remain verbs such as `add` and `reshape`. A `try_`
prefix is not required when the operation is fundamentally fallible and there
is no infallible peer.

`Add<Output = Result<_>>` remains available as convenience notation and
delegates to the same validation path. It does not panic on shape mismatch.
Documentation must show its composition limitation:

```rust
// Does not compose: `&a + &b` is a Result, not a Tensor.
// let abc = &a + &b + &c;

let ab = (&a + &b)?;
let abc = (&ab + &c)?;
```

The explanation presented to users is explicit:

> Tenferro prioritizes robust error handling over the conciseness of chained
> operator notation. Tensor operators return `Result` because compatibility
> may only be known at runtime.

Consequences that must also be documented:

- Generic numeric bounds such as `T: Add<Output = T>` do not accept these
  dynamic tensors.
- `AddAssign` and similar assignment traits are not implemented for operations
  that may fail, because those traits cannot return `Result`.
- A future statically shape-safe tensor type may provide an infallible operator
  without changing the dynamic tensor contract.

## 5. Panic policy

The public contract is not an absolute claim that no panic can ever occur. It
is the narrower and enforceable rule that caller-controlled invalid input is
not converted into a panic.

- Shape, rank, axis, dtype, configuration, I/O, and numerical convergence
  failures use `Result` when they are part of the callable contract.
- Floating-point division follows the documented numeric semantics; integer or
  operation-specific invalid division must be validated according to its
  contract.
- Allocator out-of-memory behavior is outside this recoverable error model.
- `unwrap`, `expect`, unchecked indexing, and debug-only assertions must not be
  the validation mechanism for public input.
- An internal panic is allowed only near a documented proof that the state is
  unreachable through valid public APIs.

Public functions that intentionally panic on a documented contract violation
must include `# Panics` with the exact precondition.

## 6. Rustdoc requirements

The repository rules gain the following normative requirement:

> Every public function, inherent method, trait method, extension method, and
> operator whose effective output is `Result` must contain an `# Errors`
> section. It must state the concrete failure conditions and public error
> variant or stable kind. Generic text such as “returns an error on failure” is
> insufficient.

For traced APIs, `# Errors` documents errors returned by the call itself. When
the operation can record a constraint that fails later, rustdoc also includes
`# Deferred errors`:

```rust
/// # Errors
///
/// Returns [`ValidationError::ShapeMismatch`] when the element counts are
/// known and incompatible while building the graph.
///
/// # Deferred errors
///
/// If symbolic dimensions prevent validation during graph construction, the
/// same error may be reported while compiling or executing the graph.
```

The repository rules also require:

- a `# Panics` section for every public API with an intentional documented
  panic condition;
- structured error preservation across crate boundaries;
- eager/traced parity for equivalent known inputs;
- documentation of the earliest validation phase and any deferred phase.

Online documentation is updated to include:

- a correct traced `a + b` example using `?` or explicit handling;
- the `a + b + c` limitation and its two-step/method-chain alternatives;
- the robustness-over-conciseness rationale;
- immediate versus deferred traced validation.

## 7. Enforcement and audit

Enforcement has three layers.

### Deterministic lint gate

After existing violations are fixed, CI denies:

```text
clippy::missing_errors_doc
clippy::missing_panics_doc
```

The gate covers the root workspace and every separately built supported
extension manifest. Existing violations are repaired before enabling `deny`;
there is no permanent baseline allowlist.

Clippy checks header presence, not semantic accuracy. Operator implementations
whose associated `Output` resolves to `Result` may also require explicit audit
because the effective return type is indirect.

### Repository-rules semantic review

The repository-rules review audits:

- missing or generic `# Errors` text;
- mismatch between documented and implemented variants/kinds;
- missing `# Deferred errors` for symbolic constraints;
- concrete failures delayed beyond the earliest available phase;
- eager/traced kind or payload disagreement;
- `to_string()` conversions that erase a typed source;
- `BackendFailure(String)` or generic graph errors used for known validation;
- operator examples that omit `Result` handling;
- public panic paths reachable from caller-controlled invalid input.

### Behavioral tests

Tests assert both positive behavior and classification:

| Case | Required assertion |
|---|---|
| Eager concrete mismatch | Immediate shared validation kind and fields |
| Traced concrete mismatch | Graph-build phase and same shared payload |
| Traced symbolic mismatch | Deferred phase and same shared kind |
| Extension validation | Shared payload survives every boundary |
| Extension-local failure | Local type or typed source survives; no string-only conversion |
| Backend/vendor failure | Backend identity and available status code/message retained |
| `source()` traversal | Original structured source remains reachable |
| Rustdoc examples | Examples compile and demonstrate the documented error behavior |

Tests compare stable kinds and relevant payload fields instead of formatting
strings.

## 8. Breaking migration

This is a coordinated pre-1.0 breaking change. Clean API boundaries take
priority over preserving names or variants that encode the wrong model.

No compatibility layer is retained:

- no deprecated alias from `tenferro_tensor_core::Error` to
  `ValidationError`;
- no duplicate legacy runtime/tensor validation variants;
- no stringly fallback for errors that have a structured representation;
- no old direct-return traced `reshape` under another name;
- no long-lived lint allowlist.

Implementation proceeds bottom-up:

1. Inventory every public error producer, conversion, `Result` API, panic
   contract, and separately built extension.
2. Define the shared validation types and classification in
   `tenferro-tensor-core`; migrate its own APIs and tests.
3. Replace duplicated `tenferro-tensor` validation variants with a structured
   wrapper and migrate backend/provider conversions.
4. Add runtime phase-aware validation wrapping; remove string conversions in
   graph construction, shape inference, compilation, and execution.
5. Make traced operations validate known facts early, including fallible
   `reshape`, and add symbolic constraints where validation must be deferred.
6. Migrate AD eager/traced paths and verify parity.
7. Migrate einsum, linalg, FFT, XLA, CPU, GPU, and other extensions while
   retaining domain-specific outer errors.
8. Update all public rustdoc, examples, guides, and downstream workspace call
   sites, including fallible operator examples.
9. Add semantic audit rules and behavioral tests.
10. Enable workspace-wide and extension-wide lint denial only after the full
    tree is clean.

The implementation plan may split these steps into reviewable commits, but the
release must not expose a mixed old/new public model.

## Risks and mitigations

- **Large blast radius.** Error variants and traced reshape are widely used.
  Mitigation: migrate in dependency order and keep the workspace compiling at
  each implementation-plan checkpoint.
- **Over-generalized shared payloads.** A universal shape payload could discard
  einsum axes or reshape counts. Mitigation: use distinct structured
  `ShapeMismatch` forms with one coarse kind.
- **Dependency inversion.** Putting runtime phases, runtime context, or
  runtime-specific payloads into tensor-core would pollute the metadata layer.
  Mitigation: tensor-core owns shared classifications and validation facts;
  runtime owns graph phases and runtime context.
- **Type erasure at dynamic extension boundaries.** A registry cannot expose a
  closed enum for unknown out-of-tree extension errors. Mitigation: promote
  common validation directly and otherwise retain stable kind plus boxed typed
  source.
- **False confidence from Clippy.** A heading may exist but be inaccurate.
  Mitigation: pair lints with semantic repository review and behavior tests.
- **Source-breaking migration.** Downstream matches and traced reshape call
  sites must change. Mitigation: make the break deliberate before 1.0, publish
  migration notes, and avoid a confusing dual API.

## Success criteria

- One shared tensor validation vocabulary is owned by
  `tenferro-tensor-core` and constructible by extension crates.
- No known structured Rust error is converted to a string at an internal crate
  boundary.
- Equivalent eager and traced failures expose the same validation kind and
  relevant payload fields.
- Concrete traced validation fails at graph build; symbolic validation fails
  only when bindings make the contradiction knowable.
- Traced reshape is fallible and records deferred element-count constraints
  when necessary.
- Explicit methods are canonical; fallible operators remain documented sugar.
- Online docs show correct `a + b`, `a + b + c`, and deferred-error examples.
- Every public fallible API has a concrete `# Errors` section, and every public
  intentional panic contract has `# Panics`.
- Clippy, semantic repository audit, doctests, and behavioral tests enforce the
  model across the workspace and supported extensions.
- No compatibility layer or deprecated legacy error surface remains.
