# ExtensionOp Contract

**Date:** 2026-04-19
**Parent:** [`../index.md`](../index.md)
**Related:** [`ad-contract.md`](ad-contract.md), [`primitive-catalog.md`](primitive-catalog.md), [`backend-contract.md`](backend-contract.md), [`tensor-semantics.md`](tensor-semantics.md), [`../design/dynamic-symbolic-shapes.md`](../design/dynamic-symbolic-shapes.md)

---

## 1. Scope and Status

This document is the normative specification for the `ExtensionOp` contract
implemented by the traced `StdTensorOp` graph. `ExtensionOp` enables out-of-tree
extension primitives (e.g. `FusedTropicalDotGeneral`) to participate in the
graph as `StdTensorOp::Extension(Arc<dyn ExtensionOp>)` variants without
modifying the core workspace.

Status: **normative**. Implementations must preserve this contract unless this
document is revised first.

The vocabulary (`MUST` / `SHOULD` / `MAY`) follows RFC 2119 conventions used
by the rest of `docs/spec/`. Unless explicitly marked *informative* (e.g. the
worked example in Section 14), every statement in this document is part of
the contract.

Where this document fixes a precise Rust signature in a code block, that
signature is part of the contract. Implementations may refine names and module
paths to match the surrounding codebase, but they may not change the semantic
shape (arguments, return types, blanket bounds) of any signature fixed here.

---

## 2. Relation to existing docs

This spec extends the three normative contracts that already exist in
`docs/spec/`:

- [`ad-contract.md`](ad-contract.md) owns the `PrimitiveOp` trait. This
  document extends it by specifying how an `ExtensionOp` participates in
  AD **without** itself implementing `PrimitiveOp`: the dispatcher in
  `tenferro-internal-ops/src/ad/mod.rs` routes `StdTensorOp::Extension(op)` to a
  registered rule for `op.family_id()`. The rule emits tangents and
  cotangents expressed in the core `StdTensorOp` vocabulary, or in other
  registered extension families. The ad-contract closure rule (emit only
  ops that implement `PrimitiveOp`) is preserved at the carrier level
  because the graph still contains only `StdTensorOp` values.
- [`primitive-catalog.md`](primitive-catalog.md) owns the core op
  vocabulary. `ExtensionOp` does **not** add to that vocabulary; it is a
  single carrier variant `StdTensorOp::Extension(Arc<dyn ExtensionOp>)`.
  Per-op semantics for extension payloads live in the implementer's
  documentation, not in primitive-catalog.md.
- [`backend-contract.md`](backend-contract.md) owns the execution IR and
  dispatch categories. This document defers the compiled-execution story
  to that contract: extensions enter the execution pipeline as a single
  instruction category (see Section 8 for the split).

Where this spec and the above documents disagree, this spec wins for
`ExtensionOp`-specific behaviour; the three base contracts win for
everything else.

This document does **not** own:

- the concrete registry data structure (see Section 15)
- cross-process graph serialization format (Section 11, Section 15)
- per-extension semantics (those live with each extension crate)

---

## 3. Why a spec is needed before implementation

A raw `StdTensorOp::Extension(Arc<dyn ExtensionOp>)` carrier is simple to add
but underspecified on its own. This specification answers six questions that
must be fixed for graph interning, AD caching, serialization boundaries, and
runtime dispatch to remain deterministic:

1. **What makes two extension ops equal?** Answered normatively in
   Section 5.
2. **How are extension parameters hashed?** Answered normatively in
   Section 5.
3. **How does serialization identify the op family?** Answered normatively
   in Section 5 (family_id) and Section 11 (versioning).
4. **How does the runtime decide whether two graph nodes are the same
   operation?** Answered normatively in Sections 4–5 (equality and
   hashing across `Arc<dyn ExtensionOp>`).
5. **How do caches stay stable across processes or versions?** Answered
   normatively in Section 11 (serialization compatibility) and
   Section 12 (failure modes).
6. **Where may extension runtime caches live?** Answered normatively in
   Section 4 under runtime cache ownership.

This document is the normative answer for all five.

---

## 4. Trait shape: `ExtensionOp`

The `ExtensionOp` trait is the Rust trait that every extension
implementation MUST satisfy. The core op enum carries one extension variant:

```rust
// In tenferro-internal-ops/src/std_tensor_op.rs:
pub enum StdTensorOp {
    // ... existing variants ...
    Extension(std::sync::Arc<dyn ExtensionOp>),
}
```

The `ExtensionOp` trait itself is the following contract. All methods are
required unless explicitly marked `provided`.

```rust
/// An out-of-tree operation that participates in the `StdTensorOp` graph
/// via the `StdTensorOp::Extension(Arc<dyn ExtensionOp>)` carrier.
///
/// Every method on this trait is part of the ExtensionOp contract. See
/// `docs/spec/extension-op.md` for normative requirements.
pub trait ExtensionOp: std::fmt::Debug + Send + Sync + 'static {
    // ----- Identity, hashing, equality (Section 5) -----

    /// Stable, process-independent family identifier.
    ///
    /// MUST be unique per extension *family* (payload schema), not per
    /// *instance*. MUST NOT change when the payload changes. MUST be
    /// chosen from the reserved-namespace format specified in Section 5.
    fn family_id(&self) -> &'static str;

    /// Hash the payload (everything except `family_id`).
    ///
    /// The carrier's `Hash` impl combines `family_id` (hashed as a byte
    /// string) with this method. Implementations MUST be pure and
    /// deterministic across calls on the same value.
    fn payload_hash(&self, hasher: &mut dyn std::hash::Hasher);

    /// Structural equality against another extension value.
    ///
    /// The carrier's `PartialEq` / `Eq` impl first compares
    /// `family_id`s; if they match it calls `payload_eq`. Implementations
    /// MUST return `true` iff the payloads are semantically equal and
    /// `other.family_id() == self.family_id()`.
    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool;

    /// Produce a clone of this extension behind an `Arc`.
    ///
    /// The carrier's `Clone` impl delegates to this method via a
    /// cheap `Arc::clone`; this method exists only for the rare case
    /// where a deep clone is actually needed (registry bootstrap or
    /// cross-graph duplication). Implementations SHOULD return
    /// `Arc::new(self.clone_inner())` where `clone_inner` is a regular
    /// `Clone` on the concrete type.
    fn clone_arc(&self) -> std::sync::Arc<dyn ExtensionOp>;

    // ----- Arity (Section 6) -----

    /// Number of primal inputs. MUST be consistent with
    /// `infer_output_shapes` (same input count).
    fn n_inputs(&self) -> usize;

    /// Number of outputs. MUST match the length of the returned
    /// `Vec` from `infer_output_shapes`.
    fn n_outputs(&self) -> usize;

    // ----- Shape and dtype inference (Section 7) -----

    /// Infer output dtype and shape for each output slot.
    ///
    /// Returned vector length MUST equal `self.n_outputs()`. Shapes use
    /// graph-global `SymDim` (symbolic dimensions), consistent with
    /// `TensorMeta::shape` in `tenferro-internal-ops/src/ad/context.rs`. Input
    /// metadata is given as slices of `SymDim` / `DType`; see Section 7
    /// for the detailed invariants.
    fn infer_output_meta(
        &self,
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> Vec<(DType, Vec<SymDim>)>;

    // ----- Forward execution dispatch (Section 8) -----

    /// Eager forward execution.
    ///
    /// Called from the runtime extension dispatcher when the eager path
    /// encounters an `Extension` variant. Input tensors are on the device
    /// the caller already arranged. Returned tensors MUST have shapes that
    /// match `infer_output_meta` and MUST be placed on a device the caller
    /// can consume (per `backend-contract.md`'s device-transfer policy,
    /// there is no implicit cross-device transfer).
    fn eager_execute(
        &self,
        inputs: &[&tenferro_tensor::Tensor],
    ) -> tenferro_tensor::Result<Vec<tenferro_tensor::Tensor>>;

    // AD rules are registered separately; see Section 10.
}
```

### Carrier traits: how `StdTensorOp::Extension` gets `Clone + Hash + Eq`

The core op enum requires `Clone + Debug + Hash + Eq + Send + Sync +
'static` (per `computegraph::GraphOp`). `Arc<dyn ExtensionOp>` satisfies
these through delegation:

- `Clone` via `Arc::clone` (cheap, reference-counted). No deep clone
  happens on the fast path.
- `Hash` via the extension variant implementation:

  ```rust
  impl Hash for StdTensorOp {
      fn hash<H: Hasher>(&self, state: &mut H) {
          std::mem::discriminant(self).hash(state);
          match self {
              // ... existing arms ...
              Self::Extension(op) => {
                  op.family_id().hash(state);
                  op.payload_hash(&mut DynHasherProxy::new(state));
              }
          }
      }
  }
  ```

  The `DynHasherProxy` wraps a generic `H: Hasher` behind `&mut dyn Hasher`
  to satisfy `ExtensionOp::payload_hash`'s object-safe signature.

- `PartialEq` / `Eq` via a family-id shortcut then `payload_eq`:

  ```rust
  impl PartialEq for StdTensorOp {
      fn eq(&self, other: &Self) -> bool {
          match (self, other) {
              // ... existing arms ...
              (Self::Extension(a), Self::Extension(b)) => {
                  a.family_id() == b.family_id() && a.payload_eq(&**b)
              }
              _ => false,
          }
      }
  }
  ```

This design parallels how `std::any::Any`-style downcasts work in Rust:
identity and equality are carried by a type-erased handle, and the
concrete type is recovered only when the implementer explicitly chooses
to compare.

### Runtime cache ownership

Extension payload identity and extension runtime caching are separate
contracts.

An `ExtensionOp` payload MUST describe operation semantics. Payload hashing and
equality MUST NOT depend on cache warmth, vendor plan handles, allocated
workspace, stream state, or other mutable runtime state. If such state changes
the mathematical result or device placement contract, it is not a cache and
MUST be represented as semantic payload instead.

Extension crates that need plan caches or vendor handles MUST own them in an
explicit runtime/cache object outside the semantic payload contract. That owner
MUST be bounded by default and SHOULD expose clear, capacity, and stats APIs
consistent with the workspace cache ownership rules. The op payload MAY hold an
`Arc` to the extension-owned cache object only when that cache is a performance
detail and two equal payloads remain interchangeable when their cache handles
differ.

There is no monolithic core runtime owner for arbitrary extension cache state.
`EagerRuntime`, `GraphCompiler`, and `GraphExecutor` own explicit generic
extension cache stores. Extension crates may use those stores only through the
owning compiler/runtime/executor context; they MUST NOT hide long-lived caches
inside semantic payloads, process globals, or thread-local state.

### Rationale

Identity / hash / eq MUST live on the trait rather than on a concrete
type because a `Box<dyn ExtensionOp>` has no visible payload type from
the carrier's perspective. If these methods were not on the trait, the
carrier could not implement `Hash` or `Eq`, which would break computegraph's
op interner, AD rule caching, and structural graph comparison.

### Failure signature

- An implementer that does not supply a stable `family_id` breaks op
  interning across graph builds. Symptom: every call to `builder.add_op`
  with the "same" extension creates a fresh node, exploding memory and
  defeating CSE.
- An implementer whose `payload_hash` disagrees with `payload_eq`
  breaks `HashMap`-keyed caches. Symptom: AD caches return wrong
  cotangents or miss.
- An implementer whose registered AD rule emits an `Extension` whose family
  has no registered AD rule gets `ADRuleError::Unsupported` on the next AD
  pass.

---

## 5. Identity, hashing, equality

### `family_id` format (normative)

Every `family_id` MUST follow the namespaced format:

```
"<crate-name>.<op-name>.v<major>"
```

- `<crate-name>` MUST be the publishing crate's canonical name as it
  appears on crates.io or in the workspace `Cargo.toml` (hyphens
  permitted, no spaces).
- `<op-name>` SHOULD be a stable ASCII identifier (snake_case permitted)
  that uniquely identifies the op family within that crate.
- `<major>` is the family-version integer. It MUST be bumped on any
  breaking change to the payload schema, shape-inference rule, AD rule,
  or numerical semantics. It MUST NOT be bumped for pure refactors that
  preserve all contract-visible behaviour.

Example:

```
"tenferro-ext-tropical.fused_dot_general.v1"
```

Extension crates MAY use the `ExtensionFamilyId` derive macro re-exported by
`tenferro_runtime::extension` / `tenferro_ops` to generate this string as an
inherent `FAMILY_ID` constant:

```rust
use tenferro_ops::ExtensionFamilyId;

#[derive(ExtensionFamilyId)]
#[tenferro_extension(namespace = "my-crate", name = "fft", version = 1)]
struct FftOp;

assert_eq!(FftOp::FAMILY_ID, "my-crate.fft.v1");
```

### Uniqueness

`family_id` uniqueness is enforced at registration (see Section 9). The
registry MUST reject a second registration under an already-registered
`family_id`. This makes collisions a contract violation surfaced at
registration time, not a silent equality bug.

### Hashing derivation

The op's overall hash, as seen by `StdTensorOp::hash`, MUST include:

1. the carrier discriminant (distinguishing `Extension` from other
   `StdTensorOp` variants);
2. the bytes of `family_id` (e.g. `family_id().as_bytes().hash(state)`
   or an equivalent fixed-endian encoding);
3. the payload hash produced by `payload_hash`.

`payload_hash` implementations SHOULD hash all fields that participate in
`payload_eq`, and MUST NOT include transient state (allocation addresses,
Mutex poison flags, atomically updated counters, etc.).

### Equality shortcut

The carrier's `PartialEq` MUST short-circuit on `family_id` inequality
(different families are never equal, regardless of payload resemblance).
This guarantees that two extensions with structurally identical payloads
but different families are not accidentally unified by the op interner.

### Worked example

For `FusedTropicalDotGeneral` in `tenferro-ext-tropical`:

- `family_id = "tenferro-ext-tropical.fused_dot_general.v1"`
- payload = `DotGeneralConfig` (from `tenferro-internal-tensor`)
- `payload_hash` hashes the four `Vec<usize>` fields of `DotGeneralConfig`
  in the order they are declared, via `DotGeneralConfig: Hash`
- `payload_eq` downcasts `other` to the concrete type and defers to
  `PartialEq` on `DotGeneralConfig`

Downcasting across the trait boundary SHOULD use the standard pattern:

```rust
fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
    // Family id is the invariant; we only reach here after family match.
    match (other as &dyn std::any::Any).downcast_ref::<FusedTropicalDotGeneral>() {
        Some(that) => self.config == that.config,
        None => false,
    }
}
```

Note: for `(other as &dyn Any).downcast_ref` to work, `ExtensionOp`
implementations MUST add `std::any::Any` as a supertrait bound or carry
a `fn as_any(&self) -> &dyn Any` helper method. The implementation SHOULD
choose one convention and document it in the trait definition comment; the
default choice is to add `Any` via a method-based helper to keep
`ExtensionOp` object-safe.

---

## 6. Arity and I/O shape

### Fixed-arity contract

Every `ExtensionOp` MUST declare fixed `n_inputs` and `n_outputs` values
whose return is independent of runtime input sizes. This aligns with the
`StdTensorOp` arity dispatcher in
`tenferro-internal-ops/src/std_tensor_op.rs` (e.g. `n_inputs` for `DotGeneral` is
always `2`).

The dispatcher integration in `tenferro-internal-ops/src/ad/mod.rs` MUST treat
`StdTensorOp::Extension(op)` as having `op.n_inputs()` inputs and
`op.n_outputs()` outputs. Validation:

- The number of `primal_in` keys passed to `linearize` MUST equal
  `op.n_inputs()`.
- The number of `tangent_in` entries MUST equal `op.n_inputs()`.
- The number of `primal_out` keys MUST equal `op.n_outputs()`.
- The returned tangent-output vector from `linearize` MUST have length
  `op.n_outputs()`.
- The returned cotangent-input vector from `transpose_rule` MUST have
  length `op.n_inputs()`.

### Variable-arity extensions (discouraged)

Extensions with variable arity (for example, a variadic `Concatenate`
analogue) are discouraged because they force the dispatcher to
re-derive arity per instance. If such an op is unavoidable, the
implementer MUST:

1. Store the arity in the payload (so `n_inputs` reads it from `self`,
   not from the arguments).
2. Document in the extension's own doc comment that the arity is
   dynamic, together with the payload field that determines it.
3. Preserve the invariant that for a given `Arc<dyn ExtensionOp>`
   value, `n_inputs()` returns the same value every time.

Variable-arity extensions remain outside `StdTensorOp`'s own variable-arity
branch. Core variants with dynamic arity, such as `StdTensorOp::Concatenate`,
store the arity in their payload and are handled by the core enum directly.

### Failure signature

- `n_inputs` disagreeing with `linearize`'s `primal_in` length in the
  dispatcher causes `Error::InvalidConfig` (see Section 12).
- `n_outputs` disagreeing with `eager_execute`'s returned vector
  length causes `Error::InvalidConfig`.

---

## 7. Shape and dtype inference

### Signature

```rust
fn infer_output_meta(
    &self,
    input_dtypes: &[DType],
    input_shapes: &[&[SymDim]],
) -> Vec<(DType, Vec<SymDim>)>;
```

This method's responsibility mirrors
`tenferro-runtime/src/shape_infer.rs::infer_output_dtype` and
`infer_output_shapes` for core ops, packaged as a single method per
extension.

### Contract

- `input_dtypes.len()` and `input_shapes.len()` MUST both equal
  `self.n_inputs()`. Callers (`compile_std_to_exec`, eager execution)
  guarantee this.
- The returned vector MUST have length `self.n_outputs()`.
- Each `(dtype, shape)` pair gives the inferred dtype and symbolic shape
  for the corresponding output slot.
- Shapes are expressed as `Vec<SymDim>` to match
  `TensorMeta::shape` (see `tenferro-internal-ops/src/ad/context.rs:49-109`).
  Concrete and symbolic inputs use the same representation:
  `SymDim::from(usize)` for concrete extents, symbolic placeholders for
  unknown-at-build-time dimensions.
- If the implementer needs dimension arithmetic (e.g. output dim =
  `lhs_m * rhs_n`), it MUST use the `SymDim` arithmetic API. Collapsing
  an unknown symbolic input to `0` or panicking is a contract violation.

### Symbolic-shape interaction

Per [`../design/dynamic-symbolic-shapes.md`](../design/dynamic-symbolic-shapes.md),
every extension's `infer_output_meta` MUST be **total** over both concrete and
symbolic inputs. Total means:

- The method returns without panicking for any `input_shapes` that would
  also be accepted by the ambient core ops this extension composes with.
- Where the output dimension is symbolic, the returned `SymDim` explicitly
  represents the symbolic expression rather than silently collapsing to
  a constant.

### Failure signature

- An implementer that returns the wrong number of outputs causes
  `compile_std_to_exec` to panic when assigning output slot metadata
  (the panic already exists for core ops; see
  `tenferro-runtime/src/compiler/mod.rs`). The same panic applies to
  extensions.
- An implementer that panics on valid symbolic inputs surfaces as a
  hard crash in symbolic-shape composition tests. This is a contract violation.

---

## 8. Forward execution dispatch

Tenferro has two forward-execution routes; extensions participate in
both, with a normatively-split responsibility.

### Eager path

The eager path runs through `EagerRuntime` and the runtime extension
dispatcher in `tenferro-runtime/src/extension_runtime.rs`.
When execution is attached to an `EagerRuntime` and a runtime is registered for
the extension family, the implementation MUST route `StdTensorOp::Extension(op)`
through that runtime's `ExtensionExecutor`:

```rust
// Conceptual:
StdTensorOp::Extension(ext) => extension_executor.execute(backend, ext, inputs)?,
```

The eager path MUST NOT open a backend execution session for extension
ops before handing control to the extension runtime; the extension runtime owns
its execution model and receives the backend plus its runtime-owned cache store.
The context-free `ExtensionOp::eager_execute` method is a compatibility
fallback for direct execution helpers and for unregistered eager extension
families. Built-in extensions with long-lived caches MUST register a runtime so
`EagerRuntime`-attached execution uses the runtime-owned cache store.

### Compiled path

The compiled path runs through
`tenferro-runtime/src/compiler/mod.rs::compile_std_to_exec` and
`tenferro-runtime/src/exec.rs`. The compiled path MUST include:

1. An `ExecOp::Extension(Arc<dyn ExtensionOp>)` variant (or an
   equivalent carrier) in the execution IR, mirroring the `StdTensorOp`
   variant.
2. Shape / dtype lowering in `compile_std_to_exec` that calls
   `op.infer_output_meta(...)` to populate
   `ExecInstruction::dtype` and `ExecInstruction::output_shapes`.
3. An `execute_extension_op` dispatcher in `tenferro-runtime/src/exec.rs` that,
   at runtime, calls the registered `ExtensionExecutor<B>`.
4. A single-instruction-boundary category for extensions in
   `tenferro-runtime/src/segment.rs` (similar to `DotGeneral` / `NaryEinsum`).
   Extensions MUST NOT participate in elementwise fusion planning
   because their fusion semantics are implementer-defined.

### Responsibility split (normative)

- **`compile_std_to_exec`** is responsible for: lowering the
  `StdTensorOp::Extension` variant to `ExecOp::Extension`, calling
  `infer_output_meta` for metadata population, and assigning
  `last_use` markers. It does not invoke backend kernels.
- **`eager_exec` / `eager_emitter`** are responsible for: resolving
  inputs from the emitter's tensor cache and calling the runtime-owned
  extension executor when one is available.
- **Extension runtime (`ExtensionRuntime<B>`)** is responsible for: actual
  forward computation, backend use, runtime cache entries, and device placement
  of outputs. The core pipeline MUST NOT second-guess these choices.

### Rationale

Using one `ExtensionRuntime<B>` contract for eager and compiled execution keeps
runtime caches tied to explicit owners (`EagerRuntime` or `GraphExecutor`) and
avoids hidden thread-local or process-global state in extension crates.

### Failure signature

- An extension whose `eager_execute` errors surfaces as
  `Error::BackendFailure { op, message }` with `op = "extension"` (or
  a similar constant) and the `family_id` in `message`; see Section 12.
- If a backend lacks a capability the extension needs (e.g.
  cuTENSOR unavailable on CPU-only builds), the extension SHOULD
  produce a descriptive `Error::BackendFailure` rather than panicking;
  the caller decides whether to fall back.

---

## 9. Operation construction and AD rule registration

### Operation construction

Extension op payloads are not registered in a process-global factory registry.
The frontend carries the concrete payload directly as
`StdTensorOp::Extension(Arc<dyn ExtensionOp>)`; extension crates should expose
small public wrapper functions that construct that `Arc` and call
`tenferro_runtime::extension::apply` or `tenferro_ad::extension::apply_eager`.

The removed factory registry (`ExtensionFactory` / `register_extension`) MUST
NOT be reintroduced as a prerequisite for ordinary graph construction. If a
future serialization format needs cross-process reconstruction by
`family_id`, that format must define an explicit serialization registry rather
than overloading runtime graph construction.

### AD rule registry

Only AD rules are process-global. The implementation uses a process-local
`OnceLock<RwLock<HashMap<&'static str, Arc<dyn ExtensionAdRule>>>>`, keyed by
`family_id`. New extension crates SHOULD register a ChainRules-style rule with
`register_extension_chain_rule`; advanced code MAY use the low-level
`register_extension_rule`.

Double-registration under the same `family_id` MUST be rejected with
`RegistrationError::DuplicateRule { family_id }`.

`RegistrationError`:

```rust
#[derive(Debug, thiserror::Error)]
pub enum RegistrationError {
    #[error("AD rule for family_id {family_id:?} already registered")]
    DuplicateRule { family_id: &'static str },
    #[error("family_id {family_id:?} does not match the namespaced format")]
    MalformedFamilyId { family_id: &'static str },
}
```

### Lookup

The public API MUST expose a rule lookup function:

```rust
pub fn lookup_extension_rule(family_id: &str) -> Option<std::sync::Arc<dyn ExtensionAdRule>>;
```

Lookup MUST NOT panic on a missing `family_id`. Callers decide how to
handle absence (see Section 12).

### Thread safety

The registry MUST be safe to read from any thread. Writes happen only
during initialization (the `OnceLock` wrapper permits exactly this).
Concurrent readers use the `RwLock`; writers MUST complete before any
graph-building or execution work begins on the `family_id` they added.

### Failure signature

- Registering two AD rules with the same `family_id` returns
  `RegistrationError::DuplicateRule`.
- Looking up an unregistered AD rule `family_id` returns `None`.
- Running a graph that references an extension op with no registered AD rule is
  valid for forward execution. AD through that op returns
  `ADRuleError::Unsupported` with the `family_id` and rule kind.

---

## 10. AD API surface

### Preferred ChainRules-style API

Extension AD is registered independently from the primal op. New
extension crates SHOULD use `register_extension_chain_rule` and implement
`ExtensionChainRule`:

```rust
pub trait ExtensionChainRule: Debug + Send + Sync + 'static {
    fn family_id(&self) -> &'static str;

    fn frule(&self, op: &dyn ExtensionOp, cx: &mut FruleBuilder<'_>) -> ADRuleResult<()>;

    fn rrule(&self, op: &dyn ExtensionOp, cx: &mut RRuleBuilder<'_>) -> ADRuleResult<()>;
}
```

`FruleBuilder` exposes primal inputs and input tangents through
`primal(i)` and `tangent(i)`, and records output tangents through
`set_output_tangent(i, value)`. `RRuleBuilder` exposes primal inputs and
output cotangents through `primal(i)` and `cotangent(i)`, and records input
cotangents through `set_input_cotangent(i, value)`.

The builders may emit any core `StdTensorOp` through `emit`, convenience
helpers such as `add`, `mul`, `reduce_sum_all`, and registered extension ops
through `apply_extension`. They intentionally hide `LocalValId`; extension
authors pass opaque `AdValue` handles between builder methods.

### Low-level adapter API

The ChainRules-style API is adapted to the lower-level
`register_extension_rule(Arc<dyn ExtensionAdRule>)` registry. Advanced code may
implement `ExtensionAdRule` directly when it needs exact control over
`FragmentBuilder`, `OpEmitter`, `LocalValId`, or `ValRef`. Rule signatures
mirror `PrimitiveOp::try_linearize` and `PrimitiveOp::try_transpose_rule` and
return `ADRuleResult<_>` so missing rules can propagate without panic:

```rust
pub trait ExtensionAdRule: Debug + Send + Sync + 'static {
    fn family_id(&self) -> &'static str;

    fn linearize(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut FragmentBuilder<StdTensorOp>,
        primal_in: &[GlobalValKey<StdTensorOp>],
        primal_out: &[GlobalValKey<StdTensorOp>],
        tangent_in: &[Option<LocalValId>],
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>>;

    fn transpose_rule(
        &self,
        op: &dyn ExtensionOp,
        emitter: &mut dyn OpEmitter<StdTensorOp>,
        cotangent_out: &[Option<LocalValId>],
        inputs: &[ValRef<StdTensorOp>],
        mode: &OpMode,
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>>;
}
```

In both APIs, the `op` argument is the concrete extension payload as a trait
object. Rules that need payload parameters should downcast via `op.as_any()`.

### AD closure

`frule`, `rrule`, `linearize`, and `transpose_rule` may emit core
`StdTensorOp` values and `StdTensorOp::Extension` values. Emitted extension
families MUST have their own registered `ExtensionChainRule` or
`ExtensionAdRule` before a subsequent AD pass reaches them.
This keeps out-of-tree operations in the same compute graph while preserving
the `PrimitiveOp` closure invariant at the `StdTensorOp` carrier level.

### `ShapeGuardContext` interaction

Extension AD rules MUST use `ctx.shape_of(val)`, `ctx.dtype_of(val)`,
and `ctx.metadata_of(val)` to query input metadata, exactly like the
core AD rules (see `tenferro-internal-ops/src/ad/linalg.rs` for a reference
implementation). They MUST NOT reach around the context to fetch
metadata from elsewhere.

Guards recorded through `ctx` are part of the cache-invalidation
contract; implementers that compare symbolic dimensions via
`resolve_and_guard`-like helpers are responsible for recording the
comparisons.

### Deferred zero-tangent policy

Extensions MUST NOT materialise zero cotangents for symbolic-shape inputs at
linearize time. A tangent slot that is inactive MUST be represented as `None`
in both `tangent_in` and the returned tangent-output vector. Zero synthesis
happens at the `GraphExecutor::run_with_inputs` boundary, not inside the
extension's AD rules.

### Failure signature

- Dispatcher reaching a `StdTensorOp::Extension` variant for an
  `family_id` with no registered `ExtensionChainRule` or `ExtensionAdRule` returns
  `ADRuleError::Unsupported` with the family ID and rule kind.

---

## 11. Serialization compatibility

### Scope

This document does not mandate a cross-process graph serialization format;
that is an Open Question (Section 15). However, any implementation
that does serialize graphs containing `StdTensorOp::Extension` nodes
MUST respect the following invariants.

### Family-id versioning

The `family_id` string is the on-wire identity of an extension. A
serializer MUST write `family_id` verbatim (no remapping, no
abbreviation). A deserializer MUST reject any family_id that violates
the namespaced format in Section 5 before attempting lookup.

Per Section 5, a major-version change in the `family_id` indicates a
breaking payload / semantics change. A future deserializer MUST use an
explicit serialization registry for payload reconstruction and MUST refuse
to load a graph whose on-wire `family_id` is unsupported by that registry.
The refusal MUST produce `Error::Unsupported` carrying the on-wire
`family_id`.

### Cross-process policy

In cross-process scenarios (e.g. a serialized graph produced on one
machine and loaded on another):

- Consumers lacking the producer's extension family MUST fail loudly
  with `Error::Unsupported`, unless the caller opts into a
  `skip_missing_extensions=true` mode (which, if implemented, replaces
  the extension with an error-producing placeholder rather than
  silently dropping it).
- Consumers whose registered version is behind the producer's version
  MUST also fail with `Error::Unsupported` and include both versions
  in the error message.

### In-process stability

Within a single process, the `family_id` uniqueness invariant
(Section 5, Section 9) is what keeps op interning and AD caches stable.
Serialization adds no new in-process constraints.

### Failure signature

- On-wire `family_id` that is absent from the consumer's registry:
  `Error::Unsupported { op: "extension", message: "<family_id>: not registered" }`.
- On-wire `family_id` whose version is newer than registered:
  `Error::Unsupported { op: "extension", message: "<family_id>: version mismatch, graph has vN, runtime has vM" }`.

---

## 12. Failure modes

Every failure mode below is normative. Implementations MUST surface exactly
these error types / behaviours in the listed scenarios.

| Scenario | Required behaviour |
|---|---|
| Extension runtime execution returns `Err` | Propagate to caller as `Error::BackendFailure { op: "extension", message }` with `family_id` included in `message`. MUST NOT retry, MUST NOT swallow. |
| Backend lacks a capability the extension needs | The extension runtime SHOULD return `Error::BackendFailure` with a descriptive message that includes `family_id` and the missing capability name. The core pipeline MUST NOT fall back to a different backend. |
| Graph references an unregistered `family_id` at eager or graph runtime execution time | Return a backend/config error with `family_id` and registration guidance. |
| Graph references an unregistered `family_id` at compile time | Return `Error::Unsupported` from `compile_std_to_exec`. |
| AD rules (`frule` / `rrule`, or low-level `linearize` / `transpose_rule`) encounter an `Extension` with no registered AD rule | Return `ADRuleError::Unsupported` with `family_id` and rule kind; traced `grad` / eager `backward` propagate it through the public `Error` type re-exported by the owning surface crate. |
| Hash collision on `family_id` (second registration attempt) | Registry MUST reject with `RegistrationError::Duplicate`. |
| Arity mismatch: `n_inputs()` disagrees with the `primal_in.len()` the dispatcher passed | `Error::InvalidConfig { op: "extension", message: "family_id=<id>: expected N inputs, got M" }`. |
| Output shape disagrees with `infer_output_meta` result length | `Error::InvalidConfig` with `family_id` and the mismatched counts. |
| Extension runtime returns a tensor on the wrong device | Propagate to the caller as `Error::BackendFailure` (the core pipeline does not re-locate tensors). |
| Registration with malformed `family_id` | `RegistrationError::MalformedFamilyId`. |

### Constants for `op` field

Where the table specifies `op: "extension"`, that is the recommended
constant. Implementations MAY refine it (e.g. `op: "ExtensionOp::linearize"` vs
`op: "ExtensionOp::eager_execute"`) to give better error messages, as
long as every `Error` value for an extension includes the `family_id`
somewhere in its message or fields.

---

## 13. Legacy-substrate retirement (normative, historical)

### What was retired

The legacy semiring pipeline, specifically:

- `SemiringOp<Alg>` (the parallel graph-level op type)
- `SemiringOpKind`
- `SemiringOps` trait
- `SemiringBackend<Alg>` trait and all CPU / CUDA / CubeCL / ROCm impls
- `compile_semiring_to_exec` (the parallel compile path)
- `eval_semiring_ir`
- the in-tree `tenferro/tests/tropical.rs`

...was retired during extension-substrate cleanup (commit `39f1b60` on
`refactor_ad_v3`), with additional cleanup of ad module renaming (`e1af8e9`),
compile-path isolation (`d134763`), and docs demotion (`0258531`). Equivalent
test coverage for tropical moved to a temporary `tenferro-ext-tropical` proof
of concept in commits `7317268` and `188a278`. That in-tree POC was later
removed during the no-facade crate-boundary cleanup; this section remains a
historical record of the extension contract it exercised.

### What `ExtensionOp` is NOT

`ExtensionOp` is **not** a replacement for `SemiringOp<Alg>`:

- The graph is no longer algebra-parameterized. Tropical and other
  non-standard-arithmetic paths live outside the core graph, either as
  compositions of core primitives or as fused extensions such as
  `FusedTropicalDotGeneral`.
- `ExtensionOp` provides a single variant `StdTensorOp::Extension(Arc<dyn
  ExtensionOp>)` carrying arbitrary payloads. It does not bring back a
  parallel graph vocabulary keyed on an algebra type parameter.
- Eager T-generic tropical execution continues to work through scalar
  newtypes (`MaxPlus<T>`, `MinPlus<T>`, `MaxMul<T>`) driving the
  existing `TypedTensor<T>` kernels. That path is independent of
  `ExtensionOp` and was not affected by the `SemiringBackend` removal.

### Historical record

This section exists so that any future reader encountering references
to `SemiringOp` / `SemiringBackend` in the commit log, older design
docs, or out-of-tree code lands here first. The short form:

> `SemiringOp` / `SemiringBackend` is gone. `ExtensionOp` is a different
> mechanism for a narrower purpose: single-variant carrier for out-of-tree
> fused ops. Tropical lives as core-op composition or as a fused
> `ExtensionOp`, not as a second graph vocabulary.

`ExtensionOp` therefore does **not** re-introduce a `SemiringOp`-shaped layer,
nor does it tie identity to an algebra type parameter. Identity is carried by
`family_id`, per Section 5.

---

## 14. Worked example: `FusedTropicalDotGeneral` (informative)

This section is **informative** (non-normative). It exists to
cross-validate the normative contract against an external consumer. If
the spec above is insufficient to guide a working `FusedTropicalDotGeneral`
implementation, the spec is wrong and MUST be revised.

### Sketch

```rust
// In tenferro-ext-tropical:

use std::sync::Arc;
use tenferro_ops::{ExtensionOp, StdTensorOp};
use tenferro_tensor::DotGeneralConfig;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FusedTropicalDotGeneral {
    pub config: DotGeneralConfig,
}

impl ExtensionOp for FusedTropicalDotGeneral {
    fn family_id(&self) -> &'static str {
        "tenferro-ext-tropical.fused_dot_general.v1"
    }

    fn payload_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        use std::hash::Hash;
        self.config.hash(hasher);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        // See Section 5 on the Any downcast convention.
        (other as &dyn std::any::Any)
            .downcast_ref::<FusedTropicalDotGeneral>()
            .is_some_and(|that| self.config == that.config)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn n_inputs(&self) -> usize { 2 }
    fn n_outputs(&self) -> usize { 1 }

    fn infer_output_meta(
        &self,
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> Vec<(DType, Vec<SymDim>)> {
        // Same shape rule as DotGeneral: lhs_batch + lhs_remaining + rhs_remaining.
        // Dtype promotion follows Standard DotGeneral; tropical semantics do
        // not change dtype.
        todo!("use the same shape rule as StdTensorOp::DotGeneral")
    }

    fn eager_execute(
        &self,
        inputs: &[&tenferro_tensor::Tensor],
    ) -> tenferro_tensor::Result<Vec<tenferro_tensor::Tensor>> {
        // Fused tropical GEMM: op is (max, +) over the contracting axes.
        // Implementation dispatches to a CPU / GPU kernel that also records
        // argmax indices for use in AD.
        todo!("tropical fused GEMM kernel")
    }
}
```

AD registration:

```rust
#[derive(Debug)]
struct FusedTropicalDotGeneralRule;

impl ExtensionChainRule for FusedTropicalDotGeneralRule {
    fn family_id(&self) -> &'static str {
        "tenferro-ext-tropical.fused_dot_general.v1"
    }

    fn frule(&self, op: &dyn ExtensionOp, cx: &mut FruleBuilder<'_>) -> ADRuleResult<()> {
        let op = op.as_any().downcast_ref::<FusedTropicalDotGeneral>().unwrap();
        // Sketch only; the real implementation lives in tenferro-ext-tropical.
        // Emit a tangent graph using core StdTensorOp values.
        todo!("emit the tropical JVP graph for op.config");
    }

    fn rrule(&self, op: &dyn ExtensionOp, cx: &mut RRuleBuilder<'_>) -> ADRuleResult<()> {
        let op = op.as_any().downcast_ref::<FusedTropicalDotGeneral>().unwrap();
        // Sketch only. Scatter or indicator-mask the incoming cotangent
        // back to lhs and rhs cotangents using core StdTensorOp values.
        todo!("emit the tropical VJP graph for op.config");
    }
}
```

Registration:

```rust
// In tenferro-ext-tropical/src/lib.rs:
pub fn register() -> Result<(), RegistrationError> {
    tenferro_ad::extension::register_extension_chain_rule(
        Arc::new(FusedTropicalDotGeneralRule),
    )
}
```

### What this sketch demonstrates

- Identity (`family_id`, `payload_hash`, `payload_eq`) is carried by the
  trait (Section 5).
- Arity is fixed at 2-in / 1-out (Section 6).
- Shape inference mirrors a core op's shape rule (Section 7).
- Eager execution is the extension's own kernel; the core pipeline does
  not know about tropical semantics (Section 8).
- AD emits only core ops (Section 10), preserving ad-contract.md's
  closure rule.
- Registration is explicit (Section 9).

If an external fused op cannot be implemented from this spec, the spec is
insufficient and MUST be revised.

### Historical note — reconciliation with the temporary external implementation

The former in-tree `tenferro-ext-tropical` proof of concept implemented the op
shape described above with two deviations from this informative sketch, both
within the spec's flexibility. That proof of concept is no longer part of the
workspace:

- **Payload**: the actual `FusedTropicalDotGeneralOp` carries a small
  `TropicalKind { MaxPlus, MinPlus }` enum instead of a
  `DotGeneralConfig`. The current external implementation is scoped to rank-2
  inputs with fixed contracting axes, so the full `DotGeneralConfig` is not needed; a
  richer payload is a straightforward later bump to
  `tenferro-ext-tropical.fused_dot_general.v2` (Section 5 versioning).
- **AD emission**: the sketch suggests `Gather` / `Scatter` on saved
  argmax indices. The core op vocabulary intentionally does not
  include an `ArgMax` variant, so the implementation uses the mathematically
  equivalent **indicator-mask**
  construction (the same `Compare(Eq) + Convert + Mul + ReduceSum + Div` pattern
  used by the core `ReduceMax` / `ReduceMin` AD rule in
  `tenferro-internal-ops/src/ad/contraction.rs`). The two are two expressions
  of the same subgradient; only the indicator form is expressible in
  the current core op vocabulary.

Neither deviation weakens the normative contract — identity, arity,
shape inference, forward dispatch, registry, AD closure, serialization
versioning, and failure modes all hold unchanged.

---

## 15. Open questions

The following are explicitly deferred. Future implementations may decide these
without revisiting this document.

1. **Exact AD rule registry data structure.** Section 9 normatively picks
   `OnceLock<RwLock<HashMap<&'static str, Arc<dyn ExtensionAdRule>>>>`.
   A future evidence-driven migration to `linkme`-style distributed
   slices is permitted but out of scope for the current contract.

2. **`no_std` / wasm targets.** The initial implementation
   MAY restrict `ExtensionOp` to `std`-targets. Widening to `no_std`
   (e.g. for embedded or wasm backends) is deferred until a concrete
   consumer appears.

3. **Cross-process graph serialization format.** Section 11 fixes
   required invariants for any future serializer, but does not
   mandate a specific format. Choosing one (e.g. a bincode / StableHLO
   / protobuf encoding) is out of scope for this contract.

4. **Deep-clone semantics for `Arc<dyn ExtensionOp>`.** Section 4's
   `clone_arc` is intended to be rarely invoked. If a future consumer
   needs a principled "split one Arc into two independent Arcs" path
   (e.g. for cross-thread isolation), the concrete semantics of that
   split are open.

5. **Downcast convention.** Section 5 allows either `Any`
   supertrait or `fn as_any(&self) -> &dyn Any`. Implementations pick one
   convention and document it on the trait.

6. **Metrics / observability hooks.** Whether `eager_execute` should
   emit tracing spans (via `tracing` crate or similar) is deferred.
   Extensions MAY emit their own; the core pipeline does not
   instrument extension calls today.

---

## 16. Change log

- 2026-04-19: Initial draft landed in commit `efd91a7` on `refactor_ad_v3`.
- 2026-04-20: Implementation — `ExtensionOp` trait, registry,
  `StdTensorOp::Extension` carrier, and full forward / AD / shape-infer
  / compile / eager wiring landed in commit `2c7e26c` on
  `codex-stage-6` (branched from `efd91a7`). The original public
  `tenferro::extension` facade was later replaced by the no-facade
  `tenferro_runtime::extension` / `tenferro_ad::extension` surfaces.
- 2026-04-20: External tropical self-test — `FusedTropicalDotGeneralOp` landed in
  `tenferro-ext-tropical` on branch `codex-stage-7` (branched from
  `c9266f9`). The fused op and public traced wrappers landed in commit
  `e03ea60`; the AD parity and contract self-tests in commit
  `1d9c343`. Section 14 was updated in the same branch to reconcile its
  informative sketch with the realised implementation (payload is
  `TropicalKind`, AD emits indicator-mask rather than Gather/Scatter —
  the latter requires an `ArgMax` op the core does not ship).
