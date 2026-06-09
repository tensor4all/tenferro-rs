# ExtensionOp Contract

**Date:** 2026-04-19
**Parent:** [`../index.md`](../index.md)
**Related:** [`ad-contract.md`](ad-contract.md), [`primitive-catalog.md`](primitive-catalog.md), [`backend-contract.md`](backend-contract.md), [`tensor-semantics.md`](tensor-semantics.md), [`../design/dynamic-symbolic-shapes.md`](../design/dynamic-symbolic-shapes.md)

---

## 1. Scope and Status

This document is the normative specification for the `ExtensionOp` contract
implemented by the traced `StdTensorOp` graph. `ExtensionOp` enables out-of-tree
extension primitives (e.g. `tenferro-ext-tropical.einsum.v1`) to participate in the
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

- [`ad-contract.md`](ad-contract.md) owns the `Primitive` trait. This
  document extends it by specifying how an `ExtensionOp` participates in
  AD **without** itself implementing `Primitive`: the dispatcher in
  `crates/tenferro-internal-ops/src/ad/mod.rs` routes `StdTensorOp::Extension(op)` to a
  rule for `op.family_id()` in the active `ExtensionRuleSet`. The rule emits
  tangents and cotangents expressed in the core `StdTensorOp` vocabulary, or
  in extension helper families covered by Section 10's AD-closure rules. The
  ad-contract closure rule (add only ops that implement `Primitive`) is
  preserved at the carrier level because the graph still contains only
  `StdTensorOp` values.
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

- compatibility global registry internals
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
// In crates/tenferro-internal-ops/src/std_tensor_op.rs:
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

    /// Upcast this extension to `&dyn Any` for payload downcasting.
    ///
    /// Implementations SHOULD return `self` verbatim.
    fn as_any(&self) -> &dyn std::any::Any;

    // ----- Arity (Section 6) -----

    /// Number of primal inputs. MUST be consistent with
    /// `infer_output_meta` (same input count).
    fn input_count(&self) -> usize;

    /// Number of outputs. MUST match the length of the returned
    /// `Vec` from `infer_output_meta`.
    fn output_count(&self) -> usize;

    // ----- Shape and dtype inference (Section 7) -----

    /// Infer output dtype and shape for each output slot.
    ///
    /// Returned vector length MUST equal `self.output_count()`. Shapes use
    /// graph-global `SymDim` (symbolic dimensions), consistent with
    /// `TensorMeta::shape` in `crates/tenferro-internal-ops/src/ad/context.rs`. Input
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
'static` (per `computegraph::GraphOperation`). `Arc<dyn ExtensionOp>` satisfies
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
  interning across graph builds. Symptom: every call to `builder.add_operation`
  with the "same" extension creates a fresh node, exploding memory and
  defeating CSE.
- An implementer whose `payload_hash` disagrees with `payload_eq`
  breaks `HashMap`-keyed caches. Symptom: AD caches return wrong
  cotangents or miss.
- An implementer whose AD rule emits an `Extension` whose family has no rule in
  the active `ExtensionRuleSet` gets `ADRuleError::Unsupported` on the next AD
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
"tenferro-ext-tropical.einsum.v1"
```

Extension crates MAY use the `ExtensionFamilyId` derive macro re-exported by
`tenferro_runtime::extension` to generate this string as an inherent
`FAMILY_ID` constant:

```rust
use tenferro_runtime::extension::ExtensionFamilyId;

#[derive(ExtensionFamilyId)]
#[tenferro_extension(namespace = "my-crate", name = "fft", version = 1)]
struct FftOp;

assert_eq!(FftOp::FAMILY_ID, "my-crate.fft.v1");
```

### Uniqueness

`family_id` uniqueness is an implementer contract for operation identity.
Extension crates MUST choose a family id that uniquely names one payload schema
and semantic family. The core graph does not globally register op payloads, so
payload family collisions are not caught when a payload is constructed.

Registration surfaces enforce narrower duplicate rules:

- `ExtensionRuleSet` rejects two AD rules with the same `family_id`.
- runtime executor registration is idempotent by `family_id` and keeps the
  first executor.

These checks catch common configuration mistakes, but they do not replace the
requirement that extension authors reserve stable, collision-free family ids.

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

For `TropicalEinsumOp` in `tenferro-ext-tropical`:

- `family_id = "tenferro-ext-tropical.einsum.v1"`
- payload = `TropicalKind` plus parsed `tenferro_einsum::Subscripts`
- `payload_hash` hashes the semiring kind and the parsed input/output label
  lists in a stable order
- `payload_eq` downcasts `other` to the concrete type and defers to
  `PartialEq` on the concrete payload

Downcasting across the trait boundary SHOULD use the standard pattern:

```rust
fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
    // Family id is the invariant; we only reach here after family match.
    match other.as_any().downcast_ref::<TropicalEinsumOp>() {
        Some(that) => self == that,
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

### Non-Tensor arguments and payload identity

Tensor arguments are graph inputs. They may be AD-active and participate in
graph construction through `ValueKey` dependencies.

Non-Tensor arguments are not graph inputs. Extension implementations MUST
capture them in the concrete `ExtensionOp` payload. Examples include axes,
modes, tolerances, normalization choices, and small configuration structs.

Any payload field that affects output values, output metadata, backend
behavior, or AD behavior is semantic payload. Semantic payload MUST participate
in both `payload_hash` and `payload_eq`. Equal payloads MUST be interchangeable
for graph interning, graph equality, AD rule lookup, and compile/runtime cache
identity.

Runtime cache handles, vendor plans, warm cache state, allocation addresses,
random UUIDs, closures, process-local counters, and mutable registries MUST NOT
participate in semantic payload identity. If a large external resource is
needed, the payload MAY contain a stable semantic key or content hash, but not
a process-local handle as identity.

`AttrValue` or another generic attribute container is not required by the
current Rust-native extension API. Extension authors SHOULD prefer concrete
Rust payload types whose equality and hashing match the operation semantics.

Example:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum WindowMode {
    Valid,
    Same,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct WindowedReduceOp {
    axis: usize,
    mode: WindowMode,
}

impl ExtensionOp for WindowedReduceOp {
    fn family_id(&self) -> &'static str {
        "example.windowed_reduce.v1"
    }

    fn payload_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        hasher.write_usize(self.axis);
        match self.mode {
            WindowMode::Valid => hasher.write_u8(0),
            WindowMode::Same => hasher.write_u8(1),
        }
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<WindowedReduceOp>()
            .is_some_and(|op| op == self)
    }

    fn input_count(&self) -> usize {
        1
    }

    // Other required methods omitted.
}
```

Here `axis` and `mode` are deterministic payload fields, not differentiable
tensor inputs. `input_count()` returns only the number of tensor inputs.

---

## 6. Arity and I/O shape

### Fixed-arity contract

Every `ExtensionOp` MUST declare fixed `input_count` and `output_count` values
whose return is independent of runtime input sizes. This aligns with the
`StdTensorOp` arity dispatcher in
`crates/tenferro-internal-ops/src/std_tensor_op.rs` (e.g. `input_count` for `DotGeneral` is
always `2`).

The dispatcher integration in `crates/tenferro-internal-ops/src/ad/mod.rs` MUST treat
`StdTensorOp::Extension(op)` as having `op.input_count()` inputs and
`op.output_count()` outputs. Validation:

- The number of `primal_in` keys passed to `linearize` MUST equal
  `op.input_count()`.
- The number of `tangent_in` entries MUST equal `op.input_count()`.
- The number of `primal_out` keys MUST equal `op.output_count()`.
- The returned tangent-output vector from `linearize` MUST have length
  `op.output_count()`.
- The returned cotangent-input vector from `transpose_rule` MUST have
  length `op.input_count()`.

### Variable-arity extensions (discouraged)

Extensions with variable arity (for example, a variadic `Concatenate`
analogue) are discouraged because they force the dispatcher to
re-derive arity per instance. If such an op is unavoidable, the
implementer MUST:

1. Store the arity in the payload (so `input_count` reads it from `self`,
   not from the arguments).
2. Document in the extension's own doc comment that the arity is
   dynamic, together with the payload field that determines it.
3. Preserve the invariant that for a given `Arc<dyn ExtensionOp>`
   value, `input_count()` returns the same value every time.

Variable-arity extensions remain outside `StdTensorOp`'s own variable-arity
branch. Core variants with dynamic arity, such as `StdTensorOp::Concatenate`,
store the arity in their payload and are handled by the core enum directly.

### Failure signature

- `input_count` disagreeing with `linearize`'s `primal_in` length in the
  dispatcher causes `Error::InvalidConfig` (see Section 12).
- `output_count` disagreeing with `eager_execute`'s returned vector
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
`crates/tenferro-runtime/src/shape_infer.rs::infer_output_dtype` and
`infer_output_shapes` for core ops, packaged as a single method per
extension.

### Contract

- `input_dtypes.len()` and `input_shapes.len()` MUST both equal
  `self.input_count()`. Callers (`compile_std_to_exec`, eager execution)
  guarantee this.
- The returned vector MUST have length `self.output_count()`.
- Each `(dtype, shape)` pair gives the inferred dtype and symbolic shape
  for the corresponding output slot.
- Shapes are expressed as `Vec<SymDim>` to match
  `TensorMeta::shape` (see `crates/tenferro-internal-ops/src/ad/context.rs:49-109`).
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
  `crates/tenferro-runtime/src/compiler/mod.rs`). The same panic applies to
  extensions.
- An implementer that panics on valid symbolic inputs surfaces as a
  hard crash in symbolic-shape composition tests. This is a contract violation.

---

## 8. Forward execution dispatch

Tenferro has two forward-execution routes; extensions participate in
both, with a normatively-split responsibility.

### Eager path

The eager path runs through `EagerRuntime` and the runtime extension
dispatcher in `crates/tenferro-runtime/src/extension_runtime.rs`.
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
The context-free `ExtensionOp::eager_execute` method is a host/reference
implementation for direct extension-op calls. Runtime-owned execution MUST NOT
fall back to it when an extension family is unregistered; that is a missing
runtime error. Built-in extensions with long-lived caches MUST register a
runtime so `EagerRuntime`-attached execution uses the runtime-owned cache store.

### Compiled path

The compiled path runs through
`crates/tenferro-runtime/src/compiler/mod.rs::compile_std_to_exec` and
`crates/tenferro-runtime/src/exec.rs`. The compiled path MUST include:

1. An `ExecOp::Extension(Arc<dyn ExtensionOp>)` variant (or an
   equivalent carrier) in the execution IR, mirroring the `StdTensorOp`
   variant.
2. Shape / dtype lowering in `compile_std_to_exec` that calls
   `op.infer_output_meta(...)` to populate
   `ExecInstruction::dtype` and `ExecInstruction::output_shapes`.
3. An `execute_extension_op` dispatcher in `crates/tenferro-runtime/src/exec.rs` that,
   at runtime, calls the registered `ExtensionExecutor<B>`.
4. A single-instruction-boundary category for extensions in
   `crates/tenferro-runtime/src/segment.rs` (similar to `DotGeneral`).
   Extensions MUST NOT participate in elementwise fusion planning
   because their fusion semantics are implementer-defined.

### Responsibility split (normative)

- **`compile_std_to_exec`** is responsible for: lowering the
  `StdTensorOp::Extension` variant to `ExecOp::Extension`, calling
  `infer_output_meta` for metadata population, and assigning
  `last_use` markers. It does not invoke backend kernels.
- **`eager_exec` / `eager_builder`** are responsible for: resolving
  inputs from the builder's tensor cache and calling the runtime-owned
  extension executor for extension ops. If no extension executor is available,
  extension execution is an error.
- **Extension runtime (`ExtensionRuntime<B>`)** is responsible for: actual
  forward computation, backend use, runtime cache entries, and device placement
  of outputs. The core pipeline MUST NOT second-guess these choices.

### Rationale

Using one `ExtensionRuntime<B>` contract for eager and compiled execution keeps
runtime caches tied to explicit owners (`EagerRuntime` or `GraphExecutor`) and
avoids hidden thread-local or process-global state in extension crates.

### Failure signature

- An extension whose registered runtime errors surfaces as
  `Error::backend_failure("extension", message)` (or a similar constant)
  with the `family_id` in `message`; see Section 12.
- If a backend lacks a capability the extension needs (e.g.
  cuTENSOR unavailable on CPU-only builds), the extension SHOULD
  produce a descriptive `Error::BackendFailure` rather than panicking;
  the caller decides whether to fall back.

---

## 9. Operation construction and AD rule ownership

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

### AD rule ownership

AD rules are owned by an explicit `ExtensionRuleSet` attached to
`tenferro_ad::AdContext`. New extension crates should expose a helper that
constructs a fresh rule set and registers one `ExtensionAdRule` per supported
family:

```rust
pub fn extension_rules() -> Result<ExtensionRuleSet, ExtensionRegistryError> {
    let mut rules = ExtensionRuleSet::new();
    rules.register_rule(Arc::new(MyExtensionRule))?;
    Ok(rules)
}
```

Applications then pass that set through
`tenferro_ad::AdContext::builder().with_extension_rules(...)`; eager runtimes
that need extension AD should be built from the same `AdContext`.

Double-registration under the same `family_id` in a rule set MUST be rejected
with `ExtensionRegistryError::DuplicateRule { family_id }`.

`ExtensionRegistryError`:

```rust
#[derive(Debug, thiserror::Error)]
pub enum ExtensionRegistryError {
    #[error("AD rule for family_id {family_id:?} already registered")]
    DuplicateRule { family_id: &'static str },
    #[error("family_id {family_id:?} does not match the namespaced format")]
    MalformedFamilyId { family_id: &'static str },
}
```

### Compatibility global lookup

The process-global AD rule registry and its `register_extension_rule` helper
remain a compatibility bridge for older global lookup flows. It is not the
primary extension AD path. Code that needs reproducible tests, multiple
independent AD contexts, or explicit dependency ownership SHOULD use
`ExtensionRuleSet` instead.

The compatibility API exposes a rule lookup function:

```rust
pub fn lookup_extension_rule(family_id: &str) -> Option<std::sync::Arc<dyn ExtensionAdRule>>;
```

Lookup MUST NOT panic on a missing `family_id`. Callers decide how to
handle absence (see Section 12).

### Thread safety

An `ExtensionRuleSet` MUST be safe to clone and read from any thread used by AD
graph construction. The compatibility global registry MUST also be safe to read
from any thread; writes happen only during initialization.

### Failure signature

- Registering two AD rules with the same `family_id` in one rule set returns
  `ExtensionRegistryError::DuplicateRule`.
- Looking up an unregistered AD rule `family_id` returns `None`.
- Running a graph that references an extension op with no rule in the active
  `AdContext` is valid for forward execution. AD through that op returns
  `ADRuleError::Unsupported` with the `family_id` and rule kind.

---

## 10. AD API surface

Extension AD is registered independently from the primal op. Extension crates
implement `ExtensionAdRule` and add it to an `ExtensionRuleSet`. Rule
signatures mirror `Primitive::try_linearize` and
`Primitive::try_transpose_rule` and return `ADRuleResult<_>` so missing rules
can propagate without panic:

```rust
pub trait ExtensionAdRule: Debug + Send + Sync + 'static {
    fn family_id(&self) -> &'static str;

    fn linearize(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut GraphBuilder<StdTensorOp>,
        primal_in: &[ValueKey<StdTensorOp>],
        primal_out: &[ValueKey<StdTensorOp>],
        tangent_in: &[Option<LocalValueId>],
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>>;

    fn transpose_rule(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveBuilder<StdTensorOp>,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[ValueRef<StdTensorOp>],
        mode: &OperationRole,
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>>;
}
```

The `op` argument is the concrete extension payload as a trait object. Rules
that need payload parameters should downcast via `op.as_any()`.

### AD closure

`linearize` and `transpose_rule` may add core `StdTensorOp` values and
`StdTensorOp::Extension` values. Emitted extension families MUST have their
own `ExtensionAdRule` in the active `ExtensionRuleSet` before a subsequent AD
pass reaches them. Terminal first-order helper families MAY omit a separate AD
rule when the owning extension documents that higher-order AD through that
helper is unsupported. This keeps out-of-tree operations in the same compute
graph while preserving the `Primitive` closure invariant at the
`StdTensorOp` carrier level.

### `ShapeGuardContext` interaction

Extension AD rules MUST use `ctx.shape_of(val)`, `ctx.dtype_of(val)`,
and `ctx.metadata_of(val)` to query input metadata, exactly like the
core AD rules (see `crates/tenferro-internal-ops/src/ad/linalg.rs` for a reference
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
  `family_id` with no `ExtensionAdRule` in the active rule set returns
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
| Extension runtime execution returns `Err` | Propagate to caller with `Error::backend_failure("extension", message)` and include `family_id` in `message`. MUST NOT retry, MUST NOT swallow. |
| Backend lacks a capability the extension needs | The extension runtime SHOULD return `Error::backend_failure(...)` with a descriptive message that includes `family_id` and the missing capability name. The core pipeline MUST NOT fall back to a different backend. |
| Graph references an unregistered `family_id` at eager or graph runtime execution time | Return a backend/config error with `family_id` and registration guidance. |
| Graph references an unregistered `family_id` at compile time | Return `Error::Unsupported` from `compile_std_to_exec`. |
| AD rules (`linearize` / `transpose_rule`) encounter an `Extension` with no rule in the active `ExtensionRuleSet` | Return `ADRuleError::Unsupported` with `family_id` and rule kind; traced `grad` / eager `backward` propagate it through the public `Error` type re-exported by the owning surface crate. |
| Duplicate AD rule `family_id` in one `ExtensionRuleSet` or the compatibility global registry | Rule registration MUST reject with `ExtensionRegistryError::DuplicateRule`. |
| Arity mismatch: `input_count()` disagrees with the `primal_in.len()` the dispatcher passed | `Error::InvalidConfig { op: "extension", message: "family_id=<id>: expected N inputs, got M" }`. |
| Output shape disagrees with `infer_output_meta` result length | `Error::InvalidConfig` with `family_id` and the mismatched counts. |
| Extension runtime returns a tensor on the wrong device | Propagate to the caller as a backend failure (the core pipeline does not re-locate tensors). |
| AD rule registration with malformed `family_id` | `ExtensionRegistryError::MalformedFamilyId`. |
| Runtime executor registration with malformed `family_id` | `ExtensionRuntimeRegistryError::MalformedFamilyId`. |

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
of concept in commits `7317268` and `188a278`. That early POC was later
removed during the no-facade crate-boundary cleanup. A current nested
`ext/tropical` crate exists again, but it is an extension consumer of this
contract, not a revival of the retired semiring substrate.

### What `ExtensionOp` is NOT

`ExtensionOp` is **not** a replacement for `SemiringOp<Alg>`:

- The graph is no longer algebra-parameterized. Tropical and other
  non-standard-arithmetic paths live outside the core graph, either as
  compositions of core primitives or as fused extensions such as
  `tenferro-ext-tropical.einsum.v1`.
- `ExtensionOp` provides a single variant `StdTensorOp::Extension(Arc<dyn
  ExtensionOp>)` carrying arbitrary payloads. It does not bring back a
  parallel graph vocabulary keyed on an algebra type parameter.
- Scalar tropical newtypes (`MaxPlus<T>`, `MinPlus<T>`, `MaxMul<T>`) remain
  useful for documenting and testing algebraic scalar semantics. They do not
  make the sealed tensor runtime scalar surface algebra-generic; traced or
  backend-visible tropical behavior still uses composition or `ExtensionOp`.

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

## 14. Worked implementation: tropical einsum (informative)

This section is **informative** (non-normative). It exists to
cross-validate the normative contract against a concrete consumer. If the spec
above is insufficient to guide the current `ext/tropical` implementation, the
spec is wrong and MUST be revised.

The nested `ext/tropical` crate demonstrates the contract with fused binary
tropical einsum:

- The primal payload is `TropicalEinsumOp { kind, subscripts }` with family id
  `tenferro-ext-tropical.einsum.v1`.
- The public traced helpers validate notation, dtype, rank, concrete shared
  label sizes, and output labels before calling `tenferro_runtime::extension`.
- Arity is fixed at 2 inputs / 1 output. Output metadata is inferred from
  `tenferro_einsum::Subscripts` and `SymDim`, so symbolic and concrete shapes
  use the same carrier.
- Forward execution delegates to `tropical_einsum_subscripts_with_argmax`,
  which reuses `tenferro-einsum` contraction-tree lowering for parsing,
  contraction order, shape planning, and GEMM layout. The tropical crate owns
  tropical arithmetic, first-winner argmax metadata, fallback execution, and
  optional `tropical-gemm` dispatch.
- Runtime registration is explicit through `tenferro_ext_tropical::register_runtime`.
- With the `autodiff` feature, `tenferro_ext_tropical::tropical_ad_rules()`
  builds an explicit rule set for `tenferro_ad::AdContext`. That rule set
  registers the primal family `tenferro-ext-tropical.einsum.v1` and the JVP
  helper family `tenferro-ext-tropical.einsum_jvp.v1`. The transposed JVP emits
  runtime-registered VJP execution helper `tenferro-ext-tropical.einsum_vjp.v1`;
  it is a terminal linear_transpose helper, not a separately registered AD rule family
  for higher-order AD.
- Tropical AD follows the unique-winner subgradient. The JVP gathers the active
  tangent at each first-winning contracted coordinate; the VJP scatters the
  cotangent back to the first-winning input coordinates. Oracle replay and
  finite-difference tests cover the supported unique-winner path.

This implementation keeps ordinary einsum semantics unchanged. Ordinary einsum
owns `sum(mul(...))` values and standard AD, while tropical einsum reuses only
the semiring-neutral public lowering information needed to build equivalent
binary contraction loops.

---

## 15. Open questions

The following are explicitly deferred. Future implementations may decide these
without revisiting this document.

1. **`no_std` / wasm targets.** The initial implementation
   MAY restrict `ExtensionOp` to `std`-targets. Widening to `no_std`
   (e.g. for embedded or wasm backends) is deferred until a concrete
   consumer appears.

2. **Cross-process graph serialization format.** Section 11 fixes
   required invariants for any future serializer, but does not
   mandate a specific format. Choosing one (e.g. a bincode / StableHLO
   / protobuf encoding) is out of scope for this contract.

3. **Deep-clone semantics for `Arc<dyn ExtensionOp>`.** Section 4's
   `clone_arc` is intended to be rarely invoked. If a future consumer
   needs a principled "split one Arc into two independent Arcs" path
   (e.g. for cross-thread isolation), the concrete semantics of that
   split are open.

4. **Metrics / observability hooks.** Whether `eager_execute` should
   add tracing spans (via `tracing` crate or similar) is deferred.
   Extensions MAY add their own; the core pipeline does not
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
  `1d9c343`. The then-current Section 14 sketch was updated in the same
  branch to reconcile with that realised implementation.
- 2026-05-29: Section 14 was refreshed to describe the current nested
  `ext/tropical` binary tropical einsum implementation, its shared
  `tenferro-einsum` lowering dependency, its JVP rule helper, and its
  runtime-registered VJP execution helper.
