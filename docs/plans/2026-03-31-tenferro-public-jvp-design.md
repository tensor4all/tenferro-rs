# Tenferro Public JVP Design

**Date:** 2026-03-31

**Status:** Proposed and approved

## Goal

Add public forward-mode AD to `tenferro::Tensor` as a functional `jvp`
transform, without exposing dual-carrier internals and without expanding the
public API into per-op forward helpers.

## Scope

This design covers:

- the public `tenferro::jvp(...)` API
- result packaging for primal outputs and output tangents
- internal layering onto `LinearizableOp` / `LinearizedOp`
- error semantics
- testing requirements

This design explicitly excludes:

- public `dual_level`, `make_dual`, or `unpack_dual`
- public `linearize(...)`
- HVP / forward-over-reverse
- storing forward-mode state inside `Tensor`

## Current State

The current upstream public surface intentionally centers on:

- `Tensor`
- reverse-mode helpers (`requires_grad_`, `grad`, `backward`)
- runtime-backed tensor/linalg methods
- downstream custom-op seams (`LinearizableOp`, `LinearizedOp`)

Internally, forward-mode semantics already exist at the custom-op seam through:

- `primal`
- `linearize`
- `jvp`
- `vjp`

The missing piece is a public transform that lets downstream users run forward
AD over ordinary `Tensor` computations.

## Design Decision

Expose public forward AD as a free function:

```rust
pub struct JvpResult {
    pub outputs: Vec<Tensor>,
    pub output_tangents: Vec<Option<Tensor>>,
}

pub fn jvp<F>(
    f: F,
    primals: &[Tensor],
    tangents: &[Option<Tensor>],
) -> Result<JvpResult>
where
    F: FnOnce(&[Tensor]) -> Result<Vec<Tensor>>;
```

This is the primary public forward-mode entry point.

## Why Functional JVP Instead Of Method-Based JVP

`jvp` is a transform on a function, not a property of one tensor. The
functional shape is the only public shape that naturally handles:

- unary ops like `exp`
- binary ops like `add`
- variadic ops like `einsum`
- multi-output ops like `qr` and `svd`

By contrast, method-oriented APIs such as `Tensor::jvp(...)` become awkward as
soon as the differentiated computation has more than one input or more than one
output.

## Why Public JVP Instead Of Public Dual Builders

The public `Tensor` type is a facade over `tidu::Value<DynTensor>`. The library
should not re-expose a second public AD carrier story through dual builders.

Public dual-level APIs would:

- leak implementation detail from the forward runtime
- make the public story harder to align with `linearize-first`
- complicate future migration toward more advanced transforms

The library should instead let users describe an ordinary `Tensor` computation
and request its JVP through one explicit transform.

## Result Shape

`JvpResult` returns:

- `outputs`: the primal outputs of the computation
- `output_tangents`: the forward tangents corresponding to each output

Tangents remain optional because zero tangents are common and sparse tangent
seeding should not force materialization of dense zero tensors.

Phase 1 semantics:

- `output_tangents[i] == None` means the output tangent is structurally zero
- returned output tangents are detached values

Detached output tangents keep the contract simple and avoid pretending that
FoR/HVP is already supported.

## Semantics

The public contract is:

- `primals.len() == tangents.len()`
- each `Some(tangent)` must match its primal in dtype, shape, and layout
- `None` tangent means zero tangent
- `f` may return one or many outputs
- `jvp(...)` returns those outputs plus their tangents in matching order

Runtime-backed operations keep their normal runtime requirements. If `f`
evaluates `einsum`, `solve`, `det`, `norm`, `qr`, or `svd`, the caller must
still install a runtime through `set_default_runtime(...)` or
`runtime::with_runtime(...)`.

## Internal Layering

The internal layering remains:

- public transform: `tenferro::jvp(...)`
- runtime seam: `LinearizableOp` / `LinearizedOp`
- math reference layer: `frule` / `rrule` where applicable

Important distinction:

- `frule` is a local mathematical forward rule
- `LinearizedOp::jvp` is the runtime linearization seam

`LinearizedOp::jvp` may delegate to `frule`, but it is not defined by that
delegation. It may also use cached primal outputs, saved residuals, or runtime
packaging logic that a standalone `frule` does not own.

Examples:

- `exp`: optimized `jvp` may reuse cached `exp(x)` instead of recomputing it
- `qr` / `svd`: `jvp` naturally depends on saved factorizations and options

## Error Semantics

Public `jvp(...)` should report:

- tangent count mismatch
- primal/tangent dtype mismatch
- primal/tangent shape mismatch
- primal/tangent layout mismatch when layout matters
- runtime-missing errors from runtime-backed ops
- ordinary errors returned by `f`

The API should not silently coerce tangents.

## Testing Strategy

The public transform needs tests at three levels.

### 1. Public transform tests

Cover:

- unary path: `exp`, `sum`
- binary path: `add`
- contraction path: `einsum`
- linalg path: `solve`, `qr`, `svd`
- `None` tangent inputs
- multi-output results
- runtime-required error path

### 2. Linearized seam tests

If an implementation is not a thin wrapper over an existing `frule`/`rrule`,
add focused seam tests for:

- saved linearization state
- optional tangents/cotangents
- schema handling
- multi-output packaging

This follows the repository rule that optimized `LinearizedOp::jvp/vjp`
implementations must be tested as runtime seams, not only as math rules.

### 3. Docs/rustdoc checks

README, rustdoc, and examples must not claim:

- public dual builders
- public HVP
- more forward AD coverage than the current public surface actually exposes

## Migration Impact

This public shape aligns with the existing `tensor4all-core` wrapper, which can
later migrate from a dual-builder facade toward wrapping `tenferro::jvp(...)`
directly.

It also leaves room for future transforms:

- public `linearize(...)` if ever needed
- FoR/HVP in a later phase
- richer transform options such as `create_graph`-like behavior

None of those are part of this phase.

## Summary

The recommended public forward-mode API for upstream `tenferro::Tensor` is:

- a free `jvp(...)` transform
- detached `JvpResult`
- no public dual carrier
- no public HVP
- internal reuse of `LinearizableOp` / `LinearizedOp`

This keeps the public surface small, matches the `linearize-first` internal
architecture, and leaves a clean path for future FoR support.
