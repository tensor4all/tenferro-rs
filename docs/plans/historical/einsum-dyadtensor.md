# Historical Einsum + DyadTensor AD Design

This is a historical design record for the pre-extension-runtime AD model. It
is retained for scalar/loss semantics background only; the tape and dual-value
sketches below are not current public API guidance.

`tenferro-einsum` now records graph-level extension operations and routes AD
through extension rules registered with `tenferro-ad`.

For the core AD contracts, see [autodiff.md](../../architecture/ad-pipeline.md). For math
derivations, see [AD Formula Notes](../../AD/index.md).

## Scope

This document covers:

- how einsum uses `Tape<V>`, `TrackedValue<V>`, and `DualValue<V>`
- tensor scalar semantics for loss construction
- `retain_graph` / `create_graph` expectations at the integration boundary

It does not define a separate heterogeneous AD system. The current integration
model is homogeneous only.

## Core Decision

Einsum and frontend layers integrate with a single AD execution model:

- reverse mode: homogeneous `Tape<V>` graphs
- forward mode: `DualValue<V>`

There is no mixed runtime-erased tape path for custom values.

Historical examples:

- a reverse-mode wrapper worked with `TrackedValue<Tensor<T>>`
- a forward-mode wrapper worked with `DualValue<Tensor<T>>`
- frontend wrappers may expose eager convenience APIs, but they still
  lower to homogeneous `TrackedValue<V>` / `Tape<V>` execution

## Tensor Scalar Semantics

Einsum losses follow PyTorch-style tensor scalar conventions.

- scalar tensor means rank-0 (`shape=[]`)
- shape `[1]` is not scalar
- elementwise tensor-scalar sugar should normalize to rank-0 tensor semantics

Typical loss construction sketch:

```text
let loss = /* legacy tracked einsum call returning a rank-0 tensor */;
```

or equivalently, higher layers may use an explicit reduction that returns a
rank-0 tensor.

Implicit reverse seed creation still follows `num_elements() == 1` in the core
AD engine. This is separate from the tensor-operation scalar definition above.

## Reverse-Mode Integration

### Explicit einsum AD entry points

The historical design used explicit interfaces:

- reverse-mode tracked wrapper
- reverse-mode rule helper
- HVP helper

The reverse-mode flow is:

1. build leaves on `Tape<Tensor<T>>`
2. execute tracked einsum operations
3. obtain a rank-0 loss tensor
4. call `tape.pullback(&loss)` or the higher-level `tenferro_ad` query APIs

### Query APIs

Higher layers that need query-style gradients use:

- `tenferro_ad::TracedTensorAdExt::grad`

These remain monomorphic. They do not mutate `.grad()` / `.hvp()` buffers.

### HVP

Forward-over-reverse HVP uses tangent-seeded leaves and the monomorphic tape:

```text
let tape = Tape::<Tensor<f64>>::new();
let x = tape.leaf_with_tangent(x0, v0).unwrap();
let loss = /* legacy tracked einsum call returning a rank-0 tensor */;
let hv = tape.hvp(&loss).unwrap();
```

High-level eager HVP wrappers are still intentionally limited in the current
phase; the low-level `tidu::Tape::hvp` path is the source of truth.

## Forward-Mode Integration

The historical dual-mode wrapper was the forward-mode entry point.

- primal values live in `DualValue<V>::primal()`
- tangents live in `DualValue<V>::tangent()`
- tangent shape validation is the responsibility of the tensor-facing layer
  around `DualValue`

## `retain_graph` / `create_graph`

The integration layer should assume the following core rules:

- `retain_graph = false` frees the shared tape after eager reverse queries
- graph-building higher-order gradient queries are currently unsupported
- non-scalar outputs require explicit seed cotangents
- single-element outputs may omit the seed

For tensor losses, higher layers should still prefer genuine rank-0 outputs
because that matches PyTorch user expectations and avoids ambiguity around
shape `[1]`.

## DyadTensor Wrapper Guidance

Dyadtensor-style wrappers should preserve the core AD contracts instead of
inventing separate scalar rules.

- use rank-0 tensors for tracked tensor scalars
- keep broadcast and elementwise semantics in the tensor layer
- do not special-case shape `[1]` as scalar
- keep custom homogeneous value types on `Tape<MyType>` rather than introducing
  mixed graph requirements

## Test Expectations

Integration tests should cover at least:

- tracked einsum producing rank-0 tensor losses
- backward on homogeneous tensor graphs
- `tenferro_ad::TracedTensorAdExt::grad` behavior on supported einsum wrappers
- HVP on homogeneous tensor graphs with tangent-seeded leaves
- shape `[1]` tensors continuing to follow normal tensor semantics instead of
  scalar shortcuts
