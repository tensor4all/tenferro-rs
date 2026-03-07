# Einsum + DyadTensor AD Design

This document describes how `tenferro-einsum` and dyadtensor-style higher
layers integrate with the current `chainrules` AD model.

For the core AD contracts, see [autodiff.md](./autodiff.md). For math
derivations, see [AD Formula Notes](../AD/index.md).

## Scope

This document covers:

- how einsum uses `Tape<V>`, `TrackedTensor<V>`, `DualTensor<V>`, and
  `Variable<V>`
- tensor scalar semantics for loss construction
- `retain_graph` / `create_graph` expectations at the integration boundary

It does not define a separate heterogeneous AD system. The current integration
model is homogeneous only.

## Core Decision

Einsum and dyadtensor layers integrate with a single AD execution model:

- reverse mode: homogeneous `Tape<V>` graphs
- forward mode: `DualTensor<V>`
- torch-like wrapper APIs: `Variable<V>`

There is no mixed runtime-erased tape path for custom values.

Examples:

- `tracked_einsum` works with `TrackedTensor<Tensor<T>>`
- `dual_einsum` works with `DualTensor<Tensor<T>>`
- dyadtensor wrappers may expose torch-like convenience APIs, but they still
  lower to homogeneous `Variable<V>` / `Tape<V>` execution

## Tensor Scalar Semantics

Einsum losses follow PyTorch-style tensor scalar conventions.

- scalar tensor means rank-0 (`shape=[]`)
- shape `[1]` is not scalar
- elementwise tensor-scalar sugar should normalize to rank-0 tensor semantics

Typical loss construction:

```rust,ignore
let loss = tracked_einsum("ij,ij->", &[&x, &x]).unwrap(); // rank-0 tensor
```

or equivalently, higher layers may use an explicit reduction that returns a
rank-0 tensor.

Implicit reverse seed creation still follows `num_elements() == 1` in the core
AD engine. This is separate from the tensor-operation scalar definition above.

## Reverse-Mode Integration

### Explicit einsum AD entry points

`tenferro-einsum` keeps explicit interfaces:

- `tracked_einsum`
- `einsum_rrule`
- `einsum_hvp`

The reverse-mode flow is:

1. build leaves on `Tape<Tensor<T>>`
2. execute tracked einsum operations
3. obtain a rank-0 loss tensor
4. call `tape.pullback(&loss)` or use `Variable<V>::backward`

### Query APIs

Higher layers that need query-style gradients use:

- `autograd::grad_tangent`
- `autograd::grad_variable`

These remain monomorphic. They do not mutate `.grad()` / `.hvp()` buffers.

### HVP

Forward-over-reverse HVP uses tangent-seeded leaves and the monomorphic tape:

```rust,ignore
let tape = Tape::<Tensor<f64>>::new();
let x = tape.leaf_with_tangent(x0, v0).unwrap();
let loss = tracked_einsum("i,i->", &[&x, &x]).unwrap();
let hv = tape.hvp(&loss).unwrap();
```

`backward_hvp(create_graph = true)` is still unsupported in the current phase.

## Forward-Mode Integration

`dual_einsum` remains the forward-mode entry point.

- primal values live in `DualTensor<V>::primal()`
- tangents live in `DualTensor<V>::tangent()`
- tangent shape validation is the responsibility of the tensor-facing layer
  around `DualTensor`

## `retain_graph` / `create_graph`

The integration layer should assume the following core rules:

- `effective_retain_graph = retain_graph.unwrap_or(create_graph)`
- `grad_tangent(create_graph = true)` is unsupported
- `grad_variable(create_graph = true)` is supported only when `V::Tangent = V`
  and the participating rules expose graph-aware pullback wiring
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
- `grad_tangent` and `grad_variable` behavior on supported einsum wrappers
- HVP on homogeneous tensor graphs with tangent-seeded leaves
- shape `[1]` tensors continuing to follow normal tensor semantics instead of
  scalar shortcuts
