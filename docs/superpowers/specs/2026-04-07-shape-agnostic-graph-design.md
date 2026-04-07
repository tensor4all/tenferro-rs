# Shape-Agnostic Graph + N-ary Einsum Design

**Issue:** #651
**Date:** 2026-04-07

## Motivation

TracedTensor graphs currently bake all tensor shapes (`Vec<usize>`) into op
definitions. Tensor network algorithms (DMRG, TT decomposition, ALS) change
bond dimensions dynamically via truncated SVD, making shape-fixed graphs
non-reusable across iterations.

This design removes axis sizes from the graph IR and adds an N-ary einsum op,
enabling graph reuse with varying tensor sizes.

## Design Principles

1. **Ops store only structural parameters** (axis indices, dimension mappings,
   subscripts), never axis sizes.
2. **Rank (ndim) is tracked statically** in `TracedTensor`; sizes are not.
3. **AD rules run as graph transformations** (not deferred to runtime).
   Shape-dependent ops use `*Like` variants with a shape-source graph edge.
4. **StableHLO lowering happens at execution time**, when actual tensor shapes
   are available to resolve `*Like` variants and `NaryEinsum`.

## Change 1: Remove Size Information from StdTensorOp

### Fields to Remove

| Op | Remove | Keep |
|---|---|---|
| `ReduceSum` | `input_shape` | `axes` |
| `ReduceProd` | `input_shape` | `axes` |
| `ReduceMax` | `input_shape` | `axes` |
| `ReduceMin` | `input_shape` | `axes` |
| `Reshape` | `from_shape` | `to_shape` |
| `Svd` | `input_shape` | `eps` |
| `Qr` | `input_shape` | (unit variant) |
| `Cholesky` | `input_shape` | (unit variant) |
| `Lu` | `input_shape` | (unit variant) |
| `Eigh` | `input_shape` | `eps` |
| `Eig` | `input_shape` | `input_dtype` |
| `TriangularSolve` | `lhs_shape`, `rhs_shape` | `left_side`, `lower`, `transpose_a`, `unit_diagonal` |

### Unchanged Ops

- `Reshape { to_shape: Vec<usize> }` -- user-specified output shape stays.
  Using this op makes the graph non-reusable with different sizes.
- `BroadcastInDim { shape: Vec<usize>, dims: Vec<usize> }` -- retained for
  runtime einsum expansion. Using this op also makes the graph non-reusable.
- `DotGeneral { config }` -- contains axis indices, not sizes.

## Change 2: New Op Variants

### `ReshapeLike`

```rust
ReshapeLike {}
// n_inputs: 2  -- [data, shape_source]
// n_outputs: 1
// Execution: reshape data to shape_source.shape()
```

Used by AD rules (e.g., Reshape VJP) to reshape cotangent back to the primal
input shape without embedding concrete sizes.

### `BroadcastInDimLike`

```rust
BroadcastInDimLike { dims: Vec<usize> }
// n_inputs: 2  -- [data, shape_source]
// n_outputs: 1
// Execution: broadcast data to shape_source.shape(), mapping along dims
```

Used by AD rules (e.g., ReduceSum VJP) to broadcast cotangent back to the
primal input shape.

### `NaryEinsum`

```rust
NaryEinsum {
    subscripts: String,   // e.g. "ij,jk,kl->il"
    n_inputs: usize,      // number of input tensors
}
// n_inputs: n_inputs
// n_outputs: 1
```

Records N-ary einsum as a single graph node. Contraction path optimization is
deferred to execution time.

## Change 3: TracedTensor -- Shape to Rank

```rust
// Before
pub struct TracedTensor {
    pub shape: Vec<usize>,
    // ...
}

// After
pub struct TracedTensor {
    pub rank: usize,
    // ...
}
```

### Rank Inference Rules

| Op | Output rank |
|---|---|
| Unary elementwise (Neg, Exp, ...) | same as input |
| Binary elementwise (Add, Mul, ...) | same as input |
| `ReduceSum { axes }` | input rank - axes.len() |
| `Reshape { to_shape }` | to_shape.len() |
| `ReshapeLike` | shape_source.rank |
| `BroadcastInDimLike { dims }` | shape_source.rank |
| `BroadcastInDim { shape, dims }` | shape.len() |
| `Transpose { perm }` | same as input |
| `DotGeneral` | computed from config |
| `Svd` | 3 outputs: (rank, rank-1, rank) |
| `Qr` | 2 outputs: (rank, rank) |
| `NaryEinsum { subscripts }` | output subscript length |
| `ReshapeLike` | shape_source.rank |

Rank inference is trivially computable for all ops, unlike size inference.

## Change 4: AD Rules

AD rules continue to run as graph transformations (linearization/transpose).
They use `*Like` variants with primal nodes as shape sources.

### ReduceSum VJP

```rust
// Before
BroadcastInDim { shape: input_shape.clone(), dims: kept_dims }
inputs: [cotangent]

// After
BroadcastInDimLike { dims: kept_dims }
inputs: [cotangent, primal_input]
```

`kept_dims` is computed from `axes` and the primal input's rank (available via
`TracedTensor.rank`).

### ReduceProd / ReduceMax / ReduceMin VJP

Same pattern as ReduceSum: use `BroadcastInDimLike` with primal input as
shape source.

### Reshape VJP

```rust
// Before
Reshape { from_shape: to_shape.clone(), to_shape: from_shape.clone() }
inputs: [cotangent]

// After
ReshapeLike {}
inputs: [cotangent, primal_input]
```

### BroadcastInDim VJP (concrete shape variant)

```rust
// Before
ReduceSum { axes: broadcast_axes, input_shape: shape.to_vec() }
inputs: [cotangent]

// After
ReduceSum { axes: broadcast_axes }
inputs: [cotangent]
```

`broadcast_axes` is computed from `dims` and output rank at graph
transformation time.

### BroadcastInDimLike VJP

Same as above: produce `ReduceSum { axes }` where axes are computed from
`dims` and shape_source rank.

### Linalg AD (SVD, QR, Cholesky, Eigh, TriangularSolve)

Linearization rules extract `(m, n, batch_shape)` from primal input/output
tensors at execution time (during StableHLO lowering), not from op metadata.
Intermediate ops (DotGeneral, TriangularSolve) carry no size information.
Where intermediate Reshape is needed, `ReshapeLike` references a primal
output tensor as shape source.

## Change 5: Execution Pipeline

```
Trace time:
  StdTensorOp graph (shape-agnostic, rank only)
  AD transformations (linearization/transpose) applied here

Execution time:
  1. Input tensors provide actual shapes
  2. StdTensorOp -> StableHloOp (shapes resolved from tensors)
     - ReshapeLike + shape_source.shape()  -> Reshape { concrete shape }
     - BroadcastInDimLike + shape_source.shape() -> BroadcastInDim { concrete shape, dims }
     - NaryEinsum -> custom_call (or binary ops expansion)
     - ReduceSum { axes } -> ReduceSum { axes } (no change needed)
     - Linalg ops -> same (shapes from tensors)
  3. StableHloOp -> ExecOp (unchanged)
  4. Backend execution (unchanged)
```

**StableHloOp, ExecOp, and Backend layers require no changes.** All shape
resolution happens during the StdTensorOp-to-StableHloOp lowering step.

## Change 6: NaryEinsum Execution

At execution time when `NaryEinsum` is encountered:

1. Read actual input tensor shapes
2. Look up `EinsumCache` with key `(subscripts, input_shapes)`
3. Cache miss: run `ContractionTree::optimize()`, store result
4. Cache hit: reuse stored `ContractionTree`
5. Expand tree to binary ops (DotGeneral, Transpose, ReduceSum) with concrete
   shapes and execute

The `EinsumCache` key type remains `(String, Vec<Vec<usize>>)`.

When bond dimensions change (e.g., between DMRG sweeps), a cache miss triggers
re-optimization. When the same size pattern recurs (e.g., left-to-right then
right-to-left sweep), the cache hits.

## Graph Reuse

A graph is reusable with different input sizes if and only if it contains no
concrete-size ops:

| Op | Reusable? |
|---|---|
| `Reshape { to_shape }` | No -- concrete sizes |
| `BroadcastInDim { shape, dims }` | No -- concrete sizes |
| All other ops | Yes |

In typical tensor network workloads (einsum + QR/SVD), users do not call
`Reshape` explicitly, so graphs are fully reusable.

## Constraints and Limitations

- **Reshape with concrete sizes breaks graph reuse.** This is documented as a
  known limitation. Tensor network workloads (einsum + QR/SVD) do not require
  explicit Reshape.
- **XLA backend**: `NaryEinsum` lowers to `custom_call`. `*Like` variants
  resolve to concrete StableHLO ops at execution time.
- **Rank must be statically known.** All ops must support rank inference from
  their parameters and input ranks. This is satisfied by the current op set.

## Scope

### In Scope

- Remove size fields from `StdTensorOp` variants
- Add `ReshapeLike`, `BroadcastInDimLike`, `NaryEinsum` variants
- Change `TracedTensor.shape` to `TracedTensor.rank`
- Update all AD rules to use `*Like` variants with shape-source edges
- Update StableHLO lowering to resolve shapes at execution time
- Update `einsum()` to emit `NaryEinsum` instead of expanding to binary ops

### Out of Scope

- StableHloOp changes (none needed)
- ExecOp changes (none needed)
- Backend changes (none needed)
- XLA backend implementation (future work)

### Files to Modify

- `tenferro-ops/src/std_tensor_op.rs` -- op definitions
- `tenferro-ops/src/ad/` -- all AD rule files
- `tenferro/src/traced.rs` -- TracedTensor shape->rank
- `tenferro/src/compiler.rs` -- StableHLO lowering
- `tenferro/src/einsum.rs` -- NaryEinsum emission
- `tenferro-einsum/` -- NaryEinsum support
