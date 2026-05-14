# Issue 856 Index-Select Design

**Goal:** Add AD-preserving trailing-batch gather/index-select support for
TreeTN-style batched contractions without adding TreeTN-specific operations or
backward-compatibility shims.

**Architecture:** Expose `index_select` as a first-class standard tensor API
and lower it to the existing `Gather` primitive. Keep reverse-mode semantics
owned by the existing `Gather` to `Scatter` transpose rule, so repeated
positions accumulate through scatter-add. Add only the minimal missing
rightmost-batch packing surface needed by #856; do not duplicate a separate
stack/cat subsystem.

**Tech Stack:** Rust, `tenferro-tensor`, `tenferro`, `tenferro-ops`,
existing `StdTensorOp::Gather` / `StdTensorOp::Scatter`, CPU `BufferPool`,
existing traced/eager AD pipeline.

---

## Current State

`origin/main` already has the core substrate that should own this behavior:

- `tenferro_tensor::GatherConfig` and `ScatterConfig`.
- CPU and CubeCL backend entrypoints for `gather` and `scatter`.
- `StdTensorOp::Gather` and `StdTensorOp::Scatter`.
- Forward-mode gather linearization that gathers the operand tangent.
- Reverse-mode gather transpose that emits scatter-add into a zero operand.
- `StdTensorOp::Concatenate`, including eager/traced execution and AD.

The missing piece is not a new primitive. The missing piece is a public,
ergonomic axis-select API that builds the right `GatherConfig` and a
rightmost-batch packing path that downstream code can use without host-side
scalar materialization.

The current CPU indexing helpers allocate some fully-overwritten outputs with
ad hoc `Vec::with_capacity` / uninitialized vectors outside the backend buffer
pool. Issue 856 makes this an API-contract concern for stack/gather workloads,
so the CPU indexing path should be moved onto the normal backend/session
allocation path where practical.

## Public API

Add axis-select methods at the three surfaces that users currently exercise:

```rust
impl Tensor {
    pub fn index_select(
        &self,
        axis: isize,
        positions: &[usize],
        ctx: &mut impl TensorBackend,
    ) -> tenferro_tensor::Result<Self>;
}

impl<B: TensorBackend> EagerTensor<B> {
    pub fn index_select(&self, axis: isize, positions: &[usize]) -> tenferro::Result<Self>;
}

impl TracedTensor {
    pub fn index_select(&self, axis: isize, positions: &[usize]) -> tenferro::Result<Self>;
}
```

`axis` follows torch-style negative indexing. For a rank `r` tensor, valid
values are `[-r, r - 1]`. `axis = -1` selects the trailing dimension and is the
canonical TreeTN batch-axis path.

`positions` is host-known and non-differentiable. Repeated positions are valid.
An empty `positions` slice is valid and produces an output whose selected axis
has extent zero.

Add only the minimal stack API needed for the rightmost-batch packing path if
it is still absent from the branch being implemented:

```rust
impl Tensor {
    pub fn stack(
        tensors: &[&Self],
        dim: isize,
        ctx: &mut impl TensorBackend,
    ) -> tenferro_tensor::Result<Self>;
}

impl<B: TensorBackend> EagerTensor<B> {
    pub fn stack(tensors: &[&Self], dim: isize) -> tenferro::Result<Self>;
}

impl TracedTensor {
    pub fn stack(tensors: &[&Self], dim: isize) -> tenferro::Result<Self>;
}
```

`stack(..., dim = -1)` inserts a new trailing axis and then concatenates along
that axis. It should be implemented as reshape/concatenate composition rather
than a second packing primitive.

## Index-Select Lowering

`index_select(axis, positions)` lowers to `Gather` with one indexed operand
axis:

```text
operand shape:      [d0, ..., d_axis, ..., dn]
positions:          [p]
output shape:       [d0, ..., p, ..., dn]
collapsed dim:      axis
slice size:         1 on axis, full size on other axes
start_index_map:    [axis]
index_vector_dim:   1
offset dims:        all output axes except axis
```

The generated start-index tensor has shape `[positions.len(), 1]` and dtype
`I64`. It is represented as a constant leaf/input for traced/eager execution,
not as a differentiable value.

Out-of-range positions should be rejected before building the op. This avoids
using StableHLO gather clamping for a user-facing index-select API, where
silent clamping would hide mistakes.

## AD Semantics

No new AD rule is required.

- Forward mode: existing `linearize_gather` applies the same gather to the
  operand tangent and ignores the index operand.
- Reverse mode: existing `transpose_gather` emits inverse `Scatter` into a
  zero-like operand. Since scatter has add semantics, duplicate positions
  accumulate correctly.

For example:

```text
x shape:    [3]
positions:  [1, 1, 2]
weights:    [10, 20, 30]
loss:       sum(weights * x.index_select(0, positions))
grad x:     [0, 30, 30]
```

This should be covered by traced reverse-mode and forward-mode tests, plus an
eager reverse-mode test where the eager surface is used directly.

## Allocation And Initialization

Forward stack and index-select outputs are fully overwritten. The CPU backend
should therefore allocate their output buffers through the active
`CpuExecSession` buffer pool and avoid zero-initializing those buffers.

Design changes:

- Thread `&mut BufferPool` into CPU indexing helpers used by `CpuExecSession`.
- Use pooled uninitialized buffers for fully-overwritten gather, concatenate,
  stack, slice, reverse, and dynamic-slice outputs where the implementation
  can prove every element is written before read.
- Keep zero initialization only where required by semantics, especially
  reverse-mode scatter-add accumulation into a zero cotangent.
- Retain explicit typed errors for unsupported dtype/device/layout cases.

This issue does not need to solve every historical indexing allocation in one
large refactor. It should cover the paths used by `stack(..., -1)` and
`index_select(..., -1)` and avoid introducing new ad hoc allocations.

## Error Behavior

Use existing typed error variants where possible:

- invalid normalized axis: `AxisOutOfBounds { op: "index_select", ... }`
- position outside the selected axis extent: `InvalidConfig`
- mismatched stack input shape: `ShapeMismatch`
- empty stack inputs: `InvalidConfig`
- unsupported backend/device/layout: `BackendFailure`

CPU tensors should not silently transfer between devices. GPU-native support is
not required for phase 1 if the existing CubeCL gather path cannot support the
generated config. A typed unsupported error is acceptable.

## Retained-Batch Contraction Contract

The rightmost dimension is the canonical batch axis:

```text
input messages:       [bond_dims...]
stack(..., -1):       [bond_dims..., source_assignments]
index_select(-1):     [bond_dims..., target_assignments]
dot_general retain b: [lhs_free..., rhs_free..., b]
```

Tests should include a small retained-batch contraction:

```text
A: [m, k, b]
B: [k, n, b]
C: [m, n, b]
```

This verifies that operands produced by trailing-axis stack/index-select flow
into the existing batch-trailing GEMM convention.

## Testing Plan

Write tests before implementation:

- `stack(..., dim = -1)` packs rank-0, rank-1, and rank-2 tensors into a
  trailing batch axis.
- `index_select(axis = -1, positions)` returns expected values for a
  column-major dense tensor.
- `index_select` supports repeated positions.
- repeated-position reverse-mode gradient uses scatter-add.
- forward-mode tangent follows the same positions.
- invalid axis and out-of-range positions return typed errors.
- retained-batch contraction keeps the batch dimension trailing.
- CPU buffer-pool reuse is visible for forward stack/index-select outputs.

Do not add an `ExtensionOp` or a TreeTN-specific API. Do not preserve old
host-materialization workarounds.

