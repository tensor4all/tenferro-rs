# EagerTensor API Extensions for tensor4all-rs Migration

**Issue:** #706

**Goal:** Extend `EagerTensor` from 6 ops to full coverage of `StdTensorOp`,
enabling tensor4all-rs to adopt it as the core tensor type. Also add
`TypedTensor<T>` convenience methods.

## Error Handling Convention

All `EagerTensor` methods return `Result`. Operator overloads (`+`, `*`,
unary `-`) unwrap internally and panic on error, matching PyTorch ergonomics.

Existing methods (`add`, `mul`, `neg`, `exp`, `reduce_sum`, `dot_general`)
change from `-> Self` to `-> Result<Self>`. This is a breaking change;
existing tests and callers update to `.unwrap()`.

## Multi-Output Op Pattern

A new internal helper sits alongside `unary_op` and `binary_op`:

```rust
fn multi_output_unary_op(
    &self,
    op: StdTensorOp,
    num_outputs: usize,
) -> Result<Vec<Self>>
```

Behavior:
- Calls `exec_op_on_tensors` which already returns `Vec<Tensor>`.
- Assigns a unique `GlobalValKey` to each output.
- When `requires_grad`, builds one `GradNode` per output, sharing the same
  `op` / `primal_in_keys` / `saved_data` / `input_edges` but with distinct
  `output_idx` (0, 1, 2, ...).
- When `!requires_grad`, `grad_node` is `None` for all outputs (zero overhead).

Public linalg methods wrap this:

| Method | Returns | num_outputs |
|--------|---------|-------------|
| `svd()` | `Result<(Self, Self, Self)>` | 3 |
| `qr()` | `Result<(Self, Self)>` | 2 |
| `lu()` | `Result<(Self, Self, Self, Self)>` | 4 |
| `eigh()` | `Result<(Self, Self)>` | 2 |
| `eig()` | `Result<(Self, Self)>` | 2 |

Single-output linalg (`cholesky`, `triangular_solve`) uses `unary_op` or
`binary_op`.

## EagerContext Public API

`EagerContext<B>` becomes `pub struct`. New constructors:

```rust
impl<B: TensorBackend> EagerContext<B> {
    pub fn with_backend(backend: B) -> Rc<Self>;
}

impl<B: TensorBackend> EagerTensor<B> {
    pub fn from_tensor_in(tensor: Tensor, ctx: Rc<EagerContext<B>>) -> Self;
    pub fn requires_grad_in(tensor: Tensor, ctx: Rc<EagerContext<B>>) -> Self;
}
```

`with_backend` creates an `Rc<EagerContext<B>>`. The `_in` suffixed
constructors take a shared context, avoiding repeated `absorb_from()` calls
when creating many tensors in the same computation.

Existing `from_tensor` and `requires_grad` (CpuBackend-only convenience)
remain unchanged.

## Einsum on EagerTensor

New free function:

```rust
pub fn eager_einsum_ad<B: TensorBackend>(
    inputs: &[&EagerTensor<B>],
    subscripts: &str,
) -> Result<EagerTensor<B>>
```

- Extracts `&Tensor` from each input.
- Calls `tenferro_einsum::eager_einsum(ctx, tensors, subscripts)`.
- Merges `EagerContext`s across all inputs (same `absorb_from` pattern as
  `binary_op`).
- Builds a `GradNode` with N input edges if any input has `requires_grad`.
- The op stored in `GradNode` is `StdTensorOp::DotGeneral` for 2-input
  contractions, or a sequence of `DotGeneral` ops following the contraction
  tree for N-ary. The exact op depends on what `eager_einsum` lowers to
  internally — we record the ops as the contraction tree executes them.

**Alternative (simpler):** If recording the full contraction tree's ops is
complex, an initial version can require that callers decompose N-ary einsum
into pairwise `dot_general` calls themselves. The AD would then flow through
the existing `binary_op` path. This defers the `eager_einsum_ad` wrapper to
a follow-up.

## Structural Ops

All follow the existing `unary_op` pattern. Each method constructs a
`StdTensorOp` variant and delegates:

| Method | Signature | StdTensorOp |
|--------|-----------|-------------|
| `transpose` | `(&self, perm: &[usize]) -> Result<Self>` | `Transpose` |
| `reshape` | `(&self, shape: &[usize]) -> Result<Self>` | `Reshape` |
| `slice` | `(&self, config: SliceConfig) -> Result<Self>` | `Slice` |
| `concatenate` | `(tensors: &[&Self], axis: usize) -> Result<Self>` | `Concatenate` |
| `broadcast_in_dim` | `(&self, shape: &[usize], dims: &[usize]) -> Result<Self>` | `BroadcastInDim` |
| `pad` | `(&self, config: PadConfig) -> Result<Self>` | `Pad` |
| `reverse` | `(&self, axes: &[usize]) -> Result<Self>` | `Reverse` |
| `gather` | `(&self, indices: &Self, config: GatherConfig) -> Result<Self>` | `Gather` |
| `dynamic_slice` | `(&self, starts: &Self, sizes: &[usize]) -> Result<Self>` | `DynamicSlice` |

`concatenate` is a static method taking `&[&Self]` since it has N inputs.
Context merging follows the same pattern as `binary_op` extended to N inputs.

## Elementwise Ops

Unary ops (same pattern as `exp`):

| Method | StdTensorOp |
|--------|-------------|
| `abs` | `Abs` |
| `conj` | `Conj` |
| `sign` | `Sign` |
| `log` | `Log` |
| `sqrt` | `Sqrt` |
| `rsqrt` | `Rsqrt` |
| `sin` | `Sin` |
| `cos` | `Cos` |
| `tanh` | `Tanh` |
| `expm1` | `Expm1` |
| `log1p` | `Log1p` |

Binary ops (same pattern as `mul`):

| Method | StdTensorOp |
|--------|-------------|
| `div` | `Div` |
| `pow` | `Pow` |
| `maximum` | `Maximum` |
| `minimum` | `Minimum` |

Ternary op:

| Method | StdTensorOp |
|--------|-------------|
| `select` | `Select` |

`select` needs a `ternary_op` internal helper (condition, on_true, on_false).

## Diagonal Ops

| Method | Signature | StdTensorOp |
|--------|-----------|-------------|
| `extract_diag` | `(&self, axis_a: usize, axis_b: usize) -> Result<Self>` | `ExtractDiag` |
| `embed_diag` | `(&self, axis_a: usize, axis_b: usize) -> Result<Self>` | `EmbedDiag` |
| `tril` | `(&self, k: i64) -> Result<Self>` | `Tril` |
| `triu` | `(&self, k: i64) -> Result<Self>` | `Triu` |

## Reduction Ops

| Method | Signature | StdTensorOp |
|--------|-----------|-------------|
| `reduce_prod` | `(&self, axes: &[usize]) -> Result<Self>` | `ReduceProd` |
| `reduce_max` | `(&self, axes: &[usize]) -> Result<Self>` | `ReduceMax` |
| `reduce_min` | `(&self, axes: &[usize]) -> Result<Self>` | `ReduceMin` |

## TensorScalar::Real Associated Type

`TensorScalar` gains a `Real` associated type so that decompositions
returning real-valued outputs (e.g., singular values from SVD) can be
properly typed:

```rust
pub trait TensorScalar: Copy + Clone + Send + Sync + 'static + private::Sealed {
    type Real: TensorScalar;

    fn into_tensor(shape: Vec<usize>, data: Vec<Self>) -> Tensor;
    fn try_as_slice(tensor: &Tensor) -> Option<&[Self]>;
}
```

Implementations:

| T | T::Real |
|---|---------|
| `f64` | `f64` |
| `f32` | `f32` |
| `Complex64` | `f64` |
| `Complex32` | `f32` |

This is a prerequisite for the `TypedTensor` convenience methods below and
may also benefit existing code that currently uses `Tensor` (dynamic dtype)
where `TypedTensor<T::Real>` would be more precise.

## TypedTensor Convenience Methods (P2)

Convenience methods on `TypedTensor<T>` wrapping free functions from
`tenferro-tensor`. These take `&mut CpuBackend` and delegate to backend
methods. No AD involvement.

```rust
impl<T: TensorScalar> TypedTensor<T> {
    pub fn einsum(ctx: &mut CpuBackend, inputs: &[&Self], subscripts: &str) -> Result<Self>;
    pub fn svd(&self, ctx: &mut CpuBackend) -> Result<(Self, TypedTensor<T::Real>, Self)>;
    pub fn qr(&self, ctx: &mut CpuBackend) -> Result<(Self, Self)>;
}
```

These live in `tenferro-tensor` since they don't depend on the traced/eager
graph infrastructure. Start with einsum + linalg decompositions.

## File Structure

Split `eager.rs` (currently 631 lines) to keep files focused:

| File | Responsibility |
|------|---------------|
| `eager.rs` | `EagerTensor` struct, `EagerContext` (pub), `new_leaf`, `new_result`, `backward()`, `detach()`, `data()`, `grad()`, internal helpers (`saved_forward_values`, `derived_output_key`, etc.) |
| `eager_ops.rs` | All op methods: `unary_op`, `binary_op`, `ternary_op`, `multi_output_unary_op`, `nary_op`, and every public method (`add`, `mul`, ..., `svd`, `transpose`, etc.) |
| `eager_einsum.rs` | `eager_einsum_ad` free function |
| `eager_exec.rs` | Unchanged — `exec_op_on_tensors` |
| `eager_emitter.rs` | Unchanged — `EagerEmitter` |

Tests:

| File | Covers |
|------|--------|
| `tests/eager_tensor.rs` | Existing tests (updated for Result) + new elementwise/structural primal + AD tests |
| `tests/eager_linalg.rs` | Multi-output linalg primal + AD tests |
| `tests/eager_einsum_ad.rs` | `eager_einsum_ad` primal + AD tests |

## Implementation Phases

**Phase 1 (foundation):**
- Change existing methods to `Result`
- Split `eager.rs` → `eager.rs` + `eager_ops.rs`
- Add `multi_output_unary_op`, `ternary_op`, `nary_op` internal helpers
- Make `EagerContext` public, add `with_backend` / `from_tensor_in` / `requires_grad_in`

**Phase 2 (ops — parallelizable):**
- Structural ops (transpose, reshape, slice, etc.)
- Elementwise ops (div, abs, log, sin, etc.)
- Diagonal ops (extract_diag, embed_diag, tril, triu)
- Additional reductions (reduce_prod, reduce_max, reduce_min)

**Phase 3 (linalg):**
- Multi-output: svd, qr, lu, eigh, eig
- Single-output: cholesky, triangular_solve

**Phase 4 (einsum):**
- `eager_einsum_ad` free function

**Phase 5 (TypedTensor):**
- Convenience methods on `TypedTensor<T>`
