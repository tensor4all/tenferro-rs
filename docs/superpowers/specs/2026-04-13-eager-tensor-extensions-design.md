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

### How backward_dag processes multi-output ops

`tidu::backward_dag` (backward.rs:83-95) iterates `node.primal_out_keys` to
collect cotangents for all outputs, then calls `build_single_op_linear` which
invokes `PrimitiveOp::linearize` / `transpose_rule`. For linalg ops like SVD,
the transpose rule consumes all primal outputs (u, s, vt) from `saved_data`
to produce cotangents for all inputs.

This means multi-output ops need **one shared `GradNode`** that:
1. Lists all output keys in `primal_out_keys` (not just one).
2. Saves all output tensors in `saved_data` (one per `output_slot`).
3. Is referenced by all output `EagerTensor`s (via `Arc`).

Each output `EagerTensor` holds `Arc<GradNode>` pointing to the same node,
plus its own `output_idx` to identify which output it represents. When
`topo_sort_grad_dag` walks the DAG, `Arc` pointer deduplication ensures the
shared node is visited exactly once.

### Changes to saved_forward_values

The current `saved_forward_values` helper stores only `output_slot: 0` via
`derived_output_key`. For multi-output ops, it must store each output:

```rust
fn saved_forward_values_multi(
    op: &StdTensorOp,
    input_keys: &[GlobalValKey<StdTensorOp>],
    inputs: &[Arc<Tensor>],
    outputs: &[Arc<Tensor>],
) -> HashMap<GlobalValKey<StdTensorOp>, Arc<Tensor>> {
    let mut saved = HashMap::with_capacity(input_keys.len() + outputs.len());
    for (key, value) in input_keys.iter().zip(inputs.iter()) {
        saved.insert(key.clone(), Arc::clone(value));
    }
    for (slot, output) in outputs.iter().enumerate() {
        saved.insert(
            GlobalValKey::Derived {
                op: GlobalOpKey {
                    primitive: op.clone(),
                    inputs: input_keys.to_vec(),
                    mode: OpMode::Primal,
                },
                output_slot: slot,
            },
            Arc::clone(output),
        );
    }
    saved
}
```

### Construction flow

```
exec_op_on_tensors(&op, &[input], backend) -> Vec<Tensor>  // e.g. [u, s, vt]
                         |
      build one shared GradNode with:
        primal_out_keys = [key_u, key_s, key_vt]
        saved_data      = {input_keys... , derived(slot=0) -> u, derived(slot=1) -> s, derived(slot=2) -> vt}
        output_idx      = 0  (the node itself; each EagerTensor stores its own output_idx)
                         |
      return Vec<EagerTensor> where each holds Arc::clone of the same GradNode
```

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

### Recording model

`GradNode` stores a single `Op`. The correct op to store is
`StdTensorOp::NaryEinsum { subscripts, n_inputs }`, which already has
linearize and transpose rules in `tenferro-ops/src/ad/contraction.rs`.

The AD rules for `NaryEinsum` emit further `NaryEinsum` ops for each
input's VJP (contracting the cotangent with the remaining primals). During
eager backward, these emitted `NaryEinsum` ops reach `exec_op_on_tensors`,
which currently has `todo!("NaryEinsum eager exec requires contraction tree
planning")` at `eager_exec.rs:73-75`.

**Required change in `eager_exec.rs`:** Implement the `NaryEinsum` branch
by calling `tenferro_einsum::eager_einsum(backend, tensors, subscripts)`.
This is a one-line delegation — the eager einsum implementation already
handles contraction tree planning internally.

### Construction flow

```
eager_einsum(backend, &[tensor_a, tensor_b, ...], subscripts) -> Tensor
                         |
      build GradNode with:
        op              = NaryEinsum { subscripts, n_inputs }
        primal_in_keys  = [key_a, key_b, ...]
        primal_out_keys = [result_key]
        saved_data      = {key_a -> a, key_b -> b, ..., derived(slot=0) -> result}
        input_edges     = [edge_a, edge_b, ...]
```

### Context merging

Merges `EagerContext`s across all N inputs using the same `absorb_from`
pattern as `binary_op`, extended with a loop over all inputs.

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

### Tensor ↔ TypedTensor conversion

Backend linalg methods return `Tensor` (dynamic dtype). Typed convenience
methods must safely convert results back to `TypedTensor<T>` or
`TypedTensor<T::Real>`. Add a conversion method to `TensorScalar`:

```rust
pub trait TensorScalar: ... {
    type Real: TensorScalar;

    fn into_tensor(shape: Vec<usize>, data: Vec<Self>) -> Tensor;
    fn try_as_slice(tensor: &Tensor) -> Option<&[Self]>;

    /// Try to extract a `TypedTensor<Self>` from a dynamic `Tensor`.
    /// Returns `None` if the dtype does not match.
    fn try_into_typed(tensor: Tensor) -> Option<TypedTensor<Self>>;
}
```

`try_into_typed` is the inverse of `into_tensor`. For `f64` it matches
`Tensor::F64(inner)`, for `Complex64` it matches `Tensor::C64(inner)`, etc.

### Complex SVD / eigh: backend dtype contract

The current faer backend converts real singular values / eigenvalues into
complex tensors for complex inputs (e.g., `faer_linalg.rs:1228-1231`
wraps real SVD singular values into `Tensor::C64`). For `TypedTensor<T>::svd`
to return `TypedTensor<T::Real>`, the conversion uses `T::Real::try_into_typed`
on the backend output. For complex SVD:

- Backend returns `[Tensor::C64(u), Tensor::C64(s_complex), Tensor::C64(vt)]`
- `s` conversion: `<f64 as TensorScalar>::try_into_typed(s_complex)` → `None`
  because the tensor is `C64`, not `F64`.

This means the typed wrapper needs to handle the current backend behavior.
Two options:

**(A) Change the backend** to return real dtype for inherently real outputs
(singular values, eigenvalues). This is the correct long-term fix — the
complex wrapper in faer_linalg is an unnecessary dtype promotion.

**(B) Convert in the wrapper** by extracting the real parts from the complex
tensor. E.g., for `Complex64` singular values, take `re` of each element
and build a `TypedTensor<f64>`.

**Decision: Option (A).** Change the faer backend so that SVD/eigh return
real-dtype tensors for singular values / eigenvalues even when the input is
complex. This aligns with NumPy/JAX/PyTorch behavior (e.g.,
`np.linalg.svd(complex_matrix)` returns real singular values). The change
is localized to `faer_linalg.rs` and `cpu/backend.rs`.

### Method signatures and crate placement

Linalg convenience methods live in `tenferro-tensor` (same crate as
`TypedTensor`):

```rust
impl<T: TensorScalar> TypedTensor<T> {
    pub fn svd(&self, ctx: &mut CpuBackend) -> Result<(Self, TypedTensor<T::Real>, Self)>;
    pub fn qr(&self, ctx: &mut CpuBackend) -> Result<(Self, Self)>;
    pub fn cholesky(&self, ctx: &mut CpuBackend) -> Result<Self>;
    pub fn eigh(&self, ctx: &mut CpuBackend) -> Result<(TypedTensor<T::Real>, Self)>;
}
```

**Einsum** convenience cannot live in `tenferro-tensor` because
`eager_einsum` is in `tenferro-einsum`, and `tenferro-tensor` does not
depend on `tenferro-einsum` (dependency flows the other direction). Instead,
`TypedTensor::einsum` lives in `tenferro-einsum` as a free function or
extension trait:

```rust
// In tenferro-einsum/src/eager.rs (or a new typed_eager.rs)
pub fn typed_eager_einsum<T: TensorScalar>(
    ctx: &mut CpuBackend,
    inputs: &[&TypedTensor<T>],
    subscripts: &str,
) -> Result<TypedTensor<T>>
```

This wraps inputs via `T::into_tensor`, calls `eager_einsum`, and converts
back via `T::try_into_typed`.

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
- Add `multi_output_unary_op` (with `saved_forward_values_multi`), `ternary_op`, `nary_op` internal helpers
- Make `EagerContext` public, add `with_backend` / `from_tensor_in` / `requires_grad_in`

**Phase 2 (ops — parallelizable):**
- Structural ops (transpose, reshape, slice, etc.)
- Elementwise ops (div, abs, log, sin, etc.)
- Diagonal ops (extract_diag, embed_diag, tril, triu)
- Additional reductions (reduce_prod, reduce_max, reduce_min)

**Phase 3 (linalg):**
- Multi-output: svd, qr, lu, eigh, eig (using shared `GradNode` pattern)
- Single-output: cholesky, triangular_solve

**Phase 4 (einsum):**
- Implement `NaryEinsum` branch in `eager_exec.rs` (delegate to `eager_einsum`)
- `eager_einsum_ad` free function (stores `NaryEinsum` in `GradNode`)

**Phase 5 (TypedTensor):**
- Add `TensorScalar::Real` associated type + `try_into_typed` method
- Change faer backend to return real dtype for SVD/eigh singular values / eigenvalues
- Linalg convenience methods on `TypedTensor<T>` (in `tenferro-tensor`)
- `typed_eager_einsum` free function (in `tenferro-einsum`)
