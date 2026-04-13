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

This means multi-output ops need **one `GradNode` per output** that share
the same `Arc`-wrapped data but carry distinct `output_idx`:

1. All nodes list all output keys in `primal_out_keys` (not just their own).
2. All nodes share the same `saved_data` containing all output tensors.
3. `topo_sort_grad_dag` deduplicates by `Arc` pointer — if all output
   `EagerTensor`s point to the same `Arc<GradNode>`, the node is visited
   once. However, `GradNode.output_idx` is a field on the struct, so each
   output needs its own `GradNode` instance with a unique `output_idx`.

To avoid duplicating `saved_data`, we wrap it in `Arc`:

```rust
pub struct GradNode<Op: GraphOp> {
    pub op: Op,
    pub primal_in_keys: Vec<GlobalValKey<Op>>,
    pub primal_out_keys: Vec<GlobalValKey<Op>>,
    pub saved_data: Arc<HashMap<GlobalValKey<Op>, Arc<Op::Operand>>>,  // shared
    pub input_edges: Vec<GradEdge<Op>>,
    pub output_idx: usize,  // unique per output
}
```

This requires a change in `tidu::GradNode` to wrap `saved_data` in `Arc`.
The change is backward-compatible: single-output ops create `Arc::new(map)`
as before; multi-output ops clone the `Arc` for each output node.

With separate `GradNode` instances per output, `topo_sort_grad_dag` may
visit the SVD node up to 3 times (once per output). `backward_dag` will
then process it multiple times, but only the first visit produces
cotangent inputs (subsequent visits find cotangents already consumed).
This is acceptable for correctness; an optimization to deduplicate can
be added later if needed.

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
      build Arc<saved_data> shared across all outputs:
        saved_data = {input_keys..., derived(slot=0)->u, derived(slot=1)->s, derived(slot=2)->vt}
                         |
      build one GradNode per output:
        node_u:  { op, primal_out_keys=[key_u,key_s,key_vt], saved_data=Arc::clone, output_idx=0 }
        node_s:  { op, primal_out_keys=[key_u,key_s,key_vt], saved_data=Arc::clone, output_idx=1 }
        node_vt: { op, primal_out_keys=[key_u,key_s,key_vt], saved_data=Arc::clone, output_idx=2 }
                         |
      return Vec<EagerTensor> where each holds its own Arc<GradNode>
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

Note: `StdTensorOp::Concatenate` currently lacks arity metadata, and
`n_inputs()` / `n_outputs()` are `todo!()` in `std_tensor_op.rs`. As a
prerequisite, add `n_inputs: usize` to `StdTensorOp::Concatenate` (or
implement `n_inputs()` to return `None` and handle variable arity in the
eager path). For the eager `nary_op` helper, the input count is known at
call time from the slice length, so the `todo!()` does not block eager
execution — it only matters for traced graph validation.

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

**Decision: Change the full stack** so that SVD/eigh return real-dtype
tensors for singular values / eigenvalues, matching NumPy/JAX/PyTorch
behavior. This requires coordinated changes across backend, op definition,
and AD rules.

### Required changes for real-valued SVD s / Eigh w

**1. Backend (`tenferro-tensor/src/cpu/linalg/faer_linalg.rs`, `cpu/backend.rs`):**
Remove the complex wrapper around real singular values / eigenvalues. For
complex SVD, return `[Tensor::C64(u), Tensor::F64(s), Tensor::C64(vt)]`
instead of `[Tensor::C64(u), Tensor::C64(s_complex), Tensor::C64(vt)]`.
Same for Eigh.

**2. Op definition (`tenferro-ops/src/std_tensor_op.rs`):**
Add `input_dtype: DType` to `StdTensorOp::Svd` and `StdTensorOp::Eigh`,
following the existing pattern in `StdTensorOp::Eig`. This allows AD rules
to know the input dtype and insert Convert ops when needed.

**3. AD rules (`tenferro-ops/src/ad/linalg.rs`):**
In `linearize_svd` and `linearize_eigh`, when `input_dtype` is complex:

*Forward (linearize):*
- At entry: Convert real primal `s`/`w` → complex before mixed-dtype
  operations with complex U, Vt, V. Use `StdTensorOp::Convert { to:
  input_dtype, from: real_dtype }`.
- At exit: The tangent outputs `ds`/`dw` are complex (result of complex
  arithmetic). Convert complex `ds`/`dw` → real at the output boundary
  using `StdTensorOp::Convert { to: real_dtype, from: input_dtype }`,
  since `s`/`w` are real-valued and their tangents must match.

*Reverse (transpose):*
- The cotangent seed for `s`/`w` arrives as real (matching the primal's
  real dtype). The transpose rule must Convert it to complex before use
  in internal complex arithmetic, then Convert the final cotangent for
  the input `a` stays complex (matching `a`'s dtype).

This follows the existing pattern in `linearize_eig` (lines 143-159 in
`linalg.rs`), which already inserts `Convert` ops for dtype promotion.

**4. Traced API (`tenferro/src/linalg_api.rs`):**
`TracedTensor` already carries a `dtype: DType` field (`traced.rs:50`).
`svd()` / `eigh()` pass `input_dtype: a.dtype` when constructing the op,
following the existing pattern in `eig()` (`linalg_api.rs:251-252`).

After `apply_multi_output`, output dtypes must be patched per-output.
`apply_multi_output` currently stamps all outputs with the input dtype
(`traced.rs:1324`). For complex SVD, the `s` output must have its dtype
patched to the real counterpart:

```rust
// In svd() after apply_multi_output:
let real_dtype = real_dtype_of(a.dtype); // C64→F64, C32→F32, Fxx→Fxx
u.dtype = a.dtype;      // complex
s.dtype = real_dtype;    // real
vt.dtype = a.dtype;      // complex
```

This follows the existing pattern in `eig()` (`linalg_api.rs:261-262`)
which manually patches output dtypes.

**5. Eager path:**
`EagerTensor` has the concrete `Tensor`, so `input_dtype` is known at
construction: `self.data.dtype()`.

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
| `eager_exec.rs` | `exec_op_on_tensors` — add `NaryEinsum` branch (delegate to `eager_einsum`) |
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
- Change `tidu::GradNode.saved_data` to `Arc<HashMap<...>>` for multi-output sharing
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

**Phase 5 (real-dtype SVD/eigh + TypedTensor):**
- Add `input_dtype: DType` to `StdTensorOp::Svd` and `StdTensorOp::Eigh`
- Change faer backend to return real dtype for SVD/eigh singular values / eigenvalues
- Update `linearize_svd` / `linearize_eigh` AD rules to insert `Convert` ops for real→complex promotion
- Update `svd()` / `eigh()` in traced API (`linalg_api.rs`) and eager API to pass `input_dtype`
- Add `TensorScalar::Real` associated type + `try_into_typed` method
- Linalg convenience methods on `TypedTensor<T>` (in `tenferro-tensor`)
- `typed_eager_einsum` free function (in `tenferro-einsum`)
