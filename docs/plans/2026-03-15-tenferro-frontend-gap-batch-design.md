# Tenferro Frontend Gap Batch Design

This design closes the remaining public-surface gaps in `tenferro::Tensor` so
the crate behaves more like a PyTorch-style frontend while keeping
`StructuredTensor<T>` and other typed internals out of the main user path.

## Goals

- Add missing frontend tensor methods that downstream code expects:
  - `Tensor::permute(&[usize]) -> Result<Tensor>`
  - `Tensor::conj(&self) -> Tensor`
  - `Tensor::try_scalar_value() -> Result<ScalarValue>`
  - `Tensor::mode() -> AdMode`
- Add a clean primal-only export boundary:
  - `Tensor::primal_snapshot() -> snapshot::DynTensor`
  - `snapshot::DynTensor::to_dense() -> Result<snapshot::DynTensor>`
- Add a PyTorch-like functional HVP surface:
  - `tenferro::functional::hvp(...)`
- Keep `Tensor` as the only compute protagonist in the root crate.

## Non-Goals

- Do not expose `StructuredTensor<T>` from the root frontend.
- Do not add leaf-accumulating HVP APIs.
- Do not broaden structured linalg support beyond the existing dense-only gate.
- Do not redesign `backward` / `grad` / `forward_ad` again in this batch.

## Public Surface

### 1. `Tensor::permute`

```rust
impl Tensor {
    pub fn permute(&self, perm: &[usize]) -> Result<Tensor>;
}
```

Semantics:

- `permute` acts on logical axes, not just dense payload axes.
- It supports dense, diagonal, and general `axis_classes` layouts.
- Invalid permutations return an error.

Implementation model:

- delegate to `StructuredTensor<T>` helpers
- permute logical dims
- permute logical axis classes
- derive the induced permutation on compressed payload axes
- permute the payload tensor accordingly
- rebuild a validated structured tensor

### 2. `Tensor::conj`

```rust
impl Tensor {
    pub fn conj(&self) -> Tensor;
}
```

Semantics:

- Returns the complex conjugate of the tensor.
- Real tensors are returned unchanged.
- AD mode is preserved.
- Structured layout is preserved.

This should lower to the existing lazy conjugation support in
`tenferro-tensor`.

### 3. `Tensor::try_scalar_value`

```rust
pub enum ScalarValue {
    F32(f32),
    F64(f64),
    C32(num_complex::Complex32),
    C64(num_complex::Complex64),
}

impl Tensor {
    pub fn try_scalar_value(&self) -> Result<ScalarValue>;
}
```

Semantics:

- Succeeds only when `dims() == []`.
- Does not cast or coerce.
- Returns an error for non-rank-0 tensors.

This is the canonical scalar extraction API for the dynamic frontend. Any
future `item_f64()` / `item_c64()` helpers should be thin wrappers over this.

### 4. `Tensor::mode`

```rust
impl Tensor {
    pub fn mode(&self) -> AdMode;
}
```

This exposes the existing AD-state classification in a frontend-safe way so
downstream code can distinguish primal / forward / reverse tensors without
using internal accessors.

### 5. Snapshot boundary

Introduce a dedicated snapshot submodule:

```rust
pub mod snapshot {
    pub enum DynTensor {
        F32(StructuredTensor<f32>),
        F64(StructuredTensor<f64>),
        C32(StructuredTensor<Complex32>),
        C64(StructuredTensor<Complex64>),
    }
}

impl Tensor {
    pub fn primal_snapshot(&self) -> snapshot::DynTensor;
}
```

Important boundary decisions:

- `Tensor::detach() -> Tensor` stays as the PyTorch-like compute-side API.
- `Tensor::primal_snapshot() -> snapshot::DynTensor` becomes the explicit export
  / FFI / storage boundary.
- `snapshot::DynTensor` is **not** re-exported at crate root.
- `Tensor` remains the public compute protagonist.

Snapshot methods:

- `scalar_type()`
- `dims()`
- `axis_classes()`
- `is_dense()`
- `is_diag()`
- `ndim()`
- `len()`
- `is_empty()`
- dtype-specific typed accessors as needed for bridge code
- `to_dense() -> Result<snapshot::DynTensor>`

`to_dense()` semantics:

- If already dense, returns an equivalent dense snapshot.
- If structured, materializes the dense logical tensor and returns the
  corresponding dense snapshot variant.

### 6. Functional HVP

PyTorch reference point:

- `torch.autograd.backward(...)` accumulates into leaves
- `torch.autograd.grad(...)` returns gradients
- `torch.autograd.functional.hvp(func, inputs, v)` returns `(output, hvp)`

`tenferro` should mirror the functional, side-effect-free form.

Recommended surface:

```rust
pub mod functional {
    pub fn hvp<F, I, V>(
        func: F,
        inputs: I,
        vectors: V,
        options: HvpOptions,
    ) -> Result<(Tensor, V::Output)>;
}
```

Exact generic shape can be adapted to Rust ergonomics, but the semantic
contract should match PyTorch:

- `func` returns a rank-0 `Tensor`
- `inputs` and `vectors` have matching structure
- return value is `(func_output, hvp)`
- no mutation of leaf `.grad`
- no leaf `.hvp` accumulation API in the frontend

This keeps HVP in the same family as PyTorch's
`torch.autograd.functional.hvp`, not in the `backward` family.

## Testing Policy

### `permute`

- dense transpose preserves values and shape
- diagonal transpose is stable
- multi-class structured permutation works
- invalid permutations error
- reverse-mode metadata survives through the view op

### `conj`

- real tensors unchanged
- complex tensors conjugated
- AD mode preserved

### `try_scalar_value`

- rank-0 real and complex tensors round-trip
- non-rank-0 errors

### snapshot

- `primal_snapshot()` drops AD metadata
- `snapshot::DynTensor` reports scalar type, dims, and axis classes correctly
- `snapshot::DynTensor::to_dense()` materializes structured snapshots

### `mode`

- primal / forward / reverse tensors report the right `AdMode`

### HVP

Minimal frontend contract tests:

1. single-input quadratic
   - `f(x) = sum(x^2)` gives `H v = 2v`
2. two-input separable quadratic
   - `f(x, y) = sum(x^2) + 3 sum(y^2)`
   - verifies one HVP per requested input
3. non-scalar output
   - `functional::hvp` errors when `func` does not return rank-0

## Documentation Updates

Update:

- `tenferro/src/lib.rs`
- `tenferro/src/core/dynamic/dyn_ad_tensor/*.rs`
- `tenferro/src/core/dynamic/dyn_tensor.rs` docs as moved into `snapshot`
- crate-level docs and examples

The docs should teach this split:

- `Tensor` for computation
- `snapshot::DynTensor` for export / storage / bridge code
- `functional::hvp` for side-effect-free Hessian-vector products
