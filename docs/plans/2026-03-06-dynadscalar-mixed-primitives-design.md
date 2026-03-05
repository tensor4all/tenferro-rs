# DynAdScalar and Mixed Scalar/Tensor AD Design

**Date**: 2026-03-06
**Issues**:
- [#269](https://github.com/tensor4all/tenferro-rs/issues/269)
- [#270](https://github.com/tensor4all/tenferro-rs/issues/270)
- [#271](https://github.com/tensor4all/tenferro-rs/issues/271)

## Summary

Define `DynAdScalar` as the canonical AD-aware dynamic scalar type and make it
the scalar boundary used by `DynAdTensor` mixed primitives. Add first-class
mixed scalar/tensor operations needed by tensor-network code:

- `scale(tensor, scalar)`
- `axpby(a, x, b, y)`
- `scalar * tensor`
- `tensor / scalar`
- AD-preserving `real` / `imag` / `complex` helpers

The design keeps AD ownership inside `DynAdScalar` / `DynAdTensor`, preserves
zero-copy tensor views, and avoids duplicating scalar derivative formulas in
tensor wrappers.

## Current Problems

### 1. No canonical dynamic AD scalar type

The current implementation exposes `DynAdValue` rather than `DynAdScalar`.
Functionally it already wraps `AdValue<T>`, but the API is not shaped as a
stable downstream scalar abstraction. In particular, the public surface lacks a
clear split between:

- AD-preserving access (`primal_ref`, `tangent_ref`, metadata access)
- explicit metadata drop (`detach`, `primal_into`)

Downstream replacement of `AnyScalar` would currently depend on ad hoc use of
typed downcasts and value-returning helpers.

### 2. Reverse scalar composition is not graph-correct

`AdScalar` binary operators currently merge reverse-mode operands by keeping the
left-hand node ID rather than registering a new reverse output node. The same
pattern is reused by `DynAdValue`. This means scalar expressions that depend on
multiple reverse inputs do not form an explicit output node and can lose graph
structure for pullback propagation.

### 3. Tensor mixed primitives are declared but not implemented

`TensorKernel` already declares `scale`, `axpby`, and `inner_product`, but the
dynamic dyadtensor API does not yet expose `DynAdTensor::{scale, axpby,
div_scalar}` or operator forms for `scalar * tensor` and `tensor / scalar`.

### 4. AD-preserving helper boundary is incomplete

`DynAdTensor::{real_part, imag_part, compose_complex}` exists, but mixed
scalar/tensor workflows need the same helper family anchored on the canonical
scalar abstraction and wired through both forward- and reverse-mode paths.

## Design Goals

- Make `DynAdScalar` the canonical dynamic scalar abstraction
- Keep `DynScalar` as the primal-only dynamic scalar
- Keep AD inside `DynAdScalar` / `DynAdTensor`
- Preserve zero-copy view semantics for tensors
- Keep scalar derivative formulas in `chainrules-scalarops`
- Add broadly useful mixed scalar/tensor APIs at the dyadtensor layer
- Support primal, forward, and reverse behavior for real and complex dtypes

## Non-Goals

- No tensor4all-specific naming or `AnyScalar` compatibility layer
- No tensor4all storage or index semantics in tenferro
- No diagonal/generalized native representation work in this change
- No truncation/factorization result redesign in this change

## Type Boundary

### `DynScalar`

`DynScalar` remains the primal-only runtime scalar wrapper:

- `F32(f32)`
- `F64(f64)`
- `C32(Complex32)`
- `C64(Complex64)`

It is used for detached values, explicit metadata drop paths, and any APIs that
must be primal-only by construction.

### `DynAdScalar`

Replace `DynAdValue` with:

```rust
pub enum DynAdScalar {
    F32(AdValue<f32>),
    F64(AdValue<f64>),
    C32(AdValue<Complex32>),
    C64(AdValue<Complex64>),
}
```

This becomes the canonical public scalar representation for dynamic AD-aware
code.

### `DynAdTensor`

`DynAdTensor` remains the canonical AD-aware dynamic tensor representation. New
mixed primitive methods accept `DynAdScalar`, not `DynScalar`, so that AD is
preserved in the normal path.

## Public API

### `DynAdScalar` core API

```rust
impl DynAdScalar {
    pub fn scalar_type(&self) -> ScalarType;
    pub fn mode(&self) -> AdMode;

    pub fn primal_ref(&self) -> DynScalarRef<'_>;
    pub fn primal(&self) -> DynScalar;
    pub fn primal_into(self) -> DynScalar;

    pub fn tangent_ref(&self) -> Option<DynScalarRef<'_>>;
    pub fn tangent(&self) -> Option<DynScalar>;

    pub fn detach(&self) -> DynScalar;
    pub fn into_detached(self) -> DynScalar;

    pub fn node_id(&self) -> Option<NodeId>;
    pub fn tape_id(&self) -> Option<TapeId>;

    pub fn is_real(&self) -> bool;
    pub fn is_complex(&self) -> bool;

    pub fn conj(&self) -> Self;
    pub fn sqrt(&self) -> Self;
    pub fn powf(&self, exponent: f64) -> Self;
    pub fn powi(&self, exponent: i32) -> Self;

    pub fn real_part(&self) -> Self;
    pub fn imag_part(&self) -> Self;
    pub fn compose_complex(real: Self, imag: Self) -> Result<Self>;

    pub fn try_add(self, rhs: Self) -> Result<Self>;
    pub fn try_sub(self, rhs: Self) -> Result<Self>;
    pub fn try_mul(self, rhs: Self) -> Result<Self>;
    pub fn try_div(self, rhs: Self) -> Result<Self>;
}
```

`primal_ref` and `tangent_ref` should return a borrowed dynamic-scalar view
type rather than force value copies. `primal()` and `tangent()` remain as owned
convenience accessors.

### `DynAdTensor` mixed primitive API

```rust
impl DynAdTensor {
    pub fn scale(&self, a: &DynAdScalar) -> Result<Self>;
    pub fn axpby(&self, a: &DynAdScalar, y: &Self, b: &DynAdScalar) -> Result<Self>;
    pub fn div_scalar(&self, a: &DynAdScalar) -> Result<Self>;
}

impl core::ops::Mul<&DynAdTensor> for &DynAdScalar {
    type Output = Result<DynAdTensor>;
}

impl core::ops::Div<&DynAdScalar> for &DynAdTensor {
    type Output = Result<DynAdTensor>;
}
```

This gives a named API for explicit use plus operator sugar for the two most
common scalar/tensor forms.

## Module Layout

`extension/tenferro-dyadtensor/src/dyn_types.rs` is already too large. This
change should reduce, not increase, monolithic file size.

Recommended split:

- `src/dyn_scalar.rs`
  - `ScalarType`
  - `DynScalar`
  - borrowed scalar view helpers
- `src/dyn_ad_scalar.rs`
  - `DynAdScalar`
  - dynamic scalar AD operations
- `src/dyn_tensor.rs`
  - `DynTensor`
  - shared tensor utility helpers
- `src/dyn_ad_tensor.rs`
  - `DynAdTensor`
  - tensor mixed primitives and AD-preserving helpers
- `src/dyn_types.rs`
  - compatibility-free re-export hub for the four modules above

This is not a semantic redesign. It is a code-organization step required to
keep the implementation reviewable.

## Forward-Mode Contract

Forward-mode behavior must be preserved in all normal paths:

- scalar unary/binary ops use `chainrules-scalarops::*_frule`
- tensor mixed ops combine tensor tangents and scalar tangents in the obvious
  algebraic way
- no eager metadata drop in `scale`, `axpby`, `scalar * tensor`, or
  `tensor / scalar`

Examples:

- `scale(x, a)`:
  - primal: `a * x`
  - tangent: `da * x + a * dx`
- `axpby(a, x, b, y)`:
  - primal: `a*x + b*y`
  - tangent: `da*x + a*dx + db*y + b*dy`
- `x / a`:
  - primal: `x / a`
  - tangent from scalar `div_frule`, lifted pointwise over tensor entries

Forward-mode implementation should reuse scalar helper rules rather than
hand-code duplicated formulas at the tensor wrapper layer.

## Reverse-Mode Contract

### Scalar graph ownership

If a scalar operation consumes any reverse-tracked operand, the result must
carry a new reverse output node on the same tape. Reusing an input node ID is
not acceptable because it collapses graph structure.

All reverse operands in one scalar/tensor operation must share a tape. Mismatched
tapes remain an error.

### Tensor/scalar mixed pullbacks

For `scale(x, a)`:

- `dL/dx = scale(cotangent, conj(a))`
- `dL/da = inner_product(x, cotangent)`

For `axpby(a, x, b, y)`:

- `dL/dx = scale(cotangent, conj(a))`
- `dL/dy = scale(cotangent, conj(b))`
- `dL/da = inner_product(x, cotangent)`
- `dL/db = inner_product(y, cotangent)`

For `x / a`:

- treat the tensor side as multiplication by `1/a`
- derive the scalar side through `chainrules-scalarops::div_rrule`

### Bridge behavior for real/complex helpers

`real_part`, `imag_part`, and `compose_complex` must preserve reverse AD.
Complex-to-real projection should use the same policy already encoded in
`chainrules-scalarops::handle_r_to_c_*` instead of duplicating conversion logic
in higher-level wrappers.

### Tape implementation impact

Current reverse registration utilities are tensor-centric. Supporting canonical
mixed scalar/tensor operations requires one of:

1. scalar reverse rule registration alongside existing tensor rules, or
2. scalar-as-rank0-tensor registration internally

The smallest change is preferred. The public API should not expose whichever
internal representation is chosen.

## Dtype Semantics

Supported dtypes remain:

- `f32`
- `f64`
- `Complex32`
- `Complex64`

Mixed precision is still rejected unless an existing promotion rule already
exists. Mixed real/complex of matching precision is supported:

- `F32 <-> C32`
- `F64 <-> C64`

Cross-precision pairs such as `F32` with `F64` remain errors in this change.

## Error Handling

Use existing `thiserror`-based crate errors.

Expected error classes:

- scalar/tensor dtype mismatch
- scalar/tensor shape mismatch
- mixed reverse tapes
- unsupported mixed dtype pairs
- helper misuse such as `compose_complex` on non-real inputs

No panicking normal-path API should be added beyond existing operator sugar.
Checked methods remain the primary implementation surface.

## Testing Strategy

### `#270` DynAdScalar tests

Add tests covering:

- primal-only real and complex scalars
- forward-mode scalar tangent extraction and propagation
- reverse-mode scalar metadata access
- `detach` / `primal_into`
- `real_part`, `imag_part`, `compose_complex`
- mixed real/complex arithmetic of matching precision

### `#271` mixed primitive tests

Add tests covering:

- `scale`, `axpby`, `scalar * tensor`, `tensor / scalar`
- `f64` and `Complex64`
- forward-mode tangent propagation
- reverse-mode pullbacks returning both tensor and scalar gradients
- non-contiguous or memory-order-varied tensor views where relevant
- tensor-network-shaped examples, not only scalarized toy expressions

### Focused verification commands

- `cargo nextest run --release -p tenferro-dyadtensor`
- `cargo nextest run --release -p chainrules-scalarops`

## Boundary Check Against #269

This design does not require revising the high-level boundary in `#269`.

The only additional infrastructure implied here is a scalar-capable reverse rule
registration path inside `tenferro-dyadtensor`. That is an implementation detail
of the existing boundary, not a new architecture layer.
