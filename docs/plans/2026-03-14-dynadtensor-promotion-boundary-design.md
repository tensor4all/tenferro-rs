# DynAdTensor Promotion Boundary Design

## Goal

Cover issues `#494`, `#495`, and `#496` with one coherent boundary:

- mixed real/complex promotion for `DynAdTensor::scale`, `axpby`, and `div_scalar`
- a public `DynAdTensor::promote_to(...)` helper so downstream crates do not carry their own promotion logic
- a public `DynAdTensor::primal_snapshot()` boundary so downstream storage/FFI layers do not match on internal typed variants

The design must keep `DynAdTensor` as the canonical runtime payload and avoid adding new ad hoc promotion code.

## Constraints

- Backward compatibility is not a design goal.
- `DynAdTensor` remains the canonical dynamic execution payload.
- Scalar coefficients stay modeled as rank-0 `DynAdTensor`, not a separate scalar AD type.
- Promotion is intentionally narrow: same-precision real-to-complex only.
- Storage/FFI boundaries must explicitly drop AD metadata.

## Public API

Add two public APIs on `DynAdTensor`.

```rust
impl DynAdTensor {
    pub fn promote_to(&self, target: ScalarType) -> Result<Self>;
    pub fn primal_snapshot(&self) -> Result<DynStructuredPrimal>;
}
```

Add a new public snapshot enum:

```rust
pub enum DynStructuredPrimal {
    F32(StructuredTensor<f32>),
    F64(StructuredTensor<f64>),
    C32(StructuredTensor<num_complex::Complex32>),
    C64(StructuredTensor<num_complex::Complex64>),
}
```

This boundary is intentionally dtype-dynamic and structure-preserving.
It does not pretend that the structured model is only `dense` or `diag`.
Downstream code that wants a denser storage enum may derive it from
`StructuredTensor` properties, but upstream should expose the full structured
payload honestly.

## Promotion Semantics

Supported promotions:

- identity: `T -> T`
- same-precision real-to-complex:
  - `F32 -> C32`
  - `F64 -> C64`

Unsupported for now:

- real precision changes: `F32 <-> F64`
- complex precision changes: `C32 <-> C64`
- complex-to-real narrowing: `C32 -> F32`, `C64 -> F64`

This is an algebraic promotion rule, not a general numeric cast system.

For mixed tensor/scalar ops:

- `scale(self, scalar)` uses the join of `self.scalar_type()` and
  `scalar.scalar_type()`
- `div_scalar(self, scalar)` uses the same join
- `axpby(a, self, b, other)` uses the join of `self`, `other`, `a`, and `b`

Because cross-precision joins remain unsupported, the only mixed successes are:

- all operands in `{F32, C32}` with at least one `C32` -> `C32`
- all operands in `{F64, C64}` with at least one `C64` -> `C64`

## AD Semantics

`promote_to(...)` must preserve AD mode and metadata.

- primal stays primal
- forward keeps tangent
- reverse keeps tape/node linkage

This means promotion is not implemented as detach-and-rebuild logic. It is an
AD-aware payload lift.

`primal_snapshot()` is the opposite kind of boundary:

- it intentionally drops AD metadata
- it preserves only the primal `StructuredTensor<T>`

This keeps the storage/export contract explicit and simple.

## Internal Structure

The current `dyn_ad_tensor/scalar_ops.rs` mixes operation logic and promotion
logic. That is the ad hoc part we want to remove.

Split the implementation into:

- `core/dynamic/dyn_ad_tensor/promotion.rs`
  - dtype join logic
  - `promote_to(...)`
  - typed real-to-complex lifting helpers
- `core/dynamic/dyn_ad_tensor/snapshot.rs`
  - `DynStructuredPrimal`
  - `primal_snapshot()`
- `core/dynamic/dyn_ad_tensor/scalar_ops.rs`
  - `scale`, `axpby`, `div_scalar`
  - consumes promotion helpers, does not own promotion policy

This gives one source of truth for promotion semantics.

## Testing Strategy

The tests should mirror the public contract.

### Promotion tests

- identity promotion is a no-op
- `F32 -> C32` and `F64 -> C64` succeed
- unsupported promotions return clear errors
- forward metadata is preserved
- reverse tape compatibility is preserved

### Mixed op tests

- `scale`, `axpby`, and `div_scalar` all use the same join rule
- mixed real/complex same-precision cases succeed
- unsupported cross-precision cases error
- AD metadata remains attached

### Snapshot tests

- dense snapshot preserves payload and dims
- structured snapshot preserves `axis_classes`
- snapshot drops AD metadata by construction

## Docs

Update:

- rustdoc on `DynAdTensor`
- rustdoc on `DynStructuredPrimal`
- `docs/api_index.md`
- `docs/design/autodiff.md`
- `docs/design/supported-ops.md`

The docs should explain:

- rank-0 tensor scalar semantics
- canonical `DynAdTensor` promotion path
- primal snapshot as the intended storage/FFI boundary
