# API migration guide

This page is the first stop when an older tenferro-rs example fails with
`cannot find module`, `cannot find function`, or a changed-signature error.
The current public API favors explicit extension traits, fallible constructors,
and session-owned execution. Historical worklogs are not API documentation.

## Removed modules and free functions

| Older spelling | Current spelling |
| --- | --- |
| <code>tenferro_einsum::&#8203;eager_tensor</code> | Import `tenferro_einsum::EagerEinsumExt` and call the trait method on the eager value. |
| <code>tenferro_linalg::&#8203;eager_tensor</code> | Import `tenferro_linalg::EagerTensorLinalgExt` and call the trait method on the eager value. |
| <code>tenferro_runtime::&#8203;traced_tensor</code> | Use the current traced tensor types and their extension traits, such as `tenferro_linalg::TracedTensorLinalgExt`. |
| `tenferro_einsum::einsum` | Use the current trait method, for example `TraceContextEinsumExt::einsum` or `TracedTensorEinsumExt::einsum`, for the receiver you have. |
| `tenferro_einsum::einsum_subscripts_with` | Import the owning einsum extension trait and call its session/context method. |
| `tenferro_linalg::svd`, `qr`, `eigh`, `solve` | Import `TensorLinalgExt`, `TypedTensorLinalgExt`, or `TensorReadLinalgExt` and call `.svd(...)`, `.qr(...)`, `.eigh(...)`, or `.solve(...)` on the input. |

The owning crate's `prelude` re-exports the public operation traits. For a
first direct CPU program, the usual imports are:

```rust
use tenferro_cpu::CpuBackend;
use tenferro_linalg::prelude::*;
use tenferro_runtime::prelude::*;
```

Then keep execution inside one session:

```rust
let mut backend = CpuBackend::new();
let values = backend.with_backend_session(|session| input.svdvals(session))?;
```

## Autodiff context changes

`AdContextBuilder::with_core_rules()` was removed. The core primitive rules
are installed by the normal builder; start with:

```rust
let ad = tenferro_ad::AdContext::builder().build()?;
```

When an operation family supplies semantic AD rules, install that family
explicitly instead:

```rust
let ad = tenferro_ad::AdContext::builder()
    .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules()?)?
    .build()?;
```

## Fallible constructors and shape APIs

Constructors that validate storage, placement, or metadata return `Result`.
Propagate the result instead of relying on an infallible constructor:

| Older assumption | Current spelling |
| --- | --- |
| `EagerTensor::from_tensor_in(tensor, ctx)` returns a value | `EagerTensor::from_tensor_in(tensor, ctx)?` |
| `EagerTensor::requires_grad_in(tensor, ctx)` returns a value | `EagerTensor::requires_grad_in(tensor, ctx)?` |
| `TracedTensor::input_concrete_shape(dtype, shape)` is infallible | `TracedTensor::input_concrete_shape(dtype, shape)?` |
| `TypedTensor::from_vec_col_major(shape, data)` is infallible | `TypedTensor::from_vec_col_major(shape, data)?` |
| `TypedTensor::zeros(shape)` is infallible | `TypedTensor::zeros(shape)?` |

`reduce_sum` uses an explicit optional axis list on eager values:

```rust
let total = value.reduce_sum(None)?;          // all axes
let columns = value.reduce_sum(Some(&[0]))?;  // selected axes
```

An empty slice remains distinct from `None`: use `Some(&[])` when the API's
identity/no-axis behavior is what the program needs.

## Finding the current method

1. Choose the value tier: direct concrete tensor, eager tensor, or traced tensor.
2. Import the `*Ext` trait owned by the operation crate.
3. Check the method's receiver and session arity in the
   [API cheatsheet](https://tensor4all.org/tenferro-rs/skill-references/api-cheatsheet.md).
4. Use the [linear algebra guide](../guides/linear-algebra.md),
   [einsum guide](../guides/einsum.md), or
   [custom operations guide](../guides/custom-operations.md) for the relevant
   workflow.

Do not add a compatibility alias for a removed API. If a current example or
error message contradicts this page, report the documentation gap through the
[issue-intake procedure](https://github.com/tensor4all/tenferro-rs/blob/main/ai/contribution-workflows/issue-intake.md)
after obtaining maintainer/user approval.
