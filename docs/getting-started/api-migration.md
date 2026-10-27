# API migration guide

This page is the first stop when an older tenferro-rs example fails with
`cannot find module`, `cannot find function`, or a changed-signature error.
The current public API favors explicit extension traits, fallible constructors,
and caller-held engines. Historical worklogs are not API documentation.

## Removed modules and free functions

| Older spelling | Current spelling |
| --- | --- |
| <code>tenferro_einsum::&#8203;eager_tensor</code> | Import `tenferro_einsum::EagerSessionEinsumExt` and call the trait method on a borrowed session: `ctx.with_eager_session(\|s\| s.einsum(&[&a, &b], "ij,jk->ik"))?`. |
| `tenferro_einsum::EagerEinsumExt` (`[&a, &b].einsum(...)` on eager values), `EagerTensorEinsumExt` (`a.tensordot(&b, axes)`) | `EagerSessionEinsumExt` on a borrowed session: `s.einsum(&[&a, &b], "ij,jk->ik")`, `s.einsum_notation(...)`, `s.einsum_subscripts(...)`, `s.tensordot(&a, &b, axes)`. |
| <code>tenferro_linalg::&#8203;eager_tensor</code> | Import `tenferro_linalg::EagerTensorLinalgExt` and call the trait method on the eager value. |
| <code>tenferro_runtime::&#8203;traced_tensor</code> | Use the current traced tensor types and their extension traits, such as `tenferro_linalg::TracedTensorLinalgExt`. |
| `tenferro_einsum::einsum` | Use the current trait method, for example `TraceContextEinsumExt::einsum` or `TracedTensorEinsumExt::einsum`, for the receiver you have. |
| `tenferro_einsum::einsum_subscripts_with` | Import the owning einsum extension trait and call its session/context method. |
| `tenferro_linalg::svd`, `qr`, `eigh`, `solve` | Import `TensorLinalgExt`, `TypedTensorLinalgExt`, or `TensorReadLinalgExt` and call `.svd(...)`, `.qr(...)`, `.eigh(...)`, or `.solve(...)` on the input. |
| `tenferro_tensor::HostTensor<T>` / `HostTensorView<'a, T>` | `TypedTensor<T, DynRank, Host>` (`from_host_vec_col_major`, `as_slice`, `host_data_mut`) and `TypedTensorView<'a, T, DynRank, Host>` (`from_host_slice`, `as_host_slice`). Scalar-set members and `ErasedHostTensor` payloads use the same types. |
| `tenferro_tensor::core::{DefaultScalars, HostTensor, ...}` | `tenferro_tensor::core` re-exports metadata only; import tensor types, including `DefaultScalars`, from the `tenferro_tensor` root. |
| `TypedTensorView::try_slice`, `try_slice_axis`, `try_reshape` (and the `TypedTensorViewMut` forms) | `slice_view`, `slice_axis_view`, `reshape_view`. |
| Owner-side execution through a backend-owned copy/canonicalization (`to_contiguous` / `copy_into`, `TensorFusion`, `reclaim_buffer`, `with_linalg_pool`, and the CUDA/WebGPU equivalents) | Call the method on the engine you hold: `engine.to_contiguous_read(read)?`. The `CpuEngine` / `CudaEngine` value replaces the old backend handle, and engine construction is `CpuEngine::new(pool)` / `CpuEngine::serial()` / `CudaEngine::open(ordinal)`. |
| The removed backend-session host and its entry wrappers, held sessions and phase leases | Delete the wrapper and pass `&mut engine` to the operation. There is no entry protocol, no nested-entry error and no held-session type to hold. |
| The removed compiled-run entry points on `Runtime` (the compiled, prepared and deferred run calls) | `let prepared = runtime.prepare(&program, &reads, &[engine.target()])?;` then `runtime.run(&mut [&mut engine], &prepared, &reads)?`; turn each `Output` into a `Tensor` with `output.into_tensor(&mut engine)?`. |
| An eager runtime that owned its backend (backend-owning constructor, CPU placement, a raw execution-session entry) | `EagerRuntime::new()` stores no engine: call `ctx.with_eager_session(&mut engine, \|s\| ...)`, and pass `&mut engine` to `grad` / `vjp` / `jvp` / `backward` / `backward_with` / `to_tensor`. |
| Fixed thread-count backends and CPU placement values (auto/all-allowed/NUMA-node), CPU-set and CPU-id selectors | Build the pool yourself: `CpuEngine::new(tenferro_cpu::Pool::builder().threads(n).build()?)`, or `CpuEngine::serial()` for no pool. `Pool` is `threadteam::Pool`, re-exported by `tenferro-cpu` together with the `threadteam` crate; the old `thread_pool_builder` preset, `CpuEngine::pool`, `CpuEngine::pool_stats` and `CpuEngineError::CallerInOwnPool` no longer exist. tenferro no longer constructs pools, resolves placement or verifies affinity. |
| Cloneable backend handles that shared internal state | The engine is a value, not a handle; hold one and borrow it with `&mut`. Sharing work across threads means one engine per thread or per pool. |

The owning crate's `prelude` re-exports the public operation traits. For a
first direct CPU program, the usual imports are:

```rust
use tenferro_cpu::CpuEngine;
use tenferro_linalg::prelude::*;
use tenferro_runtime::prelude::*;
```

Then borrow one engine for every call:

```rust
let mut engine = CpuEngine::serial();
let values = input.svdvals(&mut engine)?;
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

## Changed signatures (2026-10)

| Older spelling | Current spelling |
| --- | --- |
| `ctx.with_eager_session(\|s\| s.neg(&x))??` (nested `Result`) | `ctx.with_eager_session(&mut engine, \|s\| s.neg(&x))?`: the callback returns `Result<T, E>` with `E: From<tenferro_ad::Error>`, and an unregistered engine error arrives as `E::from(..)`. Annotate `Ok::<_, tenferro_ad::Error>(..)` when the error type is not otherwise inferred; map tensor-level errors with `.map_err(tenferro_ad::Error::from)`. |
| `ctx.with_eager_session(\|s\| ...)` without an engine | `ctx.with_eager_session(&mut engine, \|s\| ...)`: `EagerRuntime` stores no engine, so the caller passes one. |
| `loss.backward()`, `y.backward_with(&seed)`, `ctx.grad(&loss, &x)`, `ctx.vjp(&y, &x, &seed)`, `ctx.jvp(&y, &x, &tangent)`, `value.to_tensor()` | Each takes the engine first: `backward(&mut engine)`, `backward_with(&mut engine, &seed)`, `grad(&mut engine, &loss, &x)`, `vjp(&mut engine, ...)`, `jvp(&mut engine, ...)`, `to_tensor(&mut engine)`. |
| `reduce_sum_squares(&x, &[0])` (eager), `x.reduce_sum_squares(&[0])` (traced) | `Some(&[0])`; `None` now reduces every axis, as for `reduce_sum`. |
| `x.reduce_sum(&[1], session)` on `Tensor` / `TypedTensor` | `x.reduce_sum(Some(&[1]), session)`. |
| `session.gather(&x, &idx, &cfg)` and other core ops only on the session | Also `x.gather(&idx, cfg, session)` via `TensorSessionOpsExt` (see the session-surface notes in the guides). |
| `Tensor::from_vec_col_major` with row-major data | `Tensor::from_vec_row_major(shape, data)` reorders once into column-major storage. |
| `compile_with_input_specs` with a placeholder the output does not use | Rejected at compile time (`InvalidArgument` naming the binding); drop that binding. |
| `?` on `EagerFftInPlaceError` in a `tenferro_ad::Result` function | Works: `From<EagerFftInPlaceError> for tenferro_ad::Error` drops a rejected input. |

## Finding the current method

1. Choose the value tier: direct concrete tensor, eager tensor, or traced tensor.
2. Import the `*Ext` trait owned by the operation crate.
3. Check the method's receiver and engine arity in the
   [API cheatsheet](https://tensor4all.org/tenferro-rs/skill-references/api-cheatsheet.md).
4. Use the [linear algebra guide](../guides/linear-algebra.md),
   [einsum guide](../guides/einsum.md), or
   [custom operations guide](../guides/custom-operations.md) for the relevant
   workflow.

Do not add a compatibility alias for a removed API. If a current example or
error message contradicts this page, report the documentation gap through the
[issue-intake procedure](https://github.com/tensor4all/tenferro-rs/blob/main/ai/contribution-workflows/issue-intake.md)
after obtaining maintainer/user approval.
