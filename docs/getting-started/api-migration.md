# API migration guide

This page is the first stop when an older tenferro-rs example fails with
`cannot find module`, `cannot find function`, or a changed-signature error.
The current public API uses one call grammar per operation family (0.12),
fallible constructors, and caller-held engines. Historical worklogs are not API documentation.

## 0.12: one call grammar (#2552)

Version 0.12 replaces the per-receiver extension traits with one call grammar
for every operation family. The generated
[operation registry](../spec/generated/api-ops.md) is the reference for every
operation, its options type, `_into` modes, layout handling, errors, and
omissions.

```text
recv.op(operands…, exec)                     // concrete, default options
recv.op_with(operands…, &opts, exec)         // concrete, explicit XxxOptions (only ops with options)
recv.op_into(operands…, [&opts,] exec, dest) // concrete, only where declared; no &opts for ops without options
session.op(&x, operands…)                    // eager: the session is the receiver
traced.op_with(operands…, &opts)             // traced: no exec; run with prepare → run
```

- The engine (`exec: &mut dyn Engine`) is always the last concrete argument.
- A concrete receiver is any `AsView` value: `Tensor`, `TensorRead`,
  `TypedTensor<T>`, `TypedTensorView<T>`, or a reference to one. The `_read`
  spellings are gone; pass the view to the unsuffixed method. Typed receivers
  return typed results.
- An operation has an `XxxOptions` type and an `_with` form only when it has
  semantic options; `solve`, `lstsq`, `householder_qr`, `lu`, `full_piv_lu`,
  `full_piv_lu_solve`, `eig`, `eigvals`, `slogdet`, `det`, `inv` and `add` have
  none, so their `_into` forms take no options (#2552 Q14).
- The only suffixes are `_with` (explicit options) and `_into` (a caller
  destination `Dest`); `_in_place` remains only for the consuming eager FFT.
- Options types are `#[non_exhaustive]`: build them with
  `XxxOptions::default()` and setters named after the fields.
- Axis arguments are `isize`; a negative axis counts from the end.
- Multi-output results are structs with named fields.

### Traits

| Removed | 0.12 |
| --- | --- |
| `TensorLinalgExt`, `TensorReadLinalgExt`, `TypedTensorLinalgExt` | `tenferro_linalg::LinalgOps` |
| `TracedTensorLinalgExt`, `TraceContextLinalgExt` | `tenferro_linalg::TracedLinalgOps` on `TracedTensor` |
| `EagerSessionLinalgExt`, `EagerTensorLinalgExt` | `tenferro_linalg::EagerLinalgOps` on `EagerSession` |
| `TensorEinsumExt`, `TypedTensorEinsumExt`, `TensorReadEinsumExt`, `TypedTensorReadEinsumExt`, and their `*EinsumIntoExt` variants | `tenferro_einsum::EinsumOps` on a slice, array, or `Vec` of inputs |
| `TensorTensordotExt`, `TypedTensorTensordotExt` | `tenferro_einsum::TensordotOps` |
| `TraceContextEinsumExt` | `tenferro_einsum::TracedEinsumOps` on `TraceContext` (takes and returns `TracedTensor`) |
| `TracedTensorEinsumExt` (`tensordot`) | `tenferro_einsum::TracedTensordotOps` on `TracedTensor` |
| `EagerSessionEinsumExt` | `tenferro_einsum::EagerEinsumOps` on `EagerSession` |
| `TensorFftExt`, `TensorReadFftExt` | `tenferro_fft::FftOps` |
| `TracedTensorFftExt`, `TraceContextFftExt` | `tenferro_fft::TracedFftOps` on `TracedTensor` |
| `EagerSessionFftExt`, `EagerTensorFftExt` | `tenferro_fft::EagerFftOps` on `EagerSession` |
| `TensorSessionOpsExt`, `TypedTensorSessionOpsExt`, `TypedTensorMaskSessionOpsExt` | `tenferro_runtime::CoreOps` |

Traced core operations stay inherent `TracedTensor` methods, and eager core
operations stay inherent `EagerSession` methods; only their axis types and the
conjugated `dot_general` spelling changed. Each family's `prelude` re-exports
its new traits.

### Linear algebra

| Older spelling | 0.12 |
| --- | --- |
| `let (u, s, vh) = a.svd(exec)?` | `let Svd { u, s, vh } = a.svd(exec)?` |
| `(q, r)`, `(p, l, u, parity)`, `(p, l, u, q, parity)`, `(values, vectors)`, `(sign, logabsdet)` | `Qr { q, r }`, `Lu { p, l, u, parity }`, `FullPivLu { p, l, u, q, parity }`, `Eig { values, vectors }` / `Eigh { values, vectors }`, `Slogdet { sign, log_abs_det }` |
| `TypedSvd<T>`, `TypedLu<T>`, `TypedFullPivLu<T>`, `TypedEig<T>`, `TypedRankRevealingQrResult<T>` | The same structs with typed fields, for example `Svd<TypedTensor<T>, TypedTensor<T::Real>>`; `RankRevealingQrResult` and `HouseholderQr` keep their names |
| `r.svd_read(exec)`, `r.qr_read(exec)`, any `op_read(..)` on a `TensorRead` | `r.svd(exec)`, `r.qr(exec)`: the same name without `_read` |
| `a.svd_full(exec)` | `a.svd_with(&SvdOptions::default().shape(SvdShape::Full), exec)` |
| `a.svd_with_options(opts, exec)`, `qr_with_options`, `eigh_with_options` | `a.svd_with(&opts, exec)`, `a.qr_with(&opts, exec)`, `a.eigh_with(&opts, exec)` |
| `a.rank_revealing_qr(opts, exec)` | `a.rank_revealing_qr_with(&opts, exec)` |
| `a.pinv_with_rtol(rtol, exec)` | `a.pinv_with(&PinvOptions::default().rtol(Some(rtol)), exec)` |
| `a.triangular_solve(&b, left_side, lower, transpose_a, unit_diagonal, exec)` | `a.triangular_solve_with(&b, &TriangularSolveOptions::default().side(..).uplo(..).op(..).diag(..), exec)` with `Side::{Left, Right}`, `Uplo::{Lower, Upper}`, `MatOp::{None, Transpose}`, `Diag::{NonUnit, Unit}`; `a.triangular_solve(&b, exec)` is left, lower, no transpose, non-unit |
| `a.norm(ord, dim, keepdim, exec)` | `a.norm(exec)` for the Frobenius default, otherwise `a.norm_with(&NormOptions::default().ord(NormOrd::..).axes(Some([0].into())).keep_dims(KeepDims::Yes), exec)`; `None → NormOrd::Frobenius`, `Some(0.0) → NormOrd::Zero`, `Some(±∞) → NormOrd::Inf` / `NegInf`, other `Some(p) → NormOrd::P(p)` |
| `r.solve_read_into(b, out, exec)` | `r.solve_into(&b, exec, Dest::overwrite(out))` (overwrite only) |
| `ctx.svd(value)` (`TraceContextLinalgExt` on `TraceValue`) | `t.svd()` on a `TracedTensor` |
| `t.svd_full()`, `t.norm(ord, dim, keepdim)`, `t.triangular_solve(&b, ..)` (traced) | `t.svd_with(&..)`, `t.norm_with(&NormOptions ..)`, `t.triangular_solve_with(&b, &TriangularSolveOptions ..)` |
| `x.solve(engine, &b)` (`EagerTensorLinalgExt`) | `runtime.with_eager_session(&mut engine, \|s\| s.solve(&x, &b))` |
| `session.svd_full(&x)`, `session.pinv_with_rtol(&x, r)`, `session.norm(&x, ord, dim, keepdim)` | `session.svd_with(&x, &..Full)`, `session.pinv_with(&x, &PinvOptions::default().rtol(Some(r)))`, `session.norm_with(&x, &NormOptions ..)` |

### Lanes

| Older spelling | 0.12 |
| --- | --- |
| `svd_with_exec`, `svd_values_with_exec`, `qr_with_exec`, `eigh_with_exec`, `eigh_values_with_exec`, `eig_values_with_exec` inside `engine.with_exec(\|exec\| ..)` | `engine.with_lanes(&shares, &mut pools, \|i, lane\| view.svd(lane))` returns `Result<Vec<R>>`; a full SVD is `view.svd_with(&SvdOptions::default().shape(SvdShape::Full), lane)` |
| Lanes built by hand (`exec.token().fork_sized(..)`, `Exec::from_token`, `*_with_exec`) | `CpuEngine::with_lanes(&shares, &mut pools, \|i, lane\| ..)`: one caller-owned `BufferPool` per lane; only `svd`, `svdvals`, `qr`, `eigh`, `eigvalsh`, and `eigvals` run in a lane, every other operation returns `Unsupported` |

### Einsum

| Older spelling | 0.12 |
| --- | --- |
| `[&a, &b].einsum_notation(&n, exec)`, `einsum_subscripts(&s, exec)` | `[&a, &b].einsum(&n, exec)`, `[&a, &b].einsum(&s, exec)`: the spec argument accepts a string, an `EinsumNotation`, or `EinsumSubscripts` |
| `[r1, r2].einsum_read(spec, exec)` and typed-read variants | `[r1, r2].einsum(spec, exec)` |
| `[r1, r2].einsum_read_into("ij,jk->ik", exec, out)`, `einsum_into(spec, exec, out)` | `[r1, r2].einsum_into("ij,jk->ik", &EinsumOptions::default(), exec, Dest::overwrite(out))` |
| `a.tensordot(&b, axes, exec)` (`TensorTensordotExt`) | the same call through `TensordotOps` (`tensordot` has no options, so no `tensordot_with`) |
| `ConcreteEinsumPlan::prepare(inputs, spec)`, `prepare_notation`, `prepare_subscripts`, `prepare_typed*`, `prepare_read*` | `ConcreteEinsumPlan::prepare(inputs, spec, &EinsumOptions::default())` |
| `plan.execute_typed(..)`, `execute_read(..)`, `execute_typed_into(.., out)`, `execute_read_into(.., out)` | `plan.execute(inputs, exec)`, `plan.execute_into(inputs, exec, Dest::overwrite(out))` |
| `plan.execute_read_into_accum(inputs, exec, acc, out)` | `plan.execute_into(inputs, exec, Dest::accumulate(out))` or `Dest::scaled(out, alpha, beta)`; per-operand conjugation is fixed when the plan is prepared, through `EinsumOptions::conj` |
| `ctx.einsum(&[v1, v2], spec)` and the `_notation` / `_subscripts` variants (`&mut TraceContext`, `TraceValue`) | `trace.einsum(&[&t1, &t2], spec)` on `&TraceContext`, taking and returning `TracedTensor` |
| `ctx.einsum_subscripts_with(&[v1, v2], s, optimize)` | `trace.einsum_with(&[&t1, &t2], &s, &EinsumOptions::default().optimize(optimize))` |
| `session.einsum_notation(..)`, `session.einsum_subscripts(..)` (eager) | `session.einsum(&[&a, &b], &spec)` through `EagerEinsumOps` |

Concrete einsum and plan errors are now `tenferro_tensor::Error` (an einsum
domain error is reachable through `source()`), traced and eager einsum errors
are `tenferro_runtime::Error`, and an invalid spec is
`Validation(InvalidArgument)` on every layer.

Code that built traced einsum through `TraceContextEinsumExt` and `TraceValue`
(for example yao-rs, which pins tenferro `=0.4.0`) migrates to `TracedTensor`
values: create inputs on a `TraceContext`, call `trace.einsum(&[&a, &b], spec)`,
compile the outputs with `GraphCompiler`, then execute with
`Runtime::prepare` → `Runtime::run` after installing
`tenferro_einsum::extension_module` on the runtime. There is no public
conversion from `TracedTensor` to `TraceValue`; the `TraceContext` IR builder
(`input`, `add_op`, `add_extension`, `finish`) remains for code that emits IR
directly.

### FFT

| Older spelling | 0.12 |
| --- | --- |
| `a.fft(None, -1, FftNorm::Backward, exec)` | `a.fft(exec)` |
| `a.fft(n, axis, norm, exec)` (likewise `ifft`, `rfft`, `irfft`) | `a.fft_with(&FftOptions::default().n(n).axis(axis).norm(norm), exec)` |
| `r.fft_read(n, axis, norm, exec)` | `r.fft_with(&opts, exec)` on the view |
| `t.fft(n, axis, norm)` (traced), `session.fft(&x, n, axis, norm)` (eager) | `t.fft_with(&opts)`, `session.fft_with(&x, &opts)` |
| `ctx.fft(value, ..)` (`TraceContextFftExt`) | `t.fft_with(&opts)` on a `TracedTensor` |
| `x.fft_in_place(engine, axis, norm)`, `ifft_in_place` (`EagerTensorFftExt`) | `session.fft_in_place(x, &FftOptions::default().axis(axis).norm(norm))`; an `n: Some(..)` or a real input is `Rejected` with the input returned |
| `executor.fft(&input, n, axis, norm, exec)` (`FftExecutor`, all eight transforms) | `executor.fft_with(&input, &opts, exec)` |
| Multidimensional FFT only on the traced layer | `fftn`, `ifftn`, `rfftn`, `irfftn` (and `_with(&FftnOptions)`) on all three layers; empty `axes` selects every axis |

Error labels follow the new names, for example `fft` instead of
`TensorFftExt::fft` and `FftExecutor::fft_with` for the executor.

### Core operations and signed axes

| Older spelling | 0.12 |
| --- | --- |
| `x.exp(exec)`, `a.matmul(&b, exec)`, … via `TensorSessionOpsExt` / `TypedTensorSessionOpsExt` | the same calls via `CoreOps`, for owned tensors, views, and typed tensors alike |
| `cond.where_select(&t, &f, exec)` via `TypedTensorMaskSessionOpsExt` | the same call via `CoreOps`; the receiver is the `bool` condition |
| `Tensor::concatenate(&[&a, &b], axis, exec)`, `TypedTensor::<T>::concatenate(..)` | `CoreOps::concatenate(&[&a, &b], axis, exec)` with `axis: isize` |
| `a.dot_general_with_conj(&b, cfg, true, false, exec)` | `a.dot_general_with(&b, cfg, &DotGeneralOptions::default().lhs(Conj::Yes), exec)` |
| `session.dot_general_with_conj(&a, &b, cfg, lhs, rhs)` (eager) | `session.dot_general_with(&a, &b, cfg, &DotGeneralOptions ..)` |
| `a.add_into(&rhs, &mut out, exec)` (typed) | `a.add_into(&rhs, exec, Dest::overwrite(&mut out))` (overwrite only) |
| `a.reduce_sum(Some(&[0usize]), exec)` and every other `usize` / `&[usize]` axis argument | `a.reduce_sum(Some(&[0isize]), exec)`; `-1` names the last axis. This applies to `CoreOps`, `TracedTensor` (including `slice_axis`, `take_axis`, `concatenate`, `extract_diag`, `embed_diag`, `shape_of`, `sym_size`, `dynamic_truncate`, `pad_to_match`), and `EagerSession` |

A repeated axis in a concrete reduction, `transpose`, or `reverse` is rejected
with `DuplicateAxis`. `convert` and `cast` accept views and return a
dtype-erased `Tensor`. Engine, backend, and provider hooks (`Engine`,
`TensorDot`, `LinalgBackend`, `FftBackend`, and their `*_read` /
`*_read_into_accum` methods) are provider contracts and keep their names.

### Output destinations

`Dest` (from `tenferro_tensor`) replaces the `_read_into`, `_add_to`, and
`_into_accum` spellings: `Dest::overwrite(out)` never reads the destination,
`Dest::accumulate(out)` adds the result to it, and
`Dest::scaled(out, alpha, beta)` writes `alpha * result + beta * out`. Each
`_into` operation declares which modes it supports (`solve_into` and
`add_into` overwrite only; `einsum_into` and `plan.execute_into` support all
three), and an unsupported mode is refused with `Unsupported` before any write.

The sections below describe earlier migrations. They are kept as history: where
their right-hand column names a trait or `_read` method that 0.12 removed, use
the tables above.

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
| `ctx.constant_from(tensor)` returns a value | `ctx.constant_from(tensor)?` |
| `ctx.variable_from(tensor)` returns a value | `ctx.variable_from(tensor)?` |
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
| `loss.backward()`, `y.backward_with(&seed)`, `ctx.grad(&loss, &x)`, `ctx.vjp(&y, &x, &seed)`, `ctx.jvp(&y, &x, &tangent)`, `value.to_tensor()` | Session methods since #2602; see the next table. |
| `reduce_sum_squares(&x, &[0])` (eager), `x.reduce_sum_squares(&[0])` (traced) | `Some(&[0])`; `None` now reduces every axis, as for `reduce_sum`. |
| `x.reduce_sum(&[1], session)` on `Tensor` / `TypedTensor` | `x.reduce_sum(Some(&[1]), session)`. |
| `session.gather(&x, &idx, &cfg)` and other core ops only on the session | Also `x.gather(&idx, cfg, session)` via `TensorSessionOpsExt` (see the session-surface notes in the guides). |
| `Tensor::from_vec_col_major` with row-major data | `Tensor::from_vec_row_major(shape, data)` reorders once into column-major storage. |
| `compile_with_input_specs` with a placeholder the output does not use | Rejected at compile time (`InvalidArgument` naming the binding); drop that binding. |
| `?` on `EagerFftInPlaceError` in a `tenferro_ad::Result` function | Works: `From<EagerFftInPlaceError> for tenferro_ad::Error` drops a rejected input. |

## Eager AD, value and leaf entry points on the session (#2602)

Every eager operation that needs an engine is an `EagerSession` method, so the
engine is passed once, to `with_eager_session`, and every operand and seed is
checked against the session's runtime before anything is materialized or a
gradient slot is written.

| Removed spelling | Current spelling |
| --- | --- |
| `loss.backward(&mut engine)` | `ctx.with_eager_session(&mut engine, \|s\| s.backward(&loss))` |
| `y.backward_with(&mut engine, &seed)` | `s.backward_with(&y, &seed)` |
| `ctx.grad(&mut engine, &loss, &x)`, `grad_optional` | `s.grad(&loss, &x)`, `s.grad_optional(&loss, &x)` |
| `ctx.vjp(&mut engine, &y, &x, &seed)`, `vjp_optional` | `s.vjp(&y, &x, &seed)`, `s.vjp_optional(..)` |
| `ctx.jvp(&mut engine, &y, &x, &tangent)`, `jvp_optional` | `s.jvp(&y, &x, &tangent)`, `s.jvp_optional(..)` |
| `x.detach(&mut engine)` | `s.detach(&x)` |
| `x.to_tensor(&mut engine)`, `x.duplicate_value(&mut engine)` | `s.duplicate_value(&x)` (one owned-copy spelling); `x.into_value()` still consumes the handle without copying |
| `x.detach_into(&mut engine, &ctx_b)` | `let t = ctx_a.with_eager_session(&mut engine, \|s\| s.duplicate_value(&x))?;` then `ctx_b.with_eager_session(&mut engine, \|s\| s.constant_from(t))?` |
| `EagerTensor::from_tensor_in(t, ctx)`, `EagerTensor::requires_grad_in(t, ctx)` | `ctx.constant_from(t)`, `ctx.variable_from(t)` (engine-free, owned host tensors) or `s.constant_from(t)` / `s.constant_from_host(t)` / `s.variable_from(t)` (any input the engine can read) |
| `EagerTensor::from_vec_col_major_in(shape, data, ctx)` | `ctx.constant_from(Tensor::from_vec_col_major(shape, data)?)` |
| `s.backend_session()` | `s.engine()` (`&mut dyn ExecutionEngine`, which is also an `Engine`) |

`GradientValue::to_tensor(&mut engine)` is unchanged: a stored gradient carries
no eager-runtime identity and follows the concrete grammar (`recv.op(exec)`).


## Finding the current method

1. Choose the value tier: direct concrete tensor, eager tensor, or traced tensor.
2. Import the operation trait owned by the operation crate (`LinalgOps`,
   `EinsumOps`, `FftOps`, `CoreOps`, or their `Traced*` / `Eager*` forms), or
   the crate's `prelude`.
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
