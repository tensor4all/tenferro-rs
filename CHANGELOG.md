# Changelog

All notable changes to the `tenferro-*` crates are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **One host narrowing for borrowed reads: `TensorRead::try_host(op)`,
  `TensorView::try_host(op)` and `TypedTensorView::try_host(op)`.** Each
  returns a host-marked view (`TensorView<'_, Host>` /
  `TypedTensorView<'_, T, R, Host>`) when the read borrows host elements, and
  otherwise a `RuntimeState` error naming `op` (backend storage) or an
  `UnsupportedDType` error naming `op` (a `Tensor::external` payload). The
  host-marked views widen back without copying (`into_dynamic`), and
  `TensorView<'_, Host>` reports `dtype`, `shape` and `is_owned_layout`.
  `TensorView` gains a defaulted representation parameter
  (`TensorView<'a, D: Representation = Dynamic>`); every existing spelling
  `TensorView<'a>` still names the same type.
- **The same host narrowing for write targets: `TensorWrite::try_mut_host(op)`,
  `TensorViewMut::try_mut_host(op)` and `TypedTensorViewMut::try_mut_host(op)`**
  (#2499). Each returns a host-marked mutable view (`TensorViewMut<'_, Host>` /
  `TypedTensorViewMut<'_, T, R, Host>`) with the read side's errors: a
  `RuntimeState` error naming `op` for backend storage and an `UnsupportedDType`
  error naming `op` for a `Tensor::external` payload. A device or managed
  placement label on host storage resolves to host memory as `try_host` does.
  The host-marked mutable views widen back without copying (`into_dynamic`),
  and `TensorViewMut<'_, Host>` reports `dtype`, `shape` and `strides`.
  `TensorViewMut` gains a defaulted representation parameter
  (`TensorViewMut<'a, D: Representation = Dynamic>`); every existing spelling
  `TensorViewMut<'a>` still names the same type.
- **`tenferro_linalg::error::map_error` is public** (#2462). It is the
  translation tenferro's own linalg entries apply to a `tlinalg` failure, so a
  caller that runs `tlinalg` factorizations itself reports the same error
  tenferro would instead of copying a variant table. Its rustdoc states the
  contract: the `op` argument names the result, each `tlinalg::Error` variant
  maps to one documented tenferro variant and kind, and a variant a later
  `tlinalg` adds becomes `Error::Internal` with its `Debug` text. It never
  panics. Available with the `native` or `blas` feature.
- **The BLAS-provider lane boundary is documented** (#2463) in the CPU execution
  and backend-choice guides and on the `TensorDot` entries of `CpuEngine`. A
  vendor call takes the whole engine and the library's own threads, so a
  context budget is not honoured there. Lanes must not enter the vendor
  provider concurrently unless the library tolerates it. Only the native
  provider has a per-context entry, and tenferro offers no wrapper that
  simulates per-lane widths.
- **Borrowed-`Exec` factorization entries with typed outputs** (#2462):
  `svd_with_exec`, `svd_values_with_exec`, `qr_with_exec`, `eigh_with_exec`,
  `eigh_values_with_exec` and `eig_values_with_exec` in `tenferro-linalg`
  (`native` only). They run on a caller's `Exec` bundle and `BufferPool`, read a
  `[m, n]` or batched `[m, n, b..]` view in place (a negative stride is
  `Unsupported`), take every output and workspace from that pool, and return
  typed tensors: singular values and Hermitian eigenvalues in
  `LinalgScalar::RealScalar`, general eigenvalues in `LinalgScalar::Complex`.
  The engine entries now run the same functions and only erase the results, so
  both give bit-identical outputs. `LinalgScalar` gains `RealScalar`, and its
  `Complex` now also implements `PoolScalar + Default`.

### Changed

- **`sum`, `sum of squares` and `prod` accept a zero-length reduced axis and
  return the fold's own unit** (#2453). `reduce_sum` / `reduce_prod` on a
  `[0, 2]` input over axis 0 now yields `[2]` filled with `0` / `1` instead of
  the previous `InvalidArgument`, on both the owned and the borrowed-view CPU
  entries and on the CUDA entries, and for every dtype the fold supports
  (`F32`, `F64`, complex, `I32`, `I64`). The result follows the shape rule the
  operation already had: a reduced axis is empty, so every kept position is the
  identity, and a result whose kept extent is itself zero stays empty.
  `max` and `min` keep the typed `InvalidArgument`: they have no finite
  identity, and reporting a sentinel would invent a value the input never held.
  An empty *axis list* still returns the input unchanged.
- **`rem` differentiates away from its discontinuities in the semantic AD
  transforms** (#2454). `rem(x, y) = x - trunc(x / y) * y` is linear in `x` with
  slope `1` and in `y` with slope `-trunc(x / y)`, so `linearize` emits
  `dx - trunc(x / y) * dy` and the reverse rule emits `1` and
  `-trunc(x / y)` times the cotangent. `trunc(x / y)` is built in the core
  vocabulary as `q - rem(q, 1)`, so the emitted graph agrees with the primal's
  own remainder instead of depending on a second rounding mode. The rule is
  inactive for integer and complex inputs, which have no such derivative, and
  `compare` remains explicitly inactive. This changes
  `tenferro_ops::ad::support`: `Rem` is `direct_support!`, not
  `nondiff_support!`. The derivative is undefined at `y == 0` and at an exact
  integer quotient; those single points are left out rather than given a
  one-sided value.
- **The CPU linalg and reduction entries narrow each operand to `Host` once
  and no longer re-check placement below that point.** The
  `ensure_host_tensor` / `ensure_host_tensor_read` pre-checks (55 call sites in
  `tenferro-linalg`, 5 in `tenferro-cpu`) and the helpers themselves are
  removed; every `LinalgBackend` method and
  reduction entry calls `try_host` on its operands at the boundary, and the
  private entry functions and provider kernels take host-marked views. A
  backend-buffer operand is still refused with `RuntimeState` at the CPU
  boundary (the message now reads "CPU backend received a backend buffer;
  download the tensor to host before CPU execution" for linalg too). Two
  behaviour differences: a borrowed view that already covers its buffer in
  compact column-major order is no longer copied before a non-faer linalg
  route, and a host-resident operand carrying a device or managed placement
  label produces outputs labelled as host memory, as `into_host` already did.
- **`TypedTensor::set_placement` and `set_cpu_affinity` exist only on the
  `Dynamic` representation.** A `Host` or `Gpu` owner keeps the placement it was
  built with, so a statically host-owned tensor can no longer be relabelled as
  device memory; only the runtime union can carry a placement that disagrees with
  its storage. `Representation` gains `default_placement()`, the placement a
  representation's owner receives when none is supplied (unpinned host memory,
  no device, no CPU affinity, for all three). Migration: relabel through
  `into_dynamic()`, as in `let mut t = host.into_dynamic(); t.set_placement(p);`.
  Every in-repository caller already held a `Dynamic` tensor.
- **`TypedTensor::into_host` resolves a device label instead of carrying it.**
  `Host` is now evidence rather than a hint: the narrow keeps `PinnedHost` /
  `UnpinnedHost` (and any `cpu_affinity`) and replaces `Device` / `Managed` /
  `Other` with unpinned host memory. A value labelled device-resident that is
  narrowed to host used to come out still claiming a device, which is why every
  entry point had to re-check the label at run time (`ensure_host_tensor`, 72
  call sites). Those checks are removed in the following change; this one only
  establishes the invariant they were compensating for. Fabricating the
  contradiction on the union still works, and `tests/ui/storage/fail/
  host_placement_is_not_settable.rs` pins the other half: the compiler refuses
  `set_placement` on a `Host`-typed value (`E0599`).
- **The dependency pins move to the `threadteam` 0.5 line** (#2494):
  `threadteam` 0.5.1, `strided-{basic,perm,kernel,nn,fused}` 0.12.0,
  `strided-view` 0.7.1, `t4a-faer` 0.32.0, `tcontract-exec` 0.11.0,
  `tlinalg` 0.10.0 and `cpueinsum`/`cpueinsum-blas` 0.11.0. This is breaking for
  a host even though no tenferro signature moved: an `Exec` carries a
  `threadteam` 0.5 `TeamExec`, so the pool, the token and the `Exec` must come
  from the same generation, and two `threadteam::Exec` types cannot meet in one
  graph. Before this the workspace compiled with 128 `E0308`s. `tlinalg` 0.10
  also writes the SVD and Hermitian spectra as `<T as ScalarEntity>::Real`,
  which is the type tenferro already returned, so the pooled pass that converted
  one to the other is deleted rather than adapted.
- **`TensorRead::tensor_view`, `TensorValue::tensor_view` and
  `TensorValue::tensor_read` return `Option`** (#2486). They answer `None` for a
  tensor built with `Tensor::external`, whose caller-owned element type has no
  preset-scalar `TensorView` (for `tensor_read`, only a non-compact view of such
  a value). Before, they aborted with `unreachable!("views cover the preset
  scalars")`. Callers that already return `Result` now report that case as a
  typed error: `TensorRead::as_slice` and the CPU engine's
  `download_to_host`/`upload_host_tensor` return `Error::UnsupportedDType`
  instead of aborting. Migration: `read.tensor_view()` becomes
  `read.tensor_view().ok_or(..)?`, or a `match` on `Some(TensorView::..)`.

- **The CPU complex SVD and `eigh` no longer allocate outside the session
  pool** (#2462). The real spectrum is gathered into a pooled lease instead of
  a fresh `Vec`; the recorded steady-state ceiling of a complex SVD or `eigh`
  drops from 5 to the 1 allocation now measured. The `blas` route keeps the spectrum real instead of widening it to the
  complex scalar and narrowing it again.
- **Breaking (minor): traced values are owned by a `TraceContext`.** This is a
  minor release, not a patch: `TracedTensor` no longer carries its own graph, so
  context-free constructors, `TracedTensor::graph()` and the public
  `id`/`rank`/`dtype`/`val` fields are gone. See
  [`docs/design/traced-over-trace-context.md`](docs/design/traced-over-trace-context.md)
  §3.9 for the complete list and
  [`docs/guides/custom-operations.md`](docs/guides/custom-operations.md#migrating-a-context-free-constructor)
  for the one-line shape of the migration.
  - Extension authors: `TracedTensor::input_concrete_shape(dtype, shape)` becomes
    `<ctx>.input_concrete_shape(dtype, shape)`, and a traced value can only be
    combined with values from the same context.
  - `GraphCompiler::compile_many` requires at least one output; a zero-output
    program is built with `SemanticProgramBuilder::finish(&[])`.
  - AD seeds of an externally defined scalar must carry that scalar's canonical
    identity, which the context constructors now supply.
- **CUDA reductions cost about 20 µs less per standalone call.** `tenferro-gpu`
  reductions now wait once per call instead of once per pass plus once more for
  the result squeeze. They write the squeezed result directly and no longer
  zero-fill destinations their kernels overwrite completely. For example, a full
  `reduce_max` of 1M `f32` values on an A100 drops from 50.8 to 30.2 µs. Result
  values are bit-identical (#2406).

### Removed

- `tenferro_runtime::dim_expr_to_sym_dim_public` (`#[doc(hidden)]`) is gone
  (#2443). It turned a program's `DimExpr::InputDim { input_idx, axis }` into
  `SymDim::tensor_axis(input_idx as u64, axis)`, a value indistinguishable from
  the axis of the traced handle whose id happens to be `input_idx`, so a
  conversion given one read the other's axis. A program position has no meaning
  as a `SymDim`, and no public path produces one any more. Callers migrate per
  extent: an extent that does not depend on inputs is
  `SymDim::from(expr.eval(&[])?)` (`DimExpr::eval`); a symbolic extent of a
  traced value is `TracedTensor::axis_sym_dim(axis)` (or `sym_shape()` /
  `sym_size(axis)`) on the handle that owns it, for example the program output
  or the leaf returned by the `TraceContext::input_*` constructor of that input.
- The unused `rocm` placeholder feature is gone from `tenferro-gpu` and from the
  five crates that forwarded it (`tenferro-ad`, `tenferro-einsum`,
  `tenferro-fft`, `tenferro-linalg`, `tenferro-internal-ops`): no code was gated
  on it and HIP/ROCm has no implementation. A consumer that enabled or forwarded
  `rocm` drops it from its feature list (#2310).
