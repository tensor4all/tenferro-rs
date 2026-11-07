# Changelog

All notable changes to the `tenferro-*` crates are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
