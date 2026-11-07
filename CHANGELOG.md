# Changelog

All notable changes to the `tenferro-*` crates are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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
