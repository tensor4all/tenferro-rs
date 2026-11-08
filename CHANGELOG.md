# Changelog

All notable changes to the `tenferro-*` crates are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0] - 2026-11-09

0.10.0 stopped identifying one body of content: the crates on crates.io were
built from `fd5fc73de6`, and `main` kept the `0.10.0` version string for the 48
commits that followed. This release renumbers that tree and collects what it
changes. The version number moves one minor, because no compatibility promise
exists yet; a Rust caller should still read it as a breaking release.

### Breaking changes

What is gone, and what moved:

- **`TypedTensor::set_placement` is fallible and only on the `Dynamic`
  representation** (#2484, merged as #2497 and #2507). It returns
  `crate::Result<()>` instead of `()`, and it and `TypedTensor::set_cpu_affinity`
  left `impl<T, R, D: Representation> TypedTensor<T, R, D>`, so a `Host`- or
  `Gpu`-typed owner can no longer be relabelled. `is_host_memory()` is the one
  host/device predicate the rest of the crate now asks.
- **`TensorRead::tensor_view`, `TensorValue::tensor_view` and
  `TensorValue::tensor_read` return `Option`** (#2486, merged as #2490). They
  answer `None` instead of aborting: `tensor_view` for any `Tensor::external`
  payload, and `tensor_read` only when the view layout differs from the stored
  tensor's (`as_tensor` compares layouts, not compactness: a reshaped external
  value that still covers the whole buffer in a different layout has no read).
- **`tenferro_runtime::dim_expr_to_sym_dim_public` is removed** (#2443, merged
  as #2473). A program input position has no meaning as a `SymDim`.
- **`SvdDriver::{Gesvdj, Gesvd, Xgesvdp}` became
  `SvdDriver::{Jacobi, QrBased, Polar}`, and `EighDriver::{Syevd, Syevj}` became
  `EighDriver::{DivideAndConquer, Jacobi}`** (#2516, merged as #2517), naming an
  algorithm family rather than a cuSOLVER routine.
- **The `rocm` placeholder feature is gone** from `tenferro-gpu` and the five
  crates that forwarded it (#2310, merged as #2470).
- **`TensorView<'a>` and `TensorViewMut<'a>` gained a defaulted
  representation parameter** (`D: Representation = Dynamic`). A type annotation
  still names the same type, but the `Host` specialization adds its own inherent
  `dtype`, so an associated-function reference such as `TensorView::dtype` is now
  `E0034` ambiguous; name the representation (`TensorView::<Dynamic>::dtype`).
- **Two generations cross the crate boundary** (see [Dependency
  moves](#dependency-moves)): `threadteam` 0.4.0 became 0.5.1 and `t4a-faer`
  0.30.0 became 0.33.0, so the `Exec`, `Par` and `TeamExec` a host builds no
  longer meet the ones this crate expects, and `tcontract-exec`/`cpueinsum`
  moved 0.10 → 0.12, so `CpuBatchPolicy`, `CpuBatchStrategy` and
  `CpuBatchThresholds` (re-exports of `tcontract_exec` types) do too. Two
  copies in one graph are `E0308`.

Nothing was removed from the linalg factorization surface itself (#2483, merged
as #2481, #2506 and #2508): that change is additive. It adds
`tenferro_tensor::capability_entry`, makes `tenferro_linalg::error::map_error`
public, and adds the six borrowed-`Exec` `*_with_exec` entries while the engine
entries keep their signatures and now run the same typed bodies.

### Dependency moves

#2494 moved the workspace onto the `threadteam` 0.5 line; #2521 moved it onto the
`t4a-faer` 0.33 chain, which is one release train because `tcontract-exec`,
`cpueinsum` and `tlinalg` all pin `t4a-faer` themselves.

| dependency | 0.10.0 | 0.11.0 |
|---|---|---|
| `threadteam` | 0.4.0 | 0.5.1 |
| `strided-basic`, `-perm`, `-kernel`, `-fused`, `-nn` | 0.11.0 | 0.12.0 |
| `strided-view` | 0.7.0 | 0.7.1 |
| `t4a-faer` | 0.30.0 | 0.33.0 |
| `tcontract-exec` | 0.10.0 | 0.12.0 |
| `tlinalg` | 0.9.0 | 0.11.0 |
| `cpueinsum`, `cpueinsum-blas` | 0.10.0 | 0.12.0 |
| `t4a-cuda-kernels` | `=0.1.1` (rev `b429d41`) | `=0.1.2` (rev `ab7395c`) |

`strided-traits` (0.7.0), `tlinalg-blas` (0.5.1) and `t4a-cuda-runtime` (0.1.0)
are unchanged. `scripts/check-single-versions.sh` is the check that keeps one
identity per crate in the resolved graph.

### Results that can differ in the final bit

Two of these changes pick a different algorithm, so the same call can return
different last bits than it did on 0.10.0. Neither is pinned by a bit-equality
test.

- **A serial `native` c64 contraction takes faer on every host class again**
  (#2524, merged as #2526). 0.10.0 had narrowed 0.9.0's rule to
  `(X86_64Avx2, 1)`, which left every other `HostClass` on `tcontract`'s packed
  driver at a thread budget of one above `m*n*k = 1 << 17`, with no measurement
  behind the exception. A budget of one now lifts the c64 bound on all six
  classes, and the test reaches the list through an exhaustive `match`. The two
  drivers do not accumulate in the same order; the test asserts which driver is
  selected and the performance note beside it cites a normwise bound, not bit
  equality.
- **`t4a-cuda-kernels` 0.1.2 changes reduction numerics** (#2534). Split pass 1
  now assigns lanes by group instead of per lane, so `sum` and `sum_squares`
  over a contiguous axis accumulate `f32`/`f64` partials in a different order
  and the result can differ in the final bit from 0.1.1. The NaN-propagating
  extrema are rewritten as selects, a near-overflow fold (`x0=x1=A`,
  `x32=x33=-A`, `A = 2^127`) that returned 0 now returns NaN, and the kernel
  ABI revision moves from 12 to 13 (no symbol, signature, struct or
  launch-geometry change).

### Added

- **`TraceContextLinalgExt` and `TraceContextFftExt`** (#2525, merged as #2531).
  The linalg and fft families now have the `TraceContext` entry point the other
  families already had: the context method takes the context's `TraceValue`
  inputs with a `&self` receiver, and the existing `TracedTensorLinalgExt` /
  `TracedTensorFftExt` methods delegate to it in one line, so both spellings
  record the same operations. Both traits are re-exported from their crate
  preludes. This is additive.

- **`tenferro_tensor::capability_entry`** (#2483). It is the one lookup rule
  behind `TensorBackendCapability::capability` and `::require_capability`: the
  backend id, operation, dtype and optional capability axis are matched in one
  place, so a kernel that needs to know whether its table admits a dtype calls
  the same code the trait does instead of repeating the scan next to a second
  error message. `CpuEngine`'s analytic dtype gate now goes through it.
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

- **`SvdDriver` and `EighDriver` name an algorithm family, not a cuSOLVER
  routine** (#2516). `SvdDriver::{Gesvdj, Gesvd, Xgesvdp}` became
  `{Jacobi, QrBased, Polar}` and `EighDriver::{Syevd, Syevj}` became
  `{DivideAndConquer, Jacobi}`; `Auto` and every derivation are unchanged, and
  the variant docs now state the efficacy and keep one line naming the routine
  the CUDA provider maps the family to. The long cuSOLVER notes (the
  `h_err_sigma` guard, the batched entry points, the older-library fallback)
  moved to `gpu/linalg.rs` beside the dispatch that owns them. The docs of
  `TensorReadLinalgExt`'s borrowed entries no longer promise a faer-shaped
  eligibility rule: the public contract says a provider may read the view in
  place and otherwise materializes once inside its own boundary, and the
  predicates (`faer_strided_ok` for the CPU lane, `to_contiguous_view` for the
  CUDA lane) are documented on the provider side. `tenferro_cpu::faer`'s
  re-export doc now states that no tenferro function takes a faer type in its
  signature. This is a source-breaking rename of public enum variants.
- **One host/device answer: `is_host_memory()` on `TypedTensor`, its views,
  `Tensor`, `TensorRead`, `TensorWrite` and `TensorViewMut`** (#2484). The
  payload decides — a plain host owner, a host slice and a CPU-resident
  allocation group are host memory, a backend buffer is not — and
  `Placement::memory_kind` never does. `TensorRead::try_host`,
  `try_mut_host`, the CPU provider's operand admission and linalg's borrowed
  entries all run this one predicate, so a tensor can no longer be host memory
  to linalg and device memory to the CPU provider. `TypedTensor::set_placement`
  now returns `Result` and refuses `MemoryKind::Device` or `Managed` on a plain
  host payload, which is the contradiction that made those two paths disagree;
  a host payload keeps a host label instead of being rewritten on the way into
  `Host`. The opposite direction stays legal: a backend handle is opaque to the
  tensor crate and the runtime labels one with the caller's storage class, so
  relabelling a backend payload is still accepted. Callers of `set_placement`
  handle the `Result`; the pinned-host and CPU-affinity relabels it exists for
  are unchanged.
- **One mutable-layout legality rule: `TensorLayout::validate_mutable_no_overlap`
  now runs `strided_view::is_injective_layout`** (#2484), the analysis that
  already guards strided-rs' own mutable kernels, and `tenferro-tensor-core`
  depends on `strided-view` for it. The local stride-span proof and its
  4096-element exact-enumeration ceiling are gone, so an injective interleaved
  layout such as `[3, 1500]` with strides `[2000, 3]` is accepted for writing
  instead of being reported as `OverlappingMutableLayout`. Genuinely overlapping
  layouts still get that error, and an element count that does not fit `usize`
  still gets `IntegerOverflow`.

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
  preset-scalar `TensorView` (for `tensor_read`, only when the view layout
  differs from the stored tensor's). Before, they aborted with
  `unreachable!("views cover the preset scalars")`. Callers that already return
  `Result` now report that case as a typed error: `TensorRead::as_slice` and the
  CPU engine's `download_to_host`/`upload_host_tensor` return
  `Error::UnsupportedDType` instead of aborting. Migration: `read.tensor_view()` becomes
  `read.tensor_view().ok_or(..)?`, or a `match` on `Some(TensorView::..)`.

- **The CPU complex SVD and `eigh` no longer allocate outside the session
  pool** (#2462). The real spectrum is gathered into a pooled lease instead of
  a fresh `Vec`; the recorded steady-state ceiling of a complex SVD or `eigh`
  drops from 5 to the 1 allocation now measured. The `blas` route keeps the spectrum real instead of widening it to the
  complex scalar and narrowing it again.
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

### Fixed

- **A serial `native` c64 contraction takes faer on every host class again**
  (#2524). `default_faer_limit` had narrowed 0.9.0's rule to `x86_64-avx2` at a
  thread budget of one, which put `x86_64-avx512` and `aarch64` back on
  tcontract's packed driver for c64 above `m*n*k = 1 << 17` without a
  measurement behind the exception (1.28x to 1.45x slower per call in the
  reported downstream workload). A budget of one now lifts the c64 bound on
  every `HostClass`, and the test enumerates every variant so a narrowing
  without a measurement fails there. Wider budgets keep the default.
- **A borrowed linalg entry refuses an empty operand with a negative stride**
  (#2505). The six `*_with_exec` factorizations validated the layout inside
  `raw_view`, which the zero-dimension shortcut returned before, so a `[3, 0]`
  or `[0, 0]` view whose remaining axis was reversed came back with empty
  outputs while the same layout on a non-empty operand was `Unsupported`. The
  stride rule now runs before the shortcut, so the empty operand fails with the
  non-empty error, variant and message included. Every other input, the
  shortcut's own results included, is unchanged.
- **The borrowed entries' storage guarantee is scoped to the payload and the
  workspace** (#2505). The rustdoc said tenferro "allocates nothing outside"
  the caller's `BufferPool`, but a view's and an output's dims and strides are
  shape metadata: inline up to `INLINE_RANK` (8) axes, heap-allocated above
  that, so a rank-9 operand allocates a few small metadata blocks per call. The
  contract now says what it covers, and `tests/cpu_linalg_allocation.rs` carries
  a rank-8 row (no allocations) beside a rank-9 row (some), so the wording and
  the measured counts can only move together.

## [0.10.0] - 2026-11-06

This section carries the one entry that was already true of the published
0.10.0 tree (built from `fd5fc73de6`); every entry above it describes a change
made after that commit.

### Changed

- **Traced values are owned by a `TraceContext`.** `TracedTensor` no longer
  carries its own graph, so context-free constructors, `TracedTensor::graph()`
  and the public `id`/`rank`/`dtype`/`val` fields are gone. See
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
