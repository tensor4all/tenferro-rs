# Provider-neutral full SVD, borrowed-input parity, and full Q (issue #1822)

Umbrella record for the independently mergeable slices of
[#1822](https://github.com/tensor4all/tenferro-rs/issues/1822). Each slice is a
separate PR; this file is updated as each one lands.

## Decisions

### A — public borrowed and concrete full SVD (#1814)

- `LinalgBackend::svd_full_read(TensorRead<'_>)` mirrors `svd_read`: a
  default-`Unsupported` trait hook, a faer strided-view fast path guarded by
  `faer_strided_read_ok`, and `with_materialized_tensor_read` for everything
  else. `svd_full` and `svd_full_read` now share one already-entered
  `svd_full_entered` helper, so the owned and borrowed routes cannot drift in
  provider dispatch or error typing.
- The public surface mirrors `svd` exactly: `TensorLinalgExt::svd_full`,
  `TypedTensorLinalgExt::svd_full` (real `S`), `TensorReadLinalgExt::svd_full_read`.
  `LinalgOp::SvdFull` is routed through `svd_full_read` in
  `execute_linalg_extension_reads_in_session`, so eager and traced callers reach
  the same borrowed boundary every other decomposition already uses.
- The LAPACK provider keeps returning a typed `Unsupported` from
  `svd_full_entered` rather than silently switching to faer; slice B replaces
  that arm with a real kernel.
- Zero-dimension contract fixed while defining it. The previous owned
  `svd_full` early return built the `m x m` / `n x n` factors from an empty
  data vector, which fails the shape/length check for any input with exactly
  one empty core dimension (for example `[0, 3]`). The factor for the
  *non*-empty dimension is now the identity, which is the canonical unitary
  choice: it keeps `U Uᴴ = I` and `Vᴴ V = I` true and reconstructs the empty
  input exactly. Batched inputs repeat the identity per slab.
- `docs/internals/public-boundary-overhead-inventory.md` is regenerated
  (digest only). Its `svd_full` rows report a family-wide disposition shared by
  every `linalg` operation, and `borrowed: unsupported` there means "no
  published timing evidence" — `svd` carries the same disposition despite
  having had `svd_read` since #1646. Slice A publishes no timing evidence, so
  no disposition changes; splitting `svd_full` into its own selector would make
  the document inconsistent with the rest of the family.

### B — CPU BLAS/LAPACK full SVD (#1813)

- `LapackSvd::svd_full_2d` for f32/f64/c32/c64. The default (non-`provider-inject`)
  path uses `?gesdd` with `jobz = 'A'`, `ldu = m`, `ldvt = n`, and the existing
  `iwork = 8*min(m, n)`; the `provider-inject` path uses `?gesvd` with
  `jobu = jobvt = 'A'`. Complex `rwork` reuses `complex_gesdd_rwork_len`, called
  with `b'A'`: LAPACK states one `LRWORK` bound for `JOBZ = 'S' or 'A'`, so the
  helper's existing vectors branch already covers full mode, and passing `'A'`
  keeps that reasoning visible at the call site rather than implicit.
- `svd_full_entered` now dispatches `CpuLinalgProvider::Blas` through
  `linalg::blas::svd_full`, which shares the `batched_multi` wrapper and the
  identity-based empty-dimension contract slice A defined for faer. One public
  call, one shape contract, on both providers.
- `linalg_session_supported::<CpuBackend>` returns `true` for every linalg op,
  including `SvdFull`: the type-only seam no longer has a provider-specific
  exception to encode.
- Provider equality is reconstruction, unitarity, spectra, and documented gauge,
  not identical basis bytes. The tests assert exactly that; they never compare
  faer and LAPACK factors elementwise.

## Verification conclusions and constraints

### A

- Owned and borrowed full SVD agree for F32/F64/C32/C64 across tall, wide,
  square, and both one-dimensional shapes, checked by `U Uᴴ = I (m x m)`,
  `Vᴴ V = I (n x n)`, both adjoint directions, rank-`k` reconstruction, and a
  spectrum that matches `svdvals`.
- Borrowed layouts covered: compact owned read, transposed (strided), offset
  slice, and reversed (negative stride, which must pack). The borrowed source
  bytes are compared before and after the call.
- Ownership evidence is two-sided. A source contract asserts that every
  faer-eligible read hook consults `faer_strided_read_ok` and returns through
  its `*_faer_view_entered` adapter *before* reaching
  `with_materialized_tensor_read`. A runtime differential witness compares
  retained CPU buffer-pool capacity between the eligible and the packing route
  for the same shape: only the packing route retains the `m*n` input copy.
- Not established by slice A: LAPACK and CUDA full SVD (slices B and C), any
  wall-clock claim, and AD through the full variant, which stays
  `Unsupported` in the manifest.

### B

- The dtype/shape/layout sweep from slice A now runs once per compiled CPU
  provider. A build with both `cpu-faer` and `cpu-blas` exercises the same
  public call and the same assertions twice, which is the provider-neutrality
  claim stated directly.
- The traced full-SVD tests in `full_svd_lstsq.rs` lost their
  `not(feature = "cpu-blas")` restriction: they only assert provider-neutral
  properties, so they now cover whichever provider the default backend selects.
  `svd_full_lapack_provider_is_unsupported` is replaced by
  `svd_full_lapack_provider_returns_square_unitary_factors`, which pins
  `Uᵀ U = I (m x m)` — the identity a thin factor widened after the fact could
  not satisfy.
- The `cpu-blas` branches of the `svd_full` doc examples, which asserted the
  unsupported error, are gone; those examples now run unconditionally on
  `CpuBackend::new()`.
- Verified on the faer lane, the `blas-openblas` lane, and a both-features
  build; the `provider-inject` `?gesvd` path is verified to compile only
  (`-p tenferro-linalg --lib`). The `provider-inject` integration target does
  not compile at `origin/main` for reasons unrelated to this work, and hosted
  CI's `blas-inject` profile builds `-p tenferro-cpu`, not this target.
- Not established by slice B: CUDA full SVD (slice C) and any wall-clock claim.
