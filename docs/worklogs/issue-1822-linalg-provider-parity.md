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

### C — CUDA full SVD

- `svd_typed` gained a `full` flag instead of a parallel function, so the thin
  and full variants cannot drift in driver selection, batching, solver-info
  handling, or the wide-matrix adjoint trick. Its op tag became a runtime
  binding (`op`) rather than the module constant, so errors name `svd_full`
  when that is what the caller asked for.
- `gesvdj` takes `econ = 0` for the full variant (`U` is `m x m`, `V` is
  `n x n`); cuSOLVER places no `m`/`n` ordering constraint on it, so no
  transpose is needed. `gesvd` takes `jobu = jobvt = 'A'` and keeps the
  existing adjoint trick: for `m < n` it factors `Aᴴ`, whose full `'A'` outputs
  are `n x n` and `m x m`, and the final adjoint maps them back to `U` and
  `Vt`.
- The degenerate case needed a decision the thin variant never faced. When only
  one core dimension is zero, the full variant still owes a square unitary for
  the other, so uninitialized `alloc_output` is wrong. `empty_svd_outputs`
  materializes the identity on the host and uploads it. Nothing is read back
  from the device: the input carries no elements, so this is a constant upload,
  not a device-to-host-to-device roundtrip of computed values, and it matches
  the CPU contract exactly.
- `linalg_session_supported::<CudaBackend>` drops its `SvdFull => false` arm.
  The CUDA admission table needed its own test module: `extension::tests` is
  compiled with `not(feature = "cuda")`, so a `cfg(cuda)` arm inside it would
  never build.
- Out of scope, unchanged: CUDA `eig` and `full_piv_lu` stay `Unsupported`,
  and the new admission test pins that they still are.

### D — borrowed-input parity for the remaining CPU read hooks

Two halves with different shapes: the structural borrowed-input gaps, then the
remaining faer view fast paths for rank-revealing QR and triangular solve.

#### Structural gaps

- `LinalgBackend::eig_values_read` is the missing counterpart of the hidden
  `eig_values` hook, with the same `#[doc(hidden)]` shape as
  `eigh_values_read`. `TensorReadLinalgExt::eigvals_read` and
  `LinalgOp::EigVals` now route through it instead of packing the view and
  calling the owned hook — the same fix #1703 made for SVD and EIGH.
- `LinalgOp::Solve` is routed through `solve_read`, which already had a direct
  two-view path (`solve_from_views_entered`); it was simply unreachable from
  eager and traced callers.
- The four faer eig kernels were split into `*_core` entry points over a
  `MatRef` plus thin owned wrappers, mirroring `svd_core`. `eig_read` and
  `eig_values_read` take a strided view straight to the eigensolver when
  `faer_strided_read_ok` allows it. The complex cores keep taking
  `MatRef<faer::cNN>`; the view adapters build that pointer through the layout
  equivalence `impl_complex_faer_casts` already asserts.
- `eig_values` on the CPU backend gained an already-entered
  `eig_values_entered` helper so the owned and borrowed routes share provider
  dispatch, matching every other op in that file.
- Empty core dimensions are handled before faer is entered: general
  eigendecomposition always returns complex factors, so the empty outputs are
  tagged complex exactly as the owned entry point tags them.

#### faer view paths for rank-revealing QR and triangular solve

- Both operations are destructive: CPQR copies into its own work matrix, and a
  triangular solve overwrites its right-hand side. They therefore always pay
  one copy. What the borrowed path removes is the *second* one: the pooled
  compact tensor the packing route built first. `rank_revealing_qr_core` and
  `triangular_solve_core` take an already-prepared `MatRef` and, for the solve,
  an owned destructible RHS, so the caller decides once where those elements
  come from.
- Extracting `triangular_solve_core` collapsed the eight-arm routine match that
  was duplicated across the real and complex macros into one generic
  `faer_triangular_solve_in_place`. Transposing `A` swaps which triangle is
  stored, so the four faer routines cover all eight flag combinations once that
  flip is applied, and a right-side solve is the left-side solve of the
  transposed system with the flag flipped once more.
- `triangular_solve_read` takes the direct path only when *both* operands are
  eligible. The predicates differ on purpose: `a` becomes a strided `MatRef`, so
  it needs `faer_strided_read_ok`; `b` is gathered element by element, so
  arbitrary strides are fine and only host placement, the matrix rank every
  provider requires, and a supported dtype matter.
- The borrowed route must not accept shapes the owned route rejects. The first
  draft let a rank-1 right-hand side through the view path while both providers'
  owned paths require a matrix; the view path now uses the same rank check, and
  a test pins that the two routes refuse alike.
- The owned CPQR screens its input for non-finite and all-zero values before
  factoring. The view path applies the same two guards, read through the view's
  own indexing, and reuses a shape/placement-based zero-matrix result because
  there is no owned template tensor to copy metadata from.
### E — full Q from compact Householder QR

- The bound moves from `end > k` to `end > m` in faer, LAPACK, CUDA, and
  `householder_qr_q_columns_meta`. No new provider routine was needed: all
  three backends already build Q by applying the compact reflectors to the
  matching identity columns, and that construction generalizes to the
  complement columns unchanged. The issue sketched a separate `?orgqr` path for
  LAPACK; the existing `?ormqr`-on-identity route reaches the same result and
  keeps one code path per provider.
- The `PositiveDiagonal` gauge comes from R's diagonal, so it applies to the
  first `k` columns only. Each provider now stops at `k` explicitly. On CUDA
  that was also a latent out-of-bounds: the phase kernel indexed
  `phase[q_start + column]` against a vector of length `k`, so a full-Q request
  would have read past it. The kernel now skips a column with no phase.
- AD through a complement column returns a typed `Unsupported` rather than a
  derivative. The thin `dQ` the linearize rule builds has no column there, and
  the complement basis is defined only up to a rotation inside the nullspace.
  The refusal fires when `k` is a known constant; with a symbolic `k` the rule
  cannot prove the range exceeds it, which is recorded below as a residual.

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

### C

- Verified on a local A100 80GB with CUDA 12.6: all four dtypes across tall,
  wide and square through `gesvdj`; batched inputs; a strided device view
  through `svd_full_read`; empty core dimensions; unsupported-dtype refusal;
  and both `gesvd` orientations above the 1024 `gesvdj` threshold (tall
  `1025 x 4` and wide `4 x 1025`). Each case checks both unitarity directions
  on `U` and `Vt`, rank-`k` reconstruction, a non-increasing spectrum, and
  device residency of the outputs. The existing thin-SVD GPU tests still pass
  unchanged.
- Cross-backend equality is checked as the spectrum plus output shapes, not as
  basis bytes: the device and host bases legitimately differ by phase and
  inside the null subspace.
- The GPU source contracts were updated, not relaxed. They now follow
  `svd_typed`'s runtime op tag and point the zero-dimension residency rule at
  `empty_svd_outputs`, where that fast path now lives; the batched solver-info
  rule, the JAX-compatible driver selection rule, and the ban on downloading
  factor data are unchanged in substance.
- Not established by slice C: any wall-clock claim, and AD through the full
  variant, which stays `Unsupported`.

### D

- Borrowed and owned `eig` / `eigvals` agree for all four dtypes, compared as
  multisets: the solver may order a spectrum differently between the view and
  the packed path, and neither order is a contract.
- Layout coverage: compact owned read, transposed (faer-eligible), reversed
  (negative stride, packs), and rank-3 batched (packs, keeps batch shapes).
  The source bytes are compared before and after.
- The pool-capacity differential witness from slice A is repeated for
  `eigvals_read`: only the packing route retains the `n*n` input copy.
- Two source contracts were updated rather than relaxed.
  `linalg_internal_path_contract` pinned `eigvals_read` to "materialize then
  call `eig_values`", which is exactly the behaviour this slice removes; it now
  requires the borrowed hook and forbids the pack, matching the `eigvalsh_read`
  assertion directly above it. The faer fast-path ordering contract gained the
  two new read hooks.
- Rank-revealing QR through a transposed view is checked against its own
  contract — `Qᵀ Q = I` and `Q R = A P` — plus equal rank, equal permutation
  and equal output shapes against the owned call, with the source bytes
  compared before and after. The zero-matrix and non-finite guards are checked
  through the borrowed route as well.
- Triangular solve is checked against the owned call across four
  triangle/transpose/unit combinations with a strided right-hand side, and once
  more for the right-side orientation with a strided coefficient view. The
  rank-1 refusal parity is its own test.
- The faer fast-path source contract now covers `rank_revealing_qr_read`, and
  `triangular_solve_read` gets its own two-operand assertion because its RHS
  predicate is deliberately different from its coefficient predicate.
- Verified on the faer lane, the `blas-openblas` lane and CI-parity clippy.
  Slice D is complete: every borrowed CPU read hook the issue listed now either
  reaches faer directly or packs only what it must. No wall-clock claim is made.
- Not established by D1: rank-revealing QR and triangular solve still pack
  every borrowed view (slice D2), and no wall-clock claim is made.

### E

- `Qᴴ Q = I (m x m)` for tall inputs on both CPU providers, plus `Q[:, :k]`
  equal to the thin factor, `Aᵀ Q[:, k..m] = 0`, a complement-only range equal
  to the matching slice of full Q, square (`k = m`) and wide inputs, empty
  ranges, and both range-validation errors. The same identities are verified on
  the device through the traced surface and on a local A100.
- The gauge contract is verified as a difference: the gauged and ungauged
  complement columns are identical, while the full factor stays orthonormal.
- The AD refusal is verified end to end through `AdContext::grad`, together
  with the thin range still differentiating, and the manifest caveat naming the
  value-only complement range is asserted by its own test.
- Residual: with a symbolic `k` (a traced shape that is not a constant) the AD
  rule cannot prove the range reaches past the thin width, so the refusal is
  raised only for concrete shapes. That matches the pre-existing behaviour of
  the symbolic column selector this rule already used.
