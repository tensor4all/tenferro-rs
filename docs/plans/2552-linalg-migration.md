# tenferro-linalg migration record for #2552 (stage 1)

Branch `feat/2552-call-grammar-linalg`, based on `fc2f0426b`. Scope: `crates/tenferro-linalg`
and the call sites, docs and checkers that use it. The decision implemented is the #2552
comment "Decision: one call grammar for the whole public API…" as amended by "Decision update:
withdraw materialization policy": there is no `MaterializationPolicy` and no policy argument.

## Removed names and their replacements

Generated, not hand-written:

```bash
python3 scripts/api-2552-linalg-inventory.py snapshot docs/plans/2552-linalg-public-api-after.txt
python3 scripts/api-2552-linalg-inventory.py migration --strict \
    docs/plans/2552-linalg-public-api-before.txt docs/plans/2552-linalg-public-api-after.txt
```

Both snapshots are `cargo +nightly-2026-07-18 public-api -p tenferro-linalg --features
autodiff,native,cuda -s`. A name counts as removed when it is absent from the after snapshot
once trait-method rows listed on a receiver type (`Tensor::svd`, the blanket `X::svd`) are folded
into their trait. `replacement present` checks the target against the after snapshot.
Table size: **117 removed names**.

| removed name | replacement | note | replacement present |
|---|---|---|---|
| `error::Result` | `-` | as `Result` | n/a |
| `EighOptions::derivative_eps` | `EighOptions::with_derivative_eps` | rule 3: `with_*` setters | yes |
| `EighOptions::driver` | `EighOptions::with_driver` | rule 3: `with_*` setters | yes |
| `EighOptions::gauge` | `EighOptions::with_gauge` | rule 3: `with_*` setters | yes |
| `QrOptions::gauge` | `QrOptions::with_gauge` | rule 3: `with_*` setters | yes |
| `RankRevealingQrOptions::atol` | `RankRevealingQrOptions::with_atol` | rule 3: `with_*` setters | yes |
| `RankRevealingQrOptions::gauge` | `RankRevealingQrOptions::with_gauge` | rule 3: `with_*` setters | yes |
| `RankRevealingQrOptions::rtol` | `RankRevealingQrOptions::with_rtol` | rule 3: `with_*` setters | yes |
| `SvdOptions::derivative_eps` | `SvdOptions::with_derivative_eps` | rule 3: `with_*` setters | yes |
| `SvdOptions::driver` | `SvdOptions::with_driver` | rule 3: `with_*` setters | yes |
| `SvdOptions::gauge` | `SvdOptions::with_gauge` | rule 3: `with_*` setters | yes |
| `EagerSessionLinalgExt::eigh_with_options` | `EagerSessionLinalgExt::eigh_with` |  | yes |
| `EagerSessionLinalgExt::pinv_with_rtol` | `EagerSessionLinalgExt::pinv_with` | `PinvOptions::default().with_rtol(rtol)` | yes |
| `EagerSessionLinalgExt::qr_with_options` | `EagerSessionLinalgExt::qr_with` |  | yes |
| `EagerSessionLinalgExt::svd_full` | `EagerSessionLinalgExt::svd_with` | `SvdOptions::default().with_shape(SvdShape::Full)` | yes |
| `EagerSessionLinalgExt::svd_with_options` | `EagerSessionLinalgExt::svd_with` |  | yes |
| `EagerTensorLinalgExt` | `EagerSessionLinalgExt` | issue #2552 decision 3 | yes |
| `EagerTensorLinalgExt::solve` | `EagerSessionLinalgExt::solve` | `session.solve(&a, &b)` | yes |
| `TensorLinalgExt::eigh_with_options` | `TensorLinalgExt::eigh_with` |  | yes |
| `TensorLinalgExt::pinv_with_rtol` | `TensorLinalgExt::pinv_with` | `PinvOptions::default().with_rtol(rtol)` | yes |
| `TensorLinalgExt::qr_with_options` | `TensorLinalgExt::qr_with` |  | yes |
| `TensorLinalgExt::svd_full` | `TensorLinalgExt::svd_with` | `SvdOptions::default().with_shape(SvdShape::Full)` | yes |
| `TensorLinalgExt::svd_with_options` | `TensorLinalgExt::svd_with` |  | yes |
| `TensorReadLinalgExt` | `TensorLinalgExt` | `TensorRead<'_>` implements `LinalgInput` | yes |
| `TensorReadLinalgExt::cholesky_read` | `TensorLinalgExt::cholesky` | receiver `TensorRead<'_>` | yes |
| `TensorReadLinalgExt::det_read` | `TensorLinalgExt::det` | receiver `TensorRead<'_>` | yes |
| `TensorReadLinalgExt::eig_read` | `TensorLinalgExt::eig` | receiver `TensorRead<'_>` | yes |
| `TensorReadLinalgExt::eigh_read` | `TensorLinalgExt::eigh` | receiver `TensorRead<'_>` | yes |
| `TensorReadLinalgExt::eigh_with_options_read` | `TensorLinalgExt::eigh_with` | receiver `TensorRead<'_>` | yes |
| `TensorReadLinalgExt::eigvals_read` | `TensorLinalgExt::eigvals` | receiver `TensorRead<'_>` | yes |
| `TensorReadLinalgExt::eigvalsh_read` | `TensorLinalgExt::eigvalsh` | receiver `TensorRead<'_>` | yes |
| `TensorReadLinalgExt::full_piv_lu_read` | `TensorLinalgExt::full_piv_lu` | receiver `TensorRead<'_>` | yes |
| `TensorReadLinalgExt::full_piv_lu_solve_read` | `TensorLinalgExt::full_piv_lu_solve` | receiver `TensorRead<'_>` | yes |
| `TensorReadLinalgExt::inv_read` | `TensorLinalgExt::inv` | receiver `TensorRead<'_>` | yes |
| `TensorReadLinalgExt::lu_read` | `TensorLinalgExt::lu` | receiver `TensorRead<'_>` | yes |
| `TensorReadLinalgExt::norm_read` | `TensorLinalgExt::norm` | `ord`, `usize` axes and `keepdim: bool` become `NormOptions` (`isize` axes, `KeepDims`) on `norm_with` | yes |
| `TensorReadLinalgExt::pinv_read` | `TensorLinalgExt::pinv` | receiver `TensorRead<'_>` | yes |
| `TensorReadLinalgExt::pinv_with_rtol_read` | `TensorLinalgExt::pinv_with` | `PinvOptions::default().with_rtol(rtol)` | yes |
| `TensorReadLinalgExt::qr_read` | `TensorLinalgExt::qr` | receiver `TensorRead<'_>` | yes |
| `TensorReadLinalgExt::qr_with_options_read` | `TensorLinalgExt::qr_with` | receiver `TensorRead<'_>` | yes |
| `TensorReadLinalgExt::rank_revealing_qr_read` | `TensorLinalgExt::rank_revealing_qr` | options move to `rank_revealing_qr_with`; the short form uses defaults | yes |
| `TensorReadLinalgExt::slogdet_read` | `TensorLinalgExt::slogdet` | receiver `TensorRead<'_>` | yes |
| `TensorReadLinalgExt::solve_read` | `TensorLinalgExt::solve` | receiver `TensorRead<'_>` | yes |
| `TensorReadLinalgExt::solve_read_into` | `TensorLinalgExt::solve_into` | destination stays `TensorWrite<'_>` | yes |
| `TensorReadLinalgExt::svd_full_read` | `TensorLinalgExt::svd_with` | `SvdOptions::default().with_shape(SvdShape::Full)` | yes |
| `TensorReadLinalgExt::svd_read` | `TensorLinalgExt::svd` | receiver `TensorRead<'_>` | yes |
| `TensorReadLinalgExt::svd_with_options_read` | `TensorLinalgExt::svd_with` | receiver `TensorRead<'_>` | yes |
| `TensorReadLinalgExt::svdvals_read` | `TensorLinalgExt::svdvals` | receiver `TensorRead<'_>` | yes |
| `TensorReadLinalgExt::triangular_solve_read` | `TensorLinalgExt::triangular_solve` | four `bool` flags become `TriangularSolveOptions` (`Side`, `Uplo`, `Transpose`, `Diagonal`) on `triangular_solve_with` | yes |
| `TraceContextLinalgExt` | `TracedTensorLinalgExt` | issue #2552 decision 4 | yes |
| `TraceContextLinalgExt::cholesky` | `TracedTensorLinalgExt::cholesky` | call on the `TracedTensor` instead of `ctx.op(value)` | yes |
| `TraceContextLinalgExt::det` | `TracedTensorLinalgExt::det` | call on the `TracedTensor` instead of `ctx.op(value)` | yes |
| `TraceContextLinalgExt::eig` | `TracedTensorLinalgExt::eig` | call on the `TracedTensor` instead of `ctx.op(value)` | yes |
| `TraceContextLinalgExt::eigh` | `TracedTensorLinalgExt::eigh` | call on the `TracedTensor` instead of `ctx.op(value)` | yes |
| `TraceContextLinalgExt::eigh_with_options` | `TracedTensorLinalgExt::eigh_with` | call on the `TracedTensor` instead of `ctx.op(value)` | yes |
| `TraceContextLinalgExt::eigvals` | `TracedTensorLinalgExt::eigvals` | call on the `TracedTensor` instead of `ctx.op(value)` | yes |
| `TraceContextLinalgExt::eigvalsh` | `TracedTensorLinalgExt::eigvalsh` | call on the `TracedTensor` instead of `ctx.op(value)` | yes |
| `TraceContextLinalgExt::full_piv_lu` | `TracedTensorLinalgExt::full_piv_lu` | call on the `TracedTensor` instead of `ctx.op(value)` | yes |
| `TraceContextLinalgExt::full_piv_lu_solve` | `TracedTensorLinalgExt::full_piv_lu_solve` | call on the `TracedTensor` instead of `ctx.op(value)` | yes |
| `TraceContextLinalgExt::householder_qr` | `TracedTensorLinalgExt::householder_qr` | call on the `TracedTensor` instead of `ctx.op(value)` | yes |
| `TraceContextLinalgExt::inv` | `TracedTensorLinalgExt::inv` | call on the `TracedTensor` instead of `ctx.op(value)` | yes |
| `TraceContextLinalgExt::lstsq` | `TracedTensorLinalgExt::lstsq` | call on the `TracedTensor` instead of `ctx.op(value)` | yes |
| `TraceContextLinalgExt::lu` | `TracedTensorLinalgExt::lu` | call on the `TracedTensor` instead of `ctx.op(value)` | yes |
| `TraceContextLinalgExt::norm` | `TracedTensorLinalgExt::norm` | `ord`, `usize` axes and `keepdim: bool` become `NormOptions` (`isize` axes, `KeepDims`) on `norm_with` | yes |
| `TraceContextLinalgExt::pinv` | `TracedTensorLinalgExt::pinv` | call on the `TracedTensor` instead of `ctx.op(value)` | yes |
| `TraceContextLinalgExt::pinv_with_rtol` | `TracedTensorLinalgExt::pinv_with` | `PinvOptions::default().with_rtol(rtol)` | yes |
| `TraceContextLinalgExt::qr` | `TracedTensorLinalgExt::qr` | call on the `TracedTensor` instead of `ctx.op(value)` | yes |
| `TraceContextLinalgExt::qr_with_options` | `TracedTensorLinalgExt::qr_with` | call on the `TracedTensor` instead of `ctx.op(value)` | yes |
| `TraceContextLinalgExt::rank_revealing_qr` | `TracedTensorLinalgExt::rank_revealing_qr` | options move to `rank_revealing_qr_with`; the short form uses defaults | yes |
| `TraceContextLinalgExt::slogdet` | `TracedTensorLinalgExt::slogdet` | call on the `TracedTensor` instead of `ctx.op(value)` | yes |
| `TraceContextLinalgExt::solve` | `TracedTensorLinalgExt::solve` | call on the `TracedTensor` instead of `ctx.op(value)` | yes |
| `TraceContextLinalgExt::svd` | `TracedTensorLinalgExt::svd` | call on the `TracedTensor` instead of `ctx.op(value)` | yes |
| `TraceContextLinalgExt::svd_full` | `TracedTensorLinalgExt::svd_with` | `SvdOptions::default().with_shape(SvdShape::Full)` | yes |
| `TraceContextLinalgExt::svd_with_options` | `TracedTensorLinalgExt::svd_with` | call on the `TracedTensor` instead of `ctx.op(value)` | yes |
| `TraceContextLinalgExt::triangular_solve` | `TracedTensorLinalgExt::triangular_solve` | four `bool` flags become `TriangularSolveOptions` (`Side`, `Uplo`, `Transpose`, `Diagonal`) on `triangular_solve_with` | yes |
| `TracedTensorLinalgExt::eigh_with_options` | `TracedTensorLinalgExt::eigh_with` |  | yes |
| `TracedTensorLinalgExt::pinv_with_rtol` | `TracedTensorLinalgExt::pinv_with` | `PinvOptions::default().with_rtol(rtol)` | yes |
| `TracedTensorLinalgExt::qr_with_options` | `TracedTensorLinalgExt::qr_with` |  | yes |
| `TracedTensorLinalgExt::svd_full` | `TracedTensorLinalgExt::svd_with` | `SvdOptions::default().with_shape(SvdShape::Full)` | yes |
| `TracedTensorLinalgExt::svd_with_options` | `TracedTensorLinalgExt::svd_with` |  | yes |
| `TypedTensorLinalgExt` | `TensorLinalgExt` | `TypedTensor<T>` implements `LinalgInput` | yes |
| `TypedTensorLinalgExt::cholesky` | `TensorLinalgExt::cholesky` | receiver `TypedTensor<T>`; typed outputs through `LinalgInput` | yes |
| `TypedTensorLinalgExt::det` | `TensorLinalgExt::det` | receiver `TypedTensor<T>`; typed outputs through `LinalgInput` | yes |
| `TypedTensorLinalgExt::eig` | `TensorLinalgExt::eig` | receiver `TypedTensor<T>`; typed outputs through `LinalgInput` | yes |
| `TypedTensorLinalgExt::eigh` | `TensorLinalgExt::eigh` | receiver `TypedTensor<T>`; typed outputs through `LinalgInput` | yes |
| `TypedTensorLinalgExt::eigh_with_options` | `TensorLinalgExt::eigh_with` | receiver `TypedTensor<T>`; typed outputs through `LinalgInput` | yes |
| `TypedTensorLinalgExt::eigvals` | `TensorLinalgExt::eigvals` | receiver `TypedTensor<T>`; typed outputs through `LinalgInput` | yes |
| `TypedTensorLinalgExt::eigvalsh` | `TensorLinalgExt::eigvalsh` | receiver `TypedTensor<T>`; typed outputs through `LinalgInput` | yes |
| `TypedTensorLinalgExt::full_piv_lu` | `TensorLinalgExt::full_piv_lu` | receiver `TypedTensor<T>`; typed outputs through `LinalgInput` | yes |
| `TypedTensorLinalgExt::full_piv_lu_solve` | `TensorLinalgExt::full_piv_lu_solve` | receiver `TypedTensor<T>`; typed outputs through `LinalgInput` | yes |
| `TypedTensorLinalgExt::inv` | `TensorLinalgExt::inv` | receiver `TypedTensor<T>`; typed outputs through `LinalgInput` | yes |
| `TypedTensorLinalgExt::lu` | `TensorLinalgExt::lu` | receiver `TypedTensor<T>`; typed outputs through `LinalgInput` | yes |
| `TypedTensorLinalgExt::norm` | `TensorLinalgExt::norm` | `ord`, `usize` axes and `keepdim: bool` become `NormOptions` (`isize` axes, `KeepDims`) on `norm_with` | yes |
| `TypedTensorLinalgExt::pinv` | `TensorLinalgExt::pinv` | receiver `TypedTensor<T>`; typed outputs through `LinalgInput` | yes |
| `TypedTensorLinalgExt::pinv_with_rtol` | `TensorLinalgExt::pinv_with` | `PinvOptions::default().with_rtol(rtol)` | yes |
| `TypedTensorLinalgExt::qr` | `TensorLinalgExt::qr` | receiver `TypedTensor<T>`; typed outputs through `LinalgInput` | yes |
| `TypedTensorLinalgExt::qr_with_options` | `TensorLinalgExt::qr_with` | receiver `TypedTensor<T>`; typed outputs through `LinalgInput` | yes |
| `TypedTensorLinalgExt::rank_revealing_qr` | `TensorLinalgExt::rank_revealing_qr` | options move to `rank_revealing_qr_with`; the short form uses defaults | yes |
| `TypedTensorLinalgExt::slogdet` | `TensorLinalgExt::slogdet` | receiver `TypedTensor<T>`; typed outputs through `LinalgInput` | yes |
| `TypedTensorLinalgExt::solve` | `TensorLinalgExt::solve` | receiver `TypedTensor<T>`; typed outputs through `LinalgInput` | yes |
| `TypedTensorLinalgExt::svd` | `TensorLinalgExt::svd` | receiver `TypedTensor<T>`; typed outputs through `LinalgInput` | yes |
| `TypedTensorLinalgExt::svd_full` | `TensorLinalgExt::svd_with` | `SvdOptions::default().with_shape(SvdShape::Full)` | yes |
| `TypedTensorLinalgExt::svd_with_options` | `TensorLinalgExt::svd_with` | receiver `TypedTensor<T>`; typed outputs through `LinalgInput` | yes |
| `TypedTensorLinalgExt::svdvals` | `TensorLinalgExt::svdvals` | receiver `TypedTensor<T>`; typed outputs through `LinalgInput` | yes |
| `TypedTensorLinalgExt::triangular_solve` | `TensorLinalgExt::triangular_solve` | four `bool` flags become `TriangularSolveOptions` (`Side`, `Uplo`, `Transpose`, `Diagonal`) on `triangular_solve_with` | yes |
| `TypedEig` | `Eig` | `Eig<TypedTensor<T::Complex>>` | yes |
| `TypedFullPivLu` | `FullPivLu` | `FullPivLu<TypedTensor<T>, TypedTensor<T::Real>>` | yes |
| `TypedLu` | `Lu` | `Lu<TypedTensor<T>, TypedTensor<T::Real>>` | yes |
| `TypedRankRevealingQrResult` | `RankRevealingQrResult` | `RankRevealingQrResult<TypedTensor<T>, TypedTensor<i64>>` | yes |
| `TypedSvd` | `Svd` | `Svd<TypedTensor<T>, TypedTensor<T::Real>>` | yes |
| `eig_values_with_exec` | `TensorLinalgExt::eigvals` | receiver `TypedTensorView`; engine argument | yes |
| `eigh_values_with_exec` | `TensorLinalgExt::eigvalsh` | receiver `TypedTensorView`; engine argument | yes |
| `eigh_with_exec` | `TensorLinalgExt::eigh` | receiver `TypedTensorView`; engine argument | yes |
| `qr_with_exec` | `TensorLinalgExt::qr` | receiver `TypedTensorView`; engine argument | yes |
| `svd_values_with_exec` | `TensorLinalgExt::svdvals` | receiver `TypedTensorView`; engine argument | yes |
| `svd_with_exec` | `TensorLinalgExt::svd_with` | receiver `TypedTensorView`; engine argument; `full` becomes `SvdShape` | yes |
| `Result` | `-` | no public function returned it; use `tenferro_tensor::Result` (concrete), `tenferro_runtime::Result` (traced) or `tenferro_ad::Result` (eager) | n/a |

Removed names: 117; replacements missing from AFTER: 0

## What changed, by rule

- **Rule 1 (a name says only what is computed).** One name per operation on the concrete
  (`TensorLinalgExt`), traced (`TracedTensorLinalgExt`) and eager (`EagerSessionLinalgExt`)
  layers, with `op`, `op_with` and (for `solve`) `op_into`. The engine is the last argument
  before the destination.
- **Rule 2 (one input kind).** `TensorLinalgExt` is implemented once, on `TensorRead`, through
  the sealed `LinalgInput` conversion implemented by `Tensor`, `TensorRead`, `TypedTensor<T>` and
  `TypedTensorView<'_, T>`; its associated `Out`, `Real`, `Complex` and `Index` types keep typed
  input typed. The owned-tensor concrete path now runs the backend `*_read` methods.
- **Rule 3 (typed arguments).** Every operation has an options type (`#[non_exhaustive]`,
  `Default`, `with_*` setters); empty ones exist for option-less operations. The four
  `triangular_solve` flags became `TriangularSolveOptions` (`Side`, `Uplo`, `Transpose`,
  `Diagonal`); `norm` takes `NormOptions` with `isize` axes and `KeepDims`; `pinv` takes
  `PinvOptions`; `SvdOptions` gained `SvdShape`.
- **Rule 4 (execution per call, no policy).** `Exec` and `BufferPool` no longer appear in any
  linalg operation signature; the `*_with_exec` entries are gone.
- **Rule 5 (named results, one error per layer).** `Svd`, `Qr`, `Lu`, `FullPivLu`, `Eig`,
  `Eigh`, `Slogdet` (and the existing `RankRevealingQrResult`, `HouseholderQr`) replace tuples.
  `tenferro_linalg::Result` was verified unused by any public function (0 rows in the before
  snapshot return it) and removed; the layers return `tenferro_tensor::Error`,
  `tenferro_runtime::Error` and `tenferro_ad::Error`, with `tenferro_linalg::Error` kept as the
  typed cause.
- **Normative text.** `docs/spec/api-conventions.md` naming rules 2-3 and shape rules 3-4, and
  `REPOSITORY_RULES.md` (operation naming, the engine is the *last* argument, the retired eager
  `solve` exception, the `_read` vocabulary) now describe this grammar.

## Behaviour changes, each pinned by a test written red first

| change | test | red output |
|---|---|---|
| options types, enums, `with_*` setters | `tests/integration/call_grammar.rs` | compile failure, below |
| unified entry, `SvdShape::Full`, `solve_into`, negative norm axes | `tests/integration/call_grammar_entry.rs` | compile failure, below |
| `SvdShape::Full` + non-default gauge/driver/eps is `Unsupported` | `call_grammar_entry.rs::full_svd_with_a_gauge_is_an_explicit_error` | (same) |
| named result structs, mixed dtypes | `tests/integration/call_grammar_results.rs` | compile failure, below |
| owned `TypedTensor<T>` through the unified entry | `call_grammar_results.rs::owned_typed_tensors_use_the_same_entry_as_views` | compile failure, below |
| a negative-stride typed view is gathered, not refused | `borrowed_exec_entries.rs::a_negative_stride_view_is_gathered_and_matches_the_compact_input` | replaces the `Unsupported` assertion |

Red, options commit (`cargo test -p tenferro-linalg --features autodiff --test integration call_grammar`):

```text
   1 error: could not compile `tenferro-linalg` (test "integration") due to 6 previous errors
   1 error[E0432]: unresolved imports `tenferro_linalg::CholeskyOptions`, `tenferro_linalg::DetOptions`, `tenferro_linalg::Diagonal`, `tenferro_linalg::EigOptions`, `tenferro_linalg::EigvalsOptions`, `tenferro_linalg::EigvalshOptions`, `tenferro_linalg::FullPivLuOptions`, `tenferro_linalg::FullPivLuSolveOptions`, `tenferro_linalg::HouseholderQrOptions`, `tenferro_linalg::InvOptions`, `tenferro_linalg::KeepDims`, `tenferro_linalg::LstsqOptions`, `tenferro_linalg::LuOptions`, `tenferro_linalg::NormOptions`, `tenferro_linalg::PinvOptions`, `tenferro_linalg::Side`, `tenferro_linalg::SlogdetOptions`, `tenferro_linalg::SolveOptions`, `tenferro_linalg::SvdShape`, `tenferro_linalg::SvdvalsOptions`, `tenferro_linalg::Transpose`, `tenferro_linalg::TriangularSolveOptions`, `tenferro_linalg::Uplo`
   1 error[E0599]: no method named `with_gauge` found for struct `EighOptions` in the current scope
   1 error[E0599]: no method named `with_gauge` found for struct `QrOptions` in the current scope
   1 error[E0599]: no method named `with_gauge` found for struct `RankRevealingQrOptions` in the current scope
   1 error[E0599]: no method named `with_shape` found for struct `SvdOptions` in the current scope
   1 error[E0609]: no field `shape` on type `SvdOptions`
```

Red, entry commit (`... --test integration call_grammar_entry`):

```text
   1 error: could not compile `tenferro-linalg` (test "integration") due to 18 previous errors; 1 warning emitted
   1 error[E0164]: expected tuple struct or tuple variant, found struct variant `tenferro_tensor::Error::Validation`
   1 error[E0599]: no method named `norm_with` found for mutable reference `&mut EagerSession<'_>` in the current scope
   4 error[E0599]: no method named `norm_with` found for struct `tenferro_ad::Tensor` in the current scope
   1 error[E0599]: no method named `pinv_with` found for struct `tenferro_ad::Tensor` in the current scope
   1 error[E0599]: no method named `solve_into` found for struct `tenferro_ad::Tensor` in the current scope
   1 error[E0599]: no method named `svd_with` found for mutable reference `&mut EagerSession<'_>` in the current scope
   2 error[E0599]: no method named `svd_with` found for struct `TracedTensor` in the current scope
   3 error[E0599]: no method named `svd_with` found for struct `tenferro_ad::Tensor` in the current scope
   1 error[E0599]: no method named `svd` found for enum `TensorRead<'a>` in the current scope
   1 error[E0599]: no method named `svd` found for struct `TypedTensorView<'a, T, R, D>` in the current scope
   1 error[E0599]: no method named `triangular_solve_with` found for mutable reference `&mut EagerSession<'_>` in the current scope
   1 error[E0599]: no method named `triangular_solve_with` found for struct `tenferro_ad::Tensor` in the current scope
```

Red, result-struct commit (`... --test integration call_grammar_results`):

```text
   1 error: could not compile `tenferro-linalg` (test "integration") due to 1 previous error
   1 error[E0432]: unresolved imports `tenferro_linalg::Eig`, `tenferro_linalg::Eigh`, `tenferro_linalg::FullPivLu`, `tenferro_linalg::Lu`, `tenferro_linalg::Qr`, `tenferro_linalg::Slogdet`, `tenferro_linalg::Svd`
```

Red, retirement commit (`... --test integration owned_typed`):

```text
error[E0599]: the method `svd` exists for struct `TypedTensor<f64>`, but its trait bounds were not satisfied
    --> crates/tenferro-linalg/tests/integration/call_grammar_results.rs:110:42
     |
 110 |     let owned: Svd<TypedTensor<f64>> = a.svd(&mut engine).unwrap();
     |                                          ^^^ method cannot be called on `TypedTensor<f64>` due to unsatisfied trait bounds
     |
    ::: crates/tenferro-tensor/src/types.rs:1256:1
--
error[E0599]: the method `lu` exists for struct `TypedTensor<f64>`, but its trait bounds were not satisfied
    --> crates/tenferro-linalg/tests/integration/call_grammar_results.rs:113:53
     |
 113 |     let Lu { parity, .. }: Lu<TypedTensor<f64>> = a.lu(&mut engine).unwrap();
     |                                                     ^^
     |
    ::: crates/tenferro-tensor/src/types.rs:1256:1
--
error: could not compile `tenferro-linalg` (test "integration") due to 2 previous errors
```

Assertions that changed meaning with the removed APIs:

- error labels of the concrete entries are the plain operation names (`"svd"`, `"slogdet"`, …),
  not `"*_read"` names;
- the foreign-context case now fails while the `TracedTensor` graph is built
  (`Error::Validation { phase: GraphBuild, .. }`) instead of through the removed
  `TraceContextLinalgExt` (`RuntimeStateSource`);
- source-contract tests now read the `_with` bodies and the moved typed faer bodies.

Tests deleted because their subject no longer exists:

- `cpu_linalg_allocation.rs::borrowed_entries_keep_their_steady_state_allocation_counts` and
  `borrowed_exec_entries.rs::every_buffer_comes_from_the_callers_pool_and_reaches_a_steady_state`
  (caller-held `BufferPool` leasing; this also drops the rank-8/rank-9 metadata-allocation rows
  of #2505 for those entries);
- `borrowed_exec_entries.rs::two_execs_with_two_pools_do_not_share_storage`;
- `trace_context_linalg.rs::context_and_traced_tensor_agree_on_{a_direct_decomposition,a_composite,a_symbolic_input}`
  (parity between `TraceContextLinalgExt` and `TracedTensorLinalgExt`).

## Transcripts

`cargo check --workspace --all-features` was not run after the first attempt; see Q1 below. The
substitute bundle (`/tmp` helper, three commands) ran after every commit. Baseline at
`fc2f0426b`:

```text
$ cargo check --workspace --all-targets
    Finished `dev` profile [unoptimized] target(s) in 13.76s
cargo check --workspace --all-targets  35.84s user 10.59s system 335% cpu 13.848 total
$ cargo check -p tenferro-linalg --all-targets --features autodiff,cuda
    Finished `dev` profile [unoptimized] target(s) in 5.56s
cargo check -p tenferro-linalg --all-targets --features autodiff,cuda  6.06s user 0.87s system 124% cpu 5.585 total
```

After the options commit:

```text
$ cargo check --workspace --all-targets
    Finished `dev` profile [unoptimized] target(s) in 2.32s
exit status: 0
$ cargo check -p tenferro-linalg --all-targets --features autodiff,cuda
    Finished `dev` profile [unoptimized] target(s) in 1.49s
exit status: 0
$ cargo check -p tenferro-linalg --all-targets --no-default-features --features blas,autodiff
   1 warning: `tenferro-linalg` (lib test) generated 5 warnings (5 duplicates)
   1 warning: `tenferro-linalg` (lib) generated 5 warnings
   5 warning: unused `std::result::Result` that must be used
    Finished `dev` profile [unoptimized] target(s) in 2.99s
exit status: 0
```

After the unified-entry commit:

```text
$ cargo check --workspace --all-targets
    Finished `dev` profile [unoptimized] target(s) in 2.03s
exit status: 0
$ cargo check -p tenferro-linalg --all-targets --features autodiff,cuda
    Finished `dev` profile [unoptimized] target(s) in 1.34s
exit status: 0
$ cargo check -p tenferro-linalg --all-targets --no-default-features --features blas,autodiff
   1 warning: `tenferro-linalg` (lib test) generated 5 warnings (5 duplicates)
   1 warning: `tenferro-linalg` (lib) generated 5 warnings
   5 warning: unused `std::result::Result` that must be used
    Finished `dev` profile [unoptimized] target(s) in 1.18s
exit status: 0
```

After the result-struct commit:

```text
$ cargo check --workspace --all-targets
    Finished `dev` profile [unoptimized] target(s) in 0.14s
exit status: 0
$ cargo check -p tenferro-linalg --all-targets --features autodiff,cuda
    Finished `dev` profile [unoptimized] target(s) in 0.92s
exit status: 0
$ cargo check -p tenferro-linalg --all-targets --no-default-features --features blas,autodiff
   1 warning: `tenferro-linalg` (lib test) generated 5 warnings (5 duplicates)
   1 warning: `tenferro-linalg` (lib) generated 5 warnings
   5 warning: unused `std::result::Result` that must be used
    Finished `dev` profile [unoptimized] target(s) in 0.84s
exit status: 0
```

After the retirement commit:

```text
$ cargo check --workspace --all-targets
    Finished `dev` profile [unoptimized] target(s) in 0.44s
exit status: 0
$ cargo check -p tenferro-linalg --all-targets --features autodiff,cuda
    Finished `dev` profile [unoptimized] target(s) in 1.34s
exit status: 0
$ cargo check -p tenferro-linalg --all-targets --no-default-features --features blas,autodiff
   1 warning: `tenferro-linalg` (lib test) generated 5 warnings (5 duplicates)
   1 warning: `tenferro-linalg` (lib) generated 5 warnings
   5 warning: unused `Result` that must be used
    Finished `dev` profile [unoptimized] target(s) in 0.33s
exit status: 0
```

After the docs commit (final tree):

```text
$ cargo check --workspace --all-targets
    Finished `dev` profile [unoptimized] target(s) in 0.26s
exit status: 0
$ cargo check -p tenferro-linalg --all-targets --features autodiff,cuda
    Finished `dev` profile [unoptimized] target(s) in 0.07s
exit status: 0
$ cargo check -p tenferro-linalg --all-targets --no-default-features --features blas,autodiff
   1 warning: `tenferro-linalg` (lib test) generated 5 warnings (5 duplicates)
   1 warning: `tenferro-linalg` (lib) generated 5 warnings
   5 warning: unused `Result` that must be used
    Finished `dev` profile [unoptimized] target(s) in 0.07s
exit status: 0
```

The five `unused Result` warnings of the `blas` build are in
`src/cpu/linalg/lapack_linalg/{helpers,lu,solve}.rs`, files this change does not touch.

`cargo test -p tenferro-linalg --features autodiff` after the unified-entry commit:

```text
     Running unittests src/lib.rs (target/debug/deps/tenferro_linalg-47eed76352ca18c6)
test result: ok. 191 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.32s
     Running tests/cpu_linalg_allocation.rs (target/debug/deps/cpu_linalg_allocation-98c00f3de2664136)
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.49s
     Running tests/integration.rs (target/debug/deps/integration-f98f537af68f7d67)
test result: ok. 351 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.46s
     Running tests/prelude.rs (target/debug/deps/prelude-1975f43eb8a9bb08)
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
     Running tests/small_linalg_allocation.rs (target/debug/deps/small_linalg_allocation-652d49872e6a07a4)
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.16s
   Doc-tests tenferro_linalg
test result: ok. 312 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.13s
cargo test -p tenferro-linalg --features autodiff  20.51s user 6.46s system 184% cpu 14.651 total
```

After the result-struct commit:

```text
     Running unittests src/lib.rs (target/debug/deps/tenferro_linalg-47eed76352ca18c6)
test result: ok. 191 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.31s
     Running tests/cpu_linalg_allocation.rs (target/debug/deps/cpu_linalg_allocation-98c00f3de2664136)
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.46s
     Running tests/integration.rs (target/debug/deps/integration-f98f537af68f7d67)
test result: ok. 355 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.51s
     Running tests/prelude.rs (target/debug/deps/prelude-1975f43eb8a9bb08)
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
     Running tests/small_linalg_allocation.rs (target/debug/deps/small_linalg_allocation-652d49872e6a07a4)
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.16s
   Doc-tests tenferro_linalg
test result: ok. 319 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.14s
cargo test -p tenferro-linalg --features autodiff  19.06s user 6.49s system 186% cpu 13.699 total
```

After the retirement commit and on the final tree (identical counts; the doctest count drops
from 319 to 226 because the removed traits' doctests went with them):

```text
     Running unittests src/lib.rs (target/debug/deps/tenferro_linalg-47eed76352ca18c6)
test result: ok. 191 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.31s
     Running tests/cpu_linalg_allocation.rs (target/debug/deps/cpu_linalg_allocation-98c00f3de2664136)
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.98s
     Running tests/integration.rs (target/debug/deps/integration-f98f537af68f7d67)
test result: ok. 352 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.49s
     Running tests/prelude.rs (target/debug/deps/prelude-1975f43eb8a9bb08)
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
     Running tests/small_linalg_allocation.rs (target/debug/deps/small_linalg_allocation-652d49872e6a07a4)
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.14s
   Doc-tests tenferro_linalg
test result: ok. 226 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.13s
cargo test -p tenferro-linalg --features autodiff  7.59s user 2.05s system 165% cpu 5.815 total
```

Other checks on the final tree: `cargo fmt --all -- --check` and the `ext/tropical`,
`ext/sparse` manifests clean; `scripts/check-doc-snippets.py`, `check-agent-skills.py`,
`check-prelude-contract.py`, `check-public-error-docs.py`, `check-operation-categories.py` and
(with Python 3.12) `check-guide-dependency-snippets.py` pass; `check-api-consistency.py
--fail-on-findings` exits 1 with the same 28 finding headings as at `fc2f0426b` (none in
`tenferro-linalg`); `cargo test -p tenferro-scalar-consumer-application` passes.

## Not done, and why

- **einsum, FFT, AD and runtime session surfaces** are out of scope for this stage; they still
  use `_read`, `*IntoExt`, `_into_accum` and `TraceContext*Ext` names. Only the call sites that
  use linalg were moved. The spec and `REPOSITORY_RULES.md` say so explicitly.
- **The N-ary einsum conjugation bug (#2565)** is untouched.
- **No performance work and no numerical change.** Each new entry runs the backend method or
  extension op its predecessor ran; no benchmark was run, so no performance claim is made. The
  owned-tensor path now enters through the backend `*_read` methods, which is a dispatch change,
  not a kernel change.
- **`LinalgBackend` (provider SPI)** keeps its `_read`, `_with_options`, `svd_full*`, `Vec<Tensor>`
  and `bool` shapes (Q3).
- **Generation and freeze mechanisms** of the decision (an op registry that generates the layer
  traits; `cargo public-api` snapshots under `api/` gated in CI) were not built; this stage was
  hand-implemented, and the before/after snapshots here are review aids, not a CI gate. CI files
  under `.github/` were not to be edited.
- **`svdvals` on traced/eager and `lstsq` on concrete** were not added (Q10).
- **`benches/linalg_vjp_gate.rs`** (features `autodiff,__bench_unification_semantic_ad_api`) does
  not compile at `fc2f0426b` (`reduce_sum`, `backward` signature); only its SVD destructuring was
  migrated, and the same three pre-existing errors remain.
- **`check-pr-fast.sh` and `repository-rules-review.py`** were not run; no pull request is opened.
- **GPU**: CUDA code was type-checked (`--features autodiff,cuda`) but not run.

## Open questions

Mirrored from `OPEN-QUESTIONS-2552.md`.

Questions the lead could not settle from the issue (the "Decision: one call grammar…" comment as
amended by "Decision update: withdraw materialization policy") or from the stage brief. Each entry
says what was done in the meantime, so the branch can be read without guessing; none of these is a
resolution, and each can be reversed in review.

## Q1. `cargo check --workspace --all-features` cannot succeed in this repository

The brief asks for `cargo check --workspace --all-features` after every commit. It cannot pass by
design: `tenferro-cpu` rejects `native` together with `blas`
(`compile_error!("native and blas are mutually exclusive; use --no-default-features for blas")`,
`crates/tenferro-cpu/src/lib.rs:68`), and only one BLAS provider may be enabled at a time. Running
`cargo check -p tenferro-cpu --features blas` in this worktree reproduces that error.

On this host the attempt is also unsafe. The first `--all-features` run on the branch failed inside
the `openblas-build` build script (`Cannot read log file`), and at the same minute the worktree lost
its tracked top-level files (`crates/`, `Cargo.toml`, `CHANGELOG.md`, `deny.toml`, ...) and `target/`.
The lead did not establish the mechanism; nothing of the branch was lost (no commit existed yet), and
the worktree was re-created from the branch with the remnant moved aside, not deleted
(`/Users/hiroshi/work/t2552.wiped-remnant-20261110`).

**Done meanwhile:** after every commit the lead ran, and recorded,

- `cargo check --workspace --all-targets` (default features: `native`),
- `cargo check -p tenferro-linalg --all-targets --features autodiff,cuda`,
- `cargo check -p tenferro-linalg --all-targets --no-default-features --features blas,autodiff`,

and `cargo test -p tenferro-linalg --features autodiff` in place of `--all-features` for the tests.
**Question:** which feature matrix should stand for "all features" in this brief?

## Q2. Retiring the `*_with_exec` entries versus inventory exception E08

The brief retires the `with_exec.rs` free functions; the decision comment retires them too
("`tcontract_exec::Exec` and `BufferPool` leave op signatures"). Inventory v0.2 (exception E08)
conditioned that removal on "a borrowed-engine prototype [that] verifies lifetimes, reuse, nested
execution and retention semantics", and the prototype round did not include that prototype.

**Done meanwhile:** the brief is explicit, so the entries are removed. The typed borrowed view
(`TypedTensorView`) is a `LinalgInput`, so `view.svd_with(&opts, &mut engine)` replaces
`svd_with_exec(exec, &mut buffers, &view, full)` with typed outputs. Two behaviours change with it:
the caller-held `BufferPool` is no longer reusable across calls (the engine's own pools are used), and
a negative-stride view is now gathered by the shared view path instead of refused with `Unsupported`
(the amendment permits "documented same-placement, same-dtype input-layout adaptation within
execution"). The tests whose subject was the caller-held pool (steady-state leasing, the
rank-8/rank-9 metadata rows of #2505, two-exec storage separation) were deleted with the
entries. **Question:** is E08 considered discharged by the brief, or should a borrowed-engine
type that owns an `Exec` and a `BufferPool` be added before the removal lands?

## Q3. Is the provider SPI `LinalgBackend` part of the call grammar?

`LinalgBackend` (public, implemented by `CpuEngine` and `CudaEngine`) still has `_read`,
`_with_options`, `svd_full*`, `solve_read_into` methods, `Vec<Tensor>` returns, and `bool` arguments
(`triangular_solve*` four, `full_piv_lu_solve` one). The decision's layers are concrete, eager,
traced and variadic; the provider SPI is none of them, and rule 6 ("owner-scoped bridge") may be the
category it belongs to. **Done meanwhile:** `LinalgBackend` is unchanged; the concrete entries call
its `_read` methods. **Question:** should the SPI follow rules 1, 3 and 5, become an owner-scoped
bridge (`#[doc(hidden)]`), or stay as it is?

## Q4. Full-matrices SVD combined with a gauge, driver or derivative epsilon

Prototype P1 found that `svd_full*` takes no options and `apply_svd_gauge` is crate-private, and
proposed "Add `SvdShape` to the linalg options contract (full + gauge)" for v0.3 without deciding the
combination. **Done meanwhile:** `SvdOptions::shape` exists; `SvdShape::Full` with any non-default
gauge, driver or `derivative_eps` returns `Error::Unsupported` on every layer, because implementing
it would be new numerical behaviour (out of scope). **Question:** implement the combination, or keep
the explicit error?

## Q5. Defaults of `TriangularSolveOptions`

Rule 3 requires `Default`, but the former `triangular_solve(b, left_side, lower, transpose_a,
unit_diagonal)` had no defaults. **Done meanwhile:** the default is the all-`false` flag set
(`Side::Right`, `Uplo::Upper`, `Transpose::NoTranspose`, `Diagonal::NonUnit`), which is also the
default of `jax.lax.linalg.triangular_solve`. **Question:** confirm, or choose the LAPACK-style
`Side::Left` / `Uplo::Lower`.

## Q6. The destination type of `_into`

Decision 6 names `Dest::overwrite(&mut t)` / `Dest::accumulate(&mut t)` but not where `Dest` lives.
**Done meanwhile:** `solve_into` (the only linalg `_into`) keeps `TensorWrite<'_>` as its
destination, i.e. overwrite only. **Question:** where does `Dest` live, and should linalg adopt it in
the einsum stage?

## Q7. Values-only options (`SvdvalsOptions`, `EigvalshOptions`)

The inventory lists "applicable values-only controls" for `svdvals`. The backend has a hidden driver
selection for values-only SVD and eigh (`svd_values_with_driver*`, `eigh_values_with_driver*`) that no
public entry exposes. **Done meanwhile:** both options types are empty; no driver is exposed.
**Question:** should they carry `driver`?

## Q8. "Five booleans" in `triangular_solve`

The brief mentions five booleans. The public signatures at `fc2f0426b` have four (`left_side`,
`lower`, `transpose_a`, `unit_diagonal`) on every layer; the fifth `bool` in the crate is
`LinalgBackend::full_piv_lu_solve(a, b, transpose)`, which is SPI (Q3). **Done meanwhile:** the four
became `TriangularSolveOptions`.

## Q9. Typed input for `householder_qr`

`HouseholderQr<T>` has methods only for `Tensor`, `TracedTensor` and `EagerTensor`. **Done
meanwhile:** `householder_qr` on any concrete input returns `HouseholderQr<Tensor>`, so typed input
does not give a typed result for this one operation. **Question:** add the typed state, or accept
the exception?

## Q10. Operations missing on some layers

Rule 1 says the same operation has the same name on every layer. `svdvals` has no traced or eager
entry, and `lstsq` has no concrete entry. Adding them is new functionality. **Done meanwhile:**
nothing was added. **Question:** add them in a later stage?
