# Open questions from #2552 stage 1 (tenferro-linalg)

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
