# Issue #1332 TBLIS CPU Provider

## Summary

Added an initial optional `cpu-tblis` provider path for dense CPU
`dot_general` contractions. The patch is intentionally narrow: feature wiring,
the now-superseded TBLIS backend-kind prototype, a private TBLIS FFI leaf, and dispatch from
`TensorDot`/cached session dot-general paths.

## Context Read

- `AGENTS.md`
- `REPOSITORY_RULES.md`
- `crates/tenferro-cpu/src/backend.rs`
- `crates/tenferro-cpu/src/exec_session.rs`
- `crates/tenferro-cpu/src/gemm/mod.rs`
- `crates/tenferro-cpu/src/gemm/{faer_gemm.rs,blas_gemm.rs}`
- `crates/tenferro-cpu/tests/provider_feature_contract.rs`
- `docs/architecture/tenferro-crates.md`
- shared `tensor4all-agent-rules` common/Rust rules from the upstream repository
- local cargo registry sources for `tblis-ffi 0.2.6`

## Decisions

- Kept `tblis-ffi = 0.2.6` optional behind the TBLIS provider routes; default
  features are unchanged.
- Added Tensor4All-maintained `t4a-tblis-src` source-build link glue behind
  `cpu-tblis-linked`. The independently versioned package is stored under
  `third_party/`, excluded from the tenferro workspace, and consumed through
  the neutral Cargo dependency alias `tblis-src`.
- Kept `cpu-tblis` additive to an existing CPU provider: `tenferro-cpu` still
  requires `cpu-faer` or `cpu-blas`, and `CpuBackendKind::default_compiled()`
  remains unchanged. This preserves current defaults and guarantees unsupported
  TBLIS cases have a compiled fallback path.
- Added the now-superseded TBLIS backend-kind prototype, but kept low-level TBLIS helpers `pub(crate)`.
- Supports `f32`, `f64`, `c32`, and `c64` TBLIS contractions. Conjugated
  `dot_general_with_conj` and accumulated writes use TBLIS tensor conjugation
  flags when a TBLIS plan is valid.
- Passed full tensor labels directly to `tblis_tensor_mult`; no
  `tblis_einsum`, no eager-einsum path, no reductions, no copy/transpose
  replacement, and no grouped GEMM replacement.
- Required host-backed inputs, positive strides, non-negative offsets,
  nonzero rank, nonempty dimensions, representable FFI ranks/dimensions, and
  a small ASCII label set before calling TBLIS.
- Used zero-initialized pooled owned output for the first TBLIS path even with
  `beta = 0`, avoiding assumptions about whether TBLIS reads `C`.

## Deferred

- Negative strides, zero-rank scalar contractions, and zero-size tensors.
- Upstreaming the source-build fixes to `tblis-src` remains desirable, but the
  tenferro PR no longer depends on a new upstream release.
- A richer TBLIS thread-count policy. The initial path avoids tenferro-owned
  outer Rayon installation for the now-superseded TBLIS backend-kind prototype and leaves native TBLIS
  scheduling provider-owned.
- Broader TBLIS benchmark coverage beyond the quick local release-mode run with
  pinned provider thread counts.

## Runtime-loader compatibility exception

- `cpu-tblis-runtime` uses `tblis-ffi 0.2.6` with its `dynamic_loading`
  feature. That release reports ordinary shared-library discovery failure only
  through `panic!`, so tenferro temporarily catches the first `dyload_lib()`
  probe and caches whether runtime TBLIS is available.
- The catch is confined to the Rust loader probe before any native TBLIS call.
  The source carries the repository-standard `// INVARIANT:` marker so audits
  treat it as this explicit upstream-compatibility exception rather than a
  general panic-based backend error boundary.
- Tenferro does not replace or suppress the process-global panic hook. Until a
  fallible upstream release is adopted, a missing runtime TBLIS library may
  therefore emit the upstream panic message before tenferro falls back to the
  configured base CPU provider.
- `catch_unwind` works only with `panic = "unwind"`; a downstream application
  using `panic = "abort"` cannot recover from the upstream loader panic. The
  runtime-loaded route remains opt-in, while `cpu-tblis-linked` does not use
  this compatibility path.
- <https://github.com/RESTGroup/tblis-rs/pull/4> adds a fallible
  `try_dyload_lib()` while preserving `dyload_lib()` compatibility. Remove the
  catch and this audit exception after tenferro adopts a released version that
  contains that API.

## Independent source-provider package

- `t4a-tblis-src 0.1.0` has a self-contained manifest and an empty
  `[workspace]` table. The tenferro root also lists it under `workspace.exclude`
  and consumes it through a path-plus-exact-version dependency, so local
  development and the future registry package use the same source without
  coupling its release version to tenferro.
- The package metadata includes its MSRV, docs.rs URL, keywords, categories,
  README, and an explicit package file list. `cargo package --list`,
  `cargo package`, and standalone unit/doc tests validate the package without
  publishing it.
- The Rust build glue retains the upstream RESTGroup/tblis-rs Apache-2.0
  license and attribution. License files and pinned revisions for TBLIS,
  TCI, MArray, stl_ext, and BLIS ship with the package for downstream binary
  redistribution notices.
- No crates.io publication is part of this PR update. Publishing this new
  package requires a separate, explicit user approval for `t4a-tblis-src`.

## Final PR update verification

- Feature-specific Clippy exposed argument-count warnings in the TBLIS FFI
  leaf and benchmark harness. `TblisExecution` now groups one native execution
  request, and `BenchProvider` groups one benchmark provider, without adding
  lint suppressions or changing the public API.
- `cargo test --workspace --release` passed.
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
  passed, and `python3 scripts/check-coverage.py coverage.json` reported all
  150 checked files above their thresholds.
- `cargo doc --workspace --no-deps` passed. The system `python3` is 3.9, so
  `python3.11 scripts/check-docs-site.py` was used;
  it verified 13 workspace library crates and four guide dependency snippets.
- `cargo clippy --workspace --all-targets -- -D warnings` and the equivalent
  `ext/tropical` command passed. Feature-specific Clippy also passed for both
  `cpu-tblis-runtime` and `cpu-tblis-linked`.
- Standalone metadata inspection confirms the package name, version, MSRV,
  description, license, repository/homepage/docs URLs, README, keywords,
  categories, targets, and features. `cargo package --list --allow-dirty`
  listed 17 intended files, and `cargo package --allow-dirty` packaged and
  verified `t4a-tblis-src 0.1.0` without publishing it.

## Latest-main integration

- Merged `origin/main` at `c889102e` without rewriting the PR branch. The main
  branch had replaced the old single-context CPU backend with shared NUMA-aware
  engines and placement-specific handles.
- `DotGeneralProvider` remains handle-local policy. New placement handles copy
  it, while `run_backend_session_cached` passes the selected policy into each
  engine-owned `CpuExecSession`; the shared topology, engine resources, and
  arbitration state remain provider-policy agnostic.
- The new NUMA implementation had Linux-only test imports and discovery helpers
  compiled but unused on macOS under `-D warnings`. Narrow target/test `cfg`s
  now omit those helpers only from non-test, non-Linux builds; no lint
  suppression was added, and the portable topology tests remain enabled.
- Managed-affinity integration tests now run only on Linux/Android, matching the
  public placement contract. Platform-independent placement resolver tests pass
  `managed_affinity_available = true` explicitly, while the portable capability
  test checks that explicit placement is rejected on unsupported hosts.

## Provider-source integration review

- Current feature shape separates deployment routes:
  `tenferro-cpu/cpu-tblis` aliases the runtime-loaded route, while
  `cpu-tblis-linked` enables `dep:tblis-ffi`, `dep:tblis-src`, and
  `tblis-src/build_from_source` plus `tblis-src/static`. The provider feature
  contract covers both routes.
- `tblis-ffi` has no source-build feature of its own. Tenferro must keep both
  dependencies directly: the `tblis-src` dependency alias resolves to
  `t4a-tblis-src` and supplies Cargo link directives, while `tblis-ffi`
  supplies extern declarations.
- Do not enable `tblis-ffi/dynamic_loading` for the source-backed feature. With
  `t4a-tblis-src`, the FFI should use the non-dynamic extern path, and
  runtime availability should be treated as compile/link availability.
- The source-backed TBLIS path treats runtime availability as compile/link
  availability. The FFI leaf keeps `ensure_runtime_available()` as a no-op and
  does not call `tblis_ffi::tblis::dyload_lib()`, which only exists with
  `tblis-ffi/dynamic_loading`.
- Blocking check:
  `cargo check -p tenferro-cpu --no-default-features --features cpu-faer,cpu-tblis`
  fails inside `tblis-src v0.2.6` before tenferro compiles. The `tblis-src`
  build script passes the packaged
  `.../tblis-src-0.2.6/external_deps/tblis` path as `TBLIS_SRC`, but
  `external_deps/CMakeLists.txt` uses it as `GIT_REPOSITORY`. The unpacked
  crates.io source directory is not a Git repository, so CMake fails with
  `fatal: repository '.../external_deps/tblis' does not exist`.
- `tblis-src/static` does not help this blocker. It only changes the final
  Cargo link directive to `cargo:rustc-link-lib=static=tblis`; CMake still
  attempts to clone the local unpacked source path as a Git repository.
- Fresh-source workaround that does build on this machine:
  `CARGO_TARGET_DIR=/private/tmp/tenferro-tblis-envcheck PKG_CONFIG_LIBDIR=/private/tmp/empty-pkgconfig PKG_CONFIG_PATH=/private/tmp/empty-pkgconfig CMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE TBLIS_SRC=https://github.com/MatthewsResearchGroup/tblis.git cargo check -p tenferro-cpu --no-default-features --features cpu-faer,cpu-tblis`.
  This completed successfully.
- The successful workaround really used bundled BLIS. The generated
  `/private/tmp/tenferro-tblis-envcheck/.../tblis-build/CMakeCache.txt`
  recorded empty `BLIS_FOUND`, empty `BLIS_PREFIX`, and a populated
  `_deps/blis-src` directory; the install output included `lib/libtblis.a` and
  `lib/libtblis.dylib`.
- The same command without a fresh build directory can keep stale CMake cache
  state. A previous run still used `/opt/homebrew/bin/pkg-config` and Homebrew
  BLIS 2.0 despite `PKG_CONFIG=/usr/bin/false`, then failed in the TBLIS plugin
  build with `typedef redefinition with different types ('struct auxinfo_t' vs
  'struct auxinfo_s')`.
- Passing check:
  `cargo test -p tenferro-cpu --test provider_feature_contract` succeeds.

## Source-backed link blocker fix

- Reproduced the current test link failure with the previously successful
  source-build target:
  `CARGO_TARGET_DIR=/private/tmp/tenferro-tblis-envcheck PKG_CONFIG_LIBDIR=/private/tmp/empty-pkgconfig PKG_CONFIG_PATH=/private/tmp/empty-pkgconfig CMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE TBLIS_SRC=https://github.com/MatthewsResearchGroup/tblis.git cargo test -p tenferro-cpu --no-default-features --features cpu-faer,cpu-tblis`.
  The link line had `-L .../tblis-src-.../out/lib` but no `-ltblis`, then
  failed with undefined `_tblis_tensor_mult`.
- Fixed the link retention in the initial `tblis-src` experiment by adding an
  explicit provider-source crate anchor in `tenferro-cpu`, matching the
  existing provider-source pattern for BLAS/LAPACK crates. The final patch uses
  `#[cfg(feature = "cpu-tblis-linked")] extern crate tblis_src as _;`, so
  Cargo keeps the independently versioned source/link crate in the Rust crate
  graph and propagates its native `cargo:rustc-link-lib=static=tblis`
  directive.
- Verified the cached source-build target after the fix with:
  `CARGO_TARGET_DIR=/private/tmp/tenferro-tblis-envcheck PKG_CONFIG_LIBDIR=/private/tmp/empty-pkgconfig PKG_CONFIG_PATH=/private/tmp/empty-pkgconfig CMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE TBLIS_SRC=https://github.com/MatthewsResearchGroup/tblis.git cargo test -p tenferro-cpu --no-default-features --features cpu-faer,cpu-tblis`.
  This passed: 218 unit tests, 7 backend capability contract tests, 3 provider
  feature contract tests, 25 runtime error tests, and 61 doc-tests.
- Verified a fresh source-build target after the fix with:
  `CARGO_TARGET_DIR=/private/tmp/tenferro-tblis-fresh-after PKG_CONFIG_LIBDIR=/private/tmp/empty-pkgconfig PKG_CONFIG_PATH=/private/tmp/empty-pkgconfig CMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE TBLIS_SRC=https://github.com/MatthewsResearchGroup/tblis.git cargo test -p tenferro-cpu --no-default-features --features cpu-faer,cpu-tblis`.
  This also passed: 218 unit tests, 7 backend capability contract tests, 3
  provider feature contract tests, 25 runtime error tests, and 61 doc-tests.
- Verified the focused provider contract independently:
  `cargo test -p tenferro-cpu --test provider_feature_contract` passed.
- Ran `cargo fmt --all`; the follow-up `cargo fmt --all --check` passed.

## Follow-up fallback tests and docs

- Added two `cpu-tblis` fallback behavior tests in
  `crates/tenferro-cpu/src/backend/tests.rs`:
  - scalar-output vector inner product, which TBLIS declines because the output
    rank is zero,
  - zero-size matmul, which TBLIS declines because one logical dimension is
    zero.
  Both use the now-superseded TBLIS backend-kind prototype and assert successful results through the
  compiled faer/BLAS fallback provider.
- Updated public-facing provider docs in:
  `docs/guides/parallelism-and-caching.md`,
  `docs/guides/troubleshooting.md`, `docs/getting-started/index.md`,
  `docs/design/tensor-prims.md`, `docs/design/contraction-pipeline.md`, and
  `docs/design/supported-ops.md`.
- The docs now describe `cpu-tblis` as an optional additive `dot_general`
  contraction provider, keep `cpu-faer`/`cpu-blas` as the required
  fallback/linalg providers, and note that TBLIS threading is provider-owned.
  The current `tblis-src 0.2.6` source-build workaround remains documented only
  in this worklog.
- Follow-up verification:
  `CARGO_TARGET_DIR=/private/tmp/tenferro-tblis-fresh-after PKG_CONFIG_LIBDIR=/private/tmp/empty-pkgconfig PKG_CONFIG_PATH=/private/tmp/empty-pkgconfig CMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE TBLIS_SRC=https://github.com/MatthewsResearchGroup/tblis.git cargo test -p tenferro-cpu --no-default-features --features cpu-faer,cpu-tblis`
  passed with 220 unit tests, including the two new fallback tests, plus 7
  backend capability contract tests, 3 provider feature contract tests, 25
  runtime error tests, and 61 doc-tests.
- `cargo fmt --all --check` passed after the follow-up edits.

## Quick local benchmark

- Supervisor-run quick local release benchmark, not a broad performance claim.
  Threads were pinned with `TBLIS_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, and
  `OMP_NUM_THREADS=1`.
- Command:
  `CARGO_TARGET_DIR=/private/tmp/tenferro-tblis-owned-fresh TBLIS_NUM_THREADS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo bench -p tenferro-cpu --no-default-features --features cpu-faer,cpu-tblis --bench tblis_dot_general_provider -- --quick`
- Results:
  - f64 32: baseline 3.08 us, TBLIS 5.70 us
  - c64 conj 32: baseline 6.38 us, TBLIS 10.54 us
  - f64 64: baseline 12.57 us, TBLIS 19.19 us
  - c64 conj 64: baseline 48.94 us, TBLIS 52.35 us
  - f64 128: baseline 83.49 us, TBLIS 99.34 us
  - c64 conj 128: baseline 386.16 us, TBLIS 330.41 us

## Source-build UX fix

- Checked upstream `RESTGroup/tblis-rs` main before changing tenferro. The
  `tblis-src` `build.rs` and `external_deps/CMakeLists.txt` are still identical
  to crates.io `0.2.6` for this blocker, so switching to upstream main would not
  fix the PR risk.
- Rejected a local `[patch.crates-io]` as the main fix: it would make this
  workspace pass, but downstream users would still resolve crates.io
  `tblis-src` unless an upstream release happened first.
- Added `third_party/t4a-tblis-src`, a small Tensor4All-maintained link crate
  that is independently versioned and explicitly excluded from the tenferro
  workspace. It does not vendor a TBLIS source tree. By default it clones
  `https://github.com/MatthewsResearchGroup/tblis.git` at
  `eb719e718976572e0ab53975f4e0c799faeb35f2`, still honors
  `TBLIS_SRC`/`TBLIS_VER`, uses `SOURCE_DIR` for local `TBLIS_SRC` directories,
  and passes Cargo native link directives for `libtblis`.
- For cloned TBLIS sources, the build glue patches only the downloaded TBLIS
  `CMakeLists.txt` BLIS discovery block so bundled BLIS is used. This avoids the
  Homebrew/pkg-config header mismatch without requiring users to empty
  `PKG_CONFIG_PATH` or CMake package registries.
- Updated `cpu-tblis-linked` feature wiring to:
  `["dep:tblis-ffi", "dep:tblis-src", "tblis-src/build_from_source", "tblis-src/static"]`,
  where the neutral dependency alias resolves to `t4a-tblis-src`, so the
  source-backed provider links `libtblis.a` by default.
- Static source-provider linking also emits link directives for bundled
  `libtci.a`, `libblis_tblis.a`, and `libblis_core.a`; `otool -L` on the
  resulting test binary shows no `libtblis.dylib` dependency.
- The bundled-BLIS CMake patch is idempotent so repeated Cargo rebuilds in the
  same target directory do not fail after the first patch application.
- Bare verification now passes without `TBLIS_SRC`, `PKG_CONFIG_*`, or
  `CMAKE_FIND_*` environment workarounds:
  `cargo check -p tenferro-cpu --no-default-features --features cpu-faer,cpu-tblis`
- Full bare test verification also passes:
  `cargo test -p tenferro-cpu --no-default-features --features cpu-faer,cpu-tblis`
  passed with 220 unit tests, 7 backend capability contract tests, 3 provider
  feature contract tests, 25 runtime error tests, and 61 doc-tests.
- Remaining release hygiene: add a CI/build job for the bare `cpu-faer,cpu-tblis`
  feature set on a platform with the native build prerequisites. The provider
  feature contract validates manifest wiring, but CI should also prove native
  TBLIS can be built and linked.

## TBLIS benchmark expansion

- Expanded `crates/tenferro-cpu/benches/tblis_dot_general_provider.rs` beyond
  square GEMM while keeping the existing f64 and c64 conjugated matmul cases.
- Benchmark names now use explicit provider prefixes: `faer_*`, optional
  `blas_*` under `cpu-blas`, and `tblis_*`.
- The matmul and higher-rank sizes can be overridden with
  `TENFERRO_TBLIS_BENCH_MATMUL_SIZES` and
  `TENFERRO_TBLIS_BENCH_HIGHER_RANK_NS` so large cases can be sampled without
  making the default bench heavy.
- Added optional BLAS baselines gated with `#[cfg(feature = "cpu-blas")]`, so
  the bench compiles both for `cpu-faer,cpu-tblis` and for
  `cpu-faer,cpu-tblis,blas-openblas`.
- Verification:
  `cargo fmt --all --check` passed.
- Verification:
  `CARGO_TARGET_DIR=/private/tmp/tenferro-tblis-fresh-after PKG_CONFIG_LIBDIR=/private/tmp/empty-pkgconfig PKG_CONFIG_PATH=/private/tmp/empty-pkgconfig CMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE TBLIS_SRC=https://github.com/MatthewsResearchGroup/tblis.git cargo check -p tenferro-cpu --no-default-features --features cpu-faer,cpu-tblis --bench tblis_dot_general_provider`
  passed after allowing the TBLIS source clone.
- Verification:
  `CARGO_TARGET_DIR=/private/tmp/tenferro-tblis-fresh-after PKG_CONFIG_LIBDIR=/private/tmp/empty-pkgconfig PKG_CONFIG_PATH=/private/tmp/empty-pkgconfig CMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE TBLIS_SRC=https://github.com/MatthewsResearchGroup/tblis.git OPENBLAS_FC=/opt/homebrew/bin/gfortran LIBRARY_PATH=/opt/homebrew/lib/gcc/current:/opt/homebrew/lib/gcc/15 DYLD_LIBRARY_PATH=/opt/homebrew/lib/gcc/current:/opt/homebrew/lib/gcc/15 cargo check -p tenferro-cpu --no-default-features --features cpu-faer,cpu-tblis,blas-openblas --bench tblis_dot_general_provider`
  passed after allowing OpenBLAS/cblas and TBLIS source downloads.
- Supervisor-run quick local release benchmark with BLAS baseline enabled.
  Threads were pinned with `TBLIS_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`,
  `OMP_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`.
- Command:
  `CARGO_TARGET_DIR=/private/tmp/tenferro-tblis-blas-bench OPENBLAS_FC=/opt/homebrew/bin/gfortran LIBRARY_PATH=/opt/homebrew/lib/gcc/current:/opt/homebrew/lib/gcc/15 DYLD_LIBRARY_PATH=/opt/homebrew/lib/gcc/current:/opt/homebrew/lib/gcc/15 TBLIS_NUM_THREADS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 cargo bench -p tenferro-cpu --no-default-features --features cpu-faer,cpu-tblis,blas-openblas --bench tblis_dot_general_provider -- --quick`
- Benchmark case definitions:

`N` is the square matrix size. `n` is the size of each tensor axis in the
higher-rank cases. The rank-5 case uses a fixed batch size of `4`.

| Case | Question answered | Formula | Inputs | Contract dims | Output |
| --- | --- | --- | --- | --- | --- |
| `f64_matrix_square_gemm` | Does TBLIS add overhead on the plain matrix-multiply case that BLAS/faer already handle well? | `C[i,j] = sum_k A[i,k] B[k,j]` | `A,B: f64[N,N]` | lhs `[1]`, rhs `[0]` | `C: f64[N,N]` |
| `c64_matrix_square_gemm_lhs_conj` | How does the provider handle complex GEMM when the lhs must be conjugated? | `C[i,j] = sum_k conj(A[i,k]) B[k,j]` | `A,B: c64[N,N]` | lhs `[1]`, rhs `[0]` | `C: c64[N,N]` |
| `f64_rank4_packed_contract_axes` | What happens when a 4-D contraction is essentially GEMM-shaped because the summed axes are already packed together? | `C[a,b,c,d] = sum_{x,y} A[a,b,x,y] B[x,y,c,d]` | `A,B: f64[n,n,n,n]` | lhs `[2,3]`, rhs `[0,1]` | `C: f64[n,n,n,n]` |
| `f64_rank4_mixed_contract_axes` | What happens when the summed axes are separated by free axes, so a GEMM-only path typically needs axis movement/materialization? | `C[a,b,c,d] = sum_{x,y} A[a,x,b,y] B[c,y,x,d]` | `A,B: f64[n,n,n,n]` | lhs `[1,3]`, rhs `[2,1]` | `C: f64[n,n,n,n]` |
| `f64_rank5_batched_mixed_contract_axes` | Does the same mixed-axis pattern still behave well when repeated over an explicit batch axis? | `C[b,c,d,e,a] = sum_{x,y} A[a,b,x,c,y] B[a,d,y,x,e]` | `A,B: f64[4,n,n,n,n]` | batch lhs/rhs `[0]`; contract lhs `[2,4]`, rhs `[3,2]` | `C: f64[n,n,n,n,4]` |
| `f64_rank4_row_major_view_mixed_contract_axes` | Can the provider consume borrowed row-major positive-stride views directly instead of first canonicalizing them to owned column-major tensors? | `C[a,b,c,d] = sum_{x,y} A[a,x,b,y] B[c,y,x,d]` | borrowed `A,B: f64[n,n,n,n]`, row-major strides `[n^3,n^2,n,1]` | lhs `[1,3]`, rhs `[2,1]` | `C: f64[n,n,n,n]` |

- Default/small selected results:

| Case | Parameter | faer | BLAS | TBLIS |
| --- | ---: | ---: | ---: | ---: |
| `f64_matrix_square_gemm` | `N=32` | 3.42 us | 1.98 us | 5.66 us |
| `f64_matrix_square_gemm` | `N=64` | 12.71 us | 19.77 us | 19.52 us |
| `f64_matrix_square_gemm` | `N=128` | 86.88 us | 99.27 us | 101.31 us |
| `c64_matrix_square_gemm_lhs_conj` | `N=32` | 6.38 us | 12.10 us | 10.92 us |
| `c64_matrix_square_gemm_lhs_conj` | `N=64` | 53.72 us | 60.93 us | 53.35 us |
| `c64_matrix_square_gemm_lhs_conj` | `N=128` | 385.69 us | 406.12 us | 333.33 us |
| `f64_rank4_packed_contract_axes` | `n=4` | 2.02 us | 0.88 us | 3.40 us |
| `f64_rank4_packed_contract_axes` | `n=8` | 12.82 us | 20.60 us | 19.44 us |
| `f64_rank4_mixed_contract_axes` | `n=4` | 3.37 us | 4.05 us | 3.72 us |
| `f64_rank4_mixed_contract_axes` | `n=8` | 16.79 us | 29.64 us | 19.08 us |
| `f64_rank5_batched_mixed_contract_axes` | `n=4`, batch `4` | 5.67 us | 6.42 us | 10.38 us |
| `f64_rank5_batched_mixed_contract_axes` | `n=8`, batch `4` | 62.11 us | 108.55 us | 74.58 us |
| `f64_rank4_row_major_view_mixed_contract_axes` | `n=4` | 3.49 us | 5.49 us | 3.88 us |
| `f64_rank4_row_major_view_mixed_contract_axes` | `n=8` | 17.90 us | 42.66 us | 20.13 us |
- The expanded benchmark does not show a broad TBLIS win on this machine.
  TBLIS is competitive with or faster than BLAS on several complex and mixed-axis
  cases, but faer remains faster for most measured f64 higher-rank cases at
  these sizes. PR messaging should present this as an optional provider and
  source-build/correctness integration, not a demonstrated performance win.
- Large-size quick local release sample:
  `TENFERRO_TBLIS_BENCH_MATMUL_SIZES=256,512 TENFERRO_TBLIS_BENCH_HIGHER_RANK_NS=16`
  with the same pinned single-thread settings and `blas-openblas`.
- Large-size selected results:

| Case | Parameter | faer | BLAS | TBLIS |
| --- | ---: | ---: | ---: | ---: |
| `f64_matrix_square_gemm` | `N=256` | 617.38 us | 748.55 us | 645.64 us |
| `f64_matrix_square_gemm` | `N=512` | 5.04 ms | 4.91 ms | 4.78 ms |
| `c64_matrix_square_gemm_lhs_conj` | `N=256` | 2.93 ms | 2.77 ms | 2.39 ms |
| `c64_matrix_square_gemm_lhs_conj` | `N=512` | 23.37 ms | 22.08 ms | 18.32 ms |
| `f64_rank4_packed_contract_axes` | `n=16` | 621.53 us | 621.47 us | 646.96 us |
| `f64_rank4_mixed_contract_axes` | `n=16` | 672.51 us | 676.83 us | 646.17 us |
| `f64_rank5_batched_mixed_contract_axes` | `n=16`, batch `4` | 3.03 ms | 3.10 ms | 2.67 ms |
| `f64_rank4_row_major_view_mixed_contract_axes` | `n=16` | 935.45 us | 954.64 us | 651.67 us |
- Larger cases are more favorable to TBLIS than the default small quick bench,
  especially complex GEMM, batched mixed-axis higher-rank contraction, and
  direct row-major positive-stride views. The broad claim should still remain
  measured and size-dependent.
