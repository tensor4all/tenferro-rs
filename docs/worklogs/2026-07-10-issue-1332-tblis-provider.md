# Issue #1332 TBLIS CPU Provider

## Summary

Added an initial optional `cpu-tblis` provider path for dense CPU
`dot_general` contractions. The patch is intentionally narrow: feature wiring,
`CpuBackendKind::Tblis`, a private TBLIS FFI leaf, and dispatch from
`TensorDot`/cached session dot-general paths.

## Context Read

- `AGENTS.md`
- `REPOSITORY_RULES.md`
- `crates/tenferro-cpu/src/backend.rs`
- `crates/tenferro-cpu/src/exec_session.rs`
- `crates/tenferro-cpu/src/gemm/mod.rs`
- `crates/tenferro-cpu/src/gemm/{faer_gemm.rs,blas_gemm.rs}`
- `crates/tenferro-cpu/tests/provider_feature_contract.rs`
- shared `tensor4all-agent-rules` common/Rust rules from the upstream repository
- local cargo registry sources for `tblis-ffi 0.2.6`

## Decisions

- Kept `tblis-ffi = 0.2.6` optional behind `cpu-tblis`; default features are
  unchanged.
- Added tenferro-owned `tenferro-tblis-src` source-build link glue behind
  `cpu-tblis`, so the user-facing feature is source-backed rather than
  requiring a manual `libtblis` install or caller environment workaround.
- Kept `cpu-tblis` additive to an existing CPU provider: `tenferro-cpu` still
  requires `cpu-faer` or `cpu-blas`, and `CpuBackendKind::default_compiled()`
  remains unchanged. This preserves current defaults and guarantees unsupported
  TBLIS cases have a compiled fallback path.
- Added `CpuBackendKind::Tblis`, but kept low-level TBLIS helpers `pub(crate)`.
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
  outer Rayon installation for `CpuBackendKind::Tblis` and leaves native TBLIS
  scheduling provider-owned.
- Broader TBLIS benchmark coverage beyond the quick local release-mode run with
  pinned provider thread counts.

## Provider-source integration review

- Current feature shape is the right user-facing direction:
  `tenferro-cpu/cpu-tblis` enables `dep:tblis-ffi`,
  `dep:tenferro-tblis-src`, and
  `tenferro-tblis-src/build_from_source` plus
  `tenferro-tblis-src/static`. Public passthrough crates forward `cpu-tblis`,
  and the provider feature contract covers this.
- `tblis-ffi` has no source-build feature of its own. Tenferro must keep both
  dependencies directly: `tenferro-tblis-src` supplies Cargo link directives,
  and `tblis-ffi` supplies extern declarations.
- Do not enable `tblis-ffi/dynamic_loading` for the source-backed feature. With
  `tenferro-tblis-src`, the FFI should use the non-dynamic extern path, and
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
  `#[cfg(feature = "cpu-tblis")] extern crate tenferro_tblis_src as _;`, so
  Cargo keeps the tenferro-owned source/link crate in the Rust crate graph and
  propagates its native `cargo:rustc-link-lib=static=tblis` directive.
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
  Both use `CpuBackendKind::Tblis` and assert successful results through the
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
- Added `crates/tenferro-tblis-src`, a small tenferro-owned link crate. It does
  not vendor a TBLIS source tree. By default it clones
  `https://github.com/MatthewsResearchGroup/tblis.git` at
  `eb719e718976572e0ab53975f4e0c799faeb35f2`, still honors
  `TBLIS_SRC`/`TBLIS_VER`, uses `SOURCE_DIR` for local `TBLIS_SRC` directories,
  and passes Cargo native link directives for `libtblis`.
- For cloned TBLIS sources, the build glue patches only the downloaded TBLIS
  `CMakeLists.txt` BLIS discovery block so bundled BLIS is used. This avoids the
  Homebrew/pkg-config header mismatch without requiring users to empty
  `PKG_CONFIG_PATH` or CMake package registries.
- Updated `cpu-tblis` feature wiring to:
  `["dep:tblis-ffi", "dep:tenferro-tblis-src", "tenferro-tblis-src/build_from_source", "tenferro-tblis-src/static"]`,
  so the source-backed provider links `libtblis.a` by default.
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
- Added owned higher-rank f64 contractions:
  - rank-4 x rank-4 with two contracting dimensions and no batch,
  - interleaved rank-4 x rank-4 with two contracting dimensions,
  - rank-5 x rank-5 with one batch dimension and two contracting dimensions.
- Added a non-canonical positive-stride view case through `TensorRead` using
  row-major rank-4 host views. This exercises TBLIS's direct strided tensor
  contraction path without materializing a compact col-major tensor first.
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
- Selected results:
  - f64 matmul 32: faer 3.42 us, BLAS 1.98 us, TBLIS 5.66 us
  - f64 matmul 64: faer 12.71 us, BLAS 19.77 us, TBLIS 19.52 us
  - f64 matmul 128: faer 86.88 us, BLAS 99.27 us, TBLIS 101.31 us
  - c64 lhs-conj matmul 32: faer 6.38 us, BLAS 12.10 us, TBLIS 10.92 us
  - c64 lhs-conj matmul 64: faer 53.72 us, BLAS 60.93 us, TBLIS 53.35 us
  - c64 lhs-conj matmul 128: faer 385.69 us, BLAS 406.12 us, TBLIS 333.33 us
  - rank-4 two-contract n=4: faer 2.02 us, BLAS 0.88 us, TBLIS 3.40 us
  - rank-4 two-contract n=8: faer 12.82 us, BLAS 20.60 us, TBLIS 19.44 us
  - rank-4 interleaved n=4: faer 3.37 us, BLAS 4.05 us, TBLIS 3.72 us
  - rank-4 interleaved n=8: faer 16.79 us, BLAS 29.64 us, TBLIS 19.08 us
  - rank-5 batched interleaved n=4: faer 5.67 us, BLAS 6.42 us, TBLIS 10.38 us
  - rank-5 batched interleaved n=8: faer 62.11 us, BLAS 108.55 us, TBLIS 74.58 us
  - rank-4 row-major view n=4: faer 3.49 us, BLAS 5.49 us, TBLIS 3.88 us
  - rank-4 row-major view n=8: faer 17.90 us, BLAS 42.66 us, TBLIS 20.13 us
- The expanded benchmark does not show a broad TBLIS win on this machine.
  TBLIS is competitive with or faster than BLAS on several complex/interleaved
  cases, but faer remains faster for most measured f64 higher-rank cases at
  these sizes. PR messaging should present this as an optional provider and
  source-build/correctness integration, not a demonstrated performance win.
- Large-size quick local release sample:
  `TENFERRO_TBLIS_BENCH_MATMUL_SIZES=256,512 TENFERRO_TBLIS_BENCH_HIGHER_RANK_NS=16`
  with the same pinned single-thread settings and `blas-openblas`.
- Large-size selected results:
  - f64 matmul 256: faer 617.38 us, BLAS 748.55 us, TBLIS 645.64 us
  - f64 matmul 512: faer 5.04 ms, BLAS 4.91 ms, TBLIS 4.78 ms
  - c64 lhs-conj matmul 256: faer 2.93 ms, BLAS 2.77 ms, TBLIS 2.39 ms
  - c64 lhs-conj matmul 512: faer 23.37 ms, BLAS 22.08 ms, TBLIS 18.32 ms
  - rank-4 two-contract n=16: faer 621.53 us, BLAS 621.47 us, TBLIS 646.96 us
  - rank-4 interleaved n=16: faer 672.51 us, BLAS 676.83 us, TBLIS 646.17 us
  - rank-5 batched interleaved n=16: faer 3.03 ms, BLAS 3.10 ms, TBLIS 2.67 ms
  - rank-4 row-major view n=16: faer 935.45 us, BLAS 954.64 us, TBLIS 651.67 us
- Larger cases are more favorable to TBLIS than the default small quick bench,
  especially complex GEMM, batched/interleaved higher-rank contraction, and
  direct row-major positive-stride views. The broad claim should still remain
  measured and size-dependent.
