# Issue #1397 Cargo Build-Artifact Reduction

## Session summary

Measured the tenferro workspace and a controlled CUDA-enabled downstream
fixture through four cumulative experiments. The principal workspace release
improvement came from consolidating integration tests: the fresh target fell
from 3.20 GiB to 1.49 GiB. Dependency pruning then reduced it to 1.44 GiB.

The clean release build did not reproduce the previously observed 20 GiB.
However, a fresh unoptimized workspace test build with full debuginfo did:
23.80 GiB after integration-test consolidation. The source of the observation
is therefore unoptimized generic code plus DWARF repeated across workspace test
executables, amplified further in a developer/CI target that retains feature,
coverage, and CUDA variants.

The fastest measured workspace-CI candidate is unoptimized tests with debuginfo
disabled: 7.52 GiB in 1m58s, versus 23.79 GiB in 2m37s for normal debug and
1.44 GiB in 10m23s for release. This profile experiment is evidence for a
follow-up CI change; this PR does not change CI configuration.

## Measurement environment and method

- Host: Linux 6.8, `x86_64-unknown-linux-gnu`
- CPU: AMD EPYC 7713P; Cargo constrained to four jobs
- Rust: `rustc 1.96.0 (ac68faa20 2026-05-25)`, LLVM 22.1.2
- Cargo: `cargo 1.96.0 (30a34c682 2026-05-25)`
- Every row used a separate, initially absent `CARGO_TARGET_DIR`.
- `CARGO_BUILD_JOBS=4`, `CARGO_INCREMENTAL=0`, and an empty
  `RUSTC_WRAPPER` were used throughout.
- CUDA builds used `CUDARC_CUDA_VERSION=12080`; libraries were dynamically
  loaded and no GPU execution was performed.
- Sizes are allocated bytes from `du -s -B1`, converted to binary units.
- Times are one cold sample and include dependency resolution where required;
  differences of only a few seconds should be treated as noise.

Workspace release command:

```bash
env CARGO_TARGET_DIR=<fresh> CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=0 \
  RUSTC_WRAPPER= /usr/bin/time cargo test --workspace --release --no-run
```

Controlled downstream command (`--release` omitted for debug):

```bash
env CUDARC_CUDA_VERSION=12080 CARGO_TARGET_DIR=<fresh> \
  CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= \
  /usr/bin/time cargo build --release
```

The downstream fixture used normal path dependencies on `tenferro-cpu`,
`runtime`, `ad`, `gpu`, `einsum`, `linalg`, and `fft`, with the exact
`cpu-faer`, `autodiff`, and `cuda` feature configuration from issue #1397.
It exists only to separate ordinary dependency cost from workspace-test
amplification; the workspace table below is the CI capacity result.

## Workspace CI results

| Stage | Workload | Target | deps | build | incremental | Cold time | Resolved packages / integration targets |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 baseline (`9182bff1`) | release workspace tests | 3.20 GiB | 3.01 GiB | 43.68 MiB | 0 | 10m47.1s | 214 / 89 |
| 1 consolidated tests (`faa05e0a`) | release workspace tests | 1.49 GiB | 1.30 GiB | 43.68 MiB | 0 | 10m31.5s | 214 / 12 |
| 2 minimal faer (`0c175022`) | release workspace tests | 1.44 GiB | 1.25 GiB | 43.68 MiB | 0 | 10m22.2s | 203 / 12 |
| 3 optional linalg providers (`f8d1bba4`) | release workspace tests | 1.44 GiB | 1.25 GiB | 43.68 MiB | 0 | 10m23.4s | 201 / 12 |
| 2 minimal faer | normal debug workspace tests | 23.80 GiB | 19.72 GiB | 144.21 MiB | 0 | 2m37.4s | 203 / 12 |
| 3 optional linalg providers | normal debug workspace tests | 23.79 GiB | 19.71 GiB | 144.21 MiB | 0 | 2m37.3s | 201 / 12 |
| 4 final + CI profile probe | `opt-level=0`, `debug=0` workspace tests | 7.52 GiB | 6.25 GiB | 44.02 MiB | 0 | 1m58.3s | 201 / 12 |

Experiment 4 changes only optional CUDA dependencies, so the default workspace
release workload is unchanged from experiment 3. Its CUDA effect is measured
in the downstream table.

### Workspace incremental and cumulative deltas

| Stage | Incremental target delta | Cumulative target delta from stage 0 | Incremental time delta | Cumulative time delta |
|---|---:|---:|---:|---:|
| 1 integration consolidation | -1,832,570,880 B (-53.45%) | -53.45% | -15.56s (-2.40%) | -2.40% |
| 2 minimal faer | -53,325,824 B (-3.34%) | -55.01% | -9.32s (-1.48%) | -3.84% |
| 3 optional providers | -6,516,736 B (-0.42%) | -55.19% | +1.14s (noise) | -3.67% |

The profile probe is not cumulative experiment 5. Relative to stage 3 normal
debug, setting only test-profile debuginfo to zero reduced allocated disk by
17,470,427,136 B (68.40%) and cold time by 39.01s (24.80%), while preserving
`opt-level=0`.

## Experiment 1: integration-test consolidation

Seven multi-file suites now use `autotests = false` and one explicit harness
per crate. Existing source files remain child modules, so test names and source
organization remain recognizable. Crate boundaries were retained because a
cross-crate omnibus harness would merge feature sets and weaken commands such
as `cargo test -p tenferro-linalg --no-default-features --features cpu-blas`.

- Integration targets: 89 to 12.
- Cargo-reported test/bin executables: 111 to 34.
- Their total size: 2,365,876,104 B to 534,886,072 B (-77.39%).
- Process-level test parallelism is lower, but each libtest harness still runs
  tests in parallel and failure names retain module paths.
- Failure isolation remains at the crate boundary, which is also the feature
  and ownership boundary.
- The remaining five singleton public-contract/tutorial integration targets
  were not converted to unit tests; doing so would change what they verify for
  negligible linking savings.

The disk improvement is large while the release wall-clock change is only
2.4%, because LLVM optimization of the two large `tenferro-cpu` lib and
lib-test units dominates the critical path.

## Experiment 2: minimal faer features

Both tenferro and `strided-einsum2` now request faer with defaults disabled and
only `std` and `rayon`. No sparse, NumPy, random-generation, or parser API is
used by the dense CPU backend. `strided-einsum2` remains the owner of
`dot_general_with_backend_into`; moving its implementation into tenferro would
duplicate stride, broadcast, batching, buffer, and Rayon-thread management.

The strided crates are temporarily pinned together to revision
`017c7e2413e48e5182590eed9b2e99350cbd5283`. Pinning only
`strided-einsum2` produced duplicate git/crates.io builds of its sibling
strided crates, so all five source-compatible 0.3.0 crates use the same source.
The upstream change is [strided-rs PR #143](https://github.com/tensor4all/strided-rs/pull/143).

Workspace resolution removed 11 packages: `npyz`, `py_literal`, `pest`,
`pest_derive`, `pest_generator`, `pest_meta`, `ucd-trie`, `rand 0.8`,
`rand_core 0.6`, `rand_distr`, and `num-bigint`. The CUDA downstream graph
already needs `num-bigint` through CubeCL, so that fixture removes 10 packages.
The resolved faer feature tree contains only `std` and `rayon`.

## Experiment 3: optional linalg providers

`tenferro-linalg` now activates `faer` from `cpu-faer` and `lapack` from
`cpu-blas`; both dependencies are optional. Provider injection remains behind
its existing feature. Dependency-tree and compile checks established:

- faer-only excludes `lapack` and `lapack-sys`;
- BLAS-only excludes faer;
- faer + BLAS compiles and preserves additive runtime provider selection.

The default faer workload therefore removes two packages and about 6 MiB. The
small size result is expected; the important outcome is that downstream users
no longer compile a provider they did not select.

## Experiment 4: CubeCL/cudarc dependency roles

The old CubeCL edge enabled `cuda-version-from-build-system`,
`fallback-dynamic-loading`, and `fallback-latest` in addition to tenferro's
explicit CUDA 12.8/dynamic-loading contract. These three fallback/detection
features are removed.

An initial implementation gave normal and build dependencies identical
features. Measurement rejected that design: Cargo still emits separate normal
and build units, while the build-script cudarc rlib grew from 16.9 MiB to
26.3 MiB. The downstream release target increased by about 11.0 MiB. Textual
feature unification does not merge Cargo dependency roles.

The final design instead uses role-specific minimal sets:

- normal: `std`, `driver`, `runtime`, `nvrtc`, `nccl`, `dynamic-loading`,
  `cuda-12080`;
- build: `std`, `driver`, `dynamic-loading`, `cuda-12080`.

The build script only reads `cudarc::driver::sys::CUDA_VERSION`. `dynamic-loading`
is still required by cudarc's own build contract. NCCL cannot currently be
removed from the normal dependency: CubeCL's CUDA server unconditionally uses
NCCL communicator, all-reduce, send, and receive APIs. Making communication
optional would be a separate CubeCL design change.

Two cudarc rlibs remain because Cargo separates normal and build dependencies;
this duplication is unavoidable without eliminating the build dependency.
Their release sizes changed from 16.9 + 11.9 MiB to 14.1 + 11.9 MiB. The
upstream change is [CubeCL PR #11](https://github.com/tensor4all/cubecl/pull/11).

## Controlled downstream results

This table is supplementary and must not be used as the tenferro CI capacity
estimate. It demonstrates the dependency-only effects without integration-test
executable multiplication.

| Stage | Profile | Target | deps | build | incremental | Cold time | Packages |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 pre-faer baseline | release | 884.27 MiB | 826.94 MiB | 52.11 MiB | 0 | 7m44.8s | 369 |
| 2 minimal faer | release | 842.12 MiB | 785.00 MiB | 52.11 MiB | 0 | 7m37.2s | 359 |
| 3 optional providers | release | 835.94 MiB | 778.85 MiB | 52.11 MiB | 0 | 7m39.3s | 357 |
| 4 role-minimal cudarc | release | 832.21 MiB | 775.32 MiB | 51.90 MiB | 0 | 7m33.2s | 357 |
| 1 pre-faer baseline | debug | 2.84 GiB | 2.63 GiB | 212.13 MiB | 0 | 2m16.5s | 369 |
| 2 minimal faer | debug | 2.81 GiB | 2.60 GiB | 212.13 MiB | 0 | 2m09.3s | 359 |
| 3 optional providers | debug | 2.80 GiB | 2.59 GiB | 212.13 MiB | 0 | 2m09.6s | 357 |
| 4 role-minimal cudarc | debug | 2.80 GiB | 2.59 GiB | 211.90 MiB | 0 | 2m10.9s | 357 |

### Downstream deltas

| Stage | Release incremental / cumulative disk | Debug incremental / cumulative disk | Release incremental / cumulative time |
|---|---:|---:|---:|
| 2 minimal faer | -44,191,744 B / -4.77% | -32,161,792 B / -1.06% | -7.59s / -1.63% |
| 3 optional providers | -6,483,968 B / -5.47% | -6,258,688 B / -1.26% | +2.09s / -1.18% |
| 4 role-minimal cudarc | -3,911,680 B / -5.89% | -3,706,880 B / -1.38% | -6.09s / -2.49% |

Debug time at the final stage is 5.60s (4.10%) below the stage-1 baseline;
the +1.23s stage-4 difference from stage 3 is measurement noise.

## Why normal debug is still 23.8 GiB

A representative consolidated linalg integration executable is 910.8 MB in
normal debug and 50.3 MB in release. ELF section inspection found:

| Section family | Debug | Release |
|---|---:|---:|
| `.debug_str` | 378.8 MB | removed |
| `.debug_info` | 112.3 MB | removed |
| `.debug_line` | 64.1 MB | removed |
| other DWARF (`ranges`, `loc`, `aranges`) | about 52.2 MB | removed |
| `.text` | 200.1 MB | 40.4 MB |

Roughly two thirds of this debug executable is DWARF. The remaining difference
includes unoptimized monomorphized generic code: release reduces `.text` by
about 80%. Both are repeated in independently linked test executables. This
explains why integration consolidation is necessary but does not by itself
make full-debuginfo debug tests fit a standard runner.

## Baseline largest artifacts

The 30 largest stage-0 files (hard-linked top-level/deps copies can both appear
in this inventory) were:

<details>
<summary>Stage-0 top 30</summary>

| Bytes | Path |
|---:|---|
| 61,077,594 | `release/deps/libtenferro_cpu-c9406195cf61993f.rlib` |
| 48,983,824 | `release/deps/traced_ad_explicit-82db6d70fd2ea082` |
| 48,347,552 | `release/dynamic_shape_truncated_svd` |
| 48,347,552 | `release/deps/dynamic_shape_truncated_svd-3f9e134d1a9d27d9` |
| 47,907,720 | `release/deps/eager_tensor-ac514cb91e66cc31` |
| 47,833,680 | `release/deps/tenferro_einsum-7d7ce5419da23155` |
| 47,829,024 | `release/deps/traced_extension-1346f76ef86da005` |
| 47,786,736 | `release/deps/traced_correctness-73ab20d1aab2cff8` |
| 46,576,856 | `release/deps/tenferro_cpu-abbf55bc0d8438b8` |
| 46,419,792 | `release/einsum_subscripts_to_gradients` |
| 46,419,792 | `release/deps/einsum_subscripts_to_gradients-6065b3a737780dc6` |
| 46,316,512 | `release/deps/traced_extension-42ea5ba68ea8f146` |
| 46,219,168 | `release/deps/traced_graph_cache-91bf4bb186a12f69` |
| 46,182,104 | `release/deps/traced_ad_migration-737c50318fb28e2a` |
| 46,158,272 | `release/deps/fft_ops-d66b5d25e678978d` |
| 46,017,056 | `release/deps/eager_tensor-6bb1a77c3aa45caf` |
| 45,969,232 | `release/deps/tenferro_runtime-3dbe6e5c523cac36` |
| 45,923,624 | `release/deps/extension_op-712e2e7b727facde` |
| 45,894,920 | `release/deps/ad-fbf4fe327423fb4c` |
| 45,746,456 | `release/deps/traced_correctness-6abdba2e151ebc77` |
| 45,521,496 | `release/examples/traced_fft-ee110be20077acce` |
| 45,521,496 | `release/examples/traced_fft` |
| 45,494,040 | `release/deps/cache_management-8c63687af527c08e` |
| 45,298,792 | `release/deps/tenferro_ad-396e6cc0c59d3982` |
| 45,290,680 | `release/deps/runtime_buffer_pool-69cab09ad242b90f` |
| 45,092,328 | `release/deps/graph_executor-2f4c2ea89c354402` |
| 45,080,192 | `release/xla_einsum_backend` |
| 45,080,192 | `release/deps/xla_einsum_backend-8a58f9e2be167bd0` |
| 45,079,568 | `release/deps/numpy_api-dec8eb892118e37d` |
| 45,039,600 | `release/deps/dynamic_truncate-dd277db11968c535` |

</details>

## Verification performed

- Contract tests in `scripts/ci/tests/test_build_artifact_contracts.py` cover
  consolidated harnesses, faer defaults/revision, linalg provider isolation,
  and CubeCL/cudarc revision and direct feature contract.
- `cargo check -p tenferro-linalg --no-default-features --features cpu-faer`
- `cargo check -p tenferro-linalg --no-default-features --features cpu-blas`
- `cargo check -p tenferro-linalg --no-default-features --features cpu-faer,cpu-blas`
- `cargo check -p tenferro-linalg --no-default-features` fails as designed with
  `enable at least one fallback CPU backend: cpu-faer or cpu-blas`.
- Dependency trees prove faer-only excludes LAPACK and BLAS-only excludes faer.
- `CUDARC_CUDA_VERSION=12080 cargo check -p tenferro-gpu -p tenferro-ad
  -p tenferro-linalg --features cuda`
- CubeCL: feature-contract test, `cargo fmt --all --check`, and
  `CUDARC_CUDA_VERSION=12080 cargo check -p t4a-cubecl-cuda`.
- Every measurement command completed successfully; all workspace release and
  debug rows used `cargo test --workspace --no-run`, which compiles every
  consolidated integration harness.
- A final `cargo test --workspace` using the measured unoptimized/debug-zero
  target executed all unit, integration, tutorial-contract, and doc tests with
  no failures.
- `bash scripts/check-pr-fast.sh --coverage-reviewed --ci-profile local-gate`
  passed: 34 binaries, 2,209 nextest tests, doctests, doc snippets, and fmt.
- `repository-rules-review.py --base origin/main --head HEAD` returned `pass`
  with no findings.

## Remaining risks and follow-up

- The strided and CubeCL git revisions must be merged and released before a
  crates.io publication can rely on version-only manifests.
- Full repository gates (tests, clippy/docs/coverage and CI) still need to run
  on the final branch; the fresh release builds here compiled but did not
  execute tests.
- Standard-runner suitability should be validated with peak, not only final,
  target usage. Measure disk periodically during a cold workspace CI build.
- Compare `debug=0`, `debug=1`, and line-table-only test profiles while keeping
  `opt-level=0`; prefer settings that reduce both debuginfo generation I/O and
  final/peak disk without adding a post-build strip pass.
- Determine whether CI needs debuginfo for test crash diagnosis. If not,
  `CARGO_PROFILE_TEST_DEBUG=0` is the leading fast/disk-bounded configuration.
- Incremental compilation was deliberately disabled in these cold CI
  measurements. Keep incremental enabled for local AI/developer edit loops;
  evaluate it separately for CI cache/peak-disk policy.
