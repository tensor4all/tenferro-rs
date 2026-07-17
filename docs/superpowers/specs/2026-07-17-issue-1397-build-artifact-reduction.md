# Issue #1397 build-artifact reduction design

## Goal

Reduce cold Cargo build disk usage and wall time without changing tenferro's
public API, default CPU provider, numerical behavior, AD support, or CUDA
support. Measure each change from a fresh target directory so dependency cost
and workspace test-target amplification remain distinguishable.

## Design

### Integration tests

Set `autotests = false` in crates with multiple integration tests and declare
one explicit integration-test harness per crate. Keep each existing test file
as a child module under `tests/<suite>/`; this preserves test names and source
separation while linking the crate and its generic providers once per harness.
Start with the crates responsible for most targets (`tenferro-ad`,
`tenferro-linalg`, `tenferro-einsum`) and consolidate the remaining multi-test
crates when they do not require incompatible crate-level configuration.

### faer features and strided-rs ownership

Keep `strided-einsum2` as the owner of strided dot-general planning and
execution. tenferro deliberately delegates `dot_general` to
`strided_einsum2::dot_general_with_backend_into`; copying that implementation
back into tenferro would duplicate stride, broadcast, batching, and backend
maintenance.

The current broad faer graph is instead caused by dependency declarations:
both tenferro and `strided-einsum2` must disable faer's defaults and explicitly
enable only the dense standard-library and threading capabilities used by the
CPU backend. The strided-rs correction is an upstream prerequisite; tenferro
then updates to the corrected release or revision. `npy`, `rand`, `sparse`, and
`sparse-linalg` are not part of tenferro's faer contract.

### Optional linalg providers

Keep `cpu-faer` as the default feature, but make the `faer` dependency optional
and activate it only from `cpu-faer`. Make `lapack` optional and activate it
only from `cpu-blas`. Both features remain additive, and builds with either
provider or both providers must compile. Runtime provider selection is
unchanged.

### CUDA dependency ownership

Treat `tenferro-gpu` as the owner of CubeCL/cudarc runtime integration.
Operation-family crates should request `tenferro-gpu/cuda` instead of declaring
parallel direct CubeCL/cudarc edges unless their own source directly imports
those crates. Disable dependency defaults and align the CUDA binding floor and
dynamic-loading features at the workspace owner.

Cargo builds normal and build dependencies as distinct units even when their
feature sets are identical. Do not broaden a build dependency merely to make
its features textually match the normal dependency: use the same explicit CUDA
floor and loading mode, but give each dependency role only the API families it
actually uses. For CubeCL, the normal dependency needs driver, runtime, NVRTC,
and NCCL; its build script only reads a driver version constant and therefore
uses the smaller driver-only contract. The two artifacts are unavoidable, but
avoidable fallback/version-detection features are not.

## Measurement contract

Use four Cargo jobs, an empty `RUSTC_WRAPPER`, disabled incremental compilation,
and a fresh `CARGO_TARGET_DIR` for every cold release-test measurement. Record
wall time, total/deps/build/incremental sizes, integration-test executable count
and size, largest artifacts, resolved features, and duplicate packages. Report
incremental and cumulative deltas for baseline, test consolidation, faer
pruning, optional linalg providers, and CUDA unification.

## Verification

In addition to the repository PR gate, verify faer-only, BLAS-only,
both-provider, and CUDA compile configurations. Confirm that no-default retains
the existing explicit "enable at least one fallback CPU backend" compile-time
diagnostic. Dependency-tree assertions must show that faer-only linalg excludes
LAPACK, BLAS-only linalg excludes faer, and the removed faer packages do not
re-enter through strided-rs.
