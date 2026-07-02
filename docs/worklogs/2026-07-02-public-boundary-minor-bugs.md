# 2026-07-02 Public Boundary Minor Bugs

## Summary

Addressed a small batch of validation and public-boundary issues from the
July 2 audit set plus older minor issues that shared the same typed-error
contract. The branch keeps behavior changes local to graph-build, tensor-core,
eager standard-op, and CUDA launch-metadata validation paths.

## Issue Ledger

- #1257: fixed negative-step slice normalization for negative `end` values
  below `-1`.
- #1277: changed `TensorScalar::into_tensor` to return `Result<Tensor>` and
  validate shape/data length through `HostTensor::from_vec_col_major`.
- #1245: changed `TracedTensor::reshape` to return `Result<TracedTensor>` and
  reject concrete element-count mismatches at graph build time.
- #1273: converted traced ordered op dtype inference failures for complex
  dtypes into typed graph-build errors instead of panics.
- #1274: added eager standard-op input-count validation before dispatch.
- #1251: aligned CUDA gather/scatter launch-metadata validation with CPU
  indexing contracts for offset dims, collapsed slice sizes, and update batch
  extents.
- #1278: updated stale repository-rule text for `host_data()` and
  `host_data_mut()` returning `Result`.

## Decisions

- Accepted the public trait/API return-value changes because this batch is not
  preserving backward compatibility.
- Kept symbolic traced reshape permissive when the input element count cannot
  be proven at graph build time; runtime compilation/execution still validates
  concrete bindings.
- Added CUDA validation tests against private launch metadata helpers so the
  invalid-config behavior is covered without requiring local GPU execution.

## Verification

- `cargo fmt --all --check`
- `cargo test -p tenferro-tensor-core --test core`
- `cargo test -p tenferro-runtime --lib`
- `cargo test -p tenferro-linalg --lib`
- `cargo test -p tenferro-ad --test extension_op`
- `cargo test -p tenferro-gpu --features cuda launch_meta_rejects`
- `cargo test -p tenferro-xla lowers_structural_ops_and_convert`
- `cargo check --workspace --all-targets`
- `git diff --check`

## Residual Risks

- Full CUDA kernel execution for gather/scatter still depends on GPU CI; the
  local coverage here is launch-metadata validation.
- Larger performance/design reports from the same umbrella, including cache,
  FFT, einsum, and API design proposals, remain deferred for separate design
  or measurement work.
