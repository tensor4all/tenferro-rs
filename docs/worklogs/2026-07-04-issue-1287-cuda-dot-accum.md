# Issue 1287 CUDA Native Dot-General Accumulation

## Session Summary

Implemented the CUDA-native fast path for the #1286 dot-general accumulation
contract: `CudaBackend::dot_general_read_into_accum` now executes
`out = alpha * op(lhs) * op(rhs) + beta * out` as a single cuTENSOR
contraction with `C = D = out` — no temporary result tensor, no separate
accumulator allocation, and no host transfer.

## Context Read

- Issue #1287 (scope, staged plan, non-goals, acceptance criteria) and the
  #1286 accumulation contract (`DotGeneralAccumulation`, `ContractionScalar`,
  `dot_general_read_into_accum` / `_cached` hooks, `dot_general_accum_via_temp`).
- Existing CUDA contraction machinery in `crates/tenferro-gpu/src/cubecl/`:
  `gemm.rs` (`dot_general_typed_with_conj`, `build_layout`, descriptor and
  plan construction), `ffi/cutensor.rs` (`CutensorHandle::contract` already
  passes `alpha`/`beta` pointers), `dispatch.rs` launch helpers, and the
  device-transfer contract in `REPOSITORY_RULES.md`.
- CPU accum wiring from #1286 (`dot_general_blas_accum_typed`,
  `scale_empty_contract_output`) as the semantic reference for the
  zero-contraction degenerate case.

## Decisions Made

- **Single contraction, in-place destination**: the existing overwrite path
  allocated both an output tensor and a separate zero accumulator with
  hard-coded `alpha = 1, beta = 0`. The accum path reuses the identical
  descriptor/plan construction and passes the caller's typed `alpha`/`beta`
  with the destination bound as both C and D. cuTENSOR itself skips reading C
  when `beta == 0`, satisfying that acceptance criterion without a branch.
- **Stage 1 operand scope**: compact GPU-resident owned tensors on all three
  slots. `TensorRead::View` / `TensorWrite::View` return explicit backend
  errors (per the issue's staged plan); residency is enforced by the existing
  `typed_device_ptr` checks, so CPU tensors are rejected rather than silently
  uploaded.
- **Zero-sized contraction** (`contracting_elements == 0`): `out = beta * out`
  on device. `beta == 1` is a no-op, `beta == 0` reuses `fill_zero_kernel`,
  and the general case launches new in-place scale kernels
  (`scale_in_place_float_kernel` / `scale_in_place_complex_kernel`) against a
  one-element device constant materialized via an explicit `upload_tensor`
  call (a scalar constant, never user data).
- **Coefficient extraction**: a private `FromContractionScalar` in `gemm.rs`
  converts `ContractionScalar` to the operand scalar type; mismatches are
  typed `DTypeMismatch` errors (acceptance criterion for dtype rejection).
- **Hook placement**: the override lives on
  `TensorDot::dot_general_read_into_accum`; the `BackendCachedDot` cached hook
  default already routes there, and no cache-slot ownership was moved onto
  `TensorDot` (issue non-goal).

## Rejected Alternatives

- Expressing the degenerate `out = beta * out` as a dummy cuTENSOR
  contraction with `alpha = 0`: works but abuses the contraction API for an
  elementwise scale; a dedicated 6-line kernel is clearer and reusable.
- Falling back to `dot_general_accum_via_temp` for views: the default helper
  materializes via `to_tensor`, which errors for device-backed views anyway;
  an explicit, immediate backend error is more honest than a deep failure.

## Verification

- `cargo check -p tenferro-gpu` and `cargo check -p tenferro-gpu --features
  cuda` clean; `cargo test -p tenferro-gpu --features cuda --no-run` builds.
- `cargo fmt --all --check` and full workspace tests pass; clippy introduces
  no new warnings on touched files.
- New ignored CUDA tests (`cubecl/tests/gemm_accum_tests.rs`) cover overwrite
  compatibility, nontrivial alpha/beta, complex lhs conjugation, zero-sized
  contraction beta-scaling, dtype-mismatch rejection, and the explicit
  view-output error, comparing against the CPU accumulation reference.

## Residual Risks

- GPU-runner execution of the ignored test lane is pending (no CUDA device on
  the development host); to be run on an A100 machine before merge.
- View support (stage 2) and WebGPU accumulation remain out of scope per the
  issue.
