# HANDOFF: tenferro-rs #2208 PR 2 (CUDA on the kernel catalogue), 2026-10-27

Work in progress, **not pushed, no PR**. Branch `feat/2208-linalg-port`, three commits above
`origin/main` (`234a51369`, the merge of PR 1 / #2288):

```
766813aa0 chore(gpu): retire the aliased U7 harness, the CubeCL pins and the empty webgpu feature
957b5ef1d fix(gpu): the fused update, the conjugate dot and the squared norm come back
d8ddd68de feat(gpu)!: one CUDA path — the kernel-catalogue engine replaces CubeCL
```

Diff against `origin/main`: **260 files, +8612 / −73889** (`git diff --stat -M origin/main...HEAD`).
The `cuda_next` → `cuda` move is detected as a rename, so the substance to review is roughly the
engine under `crates/tenferro-gpu/src/cuda/**`, `crates/tenferro-linalg/**` and the wiring files.

## Status update (2026-10-28): borrowed views landed, device lanes green

Uncommitted delta on top of the three commits above (12 files, +543/−246; `git diff`).

- **Borrowed views resolve.** `cuda/structural.rs`: `device_read` returns `Resolved<'a>`
  (`Borrowed`/`Owned(Box<Tensor>)`, `Deref`) and materializes a view's region through
  `materialize_storage`; `to_contiguous_read`, `copy_read_into`, `vdot_read`,
  `norm_squared_read` and every other `_read` path now serve a device view. `axpby` copies
  nothing: both operands go to `CombinedStrided::Axpby` as their own layout + allocation.
- **Three defects the views exposed, fixed.**
  1. `materialize_storage` passed the *source allocation's* capacity where the kernel expects the
     number of destination elements: a region view over a larger allocation wrote past the compact
     destination, and a stride-0 broadcast view wrote only its first element.
  2. `cuda/runtime_adapter.rs`'s ingress/residency contract answered `false` for every
     `TensorRead::View`, so a produced metadata view failed with `OutputResidencyMismatch`.
  3. `vdot_read`/`norm_squared_read`/`axpby` now call the shared `tenferro_tensor::backend`
     validators, and the allocation-domain check has one owner (`device_read`/`read_operand`).
- **Semantics taken from the CPU surface**, verified against it: a zero-length reduced axis is a
  typed `InvalidArgument`, and an empty `vdot`/`norm_squared` is the additive identity 0.
- **Cleanups** (handoff items 4/6, partially): `dot_native`'s `read_inputs` is `FallbackCopy` (a
  borrowed operand reaches cuBLAS through a copy); `assert_device_backed` pins residency, not the
  string `"cubecl"`; stale `webgpu` rows, dead Apple/WebGPU doctests, the `cuda-next` module prose,
  and the obsolete CubeCL dependency-contract CI test are gone (replaced by a `t4a-cuda-*`
  pinned-revision contract); `scripts/ci/cuda_test_partition.tsv` has the one new row.
- **Tenferro-ad's GPU lane was red on this branch (0 of 11 on the three commits below) and is 11/11
  now.** Remaining for the PR, unchanged from the list below: `launch_planned`'s per-launch module
  build, linalg aux dtype validation, `publish`'s allocation-domain check (#2268), the 123
  `#[doc(hidden)]` marks, `KERNELS.md`, and the stale `kernel_metadata_contract.rs` audit of the
  deleted CubeCL kernels.
- **Evidence (A100 80GB, driver 580.178.04, CUDA 12.6, final state):**
  `cargo test -j 16 --workspace --no-fail-fast` 5545 passed / 0 failed (188 suites);
  `-p tenferro-gpu --features cuda` 182; `blas1_cuda -- --ignored` 12/12 (was 5/12);
  `-p tenferro-linalg --features cuda --test integration -- --ignored` 64/64;
  `-p tenferro-fft --features cuda --test cuda_fft -- --ignored` 24/24;
  `-p tenferro-ad --features cuda --test integration -- --ignored` 11/11 (was 0/11);
  doc tests gpu 33 / fft 21 / linalg 155 / einsum 105;
  `cargo clippy -j 16 --workspace --all-targets --features tenferro-gpu/cuda -- -D warnings` clean;
  `cargo fmt --check` clean; `python3 -m unittest discover -s scripts/ci/tests` 414 OK (was 6 errors).
- **Independent review (gpt-6.1-sol, post mode)** found three items; all were verified against the
  source and folded in (empty-input dtype contract, foreign-domain empties, axpby validator reuse).
  It also suggested deleting `write_region_of` (done: both sides share `read_operand`) and the
  duplicated domain check in `materialize_region` (done). Dismissed: removing the empty branch in
  `norm_squared_read` (the following `reduce_axes` call rejects zero-length axes by design).
- Not a CI gate but red: `RUSTDOCFLAGS="-D warnings" cargo doc` (pre-existing broken intra-doc links
  in `tenferro-tensor-core`, `tenferro-internal-cpu-kernels`, `cuda/identity.rs`).

## What this branch delivers

1. **tenferro-linalg on the new path.** The 14 `#[cube]` linalg kernels and the whole CubeCL
   dependency are gone. Cholesky, LU, QR, compact RRQR, eigh, SVD (gesvd/gesvdj/batched), the QR
   gauges and the factor assembly issue catalogue launches through the engine's typed-operand
   helpers (`linalg_matrix_adjoint`, `linalg_column_norms`, `linalg_select_pivot`,
   `linalg_swap_columns`, `linalg_householder_explicit_v`, `linalg_qr_phase`, `linalg_qr_apply_phase`,
   `linalg_identity_shifted`, `linalg_reflector`, `linalg_apply_reflector`, `linalg_extract_r`,
   `linalg_prefix_rank`, `linalg_householder_assemble`, `linalg_lu_apply_pivots`,
   `linalg_lu_parity`, `linalg_lu_extract_outputs`, `linalg_fill_one`,
   `linalg_upper_trapezoidal_violation`, `linalg_initialize_permutation`, `linalg_triangular`,
   `linalg_slice`), with cuSOLVER/cuBLAS reached through the raw session inside the same operation.
2. **One CUDA path.** `tenferro_gpu::cuda` names the new engine and the feature is plainly `cuda`.
   Deleted: the CubeCL module (kernels, dispatch, fusion, session, interop, tests, benches), the
   WebGPU/Metal lane (`webgpu`, the `apple` namespace, its tests, tutorial Apple binaries and
   examples), the native permutation helper, the CubeCL/CubeK Cargo dependencies and workspace pins,
   the `webgpu` and `cubecl-interop` features, `samples/cubecl-kernel` and
   `docs/guides/custom-cuda-kernels.md`.
3. **`TensorBackendCapability`** on the new engine, written by hand (not derived from the catalogue's
   `accepts`), with `crates/tenferro-gpu/tests/seam_capability.rs` holding 7 dtypes against the
   engine's observed behaviour.
4. **Prepared dispatchers cached per engine** (`PreparedDispatchers` in `cuda/dispatch.rs`):
   preparing a module set is a runtime round trip (1.25 ms `CopyExact`, 1.27 ms `Reduce`, 0.25 ms
   `Add`) and was paid per call.
5. **U7, measured before the deletion** (recorded in
   `docs/worklogs/2026-10-24-cuda-adoption-resume.md`): after the dispatcher cache the replacement
   is faster than the CubeCL engine on every cell (matmul −4.9 %, add −33.7 %, reduce 4096×256
   −28.9 %, reduce 65536×256 −13.3 %, reduce 4096×8 −67.2 %, reduce ×8 −24.2 %), bit-identical
   results. One thread, one worker, bench on L3 domain `0-7`, observer on `48-63`, A100 80GB,
   driver 580.126.09, CUDA 12.6. The harness itself was deleted with the old engine.
6. **Review-driven fixes** (see below), including three restored `Engine` capabilities
   (`vdot_read`, `norm_squared_read`, `axpby_read_into_accum`).

## Verified, with the command that ran last

| gate | last result |
|---|---|
| `cargo check -j 16 --workspace --all-targets --features tenferro-gpu/cuda` | 0 errors (after the final edits) |
| `cargo clippy -j 16 --workspace --all-targets --features tenferro-gpu/cuda -- -D warnings` | clean (after the final edits) |
| `cargo test -j 16 -p tenferro-gpu --features cuda --doc` | 33 passed (after the final edits) |
| `cargo test -j 16 --workspace --no-fail-fast` | green **before** the last commit; the re-run after it was aborted, so re-run it first |
| `cargo test -p tenferro-linalg --features cuda --test integration -- --ignored` | 64/64 device tests pass (before the last two commits) |
| `python3 -m unittest discover -s scripts/ci/tests` | 414 OK (before the last two commits) |
| `scripts/repository-rules-review.py --base origin/main`, `check-public-error-docs.py`, `audit-engine-entry.py` | pass (before the last two commits) |

Known red: `cargo test -p tenferro-gpu --features cuda --test blas1_cuda -- --ignored` fails 7 of 12.
Failing: `cuda_axpby_accepts_compact_views_with_offsets`,
`cuda_axpby_accepts_a_non_contiguous_read`, `cuda_axpby_accepts_strided_destination_regions`,
`cuda_reductions_accept_compact_views_with_offsets`, `cuda_reductions_accept_a_non_contiguous_read`,
`cuda_vdot_accepts_two_strided_operands`, `cuda_blas1_handles_empty_inputs`. Passing:
`cuda_axpby_matches_cpu_with_complex_coefficients`, `cuda_blas1_covers_all_supported_dtypes`,
`cuda_blas1_cross_thread_operands_observe_vendor_writes`, and the owned-tensor norm/vdot cases.

## Independent review (gpt-6.1-sol, `independent-review` post mode)

Findings and their status in this branch:

- **done** — `copy_bytes` / `retain_tensor` were safe functions taking raw pointers; both are now
  `unsafe` with a written `# Safety` contract, and all 35 call sites carry SAFETY comments.
- **done** — `vdot_read`, `norm_squared_read`, `axpby_read_into_accum` were missing (Engine defaults
  → `Unsupported`). Now implemented: the first two as catalogue compositions, the third as the
  catalogue's `AXPBY` (in place on the destination, which the ABI allows).
- **done** — `SumSquares` over no axes returned a copy instead of the square.
- **done** — the CUDA tutorial `custom_cuda_kernels` (behind `--features cuda`) referenced the
  deleted `raw::{NvrtcOptions, LaunchConfig, KernelArg}`; deleted with the capability. The surviving
  `cuda_vendor_interop` tutorial was rewritten onto `session.resource(...)` + the stream handle.
- **done** — the U7 bench aliased both engines after the deletion and its residuals were differences
  of absolute-value sums; the harness is deleted and the worklog names the pre-deletion revision.
- **done** — leftover `cubecl-common` dev-dependency, the CubeCL/CubeK workspace pins and
  `tenferro-internal-ops`' empty `webgpu` feature.
- **open** — **borrowed views are rejected across the engine** (`device_read` refuses
  `TensorRead::View`). This is the cause of the seven failing blas1 device tests and it is a
  functional regression against the CubeCL engine, which accepted same-device views.
- **open** — the capability table advertises native borrowed-input/strided-output for dot although
  view paths refuse; tied to the previous item.
- **open** — `linalg` auxiliaries still rebuild `KernelModules` per launch (`launch_planned` in
  `cuda/structural.rs`), so the per-launch preparation cost the U7 worklog identifies is still paid
  inside the RRQR loop.
- **open** — a compiled run takes `extension_caches` out of the engine
  (`take_extension_caches`), so raw linalg sessions populate a temporary store and vendor handles are
  recreated across runs.
- **open** — `publish` compares only the device ordinal, not the allocation domain.
- **open** — `#[doc(hidden)]` migration marks and `no_run` examples on the surviving public surface.

## Next steps, in order

1. Re-run `cargo test -j 16 --workspace --no-fail-fast` and the `scripts/ci` suite; the last commit
   was not re-verified end to end.
2. **Borrowed views.** Replace the borrowed-only `device_read` with a resolver that materializes a
   view through the existing `to_contiguous_view` / `materialize_view` and keeps the owner alive for
   the launch (e.g. `enum Resolved<'a> { Borrowed(&'a Tensor), Owned(Tensor) }` with `Deref`), then
   thread it through the ~40 call sites in `cuda/structural.rs`. That closes the seven blas1 tests
   and the capability-table lie in one move.
3. **`launch_planned`** should take `&mut self` and reuse `self.dispatcher(...)` instead of building
   `KernelModules::for_entries` per launch.
4. Aux ABI dtype validation: the `linalg_*` helpers take `&impl DeviceOperand` and do not check that
   an I64 destination really is I64 (`initialize_permutation`, pivots, status, the I32 violation
   word). A safe function can currently corrupt memory through a wrong-dtype operand.
5. Optional hardening: the extension-cache borrowing during compiled runs, and `publish`'s
   allocation-domain check.
6. Then: remove the `#[doc(hidden)]` migration marks, regenerate
   `scripts/ci/cuda_test_partition.tsv` if test names changed again, run the independent review once
   more over the final diff, squash to a reviewable state, open the PR, and spend the RunPod GPU gate
   once on the finished commit.

## Resume commands

```bash
cd ~/tensor4all/tenferro-rs
git status --short                      # expect only untracked HANDOFF*/samples/ext artefacts
cargo test -j 16 --workspace --no-fail-fast
cargo test -j 16 -p tenferro-gpu --features cuda --test seam_capability --test seam_dtype
cargo test -j 16 -p tenferro-linalg --features cuda --test integration -- --ignored   # 12 min
cargo test -j 16 -p tenferro-gpu --features cuda --test blas1_cuda -- --ignored        # 7 red today
```

Local GPU: NVIDIA A100 80GB PCIe, driver 580.126.09, CUDA 12.6, cuTENSOR 2.5.0. The kernel catalogue
is the git dependency pinned in `crates/tenferro-gpu/Cargo.toml`; a local checkout with the ABI
document is at `~/tensor4all/t4a-cuda-kernels/docs/abi/c-kernel-abi.md`.

## Decisions that need shinaoka before the PR

1. **The removed capability.** The public CubeCL session (`cuda::cubecl::Session` behind
   `cubecl-interop`) was the documented way for downstream code to compile and launch its own CUDA
   kernel, with `samples/cubecl-kernel`, a guide and one test. The new raw session exposes the
   stream, device bytes, tensor borrows, byte copies and vendor resources, but no NVRTC/launch
   surface. Ship the removal as a documented breaking change, or build the launch surface first?
2. **Borrowed views.** Is closing the view regression part of PR 2 (as I would do), or is it
   acceptable to land the cutover with the regression recorded and fix it in a follow-up PR?
3. `tenferro-fft`'s CUDA path and `tenferro-linalg`'s providers now both depend on
   `tenferro-gpu/cuda` only; nothing else in the workspace enables the old feature name, so no
   downstream pin needs updating inside this repo. Confirm no external consumer of
   `tenferro-gpu`'s `cuda-next`/`webgpu` feature names exists.
