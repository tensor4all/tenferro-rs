# ATen-Aligned Low-Level Substrate Inventory Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn the PyTorch/ATen low-level helpers actually used by the next CUDA linalg target set into a concrete Tier A/Tier B substrate backlog for tenferro.

**Architecture:** Start from real PyTorch call paths for `svd`, `svdvals`, `lu`, `qr`, `cholesky`, `solve`, `solve_triangular`, `det`, `slogdet`, `pinv`, `norm`, and `matrix_exp`. Compute the Tier A closure of substrate those ops actually consume, then record adjacent Tier B helpers that are clearly reusable and already first-class in ATen. Map every substrate item to the correct tenferro layer and call out what is already present on the current CUDA-linalg tranche versus still missing.

**Tech Stack:** PyTorch/ATen C++, cuBLAS/cuSOLVER, Rust tenferro workspace, `tenferro-device` Layer 0 CUDA runtime, `tenferro-prims` family execution, `tenferro-linalg-prims` backend contracts.

---

## Scope And Assumptions

- Tier A target ops:
  - `svd`
  - `svdvals`
  - `lu`
  - `lu_factor_ex`
  - `qr`
  - `cholesky`
  - `cholesky_ex`
  - `solve`
  - `solve_ex`
  - `solve_triangular`
  - `det`
  - `slogdet`
  - `pinv`
  - `norm`
  - `matrix_exp`
- Tier B records adjacent helpers that are not strictly required by the Tier A closure but are already ATen-proven and likely to be needed in the next neighboring tranche.
- Crosswalk status is evaluated against the current CUDA-linalg tranche represented by PR `#548` / branch `feat/torch-style-shape-packing`, not just `origin/main` before that PR lands.
- Normal-path GPU payload fallback remains forbidden. Small `info` host sync is acceptable when it is the only decision signal.
- tenferro's strict column-major design means some ATen helpers map by **capability** rather than by exact implementation shape. `cloneBatchedColumnMajor` is still relevant, but in tenferro the analogue is "preserve F-ready invariants and broadcast-safe working-copy materialization", not "adopt row-major semantics".

## PyTorch Call-Path Summary

### Factorization / Solve Family

- `linalg_cholesky_ex_out` in `aten/src/ATen/native/BatchLinearAlgebra.cpp`
  - copies or aliases input into output
  - dispatches `cholesky_stub`
  - structurally cleans output with `triu_()` / `tril_()`
  - validates via `_linalg_check_errors(info, ...)`
- `_linalg_solve_ex_out` in `BatchLinearAlgebra.cpp`
  - calls `linalg_lu_factor_ex_out`
  - uses `linalg_solve_is_vector_rhs`
  - solves through `linalg_lu_solve_out`
  - checks only `info`
- `linalg_lu_factor_ex_out` in `BatchLinearAlgebra.cpp`
  - copies `A` into `LU`
  - dispatches `lu_factor_stub`
  - uses per-batch `info`
- `linalg_qr_out` in `BatchLinearAlgebra.cpp`
  - allocates `tau`
  - prepares a Fortran-contiguous `QR` working copy, often via `cloneBatchedColumnMajor`
  - dispatches `geqrf_stub`
  - materializes `R` via `triu_out` / `triu_`
  - materializes `Q` via `orgqr_stub`
- `linalg_solve_triangular_out` in `BatchLinearAlgebra.cpp`
  - validates with `checkInputsSolver`
  - prepares layout via `borrow_else_clone`, `copyBatchedColumnMajor`, transposes, and conjugation resolution
  - dispatches `triangular_solve_stub`

### Determinant / Spectral Family

- `_linalg_det_out` in `aten/src/ATen/native/LinearAlgebra.cpp`
  - calls `linalg_lu_factor_ex_out`
  - computes `lu_det_P(pivots) * prod(LU.diagonal(...))`
- `_linalg_slogdet_out` in `LinearAlgebra.cpp`
  - calls `linalg_lu_factor_ex_out`
  - computes `sign = diag_U.sgn().prod(-1) * lu_det_P(pivots)`
  - computes `logabsdet = sum(diag_U.abs().log_(), -1)`
- `linalg_pinv` in `LinearAlgebra.cpp`
  - non-Hermitian path uses `svd`
  - forms `tol = max(atol, rtol * max_val)`
  - thresholding is `where(S > tol, reciprocal(S), 0)`
  - reconstructs with `matmul(V * S_pseudoinv.unsqueeze(-2), U.mH())`
- `matrix_rank_impl` in `LinearAlgebra.cpp`
  - uses `linalg_svdvals` or `linalg_eigvalsh`
  - computes `tol = max(atol, rtol * max_S)`
  - rank is `sum(S > tol, -1)`
- `linalg_matrix_norm` / `linalg_cond` in `LinearAlgebra.cpp`
  - `ord = ±2` and `"nuc"` rely on `linalg_svdvals`
  - other norms rely on `abs`, `sum`, `amax`, `max`
- `operator_1_norm` and `linalg_matrix_exp` in `LinearAlgebra.cpp`
  - `operator_1_norm(tensor)` is `tensor.abs().sum(-2).max(-1)`
  - `matrix_exp` then uses that norm inside the matrix-exp algorithm

### SVD Family

- `_linalg_svd_out` in `BatchLinearAlgebra.cpp`
  - allocates per-batch `info`
  - dispatches `svd_stub`
  - checks convergence/errors via `_linalg_check_errors`
- `linalg_svdvals` in `BatchLinearAlgebra.cpp`
  - delegates to `_linalg_svd_out(..., compute_uv=false)`

### CUDA-Specific Lowering

- `LinearAlgebraUtils.h`
  - `cloneBatchedColumnMajor`
  - `copyBatchedColumnMajor`
  - `batch_iterator_with_broadcasting`
  - `batchCount`
  - `matrixStride`
  - `borrow_else_clone`
  - `linalg_solve_is_vector_rhs`
  - `to_transpose_type`
  - `BroadcastLinearIndices`
- `aten/src/ATen/native/TransposeType.h`
  - `TransposeType`
- `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp`
  - SVD drivers `gesvd`, `gesvdj`, `gesvdjBatched`, `gesvdaStridedBatched`
  - wide-matrix SVD handled by transpose + U/V swap
  - non-convergence is detected by copying only `info` to CPU
  - Cholesky uses both looped `potrf` and `potrfBatched`
  - Cholesky solve uses `potrs` and `potrsBatched`
  - LU solve uses `BroadcastLinearIndices` for broadcasted factors
- `aten/src/ATen/native/cuda/linalg/CUDASolver.cpp`
  - typed wrappers for `getrf`, `gesvd`, `gesvdj`, and workspace sizing
- `aten/src/ATen/native/cuda/linalg/CusolverDnHandlePool.cpp` and `CudssHandlePool.cpp`
  - thread/device handle pooling via `DeviceThreadHandlePool`
- `aten/src/ATen/native/cuda/cuBlasCommonArgs.h`
  - `resolve_conj_if_indicated`
  - `prepare_matrix_for_cublas`
  - `MaybeOwned<Tensor>` borrow-or-clone discipline
- `aten/src/ATen/native/ComplexHelper.h` and `UnaryOps.cpp`
  - `view_as_real`
  - `view_as_complex`
  - `real()`
  - `imag()`

## Tier A: Must-Have Substrate Closure

These items are in the real closure of the bounded target-op set.

| Tier A substrate | Why it is in the closure | Main ATen evidence | tenferro target layer | Current tenferro analogue / status on PR #548 branch |
| --- | --- | --- | --- | --- |
| Batched column-major working copy helpers | `svd`, `qr`, `solve`, `solve_triangular`, `cholesky`, `lu` all normalize inputs into F-contiguous working copies | `cloneBatchedColumnMajor`, `copyBatchedColumnMajor`, `batchCount`, `matrixStride` in `LinearAlgebraUtils.h` | `tenferro-tensor` + `tenferro-linalg-prims` | `partial`: analogues exist as `Tensor::contiguous`, `ensure_col_major`, and linalg batch helpers, but there is no full `copyBatchedColumnMajor` equivalent with broadcast-aware batch/nrows growth |
| Broadcast-aware batch iteration / linear-index helpers | broadcasted `solve` / `lu_solve` style paths need reusable iteration over materialized-broadcast batches without per-op host logic | `batch_iterator_with_broadcasting`, `BroadcastLinearIndices`, `lu_solve_looped_cusolver` | `tenferro-linalg-prims` + `tenferro-tensor` | `absent`: tenferro has batch-count and RHS-shape helpers, but no reusable broadcast-aware batch iterator or linear-index helper |
| Alias-safe borrow-or-clone layout prep | triangular solve and QR use borrow-or-clone instead of blindly copying every input | `borrow_else_clone` in `LinearAlgebraUtils.h`; `prepare_matrix_for_cublas` in `cuBlasCommonArgs.h` | `tenferro-tensor` + `tenferro-prims` + `tenferro-linalg-prims` | `partial`: current code has ad hoc `ensure_col_major` and op-local prep, but no shared MaybeOwned-style substrate |
| Transpose / conjugation dispatch abstraction | nearly every BLAS/cuSOLVER-facing linalg call needs a reusable encoding of no-transpose vs transpose vs conj-transpose | `TransposeType`, `to_transpose_type`, `to_cublas`, triangular solve and LU solve call paths | `tenferro-linalg-prims` + `tenferro-prims` | `absent`: tenferro has op-local booleans and conj resolution, but no reusable `TransposeType`-class substrate shared across CUDA linalg wrappers |
| Vector-RHS normalization and batched RHS shape helpers | `solve` / `solve_ex` need shared vector-vs-matrix RHS semantics before kernel dispatch | `linalg_solve_is_vector_rhs` in `LinearAlgebraUtils.h` and `_linalg_solve_ex_out` | `tenferro-linalg` + `tenferro-linalg-prims` | `present`: `validate_solve_rhs_shape` and matching solve contracts exist |
| Per-batch `info` contracts and info-only control flow | `solve_ex`, `lu_factor_ex`, `cholesky_ex`, and CUDA SVD driver fallback are all driven by `info` tensors | `_linalg_check_errors`, `_check_gesvdj_convergence`, `linalg_*_ex_out` | `tenferro-linalg-prims` | `mostly present, still partial`: `SolveTensorExResult`, `LuTensorExResult`, and `CholeskyTensorExResult` exist and current CUDA paths populate them; remaining gaps are shared QR/SVD status propagation and fallback reuse |
| cuSOLVER / cuBLAS handle-resource lifecycle | the wrapper layer depends on thread/device-safe handle creation, destruction, and stream association before any kernel wrapper can run | `CusolverDnHandlePool.cpp`, `CudssHandlePool.cpp`, `CublasHandlePool.cpp`, `DeviceThreadHandlePool` | `tenferro-linalg-prims` | `mostly present`: `CudaLinalgRuntime` owns handle load/create/destroy and stream binding, but the inventory should still compare lifecycle policy against ATen's pooled model |
| Triangular structural cleanup | `cholesky`, `qr`, and triangular result shaping depend on device-resident `tril` / `triu` instead of host cleanup | `triu_out`, `triu_`, `tril_out`, `tril_` in `BatchLinearAlgebra.cpp` | `tenferro-tensor` | `present`: CUDA `tril` / `triu` exist |
| Fixed-shape trailing zero-fill by rank/count | tenferro `svd(..., cutoff)` keeps shapes fixed and zero-fills tails; this is a first-class structural need for current semantics | tenferro-specific requirement, analogous to ATen fixed-shape postprocessing patterns | `tenferro-tensor` | `present`: `zero_trailing_by_counts` exists |
| Diagonal extraction plus multiplicative/additive reductions | `det` and `slogdet` are pure compositions of LU factors, diagonal views, `prod`, `sum`, `abs`, `log`, and sign/parity logic | `_linalg_det_out`, `_linalg_slogdet_out` in `LinearAlgebra.cpp` | `tenferro-tensor` + `tenferro-prims` + `tenferro-linalg` | `partial`: diagonal, `prod`, `log` exist; signum-style scalar composition is still thinner than ATen |
| Same-dtype scalar family for pointwise and reductions | `det`, `slogdet`, `norm`, `cond`, and QR/SVD postprocessing all depend on reusable same-dtype `mul/div/reciprocal/sum/max/prod` | `mul_out`, `sum_out`, `prod`, `amax`, `reciprocal` across `LinearAlgebra.cpp` | `tenferro-prims` | `partial`: many real/CUDA ops exist, but coverage is still uneven across dtype/backend pairs |
| Ordered comparisons as reusable tensor ops | `pinv`, `matrix_rank`, SVD cutoff, and tolerance gating need `>` / `>=` on device | `S > tol` in `pinv` and `matrix_rank_impl` | `tenferro-prims` | `partial`: `Greater` / `GreaterEqual` numeric masks exist, but the rest of thresholding pipeline is incomplete |
| Generic select / mask application | `pinv` and rank-style thresholding use `where(...)`; without it composite code reverts to host-side loops or ad hoc zeroing | `where(S > tol, reciprocal(S), zeros)` in `linalg_pinv` | `tenferro-prims` | `absent`: no first-class `where` / mask-select / masked-fill family substrate |
| Cross-dtype complex->real unary | `matrix_exp` 1-norm, complex norms, and spectral thresholding need `abs_real`-style outputs that change dtype | `operator_1_norm(tensor.abs().sum(-2).max(-1))`, complex `angle`, `real`, `imag` conventions | `tenferro-device` + `tenferro-prims` | `partial surface, absent substrate`: `ScalarUnaryOp::{Abs, Real, Imag}` exist, but `TensorScalarPrims` executes same-dtype `Tensor<Alg::Scalar> -> Tensor<Alg::Scalar>` contracts and the real CUDA path is still same-dtype only |
| Cross-dtype real reductions over complex-derived tensors | once `abs_real` exists, `sum/max/amax` on the resulting real tensors must stay on device without host extraction | `operator_1_norm`, `matrix_rank_impl`, `cond`, `pinv` | `tenferro-prims` | `absent`: no reusable cross-dtype pipeline is surfaced today |
| `svdvals` as a first-class singular-values-only contract | `norm`, `cond`, `matrix_rank`, `pinv` all prefer singular values without paying for `U/Vh` | `_linalg_svd_out(..., compute_uv=false)` and `linalg_svdvals` | `tenferro-linalg-prims` | `present`: contract exists; full CUDA coverage is still in progress |
| cuSOLVER wrapper coverage for core decompositions | target ops bottom out in `getrf/getrs/potrf/potrs/geqrf/orgqr/ormqr/gesvd` and optionally `gesvdj` | `CUDASolver.cpp`, `BatchLinearAlgebraLib.cpp` | `tenferro-linalg-prims` | `partial`: `getrf/getrs` solve path exists; LU/Cholesky/QR/SVD wrapper coverage is not complete |
| Wide-matrix transpose/swap logic for SVD | exact SVD on CUDA needs an explicit `m < n` strategy, not just the tall case | `svd_cusolver_gesvd`, `svd_cusolver_gesvdaStridedBatched` | `tenferro-linalg-prims` | `absent`: not yet wired as reusable CUDA SVD substrate |
| Matrix-exp low-level norm substrate | `matrix_exp` itself does not need a new Layer 0 kernel family beyond GEMM, but it does require `operator_1_norm = abs_real + sum + max` | `operator_1_norm` in `LinearAlgebra.cpp` | `tenferro-prims` + `tenferro-linalg` | `absent`: blocked by the cross-dtype items above, not by GEMM itself |

## Tier B: Adjacent, ATen-Proven, Likely Next

These helpers are near the traced paths and are likely to matter in the next tranche, but they are not all required to unblock the Tier A target set immediately.

| Tier B substrate | Why it is worth carrying now | Main ATen evidence | tenferro target layer |
| --- | --- | --- | --- |
| `view_as_real` / `view_as_complex` | cleanest substrate for complex-as-real views, enables `real/imag` without forced copies and reduces future dtype-bridge churn | `ComplexHelper.h` | `tenferro-tensor` |
| `real()` / `imag()` first-class tensor ops | likely needed right after Tier A for eig/eigh cleanup, validation paths, and richer complex analytics | `UnaryOps.cpp` | `tenferro-prims` + `tenferro-tensor` |
| Richer `resolve_conj_if_indicated` / `resolve_neg` semantics | ATen treats unresolved conjugation/negation as layout flags, not ad hoc copies; useful for matmul/eigen next | `cuBlasCommonArgs.h`, `UnaryOps.cpp` | `tenferro-tensor` + `tenferro-prims` |
| Generic mask-select / masked-fill / masked-scatter family | Tier A only strictly needs `where`, but broader masked ops are likely to become necessary immediately after `pinv` and rank cleanup | ATen indexing/copy conventions; `where` usage in linalg and masked ops nearby | `tenferro-prims` |
| Generic `copy_` / temp-contiguous backfill policy | ATen has a unified copy policy that handles same-device copy, temp-contiguous fallback, and flag preservation | `aten/src/ATen/native/cuda/Copy.cu` | `tenferro-device` + `tenferro-tensor` |
| Broadcast-linear-index helpers beyond Tier A closure | after the Tier A iterator/index substrate exists, broader reuse for LU solve variants and neighboring batched kernels remains valuable | `BroadcastLinearIndices` and `lu_solve_looped_cusolver` | `tenferro-linalg-prims` |
| Device-pointer-array helpers for batched cuSOLVER | small/batched kernels repeatedly need `Tensor<ptr>` materialization | `get_device_pointers`, `potrfBatched`, `potrsBatched` | `tenferro-linalg-prims` |
| Small-matrix specialized driver policy | not required for correctness, but clearly useful for later performance parity on SVD/Cholesky | `gesvdjBatched`, `potrfBatched`, default-driver heuristics | `tenferro-linalg-prims` |
| MaybeOwned-style public internal conventions | not a kernel substrate by itself, but very effective at keeping layout prep and aliasing disciplined across ops | `MaybeOwned<Tensor>` use in `LinearAlgebraUtils.h` and `cuBlasCommonArgs.h` | `tenferro-tensor` + `tenferro-prims` + `tenferro-linalg-prims` |

## tenferro Gap Matrix

### Already Present Or Mostly Present On The Current CUDA-Linalg Tranche

- `tenferro-device`
  - shared CUDA runtime
  - generic strided copy kernel
  - CUDA `tril` / `triu`
  - `zero_trailing_by_counts`
  - real unary `Log`
  - real reduction `Prod`
- `tenferro-tensor`
  - GPU-capable `contiguous(ColumnMajor)`
  - triangular tensor helpers
  - keep-count trailing zero-fill
- `tenferro-prims`
  - same-dtype scalar family surface
  - ordered-real comparison masks (`Greater`, `GreaterEqual`)
  - minimal CUDA analytic substrate for some real unary/binary ops
- `tenferro-linalg-prims`
  - backend ownership of tensor linalg contracts
  - `svdvals`
  - `solve_ex`
  - `lu_factor_ex`
  - `cholesky_ex`
  - CUDA runtime loader and `solve` wrapper
  - CUDA handle/resource lifecycle via `CudaLinalgRuntime`
- `tenferro-linalg`
  - tensor-native `svd(..., cutoff)` fixed-shape semantics
  - tensor-native `det`, `slogdet`, `matrix_power`

### Still Missing Or Only Partial

- `tenferro-tensor`
  - no `view_as_real` / `view_as_complex`
  - no general alias-safe borrow-or-clone substrate comparable to `MaybeOwned<Tensor>`
- `tenferro-prims`
  - no first-class cross-dtype `complex -> real` unary substrate
  - existing `Abs` / `Real` / `Imag` vocabulary is same-dtype, so it is only a partial analogue
  - no first-class cross-dtype real reductions over complex-derived tensors
  - no `where` / mask-select / masked-fill family surface
  - no shared `real()` / `imag()` tensor family surface
- `tenferro-linalg-prims`
  - no reusable broadcast-aware batch iterator / linear-index substrate
  - no reusable `TransposeType`-style dispatch abstraction
  - incomplete cuSOLVER wrapper coverage for LU, Cholesky, QR, SVD drivers
  - no reusable wide-matrix SVD transpose/swap path
  - no batched device-pointer-array helper
  - no small-matrix specialized driver policy
- `tenferro-linalg`
  - `matrix_exp` still blocked by missing cross-dtype norm substrate
  - `pinv`, rank-style thresholding, and some norm paths still want `where`-class functionality rather than ad hoc structural cleanup

## Recommended Execution Order

### Task 1: Cross-dtype complex->real unary substrate

**Why first:** It is the single clearest blocker shared by `matrix_exp`, complex norms, and future spectral cleanup.

**Files:**
- Modify: `tenferro-device/src/cuda/runtime.rs`
- Modify: `tenferro-prims/src/families/scalar.rs`
- Modify: `tenferro-prims/src/cuda/scalar.rs`
- Modify: `tenferro-linalg/src/prims_bridge.rs`
- Test: `tenferro-prims/src/tests/*`

**Deliverable:**
- `abs_real`-class tensor op that returns `Tensor<T::Real>` from complex input without host bounce

### Task 2: Cross-dtype real reductions over complex-derived tensors

**Why second:** `matrix_exp` and norm-style code need `sum/max/amax` after `abs_real`, not just the unary.

**Files:**
- Modify: `tenferro-prims/src/families/scalar.rs`
- Modify: `tenferro-prims/src/cuda/scalar.rs`
- Modify: `tenferro-linalg/src/prims_bridge.rs`
- Test: `tenferro-prims/src/tests/*`

**Deliverable:**
- `sum/max` over real outputs produced from complex inputs, still fully device-resident

### Task 3: Generic select / mask application

**Why third:** `pinv`, `matrix_rank`, and tolerance-based spectral logic all want `where`, not op-specific zero-fill hacks.

**Files:**
- Modify: `tenferro-prims/src/families/scalar.rs`
- Modify: `tenferro-prims/src/cuda/scalar.rs`
- Modify: `tenferro-linalg/src/prims_bridge.rs`
- Test: `tenferro-prims/src/tests/*`

**Deliverable:**
- `where(mask, on_true, on_false)` or equivalent reusable mask-select substrate

### Task 4: Shared alias-safe layout prep

**Why fourth:** this removes repeated op-local layout plumbing and matches the ATen discipline around `MaybeOwned<Tensor>`, `batch_iterator_with_broadcasting`, and `TransposeType`.

**Files:**
- Modify: `tenferro-tensor/src/*`
- Modify: `tenferro-prims/src/*`
- Modify: `tenferro-linalg-prims/src/backend/*`

**Deliverable:**
- shared borrow-or-clone / resolve-conj / layout-prep helpers, plus reusable broadcast-aware batch iteration and transpose-dispatch helpers instead of per-op ad hoc code

### Task 5: Complete cuSOLVER wrapper coverage

**Why fifth:** after the reusable tensor/prims substrate is in place, the next unblocker is missing backend kernel coverage. This task also verifies that current handle/resource management is sufficient and does not need an ATen-style pool split yet.

**Files:**
- Modify: `tenferro-linalg-prims/src/backend/cuda/wrappers.rs`
- Modify: `tenferro-linalg-prims/src/backend/cuda/runtime.rs`
- Modify: `tenferro-linalg-prims/src/backend/cuda/*.rs`

**Deliverable:**
- LU, Cholesky, QR, SVD exact wrappers with per-batch `info`, plus a documented decision on whether current `CudaLinalgRuntime` handle lifecycle is enough or should evolve toward pooled handles

### Task 6: Tier B views and copy policy

**Why sixth:** useful immediately after Tier A, but not necessary to start unblocking the currently targeted ops.

**Files:**
- Modify: `tenferro-tensor/src/*`
- Modify: `tenferro-device/src/cuda/runtime.rs`

**Deliverable:**
- `view_as_real`, `view_as_complex`, `real`, `imag`, and better generic copy/backfill conventions

## Notes For Issue Breakdown

- If this inventory is converted into issues, split Tier A by reusable substrate family, not by top-level op.
- The highest-value first issue is:
  - `Implement cross-dtype complex->real unary/reduction substrate in tenferro-prims`
- The second highest-value issue is:
  - `Add generic mask-select/where substrate for tensor-native spectral thresholding`
- Batched driver heuristics such as `gesvdjBatched` belong in performance-followup issues, not the first correctness tranche.
