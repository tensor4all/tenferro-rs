# tenferro-linalg-prims CUDA Consolidation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate backend-owned CUDA linalg into `tenferro-linalg-prims` and make `tenferro-linalg` stay a CPU/GPU-generic public/composite layer that either routes through tensor-level backend contracts or fails early through truthful capability checks.

**Architecture:** Move the tensor-level linalg backend surface out of `tenferro-linalg/src/backend/*` and into `tenferro-linalg-prims`, alongside context bindings and backend markers for CPU/CUDA/ROCm. `tenferro-linalg` should only consume `TensorLinalgPrims`, `TensorLinalgContextFor`, and result types from `tenferro-linalg-prims`, keeping backend ownership out of the public/composite crate.

**Architecture:** CUDA support should be staged. Phase 1 implements the backend kernel basis in `tenferro-linalg-prims` and advertises only the operations that are genuinely tensor-generic end-to-end. Composite and AD paths in `tenferro-linalg` that still materialize CPU slices must either be rewritten to tensor-level helpers or kept capability-gated so GPU contexts never wander into CPU-only code paths.

**Architecture:** Normal library execution must not insert ad hoc GPU→CPU transfers as a fallback. Device-resident CUDA paths either stay on GPU all the way through or fail early via capability checks. The only allowed GPU→CPU transfers in this plan are explicit user-requested transfers and test-only result comparisons.

**Tech Stack:** Rust, `tenferro-prims` contexts and CUDA runtime, `tenferro-tensor` GPU buffers, `cudarc` dynamic loading, runtime-loaded cuSOLVER/cuBLAS FFI, faer/LAPACK feature forwarding, cargo test/doc/fmt.

**Current State:** The protocol split is only partial today. `TensorLinalgPrims` and `LinalgCapabilityOp` already live in `tenferro-linalg-prims`, but `TensorLinalgContextFor`, the backend marker types, the tensor-level CPU implementation, and the internal slice-level `LinalgBackend`/provider modules still live in `tenferro-linalg`. This plan finishes that split instead of redoing it from scratch.

**Public API guardrail:** The public `tenferro_linalg::traced_tensor::svd(&mut ctx, &tensor, options)` signature should stay unchanged while `tenferro-linalg` becomes CPU/GPU-generic. Backend ownership moves, but the user-facing API does not. The backend/kernel contract remains `TensorLinalgPrims::thin_svd(ctx, a) -> SvdTensorResult<T>`.

**SVD layering guardrail:** `SvdOptions` stays in `tenferro-linalg` as public/composite-layer policy, not in `tenferro-linalg-prims`. The `options == None` path is already close to backend-generic because it can return the backend-produced tensors directly. The `options != None` path is not generic today because it reads CPU slices to truncate/repack outputs. CUDA support must not paper over that with GPU→CPU fallback.

Split the CUDA truncation story explicitly:

- `max_rank` only: supportable without host transfer by replacing the current repack logic with metadata-only `Tensor::narrow(...)` views on `u`, `s`, and `vt`
- `cutoff`: not supportable with the current primitive vocabulary alone because predicate/select-style tensor ops are intentionally absent today; supporting it on CUDA requires a dedicated device-resident linalg-side kernel (or equivalent tensor-native masking path) that computes per-batch active rank and zeroes the trailing regions of `u`, `s`, and `vt`

Until that device-resident `cutoff` path exists, `svd(..., Some(options))` on CUDA should mean:

- supported when `options.cutoff == None`
- unsupported when `options.cutoff.is_some()`

### Recommended execution phases

Run this plan as a staged rollout, not as one continuous CUDA feature branch. The recommended dependency order is:

- **Phase 0: Ownership cleanup only**
  - Land Tasks 1-2 first.
  - Goal: finish moving backend ownership into `tenferro-linalg-prims` without changing real CUDA linalg behavior.
  - Merge value on its own: `tenferro-linalg` stops being the backend owner, which makes later GPU work mechanically simpler.

- **Phase 1: Torch-aligned linalg substrate**
  - Add the missing tensor/linalg helpers that PyTorch relies on before wiring real CUDA SVD:
    - batched column-major working-copy helpers analogous to `cloneBatchedColumnMajor` / `copyBatchedColumnMajor`
    - reusable `batch_count` / `matrix_stride` / layout-normalization helpers for batched matrices
    - device-resident working-copy plus copy-back helpers exposed for linalg, not only internal CUDA pointwise paths
    - explicit result-status plumbing for CUDA solvers (`info` or `_ex`-style status surface) so non-convergence and capability gating do not require ad hoc host fallback
    - a singular-values-only path (`svdvals` or equivalent backend contract) so composite ops like spectral/nuclear norms do not depend on CPU slice extraction
  - This phase is the prerequisite for truthful GPU-generic composite behavior.

- **Phase 2: Minimal exact CUDA SVD kernel**
  - Implement only the exact reduced/thin SVD kernel path in `tenferro-linalg-prims`.
  - Prefer `cuSOLVER gesvd` first because it directly matches `vt` output and is easier to reason about for the initial exact implementation.
  - Keep this phase intentionally narrow:
    - `svd(..., None)` only
    - exact path only
    - no truncation policy
    - no `gesvdj` heuristic yet

- **Phase 3: Composite genericization**
  - Remove or capability-gate the `tenferro-linalg` code paths that still call `extract_slice`, `tensor_from_data`, or other host-only helpers after backend execution.
  - First targets should be:
    - `svd(..., None)`
    - `NormKind::Spectral`
    - `NormKind::Nuclear`
  - The rule in this phase is simple: if the composite path cannot stay tensor-native, it must remain unsupported on CUDA.

- **Phase 4: Truncation policy**
  - Implement `max_rank` first via metadata-only `narrow` views.
  - Implement `cutoff` only after a dedicated device-resident postprocess exists.
  - Do not collapse these into one task: `max_rank` is a view problem, `cutoff` is a value-dependent kernel problem.

- **Phase 5: Driver heuristics and optimization**
  - After the exact path is stable, add optional `gesvdj` or other driver heuristics for small/well-conditioned cases.
  - This is an optimization phase, not a correctness prerequisite.

The most important planning consequence is that real CUDA enablement should not start at the public `svd(options)` layer. It should start at the substrate and backend-kernel layers and move upward.

### Locked implementation decisions before coding

Because this work spans ownership refactors, tensor layout helpers, cuSOLVER wrappers, and public/composite behavior, the implementation details should be fixed at the planning stage instead of improvised task-by-task. Use the following decisions as the implementation contract.

#### 1. Backend utility layer shape

Add a reusable linalg utility layer in `tenferro-linalg-prims` for batched matrix helpers. This layer should own the tenferro equivalents of PyTorch's `LinearAlgebraUtils` helpers:

- `batch_count(dims_or_tensor)`
- `matrix_stride(dims_or_tensor)`
- `clone_batched_column_major`
- `copy_batched_column_major`
- output-preparation helpers for writable contiguous working buffers and copy-back

These helpers should operate on `Tensor<T>` and preserve:

- logical memory space
- device placement
- batch dimensions
- column-major `(m, n, batch...)` semantics

For CUDA-backed tensors, these helpers must stay device-resident. They may internally use lower-level semiring-family copy/contiguous plans, but they must not extract CPU slices.

#### 2. Tensor/backend contract changes

Do not expose a public `compute_uv` flag from `tenferro-linalg`; keep the public API close to PyTorch's `torch.linalg.svd` surface rather than the lower-level `_linalg_svd` surface.

At the backend contract level:

- keep `thin_svd(ctx, a) -> SvdTensorResult<T>` as the reduced-SVD tensor contract
- add a singular-values-only contract:

```rust
fn svdvals(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<Tensor<T::Real>>;
```

This is needed so composite ops such as spectral norm and nuclear norm can become GPU-generic without computing or materializing `u` and `vt` unnecessarily.

Placement of `svdvals` is fixed as follows:

- backend trait declaration: `tenferro-linalg-prims/src/lib.rs`
- backend implementations: `tenferro-linalg-prims/src/backend/cpu.rs` and `tenferro-linalg-prims/src/backend/cuda/**`
- public wrapper: `tenferro-linalg/src/primal/decompositions.rs`
- public re-export/docs surface: `tenferro-linalg/src/lib.rs`

Do not put `svdvals` in `tenferro-tensor`. It is a linalg operation, not a tensor-core primitive.

Do not add a public `_ex` API at this stage. If CUDA wrappers need `info`/status plumbing internally, keep that internal to `tenferro-linalg-prims::backend::cuda`.

#### 3. Initial CUDA SVD driver policy

The first exact CUDA SVD implementation should use `cuSOLVER gesvd`, not `gesvdj`.

Reasons:

- it directly matches reduced/thin SVD through `jobu='S'`, `jobvt='S'`
- it returns `V^T`/`V^H`-oriented output that aligns better with tenferro's `vt`
- it gives a simpler exact baseline before any driver heuristics are introduced

Implementation details:

- support `m >= n` directly
- support `m < n` by a device-resident transpose/conjugate-transpose route and swap-back of output roles, following the same broad strategy PyTorch uses
- do not introduce host fallback for wide matrices

`gesvdj` belongs to the later optimization phase only. `gesvda` is approximate and must not be the default exact backend path.

When the later optimization phase introduces alternative drivers, follow the PyTorch-style division of labor:

- `gesvd`: exact QR-based baseline, per-batch loop, default-safe path
- `gesvdj`: faster Jacobi path for well-conditioned cases
- `gesvdjBatched`: small-matrix specialization for approximately `m, n <= 32`
- `gesvda`: approximate strided-batched path for tall/wide large-batch cases only

However, because tenferro explicitly avoids ad hoc production GPU→CPU fallback transfers, do not copy PyTorch's default-driver policy blindly. In particular:

- do not make `gesvdj` the default driver in the first CUDA rollout
- do not require a host-side convergence check just to preserve normal exact behavior
- keep `gesvd` as the default exact path until the cost/behavior of any `info` synchronization is explicitly accepted

If `gesvdj` is added later, the non-convergence strategy should be:

1. store per-batch `info` in an internal CUDA status tensor
2. if the selected `gesvdj` path reports non-convergence, retry that batch or call site with `gesvd`
3. keep this retry policy internal to `tenferro-linalg-prims`

If this requires a small host-visible status synchronization, document it as a control-path synchronization on `info`, not as a tensor-data fallback. The automatic fallback decision must inspect `info` only.

The locked rule is:

- allowed to synchronize or copy: the minimal `info` status needed for retry control flow
- forbidden to synchronize or copy for retry control flow: input tensors, `u`, `s`, `vt`, or any other large result payload

If a fallback retry is needed, it must rebuild or reuse a fresh device-resident working copy and rerun the failing batch on GPU. Do not route fallback decisions through host-side inspection of tensor data.

#### 4. Composite-layer rewrite order

Rewrite `tenferro-linalg` from the bottom up in this exact order:

1. `svd(..., None)`
2. `svdvals`-based spectral and nuclear norms
3. `svd(..., Some(max_rank))`
4. `svd(..., Some(cutoff))`

The reason for this order is that steps 1-3 can be made GPU-generic with reduced SVD plus views/reductions, while step 4 requires value-dependent postprocessing.

Any composite or AD path that still depends on:

- `extract_slice`
- `buffer().as_slice()`
- `tensor_from_data(Vec<_>, ...)`

must remain CUDA-disabled until rewritten.

#### 5. Locked truncation semantics

Preserve current `SvdOptions` semantics unless the user explicitly asks for a behavioral change:

- `max_rank`: shrink the returned shape to `max_k`
- `cutoff`: preserve the post-`max_rank` shape but zero-fill trailing singular directions instead of producing ragged per-batch shapes

This means CUDA `cutoff` support should be implemented as a fixed-shape device-resident postprocess, not as a shape-changing per-batch truncation scheme.

#### 6. CUDA cutoff kernel design

When `cutoff` is implemented, do it as a dedicated linalg-side postprocess in `tenferro-linalg-prims`, not by routing through generic host slices and not by waiting for a general boolean tensor substrate to appear elsewhere.

The expected design is:

1. a device-resident pass over each batch's descending singular values to compute `actual_k`
2. a device-resident zero-fill over trailing regions of `s`
3. a device-resident zero-fill over trailing columns/rows of `u` and `vt`

This preserves the existing fixed-shape semantics and avoids ad hoc GPU→CPU transfer.

#### 7. Norm implementation policy

`NormKind::Spectral` and `NormKind::Nuclear` should move to the new `svdvals` backend contract rather than reusing `svd(..., None)` followed by host-side extraction.

That policy is more PyTorch-like:

- `svd` remains the factorization API
- `svdvals` is the singular-values-only API
- norms and threshold-style logic compose from `svdvals`

Until `svdvals` exists, truthful CUDA capability reporting for those norm paths must remain false.

The expected implementation shape should stay close to PyTorch:

- `svd` remains the factorization API
- `svdvals` is implemented by a singular-values-only backend path, conceptually similar to `_linalg_svd(..., compute_uv=false)`
- higher-level threshold/rank logic composes from `svdvals`, not from host-side unpacking of `svd`

---

### Task 1: Move backend ownership from `tenferro-linalg` into `tenferro-linalg-prims`

**Files:**
- Modify: `tenferro-linalg-prims/Cargo.toml`
- Modify: `tenferro-linalg-prims/src/lib.rs`
- Create: `tenferro-linalg-prims/src/backend/mod.rs`
- Create: `tenferro-linalg-prims/src/backend/context.rs`
- Create: `tenferro-linalg-prims/src/backend/tests/mod.rs`
- Modify: `tenferro-linalg/Cargo.toml`
- Modify: `tenferro-linalg/src/backend/mod.rs`
- Modify: `tenferro-linalg/src/backend/tensor_api.rs`
- Modify: `tenferro-linalg/src/backend/tensor_context.rs`
- Modify: `tenferro-linalg/src/lib.rs`

**Step 1: Write the failing boundary test**

Add a new regression test in `tenferro-linalg-prims/src/backend/tests/mod.rs` that proves the context bridge is owned by `tenferro-linalg-prims`:

```rust
use crate::backend::TensorLinalgContextFor;

fn assert_ctx<T, C>()
where
    T: crate::KernelLinalgScalar,
    C: TensorLinalgContextFor<T>,
{
}

#[test]
fn cpu_context_is_bound_from_linalg_prims() {
    assert_ctx::<f64, tenferro_prims::CpuContext>();
}

#[test]
fn cuda_context_is_bound_from_linalg_prims() {
    assert_ctx::<f64, tenferro_prims::CudaContext>();
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p tenferro-linalg-prims cpu_context_is_bound_from_linalg_prims`

Expected: FAIL because the remaining context/backend binding surface still lives in `tenferro-linalg`, even though `TensorLinalgPrims` itself already lives in `tenferro-linalg-prims`.

**Step 3: Move the ownership boundary**

In `tenferro-linalg-prims`, add a `backend` module that owns:

```rust
pub trait TensorLinalgContextFor<T: KernelLinalgScalar>:
    tenferro_prims::TensorSemiringContextFor<tenferro_algebra::Standard<T>>
{
    type Backend: crate::TensorLinalgPrims<T, Context = Self>;
}
```

Export this from `tenferro-linalg-prims/src/lib.rs` and make `tenferro-linalg/src/backend/*` thin re-export shims:

```rust
pub use tenferro_linalg_prims::backend::*;
pub use tenferro_linalg_prims::{
    EigTensorResult, EigenTensorResult, KernelLinalgScalar, LinalgCapabilityOp,
    LinalgScalar, LuTensorResult, QrTensorResult, SvdTensorResult,
    TensorLinalgPrims as TensorLinalgBackend,
};
```

Also move feature ownership down by forwarding from `tenferro-linalg/Cargo.toml`:

```toml
[features]
default = ["linalg-faer"]
cuda = ["tenferro-linalg-prims/cuda"]
linalg-faer = ["tenferro-linalg-prims/linalg-faer"]
linalg-lapack = ["tenferro-linalg-prims/linalg-lapack"]
provider-src = ["tenferro-linalg-prims/provider-src"]
provider-inject = ["tenferro-linalg-prims/provider-inject"]
```

**Step 4: Run tests to verify it passes**

Run: `cargo test -p tenferro-linalg-prims cpu_context_is_bound_from_linalg_prims`

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg-prims/Cargo.toml tenferro-linalg-prims/src/lib.rs tenferro-linalg-prims/src/backend tenferro-linalg/Cargo.toml tenferro-linalg/src/backend tenferro-linalg/src/lib.rs
git commit -m "refactor(linalg): move tensor backend boundary into linalg-prims"
```

---

### Task 2: Move slice-level and tensor-level CPU backend ownership into `tenferro-linalg-prims`

**Files:**
- Modify: `tenferro-linalg-prims/Cargo.toml`
- Create: `tenferro-linalg-prims/src/backend/cpu.rs`
- Create: `tenferro-linalg-prims/src/backend/cpu_tensor_impl.rs`
- Create: `tenferro-linalg-prims/src/backend/tensor_helpers.rs`
- Create: `tenferro-linalg-prims/src/backend/slice_bridge.rs`
- Create: `tenferro-linalg-prims/src/backend/faer_backend/mod.rs`
- Create: `tenferro-linalg-prims/src/backend/faer_backend/real.rs`
- Create: `tenferro-linalg-prims/src/backend/faer_backend/complex.rs`
- Create: `tenferro-linalg-prims/src/backend/faer_backend/helpers.rs`
- Create: `tenferro-linalg-prims/src/backend/faer_backend/conversion.rs`
- Create: `tenferro-linalg-prims/src/backend/blas_lapack_backend/mod.rs`
- Create: `tenferro-linalg-prims/src/backend/blas_lapack_backend/helpers.rs`
- Create: `tenferro-linalg-prims/src/backend/blas_lapack_backend/real/mod.rs`
- Create: `tenferro-linalg-prims/src/backend/blas_lapack_backend/real/decompositions.rs`
- Create: `tenferro-linalg-prims/src/backend/blas_lapack_backend/real/linear_systems.rs`
- Create: `tenferro-linalg-prims/src/backend/blas_lapack_backend/real/spectral.rs`
- Create: `tenferro-linalg-prims/src/backend/blas_lapack_backend/complex/mod.rs`
- Create: `tenferro-linalg-prims/src/backend/blas_lapack_backend/complex/decompositions.rs`
- Create: `tenferro-linalg-prims/src/backend/blas_lapack_backend/complex/linear_systems.rs`
- Create: `tenferro-linalg-prims/src/backend/blas_lapack_backend/complex/spectral.rs`
- Modify: `tenferro-linalg/src/backend/mod.rs`
- Modify: `tenferro-linalg/src/backend/cpu.rs`
- Modify: `tenferro-linalg/src/backend/cpu_tensor_impl.rs`
- Modify: `tenferro-linalg/src/backend/slice_bridge.rs`
- Modify: `tenferro-linalg/src/backend/tensor_helpers.rs`
- Modify: `tenferro-linalg/src/backend/faer_backend/mod.rs`
- Modify: `tenferro-linalg/src/backend/faer_backend/real.rs`
- Modify: `tenferro-linalg/src/backend/faer_backend/complex.rs`
- Modify: `tenferro-linalg/src/backend/faer_backend/helpers.rs`
- Modify: `tenferro-linalg/src/backend/faer_backend/conversion.rs`
- Modify: `tenferro-linalg/src/backend/blas_lapack_backend/mod.rs`
- Modify: `tenferro-linalg/src/backend/blas_lapack_backend/helpers.rs`
- Modify: `tenferro-linalg/src/backend/blas_lapack_backend/real/mod.rs`
- Modify: `tenferro-linalg/src/backend/blas_lapack_backend/real/decompositions.rs`
- Modify: `tenferro-linalg/src/backend/blas_lapack_backend/real/linear_systems.rs`
- Modify: `tenferro-linalg/src/backend/blas_lapack_backend/real/spectral.rs`
- Modify: `tenferro-linalg/src/backend/blas_lapack_backend/complex/mod.rs`
- Modify: `tenferro-linalg/src/backend/blas_lapack_backend/complex/decompositions.rs`
- Modify: `tenferro-linalg/src/backend/blas_lapack_backend/complex/linear_systems.rs`
- Modify: `tenferro-linalg/src/backend/blas_lapack_backend/complex/spectral.rs`

**Note:** Most `tenferro-linalg/src/backend/**` edits in this task should be deletions or thin re-export shims, not substantive new logic. The real ownership move is into `tenferro-linalg-prims`.

**Step 1: Write the failing CPU relocation test**

Add a focused CPU smoke test under `tenferro-linalg-prims/src/backend/tests/mod.rs`:

```rust
#[test]
fn cpu_backend_still_solves_after_move() {
    let mut ctx = tenferro_prims::CpuContext::new(1);
    let a = tenferro_tensor::Tensor::from_slice(
        &[2.0_f64, 0.0, 0.0, 4.0],
        &[2, 2],
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b = tenferro_tensor::Tensor::from_slice(
        &[4.0_f64, 8.0],
        &[2],
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let x = <crate::backend::CpuTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::solve(
        &mut ctx, &a, &b,
    )
    .unwrap();
    assert_eq!(x.dims(), &[2]);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p tenferro-linalg-prims cpu_backend_still_solves_after_move`

Expected: FAIL because the CPU backend implementation still lives in `tenferro-linalg`.

**Step 3: Move the CPU implementation**

Move both layers of CPU backend code into `tenferro-linalg-prims`:

- the internal slice-level `LinalgBackend<T>` trait
- the faer/LAPACK provider modules
- the tensor-level CPU backend and tensor helpers

`tenferro-linalg` should stop owning faer/LAPACK dispatch logic and instead re-export the backend names for downstream ergonomics.

Keep the current provider feature names, but define them in `tenferro-linalg-prims/Cargo.toml` and forward them from `tenferro-linalg`.

**Step 4: Run targeted tests**

Run: `cargo test -p tenferro-linalg-prims cpu_backend_still_solves_after_move`

Expected: PASS.

Run: `cargo test -p tenferro-linalg`

Expected: PASS with the public API still unchanged.

**Step 5: Commit**

```bash
git add tenferro-linalg-prims/Cargo.toml tenferro-linalg-prims/src/backend tenferro-linalg/Cargo.toml tenferro-linalg/src/backend
git commit -m "refactor(linalg): move CPU backend implementation into linalg-prims"
```

---

### Task 3: Add a CUDA backend skeleton in `tenferro-linalg-prims` with truthful capability plumbing

**Files:**
- Modify: `tenferro-linalg-prims/Cargo.toml`
- Create: `tenferro-linalg-prims/src/backend/cuda/mod.rs`
- Create: `tenferro-linalg-prims/src/backend/cuda/runtime.rs`
- Create: `tenferro-linalg-prims/src/backend/cuda/wrappers.rs`
- Create: `tenferro-linalg-prims/src/backend/cuda/scalar_type.rs`
- Create: `tenferro-linalg-prims/src/backend/cuda/tests/mod.rs`
- Create: `tenferro-linalg-prims/src/backend/hip.rs`
- Modify: `tenferro-linalg-prims/src/backend/mod.rs`
- Modify: `tenferro-linalg/src/backend/cuda.rs`
- Modify: `tenferro-linalg/src/backend/hip.rs`

**Step 1: Write the failing CUDA boundary tests**

Add tests in `tenferro-linalg-prims/src/backend/cuda/tests/mod.rs`:

```rust
#[test]
fn cuda_backend_binding_is_owned_by_linalg_prims() {
    fn assert_ctx<T, C>()
    where
        T: crate::KernelLinalgScalar,
        C: crate::backend::TensorLinalgContextFor<T>,
    {
    }
    assert_ctx::<f64, tenferro_prims::CudaContext>();
}

#[test]
fn cuda_backend_reports_capabilities_through_type_level_traits() {
    assert!(
        !<crate::backend::CudaTensorLinalgBackend as crate::TensorLinalgPrims<f64>>::has_linalg_support(
            crate::LinalgCapabilityOp::Solve
        )
    );
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p tenferro-linalg-prims cuda_backend_binding_is_owned_by_linalg_prims`

Expected: FAIL because there is no CUDA backend in `tenferro-linalg-prims` yet.

**Step 3: Add the skeleton and dtype mapping trait**

Introduce a `CudaLinalgScalar` trait for dtype/FFI mapping instead of new `TypeId` dispatch:

```rust
pub trait CudaLinalgScalar: crate::KernelLinalgScalar {
    fn cuda_data_type() -> CudaDataType;
}
```

Compute op availability in `has_linalg_support(LinalgCapabilityOp)` on the backend implementation itself, not as duplicated per-op booleans on the dtype trait.

Implement `TensorLinalgPrims<T>` for `CudaTensorLinalgBackend` in `tenferro-linalg-prims`, and make `tenferro-linalg/src/backend/cuda.rs` / `hip.rs` pure re-export shims or delete them if the public module re-export already covers those names.

**Step 4: Run tests**

Run: `cargo test -p tenferro-linalg-prims cuda_backend_binding_is_owned_by_linalg_prims`

Expected: PASS.

Run: `cargo build -p tenferro-linalg-prims --features cuda`

Expected: PASS with a skeleton CUDA backend.

**Step 5: Commit**

```bash
git add tenferro-linalg-prims/Cargo.toml tenferro-linalg-prims/src/backend tenferro-linalg/src/backend/cuda.rs tenferro-linalg/src/backend/hip.rs
git commit -m "refactor(linalg): add linalg-prims-owned CUDA backend skeleton"
```

---

### Task 4: Implement the phase-1 CUDA linalg kernel basis in `tenferro-linalg-prims`

**Files:**
- Modify: `tenferro-linalg-prims/src/backend/cuda/mod.rs`
- Modify: `tenferro-linalg-prims/src/backend/cuda/runtime.rs`
- Modify: `tenferro-linalg-prims/src/backend/cuda/scalar_type.rs`
- Create: `tenferro-linalg-prims/src/backend/cuda/solve.rs`
- Create: `tenferro-linalg-prims/src/backend/cuda/solve_triangular.rs`
- Create: `tenferro-linalg-prims/src/backend/cuda/qr.rs`
- Create: `tenferro-linalg-prims/src/backend/cuda/svd.rs`
- Create: `tenferro-linalg-prims/src/backend/cuda/lu.rs`
- Create: `tenferro-linalg-prims/src/backend/cuda/cholesky.rs`
- Modify: `tenferro-linalg-prims/src/backend/cuda/tests/mod.rs`

**Step 1: Write failing CUDA parity tests**

Add small deterministic tests, feature-gated and runtime-optional:

```rust
fn cuda_runtime_available() -> bool {
    std::env::var_os("TENFERRO_TEST_CUDA").is_some()
}

fn solve_matches_cpu_generic<T: crate::KernelLinalgScalar>() {
    if !cuda_runtime_available() {
        return;
    }

    // 1. Solve on CpuContext using a tiny 2x2 or 3x3 column-major tensor.
    // 2. Transfer the same inputs to GPU.
    // 3. Solve on CudaContext.
    // 4. In the test only, transfer the CUDA result back to CPU.
    // 5. Compare elementwise within a dtype-specific tolerance.
}

#[test]
fn cuda_solve_matches_cpu_for_small_real_matrix_f64() { solve_matches_cpu_generic::<f64>(); }

#[test]
fn cuda_solve_matches_cpu_for_small_real_matrix_f32() { solve_matches_cpu_generic::<f32>(); }
```

Instantiate these generically for `f32` and `f64`. Do not advertise complex support unless the corresponding op is implemented and tested.

**Step 2: Run tests to verify they fail**

Run: `cargo test -p tenferro-linalg-prims --features cuda cuda_solve_matches_cpu_for_small_real_matrix -- --nocapture`

Expected: FAIL or return unsupported before the implementation exists.

**Step 3: Implement the phase-1 basis**

Implement the CUDA backend in `tenferro-linalg-prims` for:

- `solve`
- `solve_triangular`
- `qr`
- `thin_svd`
- `lu_factor`
- `cholesky`

Use runtime-loaded cuSOLVER/cuBLAS wrappers and preserve tensor residence on device. Batch execution may start as a host-side loop over trailing batch slices, but the slice data must stay on GPU and each per-batch kernel must consume device pointers directly. Do not add `to_memory_space_async(MainMemory)` or any equivalent host bounce in production paths.

Do **not** advertise:

- `eigen_sym`
- `eig`
- any composite capability implemented only by CPU-slice code in `tenferro-linalg`
- `svd(..., Some(options))` with `cutoff` in `tenferro-linalg` until that path has a dedicated device-resident implementation

**Step 4: Run tests**

Run: `cargo build -p tenferro-linalg-prims --features cuda`

Expected: PASS.

Run: `cargo test -p tenferro-linalg-prims`

Expected: PASS on non-CUDA environments.

Run: `cargo test -p tenferro-linalg-prims --features cuda cuda_solve_matches_cpu_for_small_real_matrix -- --nocapture`

Expected: PASS when `TENFERRO_TEST_CUDA` and the required runtime libraries are available.

**Step 5: Commit**

```bash
git add tenferro-linalg-prims/src/backend/cuda
git commit -m "feat(linalg): implement phase-1 CUDA kernel basis in linalg-prims"
```

---

### Task 5: Make `tenferro-linalg` capability-driven and eliminate hidden CPU-only fallthroughs

**Files:**
- Modify: `tenferro-linalg/src/prims_bridge.rs`
- Modify: `tenferro-linalg/src/ad_helpers/backend_ops.rs`
- Modify: `tenferro-linalg/src/ad_helpers/layout.rs`
- Modify: `tenferro-linalg/src/primal/decompositions.rs`
- Modify: `tenferro-linalg/src/primal/linear_systems.rs`
- Modify: `tenferro-linalg/src/primal/least_squares.rs`
- Modify: `tenferro-linalg/src/primal/spectral.rs`
- Modify: `tenferro-linalg/src/primal/matrix_functions.rs`
- Modify: `tenferro-linalg/src/primal/norms.rs`
- Modify: `tenferro-linalg/src/primal/tensor_ops.rs`
- Modify: `tenferro-linalg/src/frules/least_squares.rs`
- Modify: `tenferro-linalg/src/frules/linear_systems.rs`
- Modify: `tenferro-linalg/src/frules/lu_eigen.rs`
- Modify: `tenferro-linalg/src/frules/matrix_functions.rs`
- Modify: `tenferro-linalg/src/frules/norms.rs`
- Modify: `tenferro-linalg/src/frules/spectral.rs`
- Modify: `tenferro-linalg/src/frules/svd_qr.rs`
- Modify: `tenferro-linalg/src/rrules/least_squares.rs`
- Modify: `tenferro-linalg/src/rrules/linear_systems.rs`
- Modify: `tenferro-linalg/src/rrules/lu_eigen.rs`
- Modify: `tenferro-linalg/src/rrules/matrix_functions.rs`
- Modify: `tenferro-linalg/src/rrules/norms.rs`
- Modify: `tenferro-linalg/src/rrules/spectral.rs`
- Modify: `tenferro-linalg/src/rrules/svd_qr.rs`
- Modify: `tenferro-linalg/src/tests/runtime_capability.rs`
- Modify: `tenferro-linalg/tests/linalg_tests.rs`

**Step 1: Write the failing regression tests**

Add tests that a CUDA context never reaches CPU-slice helper errors for unsupported composite ops:

```rust
#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_stub_context_fails_through_capability_before_cpu_slice_materialization() {
    let mut ctx = tenferro_prims::CudaContext::new();
    let eye = make_tensor(vec![1.0_f64, 0.0, 0.0, 1.0], &[2, 2]);
    let err = tensorinv(&mut ctx, &eye, 1).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("not supported on the current linalg backend"));
}
```

Add two source-level regression assertions in `runtime_capability.rs`:

1. a denylist test that files meant to stay runtime-generic do not mention `backend::slice_bridge::`
2. an allowlist test for the few files that still mention `backend::slice_bridge::`, asserting those files also contain `require_linalg_support(`

Implement these as the same style of grep-based source tests already used in `runtime_capability.rs`.

**Step 2: Run tests to verify they fail**

Run: `cargo test -p tenferro-linalg cuda_stub_context_fails_through_capability_before_cpu_slice_materialization`

Expected: FAIL because several composite paths still touch `extract_slice`, `tensor_from_data`, or `slice_bridge` after entering with a CUDA context.

**Step 3: Audit and fix the genericity boundary**

Make two classes of change:

1. Add or tighten `require_linalg_support(...)` on every public/composite/AD entrypoint that still bottoms out in CPU-only helpers.
2. Replace helper APIs that force CPU extraction with tensor-preserving equivalents where the op should become GPU-generic next.

Start with these hotspots:

- `tenferro-linalg/src/prims_bridge.rs`
- `tenferro-linalg/src/ad_helpers/backend_ops.rs`
- `tenferro-linalg/src/primal/linear_systems.rs`
- `tenferro-linalg/src/primal/least_squares.rs`
- `tenferro-linalg/src/primal/spectral.rs`
- `tenferro-linalg/src/primal/decompositions.rs` (`svd` with `Some(options)`)

The rule is simple: if an operation is advertised as supported for `CudaContext`, it must stay tensor-level all the way down or explicitly materialize on GPU, never through `buffer().as_slice()` and never through an ad hoc GPU→CPU transfer fallback.

**Step 4: Run tests**

Run: `cargo test -p tenferro-linalg`

Expected: PASS.

Run: `cargo test -p tenferro-linalg cuda_stub_context_fails_through_capability_before_cpu_slice_materialization`

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/prims_bridge.rs tenferro-linalg/src/ad_helpers tenferro-linalg/src/primal tenferro-linalg/src/frules tenferro-linalg/src/rrules tenferro-linalg/src/tests/runtime_capability.rs tenferro-linalg/tests/linalg_tests.rs
git commit -m "refactor(linalg): make public paths capability-driven for CUDA contexts"
```

---

### Task 6: Enable the first GPU-generic composite surface and document the remaining gaps

**Files:**
- Modify: `tenferro-linalg/src/primal/linear_systems.rs`
- Modify: `tenferro-linalg/src/primal/least_squares.rs`
- Modify: `tenferro-linalg/src/primal/decompositions.rs`
- Modify: `tenferro-linalg/src/primal/spectral.rs`
- Modify: `tenferro-linalg/src/tests/batch_b_contracts.rs`
- Modify: `tenferro-linalg/tests/linalg_tests.rs`
- Modify: `docs/design/architecture.md`
- Modify: `docs/design/linalg.md`
- Modify: `docs/design/supported-ops.md`
- Modify: `docs/api_index.md`
- Modify: `README.md` (only if the supported CUDA linalg surface becomes user-visible enough to change the top-level support statement)

**Step 1: Write failing end-to-end tests**

Add public API tests for the first CUDA-enabled composite surface:

- `solve_ex`
- `cholesky_ex`
- `inv`

Keep them runtime-optional, small, and deterministic. The assertions must confirm:

- outputs stay shape-correct on a `CudaContext`
- unsupported ops fail through capability, not CPU-slice errors
- supported ops match CPU numerically after an explicit transfer back to host in the test harness only

**Step 2: Run tests to verify they fail**

Run: `cargo test -p tenferro-linalg cuda_context_fails_through_capability_before_cpu_slice_materialization`

Expected: PASS from Task 5, while the new composite success tests still FAIL.

**Step 3: Implement the first tensor-generic composite helpers**

Refactor `solve_ex`, `cholesky_ex`, and `inv` to build on tensor-level backend calls and semiring helpers rather than `extract_slice(...)` plus `Vec<T>` assembly. Do not introduce GPU→CPU fallback transfers in these implementations. Leave `det`, `slogdet`, `pinv`, `lstsq`, `matrix_exp`, `matrix_power`, `tensorinv`, `tensorsolve`, and AD rules capability-gated until their tensor-generic rewrites are complete.

For SVD specifically:

- keep the public `svd` signature unchanged
- keep `TensorLinalgPrims::thin_svd` as the only backend kernel contract
- allow `svd(..., None)` on CUDA once `thin_svd` exists and returns device-resident tensors correctly
- enable `svd(..., Some(options))` on CUDA for the `max_rank`-only case by rewriting the current repack path to use `Tensor::narrow(...)` views instead of CPU copies
- do not enable `svd(..., Some(options))` on CUDA when `options.cutoff.is_some()` until a dedicated device-resident cutoff kernel exists
- apply the same rule to `svd_rrule` / `svd_frule` when they are invoked with truncation options

Update the docs to say exactly which CUDA linalg ops are available and which are intentionally still unsupported.

**Step 4: Run verification**

Run: `cargo fmt --all --check`

Expected: PASS. If FAIL, run `cargo fmt --all`.

Run: `cargo test -p tenferro-linalg-prims`

Expected: PASS.

Run: `cargo test -p tenferro-linalg`

Expected: PASS.

Run: `cargo build -p tenferro-linalg-prims --features cuda`

Expected: PASS.

Run: `cargo build -p tenferro-linalg --features cuda`

Expected: PASS.

Run: `cargo doc -p tenferro-linalg-prims --no-deps`

Expected: PASS.

Run: `cargo doc -p tenferro-linalg --no-deps`

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/primal tenferro-linalg/src/tests tenferro-linalg/tests README.md docs/design/architecture.md docs/design/linalg.md docs/design/supported-ops.md docs/api_index.md
git commit -m "feat(linalg): enable first GPU-generic composite surface"
```

---

### Suggested PR split

This work should be split unless the implementation stays unexpectedly small:

- **PR 1:** Tasks 1-2 only. This lands ownership cleanup and CPU backend relocation with no real CUDA linalg behavior change.
- **PR 2:** Torch-aligned substrate work. This should contain the missing linalg helpers and result-status plumbing needed for GPU-generic composite behavior, even if no end-user CUDA SVD is enabled yet.
- **PR 3:** Minimal exact CUDA SVD kernel enablement. This should land `svd(..., None)` only, plus truthful capability reporting.
- **PR 4:** Composite genericization for `svd(None)` users and tensor-native norm paths.
- **PR 5:** Truncation policy work: `max_rank` first, `cutoff` only once a dedicated device-resident postprocess exists.
- **PR 6:** Optional `gesvdj` heuristics and follow-on optimization.

If the CUDA kernel work stalls, PRs 1-2 should still be merged independently because they are the prerequisite substrate, not speculative cleanup.

---

### Final verification checklist

Run these before calling the consolidation complete:

```bash
cargo fmt --all --check
cargo test -p tenferro-linalg-prims
cargo test -p tenferro-linalg
cargo build -p tenferro-linalg-prims --features cuda
cargo build -p tenferro-linalg --features cuda
cargo doc -p tenferro-linalg-prims --no-deps
cargo doc -p tenferro-linalg --no-deps
```

If a CUDA-equipped runner is available, also run:

```bash
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-linalg-prims --features cuda -- --nocapture
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-linalg --features cuda -- --nocapture
```

### Notes and guardrails

- Do not reintroduce backend ownership into `tenferro-linalg`; re-exports are fine, wrappers are not.
- Do not advertise CUDA support for any op whose implementation still depends on `buffer().as_slice()` or `backend::slice_bridge::...`.
- Do not add ad hoc GPU→CPU transfers in production linalg code. If a CUDA path cannot stay device-resident, keep it unsupported for now.
- Keep column-major `(m, n, batch...)` semantics intact. Batch dimensions remain trailing.
- Avoid new `TypeId`-based dtype dispatch in the CUDA path. Use trait-associated constants on a dedicated CUDA dtype trait.
- Keep tests generic and small. Prefer `f32`/`f64` first for phase 1 CUDA coverage.
