# Linalg AD Cleanup Replan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the remaining host-side AD/linalg bridge paths now that the dense eager core is effectively in place, and finish the tensor-native cleanup without reintroducing `slice_bridge`, `extract_slice`, or ad hoc host reconstruction in public/composite linalg flows.

**Architecture:** Treat the remaining work as a consumer cleanup phase, not a new substrate phase. Keep `tenferro-device`, `tenferro-tensor`, and `tenferro-prims` as the substrate layers; finish `tenferro-linalg` by replacing AD helper and composite host-vector flows with tensor-native backend helpers that consume the already-completed dense eager core.

**Tech Stack:** Rust workspace crates `tenferro-linalg`, `tenferro-linalg-prims`, `tenferro-prims`, `tenferro-tensor`; existing tensor-native LU/SVD/QR/solve contracts; runtime capability source guards; CPU/CUDA parity tests.

## Status Update After Executing This Replan

This file remains the historical execution plan for the original linalg AD cleanup. Do not reinterpret the task list below as the current frontier. The cleanup described by this replan has already been executed on this line of work; use the notes in this section as the current handoff snapshot.

### What already landed from the original cleanup plan

- The host-bridge cleanup targeted by this replan was completed earlier on this branch lineage.
- The key results of that work were:
  - source/runtime guards were added so public/composite linalg code does not quietly regress to CPU-name checks or `slice_bridge` transport
  - remaining live `slice_bridge`-driven AD/composite paths were removed
  - dead `backend/slice_bridge.rs` dependence was deleted from the active linalg stack
  - focused verification, release-mode tests, docs, and coverage gates were brought back to green
  - the resulting cleanup was upstreamed earlier via PR `#553`

### Current follow-on work now happening on this branch

The active work is no longer the `slice_bridge` cleanup itself. The branch has moved on to a broader CPU/GPU-symmetric tensor-native linalg AD refactor whose immediate goal is to keep high-level AD formulas device-agnostic while pushing runtime-specific details down into `tenferro-linalg-prims` and `tenferro-prims`.

Authoritative execution context:

- worktree: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/complex-real-unary-substrate`
- branch: `feat/complex-real-unary-substrate`
- do not use the repository root checkout at `/home/shinaoka/tensor4all/tenferro-rs` as the source of truth for this refactor; that checkout was previously dirty/stale

### Follow-on phases already completed after this replan

- Backend ownership/layout cleanup is complete.
  - `tenferro-linalg-prims/src/backend/cpu.rs` now dispatches through local per-op CPU modules instead of old `cpu_faer` / `cpu_lapack` / `cpu_tensor_impl` path imports.
  - `tenferro-linalg-prims/src/backend/hip.rs` now routes through mirrored per-op HIP stub modules instead of open-coded inline unsupported bodies.
- Tensor view ergonomics is complete.
  - Added `Tensor::mT()` and `Tensor::mH()` in `tenferro-tensor/src/tensor/views/basic.rs`.
  - Added `Tensor<Complex32>::real()/imag()` and `Tensor<Complex64>::real()/imag()` in `tenferro-tensor/src/tensor/views/complex.rs`.
  - Important semantic note: `real()` / `imag()` currently require a resolved complex tensor because they are built on `view_as_real()`, which rejects lazy-conjugated inputs.
- Functional diagonal substrate is complete in the shared linalg layer.
  - Added `tenferro-linalg/src/ad_helpers/diagonal.rs` with `diag_extract`, `trace_tensor`, `diag_scatter`, `diag_embed`, and `diag_scatter_add`.
  - Added `prims_bridge::semiring_core_single_input_into(...)` in `tenferro-linalg/src/prims_bridge.rs`.
  - `diag_extract` reorders raw `Tensor::diagonal(&[(0, 1)])` output from `[batch..., diag]` to `[diag, batch...]` so downstream linalg formulas stay matrix-first.
  - `diag_scatter` auto-dispatches between `AntiTrace` and `AntiDiag` based on input shape.
  - `diag_scatter_add` makes a functional copy via `Tensor::stack(&[base], 0)?.squeeze_dim(0)?` before updating the diagonal.
- CUDA diagonal execution parity is complete.
  - `tenferro-prims/src/cuda/mod.rs` now includes `mod diagonal;` and `CudaPlanStorage::DeferredDiagonal`.
  - `tenferro-prims/src/cuda/execution.rs` now validates/plans/dispatches `Trace`, `AntiTrace`, and `AntiDiag` instead of rejecting them.

Validation already completed for the post-replan substrate phases:

- `cargo test -p tenferro-linalg-prims --release --lib`
- `cargo test -p tenferro-linalg --release --lib`
- `cargo test -p tenferro-tensor --release --lib`
- `cargo test -p tenferro-prims --release --lib`
- focused sweep also passed:
  - `cargo fmt --all --check`
  - `cargo test -p tenferro-tensor --release --lib`
  - `cargo test -p tenferro-prims --release --lib`
  - `cargo test -p tenferro-linalg-prims --release --lib`
  - `cargo test -p tenferro-linalg --release --lib`

### Current active problem: solve-class AD rewrite

The immediate in-progress task is rewriting solve-class AD rules tensor-natively, without falling back to `extract_data`, `tensor_from_data`, raw `Vec<T>` temporaries, or manual flat-buffer batch loops.

Primary target files:

- `tenferro-linalg/src/rrules/linear_systems.rs`
- `tenferro-linalg/src/frules/linear_systems.rs`

Current state of those files:

- they still contain `extract_data(...)`
- they still reconstruct outputs through `tensor_from_data(...)`
- they still use direct raw-slice backend helpers such as `backend_mat_mul`, `backend_solve`, and `backend_solve_tri`
- they still spell the batched linear algebra as manual per-batch flat-buffer loops

Preparatory helper already added for this rewrite:

- `tenferro-linalg/src/ad_helpers/views.rs`
  - `matrix_transpose`
  - `matrix_adjoint`
  - `rhs_to_matrix`
  - `matrix_to_rhs`

Why this helper exists:

- tenferro linalg treats the first two axes as matrix axes and the trailing axes as batch axes
- `Tensor::mT()` / `Tensor::mH()` transpose the last two axes, so they are not the correct helper for batched linalg formulas in this crate
- solve/QR/SVD rewrites therefore need first-two-axis matrix helpers

Important current technical issue:

- `matrix_adjoint` currently uses `tensor.conj()`
- `Tensor::conj()` requires `T: tenferro_algebra::Conjugate`
- the existing solve AD rule signatures are currently generic over `T: KernelLinalgScalar`
- `KernelLinalgScalar` is only `LinalgScalar`; it does not itself imply `tenferro_algebra::Conjugate`
- therefore the next implementation step must decide whether to:
  - add an explicit `T: Conjugate` bound to the solve-class helpers/rules that need lazy tensor conjugation, or
  - split transpose-only and adjoint-using paths more carefully so real-only formulas stay on `matrix_transpose` while complex-capable code opts into `matrix_adjoint`

The newly added `ad_helpers/views.rs` had just been introduced when work paused and had not yet been recompiled/tested in the current segment. The next agent should validate that file first, then continue the solve rewrite.

---

## Current Baseline

- Worktree: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/complex-real-unary-substrate`
- Branch: `feat/complex-real-unary-substrate`
- Branch head when this plan was written: `e09e77f`

### Dense eager core work already complete on this branch

- fallible dense constructors
- metadata phase 2 binary ops
- metadata/scalar bridge
- representation helpers (`view_as_real`, `view_as_complex`)
- RNG core, including default-generator support for tensor RNG constructors
- LU metadata tensorization in `tenferro-linalg-prims`
- tensor-native public/composite cleanup for:
  - `det`
  - `slogdet`
  - `svd` cutoff
  - `matrix_power`
  - `vander`
  - `cross`
  - `householder_product`

### What is still not clean

The remaining host-side bridge paths are concentrated in AD helpers and a few backend wrappers:

- `tenferro-linalg/src/ad_helpers/backend_ops.rs`
  - still uses `backend::slice_bridge::{solve_vec, solve_triangular_vec, thin_svd_vec, qr_vec}`
- `tenferro-linalg/src/ad_helpers/complex_ops.rs`
  - still solves through `backend::slice_bridge::solve_vec`
- `tenferro-linalg/src/ad_helpers/matrix_exp.rs`
  - still solves Padé linear systems through `backend::slice_bridge::solve_vec`
- `tenferro-linalg/src/ad_helpers/layout.rs`
  - still exposes `extract_slice`, `extract_data`, and CPU-slice assumptions
- `tenferro-linalg/src/backend/slice_bridge.rs`
  - remains as a live bridge because the AD helper layer still depends on it

These remaining paths matter because they are the last meaningful users of host-vector reconstruction in `tenferro-linalg` after the dense eager core build-out.

## Design Constraints

- Do not update historical plan files from `2026-03-23`; keep this as the new canonical execution plan.
- Do not add new low-level substrate unless the cleanup proves a real gap.
- Do not add new linalg-specific one-off helpers in `tenferro-linalg`.
- Prefer tensor-native backend helper composition over `Vec<T>` helper APIs.
- Do not add new CPU fallback in `tenferro-linalg`.
- Keep feature-first locality where it materially improves navigation.
- Preserve CPU/CUDA genericity in public/composite code paths.

## Target End State

After this plan:

- `tenferro-linalg/src/ad_helpers/backend_ops.rs` no longer mentions `backend::slice_bridge::`
- `tenferro-linalg/src/ad_helpers/complex_ops.rs` no longer mentions `backend::slice_bridge::`
- `tenferro-linalg/src/ad_helpers/matrix_exp.rs` no longer mentions `backend::slice_bridge::`
- `tenferro-linalg/src/ad_helpers/layout.rs` no longer exposes `extract_slice`-driven tensor materialization for the remaining live call paths
- `tenferro-linalg/src/backend/slice_bridge.rs` is either deleted or isolated to dead-free, no-longer-called code and ready for deletion
- runtime/source-level regression tests explicitly guard the cleanup so the bridge does not quietly come back

## Execution Order

1. Add regression guards for remaining AD host bridges
2. Introduce tensor-native AD backend helper surface
3. Migrate direct `slice_bridge` callers (`complex_ops`, `matrix_exp`)
4. Migrate AD rule call sites to tensor-native backend helpers
5. Remove dead `extract_slice` / `slice_bridge` helpers
6. Re-close the cleanup with focused and full verification

## Task 1: Add regression guards for remaining AD host bridges

**Files:**
- Modify: `tenferro-linalg/src/tests/runtime_capability.rs`

**Step 1: Write the failing source-level tests**

Add new guards covering:

- `ad_helpers/backend_ops.rs` must not contain `backend::slice_bridge::`
- `ad_helpers/complex_ops.rs` must not contain `backend::slice_bridge::`
- `ad_helpers/matrix_exp.rs` must not contain `backend::slice_bridge::`
- `ad_helpers/layout.rs` must not contain `extract_slice(` for the remaining live AD paths once migrated

Prefer the same helper style already used in `runtime_capability.rs`.

**Step 2: Run the focused test and verify it fails**

Run:

```bash
cargo test -p tenferro-linalg --release --lib runtime_capability -- --nocapture
```

Expected: FAIL because the current sources still mention `slice_bridge` and `extract_slice`.

**Step 3: Write minimal implementation**

None in this task. RED only.

**Step 4: Re-run to confirm the intended failure**

Expected: FAIL on the new source guards.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/tests/runtime_capability.rs
git commit -m "test: guard remaining ad host bridges"
```

## Task 2: Add tensor-native AD backend helper surface

**Files:**
- Modify: `tenferro-linalg/src/ad_helpers/backend_ops.rs`
- Modify: `tenferro-linalg/src/ad_helpers/layout.rs`
- Add if needed: `tenferro-linalg/src/ad_helpers/backend_ops/tests/*.rs`

**Step 1: Reuse the failing source-level tests from Task 1**

No new source-level tests yet.

**Step 2: Run focused linalg tests for the affected AD consumers**

Run:

```bash
cargo test -p tenferro-linalg --release --lib least_squares -- --nocapture
cargo test -p tenferro-linalg --release --lib linear_systems -- --nocapture
cargo test -p tenferro-linalg --release --lib svd_qr -- --nocapture
```

Expected: PASS at baseline, but helpers still use host bridges.

**Step 3: Write minimal implementation**

Refactor `ad_helpers/backend_ops.rs` so it stops returning `Vec<T>` out of `slice_bridge`.

Recommended shape:

- replace raw `Vec<T>`-returning wrappers with tensor-native wrappers that:
  - build temporary tensors from AD host data only at the edge where AD inputs are still host vectors
  - call backend tensor contracts (`solve`, `solve_triangular`, `thin_svd`, `qr`)
  - return tensor outputs or explicitly structured tensor results
- keep helper count small; do not create op-specific helpers if one generic helper suffices
- shrink `layout.rs` so it only contains conversions still genuinely needed after the rewrite

Important:

- do not let downstream callers reach into `backend::slice_bridge`
- if a helper still needs host vector conversion at the AD boundary, keep it in `ad_helpers`, not `backend`

**Step 4: Run focused tests**

Run:

```bash
cargo test -p tenferro-linalg --release --lib least_squares -- --nocapture
cargo test -p tenferro-linalg --release --lib linear_systems -- --nocapture
cargo test -p tenferro-linalg --release --lib svd_qr -- --nocapture
cargo test -p tenferro-linalg --release --lib runtime_capability -- --nocapture
```

Expected: source guard still fails for direct callers not yet migrated, but helper layer compiles and AD rule coverage stays green.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/ad_helpers/backend_ops.rs \
        tenferro-linalg/src/ad_helpers/layout.rs
git commit -m "refactor: add tensor-native ad backend helpers"
```

## Task 3: Remove direct `slice_bridge` use from `complex_ops` and `matrix_exp`

**Files:**
- Modify: `tenferro-linalg/src/ad_helpers/complex_ops.rs`
- Modify: `tenferro-linalg/src/ad_helpers/matrix_exp.rs`

**Step 1: Add focused regression coverage if needed**

If current test names are too broad, add targeted source or behavior tests near existing linalg test files to pin:

- `complex_solve_nn` still matches the previous numerical result
- `matrix_exp` Padé solve still matches the previous result

**Step 2: Run focused tests**

Run:

```bash
cargo test -p tenferro-linalg --release --lib matrix_functions -- --nocapture
cargo test -p tenferro-linalg --release --lib runtime_capability -- --nocapture
```

Expected: source guards still fail before the rewrite lands.

**Step 3: Write minimal implementation**

- Route `complex_solve_nn` through the new tensor-native AD backend helper, not `backend::slice_bridge::solve_vec`
- Route the final Padé linear solve in `matrix_exp.rs` through the new tensor-native AD backend helper
- Keep all existing math the same; only change the transport/execution path

**Step 4: Run focused tests**

Run:

```bash
cargo test -p tenferro-linalg --release --lib matrix_functions -- --nocapture
cargo test -p tenferro-linalg --release --lib runtime_capability -- --nocapture
```

Expected: source guards for `complex_ops` and `matrix_exp` now pass.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/ad_helpers/complex_ops.rs \
        tenferro-linalg/src/ad_helpers/matrix_exp.rs
git commit -m "refactor: remove ad slice bridge solve paths"
```

## Task 4: Migrate AD rule call sites to tensor-native helpers

**Files:**
- Modify: `tenferro-linalg/src/frules/least_squares.rs`
- Modify: `tenferro-linalg/src/frules/linear_systems.rs`
- Modify: `tenferro-linalg/src/frules/lu_eigen.rs`
- Modify: `tenferro-linalg/src/frules/svd_qr.rs`
- Modify: `tenferro-linalg/src/rrules/least_squares.rs`
- Modify: `tenferro-linalg/src/rrules/linear_systems.rs`
- Modify: `tenferro-linalg/src/rrules/lu_eigen.rs`
- Modify: `tenferro-linalg/src/rrules/norms.rs`
- Modify: `tenferro-linalg/src/rrules/svd_qr.rs`

**Step 1: Reuse existing behavior tests**

Do not start by widening test coverage. Reuse current frule/rrule and primal tests first.

**Step 2: Run focused tests**

Run:

```bash
cargo test -p tenferro-linalg --release --lib frules -- --nocapture
cargo test -p tenferro-linalg --release --lib rrules -- --nocapture
```

Expected: PASS at baseline, but still routed through the old helper shape.

**Step 3: Write minimal implementation**

Update the AD rule files so they consume the new tensor-native backend helper surface instead of raw `Vec<T>` wrappers.

Guidelines:

- prefer local tensor arithmetic over repeated host vector pack/unpack
- if a helper remains host-vector based only because the AD math itself is still naturally vectorized, keep that boundary in the AD file, not in `backend::slice_bridge`
- remove dead `transpose/pack/unpack` glue once callers no longer need it

**Step 4: Run focused tests**

Run:

```bash
cargo test -p tenferro-linalg --release --lib frules -- --nocapture
cargo test -p tenferro-linalg --release --lib rrules -- --nocapture
cargo test -p tenferro-linalg --release --lib runtime_capability -- --nocapture
```

Expected: PASS, and the source guards should now clear for `ad_helpers/backend_ops.rs`.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/frules/least_squares.rs \
        tenferro-linalg/src/frules/linear_systems.rs \
        tenferro-linalg/src/frules/lu_eigen.rs \
        tenferro-linalg/src/frules/svd_qr.rs \
        tenferro-linalg/src/rrules/least_squares.rs \
        tenferro-linalg/src/rrules/linear_systems.rs \
        tenferro-linalg/src/rrules/lu_eigen.rs \
        tenferro-linalg/src/rrules/norms.rs \
        tenferro-linalg/src/rrules/svd_qr.rs
git commit -m "refactor: migrate ad rules off slice bridge"
```

## Task 5: Delete or fully isolate dead bridge helpers

**Files:**
- Modify or delete: `tenferro-linalg/src/backend/slice_bridge.rs`
- Modify: `tenferro-linalg/src/backend/mod.rs`
- Modify: `tenferro-linalg/src/ad_helpers/layout.rs`

**Step 1: Add source-level guard if needed**

If `slice_bridge.rs` remains temporarily, add a source-level test asserting it is no longer referenced from public/composite and AD helper code.

**Step 2: Run focused tests**

Run:

```bash
cargo test -p tenferro-linalg --release --lib runtime_capability -- --nocapture
```

Expected: PASS or fail only on any final lingering references.

**Step 3: Write minimal implementation**

- remove dead `slice_bridge` exports from `backend/mod.rs`
- delete `slice_bridge.rs` if unused
- if deletion is not yet possible, isolate it behind dead-code-free private helpers and leave a clear follow-up marker in the new plan, not in historical docs
- simplify `layout.rs` to the conversions still actually used

**Step 4: Run focused tests**

Run:

```bash
cargo test -p tenferro-linalg --release --lib runtime_capability -- --nocapture
cargo test -p tenferro-linalg --release --lib
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/backend/mod.rs \
        tenferro-linalg/src/ad_helpers/layout.rs \
        tenferro-linalg/src/backend/slice_bridge.rs \
        tenferro-linalg/src/tests/runtime_capability.rs
git commit -m "refactor: retire linalg slice bridge"
```

## Task 6: Full verification and branch closeout

**Files:**
- No code changes required unless a gate fails

**Step 1: Run the full required verification**

Run:

```bash
cargo fmt --all --check
cargo test --workspace --release
cargo llvm-cov --workspace --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

If you are preparing a PR instead of only pushing a branch, also run:

```bash
cargo nextest run --workspace --release --no-fail-fast
cargo test --doc --workspace --release
```

**Step 2: Fix any failures**

Do not proceed until all required local gates pass.

**Step 3: Commit any final fixes**

```bash
git add <files>
git commit -m "fix: close remaining linalg cleanup regressions"
```

**Step 4: Push**

```bash
git push origin feat/complex-real-unary-substrate
```

**Step 5: Stop**

Do not create a PR unless explicitly requested. This plan is compatible with branch-only progress.
