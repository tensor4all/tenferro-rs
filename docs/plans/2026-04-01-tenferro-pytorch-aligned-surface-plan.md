# Tenferro PyTorch-Aligned Surface Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align `tenferro::Tensor` public AD surface for `norm`, `eig/eigh`, and `lstsq` with PyTorch while keeping the internal runtime-dispatched AD implementation generic across CPU / CUDA / ROCm.

**Architecture:** Public API changes happen first at the `Tensor` surface and result-type layer, then propagate down through oracle replay, `frule/rrule`, and `LinearizedOp::jvp/vjp`. Every operator family must pass oracle replay, internal seam tests, and public integration tests before moving to the next family.

**Tech Stack:** Rust, `tenferro`, `tenferro-linalg`, `tenferro-internal-ad-linalg`, `tidu`-style `LinearizedOp`, runtime dispatch, tensor-ad-oracles replay tests.

---

### Task 1: Add the failing `norm` surface tests

**Files:**
- Modify: `tenferro/tests/integration/linalg_surface_tests.rs`
- Modify: `tenferro/tests/integration/public_jvp.rs`
- Modify: `tenferro/tests/integration/autograd_surface_tests.rs`

**Step 1: Write failing public tests for `vector_norm` and `matrix_norm`**

Add tests that:

- call `Tensor::vector_norm(...)` on complex input
- call `Tensor::matrix_norm(...)` on complex input
- call `jvp(...)` through each path
- verify output tensors are real-valued

**Step 2: Run only the new failing tests**

Run:

```bash
cargo test -p tenferro --test integration --release public_jvp -- --nocapture
```

Expected: compile errors or runtime failures because the new public methods do not exist yet.

**Step 3: Commit the failing tests**

```bash
git add tenferro/tests/integration/linalg_surface_tests.rs tenferro/tests/integration/public_jvp.rs tenferro/tests/integration/autograd_surface_tests.rs
git commit -m "test: add failing norm split surface coverage"
```

### Task 2: Introduce public `vector_norm` / `matrix_norm` types and methods

**Files:**
- Modify: `tenferro/src/lib.rs`
- Modify: `internal/tenferro-internal-ad-surface/src/core/dynamic/tensor.rs`
- Modify: `internal/tenferro-internal-ad-linalg/src/linearized.rs`
- Modify: `tenferro/README.md`

**Step 1: Add public norm-order types**

Define new public order enums:

- `VectorNormOrd`
- `MatrixNormOrd`

Expose them from `tenferro/src/lib.rs`.

**Step 2: Add `Tensor::vector_norm(...)` and `Tensor::matrix_norm(...)`**

Implement the public methods in `tensor.rs`.

**Step 3: Make `Tensor::norm(...)` a wrapper**

Convert the old `norm` surface into a convenience wrapper over the split methods or remove it if the surface decision is to hard cut completely.

**Step 4: Run the public tests**

Run:

```bash
cargo test -p tenferro --test integration --release -- --nocapture
```

Expected: public tests still fail because the internal AD/oracle layers are not updated yet.

**Step 5: Commit**

```bash
git add tenferro/src/lib.rs internal/tenferro-internal-ad-surface/src/core/dynamic/tensor.rs internal/tenferro-internal-ad-linalg/src/linearized.rs tenferro/README.md
git commit -m "feat: split public norm surface"
```

### Task 3: Align oracle replay for `vector_norm` and `matrix_norm`

**Files:**
- Modify: `tenferro-linalg/tests/oracle_db/replay.rs`
- Modify: `tenferro-linalg/tests/oracle_db/support.rs`
- Modify: `tenferro-linalg/tests/oracle_db/db.rs`

**Step 1: Add or enable `vector_norm` / `matrix_norm` replay support**

Teach replay to load the oracle families and map them to tenferro kernels.

**Step 2: Run the replay tests**

Run:

```bash
cargo test -p tenferro-linalg --release oracle_db_replay_against_tensor_ad_oracles -- --nocapture
```

Expected: failures identify missing AD math or contract mismatches.

**Step 3: Commit**

```bash
git add tenferro-linalg/tests/oracle_db/replay.rs tenferro-linalg/tests/oracle_db/support.rs tenferro-linalg/tests/oracle_db/db.rs
git commit -m "test: enable norm split oracle replay"
```

### Task 4: Implement complex `vector_norm` / `matrix_norm` AD rules

**Files:**
- Modify: `tenferro-linalg/src/frules/norms.rs`
- Modify: `tenferro-linalg/src/rrules/norms.rs`
- Modify: `internal/tenferro-internal-ad-linalg/src/linearized.rs`
- Test: `internal/tenferro-internal-ad-linalg/tests/dyn_linalg_ops.rs`

**Step 1: Write focused internal seam tests**

Add complex JVP/VJP tests for the split norm operators.

**Step 2: Run the failing seam tests**

Run:

```bash
cargo test -p tenferro-internal-ad-linalg --release dyn_linalg_ops -- --nocapture
```

Expected: failures from real-only gates or wrong real/complex bridge behavior.

**Step 3: Implement minimal AD support**

Update `frule/rrule` and the dynamic `LinearizedOp` path so:

- complex input is accepted
- output remains real-valued
- backward returns complex gradient

**Step 4: Run replay + seam + public tests**

Run:

```bash
cargo test -p tenferro-linalg --release oracle_db_replay_against_tensor_ad_oracles -- --nocapture
cargo test -p tenferro-internal-ad-linalg --release dyn_linalg_ops -- --nocapture
cargo test -p tenferro --test integration --release -- --nocapture
```

Expected: all norm-related coverage passes.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/frules/norms.rs tenferro-linalg/src/rrules/norms.rs internal/tenferro-internal-ad-linalg/src/linearized.rs internal/tenferro-internal-ad-linalg/tests/dyn_linalg_ops.rs
git commit -m "feat: support complex norm ad through split surface"
```

### Task 5: Add the failing `eig/eigh` public tests

**Files:**
- Modify: `tenferro/tests/integration/linalg_surface_tests.rs`
- Modify: `tenferro/tests/integration/public_jvp.rs`
- Modify: `tenferro/tests/integration/autograd_surface_tests.rs`

**Step 1: Write failing tests for `eigh()` naming and complex `eig()` coverage**

Cover:

- `Tensor::eigh()` public method
- `Tensor::eig()` complex-input JVP
- public dtype/result expectations

**Step 2: Run failing tests**

Run:

```bash
cargo test -p tenferro --test integration --release -- --nocapture
```

Expected: failures because `eigh()` is not yet the public method or `eig()` is still real-only in AD.

**Step 3: Commit**

```bash
git add tenferro/tests/integration/linalg_surface_tests.rs tenferro/tests/integration/public_jvp.rs tenferro/tests/integration/autograd_surface_tests.rs
git commit -m "test: add failing eig and eigh surface coverage"
```

### Task 6: Rename `eigen()` to `eigh()` and preserve split public contracts

**Files:**
- Modify: `tenferro/src/lib.rs`
- Modify: `internal/tenferro-internal-ad-surface/src/core/dynamic/tensor.rs`
- Modify: `tenferro/README.md`

**Step 1: Rename the public method**

Replace `eigen()` with `eigh()` in the public surface.

**Step 2: Update docs and examples**

Make README and rustdoc examples use `eigh()`.

**Step 3: Run the public tests**

Run:

```bash
cargo test -p tenferro --test integration --release -- --nocapture
```

Expected: `eigh()` tests improve, `eig()` complex AD still fails.

**Step 4: Commit**

```bash
git add tenferro/src/lib.rs internal/tenferro-internal-ad-surface/src/core/dynamic/tensor.rs tenferro/README.md
git commit -m "refactor: rename public eigen api to eigh"
```

### Task 7: Implement complex `eig` AD support

**Files:**
- Modify: `tenferro-linalg/src/result_types/spectral.rs`
- Modify: `tenferro-linalg/src/result_types/cotangents.rs`
- Modify: `tenferro-linalg/src/frules/spectral.rs`
- Modify: `tenferro-linalg/src/rrules/spectral.rs`
- Modify: `tenferro-linalg/tests/oracle_db/replay.rs`
- Modify: `tenferro-linalg/tests/oracle_db/support.rs`
- Modify: `internal/tenferro-internal-ad-linalg/src/linearized.rs`
- Modify: `internal/tenferro-internal-ad-linalg/tests/dyn_linalg_ops.rs`

**Step 1: Add failing oracle replay support for complex `eig`**

Enable the oracle family and observe the first concrete mismatch.

**Step 2: Add failing seam tests**

Cover complex-input `eig` JVP/VJP.

**Step 3: Generalize result/cotangent contract**

Update the `EigResult` / `EigCotangent` types so complex-input `eig` is representable cleanly.

**Step 4: Implement rule support**

Update `frule`, `rrule`, and `LinearizedOp` dispatch. Preserve `eig` and `eigh` as separate public contracts even if helper logic is shared.

**Step 5: Run replay + seam + public tests**

Run:

```bash
cargo test -p tenferro-linalg --release oracle_db_replay_against_tensor_ad_oracles -- --nocapture
cargo test -p tenferro-internal-ad-linalg --release dyn_linalg_ops -- --nocapture
cargo test -p tenferro --test integration --release -- --nocapture
```

**Step 6: Commit**

```bash
git add tenferro-linalg/src/result_types/spectral.rs tenferro-linalg/src/result_types/cotangents.rs tenferro-linalg/src/frules/spectral.rs tenferro-linalg/src/rrules/spectral.rs tenferro-linalg/tests/oracle_db/replay.rs tenferro-linalg/tests/oracle_db/support.rs internal/tenferro-internal-ad-linalg/src/linearized.rs internal/tenferro-internal-ad-linalg/tests/dyn_linalg_ops.rs
git commit -m "feat: support complex eig ad"
```

### Task 8: Add the failing `lstsq` contract tests

**Files:**
- Modify: `tenferro/tests/integration/linalg_surface_tests.rs`
- Modify: `tenferro/tests/integration/public_jvp.rs`
- Modify: `tenferro/tests/integration/autograd_surface_tests.rs`
- Modify: `tenferro-linalg/tests/oracle_db/replay.rs`

**Step 1: Write failing tests for the four-output result**

Cover:

- `solution`
- `residuals`
- `rank`
- `singular_values`

and the differentiability split.

**Step 2: Run the failing tests**

Run:

```bash
cargo test -p tenferro --test integration --release -- --nocapture
```

Expected: failures because the current result contract still exposes the old residual object.

**Step 3: Commit**

```bash
git add tenferro/tests/integration/linalg_surface_tests.rs tenferro/tests/integration/public_jvp.rs tenferro/tests/integration/autograd_surface_tests.rs tenferro-linalg/tests/oracle_db/replay.rs
git commit -m "test: add failing lstsq contract coverage"
```

### Task 9: Rewrite `lstsq` result contract

**Files:**
- Modify: `tenferro-linalg/src/result_types/decomposition.rs`
- Modify: `tenferro-linalg/src/result_types/cotangents.rs`
- Modify: `tenferro-linalg/src/primal/least_squares.rs`
- Modify: `internal/tenferro-internal-ad-linalg/src/linearized.rs`
- Modify: `internal/tenferro-internal-ad-surface/src/core/dynamic/tensor.rs`
- Modify: `tenferro/src/lib.rs`
- Modify: `tenferro/README.md`

**Step 1: Replace the old two-field result type**

Add a structured result with:

- `solution`
- `residuals`
- `rank`
- `singular_values`

**Step 2: Make `residuals` match oracle / PyTorch semantics**

Return squared residual summaries as real-valued tensors and allow empty tensors when required.

**Step 3: Update schemas**

Mark differentiability as:

- `solution`: true
- `residuals`: true
- `rank`: false
- `singular_values`: false

**Step 4: Run public tests**

Run:

```bash
cargo test -p tenferro --test integration --release -- --nocapture
```

Expected: failures shift from result shape to AD rule mismatches.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/result_types/decomposition.rs tenferro-linalg/src/result_types/cotangents.rs tenferro-linalg/src/primal/least_squares.rs internal/tenferro-internal-ad-linalg/src/linearized.rs internal/tenferro-internal-ad-surface/src/core/dynamic/tensor.rs tenferro/src/lib.rs tenferro/README.md
git commit -m "refactor: align lstsq result contract with pytorch"
```

### Task 10: Implement `lstsq` AD against the new contract

**Files:**
- Modify: `tenferro-linalg/src/frules/least_squares.rs`
- Modify: `tenferro-linalg/src/rrules/least_squares.rs`
- Modify: `tenferro-linalg/tests/oracle_db/replay.rs`
- Modify: `tenferro-linalg/tests/oracle_db/support.rs`
- Modify: `internal/tenferro-internal-ad-linalg/src/linearized.rs`
- Modify: `internal/tenferro-internal-ad-linalg/tests/dyn_linalg_ops.rs`

**Step 1: Add failing seam tests for `solution` and `residuals`**

Cover both JVP and VJP, including empty residual behavior.

**Step 2: Implement minimal `frule/rrule` support**

Use the oracle/PyTorch-style residual contract rather than the old raw residual tensor.

**Step 3: Update `LinearizedOp` packaging**

Carry the four-output schema through the runtime seam.

**Step 4: Run replay + seam + public tests**

Run:

```bash
cargo test -p tenferro-linalg --release oracle_db_replay_against_tensor_ad_oracles -- --nocapture
cargo test -p tenferro-internal-ad-linalg --release dyn_linalg_ops -- --nocapture
cargo test -p tenferro --test integration --release -- --nocapture
```

**Step 5: Commit**

```bash
git add tenferro-linalg/src/frules/least_squares.rs tenferro-linalg/src/rrules/least_squares.rs tenferro-linalg/tests/oracle_db/replay.rs tenferro-linalg/tests/oracle_db/support.rs internal/tenferro-internal-ad-linalg/src/linearized.rs internal/tenferro-internal-ad-linalg/tests/dyn_linalg_ops.rs
git commit -m "feat: support pytorch-aligned lstsq ad"
```

### Task 11: Final verification and docs sweep

**Files:**
- Modify as needed: `tenferro/README.md`
- Modify as needed: rustdoc-bearing public modules

**Step 1: Check docs for public-surface drift**

Verify README, rustdoc, and examples do not mention removed names or outdated contracts.

**Step 2: Run final verification**

Run:

```bash
cargo fmt --all
cargo test -p tenferro-linalg --release oracle_db_replay_against_tensor_ad_oracles -- --nocapture
cargo test -p tenferro-internal-ad-linalg --release
cargo test -p tenferro --test integration --release
cargo test -p tenferro --doc --release
cargo check -p tenferro --tests --release
```

Expected: all updated operator families pass.

**Step 3: Commit**

```bash
git add tenferro/README.md
git commit -m "docs: finalize pytorch-aligned ad surface"
```
