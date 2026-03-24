# Integer/Bool Metadata Substrate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a PyTorch-aligned integer/bool metadata substrate so LU pivots, LU info, determinant parity, `lu_solve`, and later rank/mask-style linalg paths stay tensor-native on CPU and CUDA.

**Architecture:** Follow the ATen shape of the solution rather than its exact layering. In PyTorch, `pivots` and `info` are tensors, and higher-level linalg code composes them with generic tensor ops such as `arange`, comparisons, `where`, `sum`, and broadcast/expand. In tenferro, the equivalent should be a dedicated metadata family plus low-level runtime support for integer/bool tensors, then LU contracts should move from host `Vec<i32>` to tensor metadata, and only after that should public LU surfaces be changed to match PyTorch.

**Tech Stack:** `tenferro-device` CUDA runtime, `tenferro-tensor` metadata tensor support, `tenferro-prims` family/bridge layer, `tenferro-linalg-prims` LU backend contracts, `tenferro-linalg` public/composite APIs, local `../pytorch` references.

---

### Task 1: Freeze the metadata-substrate contract

**Files:**
- Create: `docs/plans/2026-03-23-integer-bool-metadata-substrate-design.md`
- Modify: `tenferro-prims/src/lib.rs`
- Modify: `tenferro-prims/src/families/mod.rs`
- Create: `tenferro-prims/src/families/metadata.rs`
- Test: `tenferro-prims/src/tests/metadata_contract_phase1.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn metadata_family_exposes_i32_bool_comparison_where_and_sum_descriptors() {
    use tenferro_prims::{
        MetadataBinaryOp, MetadataReductionOp, MetadataTernaryOp, MetadataUnaryOp,
    };

    assert_eq!(MetadataUnaryOp::IotaStartZero as u8, MetadataUnaryOp::IotaStartZero as u8);
    assert_eq!(MetadataBinaryOp::NotEqual as u8, MetadataBinaryOp::NotEqual as u8);
    assert_eq!(MetadataTernaryOp::Where as u8, MetadataTernaryOp::Where as u8);
    assert_eq!(MetadataReductionOp::Sum as u8, MetadataReductionOp::Sum as u8);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p tenferro-prims --lib metadata_contract_phase1`
Expected: FAIL because the metadata family surface does not exist yet.

**Step 3: Write minimal implementation**

- Add a new metadata-family module in `tenferro-prims` with:
  - `MetadataUnaryOp`
  - `MetadataBinaryOp`
  - `MetadataTernaryOp`
  - `MetadataReductionOp`
  - `MetadataPrimsDescriptor`
  - `TensorMetadataContextFor`
  - `TensorMetadataPrims`
- Scope the first tranche to the closed set needed by LU/det/slogdet:
  - unary: `IotaStartZero`
  - binary: `Equal`, `NotEqual`, `Add`, `Sub`, `Mul`, `BitAnd`
  - ternary: `Where`
  - reduction: `Sum`, `All`, `Any`
- Document explicitly that this family is for integer/bool metadata tensors, not general numeric algebra.

**Step 4: Run test to verify it passes**

Run: `cargo test -p tenferro-prims --lib metadata_contract_phase1`
Expected: PASS.

**Step 5: Commit**

```bash
git add docs/plans/2026-03-23-integer-bool-metadata-substrate-design.md \
        tenferro-prims/src/lib.rs \
        tenferro-prims/src/families/mod.rs \
        tenferro-prims/src/families/metadata.rs \
        tenferro-prims/src/tests/metadata_contract_phase1.rs
git commit -m "feat: define metadata tensor family contracts"
```

### Task 2: Add Layer 0 CUDA runtime support for metadata tensors

**Files:**
- Modify: `tenferro-device/src/cuda/runtime.rs`
- Modify: `tenferro-device/src/cuda/runtime/shared.rs`
- Modify: `tenferro-device/src/cuda/runtime/pointwise.rs`
- Create: `tenferro-device/src/cuda/runtime/pointwise/pointwise_metadata.rs`
- Modify: `tenferro-device/src/cuda/runtime/kernels.rs`
- Create: `tenferro-device/src/cuda/runtime/kernels/metadata_scalar.rs`
- Test: `tenferro-device/src/cuda/tests/mod.rs`

**Step 1: Write the failing tests**

```rust
#[test]
fn cuda_runtime_metadata_iota_i32_matches_host_reference() { /* ... */ }

#[test]
fn cuda_runtime_metadata_not_equal_and_sum_match_host_reference() { /* ... */ }

#[test]
fn cuda_runtime_metadata_where_selects_integer_values() { /* ... */ }
```

**Step 2: Run tests to verify they fail**

Run: `TENFERRO_TEST_CUDA=1 cargo test -p tenferro-device --features cuda --lib cuda::tests::cuda_runtime_metadata_iota_i32_matches_host_reference -- --exact`
Expected: FAIL because metadata kernels and entrypoints do not exist.

**Step 3: Write minimal implementation**

- Add runtime dtype support for:
  - `i32`
  - `bool` or a documented `u8`-backed bool-like representation
- Add metadata pointwise/reduction launchers for:
  - `iota`
  - `eq` / `ne`
  - `where`
  - `sum`
  - `all` / `any`
- Keep the module split strict:
  - metadata kernels live in `kernels/metadata_scalar.rs`
  - metadata runtime entrypoints live in `pointwise/pointwise_metadata.rs`
- Do not widen the public runtime API accidentally; keep helper re-exports internal.

**Step 4: Run tests to verify they pass**

Run: `TENFERRO_TEST_CUDA=1 cargo test -p tenferro-device --features cuda --lib`
Expected: PASS with new metadata runtime tests included.

**Step 5: Commit**

```bash
git add tenferro-device/src/cuda/runtime.rs \
        tenferro-device/src/cuda/runtime/shared.rs \
        tenferro-device/src/cuda/runtime/pointwise.rs \
        tenferro-device/src/cuda/runtime/pointwise/pointwise_metadata.rs \
        tenferro-device/src/cuda/runtime/kernels.rs \
        tenferro-device/src/cuda/runtime/kernels/metadata_scalar.rs \
        tenferro-device/src/cuda/tests/mod.rs
git commit -m "feat: add cuda metadata runtime support"
```

### Task 3: Wire metadata family through CPU and CUDA prims

**Files:**
- Modify: `tenferro-prims/src/cpu/mod.rs`
- Create: `tenferro-prims/src/cpu/metadata.rs`
- Modify: `tenferro-prims/src/cuda/mod.rs`
- Create: `tenferro-prims/src/cuda/metadata.rs`
- Modify: `tenferro-prims/src/families/context.rs`
- Modify: `tenferro-prims/src/tests/mod.rs`
- Create: `tenferro-prims/src/tests/metadata_phase1.rs`

**Step 1: Write the failing tests**

```rust
#[test]
fn cpu_metadata_family_builds_lu_det_parity_primitives() { /* ... */ }

#[test]
fn cuda_metadata_family_builds_lu_det_parity_primitives() { /* ... */ }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p tenferro-prims --lib metadata_phase1`
Expected: FAIL because the metadata family is not wired into CPU/CUDA backends.

**Step 3: Write minimal implementation**

- Implement CPU metadata execution with simple loops over contiguous logical values.
- Implement CUDA metadata execution as a thin wrapper over the Layer 0 runtime added in Task 2.
- Add context bridge traits so `tenferro-linalg` can call metadata ops without importing backend-specific names.
- Keep bool metadata semantics explicit in docs:
  - no implicit arithmetic outside the supported op set
  - `sum` returns an integer tensor

**Step 4: Run tests to verify they pass**

Run:
- `cargo test -p tenferro-prims --lib metadata_phase1`
- `TENFERRO_TEST_CUDA=1 cargo test -p tenferro-prims --features cuda --lib`
Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-prims/src/cpu/mod.rs \
        tenferro-prims/src/cpu/metadata.rs \
        tenferro-prims/src/cuda/mod.rs \
        tenferro-prims/src/cuda/metadata.rs \
        tenferro-prims/src/families/context.rs \
        tenferro-prims/src/tests/mod.rs \
        tenferro-prims/src/tests/metadata_phase1.rs
git commit -m "feat: wire metadata tensor prims"
```

### Task 4: Convert LU backend contracts to tensor metadata

**Files:**
- Modify: `tenferro-linalg-prims/src/lib.rs`
- Modify: `tenferro-linalg-prims/src/backend/tests/mod.rs`
- Modify: `tenferro-linalg-prims/src/backend/cuda/tests/mod.rs`
- Modify: `tenferro-linalg/src/backend/tensor_api/tests/mod.rs`

**Step 1: Write the failing tests**

```rust
#[test]
fn lu_tensor_result_exposes_pivots_as_int_tensor() { /* ... */ }

#[test]
fn lu_tensor_ex_result_exposes_info_as_int_tensor() { /* ... */ }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p tenferro-linalg-prims --lib lu_tensor_result_exposes_pivots_as_int_tensor -- --exact`
Expected: FAIL because LU contracts still expose host vectors.

**Step 3: Write minimal implementation**

- Change `LuTensorResult` to:
  - `l: Tensor<T>`
  - `u: Tensor<T>`
  - `pivots: Tensor<i32>`
- Change `LuTensorExResult` to:
  - `l: Tensor<T>`
  - `u: Tensor<T>`
  - `pivots: Tensor<i32>`
  - `info: Tensor<i32>`
- Update CPU/CUDA backend implementations and tests to materialize pivots/info on the same device as the factors.
- Preserve 1-indexed pivot semantics to match PyTorch docs.

**Step 4: Run tests to verify they pass**

Run:
- `cargo test -p tenferro-linalg-prims --lib`
- `TENFERRO_TEST_CUDA=1 cargo test -p tenferro-linalg-prims --features cuda --lib`
Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg-prims/src/lib.rs \
        tenferro-linalg-prims/src/backend/tests/mod.rs \
        tenferro-linalg-prims/src/backend/cuda/tests/mod.rs \
        tenferro-linalg/src/backend/tensor_api/tests/mod.rs
git commit -m "feat: tensorize lu pivots and info"
```

### Task 5: Rewrite det/slogdet/lu_solve to consume metadata tensors

**Files:**
- Modify: `tenferro-linalg/src/prims_bridge.rs`
- Modify: `tenferro-linalg/src/primal/linear_systems.rs`
- Modify: `tenferro-linalg/src/tests/runtime_capability.rs`
- Modify: `tenferro-linalg/src/tests/batch_a_contracts.rs`
- Modify: `tenferro-linalg/src/tests/batch_b_contracts.rs`

**Step 1: Write the failing tests**

```rust
#[test]
fn det_section_does_not_build_host_sign_tensor() { /* source-level guard */ }

#[test]
fn slogdet_section_does_not_build_host_sign_tensor() { /* source-level guard */ }

#[test]
fn lu_solve_accepts_tensor_pivots_and_matches_cpu_cuda() { /* parity test */ }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p tenferro-linalg --lib runtime_capability::det_section_does_not_build_host_sign_tensor -- --exact`
Expected: FAIL because `linear_systems.rs` still uses `sign_data`, `tensor_from_data`, and host pivot conversion.

**Step 3: Write minimal implementation**

- Add bridge helpers that compose PyTorch-style metadata paths:
  - `arange`
  - `ne`
  - `sum`
  - `where`
- Replace:
  - `backend_pivots_to_usize`
  - `permutation_sign_from_forward_pivots`
  - host `sign_data`
  with tensor-native metadata composition.
- Update `lu_solve` to accept pivot tensors and use metadata tensor broadcast/expand semantics instead of host `&[usize]`.

**Step 4: Run tests to verify they pass**

Run:
- `cargo test -p tenferro-linalg --lib`
- `cargo test -p tenferro-linalg --features cuda --lib --no-run`
Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/prims_bridge.rs \
        tenferro-linalg/src/primal/linear_systems.rs \
        tenferro-linalg/src/tests/runtime_capability.rs \
        tenferro-linalg/src/tests/batch_a_contracts.rs \
        tenferro-linalg/src/tests/batch_b_contracts.rs
git commit -m "refactor: make lu metadata paths tensor-native"
```

### Task 6: Align public LU APIs with PyTorch

**Files:**
- Modify: `tenferro-linalg/src/result_types/status.rs`
- Modify: `tenferro-linalg/src/result_types/decomposition.rs`
- Modify: `tenferro-linalg/src/primal/decompositions.rs`
- Modify: `tenferro-linalg/src/primal/linear_systems.rs`
- Modify: `tenferro-linalg/tests/linalg_tests.rs`
- Modify: `docs/design/supported-ops.md`

**Step 1: Write the failing tests**

```rust
#[test]
fn lu_factor_public_result_exposes_tensor_pivots() { /* ... */ }

#[test]
fn lu_factor_ex_public_result_exposes_tensor_info() { /* ... */ }

#[test]
fn lu_public_surface_matches_pytorch_permutation_tensor_semantics() { /* ... */ }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p tenferro-linalg --lib lu_factor_public_result_exposes_tensor_pivots -- --exact`
Expected: FAIL because public LU results still expose `Vec<usize>` / `Vec<i32>`.

**Step 3: Write minimal implementation**

- Change `LuFactorResult` to expose `pivots: Tensor<i32>`.
- Change `LuFactorExResult` to expose:
  - `pivots: Tensor<i32>`
  - `info: Tensor<i32>`
- Change `LuResult` to move toward PyTorch semantics:
  - canonical `P: Tensor<T>` when pivoting is requested
  - empty tensor or documented `None` shim only if strictly needed for Rust ergonomics
- Update docs and examples to describe:
  - 1-indexed pivot vector semantics
  - tensor shape `(*, n)`
  - CUDA `pivot=false` behavior

**Step 4: Run tests to verify they pass**

Run:
- `cargo test -p tenferro-linalg --lib`
- `cargo test -p tenferro-linalg --features cuda --lib --no-run`
Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/result_types/status.rs \
        tenferro-linalg/src/result_types/decomposition.rs \
        tenferro-linalg/src/primal/decompositions.rs \
        tenferro-linalg/src/primal/linear_systems.rs \
        tenferro-linalg/tests/linalg_tests.rs \
        docs/design/supported-ops.md
git commit -m "feat: align public lu metadata with pytorch"
```

### Task 7: Run the full gate and record follow-ups

**Files:**
- Modify: `docs/design/supported-ops.md`
- Modify: `docs/plans/2026-03-23-integer-bool-metadata-substrate-design.md`

**Step 1: Run focused crate verification**

Run:
- `cargo fmt --all --check`
- `TENFERRO_TEST_CUDA=1 cargo test -p tenferro-device --features cuda --lib`
- `TENFERRO_TEST_CUDA=1 cargo test -p tenferro-prims --features cuda --lib`
- `TENFERRO_TEST_CUDA=1 cargo test -p tenferro-linalg-prims --features cuda --lib`
- `cargo test -p tenferro-linalg --lib`
- `cargo test -p tenferro-linalg --features cuda --lib --no-run`

Expected: PASS.

**Step 2: Run workspace gate**

Run:
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`

Expected: PASS.

**Step 3: Record explicit follow-ups**

- If `bool` lands as `u8`-backed internally, record whether a true `Tensor<bool>` public type is still needed.
- Record whether `ldl` and rank/mask paths can reuse the same metadata family immediately.

**Step 4: Commit**

```bash
git add docs/design/supported-ops.md \
        docs/plans/2026-03-23-integer-bool-metadata-substrate-design.md
git commit -m "docs: record metadata substrate verification"
```
