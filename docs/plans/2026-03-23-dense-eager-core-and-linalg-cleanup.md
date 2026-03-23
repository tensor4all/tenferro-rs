# Dense Eager Core Compatibility And Linalg Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an ATen-style dense eager core for tenferro's bool/int/real/complex tensors, then remove the remaining host-side `torch.linalg` cleanup paths by rewriting them onto that substrate.

**Architecture:** Do not continue adding op-specific linalg helpers. First widen the low-level dense eager substrate in `tenferro-device`, `tenferro-tensor`, and `tenferro-prims` so integer/bool metadata tensors, real/complex scalar tensors, and mixed-dtype cast/select paths can be composed without host reconstruction. Then migrate `tenferro-linalg` and `tenferro-linalg-prims` off temporary `Vec<i32>` / CPU-copy bridges and onto tensor-native metadata paths.

**Tech Stack:** Rust workspace crates `tenferro-device`, `tenferro-tensor`, `tenferro-prims`, `tenferro-linalg-prims`, `tenferro-linalg`; CUDA runtime-loaded kernels; `strided-view`; existing family `plan/execute` protocols.

---

## Current Baseline

- This plan assumes work starts from branch `feat/complex-real-unary-substrate` at or after commit `e7683d8` (`fix: keep lu pivot metadata 1-indexed`).
- Already complete on this branch:
  - logical copy / `resolve_conj` substrate
  - GPU `cat` / `stack`
  - complex-real unary substrate
  - metadata family phase 1
  - `tenferro-linalg-prims` LU metadata tensorization
  - LU pivot semantics aligned to PyTorch-style 1-indexed step pivots with shape `min(m, n)`
- Known blocker at this stop point:
  - `tenferro-linalg/src/primal/linear_systems.rs` still reconstructs determinant/sign metadata on the host using:
    - `permutation_sign_from_forward_pivots(...)`
    - `backend_pivots_to_forward_perm(...)`
    - `sign_data`
    - `tensor_from_data(...)`
  - This cleanup is blocked because metadata phase 1 does not yet provide the integer/bool arithmetic and mixed-dtype cast/select surface needed to express the parity/sign path tensor-natively.

## Target Compatibility Contract

This plan targets **ATen dense eager core compatibility**, not full PyTorch parity.

### In Scope

- Dense tensor creation and materialization needed by linalg and decomposition call paths
- Dense pointwise ops across:
  - bool metadata
  - `i32` metadata
  - `f32`
  - `f64`
  - `Complex32`
  - `Complex64`
- Broadcast, shape/view, contiguous/materialize, `cat`, `stack`
- Mixed-dtype cast/select bridges needed to move between:
  - bool metadata and scalar tensors
  - `i32` metadata and scalar tensors
- Dense reductions needed by `torch.linalg`-style composition
- Tensor-native LU/solve/determinant metadata composition
- Public `lu_factor`, `lu_factor_ex`, and `lu_solve` surface alignment toward PyTorch-style pivot tensors

### Explicitly Out Of Scope

- Sparse tensors
- Quantized tensors
- Named tensors
- Nested tensors
- PyTorch dispatcher / operator registry / JIT / serialization
- Neural-network-specific operators
- RNG/random sampling APIs
- Sorting/top-k/indexing families not required by the linalg cleanup closure

## Design Rules

- Prefer substrate-first changes over linalg-specific helpers.
- Do not keep temporary host bridges once tensor-native paths exist.
- Do not add new CPU fallbacks in `tenferro-linalg`.
- Preserve the current layering:
  - Layer 0: `tenferro-device`
  - Layer 2: `tenferro-tensor`
  - Layer 3: `tenferro-prims`, `tenferro-linalg-prims`
  - Layer 4: `tenferro-linalg`
- Use TDD for every task.
- Keep files small; split runtime and family modules when they approach the repo size guideline.

## Dense Eager Core Work Packages

### Dense Eager Core Operator Matrix

The first complete dense-eager tranche should support the following operator classes.

| Class | Bool Metadata | I32 Metadata | Real Scalar | Complex Scalar |
| --- | --- | --- | --- | --- |
| Generate | `iota` | `iota` | existing constructors | existing constructors |
| Unary | `logical_not` later if needed | `neg` later if needed | existing scalar/analytic set | existing scalar/analytic set |
| Binary compare | `eq`, `ne` | `eq`, `ne` | existing scalar compare | via existing complex-real or explicit real compare path |
| Binary arithmetic | `bitand` | `add`, `sub`, `mul` | existing | existing / complex-scale |
| Ternary select | `where(metadata -> metadata)` | `where(metadata -> metadata)` | existing scalar `where` | existing scalar/complex-scale plus bridge |
| Reduction | `all`, `any`, `sum(bool->i32)` | `sum(i32->i32)` | existing | existing |
| Cast / bridge | bool -> scalar | i32 -> scalar | scalar -> scalar | scalar -> scalar |

The missing pieces at this stop point are:

- metadata binary:
  - `Add`
  - `Sub`
  - `Mul`
  - `BitAnd`
- metadata-to-scalar cast / bridge:
  - bool metadata -> scalar same shape
  - `i32` metadata -> scalar same shape

## Phase A: Complete The Dense Eager Core

### Task 1: Add metadata phase-2 CPU/CUDA regression tests

**Files:**
- Create: `tenferro-prims/src/tests/metadata_phase2.rs`
- Modify: `tenferro-prims/src/tests/mod.rs`

**Step 1: Write the failing tests**

Add generic CPU/CUDA tests covering:

```rust
fn run_metadata_phase2<C>(ctx: &mut C, memory_space: LogicalMemorySpace)
where
    C: TensorMetadataContextFor,
    C::MetadataBackend: TensorMetadataPrims<Context = C>,
{
    // i32 add/sub/mul
    // bool bitand
    // shape/broadcast sanity
}

#[test]
fn cpu_metadata_phase2_i32_arithmetic_matches_host_reference() { /* ... */ }

#[test]
fn cpu_metadata_phase2_bool_bitand_matches_host_reference() { /* ... */ }

#[cfg(feature = "cuda")]
#[test]
fn cuda_metadata_phase2_i32_arithmetic_matches_host_reference() { /* ... */ }

#[cfg(feature = "cuda")]
#[test]
fn cuda_metadata_phase2_bool_bitand_matches_host_reference() { /* ... */ }
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-prims --lib metadata_phase2
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-prims --features cuda --lib metadata_phase2
```

Expected: FAIL because phase-2 metadata binary ops are not wired.

**Step 3: Write minimal implementation**

None in this task. RED only.

**Step 4: Re-run to confirm the failure is still the intended missing-op failure**

Expected: FAIL for unsupported metadata binary ops.

**Step 5: Commit**

```bash
git add tenferro-prims/src/tests/mod.rs \
        tenferro-prims/src/tests/metadata_phase2.rs
git commit -m "test: add metadata phase 2 regressions"
```

### Task 2: Implement metadata phase-2 binary ops on CPU

**Files:**
- Modify: `tenferro-prims/src/cpu/metadata.rs`

**Step 1: Reuse the failing tests from Task 1**

No new tests.

**Step 2: Run the focused CPU test**

Run:

```bash
cargo test -p tenferro-prims --lib metadata_phase2
```

Expected: FAIL on CPU path.

**Step 3: Write minimal implementation**

- Extend supported binary validation to include:
  - `(I32, I32) -> I32` for `Add/Sub/Mul`
  - `(Bool, Bool) -> Bool` for `BitAnd`
- Implement the execution loops with no host staging beyond existing CPU contiguous views.
- Do not add new op-specific helpers outside `cpu/metadata.rs` unless the same logic is reused more than once.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-prims --lib metadata_phase2
```

Expected: PASS on CPU coverage; CUDA tests still fail or remain unimplemented.

**Step 5: Commit**

```bash
git add tenferro-prims/src/cpu/metadata.rs
git commit -m "feat: add cpu metadata phase 2 binary ops"
```

### Task 3: Implement metadata phase-2 binary ops on CUDA

**Files:**
- Modify: `tenferro-device/src/cuda/runtime/pointwise/pointwise_metadata.rs`
- Modify: `tenferro-device/src/cuda/runtime/kernels/metadata_scalar.rs`
- Modify: `tenferro-prims/src/cuda/metadata.rs`

**Step 1: Reuse the failing CUDA tests from Task 1**

No new tests.

**Step 2: Run the focused CUDA test**

Run:

```bash
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-prims --features cuda --lib metadata_phase2
```

Expected: FAIL on missing CUDA support for phase-2 metadata binary ops.

**Step 3: Write minimal implementation**

- Add Layer 0 kernels and runtime entrypoints for:
  - `i32 add`
  - `i32 sub`
  - `i32 mul`
  - `bool bitand`
- Wire `tenferro-prims/src/cuda/metadata.rs` support checks, plans, and execute arms.
- Keep module split strict:
  - kernels in `metadata_scalar.rs`
  - runtime dispatch in `pointwise_metadata.rs`
- Follow the same dtype/storage conventions already used in phase 1.

**Step 4: Run tests to verify they pass**

Run:

```bash
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-prims --features cuda --lib metadata_phase2
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-device/src/cuda/runtime/pointwise/pointwise_metadata.rs \
        tenferro-device/src/cuda/runtime/kernels/metadata_scalar.rs \
        tenferro-prims/src/cuda/metadata.rs
git commit -m "feat: add cuda metadata phase 2 binary ops"
```

### Task 4: Define a dense cast/bridge family for metadata -> scalar tensors

**Files:**
- Create: `tenferro-prims/src/families/cast.rs`
- Modify: `tenferro-prims/src/families/mod.rs`
- Modify: `tenferro-prims/src/families/context.rs`
- Modify: `tenferro-prims/src/lib.rs`
- Create: `tenferro-prims/src/tests/cast_phase1.rs`

**Step 1: Write the failing tests**

Add tests for:

```rust
#[test]
fn cpu_cast_phase1_bool_metadata_to_real_mask_matches_host_reference() { /* ... */ }

#[test]
fn cpu_cast_phase1_i32_metadata_to_real_matches_host_reference() { /* ... */ }

#[cfg(feature = "cuda")]
#[test]
fn cuda_cast_phase1_bool_metadata_to_real_mask_matches_host_reference() { /* ... */ }
```

The first minimal phase should only require:

- `Bool metadata -> Tensor<f32/f64/Complex32/Complex64>`
- `I32 metadata -> Tensor<f32/f64>`

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-prims --lib cast_phase1
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-prims --features cuda --lib cast_phase1
```

Expected: FAIL because no mixed-dtype cast family exists.

**Step 3: Write minimal implementation**

Add a new family rather than overloading `TensorScalarPrims`:

- `TensorCastContextFor`
- `TensorCastPrims`
- descriptors sufficient for same-shape cast

Do not redesign the existing scalar family to take mixed-dtype inputs.

**Step 4: Run tests to verify they pass**

Run the same `cast_phase1` commands.

Expected: PASS on CPU first, then CUDA.

**Step 5: Commit**

```bash
git add tenferro-prims/src/families/cast.rs \
        tenferro-prims/src/families/mod.rs \
        tenferro-prims/src/families/context.rs \
        tenferro-prims/src/lib.rs \
        tenferro-prims/src/tests/cast_phase1.rs
git commit -m "feat: define dense cast family phase 1"
```

### Task 5: Implement CPU/CUDA cast family phase 1

**Files:**
- Create: `tenferro-prims/src/cpu/cast.rs`
- Modify: `tenferro-prims/src/cpu/mod.rs`
- Create: `tenferro-prims/src/cuda/cast.rs`
- Modify: `tenferro-prims/src/cuda/mod.rs`

**Step 1: Reuse the failing tests from Task 4**

No new tests.

**Step 2: Run the focused tests**

Run:

```bash
cargo test -p tenferro-prims --lib cast_phase1
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-prims --features cuda --lib cast_phase1
```

Expected: FAIL.

**Step 3: Write minimal implementation**

- CPU:
  - simple contiguous loops are fine
- CUDA:
  - use Layer 0 pointwise cast kernels
  - add modules if `pointwise_real.rs` or `pointwise_complex.rs` need further splitting
- Keep cast semantics explicit:
  - bool `0 -> 0`
  - bool nonzero `-> 1`
  - `i32 -> real` exact representable conversion only as normal Rust cast semantics for current dtypes

**Step 4: Run tests to verify they pass**

Run the same `cast_phase1` commands.

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-prims/src/cpu/cast.rs \
        tenferro-prims/src/cpu/mod.rs \
        tenferro-prims/src/cuda/cast.rs \
        tenferro-prims/src/cuda/mod.rs
git commit -m "feat: implement dense cast family phase 1"
```

### Task 6: Add `tenferro-linalg` bridges for metadata and cast families

**Files:**
- Modify: `tenferro-linalg/src/prims_bridge.rs`
- Modify: `tenferro-linalg/src/tests/runtime_capability.rs`
- Modify: `tenferro-linalg/src/tests/mod.rs`

**Step 1: Write the failing tests**

Add source-level and behavior tests for:

```rust
#[test]
fn metadata_bridge_uses_metadata_family_instead_of_host_vecs() { /* source guard */ }

#[test]
fn cast_bridge_materializes_bool_metadata_masks_without_host_extraction() { /* source guard */ }
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg --lib runtime_capability::metadata_bridge_uses_metadata_family_instead_of_host_vecs -- --exact
```

Expected: FAIL because the bridge helpers do not exist yet.

**Step 3: Write minimal implementation**

Add thin bridge helpers for:

- metadata `iota`
- metadata binary
- metadata reduction
- metadata `where`
- cast same-shape

Do not put linalg-specific logic in these helpers.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-linalg --lib runtime_capability::metadata_bridge_uses_metadata_family_instead_of_host_vecs -- --exact
cargo test -p tenferro-linalg --lib runtime_capability::cast_bridge_materializes_bool_metadata_masks_without_host_extraction -- --exact
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/prims_bridge.rs \
        tenferro-linalg/src/tests/runtime_capability.rs \
        tenferro-linalg/src/tests/mod.rs
git commit -m "feat: add dense metadata and cast bridges"
```

## Phase B: Linalg Cleanup On Top Of The Dense Eager Core

### Task 7: Rewrite `det` to a tensor-native metadata parity path

**Files:**
- Modify: `tenferro-linalg/src/primal/linear_systems.rs`
- Modify: `tenferro-linalg/src/tests/runtime_capability.rs`
- Modify: `tenferro-linalg/src/tests/batch_a_contracts.rs`

**Step 1: Write the failing tests**

Add or strengthen:

```rust
#[test]
fn det_section_does_not_build_host_sign_tensor() { /* source guard */ }

#[test]
fn det_matches_expected_sign_for_batched_row_swaps() { /* behavior */ }
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg --lib runtime_capability::det_section_does_not_build_host_sign_tensor -- --exact
```

Expected: FAIL because `det` still uses `sign_data`, `tensor_from_data`, and host pivot reconstruction.

**Step 3: Write minimal implementation**

Replace host pivot/sign handling with a tensor-native metadata path:

- build step index via metadata `iota`
- compare step index to pivot tensor
- derive swap mask / parity through metadata arithmetic
- cast metadata parity to real scalar sign tensor
- scale diagonal product tensor-natively

Delete:

- `permutation_sign_from_forward_pivots(...)`
- host `sign_data` path in `det`

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-linalg --lib det_matches_expected_sign_for_batched_row_swaps
cargo test -p tenferro-linalg --lib runtime_capability::det_section_does_not_build_host_sign_tensor -- --exact
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/primal/linear_systems.rs \
        tenferro-linalg/src/tests/runtime_capability.rs \
        tenferro-linalg/src/tests/batch_a_contracts.rs
git commit -m "refactor: make det use tensor-native metadata parity"
```

### Task 8: Rewrite real `slogdet` to the same metadata parity path

**Files:**
- Modify: `tenferro-linalg/src/primal/linear_systems.rs`
- Modify: `tenferro-linalg/src/tests/runtime_capability.rs`
- Modify: `tenferro-linalg/src/tests/mod.rs`

**Step 1: Write the failing tests**

Add or strengthen:

```rust
#[test]
fn slogdet_section_does_not_build_host_sign_tensor() { /* source guard */ }
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg --lib runtime_capability::slogdet_section_does_not_build_host_sign_tensor -- --exact
```

Expected: FAIL because real `slogdet` still builds host `sign_data`.

**Step 3: Write minimal implementation**

- Reuse the same metadata parity path as `det`
- Keep diagonal-sign-from-data logic tensor-native
- Remove host pivot reconstruction from real `slogdet`

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-linalg --lib slogdet_matches_expected_sign_and_logabsdet
cargo test -p tenferro-linalg --lib runtime_capability::slogdet_section_does_not_build_host_sign_tensor -- --exact
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/primal/linear_systems.rs \
        tenferro-linalg/src/tests/runtime_capability.rs \
        tenferro-linalg/src/tests/mod.rs
git commit -m "refactor: make real slogdet use tensor-native metadata parity"
```

### Task 9: Rewrite complex `slogdet` to the same metadata parity path

**Files:**
- Modify: `tenferro-linalg/src/primal/linear_systems.rs`
- Modify: `tenferro-linalg/src/tests/runtime_capability.rs`
- Modify: `tenferro-linalg/src/tests/mod.rs`

**Step 1: Write the failing tests**

Strengthen:

```rust
#[test]
fn complex_slogdet_section_does_not_build_host_sign_tensor() { /* source guard */ }
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg --lib runtime_capability::complex_slogdet_path_uses_complex_real_and_complex_scale_without_slice_bridge -- --exact
```

Expected: FAIL until the pivot sign path is tensor-native.

**Step 3: Write minimal implementation**

- Reuse metadata parity
- cast parity to the appropriate real sign tensor
- use the existing complex scale / complex-real bridges for the remaining math

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-linalg --lib runtime_capability::complex_slogdet_path_uses_complex_real_and_complex_scale_without_slice_bridge -- --exact
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/primal/linear_systems.rs \
        tenferro-linalg/src/tests/runtime_capability.rs \
        tenferro-linalg/src/tests/mod.rs
git commit -m "refactor: make complex slogdet use tensor-native metadata parity"
```

### Task 10: Align the public LU surface toward PyTorch

**Files:**
- Modify: `tenferro-linalg/src/result_types/status.rs`
- Modify: `tenferro-linalg/src/primal/decompositions.rs`
- Modify: `tenferro-linalg/src/primal/linear_systems.rs`
- Modify: `tenferro-linalg/src/tests/batch_a_contracts.rs`
- Modify: `tenferro-linalg/src/tests/batch_b_contracts.rs`
- Modify: relevant rustdoc examples

**Step 1: Write the failing tests**

Add or update:

```rust
#[test]
fn lu_factor_public_surface_exposes_pivot_tensor() { /* ... */ }

#[test]
fn lu_factor_ex_public_surface_exposes_pivot_and_info_tensors() { /* ... */ }

#[test]
fn lu_solve_accepts_pivot_tensor() { /* ... */ }
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg --lib lu_factor_public_surface_exposes_pivot_tensor
```

Expected: FAIL because the public surface still uses `Vec<usize>` / `Vec<i32>`.

**Step 3: Write minimal implementation**

Move the public surface toward PyTorch:

- `LuFactorResult.pivots -> Tensor<i32>`
- `LuFactorExResult.pivots -> Tensor<i32>`
- `LuFactorExResult.info -> Tensor<i32>`
- `lu_solve(..., pivots: &Tensor<i32>, ...)`

Clean up all call sites instead of preserving old shims.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-linalg --lib
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/result_types/status.rs \
        tenferro-linalg/src/primal/decompositions.rs \
        tenferro-linalg/src/primal/linear_systems.rs \
        tenferro-linalg/src/tests/batch_a_contracts.rs \
        tenferro-linalg/src/tests/batch_b_contracts.rs
git commit -m "feat: align public lu metadata with tensor pivots"
```

### Task 11: Delete temporary host metadata bridges and complete runtime guards

**Files:**
- Modify: `tenferro-linalg/src/backend/tensor_helpers.rs`
- Modify: `tenferro-linalg/src/tests/runtime_capability.rs`

**Step 1: Write the failing tests**

Add source-level guards that forbid:

- `backend_info_to_vec`
- `backend_pivots_to_forward_perm`
- host `sign_data`

inside the `det` / `slogdet` / `lu_solve` sections.

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg --lib runtime_capability::det_section_does_not_build_host_sign_tensor -- --exact
```

Expected: FAIL until the old helper paths are removed.

**Step 3: Write minimal implementation**

- Delete the temporary host bridge helpers that are no longer needed
- Keep only reusable tensor-native backend helpers

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-linalg --lib runtime_capability
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/backend/tensor_helpers.rs \
        tenferro-linalg/src/tests/runtime_capability.rs
git commit -m "refactor: remove temporary lu metadata host bridges"
```

## Full Verification Gate Before PR

After all tasks above are green, rerun the full repo gate from the worktree root:

```bash
cargo fmt --all --check
cargo nextest run --workspace --release --no-fail-fast
cargo test --doc --workspace --release
cargo llvm-cov nextest --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

If any step fails, fix that failure before opening or updating a PR.

## Recommended Execution Order

1. Task 1 through Task 6: finish the dense eager core substrate
2. Task 7 through Task 9: delete host determinant/sign reconstruction
3. Task 10: align public LU surface to tensor metadata
4. Task 11: remove temporary host bridges
5. Run the full verification gate

## Handoff Notes For The Next Agent

- Do not restart from `det/slogdet` directly. The metadata phase-2 and cast-family substrate is the blocker.
- Keep `PyTorch` semantics straight:
  - LU pivots are 1-indexed step pivots, not forward permutations.
  - Public `lu_solve` should ultimately accept pivot tensors, not `&[usize]`.
- Keep runtime and family modules split. `tenferro-device/src/cuda/runtime` has already been partially de-monolithized; continue that direction.
- Avoid adding new host reconstruction helpers in `tenferro-linalg`. If a cleanup needs one, the substrate is still incomplete.
