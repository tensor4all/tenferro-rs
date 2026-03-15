# Tenferro Frontend Gap Batch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the remaining frontend gaps in `tenferro::Tensor` by adding
structured-aware `permute`, `conj`, scalar extraction, a public snapshot
boundary, public mode inspection, and a PyTorch-like functional HVP API.

**Architecture:** Keep `Tensor` as the only compute protagonist. Add
`snapshot::DynTensor` for primal-only export/materialization, implement view
and conjugation through `StructuredTensor<T>`, and expose HVP as a
side-effect-free functional API similar to `torch.autograd.functional.hvp`.

**Tech Stack:** Rust, `tenferro`, `chainrules`, `tenferro-tensor`

---

### Task 1: Add snapshot boundary and scalar extraction

**Files:**
- Modify: `tenferro/src/lib.rs`
- Modify: `tenferro/src/core/mod.rs`
- Modify: `tenferro/src/core/dynamic/mod.rs`
- Modify: `tenferro/src/core/dynamic/dyn_tensor.rs`
- Modify: `tenferro/src/core/dynamic/dyn_ad_tensor/snapshot.rs`
- Add: `tenferro/tests/snapshot_surface_tests.rs`

**Step 1: Write failing public-surface tests**
Add tests for:
- `Tensor::primal_snapshot()`
- `snapshot::DynTensor` visibility under `tenferro::snapshot`
- `snapshot::DynTensor::to_dense()`
- `Tensor::try_scalar_value()` on rank-0 and non-rank-0 tensors

**Step 2: Run targeted tests to confirm failure**
Run:
`env CARGO_TARGET_DIR=/tmp/tenferro-frontend-gap-target cargo test -p tenferro snapshot_surface_tests -- --nocapture`

**Step 3: Implement snapshot module and scalar extraction**
- move or re-export dynamic primal tensor under `tenferro::snapshot::DynTensor`
- add `Tensor::primal_snapshot()`
- add `snapshot::DynTensor::to_dense()`
- add `ScalarValue`
- add `Tensor::try_scalar_value()`

**Step 4: Run targeted tests**
Run the same `snapshot_surface_tests` command.

**Step 5: Commit**
`git add tenferro/src/lib.rs tenferro/src/core/mod.rs tenferro/src/core/dynamic/mod.rs tenferro/src/core/dynamic/dyn_tensor.rs tenferro/src/core/dynamic/dyn_ad_tensor/snapshot.rs tenferro/tests/snapshot_surface_tests.rs`
`git commit -m "feat: add tenferro snapshot boundary"`

### Task 2: Expose `Tensor::mode()` and `Tensor::conj()`

**Files:**
- Modify: `tenferro/src/lib.rs`
- Modify: `tenferro/src/core/dynamic/dyn_ad_tensor/accessors.rs`
- Modify: `tenferro/src/core/dynamic/dyn_ad_tensor/complex.rs`
- Add: `tenferro/tests/tensor_mode_and_conj_tests.rs`

**Step 1: Write failing tests**
Add tests for:
- primal / reverse tensors reporting the correct mode
- complex conjugation on dense tensors
- conjugation preserving AD mode

**Step 2: Run targeted tests**
Run:
`env CARGO_TARGET_DIR=/tmp/tenferro-frontend-gap-target cargo test -p tenferro tensor_mode_and_conj_tests -- --nocapture`

**Step 3: Implement frontend methods**
- make `Tensor::mode()` public
- add `Tensor::conj()`
- ensure docs/examples exist for the new public surface

**Step 4: Run targeted tests**
Run the same `tensor_mode_and_conj_tests` command.

**Step 5: Commit**
`git add tenferro/src/lib.rs tenferro/src/core/dynamic/dyn_ad_tensor/accessors.rs tenferro/src/core/dynamic/dyn_ad_tensor/complex.rs tenferro/tests/tensor_mode_and_conj_tests.rs`
`git commit -m "feat: expose tensor mode and conjugation"`

### Task 3: Implement structured-aware `Tensor::permute`

**Files:**
- Modify: `tenferro/src/structured/layout.rs`
- Modify: `tenferro/src/core/dynamic/dyn_ad_tensor/shape.rs`
- Modify: `tenferro/src/core/dynamic/dyn_ad_tensor/layout.rs`
- Add: `tenferro/tests/tensor_permute_tests.rs`

**Step 1: Write failing tests**
Cover:
- dense transpose
- diagonal transpose stability
- multi-class structured permutation
- invalid permutation errors
- reverse-mode metadata preservation through `permute`

**Step 2: Run targeted tests**
Run:
`env CARGO_TARGET_DIR=/tmp/tenferro-frontend-gap-target cargo test -p tenferro tensor_permute_tests -- --nocapture`

**Step 3: Implement logical permutation**
- add structured helper in `StructuredTensor<T>`
- permute logical dims and axis classes
- derive the payload-axis permutation
- rebuild validated structured tensors
- expose `Tensor::permute(&[usize])`

**Step 4: Run targeted tests**
Run the same `tensor_permute_tests` command.

**Step 5: Commit**
`git add tenferro/src/structured/layout.rs tenferro/src/core/dynamic/dyn_ad_tensor/shape.rs tenferro/src/core/dynamic/dyn_ad_tensor/layout.rs tenferro/tests/tensor_permute_tests.rs`
`git commit -m "feat: add structured-aware tensor permute"`

### Task 4: Add public functional HVP API

**Files:**
- Modify: `tenferro/src/lib.rs`
- Add: `tenferro/src/functional.rs`
- Modify: `tenferro/src/autograd_api.rs` or shared helpers as needed
- Add: `tenferro/tests/functional_hvp_tests.rs`

**Step 1: Write failing tests**
Add exactly these minimal tests:
- `f(x) = sum(x^2)` gives `H v = 2v`
- `f(x, y) = sum(x^2) + 3 sum(y^2)` gives one HVP per input
- non-scalar `func` output errors

**Step 2: Run targeted tests**
Run:
`env CARGO_TARGET_DIR=/tmp/tenferro-frontend-gap-target cargo test -p tenferro functional_hvp_tests -- --nocapture`

**Step 3: Implement `tenferro::functional::hvp`**
- model it after `torch.autograd.functional.hvp`
- accept a user closure `func`
- return `(func_output, hvp)`
- do not accumulate into leaf `.grad`
- do not expose leaf `.hvp` mutation in the frontend

**Step 4: Run targeted tests**
Run the same `functional_hvp_tests` command.

**Step 5: Commit**
`git add tenferro/src/lib.rs tenferro/src/functional.rs tenferro/src/autograd_api.rs tenferro/tests/functional_hvp_tests.rs`
`git commit -m "feat: add functional hvp api"`

### Task 5: Docs, examples, and full verification

**Files:**
- Modify: `tenferro/src/lib.rs`
- Modify: public rustdoc for all touched methods/types
- Modify: relevant docs pages if frontend examples mention old gaps

**Step 1: Grep for stale or internal-only guidance**
Search for:
- missing public examples on new APIs
- root docs that still imply no snapshot boundary
- direct typed-accessor guidance for scalar extraction

**Step 2: Update docs**
- crate-level docs: mention `snapshot::DynTensor`
- `Tensor` docs: `permute`, `conj`, `try_scalar_value`, `mode`, `primal_snapshot`
- `functional::hvp` docs: PyTorch-like usage examples

**Step 3: Run full required verification**
Run:
- `cargo fmt --all --check`
- `env CARGO_TARGET_DIR=/tmp/tenferro-frontend-gap-release-target cargo test --workspace --release`
- `env CARGO_TARGET_DIR=/tmp/tenferro-frontend-gap-cov-target cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `env CARGO_TARGET_DIR=/tmp/tenferro-frontend-gap-doc-target cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py --doc-root /tmp/tenferro-frontend-gap-doc-target/doc`

**Step 4: Commit**
`git add tenferro/src tenferro/tests docs`
`git commit -m "docs: finalize tenferro frontend gap batch"`
