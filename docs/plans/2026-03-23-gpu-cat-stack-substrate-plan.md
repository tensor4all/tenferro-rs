# GPU Cat/Stack Substrate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add GPU-generic `Tensor::cat` as the core packing substrate, then lift `Tensor::stack` onto it, so remaining host-loop tensor/linalg ops can be cleaned up without ad hoc GPU kernels or payload host transfers.

**Architecture:** Implement a same-dtype, same-device pack/materialization path beneath tensor combine, keep validation and shape semantics in `tenferro-tensor`, and only then consume the substrate from higher layers. `cat` is the core primitive; `stack` follows from `unsqueeze + cat`.

**Tech Stack:** Rust, `tenferro-device`, `tenferro-tensor`, CUDA runtime substrate, unit tests in module-local test directories.

---

### Task 1: Add RED coverage for GPU tensor combine behavior

**Files:**
- Modify: `tenferro-tensor/src/tests/combine.rs`
- Test: `tenferro-tensor/src/tests/combine.rs`

**Step 1: Write the failing tests**

Add tests for:
- GPU `cat` on two same-shape tensors along a nonzero axis
- GPU `stack` on two same-shape tensors
- regression that these operations no longer error with the old `MainMemory`-only message

**Step 2: Run test to verify it fails**

Run:
```bash
cargo test -p tenferro-tensor --features cuda --lib combine
```

Expected:
- FAIL because `cat` / `stack` still reject non-main-memory tensors

**Step 3: Commit**

```bash
git add tenferro-tensor/src/tests/combine.rs
git commit -m "test: cover gpu tensor combine substrate"
```

### Task 2: Add low-level GPU pack/materialization entrypoint

**Files:**
- Modify: `tenferro-device/src/cuda/runtime.rs`
- Modify: `tenferro-device/src/cuda/tests/mod.rs`

**Step 1: Write the failing runtime test**

Add a focused CUDA runtime test that packs two same-dtype source views into one contiguous output buffer for a concat-style layout.

**Step 2: Run test to verify it fails**

Run:
```bash
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-device --features cuda --lib cuda::tests::cuda_runtime_can_pack_concat_sources -- --exact
```

Expected:
- FAIL because the pack entrypoint/kernel does not exist yet

**Step 3: Write minimal implementation**

Add a low-level runtime helper that:
- validates same device
- allocates contiguous output
- launches a kernel that copies each source view into its destination region

Keep scope minimal:
- same dtype
- fresh output only
- no mixed-device support

**Step 4: Run test to verify it passes**

Run:
```bash
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-device --features cuda --lib cuda::tests::cuda_runtime_can_pack_concat_sources -- --exact
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add tenferro-device/src/cuda/runtime.rs tenferro-device/src/cuda/tests/mod.rs
git commit -m "feat: add cuda pack substrate for tensor combine"
```

### Task 3: Lift `Tensor::cat` onto the GPU substrate

**Files:**
- Modify: `tenferro-tensor/src/tensor/combine.rs`
- Modify: `tenferro-tensor/src/tests/combine.rs`

**Step 1: Run the existing RED test**

Run:
```bash
cargo test -p tenferro-tensor --features cuda --lib combine
```

Expected:
- FAIL on the GPU `cat` test

**Step 2: Write minimal implementation**

Update `Tensor::cat` so that:
- CPU path keeps the current implementation
- GPU path delegates to the new pack/materialization substrate
- public validation behavior remains unchanged except the old main-memory rejection is removed for supported GPU tensors

**Step 3: Run tests to verify they pass**

Run:
```bash
cargo test -p tenferro-tensor --features cuda --lib combine
cargo test -p tenferro-tensor --lib combine
```

Expected:
- PASS

**Step 4: Commit**

```bash
git add tenferro-tensor/src/tensor/combine.rs tenferro-tensor/src/tests/combine.rs
git commit -m "feat: enable gpu tensor cat"
```

### Task 4: Lift `Tensor::stack` onto `unsqueeze + cat`

**Files:**
- Modify: `tenferro-tensor/src/tensor/combine.rs`
- Modify: `tenferro-tensor/src/tests/combine.rs`

**Step 1: Run RED test**

Run:
```bash
cargo test -p tenferro-tensor --features cuda --lib combine
```

Expected:
- FAIL on the GPU `stack` test

**Step 2: Write minimal implementation**

Refactor `Tensor::stack` to:
- keep current CPU semantics
- implement GPU/materialized behavior via `unsqueeze` plus the now-working `Tensor::cat`
- avoid duplicating pack logic

**Step 3: Run tests to verify they pass**

Run:
```bash
cargo test -p tenferro-tensor --features cuda --lib combine
cargo test -p tenferro-tensor --lib combine
```

Expected:
- PASS

**Step 4: Commit**

```bash
git add tenferro-tensor/src/tensor/combine.rs tenferro-tensor/src/tests/combine.rs
git commit -m "feat: enable gpu tensor stack"
```

### Task 5: Add source-level regression for combine layering

**Files:**
- Modify: `tenferro-tensor/src/tests/organization.rs`

**Step 1: Write the failing test**

Add a source-level assertion that `combine.rs` no longer hard-rejects non-main-memory tensors for supported `cat` / `stack` paths.

**Step 2: Run test to verify it fails**

Run:
```bash
cargo test -p tenferro-tensor --lib organization
```

Expected:
- FAIL because the old source text is still present or the guard is missing

**Step 3: Write minimal implementation**

Update the organization/runtime regression test to pin the new layering.

**Step 4: Run test to verify it passes**

Run:
```bash
cargo test -p tenferro-tensor --lib organization
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add tenferro-tensor/src/tests/organization.rs
git commit -m "test: lock gpu tensor combine layering"
```

### Task 6: Use the new substrate in one high-level op only

**Files:**
- Modify: `tenferro-linalg/src/primal/tensor_ops.rs`
- Modify: `tenferro-linalg/src/tests/runtime_capability.rs`
- Modify: `tenferro-linalg/src/tests/batch_b_contracts.rs`

**Step 1: Write the failing test**

Choose exactly one op, starting with `vander`, and add:
- a public regression if behavior changes or broadens
- a source-level regression that the chosen op no longer calls `extract_slice`

**Step 2: Run test to verify it fails**

Run:
```bash
cargo test -p tenferro-linalg --lib vander
```

Expected:
- FAIL because the op still uses host-slice materialization

**Step 3: Write minimal implementation**

Rewrite only `vander` against tensor-native combine/materialization. Do not touch `cross` or `householder_product` in the same task.

**Step 4: Run tests to verify they pass**

Run:
```bash
cargo test -p tenferro-linalg --lib vander
cargo test -p tenferro-linalg --features cuda --lib --no-run
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add tenferro-linalg/src/primal/tensor_ops.rs tenferro-linalg/src/tests/runtime_capability.rs tenferro-linalg/src/tests/batch_b_contracts.rs
git commit -m "refactor: make vander tensor-native"
```

### Task 7: Full verification

**Files:**
- Modify: none

**Step 1: Run focused verification**

Run:
```bash
cargo fmt --all --check
cargo test -p tenferro-device --features cuda --lib
cargo test -p tenferro-tensor --features cuda --lib
cargo test -p tenferro-linalg --lib
cargo test -p tenferro-linalg --features cuda --lib --no-run
```

Expected:
- PASS

**Step 2: Commit if needed**

```bash
git status --short
```

Expected:
- clean working tree

