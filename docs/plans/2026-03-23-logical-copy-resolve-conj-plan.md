# Logical Copy Resolve Conj Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `Tensor::cat` and `Tensor::stack` handle mixed lazy-conjugated inputs correctly by introducing a PyTorch-aligned logical-copy substrate below tensor combine.

**Architecture:** Keep `stack` as `unsqueeze + cat`. Move lazy-conjugation resolution out of combine-specific logic and into a lower-level copy/materialization path: Layer 0 gets transform-aware strided copy, Layer 2 gets a context-free logical materialization helper, and Layer 3 `resolve_conj` becomes a thin wrapper over that substrate rather than the only implementation site.

**Tech Stack:** Rust, `tenferro-device`, `tenferro-tensor`, `tenferro-prims`, CUDA runtime substrate, module-local unit tests.

---

### Task 1: Add RED coverage for logical-value combine semantics

**Files:**
- Modify: `tenferro-tensor/src/tests/combine.rs`
- Test: `tenferro-tensor/src/tests/combine.rs`

**Step 1: Write the failing tests**

Add tests that pin the desired PyTorch-like semantics:
- `cat` over mixed lazy-conjugation inputs returns logical values, not an error
- `stack` over mixed lazy-conjugation inputs returns logical values, not an error
- output is no longer lazily conjugated after materialization
- stale `preferred_compute_device` remains cleared on fresh outputs

Use small `Complex64` tensors and assert against explicit logical values.

**Step 2: Run test to verify it fails**

Run:
```bash
cargo test -p tenferro-tensor --lib combine
```

Expected:
- FAIL because current code rejects mixed conjugation flags

**Step 3: Commit**

```bash
git add tenferro-tensor/src/tests/combine.rs
git commit -m "test: cover logical-value combine semantics"
```

### Task 2: Add transform-aware logical strided copy to Layer 0

**Files:**
- Modify: `tenferro-device/src/cuda/runtime.rs`
- Modify: `tenferro-device/src/cuda/tests/mod.rs`

**Step 1: Write the failing runtime test**

Add a focused CUDA runtime test for a strided copy variant that can apply a source transform while copying:
- phase 1 transform set: `None | Conj`
- test `Complex64` input with lazy-conj semantics encoded by the caller via transform selection

**Step 2: Run test to verify it fails**

Run:
```bash
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-device --features cuda --lib cuda::tests::cuda_runtime_copy_strided_with_conj_transform_matches_host -- --exact
```

Expected:
- FAIL because the transform-aware copy entrypoint does not exist yet

**Step 3: Write minimal implementation**

Add a low-level transform-aware copy helper that:
- keeps the existing plain copy path intact
- supports `None | Conj`
- stays same-dtype, same-device, fresh-output-or-explicit-dst only
- does not depend on `tenferro-prims`

**Step 4: Run test to verify it passes**

Run:
```bash
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-device --features cuda --lib cuda::tests::cuda_runtime_copy_strided_with_conj_transform_matches_host -- --exact
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add tenferro-device/src/cuda/runtime.rs tenferro-device/src/cuda/tests/mod.rs
git commit -m "feat: add logical strided copy transform to cuda runtime"
```

### Task 3: Add Layer 2 context-free logical materialization helper

**Files:**
- Modify: `tenferro-tensor/src/cuda_runtime.rs`
- Modify: `tenferro-tensor/src/tensor/data_ops.rs`
- Modify: `tenferro-tensor/src/tests/cuda.rs`

**Step 1: Write the failing tests**

Add focused tests that pin a tensor-level helper behavior:
- a lazy-conjugated complex GPU tensor can be materialized into a resolved contiguous tensor without any `CudaContext`
- the output has `conjugated = false`
- the output preserves logical values

**Step 2: Run test to verify it fails**

Run:
```bash
cargo test -p tenferro-tensor --features cuda --lib cuda
```

Expected:
- FAIL because no context-free logical materialization helper exists yet

**Step 3: Write minimal implementation**

Add an internal helper that:
- selects CPU/GPU path from `LogicalMemorySpace`
- uses plain copy for non-conjugated tensors
- uses transform-aware copy for conjugated tensors
- returns a resolved, contiguous tensor

Do not expose `CudaContext` here.

**Step 4: Run test to verify it passes**

Run:
```bash
cargo test -p tenferro-tensor --features cuda --lib cuda
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add tenferro-tensor/src/cuda_runtime.rs tenferro-tensor/src/tensor/data_ops.rs tenferro-tensor/src/tests/cuda.rs
git commit -m "feat: add logical tensor materialization helper"
```

### Task 4: Rewrite cat/stack to use logical copy semantics

**Files:**
- Modify: `tenferro-tensor/src/tensor/combine.rs`
- Modify: `tenferro-tensor/src/tests/combine.rs`

**Step 1: Run the existing RED tests**

Run:
```bash
cargo test -p tenferro-tensor --lib combine
cargo test -p tenferro-tensor --features cuda --lib combine
```

Expected:
- FAIL on mixed-conjugation combine semantics

**Step 2: Write minimal implementation**

Update `Tensor::cat` so that:
- inputs are copied as logical values, not raw buffer bytes
- mixed lazy-conjugation no longer errors
- output is resolved (`conjugated = false`)
- output clears `preferred_compute_device`

Keep `Tensor::stack` as `unsqueeze + cat`.

**Step 3: Run tests to verify they pass**

Run:
```bash
cargo test -p tenferro-tensor --lib combine
cargo test -p tenferro-tensor --features cuda --lib combine
```

Expected:
- PASS

**Step 4: Commit**

```bash
git add tenferro-tensor/src/tensor/combine.rs tenferro-tensor/src/tests/combine.rs
git commit -m "fix: make tensor combine respect logical conjugation"
```

### Task 5: Collapse Layer 3 resolve_conj onto the lower substrate

**Files:**
- Modify: `tenferro-prims/src/cpu/context.rs`
- Modify: `tenferro-prims/src/cuda/mod.rs`
- Modify: `tenferro-prims/src/tests/prims_tests.rs`
- Modify: `tenferro-prims/src/cuda/tests/mod.rs`

**Step 1: Write the failing tests**

Add or tighten tests so `CpuBackend::resolve_conj` and `CudaBackend::resolve_conj` are verified through the lower tensor substrate behavior.

**Step 2: Run test to verify it fails if wrappers diverge**

Run:
```bash
cargo test -p tenferro-prims --lib resolve_conj
cargo test -p tenferro-prims --features cuda --lib resolve_conj
```

Expected:
- FAIL if backend wrappers are still duplicating stale logic

**Step 3: Write minimal implementation**

Rewrite backend `resolve_conj` helpers as thin wrappers over the Layer 2 logical materialization helper.

**Step 4: Run tests to verify they pass**

Run:
```bash
cargo test -p tenferro-prims --lib resolve_conj
cargo test -p tenferro-prims --features cuda --lib resolve_conj
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add tenferro-prims/src/cpu/context.rs tenferro-prims/src/cuda/mod.rs tenferro-prims/src/tests/prims_tests.rs tenferro-prims/src/cuda/tests/mod.rs
git commit -m "refactor: route resolve_conj through tensor substrate"
```
