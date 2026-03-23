# Cross Cleanup And Runtime Subsplit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Finish the current logical-copy tranche by making `cross` tensor-native and by continuing the `tenferro-device` CUDA runtime split into smaller responsibility-focused modules.

**Architecture:** `cross` should stop relying on host extraction and instead compose existing tensor/scalar substrate from `tenferro-tensor` and `tenferro-prims`. The runtime split should preserve the public `tenferro_device::cuda::runtime::*` surface while moving large internal responsibilities out of the current oversized `state.rs` and `kernels.rs`.

**Tech Stack:** Rust, `tenferro-device`, `tenferro-tensor`, `tenferro-prims`, `tenferro-linalg`, CUDA runtime support, focused crate tests.

---

### Task 1: Commit The Current Logical-Copy Tranche

**Files:**
- Modify: `tenferro-device/src/cuda/runtime.rs`
- Create: `tenferro-device/src/cuda/runtime/kernels.rs`
- Create: `tenferro-device/src/cuda/runtime/shared.rs`
- Create: `tenferro-device/src/cuda/runtime/state.rs`
- Modify: `tenferro-tensor/src/cuda_runtime.rs`
- Modify: `tenferro-tensor/src/tensor/combine.rs`
- Modify: `tenferro-tensor/src/tensor/data_ops.rs`
- Test: `tenferro-tensor/src/tests/combine.rs`

**Step 1: Verify focused tests are green before commit**

Run:

```bash
cargo fmt --all --check
cargo test -p tenferro-device --features cuda --lib
cargo test -p tenferro-tensor --lib combine
cargo test -p tenferro-tensor --features cuda --lib combine
```

**Step 2: Commit runtime split**

```bash
git add tenferro-device/src/cuda/runtime.rs tenferro-device/src/cuda/runtime/
git commit -m "refactor: split cuda runtime modules"
```

**Step 3: Commit logical-copy tensor changes**

```bash
git add tenferro-tensor/src/cuda_runtime.rs tenferro-tensor/src/tensor/combine.rs tenferro-tensor/src/tensor/data_ops.rs tenferro-tensor/src/tests/combine.rs
git commit -m "feat: make tensor combine respect logical values"
```

### Task 2: Make `cross` Tensor-Native

**Files:**
- Modify: `tenferro-linalg/src/primal/tensor_ops.rs`
- Modify: `tenferro-linalg/src/prims_bridge.rs`
- Modify: `tenferro-linalg/src/tests/batch_b_contracts.rs`
- Modify: `tenferro-linalg/src/tests/runtime_capability.rs`

**Step 1: Write the failing regression**

Add a test that requires `cross` to work on CUDA-generic tensor substrate without `extract_slice()`/`tensor_from_data()` and keeps the current broadcasting/right-hand-rule behavior.

**Step 2: Run the focused test to verify RED**

Run:

```bash
cargo test -p tenferro-linalg --lib cross
```

Expected: a source-level/runtime-capability failure or behavior failure proving the old host path is still present.

**Step 3: Implement the minimal tensor-native path**

Replace host extraction with composition from existing tensor/scalar substrate. The intended shape is:
- slice the leading vector axis into `ax/ay/az` and `bx/by/bz`
- compute the three output components with scalar ops
- `stack` the three result components back on axis `0`

Do not introduce linalg-specific CUDA kernels and do not add host fallbacks.

**Step 4: Verify GREEN**

Run:

```bash
cargo fmt --all --check
cargo test -p tenferro-linalg --lib cross
cargo test -p tenferro-linalg --features cuda --lib --no-run
```

**Step 5: Commit**

```bash
git add tenferro-linalg/src/primal/tensor_ops.rs tenferro-linalg/src/prims_bridge.rs tenferro-linalg/src/tests/batch_b_contracts.rs tenferro-linalg/src/tests/runtime_capability.rs
git commit -m "refactor: make cross tensor-native"
```

### Task 3: Continue Runtime Subsplit

**Files:**
- Modify: `tenferro-device/src/cuda/runtime/state.rs`
- Modify: `tenferro-device/src/cuda/runtime/kernels.rs`
- Create: `tenferro-device/src/cuda/runtime/state_*.rs`
- Create: `tenferro-device/src/cuda/runtime/kernel_*.rs`
- Test: `tenferro-device/src/cuda/tests/mod.rs`

**Step 1: Write the failing compile/test target**

Pick one responsibility split and move it behind a new internal module boundary while preserving public imports from `runtime.rs`.

Recommended first splits:
- `state.rs` -> cache/launch/module-loading pieces
- `kernels.rs` -> copy/materialize helpers vs scalar/reduction kernels

**Step 2: Run focused verification**

```bash
cargo test -p tenferro-device --features cuda --lib
```

**Step 3: Commit**

```bash
git add tenferro-device/src/cuda/runtime.rs tenferro-device/src/cuda/runtime/
git commit -m "refactor: continue cuda runtime subsplit"
```

### Task 4: Next Host-Loop Cleanup After `cross`

**Files:**
- Modify: `tenferro-linalg/src/primal/tensor_ops.rs`
- Modify: `tenferro-linalg/src/tests/batch_b_contracts.rs`
- Modify: `tenferro-linalg/src/tests/runtime_capability.rs`

**Step 1: Re-run hotspot inventory**

Confirm `householder_product` still relies on `extract_slice()` / `tensor_from_data()`.

**Step 2: Decide if existing substrate is sufficient**

Proceed only if the needed tensor-native write/combine path exists. If a generic substrate is missing, stop and record the blocker instead of adding an ad hoc implementation.
