# Tenferro Prims CUDA Phase-1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the `tenferro-prims` CUDA phase-1 public surface with GPU-resident execution, truthful backend capability reporting, crate-local documentation, and persistent caching for custom CUDA kernels.

**Architecture:** Keep direct `cuTENSOR` paths for contraction, permutation, reduction, and simple elementwise operations. Add a small custom CUDA runtime inside `tenferro-prims` for pointwise, reduction, and diagonal-family kernels compiled with NVRTC and cached persistently. Represent multi-step CUDA execution as internal pipelines with GPU scratch buffers, without changing public family traits.

**Tech Stack:** Rust, `cudarc`, cuTENSOR, NVRTC, CUDA C++, `tenferro-tensor`, `tenferro-prims`

---

### Task 0: Fix cuTENSOR workspace plumbing in the existing CUDA path

**Files:**
- Modify: `tenferro-prims/src/cuda/mod.rs`
- Modify: `tenferro-prims/src/cuda/planning.rs`
- Modify: `tenferro-prims/src/cuda/execution.rs`
- Test: `tenferro-prims/src/cuda/tests/mod.rs`

**Step 1: Inspect current workspace handling and add a regression test scaffold**

Add a focused test or assertion path that verifies:
- plan creation retains estimated cuTENSOR workspace size
- execution does not silently drop non-zero workspace requirements

Keep the test GPU-conditional if runtime access is required.

**Step 2: Run targeted CUDA library tests**

Run:
`cargo test -p tenferro-prims --lib cuda -- --nocapture`

Expected:
- current CUDA compile-time tests pass
- new workspace-specific assertions fail or are not yet implemented

**Step 3: Implement execute-time workspace allocation**

Implement:
- storing estimated workspace size in CUDA plan metadata
- execute-time GPU allocation when workspace size is non-zero
- reusing bounded cached buffers when safe

**Step 4: Re-run targeted CUDA library tests**

Run:
`cargo test -p tenferro-prims --lib cuda -- --nocapture`

Expected:
- PASS

**Step 5: Commit**

```bash
git add tenferro-prims/src/cuda/mod.rs tenferro-prims/src/cuda/planning.rs tenferro-prims/src/cuda/execution.rs tenferro-prims/src/cuda/tests/mod.rs
git commit -m "fix: plumb cutensor workspace through cuda execution"
```

### Task 1: Freeze the public CUDA scope in crate docs

**Files:**
- Add: `tenferro-prims/README.md`
- Modify: `docs/design/tensor-prims.md`
- Modify: `docs/design/supported-ops.md`
- Test: `tenferro-prims/src/tests/mod.rs`

**Step 1: Audit the current public surface and doc gaps**

Confirm the README/doc gaps against:
- current public family traits
- current test inventory
- current truthful backend support checks

**Step 2: Write the crate-local README and align design docs**

Document:
- family boundaries
- public operation vocabulary
- backend-specific Cargo features: `cuda`, `rocm`
- CPU/CUDA/ROCm status matrix
- CPU-only default CI policy
- prohibition on CPU fallback inside CUDA execution

**Step 3: Refresh surface-consistency tests if needed**

Add or update only the lightweight tests that guard:
- primitive inventory assumptions
- truthfulness of `has_*_support()`

Do not force artificial failing tests for documentation-only changes.

**Step 4: Run targeted tests**

Run:
`cargo test -p tenferro-prims protocol_smoke -- --nocapture`

Expected:
- PASS

**Step 5: Commit**

```bash
git add tenferro-prims/README.md docs/design/tensor-prims.md docs/design/supported-ops.md tenferro-prims/src/tests/mod.rs
git commit -m "docs: freeze tenferro-prims cuda phase-1 surface"
```

### Task 2: Add backend feature propagation and CPU-only CI policy docs

**Files:**
- Modify: `tenferro-tensor/Cargo.toml`
- Modify: `tenferro-prims/Cargo.toml`
- Modify: `tenferro-einsum/Cargo.toml`
- Modify: `tenferro/Cargo.toml`
- Modify: `tenferro-prims/README.md`

**Step 1: Audit which crates should expose backend features in this phase**

Keep propagation only where it helps real standalone use now:
- `tenferro-tensor`
- `tenferro-prims`
- `tenferro-einsum`
- `tenferro`

**Step 2: Run lightweight checks**

Run:
`cargo check -p tenferro-prims`

Expected:
- PASS before Cargo feature changes

**Step 3: Implement Cargo feature propagation**

Apply these changes:
- keep standalone `cuda` on `tenferro-tensor` and `tenferro-prims`
- add `rocm` features mirroring `cuda`
- propagate `cuda` and `rocm` through `tenferro-einsum` and `tenferro`
- do not add a `gpu` umbrella feature
- do not claim GPU runtime support in crates that still lack truthful implementation
- do not wire `tenferro-linalg` or `tenferro-capi` in this phase unless a real
  gating need appears during implementation

**Step 4: Run feature compile checks**

Run:
- `cargo check -p tenferro-prims`
- `cargo check -p tenferro-einsum`
- `cargo check -p tenferro`

Expected:
- PASS

**Step 5: Commit**

```bash
git add tenferro-tensor/Cargo.toml tenferro-prims/Cargo.toml tenferro-einsum/Cargo.toml tenferro/Cargo.toml tenferro-prims/README.md
git commit -m "build: propagate backend-specific gpu features"
```

### Task 3: Introduce the custom CUDA runtime, plan model, and scratch/cache system

**Files:**
- Modify: `tenferro-prims/src/cuda/mod.rs`
- Modify: `tenferro-prims/src/cuda/planning.rs`
- Modify: `tenferro-prims/src/cuda/execution.rs`
- Add: `tenferro-prims/src/cuda/custom/mod.rs`
- Add: `tenferro-prims/src/cuda/custom/cache.rs`
- Add: `tenferro-prims/src/cuda/custom/kernel_key.rs`
- Add: `tenferro-prims/src/cuda/custom/nvrtc.rs`
- Add: `tenferro-prims/src/cuda/custom/launch.rs`
- Add: `tenferro-prims/src/cuda/custom/scratch.rs`
- Add: `tenferro-prims/src/cuda/kernel_src/pointwise_unary.cu`
- Add: `tenferro-prims/src/cuda/kernel_src/pointwise_binary.cu`
- Add: `tenferro-prims/src/cuda/kernel_src/reduction.cu`
- Add: `tenferro-prims/src/cuda/kernel_src/diagonal_family.cu`
- Test: `tenferro-prims/src/cuda/tests/mod.rs`

**Step 1: Write failing infrastructure tests**

Add tests for:
- stable kernel-key hashing
- cache-path selection (`TENFERRO_CACHE_DIR` override vs default)
- artifact metadata round-trip
- bounded scratch-pool retention policy
- plan variants for native, custom, and pipeline execution

**Step 2: Run targeted CUDA library tests to confirm failure**

Run:
`cargo test -p tenferro-prims --lib cuda -- --nocapture`

Expected:
- existing CUDA compile-time surface tests pass
- new cache/runtime tests fail because the modules do not exist yet

**Step 3: Implement the minimal custom runtime and plan model**

Implement:
- stable cache key generation
- concrete cache invalidation inputs:
  - source hash
  - entrypoint
  - compile options
  - SM arch
  - CUDA driver version
  - cuTENSOR version
  - custom-runtime ABI version
- persistent artifact storage
- NVRTC compile helper
- module/function in-process cache
- extension of `CudaContext` to own custom runtime state
- native/custom/pipeline CUDA plan representation
- bounded scratch manager with explicit retention limits

Keep tensor payloads GPU-resident and do not add CPU fallback paths.

**Step 4: Run targeted tests**

Run:
`cargo test -p tenferro-prims --lib cuda -- --nocapture`

Expected:
- PASS

**Step 5: Commit**

```bash
git add tenferro-prims/src/cuda/mod.rs tenferro-prims/src/cuda/planning.rs tenferro-prims/src/cuda/execution.rs tenferro-prims/src/cuda/custom tenferro-prims/src/cuda/kernel_src tenferro-prims/src/cuda/tests/mod.rs
git commit -m "feat: add cuda custom runtime and pipeline plans"
```

### Task 4: Finish semiring-core diagonal-family CUDA support

**Files:**
- Modify: `tenferro-prims/src/families/semiring_core.rs`
- Modify: `tenferro-prims/src/cuda/execution.rs`
- Modify: `tenferro-prims/src/cuda/planning.rs`
- Modify: `tenferro-prims/src/cuda/kernel_src/diagonal_family.cu`
- Add: `tenferro-prims/src/cuda/tests/semiring_core_cuda.rs`

**Dependency:** Requires Task 3 to be complete because diagonal-family ops use
the custom-kernel runtime and pipeline infrastructure.

**Step 1: Write failing diagonal-family tests**

Cover:
- `Trace`
- `AntiTrace`
- `AntiDiag`
- non-contiguous shapes
- GPU-only execution contract

**Step 2: Run the targeted tests**

Run:
`cargo test -p tenferro-prims --lib semiring_core_cuda -- --nocapture`

Expected:
- FAIL because CUDA diagonal-family ops are currently unimplemented

**Step 3: Implement minimal diagonal-family kernels**

Implement planning and execution for:
- `Trace`
- `AntiTrace`
- `AntiDiag`

Use the custom diagonal-family kernel source and keep all intermediate state on
GPU.

**Step 4: Re-run the targeted tests**

Run:
`cargo test -p tenferro-prims --lib semiring_core_cuda -- --nocapture`

Expected:
- PASS on CPU-only builds for compile-time coverage
- PASS on GPU-enabled environments for runtime tests

**Step 5: Commit**

```bash
git add tenferro-prims/src/families/semiring_core.rs tenferro-prims/src/cuda/execution.rs tenferro-prims/src/cuda/planning.rs tenferro-prims/src/cuda/kernel_src/diagonal_family.cu tenferro-prims/src/cuda/tests/semiring_core_cuda.rs
git commit -m "feat: add cuda diagonal-family semiring ops"
```

### Task 5: Add CUDA scalar-family support

**Files:**
- Modify: `tenferro-prims/src/families/scalar.rs`
- Add: `tenferro-prims/src/cuda/scalar_family.rs`
- Modify: `tenferro-prims/src/cuda/kernel_src/pointwise_unary.cu`
- Modify: `tenferro-prims/src/cuda/kernel_src/pointwise_binary.cu`
- Modify: `tenferro-prims/src/cuda/kernel_src/reduction.cu`
- Add: `tenferro-prims/src/cuda/tests/scalar_cuda.rs`
- Modify: `tenferro-prims/src/tests/scalar_phase1.rs`

**Step 1: Write failing scalar CUDA tests**

Cover:
- `has_scalar_support()` truthfulness
- unary ops by dtype class
- binary ops by dtype class
- reductions including `Mean`
- rejection of unsupported complex/ordered-real mismatches

**Step 2: Run targeted tests**

Run:
`cargo test -p tenferro-prims --lib scalar_phase1 -- --nocapture`

Expected:
- current CPU tests pass
- CUDA support tests fail because `CudaBackend` still reports no scalar support

**Step 3: Implement minimal scalar-family planning and execution**

Implement:
- direct `cuTENSOR` path where classified as direct
- extend cuTENSOR operator wrapper support to cover audited real unary ops
- composed path for `Sub`, `Square`, `Mean`
- custom kernels for `Imag`, complex `Real`, complex `Abs`, complex `Reciprocal`, complex `Div`
- truthful `has_scalar_support()`

**Step 4: Re-run targeted tests**

Run:
`cargo test -p tenferro-prims --lib scalar_phase1 -- --nocapture`

Expected:
- PASS

**Step 5: Commit**

```bash
git add tenferro-prims/src/families/scalar.rs tenferro-prims/src/cuda/scalar_family.rs tenferro-prims/src/cuda/kernel_src/pointwise_unary.cu tenferro-prims/src/cuda/kernel_src/pointwise_binary.cu tenferro-prims/src/cuda/kernel_src/reduction.cu tenferro-prims/src/cuda/tests/scalar_cuda.rs tenferro-prims/src/tests/scalar_phase1.rs
git commit -m "feat: add cuda scalar primitive support"
```

### Task 6: Add CUDA analytic-family support

**Files:**
- Modify: `tenferro-prims/src/families/analytic.rs`
- Add: `tenferro-prims/src/cuda/analytic_family.rs`
- Modify: `tenferro-prims/src/cuda/kernel_src/pointwise_unary.cu`
- Modify: `tenferro-prims/src/cuda/kernel_src/pointwise_binary.cu`
- Modify: `tenferro-prims/src/cuda/kernel_src/reduction.cu`
- Add: `tenferro-prims/src/cuda/tests/analytic_cuda.rs`
- Modify: `tenferro-prims/src/tests/analytic_phase1.rs`

**Step 1: Write failing analytic CUDA tests**

Cover:
- unary analytic ops by dtype class
- custom binary ops: `Pow`, `Atan2`, `Hypot`, `Xlogy`
- reductions: `Var`, `Std`
- truthful `has_analytic_support()`

**Step 2: Run targeted tests**

Run:
`cargo test -p tenferro-prims --lib analytic_phase1 -- --nocapture`

Expected:
- current CPU tests pass
- CUDA support tests fail because `CudaBackend` still reports no analytic support

**Step 3: Implement minimal analytic-family planning and execution**

Implement:
- direct `cuTENSOR` path where classified as direct
- extend cuTENSOR operator wrapper support to cover audited real unary ops
- custom pointwise kernels for complex and non-trivial unary/binary ops
- custom reduction kernels for `Var` and `Std`
- truthful `has_analytic_support()`

**Step 4: Re-run targeted tests**

Run:
`cargo test -p tenferro-prims --lib analytic_phase1 -- --nocapture`

Expected:
- PASS

**Step 5: Commit**

```bash
git add tenferro-prims/src/families/analytic.rs tenferro-prims/src/cuda/analytic_family.rs tenferro-prims/src/cuda/kernel_src/pointwise_unary.cu tenferro-prims/src/cuda/kernel_src/pointwise_binary.cu tenferro-prims/src/cuda/kernel_src/reduction.cu tenferro-prims/src/cuda/tests/analytic_cuda.rs tenferro-prims/src/tests/analytic_phase1.rs
git commit -m "feat: add cuda analytic primitive support"
```

### Task 7: Final verification, rustdoc, and status-table cleanup

**Files:**
- Modify: `tenferro-prims/src/lib.rs`
- Modify: `tenferro-prims/README.md`
- Modify: public rustdoc in touched files
- Modify: `docs/design/tensor-prims.md`
- Modify: `docs/design/supported-ops.md`

**Step 1: Re-audit the implementation against the phase-1 invariants**

Before the final completion commit, verify that the implementation still
obeys all mandatory principles from the design doc:
- GPU-resident CUDA execution only
- no CPU fallback inside CUDA execution
- truthful `has_*_support()` and planning behavior
- no public CUDA-specific trait/API expansion
- bounded workspace/scratch retention
- backend-specific optional features only
- docs/tests/support tables aligned with implementation

Treat any violation here as a blocking issue for completion.

**Step 2: Audit public docs for touched types and functions**

Check that every public type, trait, and function touched by this work has:
- minimal doc comments
- `# Examples` blocks
- truthful backend notes

**Step 3: Run formatting and targeted verification**

Run:
- `cargo fmt --all`
- `cargo test -p tenferro-prims`
- `cargo test -p tenferro-prims --features cuda`

Expected:
- CPU-only tests pass
- CUDA-feature tests compile and pass where runtime is available

**Step 4: Run full repository verification**

Run:
- `cargo fmt --all --check`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`

Expected:
- PASS, except GPU runtime tests may be skipped on non-GPU environments

**Step 5: Commit**

```bash
git add tenferro-prims/src tenferro-prims/README.md docs/design/tensor-prims.md docs/design/supported-ops.md
git commit -m "docs: finalize tenferro-prims cuda phase-1"
```
