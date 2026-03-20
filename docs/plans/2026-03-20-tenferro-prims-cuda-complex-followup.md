# Tenferro Prims CUDA Complex Follow-Up Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the remaining CUDA gaps after real phase-1 by adding truthful complex support for `TensorScalarPrims` and `TensorAnalyticPrims` where the CPU backend already supports complex execution.

**Architecture:** Keep the current GPU-resident CUDA backend structure from the real phase-1 work. Reuse cuTENSOR paths only where complex behavior is clean and truthful, and extend the in-tree NVRTC custom-kernel runtime for complex pointwise gaps. Do not rewrite the previous phase-1 plan; treat this file as a follow-up plan for the still-missing complex subset.

**Tech Stack:** Rust, `cudarc`, cuTENSOR, NVRTC, CUDA C++, `num-complex`, `tenferro-tensor`, `tenferro-prims`

---

## Scope Freeze

This follow-up plan covers only the remaining CUDA complex gaps after commit
`f3a338b`.

In scope:

- CUDA `TensorScalarPrims` complex support for the CPU-supported subset
- CUDA `TensorAnalyticPrims` complex support for the CPU-supported subset
- truthful capability predicates, tests, README, and support-table updates

Out of scope:

- rewriting the earlier phase-1 plan documents
- ROCm implementation work
- adding new public CUDA-only APIs
- changing real-only CPU semantics for ordered operations and moment reductions

Operations that remain real-only because the CPU backend is also real-only:

- scalar: `Maximum`, `Minimum`, `ClampMin`, `ClampMax`, `Max`, `Min`
- analytic: `Atan2`, `Hypot`, `Var`, `Std`

### Task 1: Freeze the remaining complex inventory with failing capability tests

**Files:**
- Modify: `tenferro-prims/src/tests/scalar_phase1.rs`
- Modify: `tenferro-prims/src/tests/analytic_phase1.rs`
- Modify: `tenferro-prims/src/cuda/tests/mod.rs`
- Modify: `tenferro-prims/README.md`

**Step 1: Write the failing complex capability tests**

Add CUDA-only tests that assert `has_*_support()` for the complex subset that
should become supported:

- scalar unary: `Neg`, `Conj`, `Abs`, `Reciprocal`, `Real`, `Imag`, `Square`
- scalar binary: `Add`, `Sub`, `Mul`, `Div`
- scalar reductions: `Sum`, `Prod`, `Mean`
- analytic unary: `Sqrt`, `Rsqrt`, `Exp`, `Expm1`, `Log`, `Log1p`, `Sin`, `Cos`, `Tan`, `Tanh`, `Asin`, `Acos`, `Atan`, `Sinh`, `Cosh`, `Asinh`, `Acosh`, `Atanh`
- analytic binary: `Pow`, `Xlogy`

Also add negative assertions that ordered-real-only ops remain unsupported for
complex CUDA.

**Step 2: Run the targeted tests to verify they fail**

Run:
`cargo test -p tenferro-prims --features cuda cuda_scalar_phase1_advertises cuda_analytic_phase1_advertises -- --nocapture`

Expected:
- FAIL because current CUDA support predicates reject all complex scalar and
  analytic descriptors

**Step 3: Update README wording to reflect the intended next scope**

Adjust the CUDA status wording in `tenferro-prims/README.md` so it is ready for
the upcoming complex parity work, but do not prematurely claim complex support.

**Step 4: Re-run the targeted tests**

Run:
`cargo test -p tenferro-prims --features cuda cuda_scalar_phase1_advertises cuda_analytic_phase1_advertises -- --nocapture`

Expected:
- still FAIL until implementation lands

**Step 5: Commit**

```bash
git add tenferro-prims/src/tests/scalar_phase1.rs tenferro-prims/src/tests/analytic_phase1.rs tenferro-prims/src/cuda/tests/mod.rs tenferro-prims/README.md
git commit -m "test: lock remaining cuda complex support inventory"
```

### Task 2: Extend the custom CUDA runtime for complex pointwise kernels

**Files:**
- Modify: `tenferro-prims/src/cuda/custom/mod.rs`
- Modify: `tenferro-prims/src/cuda/custom/cache.rs`
- Add: `tenferro-prims/src/cuda/kernel_src/pointwise_complex.cu`
- Modify: `tenferro-prims/src/cuda/pointwise_ops.rs`
- Test: `tenferro-prims/src/cuda/tests/mod.rs`

**Step 1: Write a failing smoke test for complex custom-kernel execution**

Add a focused CUDA-conditional test that will need at least one complex custom
kernel path, for example:

- scalar complex `Imag`
- analytic complex `Pow`

**Step 2: Run the targeted CUDA test**

Run:
`cargo test -p tenferro-prims --features cuda cuda_scalar_and_analytic_smoke_run_on_device_tensors_when_runtime_is_available -- --nocapture`

Expected:
- FAIL on the new complex smoke assertions

**Step 3: Implement complex custom-kernel infrastructure**

Implement:

- complex unary/binary op enums in the custom runtime
- persistent cache keys for the complex kernel module
- a `pointwise_complex.cu` source file with kernels for:
  - unary: complex `neg`, `abs`, `reciprocal`, `real`, `imag`, `rsqrt`, `expm1`, `log1p`
  - binary: complex `div`, `pow`, `xlogy`
- launch helpers for complex32 and complex64

Keep kernel outputs in the same tensor dtype as the trait contract requires.
For `Real` and `Imag`, write the real component back as a complex number with
zero imaginary part.

**Step 4: Re-run the targeted CUDA test**

Run:
`cargo test -p tenferro-prims --features cuda cuda_scalar_and_analytic_smoke_run_on_device_tensors_when_runtime_is_available -- --nocapture`

Expected:
- FAIL only on capability or family-dispatch gaps that are not yet implemented

**Step 5: Commit**

```bash
git add tenferro-prims/src/cuda/custom/mod.rs tenferro-prims/src/cuda/custom/cache.rs tenferro-prims/src/cuda/kernel_src/pointwise_complex.cu tenferro-prims/src/cuda/pointwise_ops.rs tenferro-prims/src/cuda/tests/mod.rs
git commit -m "feat: add cuda complex pointwise kernel runtime"
```

### Task 3: Implement CUDA complex scalar family parity

**Files:**
- Modify: `tenferro-prims/src/cuda/scalar_family.rs`
- Modify: `tenferro-prims/src/cuda/family_common.rs`
- Modify: `tenferro-prims/src/cuda/pointwise_ops.rs`
- Test: `tenferro-prims/src/tests/scalar_phase1.rs`
- Test: `tenferro-prims/src/cuda/tests/mod.rs`

**Step 1: Write failing execution tests for representative complex scalar ops**

Cover:

- unary: `Conj`, `Abs`, `Real`, `Imag`
- binary: `Add`, `Sub`, `Mul`, `Div`
- reductions: `Sum`, `Prod`, `Mean`

Use GPU-resident tensors and compare against CPU results after materializing
back to host only in the test harness.

**Step 2: Run the targeted tests to verify failure**

Run:
`cargo test -p tenferro-prims --features cuda scalar_phase1 cuda_scalar_and_analytic_smoke_run_on_device_tensors_when_runtime_is_available -- --nocapture`

Expected:
- FAIL on the new complex scalar cases

**Step 3: Implement truthful complex scalar support**

Implement:

- replace the current real-only `supports_real_scalar_type()` gate with scalar
  family support logic that distinguishes:
  - complex-supported ops
  - real-only ops
- route complex scalar unary ops through either:
  - direct cuTENSOR where truthful, or
  - the new custom complex kernels
- route complex scalar binary ops through:
  - trinary cuTENSOR for `Add`, `Mul`
  - scale-based or custom handling for `Sub`
  - custom handling for `Div`
- route complex `Sum`, `Prod`, and `Mean` through GPU-resident reduction paths

Do not claim support for complex extrema or clamp operations.

**Step 4: Re-run the targeted tests**

Run:
`cargo test -p tenferro-prims --features cuda scalar_phase1 cuda_scalar_and_analytic_smoke_run_on_device_tensors_when_runtime_is_available -- --nocapture`

Expected:
- PASS for the new complex scalar tests

**Step 5: Commit**

```bash
git add tenferro-prims/src/cuda/scalar_family.rs tenferro-prims/src/cuda/family_common.rs tenferro-prims/src/cuda/pointwise_ops.rs tenferro-prims/src/tests/scalar_phase1.rs tenferro-prims/src/cuda/tests/mod.rs
git commit -m "feat: implement cuda complex scalar primitives"
```

### Task 4: Implement CUDA complex analytic family parity

**Files:**
- Modify: `tenferro-prims/src/cuda/analytic_family.rs`
- Modify: `tenferro-prims/src/cuda/pointwise_ops.rs`
- Test: `tenferro-prims/src/tests/analytic_phase1.rs`
- Test: `tenferro-prims/src/cuda/tests/mod.rs`

**Step 1: Write failing execution tests for representative complex analytic ops**

Cover:

- unary: `Sqrt`, `Exp`, `Log`, `Sin`, `Cos`
- custom unary: `Rsqrt`, `Expm1`, `Log1p`
- binary: `Pow`, `Xlogy`

Also add negative assertions that complex CUDA still rejects:

- `Atan2`
- `Hypot`
- `Var`
- `Std`

**Step 2: Run the targeted tests**

Run:
`cargo test -p tenferro-prims --features cuda analytic_phase1 cuda_scalar_and_analytic_smoke_run_on_device_tensors_when_runtime_is_available -- --nocapture`

Expected:
- FAIL on the new complex analytic cases

**Step 3: Implement truthful complex analytic support**

Implement:

- complex unary dispatch for the CPU-supported subset
- custom complex kernels for the ops that do not map cleanly through cuTENSOR
- complex binary support for `Pow` and `Xlogy`
- capability predicates that keep ordered-real-only ops false for complex

Do not widen support beyond what the CPU family already supports.

**Step 4: Re-run the targeted tests**

Run:
`cargo test -p tenferro-prims --features cuda analytic_phase1 cuda_scalar_and_analytic_smoke_run_on_device_tensors_when_runtime_is_available -- --nocapture`

Expected:
- PASS for the new complex analytic tests

**Step 5: Commit**

```bash
git add tenferro-prims/src/cuda/analytic_family.rs tenferro-prims/src/cuda/pointwise_ops.rs tenferro-prims/src/tests/analytic_phase1.rs tenferro-prims/src/cuda/tests/mod.rs
git commit -m "feat: implement cuda complex analytic primitives"
```

### Task 5: Update docs and support tables to match the new truthful surface

**Files:**
- Modify: `tenferro-prims/README.md`
- Modify: `docs/design/supported-ops.md`
- Modify: `docs/design/tensor-prims.md`

**Step 1: Audit the implemented complex surface against README and support tables**

Confirm the final support matrix matches:

- `has_scalar_support()`
- `has_analytic_support()`
- actual complex smoke coverage

**Step 2: Update the docs**

Document:

- which complex scalar ops are implemented on CUDA
- which complex analytic ops are implemented on CUDA
- which operations remain real-only by design

**Step 3: Run doc generation**

Run:
`cargo doc -p tenferro-prims --features cuda --no-deps`

Expected:
- PASS

**Step 4: Re-run the focused support tests**

Run:
`cargo test -p tenferro-prims --features cuda phase1 -- --nocapture`

Expected:
- PASS

**Step 5: Commit**

```bash
git add tenferro-prims/README.md docs/design/supported-ops.md docs/design/tensor-prims.md
git commit -m "docs: record cuda complex primitive support"
```

### Task 6: Re-audit the implementation invariants and finish verification

**Files:**
- Modify: `tenferro-prims/README.md`
- Modify: `docs/design/supported-ops.md`
- Modify: `docs/design/tensor-prims.md`
- Modify: `docs/plans/2026-03-20-tenferro-prims-cuda-complex-followup.md`

**Step 1: Re-audit the mandatory invariants**

Explicitly verify the completed implementation still satisfies:

- GPU-resident CUDA execution
- no CPU fallback
- truthful `has_*_support()`
- no new public CUDA-only traits or descriptors
- bounded workspace/scratch behavior
- docs and tests aligned with implementation

Record any follow-up gaps directly in this plan file rather than silently
leaving them implied.

**Step 2: Run the verification commands**

Run:
- `cargo fmt --all --check`
- `cargo test -p tenferro-prims`
- `cargo test -p tenferro-prims --features cuda`
- `cargo test -p tenferro-tensor --features cuda`
- `cargo doc -p tenferro-prims --features cuda --no-deps`

Expected:
- PASS

**Step 3: Commit**

```bash
git add tenferro-prims/README.md docs/design/supported-ops.md docs/design/tensor-prims.md docs/plans/2026-03-20-tenferro-prims-cuda-complex-followup.md
git commit -m "chore: verify cuda complex follow-up invariants"
```
