# Binary Strict Lowering Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the bridge-local binary dense primal GEMM lowering into `tenferro-einsum`, validate CPU performance, and remove the bridge-only fast path.

**Architecture:** Add an internal strict binary lowering plan/executor in `tenferro-einsum` that performs direct permute/contiguous/reshape/GEMM/permute lowering for the narrow dense primal binary case. Route binary APIs through this strict path first with generic fallback, then delete the bridge-local helper so tensor4all always calls tenferro-owned binary einsum.

**Tech Stack:** Rust, `tenferro-einsum`, `tenferro-prims` semiring core GEMM, `tensor4all-tensorbackend`, `cargo nextest`, release-mode benchmark tests.

---

### Task 1: Add strict binary lowering tests in `tenferro-einsum`

**Files:**
- Modify: `tenferro-einsum/src/api/binary/tests.rs`
- Modify: `tenferro-einsum/src/planning/plan/tests.rs`
- Create or modify: `tenferro-einsum/tests/bench_binary_path_breakdown.rs`

**Step 1: Write the failing tests**

Add tests that:
- prove a strict binary lowering plan is built for a simple dense matmul case
- prove repeated labels reject strict lowering
- prove strict lowering and generic binary einsum produce identical results when output permutation is non-identity

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-einsum --lib api::binary::tests
cargo test -p tenferro-einsum --lib planning::plan::tests
```

Expected:
- compile or assertion failures because strict lowering helpers do not exist yet

**Step 3: Add a micro-benchmark hook**

Extend `tenferro-einsum/tests/bench_binary_path_breakdown.rs` so it can compare:
- generic binary API
- strict binary API path
- generic with prebuilt plan

Do not optimize yet; just make the benchmark file ready to observe the new path.

**Step 4: Re-run the focused tests**

Run the same `cargo test` commands and confirm the failures are still the intended missing-feature failures.

**Step 5: Commit**

```bash
git add tenferro-einsum/src/api/binary/tests.rs tenferro-einsum/src/planning/plan/tests.rs tenferro-einsum/tests/bench_binary_path_breakdown.rs
git commit -m "test: add strict binary lowering coverage"
```

### Task 2: Implement strict binary lowering inside `tenferro-einsum`

**Files:**
- Modify: `tenferro-einsum/src/planning/plan.rs`
- Modify: `tenferro-einsum/src/api/binary.rs`
- Modify: `tenferro-einsum/src/api/borrowed.rs`
- Modify: `tenferro-einsum/src/execution/mod.rs`
- Create: `tenferro-einsum/src/execution/strict_binary.rs`

**Step 1: Write the failing test for execution routing**

Add a focused unit test showing that a valid strict binary case routes through the strict lowering executor and not the generic binary path. Keep the test narrow and internal.

**Step 2: Run the focused test to verify it fails**

Run:

```bash
cargo test -p tenferro-einsum --lib api::binary::tests::strict_binary_ -- --nocapture
```

Expected:
- fail because the route or helper is not implemented

**Step 3: Implement the minimal planning type**

In `tenferro-einsum/src/planning/plan.rs`, add:
- `StrictBinaryLoweringPlan`
- strict eligibility checks
- computed permutations, fused dims, canonical output dims, and final output permutation

Do not broaden semantics beyond the bridge helper’s narrow case.

**Step 4: Implement the minimal executor**

In `tenferro-einsum/src/execution/strict_binary.rs`, implement:
- permute if needed
- contiguous if needed
- reshape into `[m, k]` / `[k, n]`
- `TensorSemiringCore::BatchedGemm`
- reshape back and final output permute

Use existing tensor APIs and semiring core GEMM only.

**Step 5: Route binary APIs**

Update:
- `tenferro-einsum/src/api/binary.rs`
- `tenferro-einsum/src/api/borrowed.rs`

so that:
- binary APIs try strict lowering first
- if strict lowering is unavailable, they fall back to the existing generic binary path
- `einsum_with_subscripts(len == 2)` follows the same rule

**Step 6: Run focused tests**

Run:

```bash
cargo test -p tenferro-einsum --lib api::binary::tests
cargo test -p tenferro-einsum --lib planning::plan::tests
```

Expected:
- all focused tests pass

**Step 7: Run broader library verification**

Run:

```bash
cargo nextest run --release -p tenferro-einsum --lib
```

Expected:
- all `tenferro-einsum` library tests pass

**Step 8: Commit**

```bash
git add tenferro-einsum/src/planning/plan.rs tenferro-einsum/src/api/binary.rs tenferro-einsum/src/api/borrowed.rs tenferro-einsum/src/execution/mod.rs tenferro-einsum/src/execution/strict_binary.rs
git commit -m "feat: add strict binary einsum lowering"
```

### Task 3: Remove the bridge-local binary fast path

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`
- Modify: `crates/tensor4all-tensorbackend/src/tenferro_bridge/tests/mod.rs`

**Step 1: Write the failing bridge test**

Add or update a bridge-side test that verifies dense primal binary einsum still returns correct results after removing the bridge helper. The test must use the public bridge entry point, not tenferro internals.

**Step 2: Run the bridge test to verify it fails**

Run:

```bash
cargo test -p tensor4all-tensorbackend --lib tenferro_bridge::tests::einsum_native_tensors_dense_primal_
```

Expected:
- fail because the bridge still depends on the removed helper path

**Step 3: Delete bridge-local lowering**

In `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`:
- remove `binary_dense_primal_gemm_with_ids`
- remove its binary-only call sites
- always route binary native einsum through tenferro binary APIs

Keep behavior identical for non-fast-path cases.

**Step 4: Run bridge tests**

Run:

```bash
cargo test -p tensor4all-tensorbackend --lib tenferro_bridge::tests
```

Expected:
- bridge tests pass

**Step 5: Commit**

```bash
git add crates/tensor4all-tensorbackend/src/tenferro_bridge.rs crates/tensor4all-tensorbackend/src/tenferro_bridge/tests/mod.rs
git commit -m "refactor: remove bridge-local binary einsum lowering"
```

### Task 4: Benchmark and regressions

**Files:**
- Modify if needed: `tenferro-einsum/tests/bench_binary_path_breakdown.rs`
- Modify if needed: `crates/tensor4all-tensorbackend/tests/bench_einsum_native.rs`

**Step 1: Run binary micro-benchmark**

Run:

```bash
cargo test --release -p tenferro-einsum --test bench_binary_path_breakdown bench_binary_path_breakdown -- --ignored --exact --nocapture --test-threads=1
```

Expected:
- strict binary path is at least as fast as the former generic binary path
- ideally matches or improves on the bridge-era binary timing

**Step 2: Run downstream fit benchmark**

Run:

```bash
cargo --config .cargo/local-tenferro.toml test --release -p tensor4all-tensorbackend --test bench_einsum_native bench_native_einsum_fit_patterns -- --ignored --exact --nocapture --test-threads=1
```

Expected:
- benchmark completes successfully
- pairwise/generic ratios are now comparing tenferro paths, not bridge vs tenferro paths

**Step 3: Run final formatting and verification**

Run:

```bash
cargo fmt --all
cargo nextest run --release -p tenferro-einsum --lib
cargo nextest run --release -p tenferro-prims --lib
```

Expected:
- all commands pass

**Step 4: Summarize residual gap**

If generic pairwise still leads, document whether the remaining gap is now:
- strict binary executor vs generic n-ary executor
- or something else in `tensor4all` bridge / native tensor handling

Do not start GPU work in this task.

**Step 5: Commit**

```bash
git add tenferro-einsum/tests/bench_binary_path_breakdown.rs crates/tensor4all-tensorbackend/tests/bench_einsum_native.rs
git commit -m "test: benchmark strict binary einsum lowering"
```
