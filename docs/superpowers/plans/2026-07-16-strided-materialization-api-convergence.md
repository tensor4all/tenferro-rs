# Strided Materialization API Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make backend-owned `strided-rs` copying the only CPU implementation for tensor-view materialization, copy-back, and structural read paths.

**Architecture:** `tenferro-tensor` retains layout metadata and object-safe backend contracts but no public context-free tensor-sized copy loops. `tenferro-cpu` converts validated tenferro views to `strided_kernel` views, allocates through its `BufferPool`, and runs every copy inside `CpuContext`; CPU structural read operations reuse that helper without intermediate materialization. CUDA keeps same-device kernels behind the same renamed copy contract.

**Tech Stack:** Rust 2021, `tenferro-tensor`, `tenferro-cpu`, `tenferro-gpu`, `strided-kernel 0.3`, Rayon through `CpuContext`, Criterion, Cargo tests/doctests.

---

### Task 1: Establish one pool-aware CPU view-copy primitive

**Files:**
- Modify: `crates/tenferro-cpu/src/lib.rs`
- Modify: `crates/tenferro-cpu/src/structural.rs`
- Test: `crates/tenferro-cpu/src/tests/cpu_tests/backend_misc.rs`
- Test: `crates/tenferro-cpu/src/tests/cpu_stub_tests.rs`

- [ ] **Step 1: Write failing materialization tests**

Add tests that call a pool-aware CPU materializer on compact, transposed,
negative-stride, zero-stride broadcast, empty, rank-zero, and nonzero-offset
views. Include a 24-axis scattered layout with explicit positive strides and
compare exact column-major output against logical `get` traversal. Exercise all
seven dtype variants through `TensorView`.

The core regression should use the public backend path:

```rust
let mut backend = CpuBackend::with_threads(4).unwrap();
let view = tensor.as_view().transpose_view(&perm).unwrap();
let compact = backend.to_contiguous(&view).unwrap();
assert_eq!(compact.as_slice().unwrap(), expected.as_slice());
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
cargo test -p tenferro-cpu --release cpu_view_materialization -- --nocapture
```

Expected: FAIL because `CpuBackend::to_contiguous` still delegates to the
serial `TypedTensorView::to_contiguous` path and the new source-contract
assertion cannot find a `strided_kernel::copy_into` materializer.

- [ ] **Step 3: Implement the typed strided copy helpers**

In `structural.rs`, add crate-private helpers with these responsibilities:

```rust
pub(crate) fn typed_materialize_view_with_pool<T, R>(
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T, R>,
    op: &'static str,
) -> crate::Result<TypedTensor<T, R>>
where
    T: Copy + Clone + PoolScalar + 'static,
    R: TensorRank;

pub(crate) fn typed_copy_view_into<T, RS, RD>(
    src: &TypedTensorView<'_, T, RS>,
    dst: &mut TypedTensorViewMut<'_, T, RD>,
    op: &'static str,
) -> crate::Result<()>
where
    T: Copy + Send + Sync + 'static,
    RS: TensorRank,
    RD: TensorRank;
```

Both helpers must construct `StridedView`/`StridedViewMut` from the existing
shape, strides, offset, and host allocation, validate shape equality, map
`strided-kernel` errors to the supplied operation name, and call
`strided_kernel::copy_into`. Allocate the materialized destination with
`typed_array_uninit_from_pool`; preserve rank and placement metadata when
wrapping the output.

Replace `materialize_tensor_read` with a `BufferPool`-accepting dtype dispatcher
that invokes `typed_materialize_view_with_pool` for views and uses the existing
compact clone path for owned tensors.

- [ ] **Step 4: Route `CpuBackend` and `CpuExecSession` through the helper**

Use `install_with_pool`/`run_native` so both public backend and session paths run
inside the configured `CpuContext` and persistent pool. Do not construct a
temporary `CpuBackend` or `BufferPool`.

- [ ] **Step 5: Run focused tests and verify GREEN**

```bash
cargo test -p tenferro-cpu --release cpu_view_materialization -- --nocapture
cargo test -p tenferro-cpu --release materialize_tensor_read -- --nocapture
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/tenferro-cpu/src/lib.rs crates/tenferro-cpu/src/structural.rs \
  crates/tenferro-cpu/src/tests/cpu_tests/backend_misc.rs \
  crates/tenferro-cpu/src/tests/cpu_stub_tests.rs
git commit -m "perf(cpu): materialize views through strided copy"
```

### Task 2: Eliminate double materialization in structural read operations

**Files:**
- Modify: `crates/tenferro-cpu/src/structural.rs`
- Modify: `crates/tenferro-cpu/src/backend.rs`
- Modify: `crates/tenferro-cpu/src/exec_session.rs`
- Test: `crates/tenferro-cpu/src/tests/cpu_tests/backend_misc.rs`
- Test: `crates/tenferro-cpu/tests/backend_capability_contracts.rs`

- [ ] **Step 1: Write failing direct-view tests**

Add exact-output tests for `transpose_read`, `reshape_read`, and
`broadcast_in_dim_read` using explicit-stride `TensorView` inputs. Add a
source-contract test requiring those methods to dispatch to typed view helpers
and forbidding `materialize_tensor_read("transpose"`,
`materialize_tensor_read("reshape"`, and
`materialize_tensor_read("broadcast_in_dim"` in their bodies.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
cargo test -p tenferro-cpu --release structural_read -- --nocapture
cargo test -p tenferro-cpu --test backend_capability_contracts --release
```

Expected: the source contract fails because all three view paths currently
materialize an intermediate tensor.

- [ ] **Step 3: Add dtype-dispatched direct view helpers**

Add these helpers in `structural.rs`:

```rust
pub(crate) fn transpose_read_with_pool(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
    perm: &[usize],
) -> crate::Result<Tensor>;

pub(crate) fn reshape_read_with_pool(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
    shape: &[usize],
) -> crate::Result<Tensor>;

pub(crate) fn broadcast_in_dim_read_with_pool(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
    shape: &[usize],
    dims: &[usize],
) -> crate::Result<Tensor>;
```

`transpose_read_with_pool` calls `typed_transpose_view_with_pool` directly.
`reshape_read_with_pool` validates element counts, copies the source logical
order once into compact pooled storage, then attaches the requested compact
shape. `broadcast_in_dim_read_with_pool` builds aligned dimensions and strides
from the original view, uses `StridedView::broadcast`, and copies directly to
the final output.

- [ ] **Step 4: Share helpers across backend and execution session**

Replace the duplicated `TensorStructural` overrides in `backend.rs` and
`exec_session.rs` with calls to the three helpers inside their existing
execution-resource scopes.

- [ ] **Step 5: Verify GREEN and commit**

```bash
cargo test -p tenferro-cpu --release structural_read -- --nocapture
cargo test -p tenferro-cpu --test backend_capability_contracts --release
git add crates/tenferro-cpu/src/structural.rs crates/tenferro-cpu/src/backend.rs \
  crates/tenferro-cpu/src/exec_session.rs \
  crates/tenferro-cpu/src/tests/cpu_tests/backend_misc.rs \
  crates/tenferro-cpu/tests/backend_capability_contracts.rs
git commit -m "perf(cpu): consume strided views in structural reads"
```

### Task 3: Replace asymmetric copy-back with backend `copy_into`

**Files:**
- Modify: `crates/tenferro-tensor/src/backend.rs`
- Modify: `crates/tenferro-tensor/src/lib.rs`
- Modify: `crates/tenferro-cpu/src/backend.rs`
- Modify: `crates/tenferro-gpu/src/cubecl/mod.rs`
- Test: `crates/tenferro-cpu/src/tests/cpu_stub_tests.rs`
- Test: `crates/tenferro-gpu/src/cubecl/tests/structural_tests.rs`
- Test: `crates/tenferro-tensor/src/tests/public_surface_contract_tests.rs`

- [ ] **Step 1: Write failing public-contract and behavior tests**

Require `TensorViewCanonicalization::copy_into` to accept readable and writable
views, including strided source and destination layouts. Assert that
`copy_from_contiguous` is absent from the public trait. Cover shape mismatch,
CPU/backend-buffer rejection, CUDA host/device rejection, and exact transpose
copy-back behavior.

- [ ] **Step 2: Verify RED**

```bash
cargo test -p tenferro-tensor --release public_surface_contract -- --nocapture
cargo test -p tenferro-cpu --release canonicalization -- --nocapture
```

Expected: compile/source-contract failure because only
`copy_from_contiguous` exists.

- [ ] **Step 3: Change the typed capability contract**

Tighten the trait scalar bound from `T: Clone` to `T: TensorScalar` and replace
the copy method with:

```rust
fn copy_into(
    &mut self,
    src: &TypedTensorView<'_, T, R>,
    dst: &mut TypedTensorViewMut<'_, T, R>,
) -> crate::Result<()>;
```

The typed trait continues to require matching rank parameters. The dtype-erased
runtime operation in Task 4 handles dynamically ranked views. Do not restore a
serial fallback.

- [ ] **Step 4: Implement CPU and CUDA adapters**

CPU delegates to `typed_copy_view_into` inside `CpuContext::install`. CUDA
renames the existing contiguous-to-view kernel path and validates that its
source is compact; arbitrary-stride CUDA source-to-destination copying is not
added in this CPU-focused change. Preserve same-device and no-hidden-transfer
errors.

- [ ] **Step 5: Verify GREEN and commit**

```bash
cargo test -p tenferro-tensor --release public_surface_contract -- --nocapture
cargo test -p tenferro-cpu --release canonicalization -- --nocapture
cargo test -p tenferro-gpu --release structural_tests -- --nocapture
git add crates/tenferro-tensor/src/backend.rs crates/tenferro-tensor/src/lib.rs \
  crates/tenferro-cpu/src/backend.rs crates/tenferro-gpu/src/cubecl/mod.rs \
  crates/tenferro-cpu/src/tests/cpu_stub_tests.rs \
  crates/tenferro-gpu/src/cubecl/tests/structural_tests.rs \
  crates/tenferro-tensor/src/tests/public_surface_contract_tests.rs
git commit -m "refactor(tensor): make backend copy_into canonical"
```

### Task 4: Make runtime and eager materialization backend-owned

**Files:**
- Modify: `crates/tenferro-tensor/src/backend.rs`
- Modify: `crates/tenferro-tensor/src/types.rs`
- Modify: `crates/tenferro-runtime/src/exec.rs`
- Modify: `crates/tenferro-runtime/src/graph/executor.rs`
- Modify: `crates/tenferro-runtime/src/graph/executor/tests.rs`
- Modify: `crates/tenferro-runtime/tests/runtime_public_api.rs`
- Modify: `crates/tenferro-ad/src/eager.rs`
- Modify: `crates/tenferro-ad/src/eager_exec.rs`
- Modify: `crates/tenferro-ad/src/eager_ops.rs`
- Modify: `crates/tenferro-ad/tests/eager_tensor.rs`
- Modify: `crates/tenferro-ad/tests/numpy_api.rs`
- Modify: `crates/tenferro-ad/tests/segment_tests.rs`
- Modify: `crates/tenferro-einsum/src/eager.rs`
- Modify: `crates/tenferro-einsum/tests/traced_extension.rs`
- Modify: `crates/tenferro-einsum/tests/traced_graph_cache.rs`
- Modify: `crates/tenferro-fft/src/lib.rs`
- Modify: `crates/tenferro-tensor/src/tests/backend_default_read_tests.rs`
- Modify: `crates/tenferro-tensor/src/tests/types_tests.rs`
- Modify: `crates/tenferro-tensor/src/tests/types_tests/strided_dynamic.rs`
- Test: `crates/tenferro-ad/tests/eager_tensor.rs`
- Test: `crates/tenferro-runtime/tests/runtime_public_api.rs`

- [ ] **Step 1: Write failing backend-owned materialization tests**

Add tests proving that lazy `EagerTensor::to_tensor` and graph/runtime result
materialization invoke a backend session canonicalizer. Use a recording backend
whose materialization method increments a counter; assert one call for a lazy
view and zero calls for an already-owned tensor.

- [ ] **Step 2: Verify RED**

```bash
cargo test -p tenferro-ad --release eager_materialization_uses_backend -- --nocapture
cargo test -p tenferro-runtime --release runtime_materialization_uses_backend -- --nocapture
```

Expected: tests fail because `TensorValue::to_tensor` and `TensorRead::to_tensor`
currently materialize without a backend.

- [ ] **Step 3: Add an object-safe runtime materialization operation**

Add to `TensorStructural`:

```rust
fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor>;
fn copy_read_into(
    &mut self,
    src: TensorRead<'_>,
    dst: TensorWrite<'_>,
) -> crate::Result<()>;
```

Provide conservative defaults that clone an owned compact input and reject
unsupported views with the existing read-boundary error. Override both methods
in CPU and CUDA backends/sessions. CPU dispatches to Tasks 1 and 3; CUDA
dispatches to its same-device typed kernels.

- [ ] **Step 4: Remove context-free data-moving methods**

Delete the public methods and their serial helpers:

```text
TypedTensorView::to_contiguous
TypedTensorViewMut::copy_from_contiguous
TensorView::to_tensor
TensorRead::to_tensor
TensorOwnedView::to_tensor
TensorValue::to_tensor
materialize_view_buffer_col_major
materialize_typed_view_col_major
```

Keep metadata-only methods and scalar indexed access. Update rustdoc links and
public-surface tests so no context-free equivalent remains.

- [ ] **Step 5: Migrate runtime/eager/extension callers**

Pass the active `dyn BackendSession` to materialization sites. Add
`GraphExecutor::materialize_value(&TensorValue) -> Result<Tensor>` for public
graph results and update examples to call it. Keep
`EagerTensor::to_tensor` as a public no-argument method because its context owns
the backend; implement it through `with_backend_session`. Ensure graph executor
internal result APIs materialize through their executor-owned backend before
returning compact output where their existing public API promises `Tensor`.
Update FFT/einsum fallback paths to call their active backend/session rather
than tensor-type methods.

- [ ] **Step 6: Verify GREEN and commit**

```bash
cargo test -p tenferro-tensor --release
cargo test -p tenferro-runtime --release
cargo test -p tenferro-ad --release
cargo test -p tenferro-einsum --release
cargo test -p tenferro-fft --release
git add crates/tenferro-tensor crates/tenferro-runtime crates/tenferro-ad \
  crates/tenferro-einsum crates/tenferro-fft
git commit -m "refactor(runtime): require backend-owned materialization"
```

### Task 5: Record ownership policy and add focused benchmarks

**Files:**
- Modify: `REPOSITORY_RULES.md`
- Modify: `docs/guides/parallelism-and-caching.md`
- Modify: `docs/guides/eager-operations.md`
- Create: `crates/tenferro-cpu/benches/view_materialization.rs`
- Modify: `crates/tenferro-cpu/Cargo.toml`
- Create: `docs/worklogs/2026-07-16-strided-materialization-api-convergence.md`
- Test: `crates/tenferro-cpu/tests/backend_capability_contracts.rs`

- [ ] **Step 1: Write failing ownership source-contract test**

Require repository rules to state that CPU affine-strided copy, permutation,
broadcast, map, zip-map, and axis reduction delegate to `strided-rs`; record
einsum as the benchmark-backed exception. Require context-free materialization
method names to be absent from `tenferro-tensor` source.

The source contract must also require the rule rationale: CPU materialization
enters through `CpuBackend` because high-performance execution needs its
persistent `BufferPool`, fully-overwritten uninitialized output allocation,
configured `CpuContext` Rayon pool, nested-execution safety, and
serial/parallel threshold policy. Merely calling `strided-rs` from a
context-free method or on Rayon's ambient global pool is explicitly
non-compliant.

- [ ] **Step 2: Verify RED**

```bash
cargo test -p tenferro-cpu --test backend_capability_contracts --release
```

Expected: FAIL until rules and removed-surface assertions are updated.

- [ ] **Step 3: Update rules and user documentation**

Document the ownership table from the approved design, the backend-required
materialization API, CPU threading/pool behavior, and the no-hidden-transfer
contract. Explain why CPU host copies still require backend ownership: memory
reuse and thread policy are execution resources, not tensor metadata. Ban
throwaway pools and ambient-global-Rayon execution for public tensor-sized CPU
operations. Remove examples using context-free `to_contiguous`/`to_tensor`.

- [ ] **Step 4: Add a non-asserting Criterion benchmark**

Benchmark `CpuBackend::to_contiguous` at one and four threads for:

- compact 3D input;
- permuted 3D input;
- high-rank contiguous permutation;
- the scattered 24D explicit-stride layout;
- a tiny transpose to expose dispatch overhead.

Allocate the destination inside each timed iteration and verify exact output
once before timing. Register the bench with `harness = false` in the CPU crate
manifest.

- [ ] **Step 5: Write the worklog, verify, and commit**

```bash
cargo test -p tenferro-cpu --test backend_capability_contracts --release
cargo bench -p tenferro-cpu --bench view_materialization --no-run
git add REPOSITORY_RULES.md docs/guides crates/tenferro-cpu/Cargo.toml \
  crates/tenferro-cpu/benches/view_materialization.rs \
  crates/tenferro-cpu/tests/backend_capability_contracts.rs \
  docs/worklogs/2026-07-16-strided-materialization-api-convergence.md
git commit -m "docs(cpu): contract strided kernel ownership"
```

### Task 6: Full verification

**Files:**
- Review: all files changed since `origin/main`

- [ ] **Step 1: Run formatting and clippy parity**

```bash
cargo fmt --all --check
```

Read `.github/workflows` and run the exact non-GPU clippy command used by CI.
Expected: exit 0 with no warnings.

- [ ] **Step 2: Run the full release test and docs gates**

```bash
cargo test --workspace --release
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: all commands exit 0.

- [ ] **Step 3: Run coverage**

```bash
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

Expected: coverage policy passes for every modified source file.

- [ ] **Step 4: Run repository-rule review**

```bash
python3 scripts/repository-rules-review.py \
  --base origin/main \
  --head HEAD \
  --output-json /tmp/repository-rules-review.json
```

Read the JSON and fix every actionable finding or record a justified residual
risk in the worklog.

- [ ] **Step 5: Inspect final diff and status**

```bash
git diff --check origin/main...HEAD
git status --short
git log --oneline origin/main..HEAD
```

Expected: no whitespace errors, no uncommitted files other than explicitly
reported generated coverage output, and coherent commits matching Tasks 1-5.
