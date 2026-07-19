# Low-risk Overhead Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove three measured-by-construction fixed overheads from tensor accumulation, tensor metadata conversion, and disabled eager profiling without changing observable behavior.

**Architecture:** Add a compact host-output accumulation path that iterates the validated backing slice and retain the indexed path for every other layout. Replace heap-backed intermediate `Vec` conversions with direct `SmallVec` collection, and centralize the optional eager-profile start timestamp behind the existing enabled predicate.

**Tech Stack:** Rust, tenferro tensor/view abstractions, Criterion, Cargo test/Clippy.

---

### Task 1: Compact `dot_general` accumulation

**Files:**
- Modify: `crates/tenferro-tensor/src/backend.rs:1081-1130`
- Modify: `crates/tenferro-tensor/src/tests/backend_default_read_tests.rs:757-894`
- Modify: `crates/tenferro-tensor/Cargo.toml`
- Create: `crates/tenferro-tensor/benches/dot_accumulation.rs`

- [ ] **Step 1: Write a failing compact-overwrite regression test**

Add a test that performs a two-element contraction result into compact output initialized with `NaN`, using `alpha = 2` and `beta = 0`, and asserts the finite scaled result. Add a unit test in `backend.rs` that calls the not-yet-defined `compact_host_accumulation_slice(out)` and expects `Some(&mut [..])` for offset-zero compact host output and `None` for a strided view.

- [ ] **Step 2: Run the test and verify RED**

Run: `cargo test -p tenferro-tensor compact_host_accumulation_slice -- --exact`

Expected: compilation fails because `compact_host_accumulation_slice` does not exist.

- [ ] **Step 3: Implement the compact fast path**

Implement an internal helper with this contract:

```rust
fn compact_host_accumulation_slice<'a, T>(
    out: &'a mut TypedTensorViewMut<'_, T>,
) -> crate::Result<Option<&'a mut [T]>> {
    if out.backend_buffer().is_some() || !out.is_col_major_contiguous()? {
        return Ok(None);
    }
    let start = usize::try_from(out.offset()).map_err(|_| {
        invalid_argument("dot_general", "output", "compact output offset is negative")
    })?;
    let end = start.checked_add(out.n_elements()).ok_or_else(|| {
        validation("dot_general", ValidationError::IntegerOverflow)
    })?;
    out.host_storage_mut()?
        .get_mut(start..end)
        .map(Some)
        .ok_or_else(|| invalid_argument("dot_general", "output", "compact output is out of bounds"))
}
```

Use the returned slice in `accumulate_typed` and apply the same `beta == 0` branch as the fallback. If the helper returns `None`, retain the existing indexed loop unchanged.

- [ ] **Step 4: Verify GREEN and existing fallback behavior**

Run: `cargo test -p tenferro-tensor backend_default_read_tests::dot_general_accum -- --nocapture`

Expected: all matching tests pass, including compact, scalar dtype, and strided-view cases.

- [ ] **Step 5: Add the focused benchmark**

Add a `dot_accumulation` Criterion target that constructs 4096-element compact `f64` dot/output tensors and repeatedly calls `accumulate_dot_result_into` with `alpha = 1`, `beta = 1`, using `iter_batched` so each iteration receives a fresh output.

- [ ] **Step 6: Commit Task 1**

```bash
git add crates/tenferro-tensor/src/backend.rs crates/tenferro-tensor/src/tests/backend_default_read_tests.rs crates/tenferro-tensor/Cargo.toml crates/tenferro-tensor/benches/dot_accumulation.rs
git commit -m "perf(tensor): fast-path compact dot accumulation"
```

### Task 2: Allocation-free inline metadata conversion

**Files:**
- Modify: `crates/tenferro-tensor/src/types.rs:822,1451,4331-4334,4416-4420,4748`
- Modify: `crates/tenferro-tensor/src/tests/types_tests.rs`

- [ ] **Step 1: Add a failing conversion helper test**

Add an internal `collect_shape` helper test that expects an inline `ShapeVec` for rank 2 and an inline `StrideVec` for rank 2. The test must call the not-yet-defined helpers so RED is a compile failure attributable to the missing implementation.

- [ ] **Step 2: Run the test and verify RED**

Run: `cargo test -p tenferro-tensor inline_metadata_collection -- --exact`

Expected: compilation fails because the internal helper is not defined.

- [ ] **Step 3: Implement direct SmallVec collection**

Define focused internal helpers:

```rust
fn shape_vec(shape: &[usize]) -> ShapeVec {
    shape.iter().copied().collect()
}

fn stride_vec(strides: &[isize]) -> StrideVec {
    strides.iter().copied().collect()
}
```

Replace every `.to_vec().into()` in `crates/tenferro-tensor/src/types.rs` with these helpers. Preserve each existing `TensorLayout`, `shape_from_vec`, `strides_from_vec`, and error mapping call.

- [ ] **Step 4: Verify GREEN**

Run: `cargo test -p tenferro-tensor inline_metadata_collection -- --nocapture && cargo test -p tenferro-tensor types_tests -- --nocapture`

Expected: metadata helper and existing shape/layout/view tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add crates/tenferro-tensor/src/types.rs crates/tenferro-tensor/src/tests/types_tests.rs
git commit -m "perf(tensor): avoid heap metadata round trips"
```

### Task 3: Disabled eager profiling clock gate

**Files:**
- Modify: `crates/tenferro-ad/src/eager.rs:121-153`
- Modify: `crates/tenferro-ad/src/eager_ops.rs:1114-1196`
- Modify: `crates/tenferro-ad/src/eager/tests.rs:446-460`

- [ ] **Step 1: Write a failing disabled-profile test**

Extend the profiling helper imports and add a test that installs `EagerOpProfileOverrideGuard::set(false, None)` and asserts `eager_op_profile_start().is_none()`. Extend the enabled-path test to assert the same helper returns `Some`.

- [ ] **Step 2: Run the test and verify RED**

Run: `cargo test -p tenferro-ad eager_op_profile_start_respects_enabled_gate -- --exact`

Expected: compilation fails because `eager_op_profile_start` does not exist.

- [ ] **Step 3: Implement the lazy clock gate**

Add:

```rust
pub(crate) fn eager_op_profile_start() -> Option<Instant> {
    eager_op_profile_enabled().then(Instant::now)
}
```

In `nary_op`, replace the unconditional `Instant::now()` with this helper and only record `nary_op.total` and call `maybe_print_eager_op_profile()` inside `if let Some(total_started)`.

- [ ] **Step 4: Verify GREEN**

Run: `cargo test -p tenferro-ad eager_op_profile -- --nocapture`

Expected: enabled and disabled profiling helper tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add crates/tenferro-ad/src/eager.rs crates/tenferro-ad/src/eager_ops.rs crates/tenferro-ad/src/eager/tests.rs
git commit -m "perf(ad): skip clock reads when profiling is disabled"
```

### Task 4: Verification and PR

**Files:**
- Modify only if verification reveals a defect in the files above.

- [ ] **Step 1: Format and inspect**

Run: `cargo fmt --all -- --check && git diff --check`

Expected: both commands exit successfully.

- [ ] **Step 2: Run targeted tests and Clippy**

Run: `cargo test -p tenferro-tensor -p tenferro-ad && cargo clippy -p tenferro-tensor -p tenferro-ad --all-targets -- -D warnings`

Expected: all tests pass and Clippy emits no warnings.

- [ ] **Step 3: Compile and sample the benchmark**

Run: `cargo bench -p tenferro-tensor --bench dot_accumulation --no-run`

Expected: benchmark target compiles successfully.

- [ ] **Step 4: Run repository pre-PR verification required by `AGENTS.md`**

Run the repository's documented bugfix/pre-PR command set and record exact outcomes in the PR body.

- [ ] **Step 5: Push and open one batch PR**

Push `codex/issue-1426-low-risk-overheads` and open a PR referencing #1426. Explain that public behavior is unchanged, identify the compact fallback boundary, and include test and benchmark evidence.
