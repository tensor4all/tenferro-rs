# Eager Tidu Recorder Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update `EagerTensor` eager AD recording to use `tidu::record_eager_op`.

**Architecture:** `tenferro` keeps concrete execution and gradient storage, while `tidu` owns generic eager AD metadata construction. A shared `EagerTensor` helper will convert concrete inputs and outputs into `tidu` recorder types, register tenferro metadata for replay keys, and return eager result tensors.

**Tech Stack:** Rust 2021, `tidu` eager recorder API at revision `97e4ae1`, `computegraph` keys, `tenferro::EagerTensor`, `tenferro-ops::StdTensorOp`.

---

### Task 1: Add Regression Coverage For Multi-Output Eager Backward

**Files:**
- Modify: `tenferro/tests/eager_linalg.rs`

**Step 1: Write the failing test**

Add a test that differentiates through the second output of a multi-output op. This guards the output-key mapping that `tidu::record_eager_op` now owns.

```rust
#[test]
fn qr_second_output_backward_records_selected_output_slot() {
    let a = EagerTensor::requires_grad(Tensor::from_vec(
        vec![2, 2],
        vec![1.0_f64, 0.0, 0.0, 2.0],
    ));
    let (_q, r) = a.qr().unwrap();
    let loss = r.reduce_sum(&[0, 1]).unwrap();

    let _cotangents = loss.backward().unwrap();

    let grad = a.grad().expect("gradient for qr input");
    assert_eq!(grad.shape(), &[2, 2]);
}
```

**Step 2: Run test to verify current behavior**

Run:

```bash
cargo test -p tenferro --test eager_linalg qr_second_output_backward_records_selected_output_slot
```

Expected before implementation: pass on the old manual path. It is a regression guard, not necessarily a red test.

**Step 3: Commit test if committing incrementally**

```bash
git add tenferro/tests/eager_linalg.rs
git commit -m "test: cover eager multi-output backward slot recording"
```

### Task 2: Update Tidu Dependency

**Files:**
- Modify: `Cargo.toml`
- Modify: `Cargo.lock`

**Step 1: Update workspace dependency**

Change the workspace dependency:

```toml
tidu = { git = "https://github.com/tensor4all/tidu-rs.git", rev = "97e4ae1" }
```

**Step 2: Refresh lockfile**

Run:

```bash
cargo update -p tidu
```

Expected: `Cargo.lock` points `tidu` at commit `97e4ae1d80e7b35dedc330d31ad15c2eef921a86`.

**Step 3: Run a compile check**

Run:

```bash
cargo check -p tenferro
```

Expected before code migration: compile errors around manual `GradNode` construction or private `GradNode` fields.

### Task 3: Replace Manual Eager Recording With Tidu Recorder

**Files:**
- Modify: `tenferro/src/eager.rs`
- Modify: `tenferro/src/eager_ops.rs`

**Step 1: Add key source and input conversion helpers**

In `tenferro/src/eager.rs`, import recorder types and add:

```rust
pub(crate) struct EagerKeySource;

impl tidu::EagerKeySource<StdTensorOp> for EagerKeySource {
    fn fresh_input_key(&mut self) -> tenferro_ops::input_key::TensorInputKey {
        next_input_key()
    }
}
```

Add a helper that converts an eager tensor to `tidu::EagerValue<StdTensorOp>`:

```rust
pub(crate) fn eager_value<B: TensorBackend>(
    tensor: &EagerTensor<B>,
) -> tidu::EagerValue<StdTensorOp> {
    tidu::EagerValue {
        key: tensor.key.clone(),
        node: tensor.grad_node.clone(),
        requires_grad: tensor.requires_grad,
        data: Arc::clone(&tensor.data),
    }
}
```

**Step 2: Register recorder metadata**

Add a helper that accepts an op, input tensors, and `Arc<Tensor>` outputs. It calls `record_eager_op`, then registers metadata for:

- each `EagerOutput::key`,
- each key/value pair in the shared `GradNode::saved_data()` when a node exists.

Use `tensor_meta_from_tensor(value.as_ref())` for each saved value.

**Step 3: Refactor `nary_op`**

In `tenferro/src/eager_ops.rs`, keep the current context merge and concrete execution. Replace manual `GradNode` construction, input alias generation, `saved_forward_values`, and `derived_output_key` calls with the shared recorder helper.

**Step 4: Refactor `multi_output_unary_op`**

In `tenferro/src/eager_ops.rs`, convert the concrete outputs to `Arc<Tensor>`, call the same recorder helper, and build each result from the matching `EagerOutput`.

**Step 5: Delete obsolete helpers**

Remove these local helpers from `tenferro/src/eager.rs` if no longer used:

```rust
saved_forward_values
saved_forward_values_multi
derived_output_key
```

Also remove now-unused imports such as `GlobalOpKey` and `OpMode`.

**Step 6: Adapt backward callbacks to `GradNode` accessors**

If compile errors remain in `tidu::backward_dag` integration, update call sites to use the new `GradNode` accessor methods rather than public fields. Prefer changing only tenferro-owned code.

### Task 4: Verify Focused Eager AD Behavior

**Files:**
- Test: `tenferro/tests/eager_tensor.rs`
- Test: `tenferro/tests/eager_linalg.rs`
- Test: `tenferro/tests/eager_einsum_ad.rs`

**Step 1: Run focused tests**

Run:

```bash
cargo test -p tenferro --test eager_tensor
cargo test -p tenferro --test eager_linalg
cargo test -p tenferro --test eager_einsum_ad
```

Expected: all pass.

**Step 2: Fix regressions minimally**

If a test fails, inspect whether metadata for saved replay values or output keys is missing. Fix the shared recorder helper rather than adding per-op special cases.

### Task 5: Format And Broader Check

**Files:**
- Modify only files touched by previous tasks.

**Step 1: Format**

Run:

```bash
cargo fmt --all
```

**Step 2: Check formatting**

Run:

```bash
cargo fmt --all --check
```

Expected: pass.

**Step 3: Run package tests**

Run:

```bash
cargo test -p tenferro
```

Expected: pass. If unrelated dirty einsum changes cause failures, report that explicitly and keep this change scoped.
