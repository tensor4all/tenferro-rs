# Borrow-Aware Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a borrowed-input execution path so `GraphExecutor` can run from `TensorRead` inputs without cloning caller-owned tensors.

**Architecture:** Introduce an internal `ExecSlot<'a>` that can hold either an owned `Tensor` or a borrowed `TensorRead<'a>`. Keep the existing owned APIs intact, route read-capable dispatch through backend `_read` methods, and reclaim only owned slots.

**Tech Stack:** Rust, `tenferro-runtime`, `tenferro-tensor`, CPU backend tests, existing cargo test suite.

---

### Task 1: Public Borrowed Input API Tests

**Files:**
- Modify: `tenferro-runtime/tests/runtime_public_api.rs`

- [ ] **Step 1: Write failing tests**

Add tests that call the new `run_many_with_input_reads` API for elementwise/reduction and dot-general:

```rust
use tenferro_runtime::{TensorRead};

#[test]
fn graph_executor_runs_elementwise_and_reduction_with_borrowed_inputs() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let y = (&x + &x).reduce_sum(&[0]);
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
        .unwrap();
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let out = executor
        .run_many_with_input_reads(&program, &[(&x, TensorRead::from_tensor(&input))])
        .unwrap();

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].as_slice::<f64>().unwrap(), &[6.0]);
    assert_eq!(input.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
}

#[test]
fn graph_executor_runs_dot_general_with_borrowed_inputs() {
    let lhs = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let rhs = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let product = lhs.dot_general(
        &rhs,
        tenferro_runtime::DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    );
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(
            &product,
            &[(&lhs, DType::F64, &[2, 3]), (&rhs, DType::F64, &[3, 2])],
        )
        .unwrap();
    let lhs_data = Tensor::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs_data = Tensor::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let out = executor
        .run_many_with_input_reads(
            &program,
            &[
                (&lhs, TensorRead::from_tensor(&lhs_data)),
                (&rhs, TensorRead::from_tensor(&rhs_data)),
            ],
        )
        .unwrap();

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].shape(), &[2, 2]);
    assert_eq!(out[0].as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
}
```

- [ ] **Step 2: Run tests and verify red**

Run: `cargo test -p tenferro-runtime --test runtime_public_api graph_executor_runs_ --no-default-features --features cpu-faer`

Expected: fail because `GraphExecutor::run_many_with_input_reads` does not exist.

### Task 2: Borrow-Aware Runtime Slots

**Files:**
- Modify: `tenferro-runtime/src/exec.rs`

- [ ] **Step 1: Add `ExecSlot<'a>` and helpers**

Add an internal enum plus helpers for read access, shape resolution, output collection, and owned-only reclaim:

```rust
pub(crate) enum ExecSlot<'a> {
    Owned(Tensor),
    Read(tenferro_tensor::TensorRead<'a>),
}
```

Helpers must support:

- `initialize_slots_in` for owned `Vec<Tensor>`.
- `initialize_read_slots_in` for borrowed `Vec<TensorRead<'a>>`.
- `get_read` returning `TensorRead<'_>`.
- `get_tensor` returning `&Tensor` for owned slots and borrowed tensor refs.
- `resolve_tensor_shape_exprs` from either slot variant.
- `collect_outputs_from` materializing borrowed final outputs only when needed.
- last-use reclaim functions that reclaim only `ExecSlot::Owned`.

- [ ] **Step 2: Run tests**

Run: `cargo test -p tenferro-runtime --test runtime_public_api graph_executor_runs_ --no-default-features --features cpu-faer`

Expected: still fail until GraphExecutor and dispatch are wired.

### Task 3: GraphExecutor Borrowed Input Resolution

**Files:**
- Modify: `tenferro-runtime/src/graph/executor.rs`

- [ ] **Step 1: Add public API**

Add:

```rust
pub fn run_many_with_input_reads<'a>(
    &mut self,
    program: &GraphProgram,
    bindings: &[(&TracedTensor, TensorRead<'a>)],
) -> Result<Vec<Tensor>>
```

This API should validate placeholders like `run_many_with_inputs`, resolve default tensors as borrowed reads, and call a new borrowed exec path. Keep `run_many_with_inputs` unchanged.

- [ ] **Step 2: Run tests**

Run: `cargo test -p tenferro-runtime --test runtime_public_api graph_executor_runs_ --no-default-features --features cpu-faer`

Expected: dispatch may still fail until read-aware execution is complete.

### Task 4: Read-Aware Dispatch and Segmentation

**Files:**
- Modify: `tenferro-runtime/src/exec/dispatch.rs`
- Modify: `tenferro-runtime/src/segment.rs`
- Modify: `tenferro-tensor/src/backend.rs`

- [ ] **Step 1: Convert dispatch signatures to `ExecSlot`**

Change backend/FFI/host dispatch functions to accept `ExecSlot` slots. Keep public `eval_exec_ir` signatures unchanged.

- [ ] **Step 2: Route read-capable ops through `_read` methods**

Use `get_read` for elementwise, analytic, reductions, structural read methods, and `dot_general_read`.

- [ ] **Step 3: Add `dot_general_with_conj_read`**

Add a default read-capable method to `TensorDot` that avoids materializing normal borrowed tensor refs and materializes only views when conjugation needs owned tensors.

- [ ] **Step 4: Preserve unsupported-op behavior**

For indexing/fusion paths that still need `&Tensor`, use `get_tensor` so `TensorRead::Tensor(&Tensor)` remains clone-free. Borrowed views may return an explicit backend-boundary error for unsupported ops.

- [ ] **Step 5: Run tests**

Run: `cargo test -p tenferro-runtime --test runtime_public_api graph_executor_runs_ --no-default-features --features cpu-faer`

Expected: pass.

### Task 5: Reclaim and Compatibility Tests

**Files:**
- Modify: `tenferro-ad/tests/exec_dispatch.rs`
- Modify: `tenferro-runtime/tests/runtime_public_api.rs`

- [ ] **Step 1: Add or update tests**

Cover that borrowed input slots are not reclaimed, owned intermediates still are, and existing `run_many_with_inputs` remains usable.

- [ ] **Step 2: Run targeted tests**

Run:

```bash
cargo test -p tenferro-runtime --test runtime_public_api --no-default-features --features cpu-faer
cargo test -p tenferro-ad --test exec_dispatch --no-default-features --features cpu-faer
```

Expected: pass.

### Task 6: Benchmark Runner Uses Borrowed Inputs

**Files:**
- Modify: `tenferro-benchmark/src/main.rs`

- [ ] **Step 1: Update traced runner**

Change `run_instance_trace` to bind operands as `TensorRead::from_tensor` and call `run_many_with_input_reads`. Keep input tensors created outside the timed loop.

- [ ] **Step 2: Remove timed output clone where present**

Ensure traced timed loop only `black_box`s the returned output vector/tensor and does not clone data.

- [ ] **Step 3: Run benchmark build**

Run: `cargo build --release --no-default-features --features system-accelerate --bin tenferro-einsum-benchmark`

Expected: pass.

### Task 7: Final Verification

**Files:**
- No additional files.

- [ ] **Step 1: Run runtime tests**

Run: `cargo test -p tenferro-runtime --no-default-features --features cpu-faer`

Expected: pass.

- [ ] **Step 2: Run einsum tests**

Run: `cargo test -p tenferro-einsum --no-default-features --features cpu-faer,autodiff`

Expected: pass.

- [ ] **Step 3: Check git status**

Run: `git status --short`

Expected: only intentional source/test/doc changes.
