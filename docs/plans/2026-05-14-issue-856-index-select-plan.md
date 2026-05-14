# Issue 856 Index-Select Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add public trailing-batch `stack` and `index_select` APIs that preserve no-grad, forward-mode, and reverse-mode AD without host-side scalar materialization.

**Architecture:** Build the public APIs as composition over existing standard operations. `index_select` lowers to `StdTensorOp::Gather`, so existing gather linearization and gather-to-scatter transpose own AD semantics. `stack` inserts a singleton axis with `Reshape` and then uses `Concatenate`; CPU forward outputs use the backend/session buffer pool where they are fully overwritten.

**Tech Stack:** Rust, `tenferro-tensor`, `tenferro`, `tenferro-ops`, existing `GatherConfig` / `ScatterConfig`, `StdTensorOp::Gather`, `StdTensorOp::Concatenate`, `CpuBackend` / `CpuExecSession`, `BufferPool`, cargo tests.

---

## Scope

Implement #856 against current `origin/main`.

In scope:

- public concrete `Tensor::index_select` and `Tensor::stack`
- public `EagerTensor::index_select` and `EagerTensor::stack`
- public `TracedTensor::index_select` and `TracedTensor::stack`
- negative-axis handling for the new APIs
- repeated-position gather with scatter-add reverse semantics
- retained-batch contraction regression with a trailing batch axis
- CPU buffer-pool allocation for forward gather/concatenate outputs used by the new APIs
- rustdoc examples for every new public API

Out of scope:

- new AD rules
- new `StdTensorOp::IndexSelect`
- TreeTN-specific `ExtensionOp`
- NumPy-style advanced indexing or index tensor broadcasting
- GPU-native work beyond the existing CubeCL gather/concatenate capabilities
- broad restructuring of every historical indexing helper

Before touching AD-related code or tests, re-read `REPOSITORY_RULES.md`. This
plan reuses existing AD rules and should not add a new `linearize` or
`transpose_rule` arm.

## Current Pitfall To Fix

The generic binary execution helpers currently promote all binary inputs. That
is wrong for indexing ops: integer indices are parameters, not numeric data to
promote with the operand. `Gather`, `DynamicSlice`, `Scatter`, and
`DynamicUpdateSlice` need operand/update promotion without converting index
operands to complex or floating dtypes.

This cleanup is required for a clean `index_select` implementation. Do it as a
root-cause fix, not as a special case inside `index_select`.

---

### Task 1: Concrete Tensor API Tests

**Files:**
- Modify: `tenferro-tensor/src/tests/cpu_tests.rs`

**Step 1: Write failing tests for `Tensor::index_select`**

Append tests near the existing gather/concatenate CPU tests:

```rust
#[test]
fn tensor_index_select_trailing_axis_returns_expected_values() {
    let mut backend = CpuBackend::new();
    let input = Tensor::from_vec(
        vec![2, 3],
        vec![
            1.0_f64, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ],
    );

    let out = input.index_select(-1, &[2, 0, 2], &mut backend).unwrap();

    assert_eq!(out.shape(), &[2, 3]);
    assert_f64_close(get_f64(&out, &[0, 0]), 5.0);
    assert_f64_close(get_f64(&out, &[1, 0]), 6.0);
    assert_f64_close(get_f64(&out, &[0, 1]), 1.0);
    assert_f64_close(get_f64(&out, &[1, 1]), 2.0);
    assert_f64_close(get_f64(&out, &[0, 2]), 5.0);
    assert_f64_close(get_f64(&out, &[1, 2]), 6.0);
}

#[test]
fn tensor_index_select_rejects_invalid_axis_and_position() {
    let mut backend = CpuBackend::new();
    let input = Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]);

    let axis_err = input.index_select(-2, &[0], &mut backend).unwrap_err();
    assert!(axis_err.to_string().contains("index_select"));
    assert!(axis_err.to_string().contains("axis"));

    let position_err = input.index_select(0, &[3], &mut backend).unwrap_err();
    assert!(position_err.to_string().contains("index_select"));
    assert!(position_err.to_string().contains("position"));
}
```

**Step 2: Write failing tests for trailing-axis `Tensor::stack`**

Add:

```rust
#[test]
fn tensor_stack_trailing_axis_packs_scalars_vectors_and_matrices() {
    let mut backend = CpuBackend::new();

    let a = Tensor::from_vec(vec![], vec![1.0_f64]);
    let b = Tensor::from_vec(vec![], vec![2.0_f64]);
    let scalars = Tensor::stack(&[&a, &b], -1, &mut backend).unwrap();
    assert_eq!(scalars.shape(), &[2]);
    assert_eq!(scalars.as_slice::<f64>().unwrap(), &[1.0, 2.0]);

    let v0 = Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]);
    let v1 = Tensor::from_vec(vec![2], vec![3.0_f64, 4.0]);
    let vectors = Tensor::stack(&[&v0, &v1], -1, &mut backend).unwrap();
    assert_eq!(vectors.shape(), &[2, 2]);
    assert_f64_close(get_f64(&vectors, &[0, 0]), 1.0);
    assert_f64_close(get_f64(&vectors, &[1, 0]), 2.0);
    assert_f64_close(get_f64(&vectors, &[0, 1]), 3.0);
    assert_f64_close(get_f64(&vectors, &[1, 1]), 4.0);

    let m0 = Tensor::from_vec(vec![2, 1], vec![1.0_f64, 2.0]);
    let m1 = Tensor::from_vec(vec![2, 1], vec![3.0_f64, 4.0]);
    let matrices = Tensor::stack(&[&m0, &m1], -1, &mut backend).unwrap();
    assert_eq!(matrices.shape(), &[2, 1, 2]);
}
```

**Step 3: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-tensor tensor_index_select tensor_stack_trailing_axis -- --nocapture
```

Expected: FAIL to compile because `Tensor::index_select` and `Tensor::stack`
do not exist.

**Step 4: Commit tests**

```bash
git add tenferro-tensor/src/tests/cpu_tests.rs
git commit -m "test: add concrete index select regressions"
```

---

### Task 2: Concrete Tensor API Implementation

**Files:**
- Modify: `tenferro-tensor/src/types.rs`
- Create: `tenferro-tensor/src/types/shape_packing.rs`

**Step 1: Add a focused shape-packing module**

In `tenferro-tensor/src/types.rs`, add this next to `mod accessors;`:

```rust
mod shape_packing;
```

Create `tenferro-tensor/src/types/shape_packing.rs`.

**Step 2: Implement axis normalization helpers**

Add:

```rust
use super::{Tensor, TensorBackend, TypedTensor};
use crate::{GatherConfig, Result};

fn normalize_existing_axis(op: &'static str, axis: isize, rank: usize) -> Result<usize> {
    let normalized = if axis < 0 {
        rank as isize + axis
    } else {
        axis
    };
    if normalized < 0 || normalized >= rank as isize {
        return Err(crate::Error::AxisOutOfBounds {
            op,
            axis: axis.unsigned_abs(),
            rank,
        });
    }
    Ok(normalized as usize)
}

fn normalize_insert_axis(op: &'static str, axis: isize, rank: usize) -> Result<usize> {
    let normalized = if axis < 0 {
        rank as isize + 1 + axis
    } else {
        axis
    };
    if normalized < 0 || normalized > rank as isize {
        return Err(crate::Error::AxisOutOfBounds {
            op,
            axis: axis.unsigned_abs(),
            rank: rank + 1,
        });
    }
    Ok(normalized as usize)
}
```

If clippy objects to the `axis.unsigned_abs()` reporting detail, replace only
the reporting expression. Keep the signed-axis validation behavior.

**Step 3: Implement `index_select_parts`**

Add:

```rust
fn index_select_parts(
    shape: &[usize],
    axis: isize,
    positions: &[usize],
) -> Result<(Tensor, GatherConfig, Vec<usize>)> {
    let axis = normalize_existing_axis("index_select", axis, shape.len())?;
    let axis_extent = shape[axis];
    for &position in positions {
        if position >= axis_extent {
            return Err(crate::Error::InvalidConfig {
                op: "index_select",
                message: format!(
                    "position {position} out of bounds for axis {axis} with extent {axis_extent}"
                ),
            });
        }
    }

    let mut out_shape = shape.to_vec();
    out_shape[axis] = positions.len();

    let mut slice_sizes = shape.to_vec();
    slice_sizes[axis] = 1;

    let offset_dims: Vec<usize> = (0..shape.len()).filter(|&dim| dim != axis).collect();
    let mut index_data = Vec::with_capacity(positions.len());
    index_data.extend(positions.iter().map(|&position| position as i64));
    let indices = Tensor::I64(TypedTensor::from_vec(vec![positions.len(), 1], index_data));

    let config = GatherConfig {
        offset_dims,
        collapsed_slice_dims: vec![axis],
        start_index_map: vec![axis],
        index_vector_dim: 1,
        slice_sizes,
    };

    Ok((indices, config, out_shape))
}
```

**Step 4: Implement `Tensor::index_select`**

Add:

```rust
impl Tensor {
    /// Select entries from one axis using host-known positions.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{cpu::CpuBackend, Tensor};
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = Tensor::from_vec(vec![3], vec![10.0_f64, 20.0, 30.0]);
    /// let y = x.index_select(-1, &[2, 0], &mut backend).unwrap();
    ///
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[30.0, 10.0]);
    /// ```
    pub fn index_select(
        &self,
        axis: isize,
        positions: &[usize],
        ctx: &mut impl TensorBackend,
    ) -> Result<Self> {
        let (indices, config, _) = index_select_parts(self.shape(), axis, positions)?;
        ctx.with_exec_session(|exec| exec.gather(self, &indices, &config))
    }
}
```

**Step 5: Implement `Tensor::stack`**

Add:

```rust
impl Tensor {
    /// Stack tensors along a newly inserted axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{cpu::CpuBackend, Tensor};
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = Tensor::from_vec(vec![], vec![1.0_f64]);
    /// let b = Tensor::from_vec(vec![], vec![2.0_f64]);
    /// let out = Tensor::stack(&[&a, &b], -1, &mut backend).unwrap();
    ///
    /// assert_eq!(out.shape(), &[2]);
    /// assert_eq!(out.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// ```
    pub fn stack(
        tensors: &[&Self],
        dim: isize,
        ctx: &mut impl TensorBackend,
    ) -> Result<Self> {
        let first = tensors.first().copied().ok_or_else(|| crate::Error::InvalidConfig {
            op: "stack",
            message: "stack requires at least one input".into(),
        })?;
        let rank = first.shape().len();
        let axis = normalize_insert_axis("stack", dim, rank)?;

        for tensor in tensors.iter().copied().skip(1) {
            if tensor.shape() != first.shape() {
                return Err(crate::Error::ShapeMismatch {
                    op: "stack",
                    lhs: first.shape().to_vec(),
                    rhs: tensor.shape().to_vec(),
                });
            }
        }

        let mut expanded_shape = first.shape().to_vec();
        expanded_shape.insert(axis, 1);

        ctx.with_exec_session(|exec| {
            let mut expanded = Vec::with_capacity(tensors.len());
            for tensor in tensors {
                expanded.push(exec.reshape(tensor, &expanded_shape)?);
            }
            let refs: Vec<&Tensor> = expanded.iter().collect();
            exec.concatenate(&refs, axis)
        })
    }
}
```

**Step 6: Run tests to verify green**

Run:

```bash
cargo test -p tenferro-tensor tensor_index_select tensor_stack_trailing_axis -- --nocapture
```

Expected: PASS for the new concrete tests.

**Step 7: Commit**

```bash
git add tenferro-tensor/src/types.rs tenferro-tensor/src/types/shape_packing.rs tenferro-tensor/src/tests/cpu_tests.rs
git commit -m "feat: add concrete tensor index select"
```

---

### Task 3: Index Operand Promotion Cleanup

**Files:**
- Modify: `tenferro/src/eager_exec.rs`
- Modify: `tenferro/src/traced.rs`
- Test: `tenferro/tests/primitive_ops.rs`
- Test: `tenferro/tests/eager_tensor.rs`

**Step 1: Write failing tests for complex operand indexing**

In `tenferro/tests/primitive_ops.rs`, add:

```rust
#[test]
fn traced_index_select_keeps_indices_integer_for_complex_operand() {
    use num_complex::Complex64;

    let x = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec(
        vec![3],
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.5),
        ],
    ));
    let mut y = x.index_select(-1, &[2, 0]).unwrap();
    let mut engine = Engine::new(CpuBackend::new());
    let out = y.eval(&mut engine).unwrap();

    assert_eq!(out.shape(), &[2]);
    assert_eq!(
        out.as_slice::<Complex64>().unwrap(),
        &[Complex64::new(3.0, 0.5), Complex64::new(1.0, 1.0)]
    );
}
```

In `tenferro/tests/eager_tensor.rs`, add:

```rust
#[test]
fn eager_index_select_keeps_indices_integer_for_complex_operand() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec(
            vec![3],
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.5),
            ],
        ),
        test_ctx(),
    );

    let y = x.index_select(-1, &[2, 0]).unwrap();

    assert_eq!(y.data().shape(), &[2]);
    assert_eq!(
        c64_data(y.data()),
        &[Complex64::new(3.0, 0.5), Complex64::new(1.0, 1.0)]
    );
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro traced_index_select_keeps_indices_integer_for_complex_operand eager_index_select_keeps_indices_integer_for_complex_operand -- --nocapture
```

Expected: FAIL because `index_select` is not exposed on `TracedTensor` /
`EagerTensor`. If the methods have been added early, the test should fail
because a complex gather index was promoted to complex.

**Step 3: Fix eager execution promotion for indexing ops**

In `tenferro/src/eager_exec.rs`, replace generic `promote_binary` use for
indexing ops:

```rust
StdTensorOp::Gather(config) => {
    vec![exec.gather(inputs[0], inputs[1], config)?]
}
StdTensorOp::GatherDynamicSliceSizes { ... } => {
    ...
    vec![exec.gather(inputs[0], inputs[1], &config)?]
}
StdTensorOp::DynamicSlice { slice_sizes } => {
    vec![exec.dynamic_slice(inputs[0], inputs[1], slice_sizes)?]
}
StdTensorOp::Scatter(config) => {
    let promoted = promote_dtype_for_binary_op(op, inputs[0].dtype(), inputs[2].dtype());
    let operand = if inputs[0].dtype() != promoted {
        exec.convert(inputs[0], promoted).map_err(Error::from)?
    } else {
        inputs[0].clone()
    };
    let updates = if inputs[2].dtype() != promoted {
        exec.convert(inputs[2], promoted).map_err(Error::from)?
    } else {
        inputs[2].clone()
    };
    vec![exec.scatter(&operand, inputs[1], &updates, config)?]
}
StdTensorOp::DynamicUpdateSlice => {
    let promoted = promote_dtype_for_binary_op(op, inputs[0].dtype(), inputs[1].dtype());
    let operand = if inputs[0].dtype() != promoted {
        exec.convert(inputs[0], promoted).map_err(Error::from)?
    } else {
        inputs[0].clone()
    };
    let update = if inputs[1].dtype() != promoted {
        exec.convert(inputs[1], promoted).map_err(Error::from)?
    } else {
        inputs[1].clone()
    };
    vec![exec.dynamic_update_slice(&operand, &update, inputs[2])?]
}
```

Keep the helper small. If duplicated promotion becomes noisy, extract a local
`promote_data_pair(exec, op, lhs, rhs)` helper in the same file.

**Step 4: Add traced construction helpers that do not promote indices**

In `tenferro/src/traced.rs`, add a `pub(crate)` helper near `apply_binary`:

```rust
pub(crate) fn apply_binary_preserve_input_dtypes(
    op: StdTensorOp,
    lhs: &TracedTensor,
    rhs: &TracedTensor,
    out_rank: usize,
    out_shape_hint: Option<Vec<SymDim>>,
    out_dtype: DType,
) -> TracedTensor {
    // Same body shape as apply_binary, but do not insert Convert ops.
}
```

Implement it by copying the graph-building, input-map merge, extra-root merge,
checkpoint-chain merge, and metadata-scope merge from `apply_binary`, but omit
the conversion block and use `out_dtype` directly.

**Step 5: Run targeted tests**

Run:

```bash
cargo test -p tenferro traced_index_select_keeps_indices_integer_for_complex_operand eager_index_select_keeps_indices_integer_for_complex_operand -- --nocapture
```

Expected: still FAIL if frontend methods are not implemented yet, but indexing
promotion is now ready for them. If methods are already present, expected PASS.

**Step 6: Commit**

```bash
git add tenferro/src/eager_exec.rs tenferro/src/traced.rs tenferro/tests/primitive_ops.rs tenferro/tests/eager_tensor.rs
git commit -m "fix: keep indexing operands unpromoted"
```

---

### Task 4: Eager And Traced Public APIs

**Files:**
- Create: `tenferro/src/shape_packing.rs`
- Modify: `tenferro/src/lib.rs`
- Modify: `tenferro/src/traced.rs`
- Modify: `tenferro/src/eager_ops.rs`
- Test: `tenferro/tests/primitive_ops.rs`
- Test: `tenferro/tests/eager_tensor.rs`

**Step 1: Write failing tests for primal APIs**

In `tenferro/tests/primitive_ops.rs`, add:

```rust
#[test]
fn traced_stack_trailing_axis_and_index_select_feed_batched_dot_general() {
    let a0 = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    let a1 = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]));
    let b0 = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![2.0, 3.0]));
    let b1 = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![4.0, 5.0]));

    let a = TracedTensor::stack(&[&a0, &a1], -1).unwrap();
    let b = TracedTensor::stack(&[&b0, &b1], -1).unwrap().index_select(-1, &[1, 0]).unwrap();
    let mut c = a.dot_general(
        &b,
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![2],
            rhs_batch_dims: vec![2],
        },
    );

    let mut engine = Engine::new(CpuBackend::new());
    let out = c.eval(&mut engine).unwrap();

    assert_eq!(out.shape(), &[2, 1, 2]);
    assert_eq!(get_f64_data(out), &[19.0, 28.0, 23.0, 34.0]);
}
```

In `tenferro/tests/eager_tensor.rs`, add:

```rust
#[test]
fn eager_stack_trailing_axis_and_index_select_primal() {
    let x0 = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]), test_ctx());
    let x1 = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2], vec![3.0_f64, 4.0]), test_ctx());

    let stacked = EagerTensor::stack(&[&x0, &x1], -1).unwrap();
    let selected = stacked.index_select(-1, &[1, 0, 1]).unwrap();

    assert_eq!(selected.data().shape(), &[2, 3]);
    assert_close_slice(f64_data(selected.data()), &[3.0, 4.0, 1.0, 2.0, 3.0, 4.0], TOL);
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro traced_stack_trailing_axis_and_index_select_feed_batched_dot_general eager_stack_trailing_axis_and_index_select_primal -- --nocapture
```

Expected: FAIL because public frontend APIs are missing.

**Step 3: Create `tenferro/src/shape_packing.rs`**

Add `mod shape_packing;` to `tenferro/src/lib.rs`.

In `shape_packing.rs`, implement shared helpers. Keep the code in this module
where possible, but import private helpers from their actual owning modules
after inspecting current visibility rather than following the illustrative list
below verbatim:

```rust
use std::collections::HashMap;
use std::sync::Arc;

use computegraph::fragment::FragmentBuilder;
use computegraph::types::{OpMode, ValRef};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{DType, GatherConfig, Tensor, TensorBackend, TypedTensor};

use crate::checkpoint::CheckpointNode;
use crate::eager::{record_eager_outputs, EagerContext, EagerTensor};
use crate::error::{Error, Result};
use crate::metadata::{metadata_scopes_with_new, push_metadata_scope, register_scoped_fragment_metadata};
use crate::shape_infer::promote_dtypes;
use crate::sym_dim::SymDim;
use crate::traced::{apply_binary_preserve_input_dtypes, TracedTensor};
```

If Rust privacy requires narrower imports or small `pub(crate)` helper
adjustments, make those changes without changing the module boundary. Keep the
implementation in this new module rather than growing `traced.rs`.

**Step 4: Add frontend `index_select` config construction**

Implement a helper equivalent to the concrete `index_select_parts`, but using
`SymDim` for traced shape hints:

```rust
fn normalize_existing_axis(op: &'static str, axis: isize, rank: usize) -> Result<usize> { ... }

fn index_select_config(
    shape: &[usize],
    axis: isize,
    positions: &[usize],
) -> Result<(Tensor, GatherConfig, Vec<usize>)> { ... }
```

For `TracedTensor::index_select`, require a concrete `shape_hint` for
position validation. If `shape_hint` is absent, return `Error::Internal` with a
message saying `index_select currently requires a concrete shape hint`. Do not
silently defer out-of-range checking to backend gather clamping.

**Step 5: Implement `EagerTensor::index_select`**

Add:

```rust
impl<B: TensorBackend> EagerTensor<B> {
    /// Select entries from one axis using host-known positions.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_backend(CpuBackend::new());
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec(vec![3], vec![10.0_f64, 20.0, 30.0]), ctx);
    /// let y = x.index_select(-1, &[2, 0]).unwrap();
    ///
    /// assert_eq!(y.data().as_slice::<f64>().unwrap(), &[30.0, 10.0]);
    /// ```
    pub fn index_select(&self, axis: isize, positions: &[usize]) -> Result<Self> {
        let (indices, config, _) = index_select_config(self.data.shape(), axis, positions)?;
        let indices = self.ctx.constant_from(indices);
        self.gather(&indices, config)
    }
}
```

**Step 6: Implement `TracedTensor::index_select`**

Add:

```rust
impl TracedTensor {
    /// Select entries from one axis using host-known positions.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, Engine, Tensor, TracedTensor};
    ///
    /// let x = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec(vec![3], vec![10.0_f64, 20.0, 30.0]));
    /// let mut y = x.index_select(-1, &[2, 0]).unwrap();
    /// let mut engine = Engine::new(CpuBackend::new());
    ///
    /// assert_eq!(y.eval(&mut engine).unwrap().as_slice::<f64>().unwrap(), &[30.0, 10.0]);
    /// ```
    pub fn index_select(&self, axis: isize, positions: &[usize]) -> Result<Self> {
        let shape = crate::traced::try_concrete_shape(self)
            .ok_or_else(|| Error::Internal("index_select currently requires a concrete shape hint".into()))?;
        let (indices_tensor, config, out_shape) = index_select_config(&shape, axis, positions)?;
        let indices = TracedTensor::from_tensor_concrete_shape(indices_tensor);
        Ok(apply_binary_preserve_input_dtypes(
            StdTensorOp::Gather(config),
            self,
            &indices,
            out_shape.len(),
            Some(out_shape.into_iter().map(SymDim::from).collect()),
            self.dtype,
        ))
    }
}
```

**Step 7: Implement `EagerTensor::stack`**

Add:

```rust
impl<B: TensorBackend> EagerTensor<B> {
    /// Stack tensors along a newly inserted axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_backend(CpuBackend::new());
    /// let a = EagerTensor::from_tensor_in(Tensor::from_vec(vec![], vec![1.0_f64]), ctx.clone());
    /// let b = EagerTensor::from_tensor_in(Tensor::from_vec(vec![], vec![2.0_f64]), ctx);
    /// let out = EagerTensor::stack(&[&a, &b], -1).unwrap();
    ///
    /// assert_eq!(out.data().as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// ```
    pub fn stack(tensors: &[&Self], dim: isize) -> Result<Self> {
        let first = tensors.first().copied().ok_or_else(|| Error::TensorRuntime(
            tenferro_tensor::Error::InvalidConfig {
                op: "stack",
                message: "stack requires at least one input".into(),
            },
        ))?;
        let axis = normalize_insert_axis("stack", dim, first.data.shape().len())?;
        let mut expanded = Vec::with_capacity(tensors.len());
        let mut expanded_shape = first.data.shape().to_vec();
        expanded_shape.insert(axis, 1);
        for tensor in tensors {
            expanded.push(tensor.reshape(&expanded_shape)?);
        }
        let refs: Vec<&Self> = expanded.iter().collect();
        Self::concatenate(&refs, axis)
    }
}
```

**Step 8: Implement `TracedTensor::stack`**

Add an internal `apply_nary` helper in `tenferro/src/traced.rs` or the new
module. It must:

- optionally convert inputs to a common dtype for `Concatenate`
- add all parent fragments
- add the n-ary op with all external value refs
- merge all input maps
- merge all extra roots and checkpoint chains
- merge metadata scopes

Then implement:

```rust
impl TracedTensor {
    /// Stack tensors along a newly inserted axis.
    pub fn stack(tensors: &[&Self], dim: isize) -> Result<Self> {
        let first = tensors.first().copied().ok_or_else(|| Error::TensorRuntime(
            tenferro_tensor::Error::InvalidConfig {
                op: "stack",
                message: "stack requires at least one input".into(),
            },
        ))?;
        let axis = normalize_insert_axis("stack", dim, first.rank)?;
        let first_shape = crate::traced::try_concrete_shape(first)
            .ok_or_else(|| Error::Internal("stack currently requires concrete shape hints".into()))?;

        for tensor in tensors.iter().copied().skip(1) {
            let shape = crate::traced::try_concrete_shape(tensor)
                .ok_or_else(|| Error::Internal("stack currently requires concrete shape hints".into()))?;
            if shape != first_shape {
                return Err(Error::TensorRuntime(tenferro_tensor::Error::ShapeMismatch {
                    op: "stack",
                    lhs: first_shape.clone(),
                    rhs: shape,
                }));
            }
        }

        let mut expanded_shape = first_shape;
        expanded_shape.insert(axis, 1);
        let expanded: Vec<_> = tensors.iter().map(|tensor| tensor.reshape(&expanded_shape)).collect();
        let refs: Vec<&TracedTensor> = expanded.iter().collect();
        Ok(apply_nary_concatenate(&refs, axis))
    }
}
```

**Step 9: Run frontend primal tests**

Run:

```bash
cargo test -p tenferro traced_stack_trailing_axis_and_index_select_feed_batched_dot_general eager_stack_trailing_axis_and_index_select_primal traced_index_select_keeps_indices_integer_for_complex_operand eager_index_select_keeps_indices_integer_for_complex_operand -- --nocapture
```

Expected: PASS.

**Step 10: Commit**

```bash
git add tenferro/src/lib.rs tenferro/src/shape_packing.rs tenferro/src/traced.rs tenferro/src/eager_ops.rs tenferro/tests/primitive_ops.rs tenferro/tests/eager_tensor.rs
git commit -m "feat: expose eager and traced index select"
```

---

### Task 5: AD Regression Tests

**Files:**
- Modify: `tenferro/tests/ad.rs`
- Modify: `tenferro/tests/eager_tensor.rs`

**Step 1: Write failing reverse-mode test for repeated positions**

In `tenferro/tests/ad.rs`, add near the existing gather AD tests:

```rust
#[test]
fn grad_traced_index_select_repeated_positions_accumulates() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let weights = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![10.0, 20.0, 30.0]));

    let selected = x.index_select(0, &[1, 1, 2]).unwrap();
    let loss = (&selected * &weights).reduce_sum(&[0]);
    let grad = eval_tensor(loss.grad(&x).unwrap());

    assert_eq!(grad.shape(), &[3]);
    assert_close_slice(get_f64_data(&grad), &[0.0, 30.0, 30.0]);
}
```

**Step 2: Write failing forward-mode test**

In `tenferro/tests/ad.rs`, add:

```rust
#[test]
fn jvp_traced_index_select_gathers_tangent() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]));
    let tangent = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4], vec![0.5, 1.5, 2.5, 3.5]));

    let y = x.index_select(0, &[3, 1, 3]).unwrap();
    let tangent_y = eval_tensor(y.jvp(&x, &tangent));

    assert_eq!(tangent_y.shape(), &[3]);
    assert_close_slice(get_f64_data(&tangent_y), &[3.5, 1.5, 3.5]);
}
```

**Step 3: Write eager reverse-mode test**

In `tenferro/tests/eager_tensor.rs`, add:

```rust
#[test]
fn eager_index_select_repeated_positions_accumulates_grad() {
    let ctx = test_ctx();
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]),
        ctx.clone(),
    );
    let weights = EagerTensor::from_tensor_in(
        Tensor::from_vec(vec![3], vec![10.0_f64, 20.0, 30.0]),
        ctx,
    );

    let selected = x.index_select(0, &[1, 1, 2]).unwrap();
    let loss = (&selected * &weights).reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();

    assert_close_slice(f64_data(x.grad().unwrap().as_ref()), &[0.0, 30.0, 30.0], TOL);
}
```

**Step 4: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro index_select_repeated_positions jvp_traced_index_select -- --nocapture
```

Expected: FAIL if any graph metadata, AD wiring, or shape hint handling is
missing.

**Step 5: Implement only missing glue**

If failures are in metadata registration for the new helper-created index
tensor, fix the `metadata_scopes` merge in the new traced helper. If failures
are in eager reverse execution, confirm that `record_eager_outputs` sees the
index tensor as inactive through existing `StdTensorOp::Gather` rules.

Do not add a new AD rule. The expected final AD graph must still use existing
`Gather` and inverse `Scatter`.

**Step 6: Run tests to verify green**

Run:

```bash
cargo test -p tenferro index_select_repeated_positions jvp_traced_index_select -- --nocapture
```

Expected: PASS.

**Step 7: Commit**

```bash
git add tenferro/tests/ad.rs tenferro/tests/eager_tensor.rs tenferro/src
git commit -m "test: cover index select ad semantics"
```

---

### Task 6: CPU Buffer-Pool Allocation For Gather And Concatenate

**Files:**
- Modify: `tenferro-tensor/src/cpu/mod.rs`
- Create: `tenferro-tensor/src/cpu/indexing_alloc.rs`
- Modify: `tenferro-tensor/src/cpu/indexing.rs`
- Modify: `tenferro-tensor/src/cpu/backend.rs`
- Modify: `tenferro-tensor/src/cpu/exec_session.rs`
- Test: `tenferro-tensor/src/tests/cpu_tests.rs`

**Step 1: Write failing buffer-pool reuse tests**

In `tenferro-tensor/src/tests/cpu_tests.rs`, add:

```rust
#[test]
fn tensor_index_select_reuses_reclaimed_cpu_buffer() {
    let mut backend = CpuBackend::new();
    let reusable = Tensor::from_vec(vec![2, 3], vec![0.0_f64; 6]);
    let expected_ptr = reusable.as_slice::<f64>().unwrap().as_ptr();
    backend.reclaim_buffer(reusable);

    let input = Tensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = input.index_select(-1, &[2, 0, 1], &mut backend).unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap().as_ptr(), expected_ptr);
}

#[test]
fn tensor_stack_reuses_reclaimed_cpu_buffer() {
    let mut backend = CpuBackend::new();
    let reusable = Tensor::from_vec(vec![2, 2], vec![0.0_f64; 4]);
    let expected_ptr = reusable.as_slice::<f64>().unwrap().as_ptr();
    backend.reclaim_buffer(reusable);

    let x0 = Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]);
    let x1 = Tensor::from_vec(vec![2], vec![3.0_f64, 4.0]);
    let out = Tensor::stack(&[&x0, &x1], -1, &mut backend).unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap().as_ptr(), expected_ptr);
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-tensor reuses_reclaimed_cpu_buffer -- --nocapture
```

Expected: FAIL because gather/concatenate allocate outside the backend pool.

**Step 3: Add pooled allocation helper**

Create `tenferro-tensor/src/cpu/indexing_alloc.rs`:

```rust
use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::TypedTensor;

pub(crate) fn pooled_uninit_tensor<T>(
    buffers: &mut BufferPool,
    shape: Vec<usize>,
) -> TypedTensor<T>
where
    T: Clone + PoolScalar,
{
    let len = shape.iter().product();
    let data = unsafe { T::pool_acquire(buffers, len) };
    TypedTensor::from_vec(shape, data)
}
```

In `tenferro-tensor/src/cpu/mod.rs`, add:

```rust
mod indexing_alloc;
```

**Step 4: Thread `BufferPool` into gather and concatenate**

In `tenferro-tensor/src/cpu/indexing.rs`:

- keep public `gather` and `concatenate` wrappers for compatibility
- add `gather_with_pool(buffers, operand, start_indices, config)`
- add `concatenate_with_pool(buffers, inputs, axis)`
- make `typed_gather` and `typed_concatenate` take `&mut BufferPool`
- replace `typed_tensor_uninit(out_shape.clone())` with
  `pooled_uninit_tensor(buffers, out_shape.clone())`

Use trait bounds:

```rust
fn typed_gather<T: Copy + Clone + Zero + PoolScalar>(...)
fn typed_concatenate<T: Copy + Clone + PoolScalar>(...)
```

For the old public wrappers, create a local `BufferPool::new()` and call the
`*_with_pool` functions. That keeps the public `cpu::gather` behavior while
making backend execution use the real pool.

**Step 5: Wire CPU backend/session to pooled helpers**

In `tenferro-tensor/src/cpu/backend.rs`, change:

```rust
self.install(|| indexing::gather(operand, start_indices, config))
```

to:

```rust
self.install_with_pool(|buffers| indexing::gather_with_pool(buffers, operand, start_indices, config))
```

Do the same for `concatenate`.

In `tenferro-tensor/src/cpu/exec_session.rs`, change gather and concatenate
delegation to call the `*_with_pool` functions with `self.buffers`.

**Step 6: Run tests to verify green**

Run:

```bash
cargo test -p tenferro-tensor reuses_reclaimed_cpu_buffer tensor_index_select tensor_stack_trailing_axis -- --nocapture
```

Expected: PASS.

**Step 7: Commit**

```bash
git add tenferro-tensor/src/cpu/mod.rs tenferro-tensor/src/cpu/indexing_alloc.rs tenferro-tensor/src/cpu/indexing.rs tenferro-tensor/src/cpu/backend.rs tenferro-tensor/src/cpu/exec_session.rs tenferro-tensor/src/tests/cpu_tests.rs
git commit -m "perf: allocate indexing outputs from cpu buffer pool"
```

---

### Task 7: Documentation And Public Surface Checks

**Files:**
- Modify: `tenferro/src/lib.rs`
- Modify: `tenferro-tensor/src/types/shape_packing.rs`
- Modify: `tenferro/src/shape_packing.rs`
- Modify: `docs/design/supported-ops.md`
- Modify if needed: `docs/guides/choosing-an-api.md`

**Step 1: Add or verify rustdoc examples**

Every new public method must have a compiling `# Examples` section:

- `tenferro_tensor::Tensor::index_select`
- `tenferro_tensor::Tensor::stack`
- `tenferro::EagerTensor::index_select`
- `tenferro::EagerTensor::stack`
- `tenferro::TracedTensor::index_select`
- `tenferro::TracedTensor::stack`

Examples must run as doctests. Do not use `ignore` or `no_run`.

**Step 2: Update user-facing docs only where appropriate**

If docs mention supported indexing or packing operations, add concise user-facing
language:

```text
Use `stack(..., -1)` to create a trailing batch axis and
`index_select(-1, positions)` to align entries along that axis.
```

Do not expose internal terms such as `StdTensorOp`, `Fragment`, or `GatherConfig`
in user-facing docs.

**Step 3: Run targeted tests and doctests**

Run:

```bash
cargo fmt --all --check
cargo test -p tenferro-tensor tensor_index_select tensor_stack_trailing_axis reuses_reclaimed_cpu_buffer -- --nocapture
cargo test -p tenferro index_select stack_trailing_axis -- --nocapture
cargo test --doc -p tenferro-tensor
cargo test --doc -p tenferro
```

Expected: PASS.

If formatting fails, run:

```bash
cargo fmt --all
```

Then rerun `cargo fmt --all --check`.

**Step 4: Commit**

```bash
git add tenferro/src/lib.rs tenferro-tensor/src/types/shape_packing.rs tenferro/src/shape_packing.rs docs/design/supported-ops.md docs/guides/choosing-an-api.md
git commit -m "docs: document trailing batch index select"
```

---

### Task 8: Final Verification

**Files:**
- No code changes expected.

**Step 1: Re-read repository rules**

Run:

```bash
sed -n '1,260p' REPOSITORY_RULES.md
```

Check the final diff against the AD rule, documentation, and CPU threading
contracts.

**Step 2: Run pre-push checks as far as practical**

Run:

```bash
cargo fmt --all --check
cargo test --workspace --release
cargo test --doc --workspace --release
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

If time permits and `cargo llvm-cov` is available:

```bash
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

Expected: PASS. If any command fails, fix the root cause before marking the
work complete.

**Step 3: Review final diff**

Run:

```bash
git diff origin/main...HEAD --stat
git diff origin/main...HEAD
```

Confirm:

- no `ExtensionOp` was added
- no new AD rule was added
- `index_select` lowers through existing `Gather`
- reverse duplicate accumulation is covered by tests
- new public APIs have doctest examples
- no host scalar materialization path was introduced

**Step 4: Commit any final fixes**

If the verification step required changes:

```bash
git add <changed-files>
git commit -m "chore: finish issue 856 verification fixes"
```
