# NumPy-Style Free-Function Tensor API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement issue #892 by adding shared NumPy-style normalization helpers, canonical module free functions for the initial core tensor API, and matching extension-crate namespaces without adding operation-family modules to `tenferro`.

**Architecture:** Shared pure shape/axis helpers live in `tenferro-internal-ops`. Core public modules lower through those helpers into existing traced/eager/concrete primitive operations. Standard operation crates add or align their own tensor-family namespaces while remaining outside the `tenferro` facade.

**Tech Stack:** Rust workspace, `tenferro-internal-ops`, `tenferro-internal-tensor`, `tenferro`, `tenferro-einsum`, `tenferro-linalg`, `tenferro-fft`, Cargo tests/doctests.

---

## Preconditions

- Worktree: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/numpy-api-unification`
- Branch: `codex/numpy-api-unification`
- Baseline already run: `cargo build` and `cargo test --workspace`
- Design checkpoint commit: `d13bbdbf docs: design numpy-style tensor API`

Follow TDD for each task: add the narrow failing test, run it and confirm the expected failure, then implement the minimum production code.

## Task 1: Shared Normalization Helper Tests

**Files:**
- Create: `tenferro-internal-ops/src/tests/normalization_tests.rs`
- Modify: `tenferro-internal-ops/src/tests/mod.rs`

**Step 1: Write the failing tests**

Add `mod normalization_tests;` to `tenferro-internal-ops/src/tests/mod.rs`.

Create `tenferro-internal-ops/src/tests/normalization_tests.rs`:

```rust
use tenferro_ops::axis::{normalize_axis, normalize_axes};
use tenferro_ops::broadcast::{broadcast_input_plan, broadcast_shape, broadcast_shapes};
use tenferro_ops::reduction::reduced_shape;

#[test]
fn broadcast_shape_accepts_scalar_rank_padding_and_singletons() {
    assert_eq!(broadcast_shape(&[], &[3, 4]).unwrap(), vec![3, 4]);
    assert_eq!(broadcast_shape(&[5], &[3, 5]).unwrap(), vec![3, 5]);
    assert_eq!(broadcast_shape(&[3, 1], &[1, 4]).unwrap(), vec![3, 4]);
    assert_eq!(
        broadcast_shapes([&[3, 1][..], &[1, 4][..], &[3, 4][..]]).unwrap(),
        vec![3, 4]
    );
}

#[test]
fn broadcast_shape_rejects_incompatible_shapes() {
    let err = broadcast_shape(&[2, 3], &[3, 2]).unwrap_err();
    assert!(err.to_string().contains("broadcast"));
}

#[test]
fn broadcast_input_plan_drops_expanding_singletons() {
    let plan = broadcast_input_plan(&[3, 1], &[3, 4]).unwrap();
    assert_eq!(plan.source_shape, vec![3]);
    assert_eq!(plan.dims, vec![0]);

    let scalar = broadcast_input_plan(&[], &[3, 4]).unwrap();
    assert_eq!(scalar.source_shape, Vec::<usize>::new());
    assert_eq!(scalar.dims, Vec::<usize>::new());

    let vector = broadcast_input_plan(&[5], &[3, 5]).unwrap();
    assert_eq!(vector.source_shape, vec![5]);
    assert_eq!(vector.dims, vec![1]);
}

#[test]
fn normalize_axis_accepts_negative_axes_and_rejects_out_of_bounds() {
    assert_eq!(normalize_axis(0, 3).unwrap(), 0);
    assert_eq!(normalize_axis(-1, 3).unwrap(), 2);
    assert_eq!(normalize_axis(-3, 3).unwrap(), 0);
    assert!(normalize_axis(3, 3).is_err());
    assert!(normalize_axis(-4, 3).is_err());
}

#[test]
fn normalize_axes_rejects_duplicates_after_normalization() {
    assert_eq!(normalize_axes(&[0, -1], 3).unwrap(), vec![0, 2]);
    let err = normalize_axes(&[1, -2], 3).unwrap_err();
    assert!(err.to_string().contains("duplicate"));
}

#[test]
fn reduced_shape_supports_keepdims() {
    assert_eq!(reduced_shape(&[2, 3, 4], &[1], false).unwrap(), vec![2, 4]);
    assert_eq!(reduced_shape(&[2, 3, 4], &[1], true).unwrap(), vec![2, 1, 4]);
    assert_eq!(reduced_shape(&[2, 3], &[0, 1], false).unwrap(), Vec::<usize>::new());
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p tenferro-internal-ops normalization_tests
```

Expected: FAIL to compile because `tenferro_ops::axis`, `tenferro_ops::broadcast`, and `tenferro_ops::reduction` do not exist.

**Step 3: Commit only the failing tests**

Do not commit failing tests alone. Keep this task open and proceed directly to Task 2.

## Task 2: Implement Shared Normalization Helpers

**Files:**
- Create: `tenferro-internal-ops/src/axis.rs`
- Create: `tenferro-internal-ops/src/broadcast.rs`
- Create: `tenferro-internal-ops/src/reduction.rs`
- Modify: `tenferro-internal-ops/src/lib.rs`
- Test: `tenferro-internal-ops/src/tests/normalization_tests.rs`

**Step 1: Add modules and error type**

In `tenferro-internal-ops/src/lib.rs`, add:

```rust
pub mod axis;
pub mod broadcast;
pub mod reduction;
```

Create `tenferro-internal-ops/src/axis.rs`:

```rust
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AxisError {
    #[error("axis {axis} is out of bounds for rank {rank}")]
    OutOfBounds { axis: isize, rank: usize },
    #[error("duplicate axis {axis}")]
    Duplicate { axis: usize },
}

pub fn normalize_axis(axis: isize, rank: usize) -> Result<usize, AxisError> {
    let rank_i = rank as isize;
    let normalized = if axis < 0 { rank_i + axis } else { axis };
    if normalized < 0 || normalized >= rank_i {
        return Err(AxisError::OutOfBounds { axis, rank });
    }
    Ok(normalized as usize)
}

pub fn normalize_axes(axes: &[isize], rank: usize) -> Result<Vec<usize>, AxisError> {
    let mut out = Vec::with_capacity(axes.len());
    let mut seen = vec![false; rank];
    for &axis in axes {
        let normalized = normalize_axis(axis, rank)?;
        if seen[normalized] {
            return Err(AxisError::Duplicate { axis: normalized });
        }
        seen[normalized] = true;
        out.push(normalized);
    }
    Ok(out)
}
```

Create `tenferro-internal-ops/src/broadcast.rs`:

```rust
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BroadcastInputPlan {
    pub source_shape: Vec<usize>,
    pub dims: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BroadcastError {
    #[error("cannot broadcast shapes {lhs:?} and {rhs:?}")]
    IncompatibleBinary { lhs: Vec<usize>, rhs: Vec<usize> },
    #[error("cannot broadcast shape {input:?} to {output:?}")]
    IncompatibleInput { input: Vec<usize>, output: Vec<usize> },
    #[error("cannot broadcast higher-rank shape {input:?} to {output:?}")]
    RankTooLarge { input: Vec<usize>, output: Vec<usize> },
}

pub fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, BroadcastError> {
    let rank = lhs.len().max(rhs.len());
    let mut out = Vec::with_capacity(rank);
    for axis in 0..rank {
        let lhs_dim = aligned_dim(lhs, rank, axis);
        let rhs_dim = aligned_dim(rhs, rank, axis);
        if lhs_dim == rhs_dim {
            out.push(lhs_dim);
        } else if lhs_dim == 1 {
            out.push(rhs_dim);
        } else if rhs_dim == 1 {
            out.push(lhs_dim);
        } else {
            return Err(BroadcastError::IncompatibleBinary {
                lhs: lhs.to_vec(),
                rhs: rhs.to_vec(),
            });
        }
    }
    Ok(out)
}

pub fn broadcast_shapes<'a>(
    shapes: impl IntoIterator<Item = &'a [usize]>,
) -> Result<Vec<usize>, BroadcastError> {
    let mut iter = shapes.into_iter();
    let Some(first) = iter.next() else {
        return Ok(Vec::new());
    };
    let mut out = first.to_vec();
    for shape in iter {
        out = broadcast_shape(&out, shape)?;
    }
    Ok(out)
}

pub fn broadcast_input_plan(
    input: &[usize],
    output: &[usize],
) -> Result<BroadcastInputPlan, BroadcastError> {
    if input.len() > output.len() {
        return Err(BroadcastError::RankTooLarge {
            input: input.to_vec(),
            output: output.to_vec(),
        });
    }
    let rank_diff = output.len() - input.len();
    let mut source_shape = Vec::with_capacity(input.len());
    let mut dims = Vec::with_capacity(input.len());
    for (src_axis, &src_dim) in input.iter().enumerate() {
        let dst_axis = src_axis + rank_diff;
        let dst_dim = output[dst_axis];
        if src_dim != dst_dim && src_dim != 1 {
            return Err(BroadcastError::IncompatibleInput {
                input: input.to_vec(),
                output: output.to_vec(),
            });
        }
        if src_dim == 1 && dst_dim != 1 {
            continue;
        }
        source_shape.push(src_dim);
        dims.push(dst_axis);
    }
    Ok(BroadcastInputPlan { source_shape, dims })
}

fn aligned_dim(shape: &[usize], output_rank: usize, output_axis: usize) -> usize {
    if output_axis < output_rank - shape.len() {
        1
    } else {
        shape[output_axis - (output_rank - shape.len())]
    }
}
```

Create `tenferro-internal-ops/src/reduction.rs`:

```rust
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ReductionShapeError {
    #[error("axis {axis} is out of bounds for rank {rank}")]
    AxisOutOfBounds { axis: usize, rank: usize },
    #[error("duplicate axis {axis}")]
    DuplicateAxis { axis: usize },
}

pub fn reduced_shape(
    input_shape: &[usize],
    axes: &[usize],
    keepdims: bool,
) -> Result<Vec<usize>, ReductionShapeError> {
    let mut reduced = vec![false; input_shape.len()];
    for &axis in axes {
        if axis >= input_shape.len() {
            return Err(ReductionShapeError::AxisOutOfBounds {
                axis,
                rank: input_shape.len(),
            });
        }
        if reduced[axis] {
            return Err(ReductionShapeError::DuplicateAxis { axis });
        }
        reduced[axis] = true;
    }
    let mut out = Vec::with_capacity(input_shape.len());
    for (axis, &dim) in input_shape.iter().enumerate() {
        if reduced[axis] {
            if keepdims {
                out.push(1);
            }
        } else {
            out.push(dim);
        }
    }
    Ok(out)
}
```

**Step 2: Run helper tests**

Run:

```bash
cargo test -p tenferro-internal-ops normalization_tests
```

Expected: PASS.

**Step 3: Run crate tests**

Run:

```bash
cargo test -p tenferro-internal-ops
```

Expected: PASS.

**Step 4: Commit**

```bash
git add tenferro-internal-ops/src/lib.rs tenferro-internal-ops/src/axis.rs tenferro-internal-ops/src/broadcast.rs tenferro-internal-ops/src/reduction.rs tenferro-internal-ops/src/tests/mod.rs tenferro-internal-ops/src/tests/normalization_tests.rs
git commit -m "feat: add tensor API normalization helpers"
```

## Task 3: Reuse Shared Broadcasting In Traced Lowering

**Files:**
- Modify: `tenferro/src/traced.rs`
- Test: `tenferro/tests/numpy_api.rs`

**Step 1: Write a focused behavior test**

Create `tenferro/tests/numpy_api.rs` if it does not exist:

```rust
use tenferro::{traced_tensor, CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

#[test]
fn traced_add_uses_numpy_broadcasting_for_rank_padding_and_singletons() {
    let lhs = TracedTensor::from_vec_row_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0]);
    let rhs = TracedTensor::from_vec_row_major(vec![1, 4], vec![10.0_f64, 20.0, 30.0, 40.0]);
    let y = traced_tensor::add(&lhs, &rhs);

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let out = GraphExecutor::new(CpuBackend::new()).run(&program).unwrap();

    assert_eq!(out.shape(), &[3, 4]);
    assert_eq!(
        out.into_vec_row_major::<f64>().unwrap().1,
        vec![
            11.0, 21.0, 31.0, 41.0,
            12.0, 22.0, 32.0, 42.0,
            13.0, 23.0, 33.0, 43.0,
        ]
    );
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p tenferro --test numpy_api traced_add_uses_numpy_broadcasting_for_rank_padding_and_singletons
```

Expected: FAIL to compile because `tenferro::traced_tensor::add` is not defined.

**Step 3: Refactor traced helper implementation**

In `tenferro/src/traced.rs`:

- Import `tenferro_ops::broadcast::{broadcast_input_plan, broadcast_shape}`.
- Remove the local `broadcast_shape` function.
- Update private `broadcast_to` to call `broadcast_input_plan`.
- Keep the existing reshape-before-broadcast behavior for expanding singleton axes.
- Mark `broadcast_to` and `broadcast_binary` as `pub(crate)` so `traced_tensor.rs` can call them.

Use this shape:

```rust
pub(crate) fn broadcast_to(tensor: &TracedTensor, target_shape: &[usize]) -> TracedTensor {
    let tensor_shape = concrete_shape(tensor);
    if tensor_shape == target_shape {
        return tensor.clone();
    }
    let plan = broadcast_input_plan(&tensor_shape, target_shape).unwrap_or_else(|err| {
        panic!("{err}");
    });
    let source = if plan.source_shape == tensor_shape {
        tensor.clone()
    } else {
        tensor.reshape(&plan.source_shape)
    };
    source.broadcast_in_dim(target_shape, &plan.dims)
}
```

**Step 4: Add `traced_tensor::add`**

In `tenferro/src/traced_tensor.rs`, import the private lowering helper and add:

```rust
/// Elementwise addition with NumPy-style broadcasting.
///
/// # Examples
///
/// ```rust
/// # use tenferro::TracedTensor;
/// # let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
/// # let y = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);
/// let z = tenferro::traced_tensor::add(&x, &y);
/// ```
pub fn add(lhs: &TracedTensor, rhs: &TracedTensor) -> TracedTensor {
    lhs.add(rhs)
}
```

**Step 5: Run test**

Run:

```bash
cargo test -p tenferro --test numpy_api traced_add_uses_numpy_broadcasting_for_rank_padding_and_singletons
```

Expected: PASS.

**Step 6: Commit**

```bash
git add tenferro/src/traced.rs tenferro/src/traced_tensor.rs tenferro/tests/numpy_api.rs
git commit -m "feat: add traced tensor add free function"
```

## Task 4: Add Core Traced Free Functions

**Files:**
- Modify: `tenferro/src/traced.rs`
- Modify: `tenferro/src/traced_tensor.rs`
- Test: `tenferro/tests/numpy_api.rs`

**Step 1: Add failing tests for traced API names**

Extend `tenferro/tests/numpy_api.rs`:

```rust
use tenferro::CompareDir;

#[test]
fn traced_tensor_module_exposes_initial_elementwise_free_functions() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]);
    let y = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]);
    let cond = traced_tensor::compare(&x, &y, CompareDir::Gt);

    let _ = traced_tensor::sub(&x, &y);
    let _ = traced_tensor::mul(&x, &y);
    let _ = traced_tensor::div(&x, &y);
    let _ = traced_tensor::pow(&x, &y);
    let _ = traced_tensor::maximum(&x, &y);
    let _ = traced_tensor::minimum(&x, &y);
    let _ = traced_tensor::where_select(&cond, &x, &y);
    let _ = traced_tensor::clamp(&x, &y, &x);
    let _ = traced_tensor::neg(&x);
    let _ = traced_tensor::exp(&x);
    let _ = traced_tensor::log(&x);
    let _ = traced_tensor::sqrt(&x);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p tenferro --test numpy_api traced_tensor_module_exposes_initial_elementwise_free_functions
```

Expected: FAIL to compile on missing functions such as `sub`, `maximum`, `where_select`, or `clamp`.

**Step 3: Add traced lowering helpers**

In `tenferro/src/traced.rs`, add crate-private helpers that call existing
`apply_unary`, `apply_binary`, and a new `apply_ternary` if no ternary helper
exists yet:

```rust
pub(crate) fn apply_broadcast_binary_op(
    op: StdTensorOp,
    lhs: &TracedTensor,
    rhs: &TracedTensor,
) -> TracedTensor {
    let (lhs, rhs) = broadcast_binary(lhs, rhs);
    apply_binary(op, &lhs, &rhs, lhs.rank, lhs.shape_hint.clone())
}
```

Add a ternary helper that computes `broadcast_shapes([cond, on_true, on_false])`,
broadcasts all inputs to the common shape, and emits `StdTensorOp::Select` or
`StdTensorOp::Clamp`. Keep it crate-private and use the existing graph metadata
patterns from `apply_binary`.

**Step 4: Add free functions in `traced_tensor.rs`**

Add free functions for:

```rust
sub, mul, div, neg, abs, sign, conj, exp, log, sin, cos, tanh, sqrt, rsqrt,
expm1, log1p, maximum, minimum, compare, where_select, clamp
```

For `sub`, lower as `add(lhs, &neg(rhs))` unless a primitive `Sub` exists.
For `where_select`, call the ternary helper with `StdTensorOp::Select`.
For `clamp`, call the ternary helper with `StdTensorOp::Clamp`.

Every public function needs a minimal rustdoc `# Examples` section.

**Step 5: Run tests**

Run:

```bash
cargo test -p tenferro --test numpy_api
cargo test -p tenferro --doc traced_tensor
```

Expected: PASS.

**Step 6: Commit**

```bash
git add tenferro/src/traced.rs tenferro/src/traced_tensor.rs tenferro/tests/numpy_api.rs
git commit -m "feat: expose traced tensor numpy-style functions"
```

## Task 5: Add Eager Free Functions With Shared Broadcasting

**Files:**
- Modify: `tenferro/src/eager_tensor.rs`
- Test: `tenferro/tests/numpy_api.rs`

**Step 1: Add failing eager tests**

Extend `tenferro/tests/numpy_api.rs`:

```rust
#[test]
fn eager_add_uses_numpy_broadcasting_for_rank_padding_and_singletons() {
    let ctx = tenferro::EagerRuntime::new();
    let lhs = tenferro::EagerTensor::from_tensor_in(
        tenferro::Tensor::from_vec_row_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0]),
        ctx.clone(),
    );
    let rhs = tenferro::EagerTensor::from_tensor_in(
        tenferro::Tensor::from_vec_row_major(vec![1, 4], vec![10.0_f64, 20.0, 30.0, 40.0]),
        ctx,
    );

    let out = tenferro::eager_tensor::add(&lhs, &rhs).unwrap();

    assert_eq!(out.data().shape(), &[3, 4]);
    assert_eq!(
        out.data().clone().into_vec_row_major::<f64>().unwrap().1,
        vec![
            11.0, 21.0, 31.0, 41.0,
            12.0, 22.0, 32.0, 42.0,
            13.0, 23.0, 33.0, 43.0,
        ]
    );
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p tenferro --test numpy_api eager_add_uses_numpy_broadcasting_for_rank_padding_and_singletons
```

Expected: FAIL to compile because `tenferro::eager_tensor::add` is not defined.

**Step 3: Implement eager broadcast helpers**

In `tenferro/src/eager_tensor.rs`, add private helpers:

```rust
fn broadcast_to(input: &EagerTensor, target_shape: &[usize]) -> crate::Result<EagerTensor> {
    let input_shape = input.data().shape().to_vec();
    if input_shape == target_shape {
        return Ok(input.clone());
    }
    let plan = tenferro_ops::broadcast::broadcast_input_plan(&input_shape, target_shape)
        .map_err(|err| crate::Error::Internal(err.to_string()))?;
    let source = if plan.source_shape == input_shape {
        input.clone()
    } else {
        input.reshape(&plan.source_shape)?
    };
    source.broadcast_in_dim(target_shape, &plan.dims)
}
```

Also add `broadcast_binary` and `broadcast_ternary`.

**Step 4: Add eager free functions**

Add free functions for the same core set as traced:

```rust
pub fn add(lhs: &EagerTensor, rhs: &EagerTensor) -> crate::Result<EagerTensor> { ... }
pub fn where_select(cond: &EagerTensor, on_true: &EagerTensor, on_false: &EagerTensor) -> crate::Result<EagerTensor> { ... }
```

For binary functions, broadcast first and then call the existing methods.
For unary functions, call the existing method directly.
For `where_select`, broadcast all three inputs and call `EagerTensor::select`.
For `clamp`, broadcast input/lower/upper and call `input.clamp`.

Every public function needs a minimal rustdoc `# Examples` section.

**Step 5: Run tests**

Run:

```bash
cargo test -p tenferro --test numpy_api eager_add_uses_numpy_broadcasting_for_rank_padding_and_singletons
cargo test -p tenferro --test eager_tensor eager_elementwise_primal_ops_div_abs_and_sin
cargo test -p tenferro --doc eager_tensor
```

Expected: PASS.

**Step 6: Commit**

```bash
git add tenferro/src/eager_tensor.rs tenferro/tests/numpy_api.rs
git commit -m "feat: expose eager tensor numpy-style functions"
```

## Task 6: Add Concrete Tensor Free Functions

**Files:**
- Modify: `tenferro/src/tensor.rs`
- Test: `tenferro/tests/numpy_api.rs`

**Step 1: Add failing concrete tensor tests**

Extend `tenferro/tests/numpy_api.rs`:

```rust
#[test]
fn tensor_add_uses_numpy_broadcasting_with_explicit_backend() {
    let mut backend = tenferro::CpuBackend::new();
    let lhs = tenferro::Tensor::from_vec_row_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0]);
    let rhs = tenferro::Tensor::from_vec_row_major(vec![1, 4], vec![10.0_f64, 20.0, 30.0, 40.0]);

    let out = tenferro::tensor::add(&lhs, &rhs, &mut backend).unwrap();

    assert_eq!(out.shape(), &[3, 4]);
    assert_eq!(
        out.into_vec_row_major::<f64>().unwrap().1,
        vec![
            11.0, 21.0, 31.0, 41.0,
            12.0, 22.0, 32.0, 42.0,
            13.0, 23.0, 33.0, 43.0,
        ]
    );
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p tenferro --test numpy_api tensor_add_uses_numpy_broadcasting_with_explicit_backend
```

Expected: FAIL to compile because `tenferro::tensor::add` is not defined.

**Step 3: Implement concrete helpers**

In `tenferro/src/tensor.rs`, keep `pub use tenferro_tensor::Tensor;` and add
free functions. Use `tenferro_ops::broadcast` for planning and
`TensorBackend::with_exec_session` for primitive calls.

Private helper shape:

```rust
fn broadcast_to(
    input: &Tensor,
    target_shape: &[usize],
    backend: &mut impl tenferro_tensor::TensorBackend,
) -> tenferro_tensor::Result<Tensor> {
    let input_shape = input.shape().to_vec();
    if input_shape == target_shape {
        return Ok(input.clone());
    }
    let plan = tenferro_ops::broadcast::broadcast_input_plan(&input_shape, target_shape)
        .map_err(|err| tenferro_tensor::Error::BackendFailure {
            op: "broadcast",
            message: err.to_string(),
        })?;
    let source = if plan.source_shape == input_shape {
        input.clone()
    } else {
        backend.with_exec_session(|exec| exec.reshape(input, &plan.source_shape))?
    };
    backend.with_exec_session(|exec| exec.broadcast_in_dim(&source, target_shape, &plan.dims))
}
```

Add functions for the core set. For primitive calls not exposed as `Tensor`
methods, call `exec.div`, `exec.maximum`, `exec.select`, etc. directly inside
`with_exec_session`.

For `compare`, keep the current primitive output dtype for now and add a TODO
comment referencing the design's bool-output migration if the primitive still
returns numeric masks.

**Step 4: Run tests**

Run:

```bash
cargo test -p tenferro --test numpy_api tensor_add_uses_numpy_broadcasting_with_explicit_backend
cargo test -p tenferro-internal-tensor runtime_error_tests::add_rejects_shape_mismatch
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro/src/tensor.rs tenferro/tests/numpy_api.rs
git commit -m "feat: expose tensor numpy-style functions"
```

## Task 7: Add Typed Tensor Free Functions

**Files:**
- Modify: `tenferro/src/typed_tensor.rs`
- Test: `tenferro/tests/numpy_api.rs`

**Step 1: Add failing typed tensor tests**

Extend `tenferro/tests/numpy_api.rs`:

```rust
#[test]
fn typed_tensor_add_uses_numpy_broadcasting_with_explicit_backend() {
    let mut backend = tenferro::CpuBackend::new();
    let lhs = tenferro::TypedTensor::<f64>::from_vec_row_major(vec![3, 1], vec![1.0, 2.0, 3.0]);
    let rhs = tenferro::TypedTensor::<f64>::from_vec_row_major(
        vec![1, 4],
        vec![10.0, 20.0, 30.0, 40.0],
    );

    let out = tenferro::typed_tensor::add(&lhs, &rhs, &mut backend).unwrap();

    assert_eq!(out.shape, vec![3, 4]);
    assert_eq!(
        out.into_vec_row_major().unwrap(),
        vec![
            11.0, 21.0, 31.0, 41.0,
            12.0, 22.0, 32.0, 42.0,
            13.0, 23.0, 33.0, 43.0,
        ]
    );
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p tenferro --test numpy_api typed_tensor_add_uses_numpy_broadcasting_with_explicit_backend
```

Expected: FAIL to compile because `tenferro::typed_tensor::add` is not defined.

**Step 3: Implement typed functions**

In `tenferro/src/typed_tensor.rs`, keep `pub use tenferro_tensor::TypedTensor;`
and add free functions. Convert typed tensors to erased `Tensor`, call the
matching `tenferro::tensor::*` free function, and convert back with
`T::try_into_typed`.

Use a helper:

```rust
fn erase<T: tenferro_tensor::TensorScalar>(input: &TypedTensor<T>) -> tenferro_tensor::Tensor {
    T::into_tensor(input.shape.clone(), input.host_data().to_vec())
}
```

For same-dtype output functions, return `TypedTensor<T>`. For `compare`, either
return the current numeric `TypedTensor<T>` and document the staged bool
migration, or implement bool output only if the primitive compare migration is
completed first.

**Step 4: Run tests**

Run:

```bash
cargo test -p tenferro --test numpy_api typed_tensor_add_uses_numpy_broadcasting_with_explicit_backend
cargo test -p tenferro --test numpy_api
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro/src/typed_tensor.rs tenferro/tests/numpy_api.rs
git commit -m "feat: expose typed tensor numpy-style functions"
```

## Task 8: Extension Namespace Cleanup

**Files:**
- Modify: `tenferro-einsum/src/lib.rs`
- Create or modify: `tenferro-einsum/src/traced_tensor.rs`
- Modify: `tenferro-linalg/src/lib.rs`
- Create or modify: `tenferro-linalg/src/traced_tensor.rs`
- Modify: `tenferro-fft/src/lib.rs`
- Test: `tenferro-einsum/tests/traced_correctness.rs`
- Test: `tenferro-linalg/tests/traced_correctness.rs`
- Test: `tenferro-fft/tests/fft_ops.rs`

**Step 1: Add failing namespace tests**

Add one focused test per extension crate that compiles against the canonical
namespace:

```rust
// tenferro-einsum/tests/traced_correctness.rs
#[test]
fn traced_tensor_namespace_exposes_einsum() {
    let mut compiler = tenferro::GraphCompiler::new();
    let a = tenferro::TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]);
    let b = tenferro::TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]);
    let y = tenferro_einsum::traced_tensor::legacy_einsum(&mut legacy_compiler, &[&a, &b], "ij,jk->ik").unwrap();
    assert_eq!(y.rank, 2);
}
```

```rust
// tenferro-linalg/tests/traced_correctness.rs
#[test]
fn traced_tensor_namespace_exposes_svd() {
    let a = tenferro::TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]);
    let (_u, s, _vt) = tenferro_linalg::traced_tensor::svd(&a);
    assert_eq!(s.rank, 1);
}
```

```rust
// tenferro-fft/tests/fft_ops.rs
#[test]
fn traced_tensor_namespace_exposes_fft() {
    let x = tenferro::TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let y = tenferro_fft::traced_tensor::rfft(&x, None, -1, tenferro_fft::FftNorm::Backward);
    assert_eq!(y.rank, 1);
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-einsum traced_tensor_namespace_exposes_einsum
cargo test -p tenferro-linalg traced_tensor_namespace_exposes_svd
cargo test -p tenferro-fft traced_tensor_namespace_exposes_fft
```

Expected: FAIL to compile because `traced_tensor` modules do not exist.

**Step 3: Add extension namespaces**

For `tenferro-einsum`:

- Create `tenferro-einsum/src/traced_tensor.rs` that re-exports or wraps
  `crate::traced::{einsum, einsum_subscripts, einsum_with, einsum_subscripts_with}`.
- In `lib.rs`, add `pub mod traced_tensor;`.
- Keep existing root-level re-exports for compatibility unless docs say to remove
  them.

For `tenferro-linalg`:

- Create `tenferro-linalg/src/traced_tensor.rs` that re-exports
  `crate::traced::*`.
- In `lib.rs`, add `pub mod traced_tensor;`.
- Keep existing root-level re-exports for compatibility.

For `tenferro-fft`:

- Add inline `pub mod traced_tensor` in `src/lib.rs` or create a file if the
  module grows.
- Re-export `fft`, `ifft`, `rfft`, and `irfft`.
- Keep existing root functions as convenience entry points.

**Step 4: Run tests**

Run:

```bash
cargo test -p tenferro-einsum traced_tensor_namespace_exposes_einsum
cargo test -p tenferro-linalg traced_tensor_namespace_exposes_svd
cargo test -p tenferro-fft traced_tensor_namespace_exposes_fft
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-einsum/src/lib.rs tenferro-einsum/src/traced_tensor.rs tenferro-einsum/tests/traced_correctness.rs tenferro-linalg/src/lib.rs tenferro-linalg/src/traced_tensor.rs tenferro-linalg/tests/traced_correctness.rs tenferro-fft/src/lib.rs tenferro-fft/tests/fft_ops.rs
git commit -m "feat: align extension traced tensor namespaces"
```

## Task 9: Documentation And API Surface Cleanup

**Files:**
- Modify: `README.md`
- Modify as needed: `docs/guides/choosing-an-api.md`
- Modify as needed: `docs/guides/tensor-ops.md`
- Modify as needed: crate-level rustdoc in `tenferro/src/lib.rs`,
  `tenferro-einsum/src/lib.rs`, `tenferro-linalg/src/lib.rs`,
  `tenferro-fft/src/lib.rs`

**Step 1: Search for stale API claims**

Run:

```bash
rg -n "tenferro::(linalg|einsum|fft)|tenferro_linalg::(svd|qr|cholesky|eig|eigh|solve)|tenferro_einsum::traced_tensor::einsum|tenferro_fft::(fft|ifft|rfft|irfft)|\\.add\\(|\\.mul\\(" README.md docs tenferro*/src
```

Identify docs that should prefer canonical module free functions.

**Step 2: Update docs after public API exists**

Update examples to show:

```rust
let z = tenferro::tensor::add(&x, &y, &mut backend)?;
let z = tenferro::eager_tensor::add(&x, &y)?;
let z = tenferro::traced_tensor::add(&x, &y);
let y = tenferro_linalg::traced_tensor::svd(&x);
let y = tenferro_einsum::traced_tensor::legacy_einsum(&mut legacy_compiler, &[&a, &b], "ij,jk->ik")?;
```

Do not add `tenferro::linalg`, `tenferro::einsum`, or `tenferro::fft`.

**Step 3: Run doc checks**

Run:

```bash
cargo test --doc --workspace
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: PASS.

**Step 4: Commit**

```bash
git add README.md docs tenferro/src/lib.rs tenferro-einsum/src/lib.rs tenferro-linalg/src/lib.rs tenferro-fft/src/lib.rs
git commit -m "docs: document canonical tensor function namespaces"
```

## Task 10: Final Verification And Compare Follow-Up

**Files:**
- Maybe create GitHub issue if bool-returning public `compare` was not completed.
- No source changes unless verification finds a bug.

**Step 1: Confirm compare status**

If public `compare` still returns numeric masks, create or update a follow-up
issue that states:

- current primitive compare returns numeric masks for AD/backend compatibility,
- target public semantics are bool tensors,
- migration must update shape/dtype inference, eager/concrete backends, tests,
  and docs together.

If bool output was implemented in this branch, skip the follow-up and verify
tests assert `DType::Bool`.

**Step 2: Run focused verification**

Run:

```bash
cargo fmt --all --check
cargo test -p tenferro-internal-ops normalization_tests
cargo test -p tenferro --test numpy_api
cargo test -p tenferro-einsum traced_tensor_namespace_exposes_einsum
cargo test -p tenferro-linalg traced_tensor_namespace_exposes_svd
cargo test -p tenferro-fft traced_tensor_namespace_exposes_fft
```

Expected: PASS.

**Step 3: Run broad verification**

Run:

```bash
cargo test --workspace
cargo test --doc --workspace
```

Expected: PASS.

Do not claim PR readiness until the repository PR checklist is rerun:

```bash
cargo fmt --all --check
cargo test --workspace --release
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

**Step 4: Commit any final fix**

If any final verification fix is required:

```bash
git add <changed files>
git commit -m "fix: finalize numpy-style tensor API"
```

If no changes are needed, do not create an empty commit.
