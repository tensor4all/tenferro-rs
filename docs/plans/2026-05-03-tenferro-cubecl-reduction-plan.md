# tenferro-cubecl Reduction Split Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split CubeCL reduction kernels into a new `tenferro-cubecl` crate and route `tenferro-tensor` GPU reductions through it, adding GPU `i64` sum/prod support.

**Architecture:** `tenferro-cubecl` owns CubeCL reduction definitions, validation, launch policy, and kernels over `ComputeClient<R>` plus `TensorBinding<R>`. `tenferro-tensor` remains responsible for `TypedTensor` ownership, residency checks, output allocation, multi-axis reduction orchestration, error mapping, and final tenferro axis-removal shape semantics.

**Tech Stack:** Rust 2021, CubeCL fork `shinaoka/cubecl` rev `929c8a96`, `cubecl-cuda`, `thiserror`, `num-complex`, `tenferro-tensor`, cubek source adaptation from `../cubek` commit `9cf90b797107d46829e1c9d9355ce801c3dd4a7d`.

---

## Context

Issue: <https://github.com/tensor4all/tenferro-rs/issues/832>

Design doc: `docs/plans/2026-05-03-tenferro-cubecl-reduction-design.md`

Important constraints:

- Do not add a direct `cubek` dependency. The CubeCL revisions are intentionally different.
- Preserve cubek attribution for any adapted source:
  - source repository,
  - source commit,
  - source path,
  - original license,
  - tenferro-specific changes.
- `tenferro-cubecl` must not depend on `tenferro-tensor`.
- `tenferro-tensor` must keep device placement checks and tenferro error mapping.
- Column-major layout is part of the API boundary. Pass explicit shape and strides through `TensorBinding`.
- The new reduction crate should use single-axis keepdims reduction as its primitive. `tenferro-tensor` handles multi-axis reductions by keeping reduced dimensions as length 1 until the final metadata reshape.
- `i64 reduce_sum` and `i64 reduce_prod` are in scope. `i64 reduce_max` and `i64 reduce_min` stay unsupported.
- Complex `sum` and `prod` are in scope. Complex `max` and `min` stay unsupported.

## Current Code Pointers

- Workspace manifest: `Cargo.toml`
- Tensor crate manifest: `tenferro-tensor/Cargo.toml`
- CubeCL backend dispatch helpers: `tenferro-tensor/src/cubecl/dispatch.rs`
- Current GPU reduction dispatch methods: `tenferro-tensor/src/cubecl/mod.rs`
- Current GPU reduction kernels: `tenferro-tensor/src/cubecl/kernels/reduction.rs`
- Current GPU reduction tests: `tenferro-tensor/src/cubecl/tests/reduction_tests.rs`
- Existing column-major stride helper: `tenferro-tensor/src/types.rs` (`col_major_strides`)
- CubeCL `TensorBinding` constructor in the pinned CubeCL checkout:
  `~/.cargo/git/checkouts/cubecl-829d9c2a32488cd7/54f0cdb/crates/cubecl-core/src/frontend/container/tensor/launch.rs`

---

### Task 1: Add GPU Reduction Regression Tests

**Files:**

- Modify: `tenferro-tensor/src/cubecl/tests/reduction_tests.rs`

**Step 1: Add `tensor_i64` to the imports**

```rust
use super::{
    assert_tensor_close, cpu_backend, download, gpu_backend, tensor_c64, tensor_f64, tensor_i64,
    upload,
};
```

**Step 2: Add a failing `i64` sum/prod GPU parity test**

Append:

```rust
#[test]
#[ignore]
fn test_cubecl_i64_sum_and_prod_match_cpu() {
    let input = tensor_i64(
        vec![2, 3, 2],
        vec![1, 2, 3, 4, 5, 6, -1, -2, 2, 3, -3, 4],
    );

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);

    let expected = cpu.reduce_sum(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_sum(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu.reduce_prod(&input, &[2]).unwrap();
    let gpu_out = gpu.reduce_prod(&gpu_input, &[2]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);
}
```

**Step 3: Add a column-major 3D axis-removal guard test**

Append:

```rust
#[test]
#[ignore]
fn test_cubecl_reductions_column_major_3d_axes_match_cpu() {
    let input = tensor_f64(
        vec![2, 3, 4],
        (1..=24).map(|value| value as f64 - 7.0).collect(),
    );

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);

    for axes in [&[0][..], &[1][..], &[2][..], &[0, 2][..]] {
        let expected = cpu.reduce_sum(&input, axes).unwrap();
        let gpu_out = gpu.reduce_sum(&gpu_input, axes).unwrap();
        let actual = download(&gpu, &gpu_out);
        assert_eq!(actual.shape(), expected.shape());
        assert_tensor_close(&actual, &expected, 1e-12);

        let expected = cpu.reduce_prod(&input, axes).unwrap();
        let gpu_out = gpu.reduce_prod(&gpu_input, axes).unwrap();
        let actual = download(&gpu, &gpu_out);
        assert_eq!(actual.shape(), expected.shape());
        assert_tensor_close(&actual, &expected, 1e-12);

        let expected = cpu.reduce_max(&input, axes).unwrap();
        let gpu_out = gpu.reduce_max(&gpu_input, axes).unwrap();
        let actual = download(&gpu, &gpu_out);
        assert_eq!(actual.shape(), expected.shape());
        assert_tensor_close(&actual, &expected, 1e-12);

        let expected = cpu.reduce_min(&input, axes).unwrap();
        let gpu_out = gpu.reduce_min(&gpu_input, axes).unwrap();
        let actual = download(&gpu, &gpu_out);
        assert_eq!(actual.shape(), expected.shape());
        assert_tensor_close(&actual, &expected, 1e-12);
    }
}
```

**Step 4: Run the ignored tests on a CUDA machine**

Run:

```bash
CUBECL_DEBUG_LOG=0 \
CUDA_PATH=/usr/local/cuda-12.0 \
LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH \
  cargo test -p tenferro-tensor --features cubecl reduction -- --ignored
```

Expected now:

- `test_cubecl_i64_sum_and_prod_match_cpu` fails with unsupported dtype for `reduce_sum` or `reduce_prod`.
- Existing float/complex tests should still compile.

If no CUDA device is available, still run:

```bash
cargo test -p tenferro-tensor --features cubecl reduction_tests --no-run
```

Expected: tests compile.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/cubecl/tests/reduction_tests.rs
git commit -m "test: add cubecl reduction parity coverage"
```

---

### Task 2: Scaffold `tenferro-cubecl`

**Files:**

- Modify: `Cargo.toml`
- Modify: `tenferro-tensor/Cargo.toml`
- Create: `tenferro-cubecl/Cargo.toml`
- Create: `tenferro-cubecl/src/lib.rs`
- Create: `tenferro-cubecl/src/error.rs`
- Create: `tenferro-cubecl/src/reduce/mod.rs`
- Create: `tenferro-cubecl/src/reduce/definition.rs`
- Create: `tenferro-cubecl/src/reduce/launch.rs`
- Create: `tenferro-cubecl/src/reduce/routines.rs`
- Create: `tenferro-cubecl/src/reduce/kernels.rs`
- Create: `tenferro-cubecl/src/reduce/cpu_reference.rs`

**Step 1: Add the workspace member and dependency**

In `Cargo.toml`, add `"tenferro-cubecl"` to `[workspace].members`.

In `[workspace.dependencies]`, add:

```toml
tenferro-cubecl = { path = "tenferro-cubecl" }
```

**Step 2: Add the optional tensor dependency**

In `tenferro-tensor/Cargo.toml`, update the feature:

```toml
cubecl = [
    "dep:cubecl",
    "dep:cubecl-cuda",
    "dep:cubecl-runtime",
    "dep:cudarc",
    "dep:tenferro-cubecl",
]
```

Add the dependency:

```toml
tenferro-cubecl = { workspace = true, optional = true }
```

**Step 3: Create `tenferro-cubecl/Cargo.toml`**

```toml
[package]
name = "tenferro-cubecl"
version.workspace = true
edition.workspace = true
license.workspace = true
authors.workspace = true
publish.workspace = true
description = "CubeCL kernels and launch helpers for tenferro."

[features]
default = []
cpu-reference = []

[dependencies]
cubecl.workspace = true
thiserror.workspace = true
num-complex.workspace = true

[dev-dependencies]
cubecl-cuda.workspace = true
```

**Step 4: Create crate root docs**

`tenferro-cubecl/src/lib.rs`:

```rust
//! CubeCL kernels and launch helpers for tenferro.
//!
//! This crate owns GPU kernel definitions but does not own tenferro tensor
//! values, device placement, or backend dispatch.
//!
//! # Examples
//!
//! ```ignore
//! use tenferro_cubecl::reduce::{ReduceOp, ReduceStrategy};
//!
//! let _op = ReduceOp::Sum;
//! let _strategy = ReduceStrategy::Auto;
//! ```

pub mod error;
pub mod reduce;

pub use error::{CubeclKernelError, Result};
```

**Step 5: Create `error.rs`**

```rust
use thiserror::Error;

/// Error returned by tenferro CubeCL kernel launch helpers.
///
/// # Examples
///
/// ```
/// use tenferro_cubecl::CubeclKernelError;
///
/// let err = CubeclKernelError::InvalidAxis { axis: 3, rank: 2 };
/// assert!(err.to_string().contains("axis"));
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum CubeclKernelError {
    #[error("axis {axis} is out of bounds for rank {rank}")]
    InvalidAxis { axis: usize, rank: usize },

    #[error("output shape {actual:?} does not match expected keepdims shape {expected:?}")]
    MismatchOutputShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("{op:?} does not support dtype {dtype:?}")]
    UnsupportedDType { op: crate::reduce::ReduceOp, dtype: crate::reduce::ReduceDType },

    #[error("invalid reduction strategy: {reason}")]
    InvalidStrategy { reason: String },
}

/// Result alias for tenferro CubeCL kernel helpers.
///
/// # Examples
///
/// ```
/// use tenferro_cubecl::Result;
///
/// fn ok() -> Result<()> { Ok(()) }
/// assert!(ok().is_ok());
/// ```
pub type Result<T> = core::result::Result<T, CubeclKernelError>;
```

**Step 6: Create `reduce/mod.rs`**

```rust
//! Reduction kernels.
//!
//! The public launch functions reduce one axis and expect keepdims output
//! shape. Higher-level tensor crates can call them repeatedly for multi-axis
//! reductions and then reshape metadata to their public output convention.
//!
//! # Examples
//!
//! ```ignore
//! use tenferro_cubecl::reduce::{ReduceOp, keepdims_output_shape};
//!
//! assert_eq!(keepdims_output_shape(&[2, 3, 4], 1).unwrap(), vec![2, 1, 4]);
//! let _op = ReduceOp::Prod;
//! ```

mod definition;
mod kernels;
mod launch;
mod routines;

#[cfg(feature = "cpu-reference")]
pub mod cpu_reference;

#[cfg(test)]
mod tests;

pub use definition::{ReduceDType, ReduceOp, keepdims_output_shape, validate_axis};
pub use launch::{
    ReduceStrategy, launch_max_float, launch_min_float, launch_prod_complex, launch_prod_float,
    launch_prod_int, launch_sum_complex, launch_sum_float, launch_sum_int,
};
```

**Step 7: Create placeholder modules**

Add minimal compiling placeholders. `reduce/mod.rs` re-exports every launch
function that `tenferro-tensor` will call later, so every function listed there
must exist before running `cargo check`.

```rust
// definition.rs
use crate::{CubeclKernelError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Prod,
    Max,
    Min,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceDType {
    F32,
    F64,
    I64,
    Complex32,
    Complex64,
}

pub fn validate_axis(rank: usize, axis: usize) -> Result<()> {
    if axis >= rank {
        return Err(CubeclKernelError::InvalidAxis { axis, rank });
    }
    Ok(())
}

pub fn keepdims_output_shape(input_shape: &[usize], axis: usize) -> Result<Vec<usize>> {
    validate_axis(input_shape.len(), axis)?;
    let mut output = input_shape.to_vec();
    output[axis] = 1;
    Ok(output)
}
```

```rust
// launch.rs
use cubecl::prelude::*;

use crate::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceStrategy {
    Auto,
    Unit,
}

pub fn launch_sum_float<R: Runtime, F: Float + CubeElement>(
    _client: &ComputeClient<R>,
    _input: TensorBinding<R>,
    _output: TensorBinding<R>,
    _axis: usize,
    _strategy: ReduceStrategy,
) -> Result<()> {
    todo!("implemented in Task 5")
}

pub fn launch_sum_int<R: Runtime, I: Int + CubeElement>(
    _client: &ComputeClient<R>,
    _input: TensorBinding<R>,
    _output: TensorBinding<R>,
    _axis: usize,
    _strategy: ReduceStrategy,
) -> Result<()> {
    todo!("implemented in Task 5")
}

pub fn launch_sum_complex<R: Runtime, C: Complex + CubeElement>(
    _client: &ComputeClient<R>,
    _input: TensorBinding<R>,
    _output: TensorBinding<R>,
    _axis: usize,
    _strategy: ReduceStrategy,
) -> Result<()> {
    todo!("implemented in Task 5")
}

pub fn launch_prod_float<R: Runtime, F: Float + CubeElement>(
    _client: &ComputeClient<R>,
    _input: TensorBinding<R>,
    _output: TensorBinding<R>,
    _axis: usize,
    _strategy: ReduceStrategy,
) -> Result<()> {
    todo!("implemented in Task 5")
}

pub fn launch_prod_int<R: Runtime, I: Int + CubeElement>(
    _client: &ComputeClient<R>,
    _input: TensorBinding<R>,
    _output: TensorBinding<R>,
    _axis: usize,
    _strategy: ReduceStrategy,
) -> Result<()> {
    todo!("implemented in Task 5")
}

pub fn launch_prod_complex<R: Runtime, C: Complex + CubeElement>(
    _client: &ComputeClient<R>,
    _input: TensorBinding<R>,
    _output: TensorBinding<R>,
    _axis: usize,
    _strategy: ReduceStrategy,
) -> Result<()> {
    todo!("implemented in Task 5")
}

pub fn launch_max_float<R: Runtime, F: Float + CubeElement>(
    _client: &ComputeClient<R>,
    _input: TensorBinding<R>,
    _output: TensorBinding<R>,
    _axis: usize,
    _strategy: ReduceStrategy,
) -> Result<()> {
    todo!("implemented in Task 5")
}

pub fn launch_min_float<R: Runtime, F: Float + CubeElement>(
    _client: &ComputeClient<R>,
    _input: TensorBinding<R>,
    _output: TensorBinding<R>,
    _axis: usize,
    _strategy: ReduceStrategy,
) -> Result<()> {
    todo!("implemented in Task 5")
}
```

Do not route `tenferro-tensor` to the new crate until Task 6.

**Step 8: Run compile checks**

Run:

```bash
cargo fmt --all --check
cargo check -p tenferro-cubecl
cargo check -p tenferro-tensor --features cubecl
```

Expected: pass after placeholders compile.

**Step 9: Commit**

```bash
git add Cargo.toml tenferro-tensor/Cargo.toml tenferro-cubecl
git commit -m "feat: add tenferro-cubecl crate"
```

---

### Task 3: Add Reduction Validation and CPU Reference Helpers

**Files:**

- Modify: `tenferro-cubecl/src/reduce/definition.rs`
- Modify: `tenferro-cubecl/src/reduce/launch.rs`
- Modify: `tenferro-cubecl/src/reduce/cpu_reference.rs`
- Create: `tenferro-cubecl/src/reduce/tests/mod.rs`
- Create: `tenferro-cubecl/src/reduce/tests/validation.rs`
- Create: `tenferro-cubecl/src/reduce/tests/cpu_reference.rs`

**Step 1: Add validation helpers**

Implement:

```rust
pub fn validate_keepdims_output_shape(
    input_shape: &[usize],
    output_shape: &[usize],
    axis: usize,
) -> Result<()> {
    let expected = keepdims_output_shape(input_shape, axis)?;
    if output_shape != expected {
        return Err(CubeclKernelError::MismatchOutputShape {
            expected,
            actual: output_shape.to_vec(),
        });
    }
    Ok(())
}

pub fn axis_reduce_len(input_shape: &[usize], axis: usize) -> Result<usize> {
    validate_axis(input_shape.len(), axis)?;
    Ok(input_shape[axis])
}

pub fn reduced_output_len(input_shape: &[usize], axis: usize) -> Result<usize> {
    let reduce_len = axis_reduce_len(input_shape, axis)?;
    Ok(input_shape.iter().product::<usize>() / reduce_len)
}
```

**Step 2: Add support table helpers**

Implement:

```rust
pub fn supports_dtype(op: ReduceOp, dtype: ReduceDType) -> bool {
    match (op, dtype) {
        (ReduceOp::Sum | ReduceOp::Prod, ReduceDType::F32 | ReduceDType::F64) => true,
        (ReduceOp::Sum | ReduceOp::Prod, ReduceDType::I64) => true,
        (ReduceOp::Sum | ReduceOp::Prod, ReduceDType::Complex32 | ReduceDType::Complex64) => true,
        (ReduceOp::Max | ReduceOp::Min, ReduceDType::F32 | ReduceDType::F64) => true,
        _ => false,
    }
}
```

**Step 3: Add tests**

`tenferro-cubecl/src/reduce/tests/mod.rs`:

```rust
mod cpu_reference;
mod validation;
```

`validation.rs`:

```rust
use crate::CubeclKernelError;
use crate::reduce::{
    ReduceDType, ReduceOp, keepdims_output_shape, supports_dtype,
    validate_keepdims_output_shape,
};

#[test]
fn keepdims_output_shape_sets_only_reduced_axis_to_one() {
    assert_eq!(keepdims_output_shape(&[2, 3, 4], 1).unwrap(), vec![2, 1, 4]);
}

#[test]
fn keepdims_output_shape_rejects_axis_equal_to_rank() {
    let err = keepdims_output_shape(&[2, 3], 2).unwrap_err();
    assert_eq!(err, CubeclKernelError::InvalidAxis { axis: 2, rank: 2 });
}

#[test]
fn validate_keepdims_output_shape_reports_expected_shape() {
    let err = validate_keepdims_output_shape(&[2, 3, 4], &[2, 3, 1], 1).unwrap_err();
    assert_eq!(
        err,
        CubeclKernelError::MismatchOutputShape {
            expected: vec![2, 1, 4],
            actual: vec![2, 3, 1],
        }
    );
}

#[test]
fn support_table_matches_first_split_scope() {
    assert!(supports_dtype(ReduceOp::Sum, ReduceDType::I64));
    assert!(supports_dtype(ReduceOp::Prod, ReduceDType::Complex64));
    assert!(supports_dtype(ReduceOp::Max, ReduceDType::F32));
    assert!(!supports_dtype(ReduceOp::Max, ReduceDType::I64));
    assert!(!supports_dtype(ReduceOp::Min, ReduceDType::Complex32));
}
```

**Step 4: Add CPU reference helper tests**

Implement simple host helpers behind `cpu-reference`:

```rust
pub fn reduce_sum_i64_keepdims(input: &[i64], input_shape: &[usize], axis: usize) -> Vec<i64> {
    reduce_keepdims(input, input_shape, axis, 0_i64, |acc, value| acc + value)
}
```

Use column-major index conversion in the helper so tests exercise the intended layout contract.

`cpu_reference.rs` test:

```rust
#[cfg(feature = "cpu-reference")]
use crate::reduce::cpu_reference::reduce_sum_i64_keepdims;

#[test]
#[cfg(feature = "cpu-reference")]
fn cpu_reference_uses_column_major_axis_order() {
    let input = vec![1, 2, 3, 4, 5, 6];
    assert_eq!(reduce_sum_i64_keepdims(&input, &[2, 3], 0), vec![3, 7, 11]);
    assert_eq!(reduce_sum_i64_keepdims(&input, &[2, 3], 1), vec![9, 12]);
}
```

**Step 5: Run tests**

Run:

```bash
cargo test -p tenferro-cubecl
cargo test -p tenferro-cubecl --features cpu-reference
```

Expected: pass.

**Step 6: Commit**

```bash
git add tenferro-cubecl/src/reduce
git commit -m "feat: add cubecl reduction validation"
```

---

### Task 4: Add TensorBinding Adapter Helpers in `tenferro-tensor`

**Files:**

- Modify: `tenferro-tensor/src/cubecl/dispatch.rs`
- Modify: `tenferro-tensor/src/cubecl/tests/mod.rs`
- Create: `tenferro-tensor/src/cubecl/tests/metadata_tests.rs`

**Step 1: Add pure metadata helper**

In `dispatch.rs`, add:

```rust
pub(crate) fn cubecl_shape_and_strides(shape: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let strides = crate::types::col_major_strides(shape)
        .into_iter()
        .map(|stride| stride as usize)
        .collect();
    (shape.to_vec(), strides)
}
```

**Step 2: Add binding helper**

In `dispatch.rs`, add:

```rust
pub(crate) fn typed_tensor_binding<T: CubeElement + Clone>(
    tensor: &TypedTensor<T>,
    op: &'static str,
) -> crate::Result<TensorBinding<CudaRuntime>> {
    let buffer = cubecl_buffer(tensor, op)?;
    let (shape, strides) = cubecl_shape_and_strides(&tensor.shape);

    // SAFETY: `buffer.handle` owns a CubeCL allocation with `buffer.len`
    // elements. `shape` is the tensor shape and `strides` is the dense
    // column-major stride vector for that shape.
    Ok(unsafe { TensorBinding::from_raw_parts(buffer.handle.clone(), strides.into(), shape.into()) })
}
```

If the exact `Shape`/`Strides` conversions do not compile, check the pinned CubeCL `TensorBinding::from_raw_parts` signature and use the concrete conversion accepted by `cubecl-zspace`.

**Step 3: Add metadata tests**

In `tenferro-tensor/src/cubecl/tests/mod.rs`, add:

```rust
mod metadata_tests;
```

Create `metadata_tests.rs`:

```rust
use crate::cubecl::dispatch::cubecl_shape_and_strides;

#[test]
fn cubecl_metadata_uses_dense_column_major_strides() {
    assert_eq!(cubecl_shape_and_strides(&[]), (vec![], vec![]));
    assert_eq!(cubecl_shape_and_strides(&[2, 3, 4]), (vec![2, 3, 4], vec![1, 2, 6]));
}
```

**Step 4: Run tests**

Run:

```bash
cargo test -p tenferro-tensor --features cubecl metadata_tests
```

Expected: pass.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/cubecl/dispatch.rs tenferro-tensor/src/cubecl/tests/mod.rs tenferro-tensor/src/cubecl/tests/metadata_tests.rs
git commit -m "feat: add cubecl tensor binding metadata"
```

---

### Task 5: Implement `tenferro-cubecl` Single-Axis Keepdims Kernels

**Files:**

- Modify: `tenferro-cubecl/src/reduce/launch.rs`
- Modify: `tenferro-cubecl/src/reduce/routines.rs`
- Modify: `tenferro-cubecl/src/reduce/kernels.rs`
- Modify: `tenferro-cubecl/src/reduce/definition.rs`

**Step 1: Add attribution headers to cubek-derived files**

At the top of `launch.rs`, `routines.rs`, and any file with adapted cubek logic:

```rust
// Portions of this file are adapted from cubek-reduce:
// https://github.com/tracel-ai/cubek/tree/9cf90b797107d46829e1c9d9355ce801c3dd4a7d/crates/cubek-reduce
//
// Original source paths:
// - crates/cubek-reduce/src/launch/base.rs
// - crates/cubek-reduce/src/launch/strategy.rs
// - crates/cubek-reduce/src/routines/unit.rs
// - crates/cubek-reduce/src/routines/blueprint.rs
//
// Original license: MIT OR Apache-2.0.
// Tenferro changes: narrowed to tenferro reduction ops, current CubeCL fork,
// single-axis keepdims output, and explicit tenferro column-major bindings.
```

Only add this header to files where cubek code or structure is actually adapted.

**Step 2: Define a small strategy and problem surface**

In `routines.rs`, implement:

```rust
use cubecl::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReduceProblem {
    pub reduce_len: usize,
    pub reduce_count: usize,
    pub axis: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UnitReduceBlueprint {
    pub idle_units: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReduceLaunchSettings {
    pub cube_count: CubeCount,
    pub cube_dim: CubeDim,
    pub blueprint: UnitReduceBlueprint,
}

pub fn unit_launch_settings<R: Runtime>(
    client: &ComputeClient<R>,
    problem: ReduceProblem,
) -> ReduceLaunchSettings {
    let plane_size = client.properties().hardware.plane_size_max;
    let cube_dim = CubeDim::new_1d(plane_size);
    let units_per_cube = cube_dim.num_elems() as usize;
    let cubes = problem.reduce_count.div_ceil(units_per_cube).max(1) as u32;
    ReduceLaunchSettings {
        cube_count: CubeCount::Static(cubes, 1, 1),
        cube_dim,
        blueprint: UnitReduceBlueprint {
            idle_units: problem.reduce_count % units_per_cube != 0,
        },
    }
}
```

If `CubeCount::Static` is rejected by the CubeCL API, use the existing tenferro pattern from `tenferro-tensor/src/cubecl/dispatch.rs`.

**Step 3: Implement launch validation**

In `launch.rs`, before each kernel launch:

```rust
fn validate_launch<R: Runtime>(
    input: &TensorBinding<R>,
    output: &TensorBinding<R>,
    axis: usize,
) -> Result<ReduceProblem> {
    crate::reduce::validate_keepdims_output_shape(&input.shape, &output.shape, axis)?;
    let reduce_len = input.shape[axis];
    let input_len = input.shape.iter().product::<usize>();
    let reduce_count = input_len / reduce_len;
    Ok(ReduceProblem {
        reduce_len,
        reduce_count,
        axis,
    })
}
```

**Step 4: Implement typed launch wrappers**

Expose these functions from `launch.rs`:

```rust
pub fn launch_sum_float<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    axis: usize,
    strategy: ReduceStrategy,
) -> Result<()>;

pub fn launch_sum_int<R: Runtime, I: Int + CubeElement>(...) -> Result<()>;
pub fn launch_sum_complex<R: Runtime, C: Complex + CubeElement>(...) -> Result<()>;
pub fn launch_prod_float<R: Runtime, F: Float + CubeElement>(...) -> Result<()>;
pub fn launch_prod_int<R: Runtime, I: Int + CubeElement>(...) -> Result<()>;
pub fn launch_prod_complex<R: Runtime, C: Complex + CubeElement>(...) -> Result<()>;
pub fn launch_max_float<R: Runtime, F: Float + CubeElement>(...) -> Result<()>;
pub fn launch_min_float<R: Runtime, F: Float + CubeElement>(...) -> Result<()>;
```

Each wrapper should:

1. Validate axis and keepdims output shape.
2. Select `Unit` for both `ReduceStrategy::Auto` and `ReduceStrategy::Unit` in this first PR.
3. Launch the matching kernel.

**Step 5: Implement kernels using TensorBinding layout**

Use `Tensor<T>` kernel arguments, not `Array<T>`, so shape/strides come from `TensorBinding`.

Keep the algorithm simple in PR1:

- one output element per unit,
- one unit loops over the reduced axis,
- use the output linear index to reconstruct coordinates excluding the reduced axis,
- set the reduced coordinate to each `k`,
- read from input using the tensor index/metadata APIs,
- write into keepdims output.

Do not reintroduce helper functions that assume dense layout from `shape` alone. When a flat offset is needed, compute it from runtime strides.

If the CubeCL `Tensor<T>` API makes dynamic rank indexing awkward, use a small fixed-rank loop over `input.rank()` or `input.shape.len()` as supported by the pinned CubeCL API. Keep shape/stride values runtime metadata, not `#[comptime] Sequence`.

**Step 6: Preserve complex support**

Do not rely only on cubek's `Numeric` traits if they reject complex types in the pinned fork. Keep separate complex kernels with `C: Complex` for `sum` and `prod`, mirroring the current `tenferro-tensor/src/cubecl/kernels/reduction.rs` trait split.

**Step 7: Run compile checks**

Run:

```bash
cargo fmt --all --check
cargo check -p tenferro-cubecl
cargo test -p tenferro-cubecl
```

Expected: pass.

**Step 8: Optional CUDA smoke tests inside `tenferro-cubecl`**

If adding crate-local ignored CUDA tests is practical, create:

- `tenferro-cubecl/tests/reduce_cuda.rs`

Mark tests `#[ignore]`, and cover:

- `f64` sum axis 0 on shape `[2, 3]`,
- `i64` prod axis 1 on shape `[2, 3]`.

Run:

```bash
CUBECL_DEBUG_LOG=0 \
CUDA_PATH=/usr/local/cuda-12.0 \
LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH \
  cargo test -p tenferro-cubecl -- --ignored
```

**Step 9: Commit**

```bash
git add tenferro-cubecl
git commit -m "feat: implement tenferro cubecl reductions"
```

---

### Task 6: Route `CubeclBackend` Reductions Through `tenferro-cubecl`

**Files:**

- Modify: `tenferro-tensor/src/cubecl/mod.rs`
- Modify: `tenferro-tensor/src/cubecl/dispatch.rs`

**Step 1: Import the new crate functions**

In `tenferro-tensor/src/cubecl/mod.rs`, remove `reduction` from:

```rust
use kernels::{diagonal, elementwise, indexing, reduction, structural};
```

Use:

```rust
use kernels::{diagonal, elementwise, indexing, structural};
use tenferro_cubecl::reduce::{self as cubecl_reduce, ReduceStrategy};
```

**Step 2: Add output shape helpers**

Keep the existing `reduction_output_shape` helper for final tenferro output shape.

Add a keepdims helper near it:

```rust
fn reduction_keepdims_shape(input_shape: &[usize], axis: usize) -> Vec<usize> {
    let mut output_shape = input_shape.to_vec();
    output_shape[axis] = 1;
    output_shape
}
```

**Step 3: Add a final metadata reshape helper**

In `dispatch.rs` or `mod.rs`, add:

```rust
fn cubecl_reshape_metadata<T: CubeElement + Clone>(
    tensor: TypedTensor<T>,
    shape: Vec<usize>,
    op: &'static str,
) -> crate::Result<TypedTensor<T>> {
    let len = shape.iter().product::<usize>();
    if len != tensor.n_elements() {
        return Err(crate::Error::BackendFailure {
            op,
            message: format!(
                "cannot reshape CubeCL output metadata from {:?} to {:?}",
                tensor.shape, shape
            ),
        });
    }
    Ok(TypedTensor { shape, ..tensor })
}
```

**Step 4: Add a generic single-axis allocation/launch helper**

In the `impl CubeclBackend` block, add a helper that allocates keepdims output, creates `TensorBinding`s, and calls a closure:

```rust
fn launch_reduce_axis_typed<T>(
    &self,
    input: &TypedTensor<T>,
    axis: usize,
    op: &'static str,
    launch: impl FnOnce(
        &cubecl::client::ComputeClient<CudaRuntime>,
        TensorBinding<CudaRuntime>,
        TensorBinding<CudaRuntime>,
    ) -> tenferro_cubecl::Result<()>,
) -> crate::Result<TypedTensor<T>>
where
    T: CubeElement + Clone,
{
    let output_shape = reduction_keepdims_shape(&input.shape, axis);
    let output = alloc_output::<T>(self.runtime(), &output_shape);
    if output.n_elements() == 0 {
        return Ok(output);
    }

    let input_binding = typed_tensor_binding(input, op)?;
    let output_binding = typed_tensor_binding(&output, op)?;
    launch(self.runtime().client(), input_binding, output_binding).map_err(|err| {
        crate::Error::BackendFailure {
            op,
            message: err.to_string(),
        }
    })?;
    Ok(output)
}
```

Adjust visibility/imports to match current module structure.

**Step 5: Add multi-axis orchestration helpers**

Add typed helpers:

```rust
fn reduce_axes_typed<T>(
    &self,
    input: &TypedTensor<T>,
    axes: &[usize],
    op: &'static str,
    mut launch_axis: impl FnMut(&Self, &TypedTensor<T>, usize) -> crate::Result<TypedTensor<T>>,
) -> crate::Result<TypedTensor<T>>
where
    T: CubeElement + Clone,
{
    ensure_axes_unique(op, "axes", axes, input.shape.len())?;
    if axes.is_empty() {
        return Ok(input.clone());
    }

    let final_shape = reduction_output_shape(input.shape.as_slice(), axes);
    let mut sorted_axes = axes.to_vec();
    sorted_axes.sort_unstable();

    let mut current = input.clone();
    for axis in sorted_axes {
        current = launch_axis(self, &current, axis)?;
    }

    cubecl_reshape_metadata(current, final_shape, op)
}
```

Because intermediate outputs use keepdims shape, original axis numbers remain valid after each reduction.

**Step 6: Replace float/complex reduction typed methods**

Replace the bodies of:

- `reduce_sum_float_typed`
- `reduce_sum_complex_typed`
- `reduce_prod_float_typed`
- `reduce_prod_complex_typed`
- `reduce_max_typed`
- `reduce_min_typed`

with calls through `reduce_axes_typed` and `launch_reduce_axis_typed`.

Example for float sum:

```rust
fn reduce_sum_float_typed<F: CubeElement + CubeFloat + Clone>(
    &self,
    input: &TypedTensor<F>,
    axes: &[usize],
) -> crate::Result<TypedTensor<F>> {
    self.reduce_axes_typed(input, axes, "reduce_sum", |backend, current, axis| {
        backend.launch_reduce_axis_typed(current, axis, "reduce_sum", |client, input, output| {
            cubecl_reduce::launch_sum_float::<CudaRuntime, F>(
                client,
                input,
                output,
                axis,
                ReduceStrategy::Auto,
            )
        })
    })
}
```

Use the matching `tenferro-cubecl` function for complex, int, max, and min.

**Step 7: Add i64 reduction typed methods**

Add:

```rust
fn reduce_sum_int_typed<I: CubeElement + Int + Clone>(
    &self,
    input: &TypedTensor<I>,
    axes: &[usize],
) -> crate::Result<TypedTensor<I>> {
    self.reduce_axes_typed(input, axes, "reduce_sum", |backend, current, axis| {
        backend.launch_reduce_axis_typed(current, axis, "reduce_sum", |client, input, output| {
            cubecl_reduce::launch_sum_int::<CudaRuntime, I>(
                client,
                input,
                output,
                axis,
                ReduceStrategy::Auto,
            )
        })
    })
}

fn reduce_prod_int_typed<I: CubeElement + Int + Clone>(
    &self,
    input: &TypedTensor<I>,
    axes: &[usize],
) -> crate::Result<TypedTensor<I>> {
    self.reduce_axes_typed(input, axes, "reduce_prod", |backend, current, axis| {
        backend.launch_reduce_axis_typed(current, axis, "reduce_prod", |client, input, output| {
            cubecl_reduce::launch_prod_int::<CudaRuntime, I>(
                client,
                input,
                output,
                axis,
                ReduceStrategy::Auto,
            )
        })
    })
}
```

Use the exact integer trait name exported by the pinned CubeCL prelude (`Int` in the current checkout).

**Step 8: Enable i64 dispatch for sum/prod**

Change:

```rust
Tensor::I64(_) => Err(unsupported_dtype("reduce_sum", input.dtype())),
```

to:

```rust
Tensor::I64(t) => self.reduce_sum_int_typed(t, axes).map(Tensor::I64),
```

Do the same for `reduce_prod`.

Leave `reduce_max` and `reduce_min` i64 arms unsupported.

**Step 9: Run checks**

Run:

```bash
cargo fmt --all --check
cargo test -p tenferro-tensor --features cubecl reduction_tests --no-run
cargo test -p tenferro-tensor
```

Expected: pass.

**Step 10: Commit**

```bash
git add tenferro-tensor/src/cubecl/mod.rs tenferro-tensor/src/cubecl/dispatch.rs
git commit -m "feat: route cubecl reductions through tenferro-cubecl"
```

---

### Task 7: Remove Old In-Crate Reduction Kernels

**Files:**

- Modify: `tenferro-tensor/src/cubecl/kernels/mod.rs`
- Delete: `tenferro-tensor/src/cubecl/kernels/reduction.rs`
- Modify: any docs that mention the old reduction kernel location

**Step 1: Remove the old module declaration**

In `tenferro-tensor/src/cubecl/kernels/mod.rs`, delete:

```rust
pub(crate) mod reduction;
```

**Step 2: Delete the old kernel file**

Delete:

```text
tenferro-tensor/src/cubecl/kernels/reduction.rs
```

**Step 3: Search for stale references**

Run:

```bash
rg -n "kernels::reduction|mod reduction|reduce_sum_float|reduce_prod_float|reduce_max_float|reduce_min_float" tenferro-tensor/src tenferro-cubecl/src docs
```

Expected:

- No stale `tenferro-tensor` references.
- New `tenferro-cubecl` references are expected.

**Step 4: Run checks**

Run:

```bash
cargo fmt --all --check
cargo check -p tenferro-tensor --features cubecl
cargo test -p tenferro-tensor
```

Expected: pass.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/cubecl/kernels/mod.rs tenferro-tensor/src/cubecl/kernels/reduction.rs
git commit -m "refactor: remove tensor-local cubecl reduction kernels"
```

---

### Task 8: Run CUDA Reduction Verification

**Files:**

- No intended source edits unless tests fail.

**Step 1: Run ignored GPU tests**

Run on a CUDA 12 machine:

```bash
CUBECL_DEBUG_LOG=0 \
CUDA_PATH=/usr/local/cuda-12.0 \
LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH \
  cargo test -p tenferro-tensor --features cubecl reduction -- --ignored
```

Expected: pass, including:

- float sum/prod/max/min,
- complex sum/prod,
- complex max/min unsupported error behavior,
- i64 sum/prod,
- 3D column-major axis 0/1/2/multi-axis cases,
- final axis-removal shape.

**Step 2: If a CUDA test fails, debug systematically**

Use `superpowers:systematic-debugging`.

Collect:

- failing op,
- dtype,
- shape,
- axis,
- expected CPU output,
- actual GPU output,
- intermediate keepdims shape if multi-axis.

Likely failure classes:

- shape/stride mismatch in `typed_tensor_binding`,
- axis validation accepts `axis == rank` incorrectly,
- output keepdims shape not matching kernel expectation,
- multi-axis order bug,
- complex kernel trait mismatch,
- `i64` CubeCL trait bound mismatch.

**Step 3: Commit fixes if needed**

```bash
git add <changed files>
git commit -m "fix: correct cubecl reduction launch"
```

---

### Task 9: Final Workspace Verification

**Files:**

- No intended source edits unless verification reveals a real issue.

**Step 1: Run formatting**

```bash
cargo fmt --all --check
```

Expected: pass. If it fails, run `cargo fmt --all`, then rerun the check and commit formatting.

**Step 2: Run targeted non-GPU tests**

```bash
cargo test -p tenferro-cubecl
cargo test -p tenferro-cubecl --features cpu-reference
cargo test -p tenferro-tensor
cargo test -p tenferro-tensor --features cubecl reduction_tests --no-run
```

Expected: pass.

**Step 3: Run full PR checklist if preparing a PR**

```bash
cargo fmt --all --check
cargo test --workspace --release
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: pass. If full coverage is slow, at minimum record which commands were run and which remain.

**Step 4: Review the diff**

```bash
git status --short
git diff --stat HEAD~9..HEAD
git diff HEAD~9..HEAD -- Cargo.toml tenferro-tensor/Cargo.toml
```

Expected:

- `tenferro-cubecl` is a workspace member.
- `tenferro-tensor` depends on `tenferro-cubecl` only under the `cubecl` feature.
- no direct `cubek` dependency was added.
- old tensor-local reduction kernels are gone.
- attribution headers exist for adapted cubek logic.

**Step 5: Commit final verification-only fixes if needed**

```bash
git add <changed files>
git commit -m "chore: finalize cubecl reduction split"
```

---

## PR Notes

Suggested PR title:

```text
Split CubeCL reductions into tenferro-cubecl
```

Suggested PR body:

```markdown
## Summary

- Adds `tenferro-cubecl` as the dedicated CubeCL kernel crate.
- Moves GPU reduction launch/kernels behind the new crate boundary.
- Routes `CubeclBackend` reductions through `tenferro-cubecl`.
- Adds GPU `i64` `reduce_sum` and `reduce_prod` support.
- Keeps `i64` max/min and complex max/min unsupported.

## Testing

- `cargo fmt --all --check`
- `cargo test -p tenferro-cubecl`
- `cargo test -p tenferro-cubecl --features cpu-reference`
- `cargo test -p tenferro-tensor`
- `cargo test -p tenferro-tensor --features cubecl reduction_tests --no-run`
- CUDA ignored reduction tests: <pass/fail/not run with reason>

## Notes

- No direct `cubek` dependency is added because tenferro currently uses a CubeCL fork.
- Cubek-derived logic includes attribution headers with source commit/path/license.
- Linalg remains cuSOLVER/cuBLAS/cuTENSOR-backed in `tenferro-tensor`.
```
