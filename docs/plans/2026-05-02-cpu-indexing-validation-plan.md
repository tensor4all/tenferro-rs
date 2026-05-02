# CPU Indexing Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make CPU indexing return explicit `Result` errors for invalid inputs
and support I64 data tensors for CPU reverse.

**Architecture:** Convert CPU indexing entry points from panic-prone helpers to
`Result<Tensor>` helpers, keep dtype dispatch explicit at the indexing boundary,
and validate index tensors before typed execution. Backend and exec-session
callers should use the Result-returning helpers directly instead of relying on
`catch_backend_panic`.

**Tech Stack:** Rust, `tenferro-tensor`, column-major `TypedTensor`, existing
`crate::Error` variants, existing `src/tests/cpu_tests.rs` test module.

**Execution Model:** Codex dispatches this plan to
`opencode run -m deepseek/deepseek-v4-pro`. The implementation must stay within
this plan and the companion spec
`docs/plans/2026-05-02-cpu-indexing-validation-design.md`.

---

### Task 1: Add CPU Indexing Regression Tests

**Files:**

- Modify: `tenferro-tensor/src/tests/cpu_tests.rs`

**Step 1: Add regression tests near the existing indexing tests**

Add tests with these names and behaviors:

```rust
#[test]
fn test_reverse_accepts_i64_data_tensor() {
    let input = Tensor::from_vec(vec![3], vec![1_i64, 2, 3]);
    let mut backend = CpuBackend::new();

    let out = backend.reverse(&input, &[0]).unwrap();

    assert_eq!(out.dtype(), DType::I64);
    assert_eq!(out.shape(), &[3]);
    assert_eq!(out.as_slice::<i64>(), Some([3, 2, 1].as_slice()));
}

#[test]
fn test_reverse_axis_out_of_bounds_returns_error() {
    let input = Tensor::F64(TypedTensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]));
    let mut backend = CpuBackend::new();

    let err = backend.reverse(&input, &[1]).unwrap_err();

    assert!(matches!(
        err,
        crate::Error::AxisOutOfBounds {
            op: "reverse",
            axis: 1,
            rank: 1,
        }
    ));
}

#[test]
fn test_gather_rejects_fractional_float_indices() {
    let operand = Tensor::F64(TypedTensor::from_vec(
        vec![5],
        vec![10.0, 20.0, 30.0, 40.0, 50.0],
    ));
    let start_indices = Tensor::F64(TypedTensor::from_vec(vec![1, 1], vec![1.5]));
    let mut backend = CpuBackend::new();

    let err = backend
        .gather(&operand, &start_indices, &simple_gather_config())
        .unwrap_err();

    assert!(matches!(err, crate::Error::InvalidConfig { op: "index_tensor", .. }));
}

#[test]
fn test_gather_rejects_complex_indices() {
    let operand = Tensor::F64(TypedTensor::from_vec(
        vec![5],
        vec![10.0, 20.0, 30.0, 40.0, 50.0],
    ));
    let start_indices = Tensor::C64(TypedTensor::from_vec(
        vec![1, 1],
        vec![Complex64::new(1.0, 0.0)],
    ));
    let mut backend = CpuBackend::new();

    let err = backend
        .gather(&operand, &start_indices, &simple_gather_config())
        .unwrap_err();

    assert!(matches!(err, crate::Error::InvalidConfig { op: "index_tensor", .. }));
}

#[test]
fn test_dynamic_slice_rejects_oversized_window() {
    let input = Tensor::F64(TypedTensor::from_vec(
        vec![2],
        vec![1.0, 2.0],
    ));
    let starts = Tensor::from_vec(vec![1], vec![0_i64]);
    let mut backend = CpuBackend::new();

    let err = backend.dynamic_slice(&input, &starts, &[3]).unwrap_err();

    assert!(matches!(err, crate::Error::InvalidConfig { op: "dynamic_slice", .. }));
}
```

If `Complex64` is not already imported at the top of `cpu_tests.rs`, add it to
the existing imports.

**Step 2: Run tests to verify current failures**

Run:

```bash
cargo test -p tenferro-tensor test_reverse_accepts_i64_data_tensor
cargo test -p tenferro-tensor test_gather_rejects_fractional_float_indices
cargo test -p tenferro-tensor test_gather_rejects_complex_indices
cargo test -p tenferro-tensor test_dynamic_slice_rejects_oversized_window
```

Expected: at least the I64 reverse test fails or errors today, and the error
tests expose panic-catcher or wrong-error behavior.

**Step 3: Commit after the implementation passes**

Do not commit failing tests alone unless the user explicitly asks for a TDD
checkpoint commit. These tests should be committed together with the
implementation task that makes them pass.

### Task 2: Convert CPU Indexing Entry Points To Result

**Files:**

- Modify: `tenferro-tensor/src/cpu/indexing.rs:57-142`
- Modify: `tenferro-tensor/src/cpu/backend.rs:358-411`
- Modify: `tenferro-tensor/src/cpu/exec_session.rs:208-245`
- Check: `tenferro-tensor/src/cpu/mod.rs:23`

**Step 1: Change public CPU indexing helper signatures**

Change these entry points to return `crate::Result<Tensor>`:

```rust
pub fn gather(
    operand: &Tensor,
    start_indices: &Tensor,
    config: &GatherConfig,
) -> crate::Result<Tensor>;

pub fn scatter(
    operand: &Tensor,
    scatter_indices: &Tensor,
    updates: &Tensor,
    config: &ScatterConfig,
) -> crate::Result<Tensor>;

pub fn slice(input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor>;

pub fn dynamic_slice(
    input: &Tensor,
    starts: &Tensor,
    slice_sizes: &[usize],
) -> crate::Result<Tensor>;

pub fn pad(input: &Tensor, config: &PadConfig) -> crate::Result<Tensor>;

pub fn reverse(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
```

Keep `try_slice` and `try_pad` during the transition if existing call sites use
them, but make `slice` and `pad` delegate without `.expect(...)`:

```rust
pub fn slice(input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor> {
    try_slice(input, config)
}

pub fn pad(input: &Tensor, config: &PadConfig) -> crate::Result<Tensor> {
    try_pad(input, config)
}
```

**Step 2: Replace macro dispatch in indexing entry points**

Do not use `dispatch_tensor!` or `dispatch_binary!` from indexing. Use explicit
matches so unsupported I64 data tensors return errors except for reverse:

```rust
pub fn reverse(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(t) => typed_reverse(t, axes).map(Tensor::F32),
        Tensor::F64(t) => typed_reverse(t, axes).map(Tensor::F64),
        Tensor::I64(t) => typed_reverse(t, axes).map(Tensor::I64),
        Tensor::C32(t) => typed_reverse(t, axes).map(Tensor::C32),
        Tensor::C64(t) => typed_reverse(t, axes).map(Tensor::C64),
    }
}
```

For `gather` and `dynamic_slice`, reject `Tensor::I64` data operands with:

```rust
Err(crate::Error::BackendFailure {
    op: "gather",
    message: "I64 data tensors are not supported by this operation".into(),
})
```

Use `op: "dynamic_slice"` for dynamic slice. For `scatter`, return
`DTypeMismatch` when operand and updates dtypes differ, and return
`BackendFailure { op: "scatter", .. }` when both are I64.

**Step 3: Update backend callers**

In `tenferro-tensor/src/cpu/backend.rs`, replace panic-catching wrappers for
these operations with direct calls:

```rust
self.install(|| indexing::gather(operand, start_indices, config))
self.install(|| indexing::scatter(operand, scatter_indices, updates, config))
self.install(|| indexing::dynamic_slice(input, starts, slice_sizes))
self.install(|| indexing::reverse(input, axes))
```

Do the same in `tenferro-tensor/src/cpu/exec_session.rs`.

**Step 4: Run a compile check**

Run:

```bash
cargo test -p tenferro-tensor test_reverse_accepts_i64_data_tensor
```

Expected: compilation reaches the implementation. Fix all call sites that still
expect non-Result `gather`, `scatter`, `slice`, `dynamic_slice`, `pad`, or
`reverse`.

### Task 3: Add Lossless Index Tensor Conversion

**Files:**

- Modify: `tenferro-tensor/src/cpu/indexing.rs:368-383`

**Step 1: Replace `index_tensor` with `try_index_tensor`**

Change:

```rust
fn index_tensor(tensor: &Tensor) -> IndexTensor
```

to:

```rust
fn try_index_tensor(tensor: &Tensor) -> crate::Result<IndexTensor>
```

Implement lossless conversion:

```rust
const F32_MAX_EXACT_INT: f32 = 16_777_216.0; // 2^24
const F64_MAX_EXACT_INT: f64 = 9_007_199_254_740_992.0; // 2^53

fn f32_index_to_i64(value: f32) -> crate::Result<i64> {
    if !value.is_finite() || value.fract() != 0.0 || value.abs() > F32_MAX_EXACT_INT {
        return Err(crate::Error::InvalidConfig {
            op: "index_tensor",
            message: format!("index value {value} is not an exactly representable i64"),
        });
    }
    Ok(value as i64)
}

fn f64_index_to_i64(value: f64) -> crate::Result<i64> {
    if !value.is_finite() || value.fract() != 0.0 || value.abs() > F64_MAX_EXACT_INT {
        return Err(crate::Error::InvalidConfig {
            op: "index_tensor",
            message: format!("index value {value} is not an exactly representable i64"),
        });
    }
    Ok(value as i64)
}
```

For complex index tensors, return:

```rust
Err(crate::Error::InvalidConfig {
    op: "index_tensor",
    message: "complex index tensors are not supported".into(),
})
```

**Step 2: Use the Result in entry points**

Use:

```rust
let start_indices = try_index_tensor(start_indices)?;
let scatter_indices = try_index_tensor(scatter_indices)?;
let starts = try_index_tensor(starts)?;
```

**Step 3: Run focused tests**

Run:

```bash
cargo test -p tenferro-tensor test_gather_accepts_i64_indices
cargo test -p tenferro-tensor test_gather_rejects_fractional_float_indices
cargo test -p tenferro-tensor test_gather_rejects_complex_indices
```

Expected: all pass.

### Task 4: Convert Typed Indexing Validation From Assertions To Result

**Files:**

- Modify: `tenferro-tensor/src/cpu/indexing.rs:386-689`

**Step 1: Add Result-returning validation helpers**

Replace panic-prone helpers with Result-returning forms:

```rust
fn try_index_vector_size(
    op: &'static str,
    shape: &[usize],
    index_vector_dim: usize,
) -> crate::Result<usize> {
    if index_vector_dim > shape.len() {
        return Err(crate::Error::AxisOutOfBounds {
            op,
            axis: index_vector_dim,
            rank: shape.len(),
        });
    }
    Ok(if index_vector_dim == shape.len() {
        1
    } else {
        shape[index_vector_dim]
    })
}

fn try_index_batch_shape(
    op: &'static str,
    shape: &[usize],
    index_vector_dim: usize,
) -> crate::Result<Vec<usize>> {
    if index_vector_dim > shape.len() {
        return Err(crate::Error::AxisOutOfBounds {
            op,
            axis: index_vector_dim,
            rank: shape.len(),
        });
    }
    if index_vector_dim == shape.len() {
        return Ok(shape.to_vec());
    }
    Ok(shape
        .iter()
        .enumerate()
        .filter_map(|(axis, &dim)| (axis != index_vector_dim).then_some(dim))
        .collect())
}

fn clamp_window_start(
    op: &'static str,
    start: i64,
    dim_size: usize,
    window_size: usize,
) -> crate::Result<usize> {
    if window_size > dim_size {
        return Err(crate::Error::InvalidConfig {
            op,
            message: format!("window size {window_size} exceeds dimension size {dim_size}"),
        });
    }
    let max_start = dim_size.saturating_sub(window_size) as i64;
    Ok(start.clamp(0, max_start) as usize)
}
```

Keep helper names if preferred, but all user-controlled invalid cases must
return `Err`.

**Step 2: Change typed helpers to return Result**

Change signatures:

```rust
fn typed_reverse<T: Copy + Clone>(
    input: &TypedTensor<T>,
    axes: &[usize],
) -> crate::Result<TypedTensor<T>>;

fn typed_gather<T: Copy + Clone + Zero>(
    operand: &TypedTensor<T>,
    start_indices: &IndexTensor,
    config: &GatherConfig,
) -> crate::Result<TypedTensor<T>>;

fn typed_scatter<T>(
    operand: &TypedTensor<T>,
    scatter_indices: &IndexTensor,
    updates: &TypedTensor<T>,
    config: &ScatterConfig,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Add<Output = T>;

fn typed_dynamic_slice<T: Copy + Clone + Zero>(
    input: &TypedTensor<T>,
    starts: &IndexTensor,
    slice_sizes: &[usize],
) -> crate::Result<TypedTensor<T>>;
```

Replace every `assert!` and `assert_eq!` in these helpers with
`RankMismatch`, `AxisOutOfBounds`, or `InvalidConfig` as appropriate.

**Step 3: Protect internal index component access**

`index_component` currently asserts for implicit index-vector shape. Convert it
to `crate::Result<i64>` and propagate `?` through gather/scatter loops.

**Step 4: Run focused tests**

Run:

```bash
cargo test -p tenferro-tensor test_reverse_axis_out_of_bounds_returns_error
cargo test -p tenferro-tensor test_dynamic_slice_rejects_oversized_window
cargo test -p tenferro-tensor test_backend_gather_scatter_dynamic_slice_dispatch
```

Expected: all pass.

### Task 5: Update Existing Tests For Result Helpers

**Files:**

- Modify: `tenferro-tensor/src/tests/cpu_tests.rs`

**Step 1: Update direct helper call sites**

Existing direct calls to `gather`, `scatter`, `dynamic_slice`, and `pad` in
`cpu_tests.rs` now return `Result<Tensor>`. Add `.unwrap()` only in tests where
success is expected:

```rust
let out = gather(&operand, &start_indices, &simple_gather_config()).unwrap();
let out = scatter(&operand, &scatter_indices, &updates, &config).unwrap();
let out = dynamic_slice(&input, &starts, &[2, 2]).unwrap();
let out = pad(&input, &config).unwrap();
```

Do not add unwraps in production code.

**Step 2: Run the CPU test slice**

Run:

```bash
cargo test -p tenferro-tensor cpu_tests
```

Expected: CPU tests compile and pass.

### Task 6: Final Verification And Commit

**Files:**

- Review: `tenferro-tensor/src/cpu/indexing.rs`
- Review: `tenferro-tensor/src/cpu/backend.rs`
- Review: `tenferro-tensor/src/cpu/exec_session.rs`
- Review: `tenferro-tensor/src/tests/cpu_tests.rs`

**Step 1: Search for remaining panic-prone indexing paths**

Run:

```bash
rg -n "panic!|assert!|assert_eq!|expect\\(|unwrap\\(" tenferro-tensor/src/cpu/indexing.rs
```

Expected: no `panic!`, `.expect(...)`, `.unwrap(...)`, or user-controlled
`assert!`/`assert_eq!` remains in CPU indexing. A non-user-controlled internal
assert requires a comment explaining why it is unreachable.

**Step 2: Run focused verification**

Run:

```bash
cargo test -p tenferro-tensor test_reverse_accepts_i64_data_tensor
cargo test -p tenferro-tensor test_reverse_axis_out_of_bounds_returns_error
cargo test -p tenferro-tensor test_gather_rejects_fractional_float_indices
cargo test -p tenferro-tensor test_gather_rejects_complex_indices
cargo test -p tenferro-tensor test_dynamic_slice_rejects_oversized_window
cargo test -p tenferro-tensor cpu_tests
cargo fmt --all --check
```

Expected: all pass.

**Step 3: Commit**

Run:

```bash
git add tenferro-tensor/src/cpu/indexing.rs \
  tenferro-tensor/src/cpu/backend.rs \
  tenferro-tensor/src/cpu/exec_session.rs \
  tenferro-tensor/src/tests/cpu_tests.rs
git commit -m "fix: return errors from cpu indexing validation"
```

**Step 4: Report**

Report:

- issues addressed,
- files changed,
- focused tests run,
- any deferred parts of #767, #804, or #814.
