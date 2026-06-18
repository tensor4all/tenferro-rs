# Context Engine Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the mixed `Engine<B>` traced-execution API with explicit graph compilation and backend execution, while making memory order and eager runtime naming explicit.

**Architecture:** Add explicit memory-order constructors/exports first so later call-site rewrites have stable targets. Then introduce `GraphCompiler`, `GraphProgram`, and `GraphExecutor<B>` beside the existing `Engine`, migrate traced/einsum/checkpoint paths, and remove the old public API in one breaking cleanup. Keep the existing `ExecProgram` and execution pipeline internally; the public split is an ownership/API refactor, not a lower-IR rewrite.

**Tech Stack:** Rust 2021, `tenferro` facade crate, `tenferro-tensor`, `tenferro-einsum`, `lru`, `computegraph`, existing integration tests, rustdoc/doctests, docs site scripts.

---

## Baseline

Worktree: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/context-engine-refactor`

Baseline already checked before writing this plan:

```bash
cargo build -p tenferro
cargo test -p tenferro --test engine_eval --test cache_management
```

Expected baseline: both commands pass. Observed baseline: build passed; `cache_management` passed 3 tests; `engine_eval` passed 1 test.

## Constraints

Relevant skills during execution: @superpowers:executing-plans, @superpowers:test-driven-development, @superpowers:systematic-debugging when tests fail, and @superpowers:verification-before-completion before claiming completion.

- Do not preserve public compatibility shims for `Engine`, `EagerContext`, ambiguous `from_vec`, or ambiguous `try_into_vec`.
- Keep `tenferro` facade tests in `tenferro/tests/*.rs`; the facade crate has `[lib] test = false`.
- Every new public type/function needs rustdoc with a compiling `# Examples` section.
- Do not add inline `#[cfg(test)]` blocks to normal modules.
- Keep cache ownership explicit, bounded, clearable, configurable, and introspectable.
- User-facing docs must import through `tenferro::{...}` and avoid internal jargon.

## Task 1: Add Explicit Memory-Order APIs to `tenferro-tensor`

**Files:**
- Modify: `tenferro-tensor/src/types.rs`
- Modify: `tenferro-tensor/src/tests/types_tests.rs`
- Optional modify: `tenferro-tensor/tests/typed_convenience.rs`

**Step 1: Write failing tensor memory-order tests**

Add tests to `tenferro-tensor/src/tests/types_tests.rs`:

```rust
#[test]
fn typed_tensor_explicit_memory_order_constructors_match_logical_matrix() {
    let row = TypedTensor::<f64>::from_vec_row_major(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    let col = TypedTensor::<f64>::from_vec_col_major(
        vec![2, 3],
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
    );

    assert_eq!(row.shape, vec![2, 3]);
    assert_eq!(row.as_slice(), col.as_slice());
    assert_eq!(row.get(&[0, 0]), &1.0);
    assert_eq!(row.get(&[1, 0]), &4.0);
    assert_eq!(row.get(&[0, 2]), &3.0);
    assert_eq!(row.get(&[1, 2]), &6.0);
}

#[test]
fn typed_tensor_explicit_memory_order_exports_requested_order() {
    let tensor = TypedTensor::<f64>::from_vec_col_major(
        vec![2, 3],
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
    );

    let (shape, col) = tensor.clone().into_vec_col_major().unwrap();
    assert_eq!(shape, vec![2, 3]);
    assert_eq!(col, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    let (shape, row) = tensor.into_vec_row_major().unwrap();
    assert_eq!(shape, vec![2, 3]);
    assert_eq!(row, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn tensor_explicit_memory_order_roundtrips_dynamic_dtype() {
    let tensor = Tensor::from_vec_row_major(
        vec![2, 2],
        vec![1_i64, 2, 3, 4],
    );

    assert_eq!(tensor.as_slice::<i64>().unwrap(), &[1, 3, 2, 4]);
    assert_eq!(
        tensor.clone().into_vec_col_major::<i64>().unwrap(),
        (vec![2, 2], vec![1, 3, 2, 4]),
    );
    assert_eq!(
        tensor.into_vec_row_major::<i64>().unwrap(),
        (vec![2, 2], vec![1, 2, 3, 4]),
    );
}
```

**Step 2: Run the tests and verify they fail**

Run:

```bash
cargo test -p tenferro-tensor explicit_memory_order
```

Expected: FAIL because `from_vec_row_major`, `from_vec_col_major`, `into_vec_row_major`, and `into_vec_col_major` do not exist.

**Step 3: Implement generic order conversion helpers**

In `tenferro-tensor/src/types.rs`, add private helpers near `linear_offset`:

```rust
fn checked_shape_len(shape: &[usize], data_len: usize, op: &str) {
    let n: usize = shape.iter().product();
    assert_eq!(
        data_len, n,
        "{op}: data length {} does not match shape product {}",
        data_len, n
    );
}

fn row_major_offset(shape: &[usize], indices: &[usize]) -> usize {
    let mut stride = 1;
    let mut offset = 0;
    for (&dim, &index) in shape.iter().rev().zip(indices.iter().rev()) {
        offset += index * stride;
        stride *= dim;
    }
    offset
}

fn for_each_index(shape: &[usize], mut f: impl FnMut(&[usize])) {
    if shape.is_empty() {
        f(&[]);
        return;
    }
    if shape.iter().any(|&dim| dim == 0) {
        return;
    }

    let mut index = vec![0; shape.len()];
    loop {
        f(&index);
        let mut axis = 0;
        loop {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
            axis += 1;
            if axis == shape.len() {
                return;
            }
        }
    }
}

fn row_major_to_col_major<T: Clone>(shape: &[usize], data: Vec<T>) -> Vec<T> {
    checked_shape_len(shape, data.len(), "from_vec_row_major");
    let mut out = Vec::with_capacity(data.len());
    for_each_index(shape, |index| {
        out.push(data[row_major_offset(shape, index)].clone());
    });
    out
}

fn col_major_to_row_major<T: Clone>(shape: &[usize], data: Vec<T>) -> Vec<T> {
    checked_shape_len(shape, data.len(), "into_vec_row_major");
    if shape.is_empty() {
        return data;
    }
    let mut indexed = Vec::with_capacity(data.len());
    for_each_index(shape, |index| {
        indexed.push((row_major_offset(shape, index), linear_offset(shape, index)));
    });
    indexed.sort_by_key(|&(row_offset, _)| row_offset);
    indexed
        .into_iter()
        .map(|(_, col_offset)| data[col_offset].clone())
        .collect()
}
```

If this helper duplicates existing shape iteration utilities, use the existing utility instead and keep this task's tests unchanged.

**Step 4: Add explicit constructors and exports**

In `impl<T: Clone> TypedTensor<T>`:

```rust
pub fn from_vec_col_major(shape: Vec<usize>, data: Vec<T>) -> Self {
    checked_shape_len(&shape, data.len(), "from_vec_col_major");
    Self {
        buffer: Buffer::Host(data),
        shape,
        placement: default_placement(),
    }
}

pub fn from_vec_row_major(shape: Vec<usize>, data: Vec<T>) -> Self {
    let data = row_major_to_col_major(&shape, data);
    Self::from_vec_col_major(shape, data)
}

pub fn into_vec_col_major(self) -> crate::Result<(Vec<usize>, Vec<T>)> {
    match self.buffer {
        Buffer::Host(data) => Ok((self.shape, data)),
        Buffer::Backend(_) => Err(crate::Error::BackendFailure {
            op: "into_vec_col_major",
            message: "backend buffers cannot be exported as host Vec".into(),
        }),
        #[cfg(feature = "cubecl")]
        Buffer::Cubecl(_) => Err(crate::Error::BackendFailure {
            op: "into_vec_col_major",
            message: "GPU buffers cannot be exported as host Vec".into(),
        }),
    }
}

pub fn into_vec_row_major(self) -> crate::Result<(Vec<usize>, Vec<T>)> {
    let (shape, data) = self.into_vec_col_major()?;
    let row_major = col_major_to_row_major(&shape, data);
    Ok((shape, row_major))
}
```

Keep `from_vec` and `try_into_vec` temporarily as deprecated wrappers so the workspace stays buildable during migration:

```rust
#[deprecated(note = "use from_vec_col_major or from_vec_row_major")]
pub fn from_vec(shape: Vec<usize>, data: Vec<T>) -> Self {
    Self::from_vec_col_major(shape, data)
}

#[deprecated(note = "use into_vec_col_major or into_vec_row_major")]
pub fn try_into_vec(self) -> crate::Result<(Vec<usize>, Vec<T>)> {
    self.into_vec_col_major()
}
```

In `impl Tensor`:

```rust
pub fn from_vec_col_major<T: TensorScalar>(shape: Vec<usize>, data: Vec<T>) -> Self {
    T::into_tensor(shape, data)
}

pub fn from_vec_row_major<T: TensorScalar>(shape: Vec<usize>, data: Vec<T>) -> Self {
    T::into_tensor(shape.clone(), row_major_to_col_major(&shape, data))
}

pub fn into_vec_col_major<T: TensorScalar>(self) -> crate::Result<(Vec<usize>, Vec<T>)> {
    let actual = self.dtype();
    let typed = T::try_into_typed(self).ok_or(crate::Error::DTypeMismatch {
        op: "into_vec_col_major",
        lhs: T::dtype(),
        rhs: actual,
    })?;
    typed.into_vec_col_major()
}

pub fn into_vec_row_major<T: TensorScalar>(self) -> crate::Result<(Vec<usize>, Vec<T>)> {
    let actual = self.dtype();
    let typed = T::try_into_typed(self).ok_or(crate::Error::DTypeMismatch {
        op: "into_vec_row_major",
        lhs: T::dtype(),
        rhs: actual,
    })?;
    typed.into_vec_row_major()
}
```

**Step 5: Run tests**

Run:

```bash
cargo test -p tenferro-tensor explicit_memory_order
```

Expected: PASS.

**Step 6: Commit**

```bash
git add tenferro-tensor/src/types.rs tenferro-tensor/src/tests/types_tests.rs
git commit -m "feat: add explicit tensor memory order APIs"
```

## Task 2: Add Explicit Memory-Order Constructors to `TracedTensor`

**Files:**
- Modify: `tenferro/src/traced.rs`
- Create: `tenferro/tests/memory_order_api.rs`

**Step 1: Write failing traced constructor tests**

Create `tenferro/tests/memory_order_api.rs`:

```rust
use tenferro::{Tensor, TracedTensor};

#[test]
fn traced_tensor_row_major_constructor_stores_column_major_input() {
    let traced = TracedTensor::from_vec_row_major(
        vec![2, 3],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    );

    let compiled = traced.compile_with_inputs(&[]).unwrap();
    assert_eq!(compiled.inputs.len(), 1);
    assert_eq!(compiled.inputs[0].shape(), &[2, 3]);
    assert_eq!(
        compiled.inputs[0].as_slice::<f64>().unwrap(),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
    );
}

#[test]
fn traced_tensor_col_major_constructor_keeps_physical_order() {
    let traced = TracedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1_i64, 3, 2, 4],
    );

    let compiled = traced.compile_with_inputs(&[]).unwrap();
    assert_eq!(
        compiled.inputs[0].clone().into_vec_col_major::<i64>().unwrap(),
        (vec![2, 2], vec![1, 3, 2, 4]),
    );
}

#[test]
fn tensor_row_major_constructor_remains_available_through_facade() {
    let tensor = Tensor::from_vec_row_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    assert_eq!(tensor.as_slice::<f64>().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
}
```

**Step 2: Run test and verify it fails**

Run:

```bash
cargo test -p tenferro --test memory_order_api
```

Expected: FAIL because `TracedTensor::from_vec_row_major` and `from_vec_col_major` do not exist.

**Step 3: Implement traced constructors**

In `tenferro/src/traced.rs`, replace the current `from_vec` implementation with explicit constructors:

```rust
pub fn from_vec_col_major<T: TensorScalar>(shape: Vec<usize>, data: Vec<T>) -> Self {
    Self::from_tensor_concrete_shape(Tensor::from_vec_col_major(shape, data))
}

pub fn from_vec_row_major<T: TensorScalar>(shape: Vec<usize>, data: Vec<T>) -> Self {
    Self::from_tensor_concrete_shape(Tensor::from_vec_row_major(shape, data))
}

#[deprecated(note = "use from_vec_col_major or from_vec_row_major")]
pub fn from_vec<T: TensorScalar>(shape: Vec<usize>, data: Vec<T>) -> Self {
    Self::from_vec_col_major(shape, data)
}
```

Keep the deprecated wrapper only until Task 9 removes old call sites and deletes wrappers.

**Step 4: Run test**

Run:

```bash
cargo test -p tenferro --test memory_order_api
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro/src/traced.rs tenferro/tests/memory_order_api.rs
git commit -m "feat: add explicit traced tensor memory order constructors"
```

## Task 3: Rename `EagerContext` to `EagerRuntime`

**Files:**
- Modify: `tenferro/src/eager.rs`
- Modify: `tenferro/src/eager_tensor.rs`
- Modify: `tenferro/src/lib.rs`
- Modify: `tenferro/src/eager_*.rs`
- Modify: `tenferro/src/shape_packing.rs`
- Modify: `tenferro/tests/*.rs`
- Modify: `tenferro/benches/*.rs`
- Create: `tenferro/tests/eager_runtime_api.rs`

**Step 1: Write failing public rename test**

Create `tenferro/tests/eager_runtime_api.rs`:

```rust
use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};

#[test]
fn eager_runtime_replaces_eager_context_public_name() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]),
        runtime.clone(),
    );
    let loss = (&x * &x).reduce_sum(&[0]).unwrap();

    loss.backward().unwrap();

    assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[2.0, 4.0]);
    runtime.clear_grads();
    assert!(x.grad().unwrap().is_none());
}
```

**Step 2: Run test and verify it fails**

Run:

```bash
cargo test -p tenferro --test eager_runtime_api
```

Expected: FAIL because `EagerRuntime` is not exported.

**Step 3: Rename implementation and exports**

In `tenferro/src/eager.rs`:

- Rename `pub struct EagerContext` to `pub struct EagerRuntime`.
- Rename `impl EagerContext` to `impl EagerRuntime`.
- Update rustdoc examples to use `EagerRuntime`.
- Keep return types as `Arc<Self>`.
- Do not add `pub type EagerContext = EagerRuntime`.

In `tenferro/src/lib.rs`:

```rust
pub use eager::{EagerRuntime, EagerTensor};
```

In `tenferro/src/eager_tensor.rs`:

```rust
pub use crate::eager::{EagerRuntime, EagerTensor};
```

Then mechanically update internal references:

```bash
rg -n "EagerContext" tenferro
```

Every source/test/bench reference under `tenferro/` should become `EagerRuntime`.

**Step 4: Run focused eager tests**

Run:

```bash
cargo test -p tenferro --test eager_runtime_api --test eager_tensor --test eager_exec --test eager_tensor_einsum
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro/src tenferro/tests tenferro/benches
git commit -m "refactor: rename eager context to eager runtime"
```

## Task 4: Introduce `GraphProgram` and `GraphCompiler`

**Files:**
- Create: `tenferro/src/graph/mod.rs`
- Create: `tenferro/src/graph/cache.rs`
- Create: `tenferro/src/graph/program.rs`
- Create: `tenferro/src/graph/compiler.rs`
- Modify: `tenferro/src/lib.rs`
- Modify: `tenferro/src/traced.rs`
- Create: `tenferro/tests/graph_compile.rs`

**Step 1: Write failing compiler tests**

Create `tenferro/tests/graph_compile.rs`:

```rust
use std::num::NonZeroUsize;

use tenferro::{DType, GraphCompiler, Tensor, TracedTensor};

#[test]
fn graph_compiler_compiles_without_backend() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let y = &x + &x;

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();

    assert_eq!(program.input_count(), 1);
    assert_eq!(program.output_count(), 1);
    assert_eq!(compiler.compile_cache_len(), 1);
}

#[test]
fn graph_compiler_validates_placeholder_specs() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let y = &x + &x;

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[3])])
        .unwrap();

    assert_eq!(program.input_count(), 1);
    assert_eq!(program.input_specs()[0].shape(), &[3]);

    let err = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F32, &[3])])
        .unwrap_err();
    assert!(format!("{err}").contains("dtype"));
}

#[test]
fn graph_compiler_cache_is_bounded_and_reports_stats() {
    let mut compiler = GraphCompiler::new();
    compiler.set_compile_cache_capacity(NonZeroUsize::new(1).unwrap());

    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let _ = compiler.compile(&(&x + &x)).unwrap();
    let _ = compiler.compile(&x.neg()).unwrap();

    let stats = compiler.cache_stats();
    assert_eq!(compiler.compile_cache_capacity().get(), 1);
    assert_eq!(stats.compile.entries, 1);
    assert!(stats.compile.retained_bytes > 0);
}
```

**Step 2: Run tests and verify they fail**

Run:

```bash
cargo test -p tenferro --test graph_compile
```

Expected: FAIL because graph types do not exist.

**Step 3: Add graph module exports**

In `tenferro/src/lib.rs`:

```rust
pub mod graph;
pub use graph::{GraphCompiler, GraphCompilerCacheStats, GraphProgram, GraphProgramInput};
```

Do not remove `Engine` yet; that happens after all call sites migrate.

**Step 4: Move cache helpers into `graph/cache.rs`**

Move or copy these existing pieces from `tenferro/src/engine.rs`:

- `ParsedEinsum`
- `EinsumCacheKey`
- `NaryEinsumCache`
- `EinsumParseCache`
- `DEFAULT_EINSUM_CACHE_CAPACITY`
- `DEFAULT_COMPILE_CACHE_CAPACITY`
- `CacheKey`
- `compute_cache_key`
- retained-byte helpers for `ExecProgram`, `ExecOp`, einsum subscripts, parse cache, n-ary cache

Rename stats types:

```rust
pub const DEFAULT_GRAPH_COMPILE_CACHE_CAPACITY: usize = 256;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphCompilerCacheStats {
    pub compile: CacheStats,
    pub static_einsum_plans: CacheStats,
    pub einsum_parse: CacheStats,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphExecutorCacheStats {
    pub runtime_einsum_plans: CacheStats,
    pub backend: CacheStats,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CpuGraphExecutorCacheStats {
    pub executor: GraphExecutorCacheStats,
    pub buffer_pool: CacheStats,
}
```

Keep cache helper functions `pub(crate)` unless they are used by public methods.

**Step 5: Implement `GraphProgram`**

In `tenferro/src/graph/program.rs`:

```rust
use std::sync::Arc;

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_tensor::{DType, Tensor};

use crate::exec::ExecProgram;

#[derive(Clone, Debug)]
pub struct GraphProgram {
    pub(crate) exec: ExecProgram,
    pub(crate) inputs: Vec<GraphProgramInput>,
}

#[derive(Clone, Debug)]
pub struct GraphProgramInput {
    pub(crate) key: TensorInputKey,
    pub(crate) dtype: DType,
    pub(crate) shape: Vec<usize>,
    pub(crate) dim_expr_shape: Vec<DimExpr>,
    pub(crate) default_tensor: Option<Arc<Tensor>>,
}

impl GraphProgram {
    pub(crate) fn new(exec: ExecProgram, inputs: Vec<GraphProgramInput>) -> Self {
        Self { exec, inputs }
    }

    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    pub fn output_count(&self) -> usize {
        self.exec.output_slots.len()
    }

    pub fn input_specs(&self) -> &[GraphProgramInput] {
        &self.inputs
    }
}

impl GraphProgramInput {
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}
```

Add rustdoc examples to all public types and methods before committing.

**Step 6: Implement `GraphCompiler`**

In `tenferro/src/graph/compiler.rs`, move compile logic out of `TracedTensor::compile_with_inputs` and `compile_with_input_specs`.

Core shape:

```rust
pub struct GraphCompiler {
    compile_cache: LruCache<CacheKey, ExecProgram>,
    static_einsum_cache: NaryEinsumCache,
    einsum_parse_cache: EinsumParseCache,
}
```

Methods:

```rust
pub fn new() -> Self;
pub fn compile(&mut self, output: &TracedTensor) -> Result<GraphProgram>;
pub fn compile_many(&mut self, outputs: &[&TracedTensor]) -> Result<GraphProgram>;
pub fn compile_with_input_specs(
    &mut self,
    output: &TracedTensor,
    bindings: &[(&TracedTensor, DType, &[usize])],
) -> Result<GraphProgram>;
pub fn compile_cache_len(&self) -> usize;
pub fn compile_cache_capacity(&self) -> NonZeroUsize;
pub fn set_compile_cache_capacity(&mut self, capacity: NonZeroUsize);
pub fn clear_compile_cache(&mut self);
pub fn clear_einsum_caches(&mut self);
pub fn clear_caches(&mut self);
pub fn cache_stats(&self) -> GraphCompilerCacheStats;
```

Implementation notes:

- Keep validation behavior identical to current `TracedTensor::compile_with_inputs` and `compile_with_input_specs`.
- `compile()` builds descriptors from attached `inputs_map` data and stores those tensors as `default_tensor`.
- `compile_with_input_specs()` records specs without default tensors for placeholders.
- The compile cache stores only `ExecProgram`; `GraphProgram` is rebuilt each call with fresh input descriptors/default tensors to avoid stale concrete input data.
- Use `compute_cache_key(&exec)` from `graph/cache.rs`.

**Step 7: Keep old hidden methods delegating temporarily**

In `tenferro/src/traced.rs`, keep `compile_with_inputs` and `compile_with_input_specs` temporarily for existing tests, but implement them by constructing a local `GraphCompiler`. They will be removed in Task 8.

**Step 8: Run tests**

Run:

```bash
cargo test -p tenferro --test graph_compile
```

Expected: PASS.

**Step 9: Commit**

```bash
git add tenferro/src/lib.rs tenferro/src/graph tenferro/src/traced.rs tenferro/tests/graph_compile.rs
git commit -m "feat: add graph compiler program API"
```

## Task 5: Introduce `GraphExecutor<B>`

**Files:**
- Create: `tenferro/src/graph/executor.rs`
- Modify: `tenferro/src/graph/mod.rs`
- Modify: `tenferro/src/lib.rs`
- Create: `tenferro/tests/graph_executor.rs`

**Step 1: Write failing executor tests**

Create `tenferro/tests/graph_executor.rs`:

```rust
use tenferro::{CpuBackend, DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};

#[test]
fn graph_executor_runs_compiled_single_output_program() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let y = &x + &x;

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let out = executor.run(&program).unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
}

#[test]
fn graph_executor_runs_compiled_multi_output_program() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let sum = &x + &x;
    let product = &x * &x;

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&[&sum, &product]).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let outputs = executor.run_many(&program).unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[2.0, 4.0]);
    assert_eq!(outputs[1].as_slice::<f64>().unwrap(), &[1.0, 4.0]);
}

#[test]
fn graph_executor_validates_runtime_bindings() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let y = &x + &x;

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[3])])
        .unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let ok = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
    let out = executor.run_with_inputs(&program, &[(&x, &ok)]).unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);

    let wrong_shape = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let err = executor
        .run_with_inputs(&program, &[(&x, &wrong_shape)])
        .unwrap_err();
    assert!(format!("{err}").contains("shape"));
}

#[test]
fn graph_executor_cache_stats_are_separate_from_compiler_stats() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let y = &x + &x;

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let _ = executor.run(&program).unwrap();

    assert!(compiler.cache_stats().compile.entries > 0);
    assert_eq!(executor.cache_stats().runtime_einsum_plans.entries, 0);
}
```

**Step 2: Run tests and verify they fail**

Run:

```bash
cargo test -p tenferro --test graph_executor
```

Expected: FAIL because `GraphExecutor` does not exist.

**Step 3: Implement executor**

In `tenferro/src/graph/executor.rs`:

```rust
pub struct GraphExecutor<B: TensorBackend> {
    backend: B,
    backend_cache: B::RuntimeCache,
    runtime_einsum_cache: NaryEinsumCache,
    slot_workspace: Vec<Option<Tensor>>,
}
```

Methods:

```rust
pub fn new(backend: B) -> Self;
pub fn backend(&self) -> &B;
pub fn run(&mut self, program: &GraphProgram) -> Result<Tensor>;
pub fn run_many(&mut self, program: &GraphProgram) -> Result<Vec<Tensor>>;
pub fn run_with_inputs(
    &mut self,
    program: &GraphProgram,
    bindings: &[(&TracedTensor, &Tensor)],
) -> Result<Tensor>;
pub fn run_many_with_inputs(
    &mut self,
    program: &GraphProgram,
    bindings: &[(&TracedTensor, &Tensor)],
) -> Result<Vec<Tensor>>;
pub fn eval_exec_ir(&mut self, program: &ExecProgram, inputs: Vec<Tensor>) -> Result<Vec<Tensor>>;
pub fn eval_exec_ir_non_consuming(&mut self, program: &ExecProgram, inputs: &[Tensor]) -> Result<Vec<Tensor>>;
pub fn clear_backend_cache(&mut self);
pub fn clear_einsum_caches(&mut self);
pub fn clear_caches(&mut self);
pub fn cache_stats(&self) -> GraphExecutorCacheStats;
```

Input resolution rules:

- `run()` uses `GraphProgramInput::default_tensor` for every program input.
- Missing default tensor returns `Error::UnboundPlaceholder`.
- `run_with_inputs()` validates placeholder key, dtype, and shape against `GraphProgramInput`.
- Duplicate bindings return `Error::DuplicateBinding`.
- Data-carrying leaves must not be rebound, preserving current `UnexpectedBinding` behavior.

Execution call:

```rust
crate::segment::eval_exec_segmented_with_cache_and_workspace(
    &mut self.backend,
    &program.exec,
    input_tensors,
    &mut self.runtime_einsum_cache,
    &mut self.slot_workspace,
    &mut self.backend_cache,
)
```

**Step 4: Add CPU-specific executor methods**

Add `impl GraphExecutor<CpuBackend>` with methods moved from `Engine<CpuBackend>`:

- `buffer_pool_len`
- `buffer_pool_stats`
- `reset_buffer_pool`
- `cpu_cache_stats`
- `clear_all_caches`
- `gemm_analysis_cache_capacity`
- `set_gemm_analysis_cache_capacity`
- `buffer_pool_limit_bytes`
- `set_buffer_pool_limit_bytes`

Use `CpuGraphExecutorCacheStats` for aggregate stats.

**Step 5: Export executor**

In `tenferro/src/graph/mod.rs` and `tenferro/src/lib.rs`, export:

```rust
pub use executor::{CpuGraphExecutorCacheStats, GraphExecutor, GraphExecutorCacheStats};
```

**Step 6: Run tests**

Run:

```bash
cargo test -p tenferro --test graph_executor
```

Expected: PASS.

**Step 7: Commit**

```bash
git add tenferro/src/graph tenferro/src/lib.rs tenferro/tests/graph_executor.rs
git commit -m "feat: add graph executor API"
```

## Task 6: Migrate Checkpoint and Traced Multi-Output Workflows

**Files:**
- Modify: `tenferro/src/traced.rs`
- Modify: `tenferro/tests/checkpoint.rs`
- Modify: `tenferro/tests/checkpoint_truncate_integration.rs`
- Modify: `tenferro/tests/graph_executor.rs`

**Step 1: Add checkpoint API test using compiler/executor**

Append to `tenferro/tests/graph_executor.rs`:

```rust
#[test]
fn checkpoint_uses_explicit_compiler_and_executor() {
    let x = TracedTensor::from_vec_col_major(vec![], vec![3.0_f64]);
    let mut y = &x * &x;

    let mut compiler = GraphCompiler::new();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    y.checkpoint(&mut compiler, &mut executor).unwrap();

    let program = compiler.compile(&y).unwrap();
    let out = executor.run(&program).unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[9.0]);
}
```

**Step 2: Run and verify failure**

Run:

```bash
cargo test -p tenferro --test graph_executor checkpoint_uses_explicit_compiler_and_executor
```

Expected: FAIL because `checkpoint` still takes `Engine`.

**Step 3: Change checkpoint signature**

In `tenferro/src/traced.rs`:

```rust
pub fn checkpoint<B: TensorBackend>(
    &mut self,
    compiler: &mut GraphCompiler,
    executor: &mut GraphExecutor<B>,
) -> Result<()> {
    let program = compiler.compile(self)?;
    let data = Arc::new(executor.run(&program)?);
    // Keep the existing graph replacement and checkpoint-chain logic.
}
```

Do not reintroduce `self.eval(...)`. The method is allowed to execute because
the caller passes explicit compiler and executor objects.

**Step 4: Update checkpoint tests**

In `tenferro/tests/checkpoint.rs` and `tenferro/tests/checkpoint_truncate_integration.rs`:

```rust
let mut compiler = GraphCompiler::new();
let mut executor = GraphExecutor::new(CpuBackend::new());
y.checkpoint(&mut compiler, &mut executor).unwrap();
```

For tests with repeated checkpoint calls, reuse the same compiler/executor to preserve cache behavior.

**Step 5: Run checkpoint tests**

Run:

```bash
cargo test -p tenferro --test checkpoint --test checkpoint_truncate_integration
```

Expected: PASS.

**Step 6: Commit**

```bash
git add tenferro/src/traced.rs tenferro/tests/checkpoint.rs tenferro/tests/checkpoint_truncate_integration.rs tenferro/tests/graph_executor.rs
git commit -m "refactor: checkpoint through graph compiler and executor"
```

## Task 7: Move Traced Einsum Caches to Compiler/Executor

**Files:**
- Modify: `tenferro/src/einsum.rs`
- Modify: `tenferro/src/traced_tensor.rs`
- Modify: `tenferro/src/graph/compiler.rs`
- Modify: `tenferro/src/graph/executor.rs`
- Modify: `tenferro/tests/nary_einsum_cache.rs`
- Modify: `tenferro/tests/cache_management.rs`

**Step 1: Add graph compiler einsum test**

Create or update `tenferro/tests/graph_einsum.rs`:

```rust
use tenferro::traced_tensor::einsum;
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

#[test]
fn traced_einsum_uses_compiler_for_static_graph_build_and_executor_for_run() {
    let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    let b = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);

    let mut compiler = GraphCompiler::new();
    let out = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();
    assert!(compiler.cache_stats().static_einsum_plans.entries > 0);
    assert!(compiler.cache_stats().einsum_parse.entries > 0);

    let program = compiler.compile(&out).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let tensor = executor.run(&program).unwrap();

    assert_eq!(tensor.shape(), &[2, 2]);
}

#[test]
fn symbolic_einsum_reuses_executor_runtime_plan_cache() {
    let a = TracedTensor::input_symbolic_shape(tenferro::DType::F64, 2);
    let b = TracedTensor::input_symbolic_shape(tenferro::DType::F64, 2);

    let mut compiler = GraphCompiler::new();
    let out = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();
    let program = compiler
        .compile_with_input_specs(&out, &[(&a, tenferro::DType::F64, &[2, 3]), (&b, tenferro::DType::F64, &[3, 2])])
        .unwrap();

    let lhs = tenferro::Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    let rhs = tenferro::Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let _ = executor.run_with_inputs(&program, &[(&a, &lhs), (&b, &rhs)]).unwrap();
    let before = executor.cache_stats().runtime_einsum_plans.entries;
    let _ = executor.run_with_inputs(&program, &[(&a, &lhs), (&b, &rhs)]).unwrap();
    let after = executor.cache_stats().runtime_einsum_plans.entries;

    assert_eq!(before, after);
    assert!(after > 0);
}
```

**Step 2: Run and verify failure**

Run:

```bash
cargo test -p tenferro --test graph_einsum
```

Expected: FAIL because `einsum` still takes `Engine`.

**Step 3: Change traced einsum signatures**

In `tenferro/src/einsum.rs`, change public traced builders:

```rust
pub fn einsum(
    compiler: &mut GraphCompiler,
    inputs: &[&TracedTensor],
    subscripts: &str,
) -> Result<TracedTensor>;

pub fn einsum_subscripts(
    compiler: &mut GraphCompiler,
    inputs: &[&TracedTensor],
    subscripts: &EinsumSubscripts,
) -> Result<TracedTensor>;

pub fn einsum_with(
    compiler: &mut GraphCompiler,
    inputs: &[&TracedTensor],
    subscripts: &str,
    optimize: EinsumOptimize,
) -> Result<TracedTensor>;

pub fn einsum_subscripts_with(
    compiler: &mut GraphCompiler,
    inputs: &[&TracedTensor],
    subscripts: &EinsumSubscripts,
    optimize: EinsumOptimize,
) -> Result<TracedTensor>;
```

Move `cached_subscripts` and static concrete-shape contraction tree caching onto `GraphCompiler` methods:

```rust
pub(crate) fn cached_subscripts(&mut self, subscripts: &str) -> Result<Arc<ParsedEinsum>>;
pub(crate) fn cached_static_einsum_tree(
    &mut self,
    key: EinsumCacheKey,
    build: impl FnOnce() -> Result<ContractionTree>,
) -> Result<Arc<ContractionTree>>;
```

**Step 4: Keep runtime symbolic cache executor-side**

Do not change `segment` / `exec` runtime `NaryEinsum` dispatch except to ensure it receives `GraphExecutor::runtime_einsum_cache`.

**Step 5: Update cache management tests**

In `tenferro/tests/cache_management.rs`, split compiler and executor expectations:

- compiler compile/static-einsum/parse caches clear through `GraphCompiler::clear_caches`.
- executor backend/runtime-einsum caches clear through `GraphExecutor::clear_caches`.
- CPU buffer pool clear through `GraphExecutor<CpuBackend>::clear_all_caches`.

**Step 6: Run tests**

Run:

```bash
cargo test -p tenferro --test graph_einsum --test cache_management --test nary_einsum_cache
```

Expected: PASS.

**Step 7: Commit**

```bash
git add tenferro/src/einsum.rs tenferro/src/traced_tensor.rs tenferro/src/graph tenferro/tests/graph_einsum.rs tenferro/tests/cache_management.rs tenferro/tests/nary_einsum_cache.rs
git commit -m "refactor: route traced einsum caches through graph runtime"
```

## Task 8: Remove `Engine` and Old Traced Eval API From Source Call Sites

**Files:**
- Modify: `tenferro/src/lib.rs`
- Modify: `tenferro/src/traced.rs`
- Modify: `tenferro/src/engine.rs` or delete after migration
- Modify: `tenferro/src/linalg_api.rs`
- Modify: `tenferro/src/shape_packing.rs`
- Modify: `tenferro/tests/*.rs`
- Modify: `tenferro/benches/*.rs`
- Modify: `tenferro-fft/src/lib.rs`
- Modify: `tenferro-fft/tests/*.rs`

**Step 1: Add contract test that old names are gone from facade exports**

Create `tenferro/tests/public_surface.rs`:

```rust
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

#[test]
fn traced_public_surface_uses_compiler_and_executor() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let y = &x + &x;

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let out = executor.run(&program).unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
}
```

**Step 2: Migrate source call sites**

Use `rg` to find old API usage:

```bash
rg -n "\bEngine\b|\.eval\(|eval_with_inputs|eval_all\(" tenferro tenferro-fft
```

Replace patterns:

```rust
let mut engine = Engine::new(CpuBackend::new());
let out = y.eval(&mut engine).unwrap();
```

with:

```rust
let mut compiler = GraphCompiler::new();
let program = compiler.compile(&y).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
let out = executor.run(&program).unwrap();
```

For repeated tests, add local helpers inside integration test files:

```rust
fn run(output: &TracedTensor) -> Tensor {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(output).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor.run(&program).unwrap()
}
```

For multi-output tests:

```rust
let program = compiler.compile_many(&[&u, &s, &vt]).unwrap();
let outputs = executor.run_many(&program).unwrap();
```

For placeholder tests:

```rust
let program = compiler
    .compile_with_input_specs(&y, &[(&x, DType::F64, &[3])])
    .unwrap();
let out = executor.run_with_inputs(&program, &[(&x, &binding)]).unwrap();
```

**Step 3: Remove old API**

In `tenferro/src/lib.rs`, remove:

```rust
pub mod engine;
pub use engine::Engine;
```

If cache code has been fully moved, delete `tenferro/src/engine.rs`. If any private helpers remain, move them into `tenferro/src/graph`.

In `tenferro/src/traced.rs`, remove:

- `pub fn eval`
- `pub fn eval_with_inputs`
- `pub fn compile_with_inputs`
- `pub fn compile_with_input_specs`
- free `pub fn eval_all`
- `CompiledTracedTensor` if no longer needed

Keep only graph-building and AD-transform APIs on `TracedTensor`.

**Step 4: Run source checks**

Run:

```bash
rg -n "\bEngine\b|\.eval\(|eval_with_inputs|eval_all\(" tenferro tenferro-fft
cargo test -p tenferro --test public_surface --test graph_compile --test graph_executor --test graph_einsum
```

Expected: `rg` returns no source/test usage except historical docs under `docs/plans`; tests pass.

**Step 5: Commit**

```bash
git add tenferro tenferro-fft
git commit -m "refactor: remove engine traced eval API"
```

## Task 9: Remove Ambiguous `from_vec` and `try_into_vec` Public APIs

**Files:**
- Modify: `tenferro-tensor/src/types.rs`
- Modify: all Rust call sites under `tenferro-*`, `tenferro`, and checked examples
- Modify: `tenferro/tests/memory_order_api.rs`
- Modify: `tenferro-tensor/src/tests/types_tests.rs`

**Step 1: Mechanically update call sites**

Find usages:

```bash
rg -n "from_vec\(|try_into_vec\(" tenferro tenferro-* --glob '*.rs'
```

Rules:

- Use `from_vec_col_major` when existing expected slices are already column-major physical buffers.
- Use `from_vec_row_major` only when the data is written in human row-major matrix order and expected logical values assume that.
- Use `into_vec_col_major` when tests assert physical buffer order.
- Use `into_vec_row_major` when docs/tests are demonstrating user-facing row-major export.

Most existing tests in this repository use column-major buffers; default to `_col_major` unless the test name or expected values clearly describe row-major import/export.

**Step 2: Remove deprecated wrappers**

In `tenferro-tensor/src/types.rs`, delete:

- `TypedTensor::from_vec`
- `TypedTensor::try_into_vec`
- `Tensor::from_vec`
- `Tensor::try_into_vec`

In `tenferro/src/traced.rs`, delete:

- `TracedTensor::from_vec`

**Step 3: Run checks and fix remaining usage**

Run:

```bash
rg -n "from_vec\(|try_into_vec\(" tenferro tenferro-* --glob '*.rs'
cargo check --workspace
cargo test -p tenferro-tensor types_tests
cargo test -p tenferro --test memory_order_api --test public_surface
```

Expected:

- `rg` returns no non-historical Rust source usage.
- `cargo check --workspace` passes.
- targeted tests pass.

**Step 4: Commit**

```bash
git add tenferro tenferro-* 
git commit -m "refactor: remove ambiguous vector memory order APIs"
```

## Task 10: Update User Docs, Rustdoc, Examples, and Architecture Docs

**Files:**
- Modify: `README.md`
- Modify: `tenferro/README.md`
- Modify: `tenferro/src/lib.rs`
- Modify: `tenferro/src/graph/*.rs`
- Modify: `tenferro/src/traced.rs`
- Modify: `tenferro/src/eager.rs`
- Modify: `tenferro/src/einsum.rs`
- Modify: `tenferro/examples/cpu_quickstart.rs`
- Modify: `tenferro/examples/cuda_quickstart.rs`
- Modify: `docs/getting-started/index.md`
- Modify: `docs/getting-started/core-concepts.md`
- Modify: `docs/getting-started/pytorch-jax-mapping.md`
- Modify: `docs/guides/choosing-an-api.md`
- Modify: `docs/guides/eager-operations.md`
- Modify: `docs/guides/memory-order.md`
- Modify: `docs/guides/devices-and-gpu.md`
- Modify: `docs/architecture/tenferro-crates.md`

**Step 1: Update docs around the new model**

Rewrite user-facing docs around:

1. Data model and memory order.
2. Immediate concrete execution with `Tensor` / `TypedTensor` + backend.
3. Eager scalar-loss AD with `EagerTensor` + `EagerRuntime`.
4. Traced graph workflow with `TracedTensor` + `GraphCompiler` + `GraphExecutor`.
5. CPU/CUDA backend and explicit transfer boundaries.

Do not present first-use docs primarily as "four tensor layers".

**Step 2: Replace old names in docs**

Run:

```bash
rg -n "\bEngine\b|EagerContext|from_vec\(|try_into_vec\(|\.eval\(" README.md tenferro docs --glob '*.md' --glob '*.rs'
```

Expected after editing:

- No user-facing docs mention `Engine`, `EagerContext`, ambiguous `from_vec`, ambiguous `try_into_vec`, or `.eval(&mut engine)`.
- Historical files under `docs/plans/` may still mention old APIs; do not edit them except the plan/design files for this work.

**Step 3: Keep examples executable**

Update `tenferro/examples/cpu_quickstart.rs` to use explicit constructors. If it demonstrates traced execution, use compiler/executor explicitly. If it remains concrete-only, keep it concrete-only but use `Tensor::from_vec_col_major` or `from_vec_row_major` intentionally.

Update `tenferro/examples/cuda_quickstart.rs` with explicit memory-order constructors and existing upload/download boundaries.

**Step 4: Run docs/example checks**

Run:

```bash
cargo test --doc -p tenferro-tensor -p tenferro
cargo check -p tenferro --examples
python3 scripts/check-docs-site.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add README.md tenferro docs
git commit -m "docs: update execution and memory order guides"
```

## Task 11: Workspace Verification and Cleanup

**Files:**
- Modify as needed based on verification failures.

**Step 1: Format**

Run:

```bash
cargo fmt --all
cargo fmt --all --check
```

Expected: PASS after formatting.

**Step 2: Run focused regression suite**

Run:

```bash
cargo test -p tenferro-tensor types_tests
cargo test -p tenferro --test graph_compile --test graph_executor --test graph_einsum --test public_surface --test memory_order_api --test eager_runtime_api --test cache_management --test checkpoint --test checkpoint_truncate_integration
```

Expected: PASS.

**Step 3: Run workspace release tests**

Run:

```bash
cargo test --workspace --release
```

If `nextest` is available and the team prefers the PR workflow command, run:

```bash
cargo nextest run --workspace --release --no-fail-fast
```

Expected: PASS. If a GPU-only ignored test requires CUDA, do not force it locally unless the machine is configured.

**Step 4: Run docs gates**

Run:

```bash
cargo test --doc --workspace --release
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: PASS.

**Step 5: Run coverage gate if time permits before PR**

Run:

```bash
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

Expected: PASS or actionable coverage failures. Add focused tests for modified files that fall below 90%.

**Step 6: Review public surface drift**

Run:

```bash
rg -n "\bEngine\b|EagerContext|from_vec\(|try_into_vec\(|\.eval\(" README.md tenferro docs --glob '*.md' --glob '*.rs'
git diff --stat
git diff --check
```

Expected:

- No stale public docs or rustdoc examples outside historical `docs/plans`.
- No whitespace errors.

**Step 7: Commit final fixes**

```bash
git status --short
git add <files fixed during verification>
git commit -m "test: verify graph execution refactor"
```

Only create this final commit if verification required additional changes.
