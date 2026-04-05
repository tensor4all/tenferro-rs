# IR Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the compiler pipeline for all remaining ops: ReduceProd/Max/Min, Concatenate, Reverse, Slice, Pad, Gather, Scatter. After this, every StdTensorOp variant lowers through StableHLO → ExecIR → eval without `todo!()`.

**Architecture:** Phase A (simple ops with existing kernels or trivial kernels) and Phase B (Gather/Scatter/Pad with StableHLO-compatible configs). All CPU kernels reference JAX's CPU implementation for semantics. Compiler lowering is 1:1 mapping for all ops.

**Tech Stack:** Rust, strided-kernel, JAX reference (`jax/_src/lax/slicing.py`, `jax/_src/lax/lax.py`)

**Reference:** JAX CPU implementations for Gather/Scatter/Slice/Pad semantics.

---

## Phase A: Simple Ops (kernel exists or trivial)

### Task 1: Wire ReduceProd/Max/Min through compiler

**Files:**
- Modify: `tenferro/src/compiler.rs` (lower_to_stablehlo + compile_to_exec)

These ops already have StableHloOp and ExecOp variants and working CPU kernels. Just add the lowering.

- [ ] In `lower_to_stablehlo`, add before the `_ => todo!()` fallback:
  ```rust
  StdTensorOp::ReduceProd { axes, .. } => StableHloOp::ReduceProd { axes: axes.clone() },
  StdTensorOp::ReduceMax { axes, .. } => StableHloOp::ReduceMax { axes: axes.clone() },
  StdTensorOp::ReduceMin { axes, .. } => StableHloOp::ReduceMin { axes: axes.clone() },
  ```
- [ ] In `compile_to_exec`, add before the `_ => todo!()` fallback:
  ```rust
  StableHloOp::ReduceProd { axes } => ExecOp::ReduceProd { axes: axes.clone() },
  StableHloOp::ReduceMax { axes } => ExecOp::ReduceMax { axes: axes.clone() },
  StableHloOp::ReduceMin { axes } => ExecOp::ReduceMin { axes: axes.clone() },
  ```
- [ ] `cargo test --workspace`
- [ ] Commit: `feat: wire ReduceProd/Max/Min through compiler pipeline`

### Task 2: Implement Slice CPU kernel + wire through compiler

**Files:**
- Modify: `tenferro-tensor/src/cpu/indexing.rs`
- Modify: `tenferro/src/compiler.rs`

SliceConfig already has `starts`, `limits`, `strides` fields.

- [ ] Implement `slice` in indexing.rs. JAX semantics: extract elements at `starts[d], starts[d]+strides[d], ...` up to `limits[d]` per dimension.
  ```rust
  pub fn slice(input: &Tensor, config: &SliceConfig) -> Tensor {
      dispatch_tensor!(input, t => typed_slice(t, config))
  }

  fn typed_slice<T: Copy + Clone + Zero>(input: &TypedTensor<T>, config: &SliceConfig) -> TypedTensor<T> {
      let rank = input.shape.len();
      let out_shape: Vec<usize> = (0..rank)
          .map(|d| (config.limits[d] - config.starts[d] + config.strides[d] - 1) / config.strides[d])
          .collect();
      let out_n: usize = out_shape.iter().product();
      let mut data = Vec::with_capacity(out_n);
      let mut out_idx = vec![0usize; rank];
      for flat in 0..out_n {
          flat_to_multi(flat, &out_shape, &mut out_idx);
          let in_idx: Vec<usize> = (0..rank)
              .map(|d| config.starts[d] + out_idx[d] * config.strides[d])
              .collect();
          data.push(*input.get(&in_idx));
      }
      TypedTensor::from_vec(out_shape, data)
  }
  ```
- [ ] Wire in compiler: `StdTensorOp::Slice(c) => StableHloOp::Slice(c.clone())` and same for compile_to_exec
- [ ] Add test in cpu_tests.rs: slice a [4,4] tensor with starts=[1,1], limits=[3,3], strides=[1,1] → [2,2]
- [ ] `cargo test --workspace`
- [ ] Commit: `feat: implement Slice CPU kernel + compiler wiring`

### Task 3: Implement Reverse CPU kernel + wire

**Files:**
- Modify: `tenferro-tensor/src/cpu/indexing.rs`
- Modify: `tenferro/src/compiler.rs`

- [ ] Implement `reverse`: flip elements along specified axes.
  ```rust
  fn typed_reverse<T: Copy + Clone + Zero>(input: &TypedTensor<T>, axes: &[usize]) -> TypedTensor<T> {
      let n = input.n_elements();
      let mut data = Vec::with_capacity(n);
      let mut in_idx = vec![0usize; input.shape.len()];
      let mut rev_idx = vec![0usize; input.shape.len()];
      for flat in 0..n {
          flat_to_multi(flat, &input.shape, &mut in_idx);
          for d in 0..input.shape.len() {
              rev_idx[d] = if axes.contains(&d) {
                  input.shape[d] - 1 - in_idx[d]
              } else {
                  in_idx[d]
              };
          }
          data.push(*input.get(&rev_idx));
      }
      TypedTensor::from_vec(input.shape.clone(), data)
  }
  ```
- [ ] Wire in compiler
- [ ] Add test: reverse [1,2,3,4,5,6] shape [2,3] along axis 0
- [ ] `cargo test --workspace`
- [ ] Commit: `feat: implement Reverse CPU kernel + compiler wiring`

### Task 4: Implement Concatenate CPU kernel + wire

**Files:**
- Modify: `tenferro-tensor/src/cpu/indexing.rs`
- Modify: `tenferro/src/compiler.rs`

- [ ] Implement `concatenate`: join tensors along given axis.
  Iterate output elements, determine which input and which offset within it.
- [ ] Wire in compiler. Note: Concatenate has variable inputs. In `lower_to_stablehlo`, `StdTensorOp::Concatenate { axis }` → `StableHloOp::Concatenate { axis }`.
- [ ] Add test: concatenate two [2,3] tensors along axis 0 → [4,3]
- [ ] `cargo test --workspace`
- [ ] Commit: `feat: implement Concatenate CPU kernel + compiler wiring`

---

## Phase B: Complex Indexing Ops (config design needed)

### Task 5: Define GatherConfig with StableHLO semantics

**Files:**
- Modify: `tenferro-tensor/src/config.rs`

Reference: StableHLO gather spec and JAX's `_gather_impl` in `jax/_src/lax/slicing.py`.

- [ ] Replace empty GatherConfig:
  ```rust
  #[derive(Clone, Debug, Hash, PartialEq, Eq)]
  pub struct GatherConfig {
      pub offset_dims: Vec<usize>,
      pub collapsed_slice_dims: Vec<usize>,
      pub start_index_map: Vec<usize>,
      pub index_vector_dim: usize,
      pub slice_sizes: Vec<usize>,
  }
  ```
- [ ] `cargo check --workspace` (will break any code constructing empty GatherConfig — fix call sites)
- [ ] Commit: `refactor: define GatherConfig with StableHLO dimension numbers`

### Task 6: Implement Gather CPU kernel

**Files:**
- Modify: `tenferro-tensor/src/cpu/indexing.rs`

Reference JAX's gather implementation for semantics. The key algorithm:
1. Iterate over all output indices
2. Decompose output index into batch_dims and offset_dims
3. Look up start indices from the index tensor
4. Compute source index = start_index + offset
5. Read from input

- [ ] Implement `typed_gather` following StableHLO semantics
- [ ] Wire in compiler (lower_to_stablehlo + compile_to_exec)
- [ ] Add test: diagonal extraction via Gather (config for extracting diagonal of 3x3 matrix)
- [ ] `cargo test --workspace`
- [ ] Commit: `feat: implement Gather CPU kernel with StableHLO semantics`

### Task 7: Define ScatterConfig + implement Scatter CPU kernel

**Files:**
- Modify: `tenferro-tensor/src/config.rs`
- Modify: `tenferro-tensor/src/cpu/indexing.rs`
- Modify: `tenferro/src/compiler.rs`

- [ ] Replace empty ScatterConfig:
  ```rust
  #[derive(Clone, Debug, Hash, PartialEq, Eq)]
  pub struct ScatterConfig {
      pub update_window_dims: Vec<usize>,
      pub inserted_window_dims: Vec<usize>,
      pub scatter_dims_to_operand_dims: Vec<usize>,
      pub index_vector_dim: usize,
  }
  ```
- [ ] Implement `typed_scatter`: inverse of gather. Initialize output to zeros, accumulate updates at scatter indices with `+`.
- [ ] Wire in compiler
- [ ] Add test: diagonal embedding via Scatter
- [ ] `cargo test --workspace`
- [ ] Commit: `feat: implement Scatter CPU kernel with StableHLO semantics`

### Task 8: Define PadConfig + implement Pad CPU kernel

**Files:**
- Modify: `tenferro-tensor/src/config.rs`
- Modify: `tenferro-tensor/src/cpu/indexing.rs`
- Modify: `tenferro/src/compiler.rs`

- [ ] Replace empty PadConfig:
  ```rust
  #[derive(Clone, Debug, Hash, PartialEq, Eq)]
  pub struct PadConfig {
      pub edge_padding_low: Vec<i64>,
      pub edge_padding_high: Vec<i64>,
      pub interior_padding: Vec<i64>,
      pub padding_value: f64,
  }
  ```
- [ ] Implement `typed_pad`: surround tensor with padding value
- [ ] Wire in compiler
- [ ] Add test: pad [2,3] with 1 on each side → [4,5]
- [ ] `cargo test --workspace`
- [ ] Commit: `feat: implement Pad CPU kernel with StableHLO semantics`

### Task 9: Implement DynamicSlice CPU kernel + wire

**Files:**
- Modify: `tenferro-tensor/src/cpu/indexing.rs`
- Modify: `tenferro/src/compiler.rs`

- [ ] Implement: like Slice but start indices come from a tensor at runtime
- [ ] Wire in compiler
- [ ] `cargo test --workspace`
- [ ] Commit: `feat: implement DynamicSlice CPU kernel + compiler wiring`

---

## Phase C: Verification

### Task 10: Verify no todo!() in compiler pipeline

- [ ] `grep -n 'todo!' tenferro/src/compiler.rs` — only the final `_ =>` fallback should remain
- [ ] `grep -n 'todo!' tenferro-tensor/src/cpu/indexing.rs` — should be empty
- [ ] `cargo fmt --all --check`
- [ ] `cargo test --workspace`
- [ ] Commit if needed
