# Einsum Performance Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce tenferro-einsum overhead from 2.5x to <1.5x vs strided-rs by eliminating unnecessary copies, conversions, and HashMap lookups.

**Architecture:** Optimize existing lazy permute + bgemm pipeline. bgemm internally handles stride compatibility (faer: direct, lapack: conditional contiguous). Use Tensor directly without StridedView conversion. Typed buffer pool instead of TypeId/Any. Pre-computed Vec-indexed plans instead of HashMap lookup.

**Tech Stack:** Rust, faer (strided GEMM), BTreeMap (typed pool), existing TensorPrims trait

---

## Task 1: Add `is_gemm_compatible` Helper Function

**Files:**
- Modify: `tenferro-prims/src/cpu.rs`

**Step 1: Add helper function**

Add after line 42 (after `tensor_to_view_mut`):

```rust
fn is_gemm_compatible<T>(tensor: &Tensor<T>, batch_dims: &[usize], m: usize, n: usize, is_a: bool) -> bool
where
    T: Scalar,
{
    let strides = tensor.strides();
    let rank = tensor.rank();
    
    if rank < 2 {
        return false;
    }
    
    let batch_rank = rank - 2;
    
    // Check batch dimensions are contiguous
    for i in 0..batch_rank {
        let expected_stride = if i == 0 {
            batch_dims.iter().skip(i + 1).product::<usize>() * m * n
        } else {
            let prev_stride = strides[i - 1] as usize;
            if prev_stride == 0 {
                return false;
            }
            strides[i] as usize == batch_dims.iter().skip(i + 1).product::<usize>() * m * n
        };
        if i == 0 && strides[i] as usize != batch_dims.iter().skip(1).product::<usize>() * m * n {
            return false;
        }
        if i > 0 && strides[i] as usize != batch_dims.iter().skip(i + 1).product::<usize>() * m * n {
            return false;
        }
    }
    
    // Check matrix dimensions
    let m_stride = strides[batch_rank] as usize;
    let k_or_n_stride = strides[batch_rank + 1] as usize;
    
    if is_a {
        // A: [batch..., m, k] - row-major preferred
        m_stride >= n && k_or_n_stride == 1
    } else {
        // B: [batch..., k, n] - row-major preferred
        m_stride >= m && k_or_n_stride == 1
    }
}
```

**Step 2: Run tests**

Run: `cargo test -p tenferro-prims`
Expected: PASS (no behavior change yet)

**Step 3: Commit**

```bash
git add tenferro-prims/src/cpu.rs
git commit -m "feat(prims): add is_gemm_compatible helper for stride check"
```

---

## Task 2: Add `maybe_make_contiguous` Function

**Files:**
- Modify: `tenferro-prims/src/cpu.rs`

**Step 1: Add function**

Add after `is_gemm_compatible`:

```rust
fn maybe_make_contiguous<T>(
    ctx: &mut CpuContext,
    tensor: &Tensor<T>,
    batch_dims: &[usize],
    m: usize,
    n: usize,
    is_a: bool,
) -> Result<Tensor<T>>
where
    T: Scalar + 'static,
{
    if is_gemm_compatible(tensor, batch_dims, m, n, is_a) {
        return Ok(tensor.clone());
    }
    
    let mut contiguous = Tensor::zeros(tensor.dims());
    let desc = PrimDescriptor::MakeContiguous;
    let shapes: Vec<&[usize]> = vec![tensor.dims()];
    let plan = CpuBackend::plan(ctx, &desc, &shapes)?;
    
    let input_views: Vec<StridedView<T>> = vec![tensor_to_view(tensor)?];
    let mut output_view = tensor_to_view_mut(&mut contiguous)?;
    
    CpuBackend::execute(ctx, &plan, &input_views.iter().collect::<Vec<_>>(), &mut output_view)?;
    
    Ok(contiguous)
}
```

**Step 2: Run tests**

Run: `cargo test -p tenferro-prims`
Expected: PASS

**Step 3: Commit**

```bash
git add tenferro-prims/src/cpu.rs
git commit -m "feat(prims): add maybe_make_contiguous for conditional copy"
```

---

## Task 3: Add Strided GEMM for faer Backend

**Files:**
- Modify: `tenferro-prims/src/cpu.rs`

**Step 1: Add `execute_bgemm_strided_faer` function**

Add in the GEMM section (around line 1186, within `#[cfg(feature = "gemm-faer")]` block):

```rust
#[cfg(feature = "gemm-faer")]
fn execute_bgemm_strided_faer<T>(
    alpha: T,
    a: &Tensor<T>,
    b: &Tensor<T>,
    beta: T,
    c: &mut Tensor<T>,
    batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()>
where
    T: Scalar + faer::Entity,
{
    let a_data = a.buffer().as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor not supported".into()))?;
    let b_data = b.buffer().as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor not supported".into()))?;
    let c_data = c.buffer_mut().as_slice_mut()
        .ok_or_else(|| Error::DeviceError("GPU tensor not supported".into()))?;
    
    let a_strides = a.strides();
    let b_strides = b.strides();
    let c_strides = c.strides();
    
    let batch_count: usize = batch_dims.iter().product();
    let batch_rank = batch_dims.len();
    
    let a_batch_stride = if batch_rank > 0 { a_strides[0] as usize * batch_dims[1..].iter().product::<usize>() } else { m * k };
    let b_batch_stride = if batch_rank > 0 { b_strides[0] as usize * batch_dims[1..].iter().product::<usize>() } else { k * n };
    let c_batch_stride = if batch_rank > 0 { c_strides[0] as usize * batch_dims[1..].iter().product::<usize>() } else { m * n };
    
    for batch in 0..batch_count {
        let a_offset = batch * a_batch_stride;
        let b_offset = batch * b_batch_stride;
        let c_offset = batch * c_batch_stride;
        
        let a_slice = &a_data[a_offset..a_offset + m * k];
        let b_slice = &b_data[b_offset..b_offset + k * n];
        let c_slice = &mut c_data[c_offset..c_offset + m * n];
        
        let a_mat = unsafe {
            faer::mat::from_raw_parts(
                a_slice.as_ptr() as *const T,
                m,
                k,
                a_strides[batch_rank] as usize,
                a_strides[batch_rank + 1] as usize,
            )
        };
        let b_mat = unsafe {
            faer::mat::from_raw_parts(
                b_slice.as_ptr() as *const T,
                k,
                n,
                b_strides[batch_rank] as usize,
                b_strides[batch_rank + 1] as usize,
            )
        };
        let mut c_mat = unsafe {
            faer::mat::from_raw_parts_mut(
                c_slice.as_mut_ptr() as *mut T,
                m,
                n,
                c_strides[batch_rank] as usize,
                c_strides[batch_rank + 1] as usize,
            )
        };
        
        faer::linalg::matmul::matmul(c_mat, a_mat, b_mat, Some(alpha), beta)?;
    }
    
    Ok(())
}
```

**Step 2: Run tests**

Run: `cargo test -p tenferro-prims --features gemm-faer`
Expected: PASS

**Step 3: Commit**

```bash
git add tenferro-prims/src/cpu.rs
git commit -m "feat(prims): add strided GEMM for faer backend"
```

---

## Task 4: Update `execute_batched_gemm` to Use Strided Path

**Files:**
- Modify: `tenferro-prims/src/cpu.rs`

**Step 1: Modify existing `execute_batched_gemm`**

Replace the function at lines 1375-1407 with:

```rust
fn execute_batched_gemm<T: Scalar + 'static>(
    ctx: &mut CpuContext,
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
    batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    // Get Tensor references from StridedView metadata (temporary bridge)
    // This is a transition step; full Tensor direct path comes later
    
    #[cfg(feature = "gemm-faer")]
    {
        // For faer, we can use strided access directly
        // The StridedView already has stride info
        execute_bgemm_strided_from_views(ctx, alpha, inputs, beta, output, batch_dims, m, n, k)
    }
    
    #[cfg(all(not(feature = "gemm-faer"), feature = "gemm-openblas"))]
    {
        // For openblas, need contiguous - check and convert if needed
        execute_bgemm_lapack_from_views(ctx, alpha, inputs, beta, output, batch_dims, m, n, k)
    }
    
    #[cfg(all(not(feature = "gemm-faer"), not(feature = "gemm-openblas")))]
    {
        execute_bgemm_naive(ctx, alpha, inputs, beta, output, batch_dims, m, n, k)
    }
}
```

**Step 2: Add helper for strided from views**

```rust
#[cfg(feature = "gemm-faer")]
fn execute_bgemm_strided_from_views<T>(
    ctx: &mut CpuContext,
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
    batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()>
where
    T: Scalar + faer::Entity,
{
    let a = inputs[0];
    let b = inputs[1];
    
    // Use existing strided view directly with faer
    // ... delegate to existing implementation
    todo!("delegate to faer strided matmul")
}
```

**Step 3: Run tests**

Run: `cargo test -p tenferro-prims`
Expected: PASS (existing tests still work)

**Step 4: Commit**

```bash
git add tenferro-prims/src/cpu.rs
git commit -m "refactor(prims): prepare execute_batched_gemm for strided path"
```

---

## Task 5: Remove `permute_or_copy` and `make_contiguous_if_needed` from einsum

**Files:**
- Modify: `tenferro-einsum/src/lib.rs`

**Step 1: Simplify `fallback_pairwise_contraction`**

Modify lines 1161-1310 to remove explicit `permute_or_copy` and `make_contiguous_if_needed` calls. The GEMM primitive now handles this internally.

Replace the section that calls `permute_or_copy` and `make_contiguous_if_needed` with direct tensor operations that pass tensors directly to BatchedGemm.

**Step 2: Run tests**

Run: `cargo test -p tenferro-einsum`
Expected: PASS

**Step 3: Commit**

```bash
git add tenferro-einsum/src/lib.rs
git commit -m "refactor(einsum): remove explicit contiguous conversion - handled by prims"
```

---

## Task 6: Add Typed BufferPool Struct

**Files:**
- Create: `tenferro-einsum/src/pool.rs`
- Modify: `tenferro-einsum/src/lib.rs`

**Step 1: Create `pool.rs`**

```rust
use std::collections::BTreeMap;

pub struct BufferPool<T> {
    buffers: BTreeMap<usize, Vec<Vec<T>>>,
    total_bytes: usize,
    max_bytes: usize,
}

impl<T: Clone + Default> BufferPool<T> {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            buffers: BTreeMap::new(),
            total_bytes: 0,
            max_bytes,
        }
    }
    
    pub fn take(&mut self, len: usize) -> Vec<T> {
        if let Some((_, bufs)) = self.buffers.range_mut(len..).next() {
            if let Some(mut buf) = bufs.pop() {
                buf.truncate(len);
                buf.resize(len, T::default());
                return buf;
            }
        }
        vec![T::default(); len]
    }
    
    pub fn return_buf(&mut self, buf: Vec<T>) {
        let cap = buf.capacity();
        let new_bytes = cap * std::mem::size_of::<T>();
        
        if self.total_bytes + new_bytes > self.max_bytes {
            return; // Drop instead of pooling
        }
        
        self.total_bytes += new_bytes;
        self.buffers.entry(cap).or_default().push(buf);
    }
}
```

**Step 2: Add mod declaration in lib.rs**

Add near top of `tenferro-einsum/src/lib.rs`:

```rust
mod pool;
pub use pool::BufferPool;
```

**Step 3: Run tests**

Run: `cargo test -p tenferro-einsum`
Expected: PASS

**Step 4: Commit**

```bash
git add tenferro-einsum/src/pool.rs tenferro-einsum/src/lib.rs
git commit -m "feat(einsum): add typed BufferPool without TypeId/Any"
```

---

## Task 7: Replace Thread-local Pool with Typed Pool in `execute_tree`

**Files:**
- Modify: `tenferro-einsum/src/lib.rs`

**Step 1: Update `execute_tree` signature and implementation**

Modify function at lines 1532-1653:

```rust
fn execute_tree<Alg, Backend>(
    ctx: &mut Backend::Context,
    tree: &ContractionTree,
    operands: &[&Tensor<Alg::Scalar>],
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
) -> Result<()>
where
    Alg: Algebra,
    Alg::Scalar: Scalar + 'static,
    Backend: TensorPrims<Alg>,
{
    let mut pool = BufferPool::<Alg::Scalar>::new(64 * 1024 * 1024);
    
    // ... rest of implementation using pool.take() and pool.return_buf()
}
```

**Step 2: Update `alloc_tensor_pooled` to use typed pool**

Replace thread_local access with direct pool parameter.

**Step 3: Run tests**

Run: `cargo test -p tenferro-einsum`
Expected: PASS

**Step 4: Commit**

```bash
git add tenferro-einsum/src/lib.rs
git commit -m "refactor(einsum): replace thread_local pool with typed BufferPool"
```

---

## Task 8: Remove Old Thread-local Buffer Pool

**Files:**
- Modify: `tenferro-einsum/src/lib.rs`

**Step 1: Remove old pool code**

Delete lines 240-332 (thread_local! BUFFER_POOL, take_from_pool, return_to_pool, alloc_tensor_pooled, return_tensor_to_pool).

**Step 2: Run tests**

Run: `cargo test -p tenferro-einsum`
Expected: PASS

**Step 3: Commit**

```bash
git add tenferro-einsum/src/lib.rs
git commit -m "refactor(einsum): remove old thread_local buffer pool"
```

---

## Task 9: Add `ContractionPlan` Struct for Pre-computed Plans

**Files:**
- Modify: `tenferro-einsum/src/lib.rs`

**Step 1: Add `ContractionPlan` struct**

Add after `ContractionTree` definition:

```rust
pub struct ContractionPlan<S: Scalar> {
    steps: Vec<StepPlan<S>>,
    output_subscripts: Subscripts,
}

pub struct StepPlan<S: Scalar> {
    pub left_idx: usize,
    pub right_idx: usize,
    pub output_idx: usize,
    pub gemm_params: GemmParams,
    pub _marker: std::marker::PhantomData<S>,
}

pub struct GemmParams {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batch_dims: Vec<usize>,
    pub a_perm: Vec<usize>,
    pub b_perm: Vec<usize>,
    pub c_perm: Vec<usize>,
}
```

**Step 2: Run tests**

Run: `cargo test -p tenferro-einsum`
Expected: PASS

**Step 3: Commit**

```bash
git add tenferro-einsum/src/lib.rs
git commit -m "feat(einsum): add ContractionPlan for pre-computed execution"
```

---

## Task 10: Implement `prepare_tree_plan` Function

**Files:**
- Modify: `tenferro-einsum/src/lib.rs`

**Step 1: Add `prepare_tree_plan` function**

```rust
pub fn prepare_tree_plan<S, Backend>(
    tree: &ContractionTree,
    input_shapes: &[&[usize]],
    ctx: &mut Backend::Context,
) -> Result<ContractionPlan<S>>
where
    S: Scalar + 'static,
    Backend: TensorPrims<Standard<S>>,
{
    let mut steps = Vec::with_capacity(tree.steps.len());
    let mut intermediate_shapes: Vec<Vec<usize>> = input_shapes.iter().map(|s| s.to_vec()).collect();
    
    for step in &tree.steps {
        let left_shape = &intermediate_shapes[step.left_idx];
        let right_shape = &intermediate_shapes[step.right_idx];
        
        // Compute GEMM params from step subscripts
        let gemm_params = compute_gemm_params(&step.subs_a, &step.subs_b, &step.subs_c, left_shape, right_shape)?;
        
        let output_shape = compute_output_shape(&gemm_params);
        let output_idx = intermediate_shapes.len();
        intermediate_shapes.push(output_shape);
        
        steps.push(StepPlan {
            left_idx: step.left_idx,
            right_idx: step.right_idx,
            output_idx,
            gemm_params,
            _marker: std::marker::PhantomData,
        });
    }
    
    Ok(ContractionPlan {
        steps,
        output_subscripts: tree.output_subscripts.clone(),
    })
}
```

**Step 2: Run tests**

Run: `cargo test -p tenferro-einsum`
Expected: PASS

**Step 3: Commit**

```bash
git add tenferro-einsum/src/lib.rs
git commit -m "feat(einsum): implement prepare_tree_plan for plan pre-computation"
```

---

## Task 11: Implement `execute_with_plan` Function

**Files:**
- Modify: `tenferro-einsum/src/lib.rs`

**Step 1: Add `execute_with_plan` function**

```rust
pub fn execute_with_plan<S, Backend>(
    ctx: &mut Backend::Context,
    plan: &ContractionPlan<S>,
    inputs: &[&Tensor<S>],
    alpha: S,
    beta: S,
    output: &mut Tensor<S>,
    pool: &mut BufferPool<S>,
) -> Result<()>
where
    S: Scalar + 'static,
    Backend: TensorPrims<Standard<S>>,
{
    let mut intermediates: Vec<Option<Tensor<S>>> = inputs.iter().map(|t| Some((*t).clone())).collect();
    intermediates.resize(inputs.len() + plan.steps.len(), None);
    
    for step in &plan.steps {
        let left = intermediates[step.left_idx].as_ref().unwrap();
        let right = intermediates[step.right_idx].as_ref().unwrap();
        
        let mut result = execute_gemm_step::<S, Backend>(ctx, left, right, &step.gemm_params, pool)?;
        
        intermediates[step.output_idx] = Some(result);
        
        // Return consumed intermediates to pool
        if step.left_idx >= inputs.len() {
            if let Some(t) = intermediates[step.left_idx].take() {
                pool.return_buf(t.into_data());
            }
        }
        if step.right_idx >= inputs.len() {
            if let Some(t) = intermediates[step.right_idx].take() {
                pool.return_buf(t.into_data());
            }
        }
    }
    
    // Copy final result to output
    let final_result = intermediates.last().unwrap().as_ref().unwrap();
    output.copy_from(final_result);
    
    Ok(())
}
```

**Step 2: Run tests**

Run: `cargo test -p tenferro-einsum`
Expected: PASS

**Step 3: Commit**

```bash
git add tenferro-einsum/src/lib.rs
git commit -m "feat(einsum): implement execute_with_plan for Vec-indexed execution"
```

---

## Task 12: Update `einsum_with_plan` API to Return `ContractionPlan`

**Files:**
- Modify: `tenferro-einsum/src/lib.rs`

**Step 1: Update `einsum_with_plan` signature**

```rust
pub fn einsum_with_plan<S, Backend>(
    ctx: &mut Backend::Context,
    plan: &ContractionPlan<S>,
    operands: &[&Tensor<S>],
    alpha: S,
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()>
where
    S: Scalar + 'static,
    Backend: TensorPrims<Standard<S>>,
{
    let mut pool = BufferPool::new(64 * 1024 * 1024);
    execute_with_plan::<S, Backend>(ctx, plan, operands, alpha, beta, output, &mut pool)
}
```

**Step 2: Add `prepare_einsum_plan` convenience function**

```rust
pub fn prepare_einsum_plan<S, Backend>(
    subscripts: &str,
    input_shapes: &[&[usize]],
    ctx: &mut Backend::Context,
) -> Result<ContractionPlan<S>>
where
    S: Scalar + 'static,
    Backend: TensorPrims<Standard<S>>,
{
    let (tree, _) = parse_and_optimize(subscripts, input_shapes)?;
    prepare_tree_plan::<S, Backend>(&tree, input_shapes, ctx)
}
```

**Step 3: Run tests**

Run: `cargo test -p tenferro-einsum`
Expected: PASS

**Step 4: Commit**

```bash
git add tenferro-einsum/src/lib.rs
git commit -m "feat(einsum): update einsum_with_plan to use ContractionPlan"
```

---

## Task 13: Run Full Test Suite and Format

**Step 1: Run all tests**

Run: `cargo test --workspace`
Expected: PASS

**Step 2: Format code**

Run: `cargo fmt --all`

**Step 3: Check formatting**

Run: `cargo fmt --all --check`
Expected: No output (already formatted)

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: format and verify all tests pass"
```

---

## Task 14: Run Benchmark Validation

**Step 1: Run quick validation**

```bash
cd ../tenferro-einsum-benchmark
BENCH_INSTANCE=lm_brackets_4_4d RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo run --release
```

Expected: Ratio should be <2.0x (improvement from 2.47x)

**Step 2: Run full benchmark suite**

```bash
./scripts/run_all.sh 1
```

Expected: All instances show improvement, no regressions on `bin_matmul_256`.

---

## Summary

| Task | Description | Impact |
|------|-------------|--------|
| 1-4 | bgemm stride handling | faer: zero copy, lapack: conditional |
| 5 | Remove explicit contiguous conversion | Eliminates redundant copies |
| 6-8 | Typed buffer pool | No TypeId/Any overhead |
| 9-12 | Pre-computed plans | No HashMap lookup in hot loop |
| 13-14 | Validation | Verify improvements |
