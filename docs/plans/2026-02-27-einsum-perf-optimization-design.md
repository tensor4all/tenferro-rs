# Einsum Performance Optimization Design

**Issue**: [#236](https://github.com/tensor4all/tenferro-rs/issues/236)
**Date**: 2026-02-27

## Summary

tenferro-einsum is 1.2x–2.5x slower than strided-rs across the einsum benchmark suite. The gap scales with the number of contraction steps. This design optimizes the existing lazy permute + bgemm pipeline without introducing new fused primitives.

## Scope

Four optimizations:

1. **bgemm stride handling** - Backend-aware contiguous conversion inside bgemm
2. **Tensor direct processing** - Eliminate Tensor→StridedView conversion
3. **Buffer pool simplification** - Typed pool without TypeId/Any
4. **Plan cache pre-computation** - Vec-indexed plans instead of HashMap

---

## 1. bgemm Stride Handling

### Current

einsum calls `permute_or_copy` → `make_contiguous_if_needed` before bgemm, always attempting contiguous conversion.

### Change

bgemm primitive internally handles stride compatibility:

```rust
fn execute_batched_gemm<S: Scalar + GemmKernel>(
    ctx: &mut CpuContext,
    alpha: S, a: &Tensor<S>, b: &Tensor<S>,
    beta: S, c: &mut Tensor<S>,
    batch_dims: &[usize], m: usize, n: usize, k: usize,
) -> Result<()> {
    match ctx.gemm_backend {
        GemmBackend::Faer => {
            // faer supports arbitrary strides → execute directly
            execute_bgemm_faer_strided(alpha, a, b, beta, c, ...)?;
        }
        GemmBackend::Lapack => {
            // lapack requires contiguous → convert only when needed
            let a_cont = maybe_make_contiguous(a, ctx)?;
            let b_cont = maybe_make_contiguous(b, ctx)?;
            execute_bgemm_lapack(alpha, &a_cont, &b_cont, beta, c, ...)?;
        }
    }
}

fn maybe_make_contiguous<S: Scalar>(t: &Tensor<S>, ctx: &mut CpuContext) -> Result<Tensor<S>> {
    if is_gemm_compatible(t) {
        return Ok(t.clone());  // already compatible → no copy
    }
    make_contiguous_into(t, ctx)
}
```

### Impact

- faer backend: zero copies for GEMM
- lapack backend: copy only when strides are incompatible

---

## 2. Tensor Direct Processing

### Current

`TensorPrims::execute` converts `Tensor` → `StridedView` internally.

### Change

Use `Tensor::buffer()`, `Tensor::dims()`, `Tensor::strides()` directly:

```rust
fn execute_bgemm_faer_strided<S: Scalar>(
    alpha: S, a: &Tensor<S>, b: &Tensor<S>,
    beta: S, c: &mut Tensor<S>,
    batch_dims: &[usize], m: usize, n: usize, k: usize,
) -> Result<()> {
    let a_data = a.buffer().as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor".into()))?;
    let b_data = b.buffer().as_slice()?;
    let c_data = c.buffer_mut().as_slice_mut()?;
    
    let a_strides = a.strides();
    let b_strides = b.strides();
    let c_strides = c.strides();
    
    for batch in 0..batch_count {
        let a_batch = ... // slice via stride calculation
        let b_batch = ...
        let c_batch = ...
        faer::matmul_with_strides(alpha, a_batch, b_batch, beta, c_batch);
    }
    Ok(())
}
```

### Impact

- Eliminates Tensor→StridedView conversion overhead
- Removes redundant bounds checks

### Dependencies

`strided-view` retained for `Shape`/`Strides` types, but not used in hot paths.

---

## 3. Buffer Pool Simplification

### Current

```rust
thread_local! {
    static BUFFER_POOL: RefCell<HashMap<TypeId, Box<dyn Any>>>
        = RefCell::new(HashMap::new());
}
```

Requires TypeId hash + Any downcast + RefCell borrow check.

### Change

Typed pool passed directly to `execute_tree`:

```rust
struct BufferPool<T> {
    buffers: BTreeMap<usize, Vec<Vec<T>>>,  // key: capacity
}

impl<T> BufferPool<T> {
    fn take(&mut self, len: usize) -> Vec<T> {
        if let Some((_, bufs)) = self.buffers.range_mut(len..).next() {
            if let Some(buf) = bufs.pop() {
                buf.truncate(len);
                return buf;
            }
        }
        vec![T::zero(); len]
    }
    
    fn return_buf(&mut self, buf: Vec<T>) {
        let cap = buf.capacity();
        self.buffers.entry(cap).or_default().push(buf);
    }
}

fn execute_tree<T: Scalar>(tree: &ContractionTree, inputs: &[Tensor<T>], ...) {
    let mut pool = BufferPool::<T>::new();
    // use pool.take() / pool.return_buf() during execution
}
```

### Impact

- Eliminates `TypeId::of::<T>()` hash
- Eliminates `dyn Any` downcast
- Eliminates `RefCell` borrow check overhead

### Note

Pool is a local variable in `execute_tree`, not thread_local. Cross-thread reuse is out of scope.

---

## 4. Plan Cache Pre-computation

### Current

Each primitive call performs HashMap lookup:

```rust
fn plan(ctx: &mut CpuContext, desc: &PrimDescriptor, shapes: &[&[usize]]) -> Result<CpuPlan> {
    if let Some(cached) = ctx.plan_cache.get(desc, shapes) {
        return Ok(cached);
    }
    // build and cache...
}
```

### Change

Pre-compute all plans before tree execution, store as `Vec<StepPlan>`:

```rust
struct ContractionPlan<S: Scalar> {
    steps: Vec<StepPlan<S>>,
    output_subscripts: Subscripts,
}

enum StepPlan<S: Scalar> {
    Binary {
        left_idx: usize,
        right_idx: usize,
        gemm_plan: GemmPlan,      // pre-computed
        output_subscripts: Subscripts,
    },
    Unary { ... },
}

fn prepare_tree_plan<S: Scalar>(
    tree: &ContractionTree,
    input_shapes: &[&[usize]],
    ctx: &mut CpuContext,
) -> Result<ContractionPlan<S>> {
    let mut steps = Vec::with_capacity(tree.steps.len());
    for step in &tree.steps {
        let gemm_plan = build_gemm_plan(...)?;
        steps.push(StepPlan::Binary { ... });
    }
    Ok(ContractionPlan { steps, ... })
}

fn execute_with_plan<S: Scalar>(plan: &ContractionPlan<S>, ...) {
    for step in &plan.steps {
        match step {
            StepPlan::Binary { gemm_plan, .. } => {
                execute_gemm_with_plan(gemm_plan, ...);  // no HashMap lookup
            }
        }
    }
}
```

### Impact

- Eliminates HashMap lookup in execution loop
- Eliminates hash computation
- O(1) index access only

### API Change

`einsum_with_plan()` returns `ContractionPlan` for reuse.

---

## Validation Strategy

Test against most affected benchmark instances:

| Instance | Steps | Current Gap | Target |
|----------|-------|-------------|--------|
| lm_brackets_4_4d | 83 | 2.47x | <1.5x |
| lm_sentence_4_4d | 83 | 2.49x | <1.5x |
| gm_queen5_5_3 | 159 | 1.93x | <1.3x |
| bin_matmul_256 | 1 | 1.04x | no regression |

```bash
# Quick validation
BENCH_INSTANCE=lm_brackets_4_4d \
  RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo run --release

# Full suite
cd ../tenferro-einsum-benchmark && ./scripts/run_all.sh 1
```

---

## Implementation Order

1. **bgemm stride handling** - Highest impact, lowest risk
2. **Tensor direct processing** - Works with (1)
3. **Buffer pool simplification** - Independent
4. **Plan cache pre-computation** - Independent, requires API change
