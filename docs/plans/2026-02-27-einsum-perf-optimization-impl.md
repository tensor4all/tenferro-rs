# Einsum Performance Optimization Implementation Plan (v2: Contract-First)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce tenferro-einsum per-step overhead from 4-6 plan/execute round-trips to 1, closing the 1.2x–3.2x gap with strided-rs.

**Architecture:** Enable `Extension::Contract` as the preferred CPU optimization path (fused permute+GEMM). Cache mode analysis in `CpuPlan::Contract`. Reduce allocation overhead via Arc dims/strides, typed buffer pool, and uninit buffers.

**Tech Stack:** Rust, faer (strided GEMM), `Arc<[usize]>` (shared dims/strides), `BTreeMap` (typed pool)

---

## Task 1: Enable Contract Extension on CPU — Update Test

**Files:**
- Modify: `tenferro-prims/tests/prims_tests.rs:302-304`

**Step 1: Update the existing test**

The test currently expects Contract to be disabled. Flip the assertion:

```rust
#[test]
fn cpu_has_extension_contract() {
    assert!(cpu_has_ext::<f64>(Extension::Contract));
}
```

**Step 2: Run test to verify it fails (current code has the fix but test is old)**

Run: `cargo test -p tenferro-prims cpu_has_extension_contract`
Expected: If the `has_extension_for` change is already applied, PASS. If not, FAIL — apply the `cpu.rs` change next.

**Step 3: Verify `has_extension_for` change in cpu.rs**

`tenferro-prims/src/cpu.rs:2637-2639` should read:

```rust
fn has_extension_for(_ext: Extension) -> bool {
    matches!(_ext, Extension::ElementwiseMul | Extension::Contract)
}
```

(Already applied in uncommitted diff.)

**Step 4: Run full prims test suite**

Run: `cargo test -p tenferro-prims`
Expected: PASS

**Step 5: Commit**

```bash
git add tenferro-prims/src/cpu.rs tenferro-prims/tests/prims_tests.rs
git commit -m "feat(prims): enable Extension::Contract on CPU backend"
```

---

## Task 2: Contract-First Dispatch in Einsum

**Files:**
- Modify: `tenferro-einsum/src/lib.rs:1964-1971` (the `StepStrategy::Gemm` and `StepStrategy::Contract` arms in `execute_pairwise_with_plan`)

**Step 1: Verify the dispatch change is applied**

`execute_pairwise_with_plan` Gemm/Contract arms should read:

```rust
StepStrategy::Gemm(gemm_plan) => {
    if Backend::has_extension_for(Extension::Contract) {
        // Preferred optimization path: fused Contract
        let desc = PrimDescriptor::Contract {
            modes_a: subs_a.to_vec(),
            modes_b: subs_b.to_vec(),
            modes_c: subs_c.to_vec(),
        };
        let shapes = [a.dims(), b.dims(), output.dims()];
        let prim_plan = Backend::plan(ctx, &desc, &shapes)?;
        Backend::execute(ctx, &prim_plan, alpha, &[a, b], beta, output)
    } else {
        // Fallback: core ops decomposition
        execute_gemm_with_plan::<Alg, Backend>(
            ctx, gemm_plan, subs_c, a, b, alpha, beta, output,
        )
    }
}
StepStrategy::Contract => {
    // Contract extension: direct fused execution
    let desc = PrimDescriptor::Contract {
        modes_a: subs_a.to_vec(),
        modes_b: subs_b.to_vec(),
        modes_c: subs_c.to_vec(),
    };
    let shapes = [a.dims(), b.dims(), output.dims()];
    let prim_plan = Backend::plan(ctx, &desc, &shapes)?;
    Backend::execute(ctx, &prim_plan, alpha, &[a, b], beta, output)
}
```

(Already applied in uncommitted diff.)

**Step 2: Run einsum test suite**

Run: `cargo test -p tenferro-einsum`
Expected: PASS — all existing einsum operations now route through Contract instead of Gemm decomposition, producing identical results.

**Step 3: Run full workspace tests**

Run: `cargo test --workspace`
Expected: PASS

**Step 4: Commit**

```bash
git add tenferro-einsum/src/lib.rs
git commit -m "perf(einsum): prefer Contract extension over Gemm decomposition

When the backend supports Extension::Contract, use fused Contract
(1 plan + 1 execute) instead of decomposed core ops (4-6 round-trips).
The Gemm path remains as fallback for backends without Contract."
```

---

## Task 3: Benchmark Validation — Layer 1

**Step 1: Run quick benchmark on worst case**

```bash
cd ../tenferro-einsum-benchmark
BENCH_INSTANCE=gm_queen5_5_3.wcsp RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo run --release 2>&1
```

Expected: Significant improvement from ~6942ms baseline. Record the result.

**Step 2: Run full benchmark suite**

```bash
./scripts/run_all.sh 1
```

Expected: No regressions on any instance. Record results for comparison.

---

## Task 4: Contract Plan Caching — Add `ContractGemmSpec`

**Files:**
- Modify: `tenferro-prims/src/cpu.rs:185-193` (`CpuPlan::Contract` variant)
- Modify: `tenferro-prims/src/cpu.rs:678-700` (`build_plan` Contract arm)

**Step 1: Add `ContractGemmSpec` struct**

Add before the `CpuPlan` enum definition (around line 90):

```rust
/// Pre-computed mode analysis for Contract GEMM fast path.
/// Cached in `CpuPlan::Contract` to avoid re-analyzing on every execute.
#[derive(Debug, Clone)]
struct ContractGemmSpec {
    /// Permutation to apply to A's modes to get [batch, m, k] order.
    a_target: Vec<u32>,
    /// Permutation to apply to B's modes to get [batch, k, n] order.
    b_target: Vec<u32>,
    /// Permutation to apply to C's modes to get [batch, m, n] order.
    c_target: Vec<u32>,
    /// Batch modes (in both A, B, and C).
    batch_modes: Vec<u32>,
    /// Left-output modes (in A and C, not B).
    m_modes: Vec<u32>,
    /// Right-output modes (in B and C, not A).
    n_modes: Vec<u32>,
    /// Contracted modes (in A and B, not C).
    k_modes: Vec<u32>,
}
```

**Step 2: Extend `CpuPlan::Contract` to include cached spec**

```rust
Contract {
    modes_a: Vec<u32>,
    modes_b: Vec<u32>,
    modes_c: Vec<u32>,
    /// Pre-computed GEMM analysis. None if not GEMM-compatible.
    gemm_spec: Option<ContractGemmSpec>,
    _marker: PhantomData<T>,
},
```

**Step 3: Update `build_plan` Contract arm to compute spec**

In the `PrimDescriptor::Contract` match arm (line ~678), after validation, add:

```rust
// Pre-compute GEMM analysis (mirrors build_mode_spec in try_execute_contract_gemm)
let gemm_spec = build_contract_gemm_spec(modes_a, modes_b, modes_c);

Ok(CpuPlan::Contract {
    modes_a: modes_a.clone(),
    modes_b: modes_b.clone(),
    modes_c: modes_c.clone(),
    gemm_spec,
    _marker: PhantomData,
})
```

Add helper function `build_contract_gemm_spec`:

```rust
fn build_contract_gemm_spec(
    modes_a: &[u32],
    modes_b: &[u32],
    modes_c: &[u32],
) -> Option<ContractGemmSpec> {
    // Classify modes (same logic as build_mode_spec in try_execute_contract_gemm)
    let batch_modes: Vec<u32> = modes_c.iter().copied()
        .filter(|m| modes_a.contains(m) && modes_b.contains(m)).collect();
    let m_modes: Vec<u32> = modes_c.iter().copied()
        .filter(|m| modes_a.contains(m) && !modes_b.contains(m)).collect();
    let n_modes: Vec<u32> = modes_c.iter().copied()
        .filter(|m| modes_b.contains(m) && !modes_a.contains(m)).collect();
    let k_modes: Vec<u32> = modes_a.iter().copied()
        .filter(|m| modes_b.contains(m) && !modes_c.contains(m)).collect();

    // Validate counts
    let expected_a = batch_modes.len() + m_modes.len() + k_modes.len();
    let expected_b = batch_modes.len() + k_modes.len() + n_modes.len();
    if expected_a != modes_a.len() || expected_b != modes_b.len() {
        return None;
    }
    if batch_modes.len() + m_modes.len() + n_modes.len() != modes_c.len() {
        return None;
    }

    // Build target mode orders
    let a_target: Vec<u32> = batch_modes.iter()
        .chain(m_modes.iter()).chain(k_modes.iter()).copied().collect();
    let b_target: Vec<u32> = batch_modes.iter()
        .chain(k_modes.iter()).chain(n_modes.iter()).copied().collect();
    let c_target: Vec<u32> = batch_modes.iter()
        .chain(m_modes.iter()).chain(n_modes.iter()).copied().collect();

    Some(ContractGemmSpec {
        a_target, b_target, c_target,
        batch_modes, m_modes, n_modes, k_modes,
    })
}
```

**Step 4: Update execute match arm for Contract**

In the `execute` method's `CpuPlan::Contract` arm (line ~2617), pass `gemm_spec`:

```rust
CpuPlan::Contract {
    modes_a, modes_b, modes_c, gemm_spec, ..
} => {
    validate_execute_inputs(inputs, 2, "Contract")?;
    execute_contract(
        alpha, &view_refs, beta, &mut out_view,
        modes_a, modes_b, modes_c, gemm_spec.as_ref(),
    )
}
```

**Step 5: Update `execute_contract` signature**

Add `gemm_spec: Option<&ContractGemmSpec>` parameter. Pass it to `try_execute_contract_gemm`. When spec is `Some`, skip `build_mode_spec` and use cached values.

**Step 6: Run tests**

Run: `cargo test -p tenferro-prims`
Expected: PASS

Run: `cargo test --workspace`
Expected: PASS

**Step 7: Commit**

```bash
git add tenferro-prims/src/cpu.rs
git commit -m "perf(prims): cache Contract GEMM analysis in CpuPlan

Pre-compute mode classification (batch/m/n/k) and target permutations
during plan(), avoiding repeated Vec allocations and O(n²) scans in
execute(). The spec is reused across all 15 benchmark runs per step."
```

---

## Task 5: Use Cached Spec in `try_execute_contract_gemm`

**Files:**
- Modify: `tenferro-prims/src/cpu.rs:2024-2100` (`try_execute_contract_gemm` and inner functions)

**Step 1: Add cached spec parameter to `try_execute_contract_gemm`**

```rust
fn try_execute_contract_gemm<T: Scalar + 'static>(
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
    modes_a: &[u32],
    modes_b: &[u32],
    modes_c: &[u32],
    cached_spec: Option<&ContractGemmSpec>,  // new parameter
) -> Result<Option<()>> {
```

**Step 2: Use cached spec when available**

Replace the `build_mode_spec` call with:

```rust
let spec = if let Some(cached) = cached_spec {
    // Use pre-computed spec from plan phase
    ModeSpec {
        batch_modes: cached.batch_modes.clone(),
        m_modes: cached.m_modes.clone(),
        n_modes: cached.n_modes.clone(),
        k_modes: cached.k_modes.clone(),
    }
} else {
    // Fallback: compute on the fly
    match build_mode_spec(modes_a, modes_b, modes_c) {
        Some(s) => s,
        None => return Ok(None),
    }
};
```

And for `reordered_dims_strides`, use cached target orders:

```rust
let a_target = if let Some(cached) = cached_spec {
    cached.a_target.clone()
} else {
    spec.batch_modes.iter().chain(spec.m_modes.iter()).chain(spec.k_modes.iter()).copied().collect()
};
// Similarly for b_target, c_target
```

**Step 3: Run tests**

Run: `cargo test -p tenferro-prims`
Expected: PASS

Run: `cargo test --workspace`
Expected: PASS

**Step 4: Commit**

```bash
git add tenferro-prims/src/cpu.rs
git commit -m "perf(prims): use cached ContractGemmSpec in execute path"
```

---

## Task 6: Benchmark Validation — Layer 2

**Step 1: Run benchmark**

```bash
cd ../tenferro-einsum-benchmark
BENCH_INSTANCE=gm_queen5_5_3.wcsp RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo run --release 2>&1
```

Expected: Further improvement from Layer 1 result.

**Step 2: Run full suite**

```bash
./scripts/run_all.sh 1
```

Record results.

---

## Task 7: Arc Dims/Strides in Tensor — Change Struct Fields

**Files:**
- Modify: `tenferro-tensor/src/lib.rs:668-693` (Tensor struct)
- Modify: `tenferro-tensor/src/lib.rs:792-810` (Tensor::zeros)
- Modify: `tenferro-tensor/src/lib.rs:908-957` (Tensor::from_vec)

**Step 1: Change Tensor struct fields**

```rust
pub struct Tensor<T: Scalar> {
    buffer: DataBuffer<T>,
    dims: Arc<[usize]>,       // was: Vec<usize>
    strides: Arc<[isize]>,    // was: Vec<isize>
    offset: isize,
    // ... rest unchanged
}
```

Add `use std::sync::Arc;` at the top of the file if not already present.

**Step 2: Update `dims()` and `strides()` accessor methods**

These likely return `&[usize]` / `&[isize]` — they should continue to work since `Arc<[T]>` derefs to `&[T]`.

**Step 3: Update `Tensor::zeros`**

```rust
pub fn zeros(dims: &[usize], memory_space: LogicalMemorySpace, order: MemoryOrder) -> Self {
    // ...
    let strides = compute_contiguous_strides(dims, order);
    Tensor {
        buffer: DataBuffer::from_vec(vec![T::zero(); n_elements]),
        dims: Arc::from(dims),           // was: dims.to_vec()
        strides: Arc::from(strides),     // compute_contiguous_strides returns Vec
        // ... rest unchanged
    }
}
```

**Step 4: Update `Tensor::from_vec`**

```rust
Ok(Tensor {
    buffer: DataBuffer::from_vec(data),
    dims: Arc::from(dims),           // was: dims.to_vec()
    strides: Arc::from(strides),     // was: strides.to_vec()
    offset,
    // ... rest unchanged
})
```

**Step 5: Fix all other constructors and methods that set dims/strides**

Search for all `dims:` and `strides:` assignments in the Tensor impl blocks. Common patterns:
- `dims.to_vec()` → `Arc::from(dims)` or `dims.into()` (when dims is `&[usize]`)
- `strides.to_vec()` → `Arc::from(strides)` (when strides is `&[isize]`)
- `self.dims.clone()` → already cheap (Arc refcount)

Use `cargo build -p tenferro-tensor` to find remaining compilation errors.

**Step 6: Run tests**

Run: `cargo test -p tenferro-tensor`
Expected: PASS

Run: `cargo test --workspace`
Expected: PASS (callers use `&[usize]` / `&[isize]` via deref)

**Step 7: Commit**

```bash
git add tenferro-tensor/src/lib.rs
git commit -m "perf(tensor): use Arc<[usize]>/Arc<[isize]> for dims/strides

Clone is now refcount increment only. Eliminates 2-4 heap allocations
per Tensor construction and StridedView conversion."
```

---

## Task 8: Update `tensor_to_view_mut` for Arc Dims/Strides

**Files:**
- Modify: `tenferro-prims/src/cpu.rs:32-42` (`tensor_to_view_mut`)

**Step 1: Update `tensor_to_view_mut`**

With Arc dims/strides, we no longer need `to_vec()`:

```rust
fn tensor_to_view_mut<T: Scalar>(t: &mut Tensor<T>) -> Result<StridedViewMut<'_, T>> {
    let dims = t.dims();       // &[usize] via Arc deref — no allocation
    let strides = t.strides(); // &[isize] via Arc deref — no allocation
    let offset = t.offset();
    let data = t
        .buffer_mut()
        .as_mut_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;
    StridedViewMut::new(data, dims, strides, offset)
        .map_err(|e| Error::StrideError(format!("{e}")))
}
```

Note: Check if `StridedViewMut::new` borrows `dims`/`strides` or takes ownership. If it needs owned data, pass borrowed slices and let it convert internally.

**Step 2: Run tests**

Run: `cargo test -p tenferro-prims`
Expected: PASS

Run: `cargo test --workspace`
Expected: PASS

**Step 3: Commit**

```bash
git add tenferro-prims/src/cpu.rs
git commit -m "perf(prims): eliminate Vec allocs in tensor_to_view_mut"
```

---

## Task 9: Typed Buffer Pool — Create `BufferPool` Struct

**Files:**
- Create: `tenferro-einsum/src/pool.rs`
- Modify: `tenferro-einsum/src/lib.rs` (add mod declaration)

**Step 1: Create `pool.rs`**

```rust
use std::collections::BTreeMap;

const MAX_POOLED_BYTES: usize = 64 * 1024 * 1024; // 64 MB

/// Typed buffer pool using BTreeMap for O(log n) best-fit allocation.
///
/// Passed as argument to `execute_tree`, not thread-local.
pub(crate) struct BufferPool<T> {
    buffers: BTreeMap<usize, Vec<Vec<T>>>,
    total_bytes: usize,
}

impl<T> BufferPool<T> {
    pub fn new() -> Self {
        Self {
            buffers: BTreeMap::new(),
            total_bytes: 0,
        }
    }

    /// Take a buffer of at least `len` capacity from the pool.
    /// Returns an uninitialized buffer (caller must fill before reading).
    pub fn take(&mut self, len: usize) -> Vec<T> {
        if let Some((&cap, bufs)) = self.buffers.range_mut(len..).next() {
            if let Some(mut buf) = bufs.pop() {
                if bufs.is_empty() {
                    self.buffers.remove(&cap);
                }
                self.total_bytes -= cap * std::mem::size_of::<T>();
                // Safety: capacity >= len; caller writes all elements before reading.
                unsafe { buf.set_len(len) };
                return buf;
            }
        }
        let mut buf = Vec::with_capacity(len);
        unsafe { buf.set_len(len) };
        buf
    }

    /// Return a buffer to the pool for reuse.
    pub fn return_buf(&mut self, mut buf: Vec<T>) {
        let cap = buf.capacity();
        let bytes = cap * std::mem::size_of::<T>();
        if bytes == 0 || self.total_bytes + bytes > MAX_POOLED_BYTES {
            return; // drop
        }
        buf.clear();
        self.total_bytes += bytes;
        self.buffers.entry(cap).or_default().push(buf);
    }
}

impl<T> Default for BufferPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_returns_correct_length() {
        let mut pool = BufferPool::<f64>::new();
        let buf = pool.take(100);
        assert_eq!(buf.len(), 100);
        assert!(buf.capacity() >= 100);
    }

    #[test]
    fn return_and_reuse() {
        let mut pool = BufferPool::<f64>::new();
        let buf = pool.take(100);
        let ptr = buf.as_ptr();
        pool.return_buf(buf);
        let buf2 = pool.take(50); // should reuse the 100-capacity buffer
        assert_eq!(buf2.as_ptr(), ptr);
        assert_eq!(buf2.len(), 50);
    }

    #[test]
    fn best_fit_selection() {
        let mut pool = BufferPool::<f64>::new();
        // Return buffers of different sizes
        let small = Vec::<f64>::with_capacity(50);
        let large = Vec::<f64>::with_capacity(200);
        pool.return_buf(small);
        pool.return_buf(large);
        // Request 60: should get the 200-cap (smallest >= 60), since 50 < 60
        let buf = pool.take(60);
        assert!(buf.capacity() >= 60);
    }
}
```

**Step 2: Add mod declaration in lib.rs**

Near the top of `tenferro-einsum/src/lib.rs`, add:

```rust
mod pool;
use pool::BufferPool;
```

**Step 3: Run tests**

Run: `cargo test -p tenferro-einsum pool`
Expected: PASS (3 pool tests)

**Step 4: Commit**

```bash
git add tenferro-einsum/src/pool.rs tenferro-einsum/src/lib.rs
git commit -m "feat(einsum): add typed BufferPool with BTreeMap best-fit lookup"
```

---

## Task 10: Wire BufferPool into `execute_tree`

**Files:**
- Modify: `tenferro-einsum/src/lib.rs`

**Step 1: Add pool parameter to `execute_tree`**

Update signature (line ~2205):

```rust
fn execute_tree<Alg, Backend>(
    ctx: &mut Backend::Context,
    tree: &ContractionTree,
    operands: &[&Tensor<Alg::Scalar>],
    alpha: Alg::Scalar,
    beta: Alg::Scalar,
    output: &mut Tensor<Alg::Scalar>,
    pool: &mut BufferPool<Alg::Scalar>,  // new parameter
) -> Result<()>
```

**Step 2: Replace `alloc_tensor_pooled` calls with pool**

In the loop body (~line 2300), replace:
```rust
let mut result = alloc_tensor_pooled::<Alg::Scalar>(result_shape, memory_space);
```
with:
```rust
let numel = result_shape.iter().product::<usize>().max(1);
let data = pool.take(numel);
let strides = compute_col_major_strides(result_shape);
let mut result = Tensor::from_vec(data, result_shape, &strides, 0)
    .expect("pooled tensor creation should not fail");
```

(Add a helper `compute_col_major_strides` if one doesn't exist.)

**Step 3: Replace `return_tensor_to_pool` calls with pool**

Replace:
```rust
if let Some(t) = intermediates[*idx].take() {
    return_tensor_to_pool(t);
}
```
with:
```rust
if let Some(t) = intermediates[*idx].take() {
    if let Some(data) = t.try_into_data_vec() {
        pool.return_buf(data);
    }
}
```

**Step 4: Update callers of `execute_tree`**

Find all calls to `execute_tree` and pass `&mut pool`. The main caller is `einsum_with_plan` — create pool there:

```rust
let mut pool = BufferPool::new();
execute_tree::<Alg, Backend>(ctx, tree, operands, alpha, beta, output, &mut pool)?;
```

**Step 5: Run tests**

Run: `cargo test -p tenferro-einsum`
Expected: PASS

Run: `cargo test --workspace`
Expected: PASS

**Step 6: Commit**

```bash
git add tenferro-einsum/src/lib.rs
git commit -m "refactor(einsum): wire typed BufferPool into execute_tree"
```

---

## Task 11: Remove Old Thread-Local Pool

**Files:**
- Modify: `tenferro-einsum/src/lib.rs:240-332`

**Step 1: Remove old pool code**

Delete:
- `thread_local! BUFFER_POOL` definition (line ~240)
- `MAX_POOL_PER_TYPE` and `MAX_POOLED_BYTES` constants (lines ~245-246)
- `take_from_pool` function (lines ~248-275)
- `return_to_pool` function (lines ~277-306)
- `alloc_tensor_pooled` function (lines ~309-325)
- `return_tensor_to_pool` function (lines ~328-332)
- Related imports: `std::any::{Any, TypeId}`, `std::cell::RefCell`, `std::collections::HashMap` (if no longer used)

**Step 2: Verify no remaining references**

Run: `cargo build -p tenferro-einsum`
Expected: No compile errors (all usages were replaced in Task 10).

**Step 3: Run tests**

Run: `cargo test --workspace`
Expected: PASS

**Step 4: Commit**

```bash
git add tenferro-einsum/src/lib.rs
git commit -m "refactor(einsum): remove old thread_local buffer pool

Replaced by typed BufferPool passed as argument to execute_tree.
Eliminates TypeId hash, dyn Any downcast, and RefCell overhead."
```

---

## Task 12: Benchmark Validation — Full Optimization

**Step 1: Run quick benchmark**

```bash
cd ../tenferro-einsum-benchmark
BENCH_INSTANCE=gm_queen5_5_3.wcsp RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo run --release 2>&1
```

Expected: gm_queen5_5_3.wcsp < 3000ms (from 6942ms baseline).

**Step 2: Run full suite comparison**

```bash
./scripts/run_all.sh 1
```

Expected: All instances improved, no regression on bin_matmul_256.

**Step 3: Record results**

Save benchmark output to `data/results/` and format with:

```bash
uv run python scripts/format_results.py data/results/tenferro_einsum_t1_*.log data/results/strided_faer_t1_*.log
```

---

## Summary

| Task | Layer | Description | Risk |
|------|-------|-------------|------|
| 1 | 1 | Enable `has_extension_for(Contract)` + update test | Low |
| 2 | 1 | Contract-first dispatch in einsum | Low |
| 3 | 1 | Benchmark validation (Layer 1) | — |
| 4-5 | 2 | Contract plan caching (`ContractGemmSpec`) | Medium |
| 6 | 2 | Benchmark validation (Layer 2) | — |
| 7 | 3 | Arc dims/strides in Tensor | Medium |
| 8 | 3 | Update tensor_to_view_mut | Low |
| 9 | 4 | Create typed BufferPool | Low |
| 10 | 4 | Wire BufferPool into execute_tree | Medium |
| 11 | 4 | Remove old thread-local pool | Low |
| 12 | — | Final benchmark validation | — |
