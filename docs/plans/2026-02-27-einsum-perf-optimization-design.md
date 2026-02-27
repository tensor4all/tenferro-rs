# Einsum Performance Optimization Design (v2: Contract-First)

**Issue**: [#236](https://github.com/tensor4all/tenferro-rs/issues/236)
**Date**: 2026-02-27
**Supersedes**: v1 (Approach B pipeline optimization)

## Summary

tenferro-einsum is 1.2x–3.2x slower than strided-rs across the einsum benchmark suite. The gap scales with the number of contraction steps — worst case `gm_queen5_5_3.wcsp` (159 steps, 3×3 tensors) shows 3.2x slowdown where per-step dispatch overhead dominates computation.

**Root cause**: The current einsum layer decomposes pairwise contractions into multiple core ops (permute_view → MakeContiguous → BatchedGemm → Permute), each requiring a separate plan/execute round-trip with Tensor→StridedView conversion. This amounts to 4-6 round-trips per step.

**Solution**: Use `Extension::Contract` as the preferred optimization path on CPU (as specified in tensor-prims.md design), reducing each step to a single plan/execute round-trip. Combined with allocation reduction (Arc dims/strides, typed pool, uninit buffers).

## Design Principles

- **Extensions are optimization paths** — GPU or CPU, `Extension::Contract` and `Extension::ElementwiseMul` are preferred when available; core ops decomposition is the fallback.
- **Contract already exists** — `execute_contract` and `try_execute_contract_gemm` are fully implemented in `cpu.rs`. Only the dispatch routing and plan caching need changes.
- **Minimize API surface change** — Optimize internal dispatch without changing public einsum API.

---

## Layer 1: Contract Extension as Preferred Path

### Problem

The design docs (`tensor-prims.md`) specify:
```
has_extension_for(Contract)? → YES → execute Contract plan
                             → NO  → decompose into core ops
```

But the implementation has two bugs:
1. `cpu.rs:has_extension_for` returns `false` for Contract (only `true` for ElementwiseMul)
2. `einsum/lib.rs:compile_step_plans` treats Contract as last-resort fallback

### Changes

**`tenferro-prims/src/cpu.rs`** — Enable Contract:
```rust
fn has_extension_for(_ext: Extension) -> bool {
    matches!(_ext, Extension::ElementwiseMul | Extension::Contract)
}
```

**`tenferro-einsum/src/lib.rs`** — `execute_pairwise_with_plan`: When Contract extension is available, use it for both `StepStrategy::Gemm` and `StepStrategy::Contract`:
```rust
StepStrategy::Gemm(gemm_plan) => {
    if Backend::has_extension_for(Extension::Contract) {
        // Preferred: fused Contract (1 plan + 1 execute)
        let desc = PrimDescriptor::Contract { modes_a, modes_b, modes_c };
        Backend::plan(ctx, &desc, &shapes)?;
        Backend::execute(ctx, &plan, alpha, &[a, b], beta, output)
    } else {
        // Fallback: core ops decomposition (4-6 plan/execute)
        execute_gemm_with_plan(ctx, gemm_plan, ...)
    }
}
```

`compile_step_plans` remains unchanged — Gemm pre-computation serves as backup for non-Contract backends.

### Impact

Per-step: 4-6 plan/execute round-trips → 1 plan + 1 execute.
Expected: ~2-3x improvement on many-step instances (gm_queen, lm_brackets, lm_sentence).

---

## Layer 2: Contract Plan Caching

### Problem

`execute_contract` → `try_execute_contract_gemm` recomputes on every call:
- `build_mode_spec`: 4 Vec allocations + contains() scans
- `reordered_dims_strides`: 2 Vec allocations per operand
- `perm_for`: Vec allocation + position() scans
- `try_fuse_group`: sorting + fusability checks

For 159 steps × 15 runs, that's ~10,000 redundant analyses.

### Changes

Extend `CpuPlan::Contract` to cache the analysis (matching the `CpuContractionPlan` in `contraction-pipeline.md`):

```rust
CpuPlan::Contract {
    modes_a: Vec<u32>,
    modes_b: Vec<u32>,
    modes_c: Vec<u32>,
    // Cached analysis from try_execute_contract_gemm:
    cached_spec: Option<ContractGemmSpec>,
}

struct ContractGemmSpec {
    a_perm: Vec<usize>,
    b_perm: Vec<usize>,
    c_perm: Vec<usize>,
    batch_size: usize,
    m: usize,
    n: usize,
    k: usize,
    // Per-operand fusability (depends on strides, checked at execute time)
}
```

`build_plan` for Contract performs the mode analysis once. `execute` uses cached spec and only checks stride-dependent fusability at runtime.

### Impact

Eliminates ~6 Vec allocations + O(n²) scans per step.

---

## Layer 3: Arc Dims/Strides in Tensor

### Problem

`Tensor` uses `Vec<usize>` for dims and `Vec<isize>` for strides. Every clone, reshape, or view conversion allocates:
- `Tensor::from_vec`: `dims.to_vec()` + `strides.to_vec()` = 2 heap allocs
- `tensor_to_view_mut`: `t.dims().to_vec()` + `t.strides().to_vec()` = 2 heap allocs (4 total per mutable view)
- `Tensor::reshape`: `new_dims.to_vec()` = 1 heap alloc

### Changes

```rust
pub struct Tensor<T: Scalar> {
    buffer: DataBuffer<T>,
    dims: Arc<[usize]>,       // was: Vec<usize>
    strides: Arc<[isize]>,    // was: Vec<isize>
    offset: isize,
    // ... other fields unchanged
}
```

- Clone: refcount increment only (no heap alloc)
- `tensor_to_view`: pass `Arc` slices directly (no copy)
- `tensor_to_view_mut`: still needs owned dims/strides for mutation; use `Arc::make_mut` or accept borrowed slices in `StridedViewMut`

### Impact

Eliminates 2-4 heap allocations per Tensor construction/view conversion. With 159 steps and ~3 tensors touched per step, this saves ~500-1000 allocations per benchmark run.

### Compatibility

`StridedView` in strided-rs already uses `Arc<[usize]>` / `Arc<[isize]>`. This aligns tenferro-tensor with that convention.

---

## Layer 4: Typed Buffer Pool (Argument-Passed)

### Problem

Current pool: `thread_local! { HashMap<TypeId, Box<dyn Any>> }` with:
- TypeId hash lookup
- `dyn Any` downcast
- RefCell borrow check
- Linear scan (max 16 entries) for best-fit

strided-rs pool: `BTreeMap<usize, Vec<Vec<T>>>` with:
- O(log n) range lookup for best-fit
- No TypeId/downcast
- Passed as argument (no TLS)

### Changes

```rust
pub(crate) struct BufferPool<T> {
    buffers: BTreeMap<usize, Vec<Vec<T>>>,
}

impl<T> BufferPool<T> {
    fn take(&mut self, len: usize) -> Vec<T> {
        // O(log n) range lookup
        if let Some((_, bufs)) = self.buffers.range_mut(len..).next() {
            if let Some(buf) = bufs.pop() { return buf; }
        }
        Vec::with_capacity(len)
    }

    fn return_buf(&mut self, buf: Vec<T>) {
        self.buffers.entry(buf.capacity()).or_default().push(buf);
    }
}
```

Pass pool as argument to `execute_tree`:
```rust
fn execute_tree<Alg, Backend>(
    ctx: &mut Backend::Context,
    tree: &ContractionTree,
    operands: &[&Tensor<Alg::Scalar>],
    pool: &mut BufferPool<Alg::Scalar>,  // new parameter
    ...
)
```

### Impact

Eliminates TLS + TypeId + downcast + RefCell overhead. O(log n) vs O(n) lookup.

---

## Layer 5: Uninit Buffers + Consuming API

### Problem

- `Tensor::zeros` zeroes all elements, but intermediates with beta=0 are fully overwritten
- `return_tensor_to_pool` must extract the Vec from Tensor (requires `Tensor::into_data()`)

### Changes

**Uninit allocation** (already partially done in `take_from_pool` via `set_len`):
```rust
fn alloc_tensor_uninit<T: Scalar>(dims: &[usize], pool: &mut BufferPool<T>) -> Tensor<T> {
    let numel = dims.iter().product::<usize>().max(1);
    let mut data = pool.take(numel);
    unsafe { data.set_len(numel); }
    Tensor::from_vec_unchecked(data, dims.into(), compute_strides(dims))
}
```

**Consuming API**:
```rust
impl<T: Scalar> Tensor<T> {
    /// Consume tensor and return owned data buffer.
    /// Panics if buffer is shared (Arc refcount > 1).
    pub fn into_data(self) -> Vec<T> {
        self.buffer.into_inner()
    }
}
```

### Impact

Eliminates zeroing cost for intermediate allocations. Enables zero-copy buffer pool return.

---

## Execution Path Comparison

### Before (per step, Gemm path):
```
prepare_one_operand(A)     → plan(Permute/MakeContiguous) + execute
prepare_one_operand(B)     → plan(Permute/MakeContiguous) + execute
alloc_tensor_pooled        → TLS + TypeId + downcast + linear scan
plan(BatchedGemm)          → Tensor→StridedView ×3
execute(BatchedGemm)       → Tensor→StridedView ×3 + faer matmul
plan(Permute) + execute    → Tensor→StridedView ×2 (if final permute needed)
Total: 4-6 plan/execute round-trips, ~12-18 Tensor→View conversions
```

### After (per step, Contract path):
```
plan(Contract)             → cached spec lookup (Layer 2)
execute(Contract)          → Tensor→View ×3 + try_execute_contract_gemm
  └→ use cached mode spec  → fuse check + faer matmul
alloc_tensor (intermediate) → typed pool O(log n) lookup (Layer 4)
Total: 1 plan + 1 execute, 3 Tensor→View conversions
```

---

## Validation Strategy

| Instance | Steps | Before (ms) | Target |
|----------|-------|-------------|--------|
| gm_queen5_5_3 (opt_flops) | 159 | 6942 | <3000 |
| gm_queen5_5_3 (opt_size) | 159 | 6949 | <3000 |
| lm_brackets_4_4d | 83 | 3428 | <2000 |
| lm_sentence_4_4d | 83 | 3457 | <2000 |
| bin_matmul_256 | 1 | 16.6 | no regression |

```bash
# Quick validation (single instance)
cd ../tenferro-einsum-benchmark
BENCH_INSTANCE=gm_queen5_5_3.wcsp RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo run --release

# Full suite
./scripts/run_all.sh 1
```

---

## Implementation Order

| Layer | Description | Dependencies | Risk |
|-------|-------------|-------------|------|
| 1 | Contract extension preferred path | None | Low (code exists) |
| 2 | Contract plan caching | Layer 1 | Medium |
| 3 | Arc dims/strides | None (parallel) | Medium (wide change) |
| 4 | Typed buffer pool | None (parallel) | Low |
| 5 | Uninit + consuming API | Layer 3, 4 | Low |

Layers 1-2 are sequential. Layers 3, 4 can be done in parallel. Layer 5 depends on 3+4.
