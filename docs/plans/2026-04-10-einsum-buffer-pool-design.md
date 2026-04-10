# Einsum Buffer Pool for CpuBackend

> **IMPORTANT**: Do NOT auto-implement this design. An agent must discuss the
> plan with a human reviewer and get explicit approval before writing any code.
> The pseudo-code below is illustrative — the actual implementation must be
> verified against the current codebase state.

## Problem

During N-ary einsum execution, the CPU backend allocates fresh `Vec<T>` buffers
for every intermediate result:

- `dot_general` output: `vec![T::zero(); out_n]` (gemm/mod.rs:411, 508)
- `canonical_gemm_layout` fallback: `typed_transpose` allocates a copy buffer
  per operand (gemm/mod.rs:370, 375, 463, 468)

These allocations hit the system allocator on every einsum step. For repeated
evaluations (e.g., optimization loops, AD backward passes), the same buffer
sizes are allocated and freed repeatedly.

### Current state

tenferro-rs has a `BufferPool` at `tenferro/src/buffer_pool.rs`:
- `BufferPool` stores `Vec<(usize, Vec<u8>)>` — capacity-keyed, no alignment.
- `eval_exec_ir` (tenferro/src/exec.rs:168) receives `pool: &mut BufferPool`.
- After each instruction, `reclaim_last_use_inputs` extracts `Vec<u8>` from
  dead `TypedTensor` slots via `extract_host_bytes` and returns them to pool.
- **Pool is reclaim-only**: no backend method ever calls `pool.allocate()`.
- Pool lives in `Engine` (tenferro crate), not accessible from `tenferro-tensor`.

### strided-rs reference

strided-rs uses a thread-local `BufferPool` (`BTreeMap<usize, Vec<Vec<T>>>`)
with `pool_acquire(pool, &dims)` / `pool_release`. Intermediates are allocated
from the pool and returned after each binary einsum step.

## Design

### Architecture overview

```
tenferro (crate)
  Engine { backend: CpuBackend, ... }
    eval_exec_ir(backend, program, inputs)
      ├── ExecOp::BatchedGemm → backend.dot_general(lhs, rhs, config)
      │     └── CpuBackend::dot_general
      │           └── gemm::dot_general_with_pool(&mut self.buffers, lhs, rhs, config)
      │                 ├── typed_faer_gemm_with_pool(pool, ...) → output from pool
      │                 └── canonical fallback → copy buffer from pool
      ├── ExecOp::Permute → backend.transpose(input, perm)
      │     └── structural::transpose → output from pool
      └── reclaim_last_use_inputs → backend.buffers.release(...)

tenferro-tensor (crate)
  buffer_pool.rs        ← NEW: moved from tenferro, with alignment support
  cpu/backend.rs        ← CpuBackend { pool, buffers }
  cpu/gemm/mod.rs       ← pool-aware GEMM functions
```

### Step 1: Create `BufferPool` in `tenferro-tensor`

New file: `tenferro-tensor/src/buffer_pool.rs`

```rust
use std::collections::BTreeMap;

/// Alignment-aware buffer pool for reusing heap allocations.
///
/// Buffers are keyed by alignment so that an f64 buffer (align 8) is never
/// handed to an f32 request (align 4). Within each alignment bin, best-fit
/// allocation selects the smallest buffer with sufficient capacity.
pub struct BufferPool {
    /// alignment → Vec<(capacity_bytes, buffer)>
    bins: BTreeMap<usize, Vec<(usize, Vec<u8>)>>,
}
```

**Public API:**

```rust
impl BufferPool {
    pub fn new() -> Self;

    /// Acquire a raw buffer with at least `size_bytes` and correct `align`.
    /// Returns Vec<u8> with len == size_bytes (zero-filled on fresh alloc,
    /// resized on reuse). Reuses best-fit from pool if available.
    pub fn acquire(&mut self, size_bytes: usize, align: usize) -> Vec<u8>;

    /// Return a raw buffer to the pool.
    pub fn release(&mut self, buf: Vec<u8>, align: usize);

    /// Acquire a typed Vec<T> with `len` elements, zero-initialized.
    pub fn acquire_vec<T: Copy + num_traits::Zero>(&mut self, len: usize) -> Vec<T>;

    /// Return a typed Vec<T> to the pool.
    pub fn release_vec<T>(&mut self, vec: Vec<T>);

    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}
```

**`acquire` implementation (best-fit):**

```rust
pub fn acquire(&mut self, size_bytes: usize, align: usize) -> Vec<u8> {
    if let Some(bin) = self.bins.get_mut(&align) {
        // Find smallest buffer with capacity >= size_bytes
        let best = bin.iter().enumerate()
            .filter(|(_, (cap, _))| *cap >= size_bytes)
            .min_by_key(|(_, (cap, _))| *cap)
            .map(|(idx, _)| idx);
        if let Some(idx) = best {
            let (_, mut buf) = bin.swap_remove(idx);
            buf.resize(size_bytes, 0);
            return buf;
        }
    }
    vec![0u8; size_bytes]
}
```

**`acquire_vec<T>` / `release_vec<T>` — safe typed wrappers:**

```rust
pub fn acquire_vec<T: Copy + num_traits::Zero>(&mut self, len: usize) -> Vec<T> {
    let byte_len = len * std::mem::size_of::<T>();
    let align = std::mem::align_of::<T>();
    let raw = self.acquire(byte_len, align);
    let mut typed = raw_to_typed::<T>(raw);
    typed.resize(len, T::zero());
    typed
}

pub fn release_vec<T>(&mut self, vec: Vec<T>) {
    let align = std::mem::align_of::<T>();
    let raw = typed_to_raw(vec);
    self.release(raw, align);
}
```

**Vec<u8> ↔ Vec<T> conversion (unsafe helpers, private):**

```rust
/// Convert Vec<u8> to Vec<T>. The input must have been allocated with
/// compatible alignment (guaranteed by acquire keying on align).
fn raw_to_typed<T>(mut raw: Vec<u8>) -> Vec<T> {
    let ptr = raw.as_mut_ptr();
    let byte_cap = raw.capacity();
    let byte_len = raw.len();
    std::mem::forget(raw);

    let elem_size = std::mem::size_of::<T>();
    // Safety: alignment is guaranteed by pool keying.
    // Vec<u8> allocated with align >= align_of::<T>().
    // However, the standard allocator aligns Vec<u8> to 1.
    // We must use Layout-aware allocation. See note below.
    unsafe {
        Vec::from_raw_parts(
            ptr as *mut T,
            byte_len / elem_size,
            byte_cap / elem_size,
        )
    }
}

fn typed_to_raw<T>(mut vec: Vec<T>) -> Vec<u8> {
    let ptr = vec.as_mut_ptr();
    let len = vec.len();
    let cap = vec.capacity();
    std::mem::forget(vec);

    let elem_size = std::mem::size_of::<T>();
    unsafe {
        Vec::from_raw_parts(
            ptr as *mut u8,
            len * elem_size,
            cap * elem_size,
        )
    }
}
```

**CRITICAL: Alignment safety**

`Vec<u8>` allocated by the standard allocator has alignment 1, which is
insufficient for `f64` (align 8) or `Complex64` (align 8). Two options:

**(A) Always allocate as `Vec<T>` and store as `Vec<u8>` only for pool storage.**
The pool receives buffers that were originally `Vec<T>` (via `release_vec`),
so their allocation alignment matches `T`. Fresh allocations in `acquire_vec`
create `Vec<T>` directly, then convert.

```rust
pub fn acquire_vec<T: Copy + num_traits::Zero>(&mut self, len: usize) -> Vec<T> {
    let byte_len = len * std::mem::size_of::<T>();
    let align = std::mem::align_of::<T>();
    if let Some(raw) = self.try_acquire(byte_len, align) {
        // raw was originally a Vec<T>, alignment preserved
        let mut typed = raw_to_typed::<T>(raw);
        typed.resize(len, T::zero());
        typed
    } else {
        // Fresh allocation as Vec<T> — correct alignment guaranteed
        vec![T::zero(); len]
    }
}
```

**(B) Use `std::alloc::Layout`-aware allocation for fresh buffers.**

Option (A) is simpler and sufficient — the pool only reuses buffers that
originated as `Vec<T>`. This is the recommended approach.

### Step 2: Add `BufferPool` to `CpuBackend`

```rust
// tenferro-tensor/src/cpu/backend.rs

pub struct CpuBackend {
    pub(crate) pool: Arc<rayon::ThreadPool>,     // rayon thread pool (existing)
    pub(crate) buffers: BufferPool,              // NEW: buffer pool
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            pool: get_or_create_pool(default_threads()),
            buffers: BufferPool::new(),
        }
    }

    pub fn with_threads(num_threads: usize) -> Self {
        Self {
            pool: get_or_create_pool(num_threads),
            buffers: BufferPool::new(),
        }
    }
}
```

### Step 3: Pool-aware `dot_general`

**Problem**: The current `dot_general` impl dispatches through
`self.install(|| gemm::dot_general(a, b, config))`. The `install` method runs
the closure on the rayon thread pool, requiring `Send`. `&mut BufferPool` is
not `Send`-safe across rayon tasks.

**Solution**: Extract pool-using code outside the rayon `install` closure, or
pass pool into the closure (BufferPool is `Send` if we own it, but `&mut` refs
across rayon boundaries require care).

Simplest approach: **do not use `install` for `dot_general`**. The GEMM kernel
(faer/BLAS) already manages its own parallelism internally. The rayon pool
in `install` is for strided-kernel operations. For `dot_general`, run on the
calling thread with direct `&mut self.buffers` access:

```rust
// tenferro-tensor/src/cpu/backend.rs

fn dot_general(
    &mut self,
    lhs: &Tensor,
    rhs: &Tensor,
    config: &DotGeneralConfig,
) -> crate::Result<Tensor> {
    match (lhs, rhs) {
        (Tensor::F64(a), Tensor::F64(b)) => {
            gemm::dot_general_pooled(&mut self.buffers, a, b, config).map(Tensor::F64)
        }
        (Tensor::F32(a), Tensor::F32(b)) => {
            gemm::dot_general_pooled(&mut self.buffers, a, b, config).map(Tensor::F32)
        }
        (Tensor::C64(a), Tensor::C64(b)) => {
            gemm::dot_general_pooled(&mut self.buffers, a, b, config).map(Tensor::C64)
        }
        (Tensor::C32(a), Tensor::C32(b)) => {
            gemm::dot_general_pooled(&mut self.buffers, a, b, config).map(Tensor::C32)
        }
        _ => Err(crate::Error::DTypeMismatch { ... }),
    }
}
```

**New function in `gemm/mod.rs`:**

```rust
pub(crate) fn dot_general_pooled<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> crate::Result<TypedTensor<T>>
where
    T: FaerGemm + Copy + Clone + Zero + One + PartialEq,
{
    validate_dot_general(lhs, rhs, config)?;

    // Fast path: try strided GEMM (no copy needed)
    if let Some(result) = typed_faer_gemm_pooled(buffers, lhs, rhs, config) {
        return Ok(result);
    }

    // Slow path: canonical layout fallback
    let (lhs_perm, rhs_perm, new_config) =
        canonical_gemm_layout(config, lhs.shape.len(), rhs.shape.len());

    let lhs_t;
    let lhs_ref = if is_identity_perm(&lhs_perm) {
        lhs
    } else {
        lhs_t = typed_transpose(lhs, &lhs_perm)?;  // TODO: pool this too
        &lhs_t
    };

    let rhs_t;
    let rhs_ref = if is_identity_perm(&rhs_perm) {
        rhs
    } else {
        rhs_t = typed_transpose(rhs, &rhs_perm)?;  // TODO: pool this too
        &rhs_t
    };

    typed_faer_gemm_pooled(buffers, lhs_ref, rhs_ref, &new_config)
        .ok_or_else(|| Error::BackendFailure { ... })
}
```

**Output buffer from pool:**

```rust
fn typed_faer_gemm_pooled<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> Option<TypedTensor<T>>
where
    T: FaerGemm + Copy + Clone + Zero + One + PartialEq,
{
    let dims = analyse_gemm(lhs, rhs, config)?;
    let out_n: usize = dims.out_shape.iter().product();

    // === CHANGE: use pool instead of vec![T::zero(); out_n] ===
    let mut out_data: Vec<T> = buffers.acquire_vec(out_n);

    // ... (existing GEMM logic unchanged) ...

    Some(TypedTensor {
        buffer: Buffer::Host(out_data),
        shape: dims.out_shape,
        placement: lhs.placement.clone(),
    })
}
```

### Step 4: Reclaim path in `eval_exec_ir`

**Current** (tenferro/src/exec.rs):
- `eval_exec_ir` takes `pool: &mut BufferPool` (engine-level pool)
- `reclaim_last_use_inputs` calls `extract_host_bytes` → `pool.return_buffer`
- `extract_host_bytes` converts `Vec<T>` → `Vec<u8>` via unsafe ptr reinterpret

**Change**: Replace engine-level pool with backend's pool.

Option A — Access through `TensorBackend` trait (adds method):
```rust
pub trait TensorBackend {
    // ... existing methods ...
    fn reclaim_buffer(&mut self, tensor: Tensor) { /* default: drop */ }
}

impl TensorBackend for CpuBackend {
    fn reclaim_buffer(&mut self, tensor: Tensor) {
        reclaim_tensor_into_pool(tensor, &mut self.buffers);
    }
}
```

Option B — Keep engine-level pool as pass-through (minimal change):
```rust
// eval_exec_ir still takes pool: &mut BufferPool
// Engine::buffer_pool now delegates to backend.buffers
// Requires BufferPool to be the same type in both crates
```

**Recommended: Option A** — cleaner separation. The `TensorBackend` trait gets
one new method with a default no-op implementation. GPU backends ignore it.
`eval_exec_ir` calls `backend.reclaim_buffer(tensor)` instead of
`pool.return_buffer(extract_host_bytes(tensor))`.

```rust
// tenferro/src/exec.rs — CHANGED

pub fn eval_exec_ir<B: TensorBackend>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
    // pool parameter REMOVED
) -> Result<Vec<Tensor>> {
    // ...
    for inst in &program.instructions {
        let result = match &inst.op { /* ... unchanged ... */ };
        slots[inst.output_slots[0]] = Some(result);
        reclaim_last_use_inputs(&mut slots, inst, backend);
    }
    // ...
}

fn reclaim_last_use_inputs<B: TensorBackend>(
    slots: &mut [Option<Tensor>],
    inst: &ExecInstruction,
    backend: &mut B,
) {
    for (i, &is_last) in inst.last_use.iter().enumerate() {
        if is_last {
            if let Some(tensor) = slots[inst.input_slots[i]].take() {
                backend.reclaim_buffer(tensor);
            }
        }
    }
}
```

### Step 5: Remove engine-level `BufferPool`

- Delete `tenferro/src/buffer_pool.rs`
- Remove `buffer_pool` field from `Engine`
- Remove `buffer_pool_len()` public API (or delegate to backend)
- Update all callers of `eval_exec_ir` to drop the pool parameter

**Files affected:**
- `tenferro/src/lib.rs` — remove `pub mod buffer_pool`
- `tenferro/src/engine.rs` — remove `buffer_pool` field, update `eval` calls
- `tenferro/src/traced.rs` — update `eval_exec_ir` call sites
- `tenferro/tests/cpu_backend.rs` — update pool tests

## Scope

**In scope:**
- `BufferPool` in `tenferro-tensor` with alignment-aware bins
- `CpuBackend.buffers` field
- Pool-aware `dot_general` (output buffer from pool)
- `TensorBackend::reclaim_buffer` for reclaim path
- Remove engine-level `BufferPool`

**Out of scope (future work):**
- Pool for `canonical_gemm_layout` copy buffers (the `typed_transpose` calls
  inside the fallback path — marked with TODO above)
- Pool for `ExecOp::Permute` (transpose in exec IR)
- Linalg operations (svd, qr, cholesky, etc.)
- Elementwise and reduction operations
- GPU backends

## Verification

### Correctness
- All existing `cargo test --workspace --release` must pass.
- Oracle replay tests verify numeric results are unchanged.

### Pool behavior tests (new)
- `buffer_pool::tests::acquire_release_reuse` — acquire, release, re-acquire
  returns same capacity buffer.
- `buffer_pool::tests::alignment_separation` — f64 buffer (align 8) not reused
  for f32 request (align 4).
- `buffer_pool::tests::best_fit` — pool with [100, 200, 300] byte buffers;
  request for 150 returns the 200-byte buffer.
- `buffer_pool::tests::fresh_alloc_fallback` — when pool is empty, fresh
  allocation succeeds.
- Integration: repeated `einsum("bij,bjk,bkl->bil", A, B, C)` calls; after
  first call, `backend.buffers.len() > 0` (buffers recycled).

### Performance (manual/benchmark)
- Steady-state: no new allocations after first einsum evaluation.
- Repeated N-ary einsum should show reduced allocation overhead.

## Migration checklist

1. Create `tenferro-tensor/src/buffer_pool.rs` with `BufferPool`
2. Add `buffers: BufferPool` to `CpuBackend`
3. Add `dot_general_pooled` in `gemm/mod.rs`
4. Wire `CpuBackend::dot_general` to use `dot_general_pooled`
5. Add `reclaim_buffer` to `TensorBackend` trait (default no-op)
6. Implement `reclaim_buffer` for `CpuBackend`
7. Update `eval_exec_ir` to use `backend.reclaim_buffer` instead of pool param
8. Remove `tenferro/src/buffer_pool.rs` and engine-level pool
9. Update all call sites and tests
10. Add new pool unit tests and integration test
