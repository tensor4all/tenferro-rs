# Einsum Buffer Pool for CpuBackend

> **IMPORTANT**: Do NOT auto-implement this design. An agent must discuss the
> plan with a human reviewer and get explicit approval before writing any code.
> The pseudo-code below is illustrative — the actual implementation must be
> verified against the current codebase state.

## Problem

During N-ary einsum execution, the CPU backend allocates fresh `Vec<T>` buffers
for every intermediate result:

- `dot_general` output: `vec![T::zero(); out_n]`
- `canonical_gemm_layout` fallback: `typed_transpose` allocates a new buffer for
  each operand that needs canonicalization

These allocations hit the system allocator on every einsum step. For repeated
evaluations (e.g., inside optimization loops or AD backward passes), the same
size buffers are allocated and freed repeatedly.

strided-rs solves this with a thread-local `BufferPool` that reuses freed
intermediate buffers via best-fit allocation. tenferro-rs has a `BufferPool` at
the `tenferro` crate level (`tenferro/src/buffer_pool.rs`) that reclaims slot
buffers after liveness analysis in `eval_exec_ir`, but:

1. The pool is not accessible from `tenferro-tensor` (the backend crate).
2. Backend methods (`dot_general`, `transpose`, etc.) never draw from the pool.
3. The pool stores `Vec<u8>` without alignment awareness.

## Proposed Changes

### Step 1: Move `BufferPool` to `tenferro-tensor`

Move the existing `BufferPool` from `tenferro/src/buffer_pool.rs` to
`tenferro-tensor/src/buffer_pool.rs`. Extend it with alignment-keyed storage:

```rust
// tenferro-tensor/src/buffer_pool.rs

use std::collections::BTreeMap;

/// Alignment-aware buffer pool for reusing heap allocations.
///
/// Buffers are keyed by `(alignment, capacity)` so that an `f64` buffer
/// (align 8) is never handed to an `f32` request (align 4).
pub struct BufferPool {
    /// align -> Vec<(capacity_bytes, buffer)>, sorted by capacity
    pools: BTreeMap<usize, Vec<(usize, Vec<u8>)>>,
}

impl BufferPool {
    pub fn new() -> Self { ... }

    /// Acquire a buffer with at least `size_bytes` bytes and the given alignment.
    /// Returns a `Vec<u8>` with `len == 0` and `capacity >= size_bytes`.
    /// Reuses a pooled buffer if one of matching alignment and sufficient
    /// capacity exists (best-fit); otherwise allocates a new one.
    pub fn acquire(&mut self, size_bytes: usize, align: usize) -> Vec<u8> { ... }

    /// Return a buffer to the pool for future reuse.
    pub fn release(&mut self, buf: Vec<u8>, align: usize) { ... }

    /// Number of buffers currently held across all alignments.
    pub fn len(&self) -> usize { ... }

    pub fn is_empty(&self) -> bool { ... }
}
```

Also provide typed convenience helpers:

```rust
impl BufferPool {
    /// Acquire a `Vec<T>` with capacity for `len` elements.
    /// The returned vector has `len` elements, all zero-initialized.
    pub fn acquire_vec<T: Copy + Default>(&mut self, len: usize) -> Vec<T> {
        let size = len * std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        let buf = self.acquire(size, align);
        // Safety: reinterpret Vec<u8> as Vec<T> (same layout, correct alignment)
        unsafe { transmute_vec(buf, len) }
    }

    /// Return a `Vec<T>` to the pool.
    pub fn release_vec<T>(&mut self, vec: Vec<T>) {
        let align = std::mem::align_of::<T>();
        let buf = unsafe { transmute_vec_back(vec) };
        self.release(buf, align);
    }
}
```

### Step 2: Add `BufferPool` to `CpuBackend`

```rust
// tenferro-tensor/src/cpu/backend.rs

pub struct CpuBackend {
    pub(crate) pool: Arc<rayon::ThreadPool>,
    pub(crate) buffers: BufferPool,
}
```

`CpuBackend::new()` initializes an empty `BufferPool`.

### Step 3: Use pool in `dot_general`

In `tenferro-tensor/src/cpu/gemm/mod.rs`, the GEMM functions currently receive
`&TypedTensor<T>` references. They need access to the pool for:

1. **Output buffer**: Replace `vec![T::zero(); out_n]` with `buffers.acquire_vec::<T>(out_n)`.
2. **Canonical fallback copy buffers**: When `canonical_gemm_layout` triggers
   `typed_transpose`, the transpose output buffer should come from the pool.

Since `dot_general` is called via `TensorBackend::dot_general(&mut self, ...)`
which has `&mut self` access to `CpuBackend`, the pool is accessible.

The internal helper functions (`typed_faer_gemm`, `typed_blas_gemm`,
`canonical_gemm_layout` fallback path) need to receive `&mut BufferPool` as an
additional parameter.

### Step 4: Reclaim buffers after slot liveness

In `tenferro/src/exec.rs`, `reclaim_last_use_inputs` currently calls
`pool.return_buffer()` on the `tenferro`-level pool. After the move:

- `eval_exec_ir` should call `backend.buffers.release_vec()` (or equivalent)
  instead of the separate engine-level pool.
- Alternatively, keep the engine-level pool as a thin wrapper that delegates to
  `backend.buffers`. This avoids changing the `eval_exec_ir` signature.

### Step 5: Update `tenferro` crate

- Remove `tenferro/src/buffer_pool.rs` (moved to `tenferro-tensor`).
- Update `Engine` to use the backend's pool, or keep a facade that delegates.
- Update `eval_exec_ir` reclaim path.

## Scope

**In scope (this issue):**
- Buffer pool for `dot_general` output and canonical fallback buffers.
- Buffer pool for `ExecOp::Permute` (physical transpose in exec IR).

**Out of scope (future work):**
- Linalg operations (svd, qr, cholesky, etc.) — these allocate internally in
  `faer_linalg.rs` and would need similar plumbing.
- Elementwise and reduction operations — typically small or in-place.
- GPU backends — different allocation model.

## Verification

### Correctness
- All existing `cargo test --workspace --release` must pass.
- Oracle replay tests verify numeric results are unchanged.

### Pool behavior tests
- Unit test: acquire, release, re-acquire returns same capacity buffer.
- Unit test: alignment separation (f64 buffer not reused for f32).
- Integration test: repeated `einsum` calls show `buffers.len() > 0` after
  first call (buffers are being recycled).

### Performance
- No new allocations after the first einsum evaluation with a given set of
  intermediate sizes (steady-state pool).
- Benchmark: repeated N-ary einsum should show reduced allocation overhead.
