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
sizes are allocated and freed repeatedly. Furthermore, `vec![T::zero(); n]`
zero-initializes the entire buffer, which is wasteful since GEMM with
`Accum::Replace` (beta=0) overwrites every element.

### Current state

tenferro-rs has a `BufferPool` at `tenferro/src/buffer_pool.rs`:
- Stores `Vec<(usize, Vec<u8>)>` — type-erased, no alignment awareness.
- `eval_exec_ir` (tenferro/src/exec.rs:168) receives `pool: &mut BufferPool`.
- After each instruction, `reclaim_last_use_inputs` extracts `Vec<u8>` from
  dead `TypedTensor` slots via `extract_host_bytes` and returns them to pool.
- **Pool is reclaim-only**: no backend method ever calls `pool.allocate()`.
- Pool lives in `Engine` (tenferro crate), not accessible from `tenferro-tensor`.
- `Vec<u8>` ↔ `Vec<T>` conversion requires unsafe alignment handling.

### strided-rs reference

strided-rs uses **typed pools** — no type-erasure, no alignment problem:

```rust
pub struct BufferPool {
    f64_pool: BTreeMap<usize, Vec<Vec<f64>>>,
    c64_pool: BTreeMap<usize, Vec<Vec<Complex64>>>,
}
```

Key properties:
- `pool_acquire` returns **uninitialized** buffers (`col_major_uninit`).
  Safety: `einsum2_into` with `beta=0` writes every output element before reading.
- `pool_release` returns `Vec<T>` directly — no type-erasure.
- Best-fit allocation via `BTreeMap::range(total..)`.
- No zero-initialization overhead.

## Design

### Architecture overview

```
tenferro (crate)
  Engine { backend: CpuBackend, ... }
    eval_exec_ir(backend, program, inputs)
      ├── ExecOp::BatchedGemm → backend.dot_general(lhs, rhs, config)
      │     └── CpuBackend::dot_general
      │           └── install_with_pool(|buffers| gemm::dot_general_pooled(buffers, ...))
      │                 ├── typed_faer_gemm_pooled(pool, ...) → output from pool (uninit)
      │                 └── canonical fallback → copy buffer from pool
      └── reclaim_last_use_inputs → backend.reclaim_buffer(tensor)

tenferro-tensor (crate)
  buffer_pool.rs        ← NEW: typed per-dtype pools, no Vec<u8>
  cpu/backend.rs        ← CpuBackend { pool, buffers }
  cpu/gemm/mod.rs       ← pool-aware GEMM functions
```

### Step 1: Create typed `BufferPool` in `tenferro-tensor`

New file: `tenferro-tensor/src/buffer_pool.rs`

Following strided-rs's design — typed pools, no type-erasure, no alignment issues.

```rust
use std::collections::BTreeMap;
use num_complex::{Complex32, Complex64};

/// Typed buffer pool for reusing tensor allocations without type-erasure.
///
/// Each scalar type (f64, f32, Complex64, Complex32) has its own pool,
/// keyed by element count. Buffers are returned **uninitialized** — callers
/// must write every element before reading.
///
/// Design follows strided-rs's BufferPool: typed storage, best-fit via BTreeMap,
/// no Vec<u8> conversion, no alignment concerns.
pub struct BufferPool {
    f64_pool: BTreeMap<usize, Vec<Vec<f64>>>,
    f32_pool: BTreeMap<usize, Vec<Vec<f32>>>,
    c64_pool: BTreeMap<usize, Vec<Vec<Complex64>>>,
    c32_pool: BTreeMap<usize, Vec<Vec<Complex32>>>,
}
```

**Public API via sealed trait:**

```rust
/// Sealed trait for typed buffer pool access. Implemented for f64, f32,
/// Complex64, Complex32.
pub trait PoolScalar: Copy + Sized + Send + private::Sealed {
    /// Acquire a Vec<Self> with `len` elements from the pool.
    /// Contents are UNDEFINED. Caller must write every element before reading.
    ///
    /// # Safety
    /// The returned vector has length set to `len` but contents are
    /// uninitialized. This is sound for Copy types (no drop glue),
    /// but reading before writing is undefined behavior.
    unsafe fn pool_acquire(pool: &mut BufferPool, len: usize) -> Vec<Self>;

    /// Return a Vec<Self> to the pool for future reuse.
    fn pool_release(pool: &mut BufferPool, buf: Vec<Self>);
}

mod private {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for f32 {}
    impl Sealed for num_complex::Complex64 {}
    impl Sealed for num_complex::Complex32 {}
}
```

**Implementation for each type (macro):**

```rust
macro_rules! impl_pool_scalar {
    ($ty:ty, $field:ident) => {
        impl PoolScalar for $ty {
            unsafe fn pool_acquire(pool: &mut BufferPool, len: usize) -> Vec<$ty> {
                match take_best_fit(&mut pool.$field, len) {
                    Some(mut buf) => {
                        buf.set_len(len);  // no zero-fill
                        buf
                    }
                    None => {
                        let mut buf = Vec::with_capacity(len);
                        buf.set_len(len);  // no zero-fill
                        buf
                    }
                }
            }

            fn pool_release(pool: &mut BufferPool, buf: Vec<$ty>) {
                let cap = buf.capacity();
                if cap > 0 {
                    pool.$field.entry(cap).or_default().push(buf);
                }
            }
        }
    };
}

impl_pool_scalar!(f64, f64_pool);
impl_pool_scalar!(f32, f32_pool);
impl_pool_scalar!(Complex64, c64_pool);
impl_pool_scalar!(Complex32, c32_pool);
```

**Best-fit helper (same as strided-rs):**

```rust
fn take_best_fit<T>(pool: &mut BTreeMap<usize, Vec<Vec<T>>>, len: usize) -> Option<Vec<T>> {
    let key = *pool.range(len..).next()?.0;
    let vecs = pool.get_mut(&key)?;
    let buf = vecs.pop();
    if vecs.is_empty() {
        pool.remove(&key);
    }
    buf
}
```

**Other methods:**

```rust
impl BufferPool {
    pub fn new() -> Self {
        Self {
            f64_pool: BTreeMap::new(),
            f32_pool: BTreeMap::new(),
            c64_pool: BTreeMap::new(),
            c32_pool: BTreeMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.f64_pool.values().map(|v| v.len()).sum::<usize>()
            + self.f32_pool.values().map(|v| v.len()).sum::<usize>()
            + self.c64_pool.values().map(|v| v.len()).sum::<usize>()
            + self.c32_pool.values().map(|v| v.len()).sum::<usize>()
    }

    pub fn is_empty(&self) -> bool {
        self.f64_pool.is_empty()
            && self.f32_pool.is_empty()
            && self.c64_pool.is_empty()
            && self.c32_pool.is_empty()
    }
}

impl Default for BufferPool {
    fn default() -> Self { Self::new() }
}
```

### Step 2: Add `BufferPool` to `CpuBackend`

```rust
// tenferro-tensor/src/cpu/backend.rs

pub struct CpuBackend {
    pub(crate) pool: Arc<rayon::ThreadPool>,     // rayon thread pool (existing)
    pub(crate) buffers: BufferPool,              // NEW: typed buffer pool
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

### Step 3: `install_with_pool` helper

The existing `self.install(|| ...)` runs closures on the rayon thread pool
via `self.pool.install(op)`, which is required for faer's `Par::rayon(0)` to
use the correct thread pool. However, `&mut BufferPool` cannot be passed into
the `Send`-requiring closure.

Solution: temporarily move `BufferPool` into the closure via `std::mem::take`:

```rust
impl CpuBackend {
    /// Run `op` on the rayon thread pool with mutable access to the buffer pool.
    /// BufferPool is temporarily moved into the closure (Send) and moved back.
    fn install_with_pool<R: Send>(
        &mut self,
        op: impl FnOnce(&mut BufferPool) -> R + Send,
    ) -> R {
        let mut buffers = std::mem::take(&mut self.buffers);
        let (result, returned) = self.pool.install(|| {
            let r = op(&mut buffers);
            (r, buffers)
        });
        self.buffers = returned;
        result
    }
}
```

This preserves the rayon thread pool context (so `Par::rayon(0)` in faer uses
the correct pool) while allowing `&mut BufferPool` access inside the closure.

Operations that don't need the pool continue to use `self.install(|| ...)`.

### Step 4: Pool-aware `dot_general`

```rust
// tenferro-tensor/src/cpu/backend.rs

fn dot_general(
    &mut self,
    lhs: &Tensor,
    rhs: &Tensor,
    config: &DotGeneralConfig,
) -> crate::Result<Tensor> {
    self.install_with_pool(|buffers| match (lhs, rhs) {
        (Tensor::F64(a), Tensor::F64(b)) =>
            gemm::dot_general_pooled(buffers, a, b, config).map(Tensor::F64),
        (Tensor::F32(a), Tensor::F32(b)) =>
            gemm::dot_general_pooled(buffers, a, b, config).map(Tensor::F32),
        (Tensor::C64(a), Tensor::C64(b)) =>
            gemm::dot_general_pooled(buffers, a, b, config).map(Tensor::C64),
        (Tensor::C32(a), Tensor::C32(b)) =>
            gemm::dot_general_pooled(buffers, a, b, config).map(Tensor::C32),
        _ => Err(crate::Error::DTypeMismatch { ... }),
    })
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
    T: FaerGemm + PoolScalar + Copy + Clone + Zero + One + PartialEq,
{
    validate_dot_general(lhs, rhs, config)?;

    if let Some(result) = typed_faer_gemm_pooled(buffers, lhs, rhs, config) {
        return Ok(result);
    }

    // Canonical fallback
    let (lhs_perm, rhs_perm, new_config) =
        canonical_gemm_layout(config, lhs.shape.len(), rhs.shape.len());
    let lhs_t;
    let lhs_ref = if is_identity_perm(&lhs_perm) {
        lhs
    } else {
        lhs_t = typed_transpose(lhs, &lhs_perm)?;
        &lhs_t
    };
    let rhs_t;
    let rhs_ref = if is_identity_perm(&rhs_perm) {
        rhs
    } else {
        rhs_t = typed_transpose(rhs, &rhs_perm)?;
        &rhs_t
    };

    typed_faer_gemm_pooled(buffers, lhs_ref, rhs_ref, &new_config)
        .ok_or_else(|| Error::BackendFailure {
            op: "dot_general",
            message: "CPU GEMM requires host-backed canonical inputs".into(),
        })
}
```

**Output buffer from pool (uninit):**

```rust
fn typed_faer_gemm_pooled<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> Option<TypedTensor<T>>
where
    T: FaerGemm + PoolScalar + Copy + Clone + Zero + One + PartialEq,
{
    let dims = analyse_gemm(lhs, rhs, config)?;
    let out_n: usize = dims.out_shape.iter().product();

    if dims.m == 0 || dims.n == 0 || dims.k == 0 || dims.batch_total == 0 {
        return Some(TypedTensor {
            buffer: Buffer::Host(vec![T::zero(); out_n]),
            shape: dims.out_shape,
            placement: lhs.placement.clone(),
        });
    }

    // SAFETY: GEMM with Accum::Replace (beta=0) writes every output element.
    let mut out_data: Vec<T> = unsafe { T::pool_acquire(buffers, out_n) };

    // ... existing GEMM kernel (unchanged) ...

    Some(TypedTensor {
        buffer: Buffer::Host(out_data),
        shape: dims.out_shape,
        placement: lhs.placement.clone(),
    })
}
```

### Step 5: Reclaim path in `eval_exec_ir`

Add `reclaim_buffer` to `TensorBackend` trait (default no-op for GPU backends):

```rust
// tenferro-tensor/src/backend.rs

pub trait TensorBackend {
    // ... existing methods ...

    /// Reclaim a tensor's buffer for potential reuse.
    /// Default implementation drops the tensor. CPU backend returns buffer to pool.
    fn reclaim_buffer(&mut self, _tensor: Tensor) {}
}
```

```rust
// tenferro-tensor/src/cpu/backend.rs

impl TensorBackend for CpuBackend {
    fn reclaim_buffer(&mut self, tensor: Tensor) {
        match tensor {
            Tensor::F64(t) => reclaim_typed(&mut self.buffers, t),
            Tensor::F32(t) => reclaim_typed(&mut self.buffers, t),
            Tensor::C64(t) => reclaim_typed(&mut self.buffers, t),
            Tensor::C32(t) => reclaim_typed(&mut self.buffers, t),
        }
    }
}

fn reclaim_typed<T: PoolScalar>(pool: &mut BufferPool, typed: TypedTensor<T>) {
    if let Buffer::Host(data) = typed.buffer {
        T::pool_release(pool, data);
    }
}
```

**Update `eval_exec_ir`:**

```rust
// tenferro/src/exec.rs

pub fn eval_exec_ir<B: TensorBackend>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
    // pool parameter REMOVED
) -> Result<Vec<Tensor>> {
    let mut slots: Vec<Option<Tensor>> = vec![None; program.n_slots];
    for (i, tensor) in inputs.into_iter().enumerate() {
        slots[program.input_slots[i]] = Some(tensor);
    }

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

### Step 6: Remove engine-level `BufferPool`

- Delete `tenferro/src/buffer_pool.rs`
- Remove `pub mod buffer_pool` from `tenferro/src/lib.rs`
- Remove `buffer_pool: BufferPool` field from `Engine`
- Remove `buffer_pool_len()` public API (or delegate to
  `self.backend.buffers.len()`)
- Update `eval_exec_ir` call sites in `tenferro/src/traced.rs` and
  `tenferro/src/engine.rs` to drop the pool parameter

## Scope

**In scope:**
- Typed `BufferPool` in `tenferro-tensor` (f64, f32, c64, c32 pools)
- `CpuBackend.buffers` field
- `install_with_pool` helper for rayon + pool access
- Pool-aware `dot_general` (uninit output buffer from pool)
- `TensorBackend::reclaim_buffer` for reclaim path
- Remove engine-level `BufferPool`

**Out of scope (future work):**
- Pool for `canonical_gemm_layout` copy buffers (typed_transpose in fallback)
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
  returns same capacity buffer (check pointer equality).
- `buffer_pool::tests::best_fit` — pool with buffers of capacity [100, 200, 300];
  request for 150 returns the capacity-200 buffer.
- `buffer_pool::tests::type_separation` — f64 and f32 pools are independent.
- `buffer_pool::tests::fresh_alloc_fallback` — empty pool allocates fresh.
- `buffer_pool::tests::uninit_no_zero_fill` — acquired buffer contents are not
  necessarily zero (test that we're not wasting cycles on initialization).
- Integration: repeated `einsum("bij,bjk,bkl->bil", A, B, C)` calls; after
  first call, `backend.buffers.len() > 0` (buffers recycled).

### Performance (manual/benchmark)
- Steady-state: no new allocations after first einsum evaluation.
- No zero-initialization overhead for GEMM output buffers.

## Migration checklist

1. Create `tenferro-tensor/src/buffer_pool.rs` with typed `BufferPool` +
   `PoolScalar` trait
2. Add `pub mod buffer_pool` to `tenferro-tensor/src/lib.rs`
3. Add `buffers: BufferPool` to `CpuBackend`
4. Add `install_with_pool` helper to `CpuBackend`
5. Add `dot_general_pooled` / `typed_faer_gemm_pooled` in `gemm/mod.rs`
6. Wire `CpuBackend::dot_general` to use `install_with_pool` +
   `dot_general_pooled`
7. Add `reclaim_buffer` to `TensorBackend` trait (default no-op)
8. Implement `reclaim_buffer` for `CpuBackend`
9. Update `eval_exec_ir` to use `backend.reclaim_buffer`, drop pool param
10. Remove `tenferro/src/buffer_pool.rs` and engine-level pool
11. Update all call sites in `engine.rs`, `traced.rs`, tests
12. Add new pool unit tests and integration test
