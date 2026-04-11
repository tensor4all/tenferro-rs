# Einsum Buffer Pool — Implementation Plan

> **IMPORTANT**: Do NOT auto-implement. An agent must discuss the plan with a
> human reviewer and get explicit approval before writing any code.

**Issue**: #671
**Design**: `docs/plans/2026-04-10-einsum-buffer-pool-design.md`
**Branch**: `optimize_einsum`

## Step 1: Create `BufferPool` in `tenferro-tensor`

**New file**: `tenferro-tensor/src/buffer_pool.rs`

Create the typed buffer pool with:
- `BufferPool` struct with `BTreeMap<usize, Vec<Vec<T>>>` per dtype (f64, f32, c64, c32)
- `PoolScalar` sealed trait with `pool_acquire` (unsafe, uninit) and `pool_release`
- `impl_pool_scalar!` macro for the 4 types
- `take_best_fit` helper (BTreeMap range search)
- `new()`, `len()`, `is_empty()`, `Default`

**Also edit**: `tenferro-tensor/src/lib.rs` — add `pub mod buffer_pool`

**Tests** (in `tenferro-tensor/src/buffer_pool.rs` or dedicated test file):
- `acquire_release_reuse`: acquire → release → acquire returns same capacity
- `best_fit`: pool with [100, 200, 300]; request 150 → gets 200-cap buffer
- `type_separation`: f64 and f32 pools independent
- `fresh_alloc_fallback`: empty pool allocates fresh
- `zero_len_not_pooled`: release of zero-capacity buffer is no-op

**Verify**: `cargo test -p tenferro-tensor`

## Step 2: Add `BufferPool` to `CpuBackend`

**Edit**: `tenferro-tensor/src/cpu/backend.rs`

- Add `buffers: BufferPool` field to `CpuBackend`
- Initialize in `new()` and `with_threads()`
- Add `install_with_pool` helper method:
  ```
  fn install_with_pool<R: Send>(&mut self, op: impl FnOnce(&mut BufferPool) -> R + Send) -> R
  ```
  Uses `std::mem::take` to move pool into rayon closure and back.

**Verify**: `cargo test -p tenferro-tensor` (existing tests still pass)

## Step 3: Pool-aware `dot_general`

**Edit**: `tenferro-tensor/src/cpu/gemm/mod.rs`

Add new functions alongside existing ones (do NOT delete the old ones yet):
- `dot_general_pooled<T>(buffers, lhs, rhs, config)` — same logic as `dot_general`
  but passes `buffers` to GEMM functions
- `typed_faer_gemm_pooled<T>(buffers, lhs, rhs, config)` — replaces
  `vec![T::zero(); out_n]` with `unsafe { T::pool_acquire(buffers, out_n) }`

The `T` bound gains `PoolScalar`:
```
T: FaerGemm + PoolScalar + Copy + Clone + Zero + One + PartialEq
```

For the zero-size edge case (`m==0 || n==0 || k==0 || batch_total==0`),
keep `vec![T::zero(); out_n]` (pool not needed for empty tensors).

Do the same for `typed_blas_gemm_pooled` if `cpu-blas` feature exists.

**Edit**: `tenferro-tensor/src/cpu/backend.rs`

Wire `CpuBackend::dot_general` to use `install_with_pool` + `dot_general_pooled`:
```rust
fn dot_general(&mut self, lhs, rhs, config) -> Result<Tensor> {
    self.install_with_pool(|buffers| match (lhs, rhs) {
        (Tensor::F64(a), Tensor::F64(b)) =>
            gemm::dot_general_pooled(buffers, a, b, config).map(Tensor::F64),
        // ... other dtypes
    })
}
```

**Verify**: `cargo test -p tenferro-tensor` — all GEMM tests pass with pooled path.

## Step 4: `TensorBackend::reclaim_buffer`

**Edit**: `tenferro-tensor/src/backend.rs`

Add to `TensorBackend` trait:
```rust
fn reclaim_buffer(&mut self, _tensor: Tensor) {}
```
Default no-op (GPU backends, tests).

**Edit**: `tenferro-tensor/src/cpu/backend.rs`

Implement for `CpuBackend`:
```rust
fn reclaim_buffer(&mut self, tensor: Tensor) {
    match tensor {
        Tensor::F64(t) => reclaim_typed::<f64>(&mut self.buffers, t),
        Tensor::F32(t) => reclaim_typed::<f32>(&mut self.buffers, t),
        Tensor::C64(t) => reclaim_typed::<Complex64>(&mut self.buffers, t),
        Tensor::C32(t) => reclaim_typed::<Complex32>(&mut self.buffers, t),
    }
}
```

Where `reclaim_typed` extracts `Buffer::Host(data)` and calls
`T::pool_release(pool, data)`.

**Verify**: `cargo test -p tenferro-tensor`

## Step 5: Update `eval_exec_ir` reclaim path

**Edit**: `tenferro/src/exec.rs`

- Change `eval_exec_ir` signature: remove `pool: &mut BufferPool` parameter
- Change `reclaim_last_use_inputs` to take `backend: &mut B` instead of
  `pool: &mut BufferPool`
- Call `backend.reclaim_buffer(tensor)` instead of
  `reclaim_tensor_buffer(tensor, pool)`
- Remove `reclaim_tensor_buffer` and `extract_host_bytes` functions

**Verify**: `cargo test -p tenferro` — check compile. Expect failures from
callers of `eval_exec_ir` that still pass the pool parameter.

## Step 6: Update callers and remove engine-level pool

**Edit**: `tenferro/src/traced.rs`
- Update calls to `eval_exec_ir` — drop the `&mut engine.buffer_pool` argument

**Edit**: `tenferro/src/engine.rs`
- Remove `buffer_pool: BufferPool` field from `Engine`
- Remove `buffer_pool_len()` method (or delegate to `self.backend.buffers.len()`)
- Update `Engine::new()` to not initialize buffer_pool

**Delete**: `tenferro/src/buffer_pool.rs`

**Edit**: `tenferro/src/lib.rs`
- Remove `pub mod buffer_pool`

**Edit**: `tenferro/tests/cpu_backend.rs`
- Remove or update buffer pool tests (move pool-specific tests to
  `tenferro-tensor` if not already covered in Step 1)

**Verify**: `cargo test --workspace --release` — full green.

## Step 7: Integration test and final verification

**Add test** (in `tenferro/tests/` or `tenferro-tensor` tests):
- Repeated einsum evaluation reuses buffers:
  ```rust
  let mut engine = Engine::new(CpuBackend::new());
  let result1 = engine.eval(einsum_expr, inputs1);
  // After first eval, pool should have reclaimed buffers
  // Second eval with same shapes should not allocate new buffers
  let result2 = engine.eval(einsum_expr, inputs2);
  ```

**Run full verification**:
```bash
cargo fmt --all --check
cargo test --workspace --release
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
```

## File change summary

| File | Action |
|---|---|
| `tenferro-tensor/src/buffer_pool.rs` | **NEW** |
| `tenferro-tensor/src/lib.rs` | add `pub mod buffer_pool` |
| `tenferro-tensor/src/backend.rs` | add `reclaim_buffer` to trait |
| `tenferro-tensor/src/cpu/backend.rs` | add `buffers` field, `install_with_pool`, wire `dot_general`, impl `reclaim_buffer` |
| `tenferro-tensor/src/cpu/gemm/mod.rs` | add `dot_general_pooled`, `typed_faer_gemm_pooled` (+ blas variant) |
| `tenferro/src/exec.rs` | drop pool param, use `backend.reclaim_buffer` |
| `tenferro/src/engine.rs` | remove `buffer_pool` field |
| `tenferro/src/traced.rs` | update `eval_exec_ir` call sites |
| `tenferro/src/lib.rs` | remove `pub mod buffer_pool` |
| `tenferro/src/buffer_pool.rs` | **DELETE** |
| `tenferro/tests/cpu_backend.rs` | update/remove pool tests |
