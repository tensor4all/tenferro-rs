# Complex Prims GEMM Dispatch Design

## Summary

Issue `#245` already routes real-valued `tenferro-linalg` GEMM helper paths through
`tenferro-prims::PrimDescriptor::BatchedGemm` in this worktree. The remaining gap is
complex-valued GEMM: `tenferro-prims` still optimizes only `f64` / `f32`, and
`tenferro-linalg` keeps a complex fallback that bypasses prims.

This design extends the optimized GEMM backend layer to `Complex64` and `Complex32`,
then removes the last complex fallback in `tenferro-linalg` so all GEMM-style helper
work uses `tenferro-prims`.

## Goals

- Add optimized complex GEMM dispatch to `tenferro-prims`
- Require exactly one CPU GEMM backend (`gemm-faer` or `gemm-openblas`)
- Remove the portable naive GEMM fallback path
- Route complex `tenferro-linalg` GEMM helpers through prims, matching the real path

## Non-Goals

- Redesign the public `TensorPrims` API
- Redesign `tenferro-linalg::backend::LinalgBackend`
- Change decomposition or solve kernels outside GEMM helper routing
- Expand issue `#246` beyond the already-added API skeleton

## Design Decisions

### 1. Enforce Exactly One CPU GEMM Backend

`tenferro-prims` should reject unsupported feature combinations at compile time:

- `gemm-faer` enabled, `gemm-openblas` disabled: supported
- `gemm-faer` disabled, `gemm-openblas` enabled: supported
- both enabled: compile error
- both disabled: compile error

This makes the backend choice explicit and removes the need for a correctness-only
fallback path.

### 2. Add Complex Optimized Dispatch for Both Supported Backends

#### `gemm-faer`

Extend the internal `FaerGemm` trait implementation set from:

- `f64`
- `f32`

to also include:

- `Complex64`
- `Complex32`

The existing `impl_faer_gemm!` macro must be generalized to compare against typed
`zero` / `one` values instead of real literals so the same accumulate semantics work
for real and complex scalars.

With `gemm-faer`, `execute_batched_gemm` should dispatch complex tensors through the
same strided zero-copy path used by real tensors.

#### `gemm-openblas`

The current OpenBLAS path only provides `dgemm` / `sgemm`. Add:

- `cblas_zgemm` for `Complex64`
- `cblas_cgemm` for `Complex32`

These should plug into the existing contiguous packed path used when `gemm-faer` is
not active. That preserves the current "pack then GEMM" architecture for OpenBLAS
while giving complex tensors an optimized backend instead of a generic scalar loop.

### 3. Remove Naive GEMM Fallback

Once exactly one backend is required and both backends support complex optimized GEMM,
the CPU GEMM execution path no longer needs the portable naive fallback.

`execute_batched_gemm` should become a two-state dispatcher:

- `gemm-faer`: real + complex use strided fast path
- `gemm-openblas`: real + complex use contiguous packed fast path

The standalone naive helpers can be deleted, along with docs that advertise them.

### 4. Simplify `tenferro-linalg` Routing

After `tenferro-prims` supports optimized complex GEMM:

- Replace the temporary real-only `batched_gemm_real` bridge with a generic
  `batched_gemm_via_prims<T>`
- Remove the `TypeId`-based real/complex split in `prims_bridge`
- Route both `backend_mat_mul` and `backend_mat_mul_nn` directly through the generic
  prims bridge
- Delete the last direct complex fallback in `backend_mat_mul_nn`

This makes `tenferro-linalg`'s internal GEMM boundary consistent: helper GEMM always
goes through prims regardless of scalar type.

## Testing Strategy

### `tenferro-prims`

- Add unit tests that exercise complex GEMM under the optimized backend path
- Keep existing `PrimDescriptor::BatchedGemm` tests for complex values as regression
  coverage
- Add feature-combination checks that fail fast on unsupported configurations

### `tenferro-linalg`

- Extend the existing `RejectingMatMulBackend` test strategy so both real and complex
  helper paths prove they do not call `backend.mat_mul`
- Re-run `matrix_exp` real and complex tests to confirm behavioral parity

## Verification

Fresh verification must include:

- `cargo fmt --all`
- `cargo test -p tenferro-prims`
- `cargo test -p tenferro-linalg`
- `cargo test --workspace`

## Rationale

This keeps issue `#245` narrowly focused on GEMM routing while still finishing the
complex side correctly. It also improves the future migration path for issue `#246`
by removing the last internal slice-GEMM special case from `tenferro-linalg`.
