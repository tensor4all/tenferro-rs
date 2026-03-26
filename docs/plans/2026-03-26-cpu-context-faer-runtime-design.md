# CPU Context-Owned Faer Runtime Design

## Status

Approved on 2026-03-26.

## Context

Two active problems meet at the same design seam:

- Issue #336 exposed that CPU contraction still spends too much time in per-call layout materialization and temporary allocation, even after removing the naive contraction fallback on standard scalar GEMM-valid paths.
- Issue #572 points out that `CpuContext` currently constructs a dedicated rayon thread pool that faer-backed execution does not actually use. The public API therefore suggests a threading contract that is not true in practice.

At the moment, CPU faer execution is split across two inconsistent models:

- `tenferro-prims` owns `CpuContext`, plan cache, and BLAS scratch, but faer GEMM still hardcodes `Par::rayon(0)`.
- `tenferro-linalg` and `tenferro-linalg-prims` mostly use faer high-level solver APIs such as `mat.qr()` and `mat.thin_svd()`, which internally read faer's global parallelism instead of consulting `CpuContext`.

This split makes it impossible to give `CpuContext::new(num_threads)` a coherent meaning, and it prevents clean reuse of temporary buffers across contraction and linalg kernels.

## Goals

- Make `CpuContext` the single owner of CPU execution policy and reusable CPU execution resources.
- Make the `num_threads` parameter real by routing all faer-backed CPU execution through the `CpuContext` thread pool.
- Add reusable temporary buffer pooling for contraction fallback and faer scratch/workspace.
- Move `prepare_one_operand`-style partial fallback down to the primitive layer so it can be reused outside `tenferro-einsum`.
- Keep partial fallback semantically pure: it may change temporary layout materialization strategy, but not tensor values or API-visible ownership semantics.

## Non-Goals

- This design does not try to unify CPU and GPU runtime ownership.
- This design does not change BLAS provider threading behavior beyond keeping CPU context semantics explicit.
- This design does not preserve the current faer high-level wrapper implementation style if that style conflicts with context-owned execution policy.

## Chosen Direction

We choose the strong form of `CpuContext`:

- `CpuContext` keeps a rayon `ThreadPool`.
- `CpuContext` becomes the owner of temporary CPU execution resources.
- All faer-backed CPU execution must become context-aware.

That means issue #572 is addressed by making the existing thread-pool contract true, not by deleting the pool.

## Ownership Model

### `CpuContext` owns

- rayon `ThreadPool`
- plan cache
- BLAS scratch pool
- typed temporary tensor/buffer pool for CPU materialization
- faer scratch/workspace pool

### `CpuContext` does not own

- global process-wide parallelism policy
- global faer parallelism state
- user-visible tensor results

`CpuContext` is therefore a reusable mutable execution context, not a plain thread-count wrapper.

## Lifetime Model

There are two distinct lifetimes:

- Context lifetime: as long as the caller keeps a `CpuContext`
- Loan lifetime: a borrowed temporary buffer or tensor is valid only for one primitive or linalg call

The pool outlives each operation, but the individual loan does not escape the operation. This keeps ownership simple and allows `Tensor::from_vec(...)` / `try_into_data_vec()` to be used for reusable temporary column-major tensors without leaking pooled storage into public API outputs.

## Faer Execution Policy

### Problem

Faer high-level solvers use `get_global_parallelism()`, not `CpuContext`.

Examples:

- `mat.qr()`
- `mat.thin_svd()`
- `mat.self_adjoint_eigen(...)`
- `mat.partial_piv_lu()`

Likewise, current faer GEMM uses `Par::rayon(0)` directly.

### Decision

For all CPU faer paths that are supposed to respect `CpuContext`, we stop using the faer high-level convenience methods and move to low-level faer APIs that accept explicit `Par` and explicit scratch/workspace.

This applies to:

- GEMM / strided GEMM
- QR
- thin SVD
- self-adjoint EVD
- LU
- any other faer-backed CPU path whose semantics should depend on `CpuContext`

Execution happens inside `ctx.thread_pool().install(...)`, and the faer call receives `Par::rayon(ctx.num_threads())`.

This gives `CpuContext::new(n)` one consistent meaning across primitive and linalg code.

## Temporary Buffer Design

## Typed pools

The context owns typed pools keyed by scalar type and capacity class. The minimum useful unit is reusable `Vec<T>`, because:

- temporary contraction operands can be represented as `Tensor<T>` from pooled `Vec<T>`
- faer scratch/workspace can be backed by reusable byte buffers
- the same pool substrate can serve both primitive and linalg paths

Recommended internal split:

- `TempVecPool<T>` for typed data buffers
- `ByteScratchPool` for faer scratch/workspace and similar raw temporary memory

## Contract temporary tensors

Contraction fallback uses pooled temporaries for:

- materialized `A`
- materialized `B`
- temporary `C` when output layout is non-fusible

The temporary tensor memory is owned by the context, wrapped into a temporary `Tensor<T>` for the duration of the call, then returned to the pool.

## Partial Fallback Design

`prepare_one_operand`-style logic moves down to the CPU primitive layer.

The key abstraction is a prepared operand that is semantically pure:

```rust
enum PreparedOperand<'a, T> {
    Borrowed(StridedView<'a, T>),
    Materialized(TemporaryTensor<T>),
}
```

Preparation may:

- keep a borrowed view
- permute metadata only
- partially materialize one non-fusible group
- fully materialize the operand

Preparation may not:

- mutate user inputs
- leak pooled memory into returned public outputs
- alter contraction algebra or ownership semantics

## Module Boundaries

## `tenferro-prims`

This crate owns CPU execution substrate:

- `cpu/context.rs`
  - extend `CpuContext`
  - add `install`, `faer_par`, temporary-pool accessors
- `cpu/temp_pool.rs`
  - typed vector pool and raw scratch pool
- `cpu/contract_prepare.rs`
  - partial fallback and prepared-operand logic
- `cpu/gemm_support.rs`
  - route faer GEMM through context-owned execution policy
- `cpu/contract_gemm.rs`
  - consume prepared operands and pooled temp outputs

## `tenferro-linalg-prims`

This crate owns low-level tensor linalg execution contracts and their context-aware CPU implementation:

- keep `type Context = tenferro_prims::CpuContext`
- stop ignoring `ctx`
- route CPU faer linalg through context-aware low-level helpers

## `tenferro-linalg`

This crate keeps the public/composite layer and should not own CPU execution policy. It may continue to expose APIs that accept `&mut CpuContext`, but the actual faer lowering must no longer bypass that context.

## API Consequences

No public API shape change is required for the core move:

- `CpuContext::new(num_threads)` stays
- public CPU APIs that already accept `&mut CpuContext` stay

What changes is the contract:

- the thread count now actually controls faer-backed CPU execution
- temporary CPU allocation reuse becomes context-scoped rather than allocator-luck

## Migration Strategy

The migration should be staged so each operation family becomes fully context-aware before the old path is deleted.

Recommended order:

1. extend `CpuContext` with temporary pools and faer execution helpers
2. move contraction partial fallback and temp reuse into `tenferro-prims`
3. make faer GEMM context-aware
4. port QR to low-level faer API
5. port thin SVD to low-level faer API
6. port EVD, LU, and remaining faer-backed CPU linalg paths
7. update downstream docs, benches, and cached-context integration points

Within each operation family, do not keep both a context-aware and a hidden global-parallelism faer path once the new one is validated.

## Testing Strategy

Testing must prove both numerical correctness and context ownership semantics.

### Primitive tests

- `CpuContext::new(n)` thread-pool behavior is observable
- standard-scalar contraction never falls back to the naive contraction loop on GEMM-valid paths
- non-fusible output uses pooled temp output rather than per-call fresh allocation
- partial fallback preserves results for borrowed, partial, and full materialization cases

### Linalg tests

- QR / thin SVD / EVD / LU remain numerically correct
- context-aware implementations no longer ignore `ctx`
- `CpuContext::new(1)` remains the stable single-thread baseline for benches and deterministic focused tests

### Integration tests

- cached `CpuContext` reuse in downstream code still works
- issue #336 benchmark improves further once contraction materialization reuse lands
- issue #572 semantics are resolved because the `CpuContext` thread count becomes real

## Risks

- Low-level faer APIs are more verbose and require explicit scratch management.
- The migration touches both primitives and linalg, so mixed old/new paths must be avoided.
- Some downstream code currently recreates `CpuContext` per call; that code will still work, but it will not benefit from reuse until follow-up cleanup adopts longer-lived contexts.

## Follow-Up Work

- Move `tensor4all-rs` hot loops that still recreate `CpuContext::new(1)` toward cached or runtime-local reuse where appropriate.
- Re-benchmark issue #336 after pooled contraction fallback lands.
- Revisit whether any CPU paths that remain BLAS-backed need analogous context semantics or explicit documentation boundaries.
