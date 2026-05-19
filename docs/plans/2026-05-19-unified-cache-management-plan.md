# Unified cache management plan

Status: TODO

After the current optimization work settles, add a unified cache-management
surface across tenferro and tenferro-tensor.

Goals:

- Define explicit upper bounds for every runtime/compiler cache.
- Provide a single user-facing way to configure cache limits.
- Provide a single user-facing way to clear caches.
- Document cache ownership, default limits, memory behavior, and clearing APIs.

Initial cache inventory:

- `Engine::compile_cache`
- `Engine::einsum_cache`
- `Engine::einsum_parse_cache`
- `Engine::backend_cache` for backend-specific prepared analysis, currently
  including CPU GEMM analysis slots keyed by compiled instruction index
- `CpuBackend` buffer pool
- Any future compiled execution-plan caches

Design notes:

- Long-lived cache lifetimes should be owned by `Engine` or another explicit
  top-level runtime object, not hidden thread-local state or backend internals.
- Backend resource pools may remain backend-owned, but must still be
  configurable and clearable through the public runtime surface.
- Defaults should be bounded and conservative.
- Clearing should be available at both fine-grained and aggregate levels.
- Documentation should explain which caches retain memory after `clear`, which
  release memory to the allocator, and which are per-backend or per-engine.
