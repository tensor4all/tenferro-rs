# Issue #1576 safety-audit remediation

## Summary

Implemented the accepted five-item remediation from #1576 on `origin/main`
(a992a44c). Overwrite proofs now name the runtime-selected zip/map, scalar-map,
add, and multiplication operations; the source-contract test checks those
phrases and rejects duplicate adjacent proofs. TBLIS thread-control FFI now
states the process-global ABI/serialized guard/Drop-restoration contract.

The erased session bridge now uses explicit `BackendSession` implementations
for each concrete backend/session and private `'static` marker structs whose
`TypeId`s identify the concrete target; it no longer uses a shared
`BackendSessionIdentity` or `type_name`. CPU tests cover foreign-marker
rejection without callback invocation and a scoped concrete borrow.

CUDA cache scans have canonical bounded `// INVARIANT:` markers. The pinned
CubeCL source was inspected at revision `1c88bb6f1a47ffb11755e05048b7828a743f53e1`:
`cubecl-cuda/src/compute/storage/gpu.rs` records deallocations and matches
`malloc_async` with `free_async` on the owning stream, while sync allocations
use `free_sync`; `cubecl-runtime` flushes storage through the CUDA server.
The cache INVARIANT marker records this handle-owned release contract. A real
CUDA eviction test retains nonzero workspace, asserts that allocation and an
actual eviction occur with `max_entries = 1`, synchronizes at the test
boundary, and validates both outputs.

## Verification

Passed:

- `cargo test -p tenferro-internal-cpu-kernels --lib internal_full_overwrite_sources_use_the_guard_boundary`
- `cargo test --manifest-path ext/tenferro-cpu-tblis/Cargo.toml`
- `cargo test -p tenferro-tensor --lib backend_default_read`
- `cargo test -p tenferro-cpu --lib with_cpu_exec_session`
- `cargo test -p tenferro-ad --test integration runtime_snapshot`
- `cargo test -p tenferro-einsum --lib typed_eager`
- `cargo test -p tenferro-fft --test backend_capability`
- `cargo test -p tenferro-linalg --test integration backend_errors`
- `cargo check -p tenferro-gpu --features cuda --lib`
- `cargo check -p tenferro-gpu --features webgpu --lib`
- `cargo test -p tenferro-gpu --features cuda --lib cuda_cutensor_cache_eviction_keeps_inflight_workspace_valid -- --ignored`
- `cargo fmt --all`
- After rebasing onto `origin/main`, the backend capability contract was
  narrowed to scan only CPU contraction entry-point bodies in
  `exec_session.rs`; `cargo test -p tenferro-cpu --test integration
  backend_capability_contracts::cpu_provider_dispatch_has_no_runtime_registry_lookup_or_legacy_staging
  -- --exact` passed.

The CUDA eviction test passed on the available CUDA GPU/toolkit environment.
No new dependency, synchronization, registry, downcast framework, or shim was
added. Generated build directories, Cargo.lock files, and subagent artifacts
were removed before handoff.
