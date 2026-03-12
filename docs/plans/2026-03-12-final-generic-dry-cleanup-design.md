# Final Generic/DRY Cleanup Design

## Goal

Close the most visible remaining production debt after the family-trait and
CPU-named-linalg cleanup:

- remove the CPU-only thread-local linalg semiring bridge in `tenferro-linalg`
- reduce concrete-backend duplication in `tenferro-dyadtensor` runtime dispatch
- remove `expect`-based library error handling in `tenferro-burn`
- update active docs so they describe the current generic layering accurately

## Scope

This bundle is intentionally limited to active production code and active docs.
It does not rewrite historical notes under `docs/design/reference/**` or
`docs/plans/**`, except where a current design page explicitly references the
live architecture.

## Current Problems

### 1. `tenferro-linalg::prims_bridge` is still CPU-only

`tenferro-linalg/src/prims_bridge.rs` currently owns a thread-local
`CpuContext` and dispatches directly to `CpuBackend`. That means a composite API
such as `matrix_power` still reaches semiring execution through an internal
CPU-only shortcut instead of a generic context/boundary.

### 2. `tenferro-dyadtensor` runtime dispatch still repeats concrete slots

`extension/tenferro-dyadtensor/src/api/runtime_dispatch.rs` already centralizes
dispatch, but still repeats CPU/CUDA/ROCm slot wiring in several helpers and
macros. The shape is better than before, but not yet the cleanest generic form.

### 3. `tenferro-burn` still uses `expect(...)` in library code

`extension/tenferro-burn` still panics for invalid subscripts, missing tree
steps, and malformed autodiff graph state. Some invariants can remain
assertions, but user-controlled failures should become explicit errors at the
bridge boundary or be downgraded to checked internal helpers.

### 4. Active docs still understate current backend-generic shape

The recent `KernelLinalgScalar` split is documented, but the supported-ops and
architecture descriptions still do not explain the remaining gaps in terms of
concrete debt items such as the CPU-only prims bridge and Burn error handling.

## Recommended Approach

### A. Replace the CPU-only prims bridge with explicit context threading

Move `prims_bridge` from "hidden thread-local CPU fallback" to "generic helper
that executes through the caller-provided semiring core context". The internal
matrix helpers that still need batched GEMM should accept the same context they
already receive from the linalg API entrypoint.

This keeps `tenferro-linalg` honest: it remains a composition layer and stops
owning a hidden runtime.

### B. Normalize dyadtensor runtime dispatch around a single slot table

Keep the public API generic over the runtime families, but reduce concrete
duplication by collapsing CPU/CUDA/ROCm slot metadata and the dispatch macros
around one shared pattern. The goal is not "erase runtime types completely";
the goal is to keep concrete runtime names in one place and keep all higher
layers dependent only on the runtime-family contracts.

### C. Replace panic-oriented Burn glue with checked helpers

For `tenferro-burn`, user-controlled parse/shape failures should flow through a
checked helper and produce a clear panic only at the absolute Burn API surface
when Burn itself requires infallible trait methods. Internal structural
expectations should be isolated in small helpers so the panic surface is
minimal and auditable.

### D. Update docs to describe the real remaining boundaries

The active design docs should describe:

- semiring/linalg composition is generic at the API boundary
- GPU coverage is still capability-gated, not "planned only"
- remaining non-generic debt is now the semiring bridge/runtime glue, not the
  removed legacy traits

## Non-Goals

- no custom CUDA kernels
- no new public numerical operations
- no historical-doc rewrite beyond minimal caveats
- no benchmark-driven refactor in this bundle

## Success Criteria

- `tenferro-linalg/src/prims_bridge.rs` no longer owns a thread-local
  `CpuContext`
- production linalg composition paths do not hide a CPU backend behind the
  bridge
- `tenferro-burn` library code no longer contains `expect(...)`
- `dyadtensor` runtime dispatch has less repeated concrete-slot wiring while
  preserving CPU/GPU generic API contracts
- active docs match the new state
- full workspace verification passes
