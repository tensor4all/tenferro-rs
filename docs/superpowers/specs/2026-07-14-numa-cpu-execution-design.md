# NUMA-Aware CPU Execution Design

## Status

This design refines GitHub issue #1345. It defines the initial NUMA-aware
execution contract for `tenferro-cpu` and deliberately limits explicit CPU
placement to execution whose parallelism tenferro controls.

## Goals

- Run independent faer/native graphs concurrently on disjoint NUMA nodes.
- Preserve an execution mode spanning the full CPU set allowed to the process.
- Pin every tenferro-owned Rayon worker to a known logical CPU.
- Keep `CpuBackend` as the public coordinator and make clones share one resource
  registry and arbiter.
- Keep faer and tenferro-native kernels in the same selected Rayon engine.
- Preserve existing BLAS execution while making clear that tenferro does not
  control external-provider worker affinity.
- Document the execution ownership model in the rendered online docs.

## Non-goals

- NUMA-aware allocation, first-touch placement, page migration, or memory
  interleaving.
- Explicit NUMA placement for MKL, OpenBLAS, Accelerate, TBLIS, or another
  external provider in the initial implementation.
- Reaching into OpenMP or provider internals to construct per-node worker teams.
- A fixed mapping from a Rayon task to one worker or core. Workers are pinned;
  Rayon may still move tasks among workers in the selected engine.
- Topology hot-plug handling after backend construction.

## NUMA Topology Definition

The OS NUMA node ID remains the public diagnostic identity. IDs are not
renumbered and may be sparse.

For each discovered OS node, tenferro computes:

```text
usable_node(id).cpus = os_node(id).cpus ∩ process_allowed_cpus
```

Empty intersections are discarded. Usable node CPU sets must be pairwise
disjoint. A topology with overlapping usable node sets is invalid.

CPU IDs identify logical CPUs, including SMT siblings when those logical CPUs
are allowed. Containers and virtual machines use the topology and affinity
visible inside that environment, not the host's hidden physical topology.

When NUMA topology discovery is unavailable, tenferro exposes only the
all-allowed domain. An explicit unknown or unavailable NUMA node request returns
a typed error and never falls back.

## Placement Model

The public request type distinguishes automatic policy from explicit placement:

```rust
pub enum CpuPlacement {
    Auto,
    NumaNode(NumaNodeId),
    AllAllowed,
}
```

`AllAllowed` means the complete logical-CPU set permitted to the process, not
all installed CPUs and not a promise that every CPU will be busy. A resolved
placement carries the concrete CPU set used for pinning and arbitration.

The initial capability matrix is:

| Runtime provider | `Auto` | `NumaNode(id)` | `AllAllowed` |
| --- | --- | --- | --- |
| faer plus tenferro-native kernels | supported | supported | supported |
| BLAS/LAPACK external provider | provider-default exclusive execution | unsupported | unsupported |
| TBLIS or another external provider | provider-default exclusive execution | unsupported | unsupported |

Only `Auto` may resolve to provider-default execution. Explicit `NumaNode` and
`AllAllowed` requests must not silently become an unmanaged provider call.

## Backend And Engine Ownership

`CpuBackend` becomes a cheap cloneable handle:

```rust
pub struct CpuBackend {
    shared: Arc<CpuBackendState>,
}
```

All clones share topology, the engine registry, the resource arbiter, external
provider synchronization, and engine-owned resource statistics. Cloning does
not create a Rayon pool or duplicate a buffer/cache owner.

`CpuBackendState` owns:

- the topology snapshot and process-allowed CPU set;
- one fixed `CpuEngine` for each usable NUMA node;
- one lazily constructed `AllAllowed` engine;
- an overlap-aware resource arbiter;
- synchronization for provider-default exclusive execution.

Each `CpuEngine` owns:

- an immutable execution-domain CPU set;
- a Rayon pool with every worker pinned and verified;
- the engine's thread count and faer `Par` policy;
- a `BufferPool`;
- a `GemmAnalysisCache`;
- other placement- or thread-count-dependent tuning state.

Default engine construction uses one worker per logical CPU in the engine CPU
set. Compatibility constructors with an explicit thread budget cap each
engine's worker count at that budget and spread the workers deterministically
over the engine CPU set. They do not create more workers than CPUs in a pinned
engine. Existing thread-count-only construction continues to work for default
execution while the placement-aware APIs expose resolved per-engine counts.

The all-allowed engine is lazy so NUMA-local workloads do not create a second
set of parked workers unless all-allowed native execution is requested.

## Resource Arbitration

The arbiter reasons about CPU-set overlap, not NUMA node numbers:

- disjoint NUMA engines may execute concurrently;
- one engine initially runs at most one graph at a time;
- the all-allowed engine conflicts with every NUMA engine;
- provider-default exclusive execution conflicts with every tenferro CPU
  engine, even though the provider's exact worker CPU set is unmanaged;
- acquisition of multiple resources uses stable ordering;
- errors and panics release every acquired permit.

External-provider synchronization must cover all handles sharing the backend
state. Synchronization state required solely because an external library owns
process-wide mutable configuration may be process-wide; engine caches and
buffers must not become globals.

## Execution Contexts

### Faer/native graph

After resolving placement and acquiring the engine permit, one session remains
alive for the complete graph. The graph enters the selected Rayon pool once.
Within that scope:

- faer uses `Par::Seq` for a one-worker engine and `Par::rayon(0)` otherwise;
- native elementwise, reduction, structural, and other Rayon-backed kernels use
  the same ambient pool;
- Host and supported extension/FFI boundaries do not recreate the engine
  session or re-enter the pool.

### External-provider graph

An external-provider graph uses provider-default exclusive execution and holds
that permit for the complete graph. It does not claim a public NUMA placement.

Native segments run in the pinned all-allowed Rayon engine. Provider operations
run outside the Rayon pool so provider-created OpenMP or native workers do not
inherit a single pinned Rayon worker as their execution context and do not
create nested parallelism under an active Rayon operation.

The provider-default exclusive permit already reserves the complete tenferro
CPU domain. Native segments borrow the all-allowed engine's pool and mutable
resources under that reservation; they do not acquire a second overlapping
all-allowed permit. This avoids self-conflict while retaining one serialization
boundary for the graph.

```text
provider-default exclusive session
    native segment   -> enter pinned AllAllowed Rayon engine
    provider call    -> leave Rayon; call provider exclusively
    native segment   -> enter pinned AllAllowed Rayon engine
```

The graph owns one permit and one execution session, but may enter the native
pool once per native segment. The one-pool-entry acceptance criterion applies
only to faer/native graphs with managed placement.

## Provider Identity

The existing public provider-selection boundary remains:

```rust
pub enum CpuBackendKind {
    Faer,
    Blas,
}
```

This NUMA work does not add a public `CpuProvider::{Mkl, OpenBlas, ...}` enum.
Concrete provider identity may be unavailable with injected symbols and is not
needed for the initial placement decision.

Internally, execution is classified by capability:

```text
TenferroRayon    -> managed explicit placement
ExternalProvider -> provider-default exclusive execution
```

Public callers may query placement capabilities and topology. Concrete provider
details may appear as diagnostic text or benchmark metadata, but are not a
stable enum used for control flow.

## Errors

Unsupported or invalid placement fails before graph execution. Errors distinguish
at least:

- unknown NUMA node ID;
- NUMA discovery unavailable;
- invalid/overlapping topology;
- worker pinning failure;
- external provider does not support managed placement;
- engine construction failure.

An error should include the requested placement, public backend kind, and a
stable reason category. It must not retry with a different placement unless the
original request was `Auto`.

## Public Documentation

The implementation PR must add a rendered user guide at
`docs/guides/cpu-execution.md` and link it from `docs/getting-started/index.md`
and the docs index/navigation.

The guide must explain:

- OS NUMA node IDs and intersection with process-allowed CPUs;
- `Auto`, `NumaNode`, `AllAllowed`, and provider-default exclusive execution;
- the provider/placement capability matrix;
- where native elementwise kernels execute in a BLAS-backed graph;
- that thread-count control is not CPU-affinity control;
- cloneable `CpuBackend` handle semantics;
- affinity versus NUMA memory placement;
- typed errors and non-fallback behavior;
- topology and resolved-placement diagnostics.

Rustdoc for the new public placement, topology, capability, and execution APIs
must contain compiling, runnable examples. User-facing examples must use public
direct crates and should verify a deterministic property rather than only print
diagnostics.

Durable internal architecture belongs in a CPU backend design document under
`docs/design/`; the implementation PR also requires a work log recording the
staged migration, verification, and residual provider risks.

## Testing

Topology and arbitration tests use injected fixtures and do not require a
multi-socket CI host. They cover:

- affinity intersection, empty nodes, sparse IDs, and overlap rejection;
- disjoint permit coexistence and overlapping permit exclusion;
- lazy all-allowed construction;
- clone handles sharing the same arbiter and engines;
- worker pinning success and failure;
- release after errors and panics;
- explicit external-provider placement rejection;
- `Auto` provider-default resolution;
- one pool entry for a managed faer/native graph across Host/FFI boundaries;
- native/provider/native context transitions for an external-provider graph.

Linux-only diagnostics may verify observed worker CPUs remain within their
engine CPU set. Tests must still pass on single-node and topology-unavailable
hosts.

## Benchmarks

An opt-in multi-NUMA benchmark compares:

- two concurrent faer/native graphs on disjoint nodes;
- one faer/native graph on the all-allowed engine;
- the existing external-provider path under provider-default exclusive
  execution.

Output reports the process-allowed CPU set, resolved topology, provider kind,
placement or unmanaged-provider mode, worker/thread count, and problem shape.
No benchmark may describe external-provider execution as NUMA-pinned without
verifying every provider worker's affinity.

## Staged Delivery

The work should be split into reviewable PRs or clearly separated commits:

1. placement/topology value types, capability checks, fixture tests, and docs
   contract without changing execution behavior;
2. topology discovery and diagnostics;
3. cloneable backend state, engine registry, and overlap arbiter;
4. pinned faer/native Rayon engines;
5. one managed engine session across complete graph execution;
6. provider-default exclusive mixed-graph execution;
7. rendered docs, durable design documentation, diagnostics, and benchmarks.

The first behavior-changing PR must preserve existing constructors and ordinary
BLAS execution. No PR may advertise external-provider NUMA placement merely
because it can set a provider thread count.

## Acceptance Criteria

- Two managed faer/native graphs execute concurrently on disjoint fixture or
  hardware NUMA nodes.
- Every managed Rayon worker is pinned to its engine CPU set.
- The lazy all-allowed engine supports full-domain faer/native execution and
  conflicts with every node engine.
- `CpuBackend` clones share engines and arbitration.
- Explicit placement for external providers returns a typed error without
  fallback.
- External-provider graphs hold a global exclusive permit; native segments use
  the pinned all-allowed Rayon engine and provider calls occur outside Rayon.
- Engine-local buffers and analysis caches are not concurrently shared.
- Managed faer/native graphs create one engine session and enter one pool once,
  including supported Host/FFI boundaries.
- Single-node and topology-unavailable hosts retain working default execution.
- Rendered online docs describe the mixed BLAS/native execution model and the
  distinction between thread count, affinity, and NUMA memory placement.
- Required tests, docs checks, coverage checks, and multi-NUMA diagnostics pass.
