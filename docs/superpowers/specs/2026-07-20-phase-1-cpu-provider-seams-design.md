# Phase 1 CPU Provider Seams Design

**Issue:** [#1434](https://github.com/tensor4all/tenferro-rs/issues/1434)

**Parent:** [#1433](https://github.com/tensor4all/tenferro-rs/issues/1433)

**Baseline:** `85855e272b1495611deb601a9ee06f3546772c3c` (`origin/main` on 2026-07-20)

## Scope

Phase 1 extracts CPU contraction providers without changing resource-domain
ownership, graph scheduling, or eager session entry. It introduces borrowed,
validated requests; direct immutable provider slots; an engine-owned
`DotGeneralRuntime`; and construction-time provider replacement. Eager and
graph execution continue to enter the existing `CpuBackend` session exactly
once.

NUMA executors, `Managed`/`ExternalManaged`, `SemanticProgram`, common graph
scheduling, GPU execution, extension lowering, linalg-family migration, and a
budget-one eager fast path remain outside this phase.

## Chosen Boundary

The engine allocates outputs and canonicalization temporaries from the existing
session `BufferPool`. Providers receive borrowed input views and a reborrowed,
preallocated output and write into it synchronously. They do not receive the
pool, create a session, enter a Rayon pool, acquire a resource permit, or own a
cache.

This boundary was selected over two alternatives:

1. Providers returning owned tensors would make allocation and pool ownership
   provider-specific and would split returning and `_into` execution paths.
2. A public mutable context exposing `BufferPool` would expose current engine
   internals and prematurely constrain the Phase 2 resource-domain design.

The write-into boundary keeps allocation and fallback in the engine and makes
provider dispatch object-safe without a generic scalar method.

## Public Provider SPI

The SPI lives in `tenferro_cpu::provider`. All traits are object-safe and use
dynamic `TensorRead`/`TensorWrite` values so one trait object covers all
supported scalar dtypes.

```rust
pub trait CpuGemmProvider: core::fmt::Debug + Send + Sync + 'static {
    fn gemm(
        &self,
        context: &CpuProviderContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome>;

    fn strided_batched_gemm(
        &self,
        context: &CpuProviderContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome>;

    fn grouped_gemm(
        &self,
        context: &CpuProviderContext<'_>,
        request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome>;
}

pub trait CpuLayoutTransformProvider: core::fmt::Debug + Send + Sync + 'static {
    fn materialize(
        &self,
        context: &CpuProviderContext<'_>,
        request: CpuLayoutTransformRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome>;
}

pub trait CpuGeneralContractionProvider:
    core::fmt::Debug + Send + Sync + 'static
{
    fn dot_general(
        &self,
        context: &CpuProviderContext<'_>,
        request: CpuDotGeneralRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome>;
}
```

The three request lifetimes are the short request borrow, the read-input
lifetime, and the writable-output lifetime. Requests borrow
`&mut TensorWrite<'out>`; they never consume the caller's output. Consequently,
an explicit `Unsupported` result ends the provider borrow and lets the
engine-owned composite safely reuse the same output for the next configured
resolution step.

Request fields are private. Public accessors expose borrowed tensor views,
scalar accumulation, dimensions, strides, and contraction-role iterators.
Constructors remain crate-private because only the engine may assert that a
request is validated.

### Outcomes

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub enum CpuProviderOutcome {
    Executed,
    Unsupported(CpuProviderUnsupported),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CpuProviderUnsupported {
    DType(DType),
    Rank { lhs: usize, rhs: usize },
    Layout(CpuOperand),
    Conjugation,
    Accumulation,
    StridedBatch,
    Grouped,
    RuntimeUnavailable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuOperand {
    Lhs,
    Rhs,
    Output,
}
```

`Unsupported` is the only result that continues configured resolution and it
must leave the output unchanged. `Err` is terminal and never triggers another
provider. A required capability maps `Unsupported` to the existing structured
`tenferro_tensor::Error::Unsupported` category while preserving provider and
reason. Provider failures preserve their typed source chain.

### Execution context

`CpuProviderContext` has private fields and exposes only:

```rust
pub fn thread_budget(&self) -> usize;
pub fn kernel_parallelism(&self) -> CpuKernelParallelism;
```

`CpuKernelParallelism` is `Sequential` or `Inner`. The engine's outer-batch
mode is not passed to a provider: under outer fan-out the engine invokes a
single-job provider request with `Sequential`. Crate-private code can reach the
existing `CpuContext` to perform the one permitted fan-out or native-layout
scope. Public providers cannot call `CpuContext::install` through this API.

## Request Shapes

`CpuGemmRequest` describes GEMM rather than semantic `dot_general`. It contains
borrowed lhs/rhs/output values, `m`, `n`, `k`, batch count, checked element
offsets, row/column/batch strides for all operands, conjugation flags, and
`DotGeneralAccumulation`. A single GEMM has batch count one. A strided-batched
request uses the same descriptor with batch count greater than one.

`CpuGroupedGemmRequest` contains borrowed lhs/rhs/output values, a borrowed
`[GroupedGemmJob]`, and shared accumulation. Jobs remain ordered and output
ranges are validated as pairwise disjoint before provider entry.

`CpuLayoutTransformRequest` contains one `TensorRead`, one reborrowed
`TensorWrite`, and an explicit materialization intent. Phase 1 has one intent,
`CanonicalColumnMajor`; metadata-only transpose/reshape/broadcast never calls
this provider.

`CpuDotGeneralRequest` contains borrowed lhs/rhs/output values,
`DotGeneralAccumulation`, and `CpuContractionAxes`. The axes value preserves
ordered contracting pairs, ordered batch pairs, and allocation-free lhs/rhs
free-axis iterators. These four role groups are the complete TBLIS-style label
relationship; providers do not reconstruct semantic roles from a GEMM plan.

No request contains `Vec`, `SmallVec`, `String`, `HashMap`, `TypeId`, `Any`, or
an owned tensor. Creating and dispatching a request performs no heap allocation.
Prepared owners and existing analysis caches may own `SmallVec` metadata and
lend slices into requests.

## Validation Ownership

`DotGeneralRuntime` performs common validation exactly once before output
mutation or provider invocation:

- lhs/rhs dtype and host placement;
- axis range, uniqueness, role disjointness, and ordered-pair counts;
- paired contracting and batch extents;
- output shape, dtype, strides, and accumulation scalar dtype;
- checked products, offsets, and reachable ranges.

Ranks up to 64 use four `u64` role masks. Higher ranks use an allocation-free
linear duplicate/disjointness check. Ordered input slices remain the semantic
source of truth in both cases. Tests compare the new validator against the
current validator for valid and invalid configurations through rank 70.

Provider-specific layout analysis occurs only after common validation.
GEMM/TBLIS analysis may return typed `Unsupported`; malformed input cannot be
reclassified as a capability miss.

## Runtime Composition and Resolution

The engine owns this immutable composition:

```rust
struct DotGeneralRuntime {
    general: Option<Arc<dyn CpuGeneralContractionProvider>>,
    gemm: Arc<dyn CpuGemmProvider>,
    layout: Arc<dyn CpuLayoutTransformProvider>,
    general_policy: GeneralContractionPolicy,
}

struct CpuProviderBundleInner {
    dot_general: DotGeneralRuntime,
}

#[derive(Clone, Debug)]
pub struct CpuProviderBundle {
    inner: Arc<CpuProviderBundleInner>,
}
```

Resolution order is fixed before execution:

1. Invoke the configured general-contraction slot when present.
2. On `Executed`, stop.
3. On `Unsupported`, either report a typed error for `Required` policy or use
   the already configured layout-plus-GEMM path for `Preferred` policy.
4. Any `Err` stops immediately.

There is no provider-name lookup, registry scan, downcast, or error-based retry.
TBLIS preferred/required modes become bundle configuration instead of branches
inside `CpuBackend` and `CpuExecSession`.

`CpuProviderBundle::builder(CpuBackendKind)` installs the built-in faer or BLAS
GEMM adapter and the strided-kernel layout adapter. Feature-gated builder
methods install TBLIS or an external `Arc<dyn ...>` provider. `build()` rejects
missing mandatory slots. `CpuBackend::with_provider_bundle` consumes the
backend and installs an immutable bundle. Mutable provider setters are removed;
provider replacement is construction-time only. Cloned backends share the
same bundle.

`CpuBackendKind` remains the convenience selection for operation families not
migrated in Phase 1. It no longer drives contraction branches after bundle
construction.

## Cache Identity

`GemmAnalysisCache` remains engine/session-owned. It binds to
`Weak<CpuProviderBundleInner>` on first contraction. Reuse requires upgrading
the weak pointer and `Arc::ptr_eq` with the executing bundle. A mismatch or a
dead weak target clears all slots before rebinding.

The retained `Weak` control block prevents pointer-reuse ABA, so no global
64-bit provider ID or exhaustion behavior is needed. The existing engine
resource lock serializes bind/clear/use. Alternating different bundles on
clones may invalidate the shared analysis cache but cannot reuse a plan under
the wrong provider.

## Existing-Path Migration

Both direct `CpuBackend` calls and graph `BackendSession` calls construct the
same `CpuExecSession` and delegate contraction to its borrowed
`DotGeneralRuntime`. The session receives the selected bundle once at session
construction. No operation creates a backend, session, pool, permit, or
provider registry.

The built-in extraction preserves current scheduling:

- grouped faer with multiple jobs and multiple threads uses engine-owned outer
  Rayon fan-out and sequential single-job provider calls;
- one grouped job or a one-thread context uses a sequential outer loop and may
  allow inner provider parallelism;
- strided-batched faer keeps a sequential batch loop with inner parallelism;
- provider/layout execution stays under the existing outer session lease.

The phase removes the duplicated `CpuBackendKind`/TBLIS contraction dispatch
from `backend.rs` and `exec_session.rs`. No temporary compatibility dispatcher
remains at phase end.

## Tests

Focused tests cover:

- request size and allocation-free construction/dispatch;
- object-safe custom GEMM, layout, and general-contraction providers;
- `Unsupported` leaves output unchanged and alone permits fallback;
- an execution error never invokes fallback;
- required and preferred general-contraction policies;
- cache reuse for one bundle and invalidation across distinct bundles;
- faer grouped and strided-batched scheduling without nested fan-out;
- eager and graph/session routes reaching the same provider slots;
- dtype, axis, shape, placement, output, and accumulation error parity;
- numerical parity for f32, f64, c32, and c64, including conjugation,
  transposed reads, accumulation, zero extents, grouped jobs, and strided output.

## Predeclared Non-Inferiority Protocol

The comparison uses the current-main eager benchmark introduced by Phase 0 and
adds `slice_f64` as the indexed family case before any candidate measurement.
The case matrix is lazy and materialized `neg_f64`, `add_f64`,
`reduce_sum_f64`, `slice_f64` at lengths 1/8/64 and `dot_general_f64` at matrix
sizes 1/2. Inputs, runtime construction, and output consumers remain outside
the timed loop except that the public operation's current config ownership is
included symmetrically.

The host protocol is:

- release profile, identical Rust toolchain and feature set;
- one fixed CPU selected from the process-allowed affinity mask;
- `RAYON_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, and `VECLIB_MAXIMUM_THREADS=1`;
- Criterion 2 s warm-up, 5 s measurement, 100 samples, 95% confidence;
- three complete baseline/candidate pairs, alternating order `A/B`, `B/A`,
  `A/B`;
- Criterion's per-pair 95% relative-change interval is the primary statistic;
- a case is `PASS` only when all three interval upper endpoints are at most
  +5%; it is `FAIL` when at least two interval lower endpoints exceed +5%; all
  other valid results are `INCONCLUSIVE`;
- the campaign is `PASS` only when every case is `PASS`, is `FAIL` when any
  case is `FAIL`, and is otherwise `INCONCLUSIVE`; median point-estimate ratios
  are recorded as diagnostics but do not override the interval rule;
- the allocation probe requires candidate allocations and allocated bytes per
  operation to be no greater than baseline for every case;
- source/behavior contract tests require direct typed slot dispatch and forbid
  hot-path `HashMap`, string lookup, `TypeId`, `Any`, and downcast.

A pair is invalid when the process loses its fixed affinity, any benchmark
process overlaps a Cargo/rustc process, the normalized one-minute load average
exceeds 0.25 of the process-allowed CPU count at either endpoint, or a case is
missing. Any invalid pair makes the whole experiment `INCONCLUSIVE`. A retry is
the complete three-pair experiment with unchanged settings; individual cases
or favorable pairs are never retried or selected.

## Phase Exit

Phase 1 exits only when correctness/provider-contract tests pass, the complete
performance experiment is `PASS`, allocation/string-dispatch gates pass,
normative parallelism/provider documentation and a worklog are updated, and no
temporary contraction dispatcher remains. A `FAIL` blocks promotion. An
`INCONCLUSIVE` result permits only a complete rerun under the fixed protocol.
