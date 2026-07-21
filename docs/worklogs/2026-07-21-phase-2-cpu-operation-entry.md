# Phase 2 CPU operation-entry type state

This worklog records the Task 6/8 refactor that separates CPU executor entry,
logical parallel mode, and provider worker ownership. It implements the
reviewed issue #1433/#1436 contract without implementing Task 7 provider
count/placement classification. The scoped strided execution policy comes from
reviewed revision `6b0b4a46b7dd9a9ea1677a0d596c0b4adab1acbc`.

## Context reviewed

- `AGENTS.md`, `REPOSITORY_RULES.md`, and the shared common, Rust,
  performance, and documentation/test rules.
- `docs/design/execution-engine-provider-architecture.md` and
  `docs/design/cpu-backend-execution.md`.
- The existing `CpuDomainExecutor`, resource-domain, arbiter, backend-session,
  dot/grouped-provider, linalg interop, allocation-audit, and packed grouped-job
  implementations and tests.

## Implemented boundary

- Added crate-private `CpuOperationEntry<'a>`, which holds the selected
  `CpuResourceDomain` and `ResourcePermit`. It alone owns checked executor
  `install` and `submit`.
- Narrowed public `CpuExecutionContext<'a>` to an always-already-entered
  borrowed policy value. Its public accessors report domain ID, CPU set, thread
  budget, placement guarantee, logical mode, and the hidden faer policy. It no
  longer exposes install, submit, mode mutation, child construction, permit
  ownership, or a context-specific scheduling error.
- Sequential and inner operation entry call executor `install` exactly once.
  Inner entry is not rejected merely because the executor reports no Rayon
  inner region: an external-worker provider may own fan-out after entry.
- Outer execution calls executor `submit` exactly once. It exposes at most
  `min(thread_budget, logical_job_count)` composite lanes to the executor; lane
  `k` runs logical indices `k, k + lane_count, ...` sequentially, constructing
  an already-entered Sequential context for each logical child. Passing Outer
  to the install entry is rejected before executor or operation mutation.
- `CpuExecSession` stores an entry rather than a provider context. Direct,
  session-native, dot/general/grouped, and `with_linalg_pool` execution all use
  the same operation boundary. The prior Managed-BLAS direct bypass was
  removed.
- Executor failures now remain a direct typed `CpuDomainExecutorError` source
  instead of being wrapped in the removed public
  `CpuExecutionContextError`.
- The entered context is the single native-policy authority. `Inner` work whose
  selected executor advertises Rayon uses only that executor up to the
  operation budget; Sequential, external-worker Inner, and every Outer child
  run native/layout kernels sequentially. The same scope wraps direct, session,
  materialization, and linalg operation families and is restored after ordinary
  errors or panic.

The preserved call-count table is:

| Operation selection | install | submit | Provider context |
|---|---:|---:|---|
| Sequential | 1 | 0 | Sequential |
| Inner, engine workers | 1 | 0 | Inner |
| Inner, external workers | 1 | 0 | Inner |
| Outer with N jobs | 0 | 1 | N Sequential logical children over at most `min(N, budget)` submitted lanes |

## TDD evidence

The focused RED test
`external_worker_inner_entry_does_not_require_rayon_capability` first failed
against the old context-owned entry with `InnerUnsupported`. After the
type-state boundary was introduced, focused tests established:

- ExternalWorkers with `inner_parallelism = None`: install 1, submit 0,
  provider call 1;
- actual selected Rayon pool: install 1, submit 0, Inner context, current pool
  size 2;
- outer execution: submit 1, install 0, every child Sequential;
- an external executor advertising four workers with a domain budget of two:
  one submit containing two lanes, maximum two concurrent logical operations,
  and six logical indices each executed exactly once;
- budget one rejects Outer before submit/install/mutation, while a budget at
  least as large as the job count submits exactly the job count with no extra
  logical calls;
- invalid Outer install selection: typed error before executor/operation
  mutation;
- executor install rejection: one install, zero submits, zero provider calls,
  unchanged output, and the original typed executor error;
- a custom executor that actively rejects reentry: direct install,
  direct-native, backend-session native, dot provider, and linalg interop each
  add exactly one install and zero submits;
- source contract: no provider-context install/submit/mode mutation, no session
  provider-context storage, and no Managed-BLAS direct bypass.

At the first complete CPU unit-test boundary after the refactor,
`cargo test -p tenferro-cpu --lib` passed all 439 tests. The subsequent focused
call-count, selected-Rayon, and source-contract tests also passed. Final
workspace verification is recorded in the task handoff rather than freezing
machine-specific build output here.

## Phase 1 BLAS regression repair

The operation-entry review exposed a pre-existing Phase 1 regression relative
to main: valid BLAS contractions stopped after a direct GEMM capability miss
such as `Layout(Lhs)`, `Layout(Rhs)`, or `Conjugation`. The smallest reproducer
was:

```text
RUSTFLAGS='-l dylib=openblas -l dylib=lapack' \
  cargo test -p tenferro-ad --profile ci --no-default-features \
  --features cpu-blas --test integration \
  cpu_backend::test_vector_dot_product -- --exact --nocapture
```

The regression is now repaired without changing provider capabilities. Only
`Layout(Lhs)`, `Layout(Rhs)`, and `Conjugation` trigger canonical operand
materialization followed by one retry of the same provider. Conjugation is
fused into materialization, retry flags are cleared, alpha/beta are retained,
all temporaries are reclaimed, and output layout, typed error, or a second
unsupported result remains terminal. The real OpenBLAS reproducer now passes.
See
[`2026-07-21-phase-1-blas-provider-fallback.md`](./2026-07-21-phase-1-blas-provider-fallback.md)
for the focused evidence.

## Deferred work and residual risk

Task 7 must replace the temporary backend-kind BLAS selection with explicit
provider count/placement capabilities and construction-time compatibility
validation. This refactor deliberately does not claim that
`ParallelMode::Inner` implies Rayon support or that an external BLAS runtime's
workers satisfy a strict placement domain. It establishes the non-bypassable
entry point on which that validation can rely.

## Linalg single-entry review follow-up

The nine public CPU linalg borrowed-read entry points now acquire the executor
and buffer pool exactly once. Input canonicalization is exposed only as
`CpuExecutionContext::with_materialized_tensor_read`: external code cannot
construct that receiver, and the raw materializer remains crate-private.
Owned tensors are borrowed directly; materialized views are returned to the
same pool after both successful closure completion and ordinary errors. The
two-input operations nest this scope so failure while materializing the second
input still returns the first temporary. Panic recovery retains the existing
pool-loan guarantee; retaining the exact in-flight temporary across unwinding
is intentionally not part of this follow-up.

Focused tests seed the pool and observe both retained capacity and allocation
identity. They cover success followed by immediate reuse, a typed numerical
error, and failure of the second input materialization. The source contract
also keeps the nine read-method entry count at one and rejects reintroduction
of a free public materialize or reshape helper.

The source-contract function slicer remains dependency-free because `syn` was
not an existing direct dev-dependency of `tenferro-linalg`. Its brace matcher
was instead upgraded to mask line comments, nested block comments, normal and
raw strings, and character literals while preserving byte offsets. A mutation
test places unmatched braces in every one of those forms and verifies that the
next function is not absorbed into the selected section.

The managed `cholesky_read` path is included in that same boundary. Before the
follow-up, nonzero managed input mapping happened outside the operation entry,
factorization entered later, and output allocation/write happened after it;
the zero-size path never entered the executor. The helper now accepts the
already-entered `CpuExecutionContext` and `BufferPool`, so validation, input
map, zero-size handling, provider work, shared-domain allocation, dtype
adaptation, and output map/write all remain below one `with_linalg_pool` call.
The owned `cholesky` surface shares this helper to preserve surface parity.

The RED managed-domain probe observed three storage operations outside entry
for the nonzero case and zero installs for the empty case. The GREEN focused
run covered both sizes with `install = 1`, `submit = 0`, no observed
allocation/map outside entry, correct factor/empty output, and a successful
next operation. The full `single_entry` module then passed 15 tests and the
managed-Cholesky module passed all 3 tests. The path-sensitive source contract
now requires entry before managed dispatch, rejects an early/fallible prefix,
and verifies that zero-size, factor, allocation, and output adaptation stay in
the non-entering managed helper.

These correctness results do not classify the Phase 2 eager performance gate,
Phase 2D, or Phase 2E. Their status remains unchanged until the umbrella
acceptance evidence is complete.

Fresh follow-up verification covered the complete linalg package under all
three CPU provider feature combinations: default faer (110 unit, 113
integration, 59 doctests), BLAS-only with OpenBLAS/LAPACK linkage (108 unit,
114 integration, 59 doctests), and combined faer+BLAS (112 unit, 114
integration, 59 doctests). The complete default CPU package also passed 462
unit tests plus its integration and 180 doctests. Default, BLAS-only, and
combined `cargo clippy` runs for `tenferro-cpu` and `tenferro-linalg` passed
with `-D warnings`; the docs-site checker, focused rustdoc build, and
`cargo fmt --all -- --check` also passed. These are correctness and build
results only, not performance-gate evidence.
