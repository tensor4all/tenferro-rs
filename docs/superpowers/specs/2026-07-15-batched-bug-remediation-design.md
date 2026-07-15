# Batched Bug Remediation Design

## Status

The six-issue batch and the one-PR strategy were approved interactively on
2026-07-15. This document records the proposed implementation boundary before
an implementation plan is written.

The work starts from `origin/main` at `50c6623d` and follows the repository
remediation workflow: one branch, coherent commits, one PR after local
verification, and a non-squash merge.

## Goals

- Fix issues #1275, #1276, #1368, #1375, #1381, and #1385 at their root
  contracts rather than adding compatibility shims around the defects.
- Use `docs/design/api-and-convention-freeze.md` as the canonical decision for
  #1276: unsuffixed methods operate on owned tensors, `_read` methods operate on
  borrowed views, and mutable output surfaces accept both owned tensors and
  mutable views.
- Preserve CPU/GPU numerical parity for NaNs, signed zero, infinities, and
  integer overflow semantics.
- Eliminate avoidable pool construction, environment reads, and tensor-read
  descriptor clones from repeated CPU execution.
- Keep each issue reviewable as a coherent commit while delivering a single
  remediation PR.

## Non-goals

- Adding a new backend, operation family, dependency, feature flag, or AD
  convention.
- Preserving public compatibility for APIs whose current shape is the confirmed
  bug. The remediation workflow explicitly prefers the canonical root contract.
- General CPU or GPU performance tuning beyond the repeated work identified by
  #1385.
- Expanding GPU integer support beyond the operation and dtype matrix already
  claimed by the repository.
- Rewriting historical documents under `docs/plans/`.

## Classification Ledger

| Issue | Classification | Current evidence | Planned disposition | Smallest verification target |
| --- | --- | --- | --- | --- |
| [#1375](https://github.com/tensor4all/tenferro-rs/issues/1375) | Auto Fix | SVD gauge batch count uses unchecked `product::<usize>()` and later derives batch offsets from it. | Introduce checked batch metadata and use it for all gauge batch traversal. | Overflow helper tests plus batched real/complex SVD gauge tests. |
| [#1368](https://github.com/tensor4all/tenferro-rs/issues/1368) | Auto Fix | CPU maximum/minimum explicitly propagate NaN; CubeCL elementwise and reduction paths use native `.max()`/`.min()` directly. | Route GPU maximum/minimum combines through shared NaN-propagating helpers. | f32/f64 elementwise and reduction parity for both NaN operand orders, signed zero, and infinities. |
| [#1381](https://github.com/tensor4all/tenferro-rs/issues/1381) | Verify First, then Auto Fix | Current CubeCL lowering wraps integer arithmetic, and tests observe wrapping, but source uses bare operators and does not state the contract. | Confine bare CubeCL IR operations to documented `wrapping_*` helpers and make all integer call sites explicit. | Existing overflow tests plus source-contract inventory and boundary-value cases. |
| [#1276](https://github.com/tensor4all/tenferro-rs/issues/1276) | Auto Fix | Typed tensor views implement unsuffixed einsum traits, and typed `_into` output accepts only `&mut TypedTensor<T>`, unlike the canonical owned/read and tensor/write vocabulary. | Split owned/read traits and add a typed write adapter accepting owned mutable tensors or mutable typed views. | Public-surface contract tests, doctests, owned/read execution, and both output forms. |
| [#1275](https://github.com/tensor4all/tenferro-rs/issues/1275) | Auto Fix | Public CPU free functions allocate a new local `BufferPool`, bypassing `CpuBackend`, `CpuContext`, placement, and reusable execution resources. | Remove the bypassing public surface and route documented usage through `CpuBackend` plus backend traits. | Public-surface inventory, migrated doctests, and backend pool-reuse tests. |
| [#1385](https://github.com/tensor4all/tenferro-rs/issues/1385) | Auto Fix | `BufferPoolLoan::new` uses `mem::take`, whose replacement constructs maps and reads the retention-limit environment variable; read GEMM internals consume descriptors and force callers to clone them. | Borrow the installed pool directly, cache the default environment-derived limit, and borrow GEMM read descriptors. | Allocation/overhead benchmarks, pool restoration tests, panic tests, and direct/read GEMM parity. |

The ledger will be reconciled against the issue tracker immediately before
starting each commit. A newly fixed or narrowed item will be marked stale or
narrowed rather than reimplemented from historical wording.

## Overall Delivery Strategy

The batch uses one branch and one final PR, but it is not one undifferentiated
change. Each commit has its own regression boundary and leaves the workspace in
a passing state. The recommended order starts with small correctness fixes,
then changes the public API, and finishes with the CPU hot-path refactor. This
keeps failures attributable and makes the last benchmark comparison measure the
final public execution path.

The final PR body will reproduce the classification ledger, link the work log,
state any GPU verification unavailable locally, and close only issues fully
resolved by the batch.

## #1375: Checked SVD Gauge Batch Metadata

The SVD gauge code must not treat a successful output-shape check as proof that
all later products and offsets fit in `usize`. A small internal preparation
helper will validate the `u`, singular-value, and `vt` batch shapes and compute:

- checked batch count;
- checked per-batch matrix spans;
- checked offsets or ranges used by the gauge loop.

The execution loop will consume this prepared metadata instead of recomputing
shape products. Errors remain typed linalg configuration/backend errors and are
reported before indexing or mutation. Zero-sized batches remain valid no-ops.

Tests will exercise synthetic overflowing dimensions through the preparation
helper so no impossible allocation is required. Existing batched f32, f64, c32,
and c64 gauge behavior will remain covered, including zero-batch and malformed
output cases. A source-contract test will prevent reintroduction of unchecked
batch `product()` in the gauge family.

## #1368: NaN-Propagating GPU Extrema

CubeCL elementwise and reduction code will share two generic helpers equivalent
to the CPU contract:

```text
maximum(a, b): if a is NaN return a; else if b is NaN return b; else native max
minimum(a, b): if a is NaN return a; else if b is NaN return b; else native min
```

Returning the encountered NaN avoids inventing host-side values and keeps the
operation device-native. Non-NaN values continue through CubeCL's native
maximum/minimum so signed-zero and infinity behavior is preserved.

The helpers will be used by:

- binary elementwise maximum/minimum;
- unit reductions;
- plane reductions;
- any adjacent combine stage found by the required neighborhood scan.

Feature-gated CUDA tests will compare GPU results with CPU results for f32 and
f64, NaN in either operand order and at different reduction positions, positive
and negative zero, infinities, unit reduction, and plane reduction. Source-only
contract tests remain useful on hosts without CUDA but do not replace runtime
CUDA verification.

## #1381: Explicit GPU Integer Wrapping

The repository already specifies wrapping integer arithmetic, and current
CubeCL lowering behaves that way. The defect is that bare operators at call
sites make the intended overflow contract implicit and vulnerable to a future
lowering or compiler change.

CubeCL does not currently expose Rust-style `wrapping_*` intrinsics for this
generic kernel layer. Therefore the implementation will centralize the
permitted bare IR expressions in small helpers named for the semantic contract,
for example `wrapping_add`, `wrapping_sub`, `wrapping_mul`, `wrapping_neg`, and
the wrapping accumulation helpers required by reduction and integer power.
Each helper will carry an `INVARIANT` comment tying the expression to verified
CubeCL lowering. Kernel call sites will use only these helpers.

Division, remainder, negative exponent, and other domain errors keep their
existing preflight validation; wrapping helpers must not weaken those checks.
Integer extrema and comparisons are not overflow operations and remain direct.

The implementation begins by compiling and running a minimal boundary probe for
the active CubeCL version. If that probe disproves wrapping lowering for any
operation, that operation stops at the design gate instead of being relabeled
with a misleading helper. Otherwise, i32/i64 tests cover min/max boundaries,
negative values, multiplication, negation, power, and reduction accumulation.
A source-contract inventory ensures new bare arithmetic cannot bypass the
documented helper set.

## #1276: Canonical Typed Einsum API

`docs/design/api-and-convention-freeze.md` is authoritative. The typed API will
match the dtype-erased API rather than preserving the current divergence.

### Input trait split

| Input surface | Trait | Method vocabulary |
| --- | --- | --- |
| Collections of owned `TypedTensor<T>` values or references to them | `TypedTensorEinsumExt<T>` | `einsum`, `einsum_subscripts` |
| Collections of `TypedTensorView<'a, T>` | `TypedTensorReadEinsumExt<T>` | `einsum_read`, `einsum_read_subscripts` |
| Owned typed inputs with caller-provided output | `TypedTensorEinsumIntoExt<T>` | `einsum_into`, `einsum_into_subscripts` |
| Typed views with caller-provided output | `TypedTensorReadEinsumIntoExt<T>` | `einsum_read_into`, `einsum_read_into_subscripts` |

Unsuffixed trait implementations for typed views will be removed. Compatibility
aliases are intentionally excluded because they would preserve the
inconsistency the issue asks to remove.

### Typed mutable output

`tenferro-tensor` will add a public `TypedTensorWrite<'a, T>` adapter with
conversion from both:

- `&'a mut TypedTensor<T>`;
- `TypedTensorViewMut<'a, T>`.

The adapter mirrors dtype-erased `TensorWrite` and exposes only the metadata and
conversion operations needed by backends. Typed einsum `_into` methods and
`ConcreteEinsumPlan::execute_typed_into` will accept any value convertible to
this adapter. Validation occurs before execution and checks output dtype, shape,
placement/residency requirements, and non-overlapping mutable layout through
the existing view constructors and backend boundary.

Every new public type, trait, and method receives runnable rustdoc examples.
Tests cover owned inputs, borrowed strided inputs, owned output, strided mutable
output, prepared-plan output, mismatched shape, and dtype/placement errors.
Public-surface contract tests ensure view implementations cannot regain
unsuffixed names.

## #1275: Remove CPU Resource-Bypassing Free Functions

The canonical CPU execution owner is `CpuBackend`, backed by `CpuContext` and
engine-owned pools/caches. Public module-level helpers that construct a fresh
`BufferPool` on every call create a second execution model and cannot be made
consistent merely by hiding a global default backend.

The fix will:

- remove the crate-root free-function reexports that bypass `CpuBackend`;
- make the corresponding module wrappers private or crate-private when they
  are no longer part of the supported surface;
- delete duplicated `with_local_pool` helpers;
- retain the pool-aware internal kernels used by `CpuBackend` and sessions;
- migrate rustdoc, README, guide, tutorial, and tests to explicit backend-trait
  calls or higher-level tensor extension methods.

No process-global default backend will be introduced. Callers that care about
threading, placement, provider choice, caches, or buffer reuse must keep a
backend/context handle. A public-surface contract test will inventory the
removed names, while behavior tests will show repeated calls reuse the selected
backend's pool.

## #1385: Steady-State CPU Install And GEMM Overhead

### Borrow execution resources without placeholder construction

`BufferPoolLoan` currently moves the pool out with `mem::take`; constructing the
temporary replacement creates fourteen `BTreeMap`s and rereads the environment.
The loan will instead hold a direct mutable borrow of the engine-owned pool.
Panic recovery remains the responsibility of pool in-flight accounting and the
existing session/resource guard; no empty placeholder pool is required.

Tests will preserve the current guarantees that buffers are returned after
normal execution and replenished/restored after panic. The neighborhood scan
will cover `install_with_pool`, `run_with_pool`, linalg pool access, and backend
sessions so no sibling path retains `mem::take` solely to satisfy borrowing.

### Read the default retention limit once

Standalone `BufferPool::new()` still honors
`TENFERRO_CPU_BUFFER_POOL_MAX_RETAINED_BYTES`, but the parsed process default is
stored in `OnceLock`. Explicit constructors and runtime limit setters remain
fully dynamic. Tests that mutate the environment will target the parsing helper
directly or use subprocess isolation rather than depending on resettable global
state.

### Borrow GEMM descriptors

Internal Faer, BLAS, and TBLIS read/into entry points will accept borrowed
`&TensorRead` descriptors where they do not take ownership. `CpuBackend` and
`CpuExecSession` will pass their existing descriptors by reference instead of
cloning them for each dispatch. Ownership is retained only at boundaries that
actually consume a tensor value.

The change includes direct, conjugated, cached, accumulated, grouped, backend,
and session paths found by the neighborhood scan. Tests cover owned tensors,
strided views, f32/f64/c32/c64, output accumulation, and provider-specific
feature builds.

### Performance acceptance

Existing `cpu_install_overhead` and `grouped_gemm` Criterion benchmarks will be
extended or supplemented with a warmed repeated-dispatch case. The acceptance
criterion is structural and measurable:

- no environment lookup or empty `BufferPool` construction per install;
- no `TensorRead` descriptor clone per direct GEMM dispatch;
- no regression in pool retention/restoration behavior;
- before/after benchmark evidence recorded in the work log and PR body.

Wall-clock improvement is expected but is not encoded as a fragile universal
percentage threshold.

## Commit Structure

1. `fix(linalg): validate SVD gauge batch layout` — #1375.
2. `fix(gpu): make numeric edge semantics explicit` — #1368 and #1381 after
   the wrapping probe succeeds.
3. `fix(einsum): align typed read and write surfaces` — #1276.
4. `refactor(cpu): remove pool-bypassing free functions` — #1275.
5. `perf(cpu): eliminate steady-state install allocations` — #1385.
6. `docs: record batched remediation results` — durable docs, final ledger,
   work log, benchmark evidence, and residual risks.

If #1381 fails its lowering probe, commit 2 contains #1368 only and the ledger
records #1381 as design-gated with the precise failing operation. It must not
block the other five confirmed fixes.

## Verification

Each commit receives targeted tests before moving to the next issue. The final
committed head must pass the repository checklist:

```bash
cargo fmt --all --check
cargo test --workspace --release
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
python3 scripts/repository-rules-review.py \
  --base origin/main \
  --head HEAD \
  --output-json /tmp/repository-rules-review.json
```

Clippy will use the exact command and feature set from the current CI job.
Changed README/getting-started samples will be extracted and run. CUDA runtime
tests will run on an available CUDA host; if no local CUDA host is available,
the work log and PR body will distinguish source-only verification from the
required remote GPU check rather than claiming runtime coverage.

Before the PR is opened, the implementation will also receive a side review
against `REPOSITORY_RULES.md`, a local LLM review on the committed head, and a
final reconciliation with all six issue threads.

## Documentation And Review Artifacts

The implementation will add one work log under `docs/worklogs/` containing:

- final issue classifications and working-hash evidence;
- context and rules read;
- chosen and rejected alternatives;
- per-commit verification;
- CPU benchmark commands and before/after results;
- GPU runtime environment and results;
- residual risk and follow-up issues.

Durable API wording affected by #1276 and CPU execution wording affected by
#1275/#1385 will be updated in active design/guides, not only in the work log.

## Primary Risks And Mitigations

- **Public API churn:** #1275 and #1276 intentionally remove inconsistent
  surfaces. Compile-time public-surface tests and fully migrated docs make the
  new contract explicit.
- **Borrow checker pressure in CPU sessions:** replace resource movement in the
  smallest scope and preserve existing panic guards; do not introduce unsafe
  aliasing to avoid a refactor.
- **CubeCL semantic assumptions:** require the #1381 lowering probe and runtime
  boundary tests before labeling helper operations as wrapping.
- **GPU availability:** separate source-contract evidence from actual CUDA
  execution and record exactly which was run.
- **Batch review size:** keep issue-scoped commits, a live ledger, and one work
  log so reviewers can inspect or revert a unit without losing the one-PR
  remediation workflow.
