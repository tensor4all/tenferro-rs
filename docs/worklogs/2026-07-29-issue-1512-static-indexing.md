# Issue 1512: static indexing plan adoption

## Summary

Replaced the local CPU per-element implementations of slice, pad,
concatenate, and reverse with the corresponding strided-kernel erased plans.
Triangular masking now fills contiguous column runs instead of performing
per-element index classification.

## Context read

- `REPOSITORY_RULES.md`, especially performance evidence, explicit CPU
  execution context, destination aliasing, and upstream ownership
- tenferro-rs issues #1490 and #1512
- strided-rs issues #149, #160, and #170
- the existing CPU backend, execution-session, buffer-pool, static-indexing,
  and structural implementations

## Design

- `CpuBackend` and `CpuExecutionSession` pass their pooled
  `ExecContext` explicitly to every erased static plan replay.
- Static outputs are acquired as `Vec<MaybeUninit<T>>` from the CPU buffer
  pool. The upstream full-overwrite descriptor never constructs `&mut [T]`
  before replay, and converts the owner to `Vec<T>` only after success.
- Short-rank column-major strides use inline `SmallVec` storage with checked
  `usize` to `isize` conversion and multiplication. A zero extent keeps all
  subsequent strides at zero without converting unreachable large extents.
- Slice, pad, concatenate, and reverse preserve the upstream plan's typed
  shape, stride, offset, and overlap validation.
- Triangular operations retain their local semantics but classify one boundary
  per matrix column, copying the kept run and filling the masked run.
- Tests cover Bool values, empty dimensions, rectangular and batched
  triangular tensors, and nonzero diagonal offsets.
- Source-contract tests now require delegation to upstream erased plans rather
  than the removed local segment-prefix and per-element loops.

## Stop-the-line and upstream fix

The first adoption benchmark found that scalar `ErasedPadPlan` replay regressed
the exact public API pad row at one thread by about 60 percent. Work stopped
before opening the tenferro PR. The missing contiguous-run coalescing was filed
as strided-rs #170 and implemented upstream as compile-time run metadata with a
generic strided fallback.

Exact `cpu/indexing_layout/pad` measurements used 15 measured runs after three
warmups:

| Row | Pre-adoption | Scalar adopted | Run-coalesced |
|---|---:|---:|---:|
| eager t1 | 16.103 ms | 25.948 ms | 1.078 ms |
| trace t1 | 15.895 ms | 25.449 ms | 1.133 ms |
| eager t4 | 16.127 ms | 4.873 ms | 1.096 ms |
| trace t4 | 17.226 ms | 4.800 ms | 1.126 ms |

strided-rs PR #171 added contiguous pad runs. The fixed tenferro allocation
gate then exposed per-call exact injectivity-check allocation during static
plan compile; strided-rs PR #173 moved the existing disjoint-stride proof
ahead of exact enumeration and added checked offset/fusion arithmetic. All
workspace strided dependencies are pinned to merged commit `ed3053a6`, which
also includes PR #175's safe uninitialized full-overwrite replay. The final
exact-row comparison used that merged state, 15 measured runs after three
warmups, t1 pinned to CPU 60, and t4 pinned to CPUs 56-59:

| Operation | Backend | t1 | t4 |
|---|---|---:|---:|
| slice | eager | -89.7% | -89.0% |
| slice | trace | -89.2% | -86.3% |
| pad | eager | -93.5% | -93.8% |
| pad | trace | -93.6% | -88.8% |
| concatenate | eager | -96.6% | -96.3% |
| concatenate | trace | -96.3% | -95.6% |
| reverse | eager | -92.7% | -91.4% |
| reverse | trace | -92.6% | -91.7% |
| tril | eager | -28.3% | -25.6% |
| tril | trace | -27.2% | -24.8% |
| triu | eager | -27.9% | -25.1% |
| triu | trace | -28.9% | -23.7% |

Every predeclared row improved; no row approached the +20 percent blocking
threshold. The measurements used the same benchmark binaries and machine
profile for each baseline/candidate pair. PR #173 changes compile-time
validation only; the final pin is rechecked by the fixed allocation gate and
the complete CPU test suite.

## Independent review follow-up

The final review identified that the pre-existing `PoolScalar::pool_acquire`
API can expose typed uninitialized storage before a kernel completes. The
temporary zero-initialized fix violated the full-overwrite materialization
rule, so strided-rs #174 / PR #175 added a narrow
`ErasedRawStridedUninitMut` contract. The CPU pool now hands these four paths
`Vec<MaybeUninit<T>>` directly and remains safely droppable on error or panic.
Issue #1516 continues to track the remaining legacy typed-uninitialized
callers outside this static indexing scope.

The same review found unchecked arithmetic in the inline stride helper. The
helper now returns typed validation errors on conversion or multiplication
overflow, and tests cover overflow, a zero extent followed by an unreachable
large extent, and rank nine spilling beyond the inline capacity.

All rows in the table were remeasured after the safe acquisition change with
the same 15-run/three-warmup protocol and CPU binding. Static indexing improves
by 86.3 to 96.6 percent and triangular masking by 23.7 to 28.9 percent relative
to the pre-adoption baseline.
