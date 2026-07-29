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
- Static outputs are acquired zero-initialized from the CPU buffer pool before
  safe typed slices and erased descriptors are
  constructed. This avoids exposing the pool's legacy typed-uninitialized
  acquisition to the new paths.
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
workspace strided dependencies are pinned to merged commit `e18ecb10`. The
final exact-row comparison used the equivalent #171 kernel state, 15 measured
runs after three warmups, t1 pinned to CPU 60, and t4 pinned to CPUs 56-59:

| Operation | Backend | t1 | t4 |
|---|---|---:|---:|
| slice | eager | -86.4% | -85.5% |
| slice | trace | -85.7% | -86.3% |
| pad | eager | -92.2% | -92.2% |
| pad | trace | -92.1% | -92.7% |
| concatenate | eager | -94.9% | -94.6% |
| concatenate | trace | -94.9% | -94.6% |
| reverse | eager | -90.4% | -88.9% |
| reverse | trace | -90.4% | -89.5% |
| tril | eager | -27.3% | -22.7% |
| tril | trace | -27.7% | -23.9% |
| triu | eager | -29.3% | -23.3% |
| triu | trace | -28.2% | -20.8% |

Every predeclared row improved; no row approached the +20 percent blocking
threshold. The measurements used the same benchmark binaries and machine
profile for each baseline/candidate pair. PR #173 changes compile-time
validation only; the final pin is rechecked by the fixed allocation gate and
the complete CPU test suite.

## Independent review follow-up

The final review identified that the pre-existing `PoolScalar::pool_acquire`
API can expose typed uninitialized storage before a kernel completes. The new
static indexing paths were changed to the safe zero-initialized acquisition
path before descriptor construction. The repository-wide `MaybeUninit` owner
and handoff redesign is tracked separately in #1516 so that it can cover every
legacy caller without hiding a broad safety change inside this performance PR.

The same review found unchecked arithmetic in the inline stride helper. The
helper now returns typed validation errors on conversion or multiplication
overflow, and tests cover overflow, a zero extent followed by an unreachable
large extent, and rank nine spilling beyond the inline capacity.

The eight static-indexing rows in the table were remeasured after the safe
acquisition change with the same 15-run/three-warmup protocol and CPU binding.
The additional initialization cost reduces part of the gain but every row
still improves by 85.5 to 94.9 percent relative to `origin/main`.
