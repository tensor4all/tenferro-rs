# Issue 1511: CPU indexed-plan cache

## Summary

Added an engine-owned bounded cache for the strided erased gather, additive
scatter, dynamic-slice, and dynamic-update-slice plans used by public eager and
traced CPU execution.

## Context read

- `REPOSITORY_RULES.md`, especially cache ownership, CPU threading, benchmark
  protocol, and public API documentation
- tenferro-rs issues #1490 and #1511
- strided-rs issue #149 and the merged erased indexed-plan implementations
- `tenferro-cpu` engine resources, backend sessions, runtime cache-owner
  adapter, buffer-pool controls, GEMM analysis cache, and indexed adapters

## Design

- Owner and lifetime: one cache in each `CpuEngine::EngineResources`, shared by
  clones of the owning `CpuBackend` for that engine's lifetime.
- Bounds: 256 entries and 8 MiB logical retained payload per engine by default.
  Either zero bound disables retention.
- Key: family, value dtype, index dtype, all operand/index/update/output
  dimensions and strides, and the complete operation configuration.
- Lookup allocation: rank/config metadata uses nested `SmallVec` values, so
  normal rank-at-most-eight calls do not allocate a temporary heap key.
- Values: compiled erased plans are retained behind `Arc`; replay clones only
  the `Arc`, not the plan metadata.
- Eviction: the repository-standard `lru` implementation, enforced in constant
  time for both entry and retained-byte bounds.
- Accounting: key and plan headers, spilled key capacity, and a conservative
  logical charge for the upstream plan's cloned/derived vectors. Tests cover
  inline and spilled-rank keys. Stats include hits, misses, evictions, clears,
  entries, and retained bytes.
- Controls: `CpuBackend` exposes cache-specific limits, aggregate stats, and
  clear methods. Existing `RuntimeCacheOwner` stats and clear include the
  current engine's indexed cache. Updated limits are inherited by lazily
  created managed engines. A single configuration lock serializes limit
  updates with lazy engine creation, so readers and new engines cannot observe
  mixed entry/byte bounds.

No process-global or thread-local cache was introduced. Additive scatter replay
remains serial as required by strided-rs #164.

## Verification and performance

Focused correctness tests cover all four public backend families sharing the
cache, compile-once reuse through direct and execution-session paths,
family/config keys, LRU eviction, byte eviction, coherent limit snapshots
across clones, inline/spilled retention charges, clear/configure controls, and
runtime cache-owner aggregation.

The benchmark source was `tenferro-benchmark`'s public API suite with
`PUBLICATION_GATE_PROFILE=full`, rows
`gather,scatter,dynamic_slice,dynamic_update_slice`, and thread counts 1 and 4.
The baseline was tenferro commit `60837e8f`; the candidate used this worktree.
Two complete counterbalanced orders were run because three warmups did not
remove a reproducible first-binary/cold-worker bias from the first t4 eager
gather row. The comparison statistic below is the geometric mean of the two
candidate/baseline ratios:

| Row | t1 | t4 |
|---|---:|---:|
| gather eager | +2.7% | -0.1% |
| gather trace | +2.1% | +0.7% |
| scatter eager | +1.1% | +1.0% |
| scatter trace | -0.8% | +0.2% |
| dynamic_slice eager | -0.6% | -0.1% |
| dynamic_slice trace | +10.3% | -0.2% |
| dynamic_update_slice eager | +0.9% | -2.3% |

All rows remained below the predeclared +20% blocking threshold. An earlier
run was classified `INCONCLUSIVE` after a competing release compile was
discovered during measurement; none of its values were promoted. The final
reviewed implementation was remeasured with t1 pinned to CPU 60 and t4 pinned
to CPUs 56-59 because three unrelated Julia processes were active elsewhere
on the 64-core host. Both baseline/candidate orders are retained. The t1
dynamic-slice trace row remained the noisiest result (+15.0% and +5.7% in the
two orders); the matching eager and t4 trace rows did not regress.

A same-binary alternating attribution probe measured t4 gather at 1.288 ms
with retention enabled and 1.300 ms with retention disabled. Plan compilation
therefore does not explain the remaining order-of-magnitude public API gap.
Source inspection attributes the residual to execute-time coordinate decode
and checked per-element offset reconstruction in the upstream erased indexed
plans. That kernel work is intentionally refiled in strided-rs rather than
being optimized inside the tenferro adapter.

## Deferred

- Contiguous-run/rank-one and general indexed execute-loop optimization belongs
  to strided-rs and requires its own benchmark-backed PR.
- Static slice/pad/concatenate/reverse and row-run triangular work remain in
  tenferro-rs #1512.
