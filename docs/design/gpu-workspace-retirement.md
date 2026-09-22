# cuTENSOR workspace retirement without a stream barrier

Status: implemented on `perf/cutensor-workspace-retirement`
(`WorkspaceRetirementQueue` in `crates/tenferro-gpu/src/cubecl/workspace_retirement.rs`).

Plan eviction keeps the shared scratch allocation alive. Scratch growth, a
lowered retention cap, whole-cache eviction, clear, and teardown retire the
allocation through this mechanism. The historical measurements below
describe the pre-sharing design; shared scratch is excluded from cache byte
accounting and bounded by its own retention cap, see
[current ownership and tradeoffs](gpu-backend-design.md).

## Problem

`gpu/tensornetwork` trace-mode execution spends 54% of each call with the
device idle between kernels (measured on the A100 GPU benchmark suite, see the
linked work log). Attribution shows the idle gaps come from stream barriers, not
from launch latency or host dispatch:

- `synchronize_raw_stream` is called ~226 times per call (2944 per run for 13
  calls). Its only non-explicit caller is `Workspace::drop`
  (`crates/tenferro-gpu/src/cubecl/gemm.rs`), reached through
  `CachedCutensorContraction` eviction from the cuTENSOR plan cache.
- Every barrier drains the pipeline, which is what produces the ~32 us median
  inter-kernel gap.
- The default cuTENSOR plan cache holds 64 entries while this workload has 74
  unique contraction specs, so eviction happens continuously.

`Workspace::drop` synchronizes because `docs/design/gpu-backend-design.md`
records that CubeCL raw pointers have no completion witness (issue #967
invariant): releasing the CubeCL handle returns the block to the pool, and
without proof that the enqueued vendor work finished, a later allocation could
receive the same block while cuTENSOR still writes it.

## Proposal

Give the workspace a completion witness instead of a barrier.

1. `Workspace` already records the exact stream it was enqueued on
   (`alloc_workspace` stores `rt.raw_cuda_stream()`, and workspaces are indexed
   by physical stream slot).
2. On eviction, record a CUDA event on that stream after the last enqueue and
   hand `(event, CubeCL handle, runtime)` to a runtime-owned retirement queue
   instead of synchronizing.
3. Drain the queue opportunistically: query pending events; when an event
   reports completion, release its handle. Reuse the existing event primitives
   and retirement state machine already used by the CUDA event domain
   (`crate::cubecl::event_domain`, `crate::event_retirement`).
4. Bound the queue. If the queue is full, or if recording or querying the event
   fails, fall back to today's behavior: synchronize the stream, or forget the
   handle and leak rather than race. The #967 invariant is preserved because a
   handle is only released after its event reports completion.

Correctness argument: `cuEventRecord` on a stream completes only after all
previously enqueued work on that stream completes. Recording at eviction time
and releasing the handle after completion means no in-flight cuTENSOR work can
reference a block that has been returned to the pool. Draining happens under the
plan-cache mutex on the enqueue thread, which already holds the primary context.

## Result

`gpu/tensornetwork` on the A100 benchmark container, three repetitions per
configuration, all verifications passing:

| Variant | Before | After |
| --- | ---: | ---: |
| `tenferro-cuda-trace` | 75.7-80.1 ms | 68.0-69.5 ms |
| `tenferro-cuda-eager` | 81.0-83.5 ms | 72.8-74.5 ms |

nsys on the same problem: `cudaStreamSynchronize` 2121 -> 27 calls,
`cuEventSynchronize` 162 calls (0.02 per kernel, the deferred waits), median
inter-kernel gap 32.5 us -> 28.4 us, kernel busy time unchanged (469.9 ->
462.8 ms). The whole `nvidia-gpu` set (standard suite plus both linalg AD
suites) reported 156 ok / 51 unsupported / 0 failures, the ignored CUDA library
tests 136 passed, and no row regressed outside the noise of the sub-millisecond
latency suite.

The remaining 28.4 us median inter-kernel gap is not retirement: the next
candidates are cubecl-cuda's per-command `cuCtxSetCurrent` (an upstream TODO)
and launch latency itself.

## Open decisions

Decisions taken for this change:

1. Drain point: **decided** — the retirement queue is drained immediately
   before allocating a new workspace, plus on explicit `synchronize()`, cache
   clear, and teardown. Memory pressure only exists where a new allocation
   happens, and evictions (which enqueue retirements) and allocations are the
   same event: a new contraction spec evicts the LRU entry and then needs a
   workspace. Draining anywhere else is either wasted work or unsafe for
   memory:
   - Per-contraction drains add an event query to the hot path for no bound
     improvement over the allocation-time drain.
   - Call-boundary drains would hold every retirement of the call: this workload
     evicts ~226 times per call, so the queue would hold ~226 workspaces at once
     and force CubeCL pool growth instead of reuse.
2. Queue bound and fallback: **decided** — `DEFAULT_WORKSPACE_RETIREMENT_CAPACITY`
   is 16; at capacity the oldest retirement is resolved with an event wait, and
   a retirement whose event cannot be created or recorded falls back to the old
   stream barrier, leaking the handle if even that fails.
3. Whether to also raise `DEFAULT_CUTENSOR_PLAN_CACHE_MAX_ENTRIES` (64): still
   open. Raising it to 512 in a single A/B cut barrier calls by 33% but did not
   improve the median while retirement was synchronous; revisit now that
   retirement is deferred.
4. Scope: still open. The same "no completion witness" reasoning also applies
   to the cuFFT work area and the cuTENSOR permutation plans; they are not part
   of this change.

## Verification plan

- Numerical: force plan-cache eviction between two contractions (or set a small
  cache bound) and assert results still match the CPU reference; cover the
  fallback path (event record/query failure) with the leak-rather-than-race
  behavior. Implemented as
  `test_workspace_retirement_defers_eviction_barrier_f64` (plan cache bound 1,
  two contractions, counters asserted, then an explicit barrier releases the
  queue).
- Accounting: `cuda_extension_cache_stats` retained-byte estimates cover
  cache-owned payloads only. Shared cuTENSOR scratch is outside them and is
  reported by `CudaBackend::cutensor_workspace_stats`; a workspace that is
  retiring or in flight is counted by neither, so neither statistic is a
  device-memory total.
- Performance: at least three repetitions per configuration (single-run variance
  is about 5%), reporting the median plus nsys kernel-busy/span and the
  inter-kernel gap distribution, for `gpu/tensornetwork` trace and eager.
  Counters are exposed through
  `CudaBackend::cutensor_workspace_retirement_stats`.
