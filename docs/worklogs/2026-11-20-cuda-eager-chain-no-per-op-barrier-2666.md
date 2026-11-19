# CUDA eager elementwise chain: no stream barrier per operation (#2666)

## Decisions

- An operation's launch no longer proves its own completion. `CudaEngine::prove_launch`
  used to end a standalone launch with `CudaRuntime::synchronize`, which records a
  completion checkpoint and waits on it, between every pair of launches. The eager
  arm of `gpu/elementwise` measured 1.009 ms against 0.348 ms traced and 0.313 ms
  PyTorch on the A100; it now returns from the launch and lets a download or the
  caller's barrier observe the work, which is 0.520 ms, with the traced arm unchanged.
  Detached transfer staging is still released after every launch, by polling its own
  completion instead of waiting for the stream.
- The rule is the one `docs/guides/devices-and-gpu.md` ("Eager GPU Synchronization")
  already stated and the one a scheduled run already followed after #2382: the
  result is a device-resident handle, a download waits for its own copy, and a
  caller that needs a host-side edge calls `ExecutionEngine::synchronize`.
- The completion-deferral scope the scheduled path used (`deferred_completions`,
  `defer_completion`, `CompletionDeferral`, `completion_deferred`) and the
  `launch_completion_syncs` counter existed only to distinguish "scheduled" from
  "standalone" completion. With one rule there is nothing to distinguish, so they
  are gone; `CudaRuntime::stream_waits` is the counter that stays meaningful, and the
  tests that pinned the removed counter now pin real waits.
- Rejected: making the eager session the deferral scope (one barrier when the session
  releases the engine) and keeping a standalone launch's drain. `EagerSession` holds
  only `&mut dyn ExecutionEngine`, so that needs a new method on a public trait, and it
  leaves two completion rules inside one engine.
- Not changed: the eager path still submits one kernel per arithmetic operation (41
  launches for the 24 operations of an 8-step chain, the rest being its own copies).
  Fusing the chain is the prepared path's job and a different change; the residual
  0.520 vs 0.344 ms is that path's per-operation host bookkeeping, which is #2079's
  subject, not this one.
- `docs/internals/cuda-surface-inventory.md` had drifted: three items added after
  #2579 had no verdict, so its generator refused to run and the table was stale. The
  three now carry verdicts, so the table is regenerated rather than hand-edited.

## Verification conclusions and constraints

- A 24-launch eager chain now counts zero stream barriers of its own and exactly one
  for the caller's `synchronize`, and it is numerically identical to the same
  arithmetic on the CPU (`rtol 1e-9`, pinned by
  `tenferro-ad::integration::gpu_eager_chain_no_stream_drain`).
- What is no longer guaranteed: a device fault is reported at the next barrier rather
  than at the launch that caused it, which was already true for a scheduled run.
- Pinned staging for a transfer is released once its own completion is observed,
  which the next launch polls for; a still-queued copy therefore keeps its source and a
  long eager loop that uploads a fresh pinned buffer per operation holds the ones whose
  copies have not finished yet.
- Benchmark medians and the raw rows for the before/after comparison are in the PR
  description and the linked benchmark-repository run directories; the case is
  `gpu/elementwise`, `elementwise_chain_f64_{1k,1m}`, arms `tenferro-cuda-eager` and
  `tenferro-cuda-trace`, one arm per process, measure CPU pinned.
