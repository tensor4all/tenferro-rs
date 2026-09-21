# CPU worker affinity: domain confinement instead of one CPU per worker

## Decisions

- A Tenferro-managed CPU domain now confines every Rayon worker to the
  **whole declared CPU set** (`CpuContext::with_pinned_cpus`) instead of
  assigning one CPU per worker. `CpuExecutorAffinity::TenferroPinnedVerified`
  is renamed to `TenferroDomainVerified` and states the verified domain set,
  not a per-worker CPU. `CpuContextError::WorkerPinning { worker, cpu, .. }`
  becomes `CpuContextError::WorkerAffinity { worker, .. }`.
- Reason: BLAS/LAPACK providers create their own thread teams, and those
  threads inherit the creating thread's CPU mask. With one CPU per worker the
  provider team was confined to that single CPU, so provider parallelism
  silently collapsed. Diagnosis came from comparing provider builds: the same
  `dgemm`/LU workloads that the OpenBLAS build spread over the domain ran on
  3-20x less efficient configurations in the MKL build.
- Rejected: per-operation affinity widening (`sched_setaffinity` around each
  provider call), running provider calls on the API-calling thread, and
  per-worker CCX CPU subsets. The first two add restore/error paths and change
  where a session executes; the third was measured to protect nothing (see
  below) and was dropped as unnecessary complexity.
- User-facing guides are part of the change: `docs/guides/cpu-execution.md`
  documented the one-CPU worker pinning and carried a `KMP_AFFINITY=...,norespect`
  workaround for the inherited one-CPU mask it produced. The guide now describes
  domain confinement, and `docs/guides/choosing-a-backend.md` /
  `docs/guides/external-linalg-interop.md` no longer claim that tenferro cannot
  place provider workers at all — the precise remaining limits are the provider's
  fan-out and any affinity policy the provider installs itself.
- Rejected: keeping per-worker single-CPU affinity for "performance-critical"
  native regions. Pinning only those regions would restore the syscall/pool
  machinery while the measurements show no native locality benefit.

## Verification conclusions and constraints

- Provider paths recover: with 4 domain CPUs, MKL `dgemm` 1024x1024
  43-53 ms -> 13.0 ms, `solve` 1024x1024 rhs=1 134-139 ms -> 6.8 ms,
  `svd` 512 112-120 ms -> 29.7 ms, `qr` 1024 320-324 ms -> 48.8 ms,
  `eigh` 512 84 ms -> 17.8 ms, `grad_sum_solve` 512 78-91 ms -> 7.8 ms.
- Native/faer parallelism is unaffected: elementwise, reduction, and transpose
  rows are unchanged, and a paired A/B run at both 4 workers / 4 CPUs and
  4 workers / 16 CPUs over two CCXs kept every faer row within +-4% with mixed
  signs. Worker migration measured by 2 ms affinity sampling was 0-0.2 moves/s
  without a provider (all four workers stayed on one CPU for entire runs) and
  0.8-2.8 moves/s with MKL.
- The old assignment spread workers evenly across the whole allowed mask
  (`select_worker_cpus`), which is neither CCX- nor NUMA-local, so no locality
  guarantee was lost by removing it.
- Regression coverage: `pinned_workers_are_confined_to_the_whole_domain_cpu_set`
  asserts each worker's verified mask equals the domain set, and
  `provider_style_thread_from_a_worker_inherits_the_domain_cpu_set` asserts a
  thread created from a worker inherits the whole set (the failure mode this
  change fixes). Both are Linux-gated and use the process's allowed CPUs.
- Unverified: multi-NUMA-node domains and domains much wider than the worker
  count. If a real locality regression appears there, the lever is selecting a
  narrower domain (existing placement/NUMA policy), not per-worker CPU subsets.
- Provider thread counts remain provider-controlled
  (`MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`); this change
  does not add any thread-count setter. The worker stack reservation from the
  preceding fix stays authoritative for provider kernels that recurse.
- Evidence: paired pinning-mode and multi-CCX measurements under
  `.worktrees/affinity-check-atEL1H/` (`pinning-mode-results.md`,
  `pinmode-*.jsonl`, `ccx-*.jsonl`, `ccxmig-*.jsonl`,
  `final-affinity-mkl-t4.jsonl`).
