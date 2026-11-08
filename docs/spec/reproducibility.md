# Reproducibility

Normative. This page owns tenferro's bitwise-reproducibility promise. The threading
mechanics it rests on are owned by
[Performance Tips: CPU Threading Contract](https://github.com/tensor4all/tenferro-rs/blob/main/PERFORMANCE_TIPS.md#cpu-threading-contract),
and how a comparison may be set up is owned by
[Performance Tips: Performance-Sensitive Tests And Benchmarks](https://github.com/tensor4all/tenferro-rs/blob/main/PERFORMANCE_TIPS.md#performance-sensitive-tests-and-benchmarks).
Other documents link here rather than restating the promise.

## The promise

Bitwise-identical results on the same machine with the same build at a **fixed thread
count** — across fresh threads, processes and repeated runs. That is the whole promise.

The count is pinned explicitly, by building the `CpuEngine` pool with the width you want
([Parallelism and Caching](../guides/parallelism-and-caching.md#cpu-thread-count)); it is never
inherited from an environment variable.

## What is not promised

Identical bits when the thread count changes. It was previously treated as a goal; it was
withdrawn as a requirement on 2026-11-08 ([#2522](https://github.com/tensor4all/tenferro-rs/issues/2522)),
because making the numeric partition independent of the thread count constrains how the work
may be split — a permanent cost — to buy a property no user needs. No public API claims it.

## Comparing results

Fix the thread count before comparing anything: caches that span runs, benchmark tables,
regression digests. A digest captured at one width and checked at another compares two
different computations.

## Why a thread count can move the last bits

The width of the parallel work is chosen by a cost model from the problem, not by the
requested degree. The faer-backed CPU kernels receive the engine placement token (through
`tcontract_exec::with_faer_par`), and a parallel region opens only where the estimate pays:
the serial time of the region, priced in flops (and therefore dtype-dependent), must be at
least `PAYS` times the dispatch cost of the sections it opens — the same threshold
`threadteam::PAYS` applies to tenferro kernels.

The token budget bounds that search; it does not scale the width. The pinned `t4a-faer`
0.33.0 considers every width in `2..=budget` and takes the **widest one that pays**
(`t4a-faer 0.33.0/src/utils/cost.rs:613-619`); when none pays it returns 1 and the region runs
on the calling thread (`:711-713`). Two consequences:

- a larger budget can make a region parallel that a smaller budget ran serially, because a
  wider width becomes available to the search — the result is not `min(width, budget)`;
- two budgets that both admit the width the model picks produce the same partition, and so the
  same bits — which is why 2 and 8 threads agree in the observations below.

"Differs across thread counts" therefore means "differs between the serial path and the
parallel path" for the cases recorded here, not "the summation order follows the count". It is
not a general rule: the tridiagonalization inside `t4a-faer` splits columns as `idx / nthreads`
(`t4a-faer 0.33.0/src/linalg/evd/tridiag.rs:386-409`) and adds the partial vectors, so there two
parallel widths group the same sum differently.

## Observations, not contract

Recorded in [#2522](https://github.com/tensor4all/tenferro-rs/issues/2522), on an AMD EPYC
7713P (Zen 3, AVX2+FMA), as evidence of the shape above rather than as a promise:

- f64 `mat_mul` gave identical bits at 1 vs 2 and at 2 vs 8 threads;
- Complex64 `mat_mul` / `contract_pair` agreed at `16³`, `32³` and `48³`, and differed at
  `64³` and larger **between 1 and 2 threads only** — 2 and 8 agreed at every size.

The complex threshold sits at a different size than the real one because the estimate is in
nanoseconds per multiply-add and the rate is dtype-specific: `gemm_mac` is `0.0641` real against
`0.2565` complex on the AVX-512 row, and `0.037` against `0.165` on the Apple row
(`t4a-faer 0.33.0/src/utils/cost.rs:382,407`). That ratio is near four, not the two that
"twice the work per element" would suggest, so the crossover moves with the measured rate.

## What this asks of a contributor

- Do not add a test that asserts identical bits across thread counts.
- Do add or keep tests that assert identical bits at a fixed thread count, with the count
  pinned explicitly rather than through an environment variable.
- When writing a performance claim, pin the thread count for both sides of the comparison
  ([Performance Tips](https://github.com/tensor4all/tenferro-rs/blob/main/PERFORMANCE_TIPS.md#performance-sensitive-tests-and-benchmarks)).
