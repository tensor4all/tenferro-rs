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
inherited from an environment variable. The pool is the only CPU thread control: the faer backend
has no thread pool of its own and reads no vendor thread variable, so nothing outside the pool
changes how many threads run.

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
on the calling thread (`:711-713`). Three consequences:

- a larger budget can make a region parallel that a smaller budget ran serially, because a
  wider width becomes available to the search — the result is not `min(width, budget)`;
- two budgets that both admit the width the model picks produce the same partition, and so the
  same bits;
- equal bits across two budgets do not by themselves show that this happened. The budget
  changes which widths are available, not the rule: a 1 ms region pays at width 2 (1423 ns of
  dispatch) and also at width 8 (2738 ns), so the search picks 2 under one budget and 8 under
  the other, and the two runs can still agree in their last bits. Agreement is an
  observation, not a consequence of the budget rule.

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

Where the product runs is decided before the width, and for `Complex64` that decision is a
driver switch rather than the cost-model threshold: above `1 << 17` elements a contraction
goes to tcontract's packed driver instead of faer, except at a thread budget of one on a
`native` build, where that bound is lifted for every host whatever `HostClass::detect`
returns (`crates/tenferro-cpu/src/contraction.rs:471-502`).
`48³ = 110 592` is below the bound and `64³ = 262 144` above it, which is exactly where the
recorded `Complex64` cells diverge, and why 2 and 8 agree with each other but not with 1 on
that host. The dtype rates (`0.0641`/`0.2565` on the AVX-512 row, `0.037`/`0.165` on the
Apple row, `t4a-faer 0.33.0/src/utils/cost.rs:382,407`) only decide the width inside
whichever driver runs; they do not set this boundary.

## What this asks of a contributor

- Do not add a test that asserts identical bits across thread counts.
- Do add or keep tests that assert identical bits at a fixed thread count, with the count
  pinned explicitly rather than through an environment variable.
- When writing a performance claim, pin the thread count for both sides of the comparison
  ([Performance Tips](https://github.com/tensor4all/tenferro-rs/blob/main/PERFORMANCE_TIPS.md#performance-sensitive-tests-and-benchmarks)).
