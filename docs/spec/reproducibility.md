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
the serial time of the region, priced as units of work times a per-kernel rate (one for real,
one for complex, halved in single precision; `colmat 0.34.0/src/utils/cost.rs:254-264`), must
be at least `PAYS` times the dispatch cost of the sections it opens — `colmat` re-exports
`threadteam::PAYS`, the threshold tenferro kernels use (`:15`).

The token budget bounds that search; it does not scale the width. The budget that reaches
faer can already be narrower than the engine's: for a contraction, tcontract limits the token to
the item width it chose (`tcontract 0.13.0/src/strategy/faer.rs:310`). Within it `colmat` 0.34.0
searches `2..=budget`, up to a cap where the region has one (the partial-pivot LU regions,
`cost.rs:658-675`, or a task count), and takes the **widest one that pays**
(`cost.rs:692-702`); when none pays it returns 1 and the region runs on the calling thread
(`colmat 0.34.0/src/utils/mod.rs:82-84`). Three consequences:

- a larger budget can make a region parallel that a smaller budget ran serially, because a
  wider width becomes available to the search — the result is not `min(width, budget)`;
- two budgets for which the search picks the same width give that region the same partition.
  This does not extend to a whole call: a recursive join only asks whether width 2 pays and
  then divides its whole budget between the halves (`cost.rs:750-768`);
- equal bits across two budgets do not by themselves show that this happened. The budget
  changes which widths are available, not the rule: on the `x86_64-avx2` dispatch row
  (`threadteam 0.6.0/src/cost.rs:388-399`) a 1 ms region pays at width 2 (1423 ns of
  dispatch) and also at width 8 (2738 ns), so the search picks 2 under one budget and 8 under
  the other, and the two runs can still agree in their last bits. Agreement is an
  observation, not a consequence of the budget rule.

"Differs across thread counts" therefore means "differs between the serial path and the
parallel path" for the cases recorded here, not "the summation order follows the count". It is
not a general rule: the tridiagonalization inside `colmat` runs each step at the searched
width (`colmat 0.34.0/src/linalg/evd/tridiag.rs:332`), puts column boundary `idx` at
`(1 - sqrt(1 - idx / width)) * n` (`:392-398`) and adds one partial vector per participant
(`:457-460`), so there two parallel widths group the same sum differently.

## Observations, not contract

Recorded in [#2522](https://github.com/tensor4all/tenferro-rs/issues/2522) on the t4a-faer
release then pinned (not repeated on colmat 0.34), on an AMD EPYC 7713P (Zen 3, AVX2+FMA),
as evidence of the shape above rather than as a promise:

- f64 `mat_mul` gave identical bits at 1 vs 2 and at 2 vs 8 threads;
- Complex64 `mat_mul` / `contract_pair` agreed at `16³`, `32³` and `48³`, and differed at
  `64³` and larger **between 1 and 2 threads only** — 2 and 8 agreed at every size.

Where the product runs is decided before the width, and for `Complex64` that decision is a
driver switch rather than the cost-model threshold: a fusable contraction whose `m * n * k`
volume is above `1 << 17` goes to tcontract's packed driver instead of faer
(`tcontract 0.13.0/src/plan/mod.rs:481-485`), except at a thread budget of one, where that
bound is lifted for every host whatever `HostClass::detect` returns
(`crates/tenferro-cpu/src/contraction.rs:415-433`).
`48³ = 110 592` is below the bound and `64³ = 262 144` above it, which is exactly where the
recorded `Complex64` cells diverge, and why 2 and 8 agree with each other but not with 1 on
that host. faer's GEMM rates (`0.0641`/`0.2565` on the AVX-512 row, `0.037`/`0.165` on the
Apple row and, as the smaller of those two, on every other host class;
`colmat 0.34.0/src/utils/cost.rs:392,418,439-450`) only decide the width inside faer. The
packed driver's width is priced by tcontract (`tcontract 0.13.0/src/plan/mod.rs:1245`).
Neither sets this boundary.

## What this asks of a contributor

- Do not add a test that asserts identical bits across thread counts.
- Do add or keep tests that assert identical bits at a fixed thread count, with the count
  pinned explicitly rather than through an environment variable.
- When writing a performance claim, pin the thread count for both sides of the comparison
  ([Performance Tips](https://github.com/tensor4all/tenferro-rs/blob/main/PERFORMANCE_TIPS.md#performance-sensitive-tests-and-benchmarks)).
