# CPU Session Entry and Rayon Dispatch Cost

Keep a session around a related sequence of operations. Reusing a backend
preserves its pool and caches; reusing one entered session additionally
amortizes the cost of reaching its workers. These are different optimizations.
A session is an execution scope, not merely a tensor wrapper.

## What costs time?

For the managed CPU path inspected at tenferro revision
`8e3f8b945ba6605ef23d80143592d7437484d3bb`, session entry obtains execution
permission, enters the selected executor, borrows buffers/caches, and calls
the user closure. A multi-thread CpuContext uses Rayon `ThreadPool::install`;
the one-thread CpuContext executes inline without constructing a one-worker
Rayon pool. Pool creation is not repeated on every session entry.

Rayon 1.12.0 / rayon-core 1.13.0 distinguishes three callers:

- Outside Rayon: put a stack job in the pool's injection queue, notify workers
  as needed, and wait on a mutex/condition-variable latch for completion.
- Already in the same pool: execute the closure directly.
- In another pool: inject into the target pool and let the current worker
  process work while waiting. This is not the same-pool fast path.

Thus an empty external `install` measures a synchronous cross-thread round
trip, not just construction of a session or a configuration change. Worker
wakeups, queue synchronization and OS scheduling can dominate tiny kernels.
See [Rayon registry implementation](https://github.com/rayon-rs/rayon/blob/rayon-core-1.13.0/rayon-core/src/registry.rs#L496)
and [tenferro CpuContext](https://github.com/tensor4all/tenferro-rs/blob/8e3f8b945ba6605ef23d80143592d7437484d3bb/crates/tenferro-cpu/src/context.rs#L524).

## Measurements: do not treat these as universal constants

Measured 2026-09-27 on Apple M5 Max, macOS 26.5.1, Rust 1.96.0, release build,
Rayon 1.12.0 / rayon-core 1.13.0. Available parallelism was 18. Each result
is the median of 15 batch means, with 10,000 calls per batch after 1,000
warmup calls. Pool construction and an initial worker broadcast are excluded.
The closure returns a black-boxed integer; this is a dispatch microbenchmark,
not GEMM or session throughput. Concurrent machine load was not controlled.

| Rayon workers | Install from outside, us/call | Install from same pool, ns/call |
|---|---:|---:|
| 1 | 3.037 | 1.54 |
| 2 | 4.121 | 1.36 |
| 8 | 14.398 | 1.31 |
| 18 | 26.380 | 1.52 |

A same-day reproduction during documentation verification gave about 21.9 us
for the 18-worker install sweep and 10.1 us in the subsequent scope comparison
(whose one-job scope measured 24.3 us). This variation reinforces that these
are observations, not a fixed 26 us constant or a performance threshold.

The same-pool numbers include compiler optimization of a minimal closure;
they are not the cost of a real operation. The observed worker-count trend
is not proof of a particular bottleneck or a portable scaling law.

A second 18-worker run compared entry strategies:

| Operation | Median, us/call |
|---|---:|
| External install | 26.128 |
| In-place scope, no spawned work | 0.056 |
| In-place scope, one spawned job and completion wait | 26.379 |

An in-place scope can avoid transferring its body to a worker, but does not
eliminate the cost of jobs it actually dispatches. Nor does it automatically
redirect ordinary Rayon calls in the caller's body to the specified pool.
It is not a drop-in replacement for tenferro session entry.

Separately, [issue #1904](https://github.com/tensor4all/tenferro-rs/issues/1904)
reported an empty tenferro session at **25.592 us** with the default
multi-thread backend and **0.118 us** with `with_threads(1)`, on an M5 Max at
revision `aebc3148`. That measurement includes session work and is historical,
not a fresh session benchmark at the revision inspected above. Its agreement
with the standalone Rayon magnitude supports the dispatch explanation; it
is not a measured subtraction of tenferro's other costs. The 3.037 us
one-worker Rayon result is not comparable to tenferro's inline one-thread
path.

The older [CPU session-open breakdown](../design/cpu-session-open-cost.md)
uses a different Linux machine, revision context, instrumentation and workload.
Do not combine its component timings with these numbers.

## How to avoid or amortize the overhead

1. Construct the backend once, then keep several related operations inside
   one explicit session callback. Use the borrowed session for each operation;
   do not open a backend/session again inside the callback or a spawned task.
   See the public examples in [session-oriented concrete APIs](../design/session-oriented-concrete-apis.md)
   and the [typed non-AD tutorial](../tutorials/typed-tensor-non-ad.md).
2. For many tiny matrices, prefer a batched operation or a sequence in one
   session. One 26 us entry spread over 100 operations contributes about
   0.26 us per operation; over 1,000, about 0.026 us. These are amortization
   calculations, not additional benchmarks. Do not hold an execution scope
   across unrelated waits: it retains execution resources/permission.
3. For an unavoidable tiny single call, measure the one-thread CPU backend.
   It avoids the worker handoff; it does not make tensor validation, allocation,
   or the numerical kernel free. Keep backend reuse even in this case.
4. For a large operation, compare total time: a 26 us entry adds about 2.6%
   to 1 ms of useful work, or 0.26% to 10 ms. Actual crossover points depend
   on the workload and provider.
5. A downstream non-AD kernel may use tensor/view storage directly. Follow
   [external linalg interop](external-linalg-interop.md) for layout, borrowing
   and provider obligations. Direct faer calls can remove tenferro dispatch,
   but do not remove Rayon handoff if they still enter a pool. Direct external
   parallel calls must respect the documented nesting and resource contracts.

The [faer re-export proposal #1939](https://github.com/tensor4all/tenferro-rs/issues/1939)
and [CUDA vendor-call contract #1940](https://github.com/tensor4all/tenferro-rs/issues/1940)
track downstream access work; an issue link is not a claim that an API has
shipped. CPU timings here say nothing about CUDA launch or synchronization cost.

**Do not remove sessions, switch to an ambient global pool, bypass resource
admission, or add nested backend entry to fix this measurement.** Sessions
amortize the round trip while retaining configured execution resources. A
hypothetical faer API accepting a pool would save the initial handoff only if
its scheduling implementation changed, not if it merely wrapped `install`.
Parallel work has coordination costs, but there is no universal 20–26 us tax
on every parallel operation.

## Reproduce the dispatch measurements

The two source snapshots and locked dependency graph used for these runs are
available as [install source](../performance/rayon-install-probe/install.rs.txt),
[scope source](../performance/rayon-install-probe/scope.rs.txt),
[manifest](../performance/rayon-install-probe/Cargo.toml.txt),
[lockfile](../performance/rayon-install-probe/Cargo.lock.txt), and raw
[install](../performance/rayon-install-probe/install-results.txt) /
[scope](../performance/rayon-install-probe/scope-results.txt) results. They are
standalone probes, not tenferro API examples or CI latency thresholds. From
the repository root:

```bash
probe_dir=$(mktemp -d)
mkdir -p "$probe_dir/src"
cp docs/performance/rayon-install-probe/Cargo.toml.txt "$probe_dir/Cargo.toml"
cp docs/performance/rayon-install-probe/Cargo.lock.txt "$probe_dir/Cargo.lock"
cp docs/performance/rayon-install-probe/install.rs.txt "$probe_dir/src/main.rs"
cargo run --release --locked --manifest-path "$probe_dir/Cargo.toml"
cp docs/performance/rayon-install-probe/scope.rs.txt "$probe_dir/src/main.rs"
cargo run --release --locked --manifest-path "$probe_dir/Cargo.toml"
```

Record compiler, OS/CPU, worker count, revisions, timing scope and load when
comparing results. Measure empty session entry, kernel work inside an already
entered session, and end-to-end calls separately when locating a regression.
Do not infer a session's cost from a benchmark that also constructs a backend.
