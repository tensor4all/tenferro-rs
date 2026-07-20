# Current-main eager dispatch baseline (2026-07-20)

This worklog records the pre-refactor eager API baseline requested by issue
#1433. It is a measurement of the current implementation, not evidence about
the proposed runtime architecture and not a post-hoc non-inferiority threshold.

## Source and environment

- Upstream baseline: `origin/main` at
  `85855e272b1495611deb601a9ee06f3546772c3c`.
- Measurement branch parent:
  `2e82107a5080b8290e8cbae05d643f4a78a1eb54`. Relative to the upstream
  baseline, that parent changes documentation only; there were no Rust source
  changes before adding this benchmark.
- Host: AMD EPYC 7713P 64-Core Processor, 64 physical cores, one socket, one
  NUMA node, 1,019,435 MiB reported node memory.
- OS: Linux 6.8.0-101-generic, x86_64.
- Compiler: `rustc 1.96.0 (ac68faa20 2026-05-25)`, LLVM 22.1.2.
- Cargo: `cargo 1.96.0 (30a34c682 2026-05-25)`.
- CPU frequency governor: `schedutil`.
- Host load immediately after the run: 3.97, 4.95, 4.63 (1, 5, and 15
  minutes). This is a shared host rather than an isolated performance runner.
- Measurement time: 2026-07-20 14:55-14:59 JST.

The process was pinned to CPU 0. Rayon and the CPU backend were both limited to
one thread:

```console
RAYON_NUM_THREADS=1 taskset -c 0 \
  cargo bench -p tenferro-ad --bench eager_dispatch_baseline
```

Criterion used a 2-second warm-up, a 5-second measurement interval, and 100
samples per case. Inputs were created outside the timed loops. Each iteration
called the public `EagerTensor` API and consumed its result. `dot_general`
clones the public `DotGeneralConfig` inside the timed loop, so the baseline
includes that existing request-construction cost.

## Results

Times below are Criterion median estimates in microseconds with 95% confidence
intervals. `lazy` consumes `shape()` and `tensor_read()` from the returned
`EagerTensor`; `materialized` additionally calls `materialized()` and reads the
first `f64` value.

| Consumption | Operation | Size | Median (us) | 95% CI (us) |
|---|---|---:|---:|---:|
| lazy | neg | 1 | 9.590 | 9.407-9.741 |
| lazy | neg | 8 | 9.501 | 9.362-9.668 |
| lazy | neg | 64 | 10.535 | 10.387-10.718 |
| lazy | add | 1 | 10.330 | 10.211-10.617 |
| lazy | add | 8 | 10.365 | 10.137-10.482 |
| lazy | add | 64 | 9.744 | 9.721-9.771 |
| lazy | reduce_sum | 1 | 9.874 | 9.753-10.097 |
| lazy | reduce_sum | 8 | 9.374 | 9.321-9.519 |
| lazy | reduce_sum | 64 | 9.535 | 9.472-9.625 |
| lazy | dot_general | 1x1 | 11.407 | 11.366-11.469 |
| lazy | dot_general | 2x2 | 11.413 | 11.377-11.454 |
| materialized | neg | 1 | 8.866 | 8.848-8.913 |
| materialized | neg | 8 | 9.046 | 9.018-9.082 |
| materialized | neg | 64 | 9.155 | 9.119-9.191 |
| materialized | add | 1 | 9.525 | 9.500-9.564 |
| materialized | add | 8 | 9.738 | 9.661-9.806 |
| materialized | add | 64 | 9.852 | 9.784-9.986 |
| materialized | reduce_sum | 1 | 9.161 | 9.130-9.192 |
| materialized | reduce_sum | 8 | 9.496 | 9.462-9.594 |
| materialized | reduce_sum | 64 | 10.885 | 9.699-10.954 |
| materialized | dot_general | 1x1 | 11.613 | 11.381-12.127 |
| materialized | dot_general | 2x2 | 11.413 | 11.364-11.457 |

The small elementwise and reduction calls are approximately 9-11 us on this
host; the tiny `dot_general` calls are approximately 11.4-11.6 us. Because
useful work is intentionally tiny, these cases expose fixed eager-path costs,
but they measure the complete public call rather than isolating one dispatch
instruction.

## Use as a future gate

The post-refactor comparison must reuse this benchmark source, compiler/profile,
CPU pinning, backend thread count, and consumption semantics. It should run the
baseline and candidate close together on an otherwise quiet machine; this
single shared-host run is not sufficient for an absolute CI gate.

Before collecting candidate results, the implementation child issue must state
its non-inferiority statistic, allowed regression threshold, repetition policy,
and noisy-run handling. Those choices must not be derived from candidate
measurements. Phases 1-2 of the architecture work remain independent of this
gate; changing the eager execution path does not.
