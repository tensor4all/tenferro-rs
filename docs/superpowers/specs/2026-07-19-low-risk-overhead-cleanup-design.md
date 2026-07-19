# Low-risk overhead cleanup design

## Scope

Address three internal overhead findings from #1426 without changing public APIs,
numeric results, validation, error behavior, layout semantics, or profiling output:

1. Avoid per-element index-vector allocation when `dot_general` accumulates into
   a compact column-major output.
2. Avoid `SmallVec -> Vec -> SmallVec` metadata round trips in tensor view
   construction where the destination rank representation can be built directly.
3. Avoid reading the clock for eager AD profiling when profiling is disabled.

## Considered approaches

For `dot_general`, replacing the whole accumulation loop with unchecked pointer
arithmetic would also remove overhead, but would duplicate layout logic and widen
the unsafe surface. Reworking the generic tensor iterator would have a larger
blast radius. The selected approach adds a compact-output slice fast path and
keeps the existing indexed implementation as the fallback for non-compact views.

For view metadata, changing the public shape types could eliminate more copies,
but would be an API change. The selected approach only removes conversions that
are redundant for the existing `SmallVec`-backed representations.

For profiling, changing the profiler API is unnecessary. The selected approach
moves clock acquisition behind the existing enabled predicate.

## Behavioral contracts

- The compact accumulation fast path preserves `out = alpha * dot + beta * out`.
- When `beta == 0`, accumulation does not read the previous output value.
- Non-compact and strided outputs continue through the existing checked path.
- Shape, stride, offset, dtype, validation, and error behavior remain unchanged.
- Profiling-enabled event contents and timings retain the same meaning.
- No public symbols or dependencies are added.

## Verification

- Add or extend tests for compact overwrite and additive accumulation, including
  a `beta == 0` case whose destination initially contains non-finite values.
- Retain explicit coverage of a non-compact/strided accumulation destination.
- Add structural unit coverage where practical for the profiling gate and direct
  metadata conversion; otherwise rely on existing behavioral tests plus source
  inspection and benchmarks.
- Add a focused accumulation benchmark that exercises the compact hot path.
- Run formatting, targeted crate tests, Clippy for touched crates, and the
  repository's required pre-PR verification.

## Non-goals

This change does not address eager einsum planning, runtime segment caches, GPU
planning, host-transfer ownership, dynamic slicing, FFT planning, or public API
design. Those findings retain their separate performance and ownership risks.
