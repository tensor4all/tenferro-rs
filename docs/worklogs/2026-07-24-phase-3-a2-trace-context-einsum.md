# Phase 3 A2 TraceContext and einsum checkpoint

Date: 2026-07-24

## Session summary

This checkpoint moved backend-neutral trace ownership into `TraceContext` and
replaced compiler-owned einsum tracing with exactly four
`TraceContextEinsumExt` entry points. The implementation landed in
`1c495298` and `e3ea1e58`; the remaining caller cleanup was included in the
Phase 3 closure at `e40bcd3c`.

This is an A2 checkpoint record, not a claim that Phase 3 was complete at the
time. Semantic AD and final compatibility-surface deletion still belonged to
A3.

## Decisions made

- `TraceContext` is the only mutable owner of one semantic trace.
- `TraceValue` remains opaque and context-owned; foreign values are rejected
  without mutating the destination trace.
- Ordered input descriptors and defaults are stored once in the semantic
  builder/frozen program rather than duplicated in a parallel trace registry.
- `finish` consumes the context and preserves ordered inputs and duplicate
  outputs in `TracedGraph`.
- `GraphCompiler::compile_traced_graph` accepts the immutable semantic artifact
  and creates only runtime-private execution staging.
- `TraceContextEinsumExt` exposes `einsum`, `einsum_subscripts`,
  `einsum_with`, and `einsum_subscripts_with`. Text parsing uses the
  context-owned extension cache; each successful call emits one semantic
  extension operation.
- `GraphCompilerEinsumExt` and its free-function compatibility surface were
  removed. `TracedTensor` retains only the separate `tensordot` contraction
  sugar needed by existing core tracing.

## Verification performed

- Runtime tests cover foreign ownership, opaque Debug output, ordered inputs,
  defaults, duplicate outputs, and ordered execution.
- Einsum integration tests cover all four methods, n-ary and parsed notation,
  malformed input, foreign values, explicit path/tree policy, symbolic
  metadata, cache behavior, compilation, and execution.
- Runtime/einsum tests, doctests, all-target clippy, formatting, documentation
  checks, and `git diff --check` passed as part of the Phase 3 closure.

## Performance policy

The eager einsum path was remeasured after the first comparison crossed the
5% noise trigger. The valid repeat measured approximately +3.46%, below the
remeasure threshold and far below the roughly 50% blocking threshold.

## Remaining work at this checkpoint

A3 still owned whole-program semantic AD, extension-family migration,
production transform-cache wiring, and deletion of old public
graph/execution/rule surfaces. Those items were subsequently closed by
`e40bcd3c` and its Phase 3 audit follow-up.
