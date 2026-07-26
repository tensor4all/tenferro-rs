# Phase 3 A2 CompiledGraph checkpoint

Date: 2026-07-24

## Session summary

This checkpoint replaced the transitional graph-program artifact with
`CompiledGraph`, whose public contract is the frozen semantic program,
process-local bindings, and ordered input/output counts. The boundary landed
across `831760c6`, `66609b5f`, `6411c0ba`, and `ef1ffa40`; final caller
migration was included in `e40bcd3c`.

This log records the A2 boundary. Semantic AD and the last serial migration
were still A3 work at that point.

## Decisions made

- Public consumers inspect `program()`, `bindings()`, `input_count()`, and
  `output_count()` only.
- `ExecProgram`, slot tables, lowering views, and compiled staging remain
  private to `tenferro-runtime`.
- `CompiledGraph::Debug` reports bounded counts and the semantic fingerprint;
  it does not traverse tensor bindings or extension payloads.
- `GraphExecutor` accepts ordered tensor slices and offers `Tensor`,
  `TensorRead`, and `TensorValue` variants for single- and multi-output
  execution.
- An empty explicit input slice means “use all frozen defaults.” Arity,
  dtype, shape, placement, and missing-default validation occur before backend
  dispatch.
- The only forward adapter is the crate-private
  semantic-program-to-execution-staging lowering. No reverse staging adapter
  was introduced.

## Source-contract evidence

- Public `GraphProgram` and `GraphProgramInput` definitions are absent.
- Public `ExecProgram` construction and re-export are absent; its module is
  private to `tenferro-runtime`.
- Keyed `GraphExecutor::*_with_bindings` compatibility methods are absent.
- Raw `eval_exec_ir*` entry points are crate-private.

Repository source-contract tests enforce these boundaries.

## Verification performed

Runtime debug/release tests, doctests, workspace all-target clippy, formatting,
documentation checks, CUDA/WebGPU compile gates, external sparse/tropical
tests, and `git diff --check` passed as part of the Phase 3 closure.

The trace build-and-compile benchmark improved reproducibly by approximately
73%. The eager einsum remeasurement was approximately +3.46%, below the 5%
trigger and the roughly 50% blocking threshold.

## Remaining work at this checkpoint

A3 still owned semantic AD for core and all extension families, the
TraceContext einsum caller closure, and removal of old AD/execution surfaces.
Those items were subsequently closed by `e40bcd3c` and its audit follow-up.
