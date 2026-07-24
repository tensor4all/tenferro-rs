# Phase 8 XLA compiled-graph API checkpoint

This worklog records the Phase 8 local checkpoint after
[`2026-07-25-phase-7-webgpu-runtime-registration.md`](2026-07-25-phase-7-webgpu-runtime-registration.md).

## Session summary

Phase 8 was implemented locally on
`codex/execution-engine-phase9-restart`. Per maintainer direction, no PR is
created until Phase 8 is complete and the AMD CPU/CUDA benchmark gate has
passed.

The implemented slice is intentionally bounded:

- `tenferro_xla::lower_compiled_to_stablehlo(&CompiledGraph)` is the preferred
  public graph lowering boundary;
- `XlaExecutor` exposes `CompiledGraph` wrappers for lowering and ordered-input
  PJRT execution calls;
- existing `SemanticProgram` APIs remain compatible for operation crates,
  lower-level tests, and internal lowering callers;
- full runtime-owned `SubgraphCompiler` selection, one-node scheduled XLA
  subgraphs, and PJRT executable caching remain deferred.

## Context read

- Workspace and repository rules: `AGENTS.md`, `REPOSITORY_RULES.md`, workspace
  `CODING_RULES.md`, and shared tensor4all rules.
- Design authority:
  `docs/design/execution-engine-provider-architecture.md` and
  `docs/design/xla-backend.md`.
- XLA public API and executor:
  `crates/tenferro-xla/src/lib.rs` and
  `crates/tenferro-xla/src/executor.rs`.
- PJRT execution boundary:
  `crates/tenferro-xla/src/pjrt/execute.rs`.
- Runtime compiled artifact:
  `crates/tenferro-runtime/src/graph/program.rs`.

## Implementation decisions

1. The new public helper accepts `tenferro_runtime::CompiledGraph` and delegates
   to existing StableHLO lowering through `program.program()`. It does not
   construct or consume `ExecProgram`.
2. `XlaExecutor::lower_compiled_to_stablehlo` delegates to the new public helper
   so executor users follow the same boundary.
3. `XlaExecutor::run_compiled_many_with_inputs` and
   `XlaExecutor::run_compiled_with_inputs` preserve the current ordered-input
   PJRT contract and delegate to the existing semantic-program execution path.
4. Default-input XLA execution is not added in this slice. The PJRT boundary
   still requires explicit ordered host tensors.
5. The `SemanticProgram` lowering and execution APIs stay public for lower-level
   use, but examples now prefer `CompiledGraph` for graph users.

## TDD evidence

The following RED check was observed before the implementation:

```text
cargo test -p tenferro-xla xla_accepts_compiled_graph_without_program_peeking
  -> compile failure: no lower_compiled_to_stablehlo,
     XlaExecutor::lower_compiled_to_stablehlo, or
     XlaExecutor::run_compiled_with_inputs
```

The corresponding focused GREEN check passed:

```text
cargo test -p tenferro-xla xla_accepts_compiled_graph_without_program_peeking
cargo test -p tenferro-xla
cargo test -p tenferro-xla --features pjrt
cargo fmt --all --check
python3 scripts/check-doc-snippets.py
python3 scripts/check-public-error-docs.py
python3 scripts/test-doc-consistency.py
python3 scripts/check-guide-dependency-snippets.py
git diff --check
```

## Open-decision ledger

These items are intentionally not implemented in this Phase 8 slice:

- runtime `SubgraphCompiler` capability and deterministic subgraph selection;
- prepared one-node XLA scheduled subgraph operations;
- PJRT executable cache identity and eviction statistics;
- environment-gated CPU-vs-XLA value comparisons beyond the existing XLA test
  harness.

The next gate before PR creation is the Phase 8 closeout verification plus the
AMD CPU/CUDA benchmark gate requested by maintainers.
