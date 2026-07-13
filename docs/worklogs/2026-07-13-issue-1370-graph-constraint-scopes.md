# Issue 1370 graph-owned shape-constraint scopes

## Scope

This stage preserves extension shape constraints across traced graph
composition and lowers reachable scopes in `GraphCompiler`. It does not add
constraints to real extension families or transfer scopes through
`TracedTensorParts`; those remain separate follow-up stages.

## Context read

- Shared tensor4all repository, Rust, performance, documentation, testing, and
  numerical rules.
- `AGENTS.md`, `REPOSITORY_RULES.md`, and the approved extension shape equality
  design.
- CodeGraph call paths for graph metadata registration, extension application,
  traced unary/binary construction, checkpointing, and graph compilation.

## Design

- One graph analysis walk registers metadata and records extension-local
  constraints with every output origin and the ordered graph input keys.
- Traced tensors carry an immutable `Arc`-backed constraint-scope chain.
  Materialization uses pointer-identity deduplication; semantic deduplication
  remains in the normalized equality solver after lowering.
- The compiler gathers output chains before graph materialization, maps live
  value keys to SSA slots once, and prunes a scope only when none of its origin
  keys is live.
- Scoped `InputDim` expressions are substituted through the complete
  pre-optimizer symbolic slot-shape table. Executable instruction metadata
  continues to use a separate concrete shape/extent table. Extension inference
  runs once against op-local placeholders and its results are explicitly
  substituted into both tables, so global symbolic indices cannot leak into
  instruction-local extent resolution. Missing keys, slots, and axes return
  typed `ShapeConstraintEvaluation` errors.
- Compiler-inferred and graph-scoped constraints share the same provenance,
  discharge, normalization, and cache-identity pipeline.
- `TracedTensorParts` deliberately constructs an empty constraint chain in this
  stage. Explicit AD transform transfer is deferred.

## TDD evidence

The first focused run failed because the scope types and traced field did not
exist. After the scope substrate was added, compiler-focused tests failed with
zero guards for live scoped relations because graph input slot shapes were
initialized as constants. Separating concrete runtime descriptors from
program-input shape expressions made the live constraints compile to one
guard while preserving typed missing-reference failures and all-dead pruning.
An initial attempt to reuse symbolic slot shapes for executable extents exposed
an XLA tutorial regression: an einsum output's global `InputDim` index was
misread as instruction-local and propagated an unknown extent into `Abs`.
The focused tutorial test stayed red until the concrete and symbolic tables
were separated at the inference boundary described above.

## Residual risk

This stage preserves constraints through ordinary traced composition and
checkpoint roots. AD transform output parts still initialize an empty chain by
design until the explicit transfer boundary is implemented.

## Verification

- `cargo test --workspace --all-targets --release`
- workspace and standalone tropical Clippy parity with `-D warnings`
- `cargo doc --workspace --no-deps` and `scripts/check-docs-site.py`
- `cargo llvm-cov --workspace --release` and the per-file coverage checker
- focused constraint-scope, symbolic cache-identity, and XLA tutorial
  regressions
