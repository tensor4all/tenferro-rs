# Phase 3 A1 semantic-staging migration plan

**Goal:** Establish the only forward `SemanticProgram` to private execution
staging path, make traced graph compilation publish the semantic artifact, and
migrate peer callers away from public execution staging without violating the
accepted serial ownership of einsum semantic AD.

**Architecture:** The existing trace materialization remains a temporary input
to semantic compilation. It no longer lowers directly to `ExecProgram`.
`GraphCompiler` first constructs and freezes a backend-neutral
`SemanticProgram` plus `ProgramBindings`; one crate-private adapter lowers that
artifact into the temporary execution staging type. `GraphProgram` temporarily
retains both sides only until P3-A2/A3 migrate its executor callers. XLA reads
public semantic views directly and never calls the private adapter.

## Task 1: Freeze the sole forward staging adapter

- [x] Add failing runtime tests for core, multi-output extension, ordered
  inputs/outputs, bindings exclusion, exact symbolic metadata, guards, and
  typed rejection of non-exact staging metadata.
- [x] Add one crate-private `SemanticProgram -> ExecProgram` adapter under the
  runtime compiler. Do not add a reverse adapter or expose raw program slots.
- [x] Reuse existing execution optimization and shape-constraint machinery;
  preserve semantic operation/output order and exact extension payloads.
- [x] Run focused program/compiler/graph tests and commit the adapter.

## Task 2: Route traced graph compilation through SemanticProgram

- [ ] Add failing tests proving `GraphCompiler` publishes a semantically
  equivalent artifact, freezes default tensors into `ProgramBindings`, and
  obtains execution staging only through the adapter.
- [ ] Convert the temporary computegraph materialization directly into
  `SemanticProgramBuilder` operations, with explicit extension
  effects/aliases and failure-atomic finish.
- [ ] Add temporary read-only semantic accessors to `GraphProgram`; document
  that the legacy container is deleted in P3-A3.
- [ ] Migrate in-repository extension payloads reached by graph compilation to
  explicit pure/fresh (or their actual effect/alias) declarations.
- [ ] Run runtime, extension, FFT, linalg, and einsum graph compilation tests;
  commit the semantic route.

## Task 3: Migrate XLA to public semantic views

- [ ] Add failing XLA tests/source contracts forbidding `GraphProgram`,
  `GraphProgramLoweringView`, `GraphInstructionView`, and public
  `ExecProgram` dependencies.
- [ ] Change StableHLO lowering and PJRT execution APIs to accept the frozen
  semantic artifact and ordered runtime inputs.
- [ ] Lower core/extension operations from `SemanticOperationView`,
  `SemanticOpRef`, and allocation-free metadata access. Reject bounded/unknown
  extents with typed `NonStaticShape`.
- [ ] Remove XLA use of runtime execution-staging compilation for extension
  subgraphs by constructing/lowering a temporary semantic subprogram.
- [ ] Run XLA unit/integration/doctests with default and PJRT feature gates;
  commit the caller migration.

## Task 4: Remove sibling-crate public ExecProgram construction

- [ ] Replace tenferro-ad integration fixtures that construct `ExecProgram`
  with traced/semantic public APIs, moving genuinely runtime-private staging
  coverage into tenferro-runtime tests.
- [ ] Add source contracts proving no sibling crate constructs or imports
  public `ExecProgram`.
- [ ] Run focused AD/runtime integration and doctests; commit the fixture
  migration.

## Task 5: Serial einsum staging migration after semantic AD freezes

- [ ] Wait for P3 semantic AD traits/registry and `CompiledGraph` API to be
  frozen.
- [ ] In one serial commit owning `crates/tenferro-einsum/src/extension.rs`,
  migrate both `EinsumAdRule` and its cached runtime-program boundary.
- [ ] Migrate compile benches/cache tests, run einsum AD/eager/traced/benchmark
  correctness gates, and verify cache retention accounting.

## A1 checkpoint gate

- [ ] Search proves exactly one private forward semantic-to-staging adapter,
  no reverse adapter, no XLA `GraphProgram`/execution-view dependency, and no
  sibling public `ExecProgram` construction.
- [ ] Run affected workspace debug/release tests, doctests, clippy, formatting,
  documentation/source-contract scripts, and `git diff --check`.
- [ ] If a measured eager path changes by at least 5%, remeasure; block only a
  reproducible slowdown of roughly 50% or more.
- [ ] Record the A1 worklog and remaining A2/A3 serial boundaries.
