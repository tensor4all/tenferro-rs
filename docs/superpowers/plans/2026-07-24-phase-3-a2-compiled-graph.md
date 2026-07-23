# Phase 3 A2 CompiledGraph Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the public `GraphProgram` execution boundary with
`CompiledGraph`, route `GraphExecutor` through semantic artifacts plus the sole
private staging adapter, and remove peer access to execution-slot views.

**Architecture:** `CompiledGraph` owns one `FrozenProgram` and the temporary
runtime-private execution staging generated from it. Public input/output
contracts come from `SemanticProgram`; tensor defaults remain in
`ProgramBindings`. Native execution accepts ordered tensors, while only
`tenferro-runtime` can access staging until Phase 5 deletes it.

**Tech Stack:** Rust, `tenferro-runtime::program`, computegraph compatibility
tracing, CPU/GPU graph executors, Cargo tests/clippy/rustdoc.

---

### Task 1: Freeze the `CompiledGraph` artifact

**Files:**
- Modify: `crates/tenferro-runtime/src/graph/program.rs`
- Modify: `crates/tenferro-runtime/src/graph/mod.rs`
- Modify: `crates/tenferro-runtime/src/lib.rs`
- Test: `crates/tenferro-runtime/src/graph/program.rs`
- Test: `crates/tenferro-runtime/tests/integration/public_surface_contract.rs`

- [x] Add a public-surface test requiring `CompiledGraph`; Task 4 strengthens
  it to reject `GraphProgram`, `GraphProgramInput`, and lowering-view exports
  after their callers migrate.
- [x] Replace the legacy container with:

```rust
pub struct CompiledGraph {
    pub(crate) staging: ExecProgram,
    pub(crate) frozen: FrozenProgram,
}

impl CompiledGraph {
    pub fn program(&self) -> &SemanticProgram;
    pub fn bindings(&self) -> &ProgramBindings;
    pub fn input_count(&self) -> usize;
    pub fn output_count(&self) -> usize;
}
```

  `Debug` must remain a bounded count/fingerprint summary and must not traverse
  bindings or extension payload debug output.
- [x] Keep the staging field and all slot/lowering-view access private to
  `tenferro-runtime`; the temporary public staging re-export needed by the
  serial einsum owner is removed only after semantic AD freezes. Do not add a
  reverse staging adapter.
- [x] Run `cargo test -p tenferro-runtime --lib graph::program` and the public
  surface integration test; expected result is PASS.
- [x] Commit the artifact boundary.

### Task 2: Return `CompiledGraph` from graph compilation

**Files:**
- Modify: `crates/tenferro-runtime/src/graph/compiler.rs`
- Modify: `crates/tenferro-runtime/src/graph/cache.rs`
- Modify: `crates/tenferro-runtime/src/compiler/semantic_staging.rs`
- Test: `crates/tenferro-runtime/src/graph/compiler.rs`
- Test: `crates/tenferro-runtime/tests/integration/semantic_program.rs`

- [x] Add failing tests proving every public compile entry returns the same
  frozen semantic artifact/bindings as before and staging is created only by
  `lower_semantic_to_exec_staging`.
- [x] Change compiler construction to:

```rust
let frozen = self.compile_materialized_semantic_program(...)?;
let staging = lower_semantic_to_exec_staging(
    frozen.program.as_ref(),
    &exact_input_shapes,
)?;
Ok(CompiledGraph::new(frozen, staging))
```

- [x] Preserve the existing private staging cache and exact collision checks
  for this compatibility stage, and never expose cached staging through public
  accessors. The TraceContext/pure-compiler task replaces this cache with the
  accepted semantic compiler key before Phase 3 closes.
- [x] Run runtime compiler, semantic-program, cache, and release tests;
  expected result is PASS.
- [x] Commit the compiler migration.

### Task 3: Migrate `GraphExecutor` to ordered semantic inputs

**Files:**
- Modify: `crates/tenferro-runtime/src/graph/executor.rs`
- Modify: `crates/tenferro-runtime/src/graph/executor/tests.rs`
- Modify: `crates/tenferro-runtime/src/graph/executor/tests/preflight.rs`
- Test: `crates/tenferro-runtime/tests/integration/graph_default_input_placement.rs`

- [ ] Add failing tests for ordered input count/dtype/shape validation, default
  binding lookup, borrowed reads, multi-output order, deferred-zero synthesis,
  shape guards, and lazy-output preservation using `CompiledGraph`.
- [ ] Replace traced-tensor keyed public bindings with ordered inputs:

```rust
pub fn run_with_inputs(
    &mut self,
    graph: &CompiledGraph,
    inputs: &[&Tensor],
) -> Result<Tensor>;

pub fn run_many_with_inputs(
    &mut self,
    graph: &CompiledGraph,
    inputs: &[&Tensor],
) -> Result<Vec<Tensor>>;
```

  Apply the same ordered contract to `TensorRead` and `TensorValue` variants.
  Empty explicit input slices mean “use frozen defaults”; non-empty slices
  must cover every semantic input in order.
- [ ] Resolve metadata through `SemanticProgram::inputs()` and
  `value_metadata`; resolve defaults through the frozen binding map. Perform
  all validation before entering a backend session.
- [ ] Keep raw staging evaluation methods crate-private and covered only by
  runtime unit tests.
- [ ] Run executor unit/integration tests on CPU and compile all GPU-feature
  entry points; expected result is PASS.
- [ ] Commit the executor migration.

### Task 4: Migrate in-repository callers

**Files:**
- Modify: all Rust/docs callers returned by
  `rg -n 'GraphProgram|input_specs\\(|lowering_view\\(|run_.*\\(&program, &\\[\\(&' crates docs`
- Test: affected crate integration suites and doctests

- [ ] Add a repository source contract that permits `GraphProgram` only in the
  contract's forbidden-symbol string and permits `ExecProgram` only in
  `tenferro-runtime` plus the serial einsum owner pending semantic AD.
- [ ] Change compiler result annotations to `CompiledGraph`, semantic queries
  to `program()`/`bindings()`, and explicit executions to ordered tensors such
  as:

```rust
let graph = compiler.compile_with_input_specs(&output, &specs)?;
let value = executor.run_with_inputs(&graph, &[&lhs, &rhs])?;
```

- [ ] Update docs/examples without compatibility shims or deprecated aliases.
- [ ] Run affected runtime, AD, FFT, linalg, einsum, XLA, GPU compile, and
  doctest gates; expected result is PASS except the already declared serial
  symbolic-einsum/XLA hold.
- [ ] Commit caller migration.

### Task 5: A2 checkpoint

**Files:**
- Create: `docs/worklogs/2026-07-24-phase-3-a2-compiled-graph.md`
- Modify: `docs/design/execution-engine-provider-architecture.md`

- [ ] Prove by source search that public `GraphProgram`, `GraphProgramInput`,
  and graph lowering views are absent and exactly one private semantic-to-
  staging adapter remains.
- [ ] Run debug/release runtime and affected workspace tests, doctests, clippy,
  formatting, and `git diff --check`.
- [ ] Apply the agreed benchmark rule: remeasure only if a measured eager path
  changes by at least 5%, and block only a reproducible slowdown of roughly
  50% or more.
- [ ] Record the semantic-AD/TraceContext/einsum serial boundaries still owned
  by P3-A3 and commit the checkpoint.
