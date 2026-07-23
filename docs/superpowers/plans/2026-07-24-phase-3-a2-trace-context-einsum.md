# Phase 3 A2 TraceContext and Einsum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the semantic `TraceContext -> TracedGraph -> GraphCompiler` boundary and atomically replace `GraphCompilerEinsumExt` with four `TraceContextEinsumExt` methods accepting opaque `TraceValue`s.

**Architecture:** `TraceContext` wraps `SemanticProgramBuilder`, owns ordered input descriptors, defaults, and trace-time extension caches, and issues opaque `TraceValue` handles. Consuming `finish` produces an immutable `TracedGraph` containing one `FrozenProgram`; `GraphCompiler::compile_traced_graph` accepts that artifact without inspecting a runtime or backend. Einsum tracing emits one pure `EinsumExtensionOp` directly into the semantic builder; concrete lowering remains a compiler/runtime concern.

**Tech Stack:** Rust, `tenferro-runtime::program`, `tenferro-runtime::extension_cache`, `tenferro-einsum`, semantic extension ops, Cargo tests/clippy/rustdoc.

---

### Task 1: Add the semantic trace frontier

**Files:**
- Create: `crates/tenferro-runtime/src/trace.rs`
- Modify: `crates/tenferro-runtime/src/lib.rs`
- Test: `crates/tenferro-runtime/src/trace/tests.rs`
- Test: `crates/tenferro-runtime/tests/integration/public_surface_contract.rs`

- [ ] **Step 1: Add failing ownership and freeze tests**

  Test that two contexts reject each other's `TraceValue` with
  `ProgramBuildError::ForeignValue`, input/default order survives `finish`,
  duplicate output handles remain ordered, and debug output never reveals raw
  slots or builder nonces.

- [ ] **Step 2: Run the focused tests and observe missing symbols**

  Run:

  ```bash
  cargo test -p tenferro-runtime trace::tests -- --nocapture
  ```

  Expected: compile failure because `TraceContext`, `TraceValue`, and
  `TracedGraph` do not exist.

- [ ] **Step 3: Implement the opaque semantic trace types**

  Add these public shapes without exposing `ProgramValue`:

  ```rust
  pub struct TraceContext {
      builder: SemanticProgramBuilder,
      inputs: Vec<TraceInputDescriptor>,
      extension_caches: ExtensionCacheStore,
  }

  #[derive(Clone, Copy)]
  pub struct TraceValue {
      value: ProgramValue,
  }

  #[derive(Clone)]
  pub struct TracedGraph {
      frozen: FrozenProgram,
      inputs: Box<[TraceInputDescriptor]>,
  }
  ```

  Provide `new`, `input`, `input_with_default`, `bind_input`, `add_op`,
  `add_extension`, `value_metadata`, `extension_caches_mut`, and consuming
  `finish`. Delegate all ownership, arity, effect, alias, metadata, and
  binding validation to `SemanticProgramBuilder`; do not add a second token
  registry or process-global fallback.

- [ ] **Step 4: Verify the trace frontier**

  Run:

  ```bash
  cargo test -p tenferro-runtime trace::tests
  cargo test -p tenferro-runtime --test integration public_surface_contract
  ```

  Expected: PASS.

### Task 2: Add the pure traced-graph compiler entry

**Files:**
- Modify: `crates/tenferro-runtime/src/graph/compiler.rs`
- Modify: `crates/tenferro-runtime/src/graph/program.rs`
- Test: `crates/tenferro-runtime/src/graph/compiler.rs`
- Test: `crates/tenferro-runtime/tests/integration/semantic_program.rs`

- [ ] **Step 1: Add a failing compile-and-execute test**

  Build two ordered trace inputs, add a core `Add`, finish a `TracedGraph`,
  compile it, execute it with two ordered tensors, and assert the semantic
  program/output order and numerical result.

- [ ] **Step 2: Run the focused test and observe the missing compiler entry**

  Run:

  ```bash
  cargo test -p tenferro-runtime graph::compiler::tests::compile_traced_graph
  ```

  Expected: compile failure because `compile_traced_graph` does not exist.

- [ ] **Step 3: Implement forward-only semantic compilation**

  Add:

  ```rust
  pub fn compile_traced_graph(
      &mut self,
      graph: &TracedGraph,
  ) -> Result<CompiledGraph>
  ```

  Clone the immutable `FrozenProgram`, lower it only through
  `lower_semantic_to_exec_staging`, reuse the existing exact staging cache,
  and construct private compatibility input descriptors in the same semantic
  input order. Do not add a semantic-to-legacy reverse adapter.

- [ ] **Step 4: Verify compiler and executor behavior**

  Run:

  ```bash
  cargo test -p tenferro-runtime graph::compiler
  cargo test -p tenferro-runtime graph::executor
  cargo test -p tenferro-runtime --test integration semantic_program
  ```

  Expected: PASS.

### Task 3: Replace GraphCompilerEinsumExt atomically

**Files:**
- Modify: `crates/tenferro-einsum/src/traced.rs`
- Modify: `crates/tenferro-einsum/src/extension.rs`
- Modify: `crates/tenferro-einsum/src/lib.rs`
- Modify: all Rust and documentation callers found by
  `rg -n 'GraphCompilerEinsumExt|compiler\\.einsum|einsum\\(&mut (compiler|engine)' crates ext docs`
- Test: `crates/tenferro-einsum/tests/integration/traced_correctness.rs`
- Test: `crates/tenferro-einsum/tests/integration/traced_graph_cache.rs`
- Test: `crates/tenferro-xla/tests/integration/stablehlo_lowering.rs`

- [ ] **Step 1: Add failing public API and ownership tests**

  Require exactly these four methods:

  ```rust
  pub trait TraceContextEinsumExt {
      fn einsum(&mut self, inputs: &[TraceValue], subscripts: &str)
          -> Result<TraceValue>;
      fn einsum_subscripts(
          &mut self,
          inputs: &[TraceValue],
          subscripts: &EinsumSubscripts,
      ) -> Result<TraceValue>;
      fn einsum_with(
          &mut self,
          inputs: &[TraceValue],
          subscripts: &str,
          optimize: EinsumOptimize,
      ) -> Result<TraceValue>;
      fn einsum_subscripts_with(
          &mut self,
          inputs: &[TraceValue],
          subscripts: &EinsumSubscripts,
          optimize: EinsumOptimize,
      ) -> Result<TraceValue>;
  }
  ```

  Test ordered n-ary inputs, malformed notation, foreign values, explicit
  paths, concrete precomputed trees, symbolic constraints, and execution
  after `finish` plus `compile_traced_graph`.

- [ ] **Step 2: Run the focused API tests and observe the old boundary**

  Run:

  ```bash
  cargo test -p tenferro-einsum --features autodiff \
    traced_correctness::trace_context_einsum_ext_exposes_einsum
  ```

  Expected: compile failure because only `GraphCompilerEinsumExt` exists.

- [ ] **Step 3: Emit semantic einsum directly**

  Parse through `TraceContext::extension_caches_mut`, validate ordered input
  metadata, convert `EinsumOptimize` to `EinsumPlanSpec`, attach a concrete
  `ContractionTree` only as an identity-excluded hint, and call
  `TraceContext::add_extension(Arc::new(EinsumExtensionOp), inputs)`. Do not
  expand to computegraph or call `GraphCompiler` while tracing.

- [ ] **Step 4: Remove the old trait and migrate every caller in the same commit**

  Delete `GraphCompilerEinsumExt`, its implementation, and the four free
  functions that accept `GraphCompiler`. Migrate examples/tests to:

  ```rust
  let mut trace = TraceContext::new();
  let lhs = trace.input_with_default(lhs_spec, lhs_tensor)?;
  let rhs = trace.input_with_default(rhs_spec, rhs_tensor)?;
  let output = trace.einsum(&[lhs, rhs], "ij,jk->ik")?;
  let graph = trace.finish(&[output])?;
  let compiled = compiler.compile_traced_graph(&graph)?;
  ```

  Keep `TracedTensorEinsumExt::tensordot` only where the eager/traced AD
  compatibility surface still consumes `TracedTensor`; it is not a compiler
  extension point.

- [ ] **Step 5: Prove the replacement is atomic**

  Run:

  ```bash
  test -z "$(rg -n 'GraphCompilerEinsumExt' crates ext docs --glob '*.rs' --glob '*.md')"
  rg -n 'TraceContextEinsumExt' crates/tenferro-einsum/src crates/tenferro-einsum/tests
  ```

  Expected: the forbidden-symbol search is empty and the new trait appears in
  implementation plus tests.

- [ ] **Step 6: Verify einsum and XLA**

  Run:

  ```bash
  cargo test -p tenferro-einsum --features autodiff
  cargo test -p tenferro-xla --tests
  ```

  Expected: PASS.

### Task 4: Checkpoint the trace boundary

**Files:**
- Modify: `docs/design/execution-engine-provider-architecture.md`
- Modify: `docs/superpowers/plans/2026-07-24-phase-3-a3-semantic-ad.md`
- Create: `docs/worklogs/2026-07-24-phase-3-a2-trace-context-einsum.md`

- [ ] **Step 1: Record exact completed and remaining contracts**

  Record semantic trace ownership, ordered inputs, pure compiler entry,
  extension-first einsum, and remaining legacy `TracedTensor` consumers owned
  by semantic AD/linalg migration. Do not claim Phase 3 complete.

- [ ] **Step 2: Run the checkpoint gates**

  Run:

  ```bash
  cargo fmt --all -- --check
  cargo clippy -p tenferro-runtime --all-targets -- -D warnings
  cargo clippy -p tenferro-einsum --features autodiff --all-targets -- -D warnings
  cargo test -p tenferro-runtime
  cargo test -p tenferro-einsum --features autodiff
  git diff --check
  ```

  Expected: PASS.

- [ ] **Step 3: Apply the benchmark policy**

  This trace/compile-only API migration does not change the standalone eager
  execution path. Run a benchmark only if an eager-path measurement changes
  by at least 5%; stop only for a reproducible slowdown of roughly 50% or
  more.

- [ ] **Step 4: Commit the atomic replacement**

  ```bash
  git add crates/tenferro-runtime crates/tenferro-einsum crates/tenferro-xla docs
  git commit -m "feat(trace): move einsum tracing into trace contexts"
  ```
