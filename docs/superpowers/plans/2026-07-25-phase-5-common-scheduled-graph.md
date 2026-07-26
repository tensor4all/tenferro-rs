# Phase 5 Common Scheduled Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Do not use subagents unless the user or active agent instructions explicitly re-enable them. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move CPU compiled graph execution onto a runtime-owned Phase 5 path backed by prepared programs, scheduled graph metadata, explicit transfer/event boundaries, and a generic tensor-backend execution bridge.

**Architecture:** Keep the public `GraphExecutor<B>` as a legacy facade through Phase 5, but make the new execution contract `Runtime::run_compiled*`. `GraphCompiler` returns only the frozen semantic program and bindings; runtime preparation owns staging and schedule construction. `EngineRegistration::with_tensor_backend_executor` stores a runtime-private erased executor over any `TensorBackend`, so `tenferro-runtime` remains independent of `tenferro-cpu`.

**Tech Stack:** Rust workspace crates `tenferro-runtime`, `tenferro-cpu`, `tenferro-ad`; existing `ExecProgram`/segment evaluator as private transitional CPU payload; Python documentation/source consistency checks; cargo unit/integration/doc tests.

---

## File map

- Modify `crates/tenferro-runtime/src/graph/program.rs`: remove direct `ExecProgram` storage from `CompiledGraph`; expose only `FrozenProgram`, compiler options, semantic program, bindings, input/output counts.
- Modify `crates/tenferro-runtime/src/graph/compiler.rs`: return compiled graphs without retaining execution staging; keep transient staging for validation/cache stats until Phase 8 removes the old compiler cache surface.
- Modify `crates/tenferro-runtime/src/graph/executor.rs`: keep legacy `GraphExecutor<B>` working by staging with the compiled graph's stored compiler options at the legacy execution boundary; mark this as transitional.
- Modify `crates/tenferro-runtime/src/compiler/semantic_staging.rs`: rename/delete the Phase-4-only private adapter name `lower_semantic_to_exec_staging`; keep `stage_semantic_program` as the only private staging builder until Phase 8.
- Modify `crates/tenferro-runtime/src/runtime/engine_registration.rs`: add runtime-owned tensor-backend execution bridge fields and public builder/accessor methods.
- Create `crates/tenferro-runtime/src/runtime/execution.rs`: private erased execution bridge, `Runtime::run_compiled*` implementation helpers, input resolution, prepared projection validation, and execution through the bridge.
- Create `crates/tenferro-runtime/src/runtime/schedule.rs`: crate-private `ScheduledGraph`, node families, event-domain IDs, buffer/admission summaries, and validation helpers.
- Modify `crates/tenferro-runtime/src/runtime/mod.rs`: wire `execution` and `schedule` modules and re-export only deliberate public Phase 5 items.
- Modify `crates/tenferro-runtime/src/runtime/preparation.rs`: expose crate-private prepared root/staging/identity accessors for runtime execution, and add an options-aware `CompiledGraph` preparation helper; no new public API.
- Modify `crates/tenferro-runtime/src/error.rs`: add a typed runtime execution/preparation conversion only if existing `RuntimeStateSource` cannot preserve the source. Prefer existing variants.
- Modify `crates/tenferro-cpu/src/runtime_adapter.rs` and `crates/tenferro-cpu/src/lib.rs`: add public `runtime_engine_registration(&CpuBackend)` that includes preparation slots, cache owner, and tensor-backend execution bridge.
- Modify `crates/tenferro-ad/src/eager_backend.rs`: replace duplicate private CPU registration construction with `tenferro_cpu::runtime_engine_registration`.
- Add/update tests under `crates/tenferro-runtime/src/runtime/tests/`, `crates/tenferro-runtime/src/graph/executor/tests.rs`, `crates/tenferro-runtime/tests/integration/`, `crates/tenferro-cpu/src/runtime_adapter/tests.rs`, and `crates/tenferro-ad/src/eager/tests/`.
- Modify docs after code lands: `docs/design/execution-engine-provider-architecture.md`, `docs/architecture/tenferro-crates.md`, `docs/guides/parallelism-and-caching.md`, and add a Phase 5 worklog.

## Task 1: RED source contracts for Phase 5 ownership

**Files:**
- Modify: `crates/tenferro-runtime/src/graph/executor/tests.rs`
- Modify: `crates/tenferro-runtime/src/runtime/tests/preparation.rs`
- Test: `cargo test -p tenferro-runtime graph::executor::tests::phase5_source_contracts --lib`
- Test: `cargo test -p tenferro-runtime runtime::tests::preparation::phase5_preparation_source_contracts --lib`

- [ ] **Step 1: Add failing graph source-contract tests**

Add this module to `crates/tenferro-runtime/src/graph/executor/tests.rs`:

```rust
mod phase5_source_contracts {
    fn repo_file(path: &str) -> String {
        let mut root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        root.push("../..");
        root.push(path);
        std::fs::read_to_string(root).expect("source file must be readable")
    }

    #[test]
    fn compiled_graph_no_longer_retains_execution_staging() {
        let source = repo_file("crates/tenferro-runtime/src/graph/program.rs");
        let body = source
            .split_once("pub struct CompiledGraph")
            .and_then(|(_, rest)| rest.split_once("impl CompiledGraph"))
            .map(|(body, _)| body)
            .expect("CompiledGraph struct body");

        assert!(body.contains("CompilerOptions"));
        assert!(!body.contains("ExecProgram"));
        assert!(!body.contains("staging"));
    }

    #[test]
    fn graph_compiler_does_not_stage_on_compile() {
        let source = repo_file("crates/tenferro-runtime/src/graph/compiler.rs");
        let compile_frozen = source
            .split_once("fn compile_frozen")
            .and_then(|(_, rest)| rest.split_once("/// Compile multiple traced outputs"))
            .map(|(body, _)| body)
            .expect("compile_frozen body");

        assert!(compile_frozen.contains("CompiledGraph::new"));
        assert!(compile_frozen.contains("self.compiler_options"));
        assert!(!compile_frozen.contains("staging,"));
    }
}
```

- [ ] **Step 2: Add failing preparation source-contract tests**

Add to `crates/tenferro-runtime/src/runtime/tests/preparation.rs`:

```rust
#[test]
fn phase5_deletes_phase4_only_staging_adapter_name() {
    let source = repo_file("crates/tenferro-runtime/src/compiler/semantic_staging.rs");

    assert!(!source.contains("lower_semantic_to_exec_staging"));
    assert_eq!(source.matches("pub(crate) fn stage_semantic_program").count(), 1);
}

#[test]
fn phase5_runtime_execution_module_is_the_only_new_execution_owner() {
    let runtime_mod = repo_file("crates/tenferro-runtime/src/runtime/mod.rs");
    let graph_executor = repo_file("crates/tenferro-runtime/src/graph/executor.rs");

    assert!(runtime_mod.contains("mod execution;"));
    assert!(runtime_mod.contains("mod schedule;"));
    assert!(graph_executor.contains("legacy"));
}
```

- [ ] **Step 3: Run RED tests**

Run:

```bash
cargo test -p tenferro-runtime graph::executor::tests::phase5_source_contracts --lib
cargo test -p tenferro-runtime runtime::tests::preparation::phase5 --lib
```

Expected: fail because `CompiledGraph` still stores `staging`, `CompiledGraph` does not store compiler options, `lower_semantic_to_exec_staging` still exists, and runtime modules are absent.

## Task 2: Move staging out of `CompiledGraph`

**Files:**
- Modify: `crates/tenferro-runtime/src/graph/program.rs`
- Modify: `crates/tenferro-runtime/src/graph/compiler.rs`
- Modify: `crates/tenferro-runtime/src/graph/executor.rs`
- Modify: `crates/tenferro-runtime/src/compiler/semantic_staging.rs`
- Test: Task 1 tests
- Test: `cargo test -p tenferro-runtime graph::compiler --lib`
- Test: `cargo test -p tenferro-runtime graph::executor --lib`

- [ ] **Step 1: Rename the private staging builder internals**

In `crates/tenferro-runtime/src/compiler/semantic_staging.rs`, rename the private function body from `lower_semantic_to_exec_staging` to `build_exec_staging`. Keep only:

```rust
pub(crate) fn stage_semantic_program(
    program: &SemanticProgram,
    options: CompilerOptions,
) -> Result<ExecProgram> {
    build_exec_staging(program, options)
}
```

Do not keep the old function name in comments or tests.

- [ ] **Step 2: Remove staging from `CompiledGraph`**

Change `CompiledGraph` to:

```rust
use crate::compiler::CompilerOptions;

#[derive(Clone)]
pub struct CompiledGraph {
    pub(crate) frozen: FrozenProgram,
    pub(crate) compiler_options: CompilerOptions,
}

impl CompiledGraph {
    pub(crate) fn new(frozen: FrozenProgram, compiler_options: CompilerOptions) -> Self {
        Self {
            frozen,
            compiler_options,
        }
    }

    pub(crate) fn frozen(&self) -> &FrozenProgram {
        &self.frozen
    }

    pub(crate) fn compiler_options(&self) -> CompilerOptions {
        self.compiler_options
    }
}
```

Keep the existing public `program`, `bindings`, `input_count`, `output_count`, and `Debug` behavior.

- [ ] **Step 3: Make `GraphCompiler` stop retaining staging in the returned graph**

Change `compile_frozen` to transiently stage with `self.compiler_options` for validation/cache stats, call `get_or_compile`, and then return `CompiledGraph::new(frozen.clone(), self.compiler_options)`. Do not store the returned `ExecProgram` in `CompiledGraph`.

Change `compile_many_with_descriptors` similarly: keep transient staging and `get_or_compile` so existing compile-cache behavior remains observable, but return `CompiledGraph::new(semantic, self.compiler_options)`.

- [ ] **Step 4: Keep legacy `GraphExecutor<B>` functional by restaging at execution boundary**

Add a private helper in `graph/executor.rs`:

```rust
fn legacy_stage_compiled_graph(program: &CompiledGraph) -> Result<ExecProgram> {
    crate::compiler::semantic_staging::stage_semantic_program(
        program.program(),
        program.compiler_options(),
    )
}
```

Replace `&program.staging` uses with a local `let staging = legacy_stage_compiled_graph(program)?;`.

Add an `// INVARIANT:` comment near the helper:

```rust
// INVARIANT: `GraphExecutor<B>` is the Phase 5 legacy compatibility facade.
// Runtime-owned execution reaches staging through `Runtime::prepare_for`;
// Phase 8 owns this facade's retirement.
```

- [ ] **Step 5: Run GREEN tests**

Run:

```bash
cargo test -p tenferro-runtime graph::executor::tests::phase5_source_contracts --lib
cargo test -p tenferro-runtime runtime::tests::preparation::phase5 --lib
cargo test -p tenferro-runtime graph::compiler --lib
cargo test -p tenferro-runtime graph::executor --lib
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add crates/tenferro-runtime/src/compiler/semantic_staging.rs \
        crates/tenferro-runtime/src/graph/program.rs \
        crates/tenferro-runtime/src/graph/compiler.rs \
        crates/tenferro-runtime/src/graph/executor.rs
git commit -m "refactor(runtime): move staging behind execution boundary"
```

## Task 3: Add runtime-owned tensor-backend execution bridge

**Files:**
- Create: `crates/tenferro-runtime/src/runtime/execution.rs`
- Modify: `crates/tenferro-runtime/src/runtime/engine_registration.rs`
- Modify: `crates/tenferro-runtime/src/runtime/mod.rs`
- Test: `crates/tenferro-runtime/src/runtime/tests/snapshot.rs`
- Test: `crates/tenferro-runtime/tests/integration/runtime_public_api.rs`

- [ ] **Step 1: Write failing execution-bridge tests**

Add an integration test to `crates/tenferro-runtime/tests/integration/runtime_execution.rs`:

```rust
use std::sync::Arc;

use tenferro_cpu::CpuBackend;
use tenferro_runtime::{
    CoreCapabilityBundle, EngineId, EngineRegistration, ExecutionContextIdentity,
    HardwareClassId, StorageClass,
};

#[test]
fn engine_registration_records_tensor_backend_execution_bridge() {
    let storage = StorageClass::new("tenferro.storage.host").unwrap();
    let registration = EngineRegistration::new(
        EngineId::new("tenferro.engine.exec").unwrap(),
        ExecutionContextIdentity::of::<()>(),
        HardwareClassId::new("tenferro.hardware.host").unwrap(),
        Arc::from(vec![storage.clone()]),
        storage,
        CoreCapabilityBundle::builder().build(),
    )
    .unwrap();

    assert!(!registration.has_execution_engine());

    let registration = registration.with_tensor_backend_executor(CpuBackend::new());

    assert!(registration.has_execution_engine());
}
```

- [ ] **Step 2: Implement private erased bridge**

Create `runtime/execution.rs` with:

```rust
use std::fmt;
use std::sync::Mutex;

use tenferro_tensor::{Tensor, TensorBackend, TensorValue};

use crate::error::{Error, ErrorPhase, Result};
use crate::exec::{ExecProgram, ExecSlot};
use crate::extension_runtime::ExtensionExecutor;

pub(super) trait ErasedTensorBackendExecutor: fmt::Debug + Send + Sync {
    fn run_exec_program(
        &self,
        program: &ExecProgram,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>>;

    fn run_exec_program_values(
        &self,
        program: &ExecProgram,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<TensorValue>>;
}

pub(super) struct TensorBackendExecutor<B: TensorBackend + Clone + Send + Sync + 'static> {
    state: Mutex<TensorBackendExecutorState<B>>,
}

struct TensorBackendExecutorState<B: TensorBackend + Clone + Send + Sync + 'static> {
    backend: B,
    backend_cache: B::RuntimeCache,
    extension_executor: ExtensionExecutor<B>,
    slot_workspace: Vec<Option<ExecSlot<'static>>>,
}
```

Implement `Debug` without dumping backend internals. Implement execution by calling existing private segment helpers with the stored backend/cache/workspace. Convert poisoned locks to `Error::RuntimeState`.

- [ ] **Step 3: Add `EngineRegistration` methods and field**

Add:

```rust
pub(super) execution_engine: Option<Arc<dyn ErasedTensorBackendExecutor>>,
```

Initialize it to `None` in `EngineRegistration::new`, clone it in derived clone, and include `execution_engine` boolean in `Debug`.

Add public methods:

```rust
pub fn with_tensor_backend_executor<B>(mut self, backend: B) -> Self
where
    B: TensorBackend + Clone + Send + Sync + 'static,
{
    self.execution_engine = Some(Arc::new(super::execution::TensorBackendExecutor::new(backend)));
    self
}

pub fn has_execution_engine(&self) -> bool {
    self.execution_engine.is_some()
}
```

Every new public method needs rustdoc and a compiling example.

- [ ] **Step 4: Wire module**

In `runtime/mod.rs`, add:

```rust
mod execution;
mod schedule;
```

Do not re-export private bridge types.

- [ ] **Step 5: Run bridge tests**

Run:

```bash
cargo test -p tenferro-runtime runtime::tests::snapshot::engine_registration_records_tensor_backend_execution_bridge --lib
cargo test -p tenferro-runtime runtime::tests::snapshot --lib
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add crates/tenferro-runtime/src/runtime/execution.rs \
        crates/tenferro-runtime/src/runtime/engine_registration.rs \
        crates/tenferro-runtime/src/runtime/mod.rs \
        crates/tenferro-runtime/src/runtime/tests/snapshot.rs
git commit -m "feat(runtime): add tensor backend execution bridge"
```

## Task 4: Add `Runtime::run_compiled*`

**Files:**
- Modify: `crates/tenferro-runtime/src/runtime/execution.rs`
- Modify: `crates/tenferro-runtime/src/runtime/snapshot.rs`
- Modify: `crates/tenferro-runtime/src/runtime/preparation.rs`
- Test: `crates/tenferro-runtime/tests/integration/runtime_execution.rs` or existing integration module

- [ ] **Step 1: Write failing runtime execution integration tests**

Create or extend an integration test with:

```rust
use tenferro_cpu::{runtime_engine_registration, CpuBackend};
use tenferro_runtime::{GraphCompiler, Runtime, Tensor, TracedTensor};

#[test]
fn runtime_run_compiled_matches_legacy_graph_executor_for_add() {
    let x = TracedTensor::input_symbolic_shape(tenferro_runtime::DType::F64, 1).unwrap();
    let y = (&x + &x).unwrap();
    let program = GraphCompiler::new()
        .compile_with_input_specs(&y, &[(&x, tenferro_runtime::DType::F64, &[3])])
        .unwrap();
    let input = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();

    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(runtime_engine_registration(&backend).unwrap())
        .unwrap();
    let runtime = builder.build().unwrap();

    let out = runtime.run_compiled(&program, &[&input]).unwrap();

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);
}

#[test]
fn runtime_run_compiled_uses_prepared_cache_on_second_call() {
    let x = TracedTensor::input_symbolic_shape(tenferro_runtime::DType::F64, 1).unwrap();
    let y = (&x + &x).unwrap();
    let program = GraphCompiler::new()
        .compile_with_input_specs(&y, &[(&x, tenferro_runtime::DType::F64, &[3])])
        .unwrap();
    let input = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();

    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(runtime_engine_registration(&backend).unwrap())
        .unwrap();
    let runtime = builder.build().unwrap();

    runtime.run_compiled(&program, &[&input]).unwrap();
    let after_first = runtime.cache_stats().unwrap().prepared_plans;
    runtime.run_compiled(&program, &[&input]).unwrap();
    let after_second = runtime.cache_stats().unwrap().prepared_plans;

    assert_eq!(after_first.misses, 1);
    assert_eq!(after_second.hits, after_first.hits + 1);
    assert_eq!(after_second.entries, after_first.entries);
}
```

- [ ] **Step 2: Add prepared accessors and options-aware preparation**

In `runtime/preparation.rs`, add crate-private accessors:

```rust
impl PreparedProgramRoot {
    pub(crate) fn staging(&self) -> &ExecProgram {
        &self.staging
    }

    pub(crate) fn engine_id(&self) -> &EngineId {
        &self.identity.engine_id
    }

    pub(crate) fn epoch(&self) -> RuntimeEpoch {
        self.identity.epoch
    }
}

impl PreparedProgram {
    pub(crate) fn root(&self) -> &Arc<PreparedProgramRoot> {
        &self.root
    }

    pub(crate) fn specialization(&self) -> &SpecializationProjection {
        &self.specialization
    }
}
```

Also add an options-aware preparation path used only by Phase 5 runtime
execution:

```rust
pub(crate) fn prepare_compiled_for(
    runtime: &Runtime,
    caches: &RuntimeCacheSet<PreparedEntryKey, PreparedProgram>,
    program: &CompiledGraph,
    signature: &InputSignature,
    options: &PrepareOptions,
) -> PreparedProgramResult<Arc<PreparedProgram>> {
    prepare_for_with_compiler_options(
        runtime,
        caches,
        program.frozen(),
        program.compiler_options(),
        signature,
        options,
    )
}
```

Refactor the existing `prepare_for` implementation so default-options callers
delegate to `prepare_for_with_compiler_options(..., CompilerOptions::default(),
...)`, while `PreparationContext`/`PreparedProgramRoot` construction stages
with the supplied options instead of hard-coded defaults.

- [ ] **Step 3: Implement input resolution and signature derivation**

In `runtime/execution.rs`, implement helpers that:

- use supplied ordered tensors when `inputs` is non-empty;
- otherwise use `CompiledGraph::bindings()` defaults in semantic input order;
- validate dtype/rank/shape with the same logic as legacy `GraphExecutor`;
- build `InputSignature::from_reads`;
- convert borrowed reads to owned tensors only at execution boundary.

Do not retain `ProgramBindings` or borrowed tensor reads after the function returns.

- [ ] **Step 4: Implement `Runtime::run_compiled*`**

In `runtime/snapshot.rs`, add public methods with rustdoc examples:

```rust
pub fn run_compiled(
    &self,
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> crate::Result<Vec<Tensor>> {
    super::execution::run_compiled(self, program, inputs)
}

pub fn run_compiled_values(
    &self,
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> crate::Result<Vec<TensorValue>> {
    super::execution::run_compiled_values(self, program, inputs)
}
```

Implementation flow:

1. Resolve input tensors and reads.
2. Derive `InputSignature`.
3. Call the crate-private options-aware preparation helper for `CompiledGraph`.
4. Re-project `prepared.specialization().requirements().project(&signature)` and require equality with `prepared.specialization()`.
5. Look up the selected engine in the current snapshot.
6. Require `execution_engine.is_some()`.
7. Execute `prepared.root().staging()` through the bridge.

- [ ] **Step 5: Add missing-bridge and stale-epoch tests**

Add tests:

- an engine with preparation slots but no tensor-backend bridge prepares but `run_compiled` returns a runtime-state error before admission;
- after reconfiguration changes epoch, the convenience path re-prepares and still returns the correct output.

- [ ] **Step 6: Run runtime execution tests**

Run:

```bash
cargo test -p tenferro-runtime --test integration runtime_execution
cargo test -p tenferro-runtime runtime::tests::preparation --lib
cargo test -p tenferro-runtime
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add crates/tenferro-runtime/src/runtime/execution.rs \
        crates/tenferro-runtime/src/runtime/snapshot.rs \
        crates/tenferro-runtime/src/runtime/preparation.rs \
        crates/tenferro-runtime/tests/integration
git commit -m "feat(runtime): execute compiled graphs through runtime"
```

## Task 5: Add scheduled graph/event/transfer boundary

**Files:**
- Create/modify: `crates/tenferro-runtime/src/runtime/schedule.rs`
- Modify: `crates/tenferro-runtime/src/runtime/execution.rs`
- Test: `crates/tenferro-runtime/src/runtime/tests/schedule.rs`
- Modify: `crates/tenferro-runtime/src/runtime/tests/mod.rs`

- [ ] **Step 1: Write failing schedule boundary tests**

Create `crates/tenferro-runtime/src/runtime/tests/schedule.rs` with tests:

```rust
#[test]
fn transfer_node_bridges_distinct_event_domains() {
    let source = EventDomainId::runtime_created_for_test(1);
    let destination = EventDomainId::runtime_created_for_test(2);
    let transfer = ScheduledTransfer::for_test(source, destination);

    assert_ne!(transfer.source_event_domain(), transfer.destination_event_domain());
    assert_eq!(transfer.dependencies()[0].domain(), source);
    assert_eq!(transfer.completion().domain(), destination);
}

#[test]
fn collective_node_is_representable_but_execution_is_unsupported() {
    let graph = ScheduledGraph::for_test(vec![ScheduledNode::Collective(
        ScheduledCollective::unsupported_for_test(),
    )]);

    assert!(graph.contains_collective());
    assert!(graph.validate().is_ok());
    assert!(graph.execute_for_test().unwrap_err().to_string().contains("collective"));
}
```

Use `pub(crate)` test constructors behind `#[cfg(test)]`; do not make these public APIs.

- [ ] **Step 2: Implement schedule node types**

Add `ScheduledGraph`, `ScheduledNode`, `ScheduledOperation`, `ScheduledTransfer`, `ScheduledCollective`, `ScheduledBarrier`, `EventDomainId`, `EventSlotId`, `EventDependency`, `EventCompletion`, `BufferPlan`, `RunAdmissionSummary`, and `NodeAdmissionSummary`.

Keep constructors crate-private. Implement `Debug`, `Clone` only where needed, and retained-byte helpers where schedule metadata owns heap allocations.

- [ ] **Step 3: Integrate schedule construction into runtime execution**

In `execution.rs`, build a `ScheduledGraph` from the prepared root before bridge execution. The initial CPU schedule is one segment containing operation nodes derived from `PreparedProgram::operations()` and the prepared staging payload.

Do not execute transfer/collective nodes in production except for the synchronous mock transfer test path.

- [ ] **Step 4: Add two-mock-engine transfer test**

Add a unit test that constructs:

```text
engine-a operation -> transfer a-to-b -> engine-b operation
```

Assert the source operation completion domain is not reused as the destination completion domain. This test is schedule-level and hardware-free.

- [ ] **Step 5: Run schedule tests**

Run:

```bash
cargo test -p tenferro-runtime runtime::tests::schedule --lib
cargo test -p tenferro-runtime runtime::tests::preparation --lib
cargo test -p tenferro-runtime
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add crates/tenferro-runtime/src/runtime/schedule.rs \
        crates/tenferro-runtime/src/runtime/execution.rs \
        crates/tenferro-runtime/src/runtime/tests/mod.rs \
        crates/tenferro-runtime/src/runtime/tests/schedule.rs
git commit -m "feat(runtime): add scheduled graph boundary"
```

## Task 6: CPU helper, eager bridge cleanup, docs, and Phase 5 closeout

**Files:**
- Modify: `crates/tenferro-cpu/src/runtime_adapter.rs`
- Modify: `crates/tenferro-cpu/src/lib.rs`
- Modify: `crates/tenferro-cpu/src/runtime_adapter/tests.rs`
- Modify: `crates/tenferro-ad/src/eager_backend.rs`
- Modify: `crates/tenferro-ad/src/eager/tests/runtime_snapshot.rs`
- Modify: `docs/architecture/tenferro-crates.md`
- Modify: `docs/design/execution-engine-provider-architecture.md`
- Modify: `docs/guides/parallelism-and-caching.md`
- Create: `docs/worklogs/2026-07-25-phase-5-common-scheduled-graph.md`
- Test: `cargo test -p tenferro-cpu runtime_adapter --lib`
- Test: `cargo test -p tenferro-ad eager::tests::runtime_snapshot --lib`

- [ ] **Step 1: Write failing CPU helper tests**

Add to `crates/tenferro-cpu/src/runtime_adapter/tests.rs`:

```rust
#[test]
fn public_cpu_runtime_registration_includes_execution_bridge() {
    let backend = CpuBackend::new();
    let registration = crate::runtime_engine_registration(&backend).unwrap();

    assert_eq!(registration.engine_id().as_str(), "tenferro-cpu.default.v1");
    assert!(registration.has_execution_engine());
}
```

- [ ] **Step 2: Implement CPU helper**

Move the engine ID/hardware/storage constants and registration builder from `tenferro-ad/src/eager_backend.rs` into `tenferro-cpu/src/runtime_adapter.rs` as public helper functions:

```rust
pub fn runtime_engine_registration(
    backend: &CpuBackend,
) -> Result<EngineRegistration, RuntimeConfigError> {
    // Build direct core capability bundle from Arc<CpuBackend>.
    // Attach cache owner.
    // Attach with_tensor_backend_executor(backend.clone()).
}
```

Re-export it from `crates/tenferro-cpu/src/lib.rs`.

- [ ] **Step 3: Delegate eager CPU registration**

In `tenferro-ad/src/eager_backend.rs`, replace the private registration builder with:

```rust
pub(crate) fn cpu_runtime_engine_registration(
    backend: &CpuBackend,
) -> Result<EngineRegistration, RuntimeConfigError> {
    tenferro_cpu::runtime_engine_registration(backend)
}
```

Keep private helper names used by existing eager tests if needed, but do not duplicate constants or capability assembly.

- [ ] **Step 4: Update docs/worklog**

Document:

- Phase 5 runtime-owned blocking path exists as `Runtime::run_compiled*`;
- `GraphExecutor<B>` is legacy staging through Phase 5;
- `EngineRegistration::with_tensor_backend_executor` is the runtime-owned bridge and does not add a runtime-to-CPU dependency;
- transfer/collective/barrier nodes are scheduler-owned boundaries; collectives remain unsupported until a later child.

- [ ] **Step 5: Run Phase 5 closeout gates**

Run:

```bash
cargo test -p tenferro-runtime
cargo test -p tenferro-cpu
cargo test -p tenferro-ad
cargo clippy -p tenferro-runtime --all-targets -- -D warnings
cargo clippy -p tenferro-cpu --all-targets -- -D warnings
cargo clippy -p tenferro-ad --all-targets -- -D warnings
cargo fmt --all --check
python3 scripts/test-doc-consistency.py
python3 scripts/check-doc-snippets.py
python3 scripts/check-guide-dependency-snippets.py
python3 scripts/check-docs-site.py
python3 scripts/check-public-error-docs.py
git diff --check
scripts/check-pr-fast.sh --coverage-reviewed \
  --test 'cargo test -p tenferro-runtime' \
  --test 'cargo test -p tenferro-cpu' \
  --test 'cargo test -p tenferro-ad'
```

- [ ] **Step 6: Repository-rules review and commit**

Run deterministic review if the external LLM reviewer is still unusable:

```bash
python3 scripts/repository-rules-review.py \
  --base HEAD \
  --worktree \
  --dry-run \
  --llm-skipped-reason 'Phase 5 local deterministic review; external LLM reviewer unavailable if HTTP 400 recurs' \
  --output-json /tmp/repository-rules-review-p5-worktree-dry-run.json
```

Then commit:

```bash
git add crates/tenferro-runtime crates/tenferro-cpu crates/tenferro-ad \
        docs/architecture/tenferro-crates.md \
        docs/design/execution-engine-provider-architecture.md \
        docs/guides/parallelism-and-caching.md \
        docs/worklogs/2026-07-25-phase-5-common-scheduled-graph.md
git commit -m "feat(runtime): route graph execution through runtime"
```

## Self-review checklist

- Spec coverage: the tasks cover `CompiledGraph` staging removal, runtime-owned execution bridge, `Runtime::run_compiled*`, schedule node families, explicit transfer/event boundary, CPU helper, docs, worklog, and closeout gates.
- Intentional deferrals: public async `submit`, `ExecutionHandle`, real device events, real transfer providers, collective providers, GPU native execution, XLA `SubgraphCompiler`, and Phase 6 extension-family migration remain out of Phase 5.
- Public surface discipline: only `Runtime::run_compiled*`, `EngineRegistration::with_tensor_backend_executor`, `EngineRegistration::has_execution_engine`, and `tenferro_cpu::runtime_engine_registration` are new public APIs in this plan.
- TDD: every production task begins with a failing behavior/source-contract test.
- No PR: this plan stops at local commits; PR creation remains deferred until Phase 8 and benchmark campaign completion.
