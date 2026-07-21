# Phase 2 CPU Domain Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement issue #1436: object-safe CPU domain executors, explicit
parallel modes, managed and externally managed NUMA domains, independent
provider count/placement capabilities, and a placement-bound eager session.

**Architecture:** Preserve the current `CpuBackend` session and global
`ResourceArbiter` as the sole admission boundary. Generalize `CpuEngine` to own
an immutable resource domain containing an executor, then pass one borrowed
`CpuExecutionContext` through engine composites and providers. External domains
are retained in one placement-indexed coordinator and use declared CPU sets for
cooperative arbitration without tenferro repinning or claiming live affinity.

**Tech Stack:** Rust 2021, Rayon, faer, optional CBLAS/LAPACK/TBLIS providers,
Criterion, tenferro typed errors, rustdoc/doctest, repository source-contract
tests.

---

## File Structure

- `crates/tenferro-tensor/src/types.rs`: backend-neutral `CpuDomainId` and
  optional CPU-affinity placement metadata.
- `crates/tenferro-tensor/src/tests/types_tests.rs`: placement metadata tests.
- `crates/tenferro-cpu/src/domain_executor.rs`: object-safe jobs, executor
  capabilities, errors, and stack-owned typed helper adapters.
- `crates/tenferro-cpu/src/domain_executor/tests.rs`: executor contract tests.
- `crates/tenferro-cpu/src/resource_domain.rs`: resource-domain values,
  `ExternalCpuDomain`, placement guarantees, provider-axis validation, and
  `CpuExecutionContext`.
- `crates/tenferro-cpu/src/resource_domain/tests.rs`: descriptor, lifetime, and
  validation tests.
- `crates/tenferro-cpu/src/provider_capability.rs`: BLAS/native count and
  placement classifications resolved at construction.
- `crates/tenferro-cpu/src/provider_capability/tests.rs`: injected probe tests.
- `crates/tenferro-cpu/src/affinity_policy.rs`: dominant-input and
  require-single-domain resolution.
- `crates/tenferro-cpu/src/affinity_policy/tests.rs`: deterministic resolver
  tests.
- `crates/tenferro-cpu/src/context.rs`: pinned Rayon implementation of
  `CpuDomainExecutor`.
- `crates/tenferro-cpu/src/engine.rs`: one `CpuResourceDomain` plus mutable
  engine-local resources.
- `crates/tenferro-cpu/src/backend.rs`: managed/external registries,
  construction, selection, diagnostics, admission, and placement-tagged output.
- `crates/tenferro-cpu/src/provider.rs`: `ParallelMode`, provider capability
  methods, and `CpuExecutionContext` request arguments.
- `crates/tenferro-cpu/src/dot_runtime.rs`: engine-owned outer/inner grouped
  scheduling through the executor.
- `crates/tenferro-cpu/src/exec_session.rs`: create and reuse one execution
  context below one backend-session entry.
- `crates/tenferro-ad/src/eager.rs`: `CpuPlacementBoundEager` bridge.
- `crates/tenferro-ad/src/eager_backend.rs`: checked clone of a CPU coordinator
  handle without exposing enum internals.
- `crates/tenferro-cpu/benches/numa_execution.rs`: executor and placement
  diagnostics/benchmarks.
- `docs/design/execution-engine-provider-architecture.md`,
  `docs/design/cpu-backend-execution.md`, `docs/guides/cpu-execution.md`, and
  `docs/guides/parallelism-and-caching.md`: normative and rendered behavior.
- `docs/worklogs/2026-07-21-phase-2-cpu-domain-executors.md`: curated evidence.

### Task 1: Add backend-neutral CPU-affinity metadata

**Files:**

- Modify: `crates/tenferro-tensor/src/types.rs`
- Modify: `crates/tenferro-tensor/src/lib.rs`
- Modify: `crates/tenferro-tensor/src/tests/types_tests.rs`
- Modify: every existing `Placement { ... }` literal reported by
  `rg -l 'Placement\s*\{' crates docs -g '*.rs'`

- [ ] **Step 1: Write failing placement tests**

Add tests that demand a caller-stable ID and optional affinity distinct from
device placement:

```rust
#[test]
fn cpu_domain_id_is_stable_caller_metadata() {
    let id = CpuDomainId::new(17);
    assert_eq!(id.as_u64(), 17);
    assert_eq!(id, CpuDomainId::new(17));
}

#[test]
fn cpu_affinity_is_not_a_device_boundary() {
    let placement = Placement {
        memory_kind: MemoryKind::UnpinnedHost,
        device: None,
        cpu_affinity: Some(CpuDomainId::new(3)),
    };
    assert_eq!(placement.cpu_affinity, Some(CpuDomainId::new(3)));
    assert!(placement.device.is_none());
    assert!(Placement::default().cpu_affinity.is_none());
}
```

- [ ] **Step 2: Verify RED**

Run:

```bash
cargo test -p tenferro-tensor cpu_domain_id_is_stable_caller_metadata
```

Expected: compile failure because `CpuDomainId` and `Placement::cpu_affinity`
do not exist.

- [ ] **Step 3: Implement the ID and metadata**

Add:

```rust
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CpuDomainId(u64);

impl CpuDomainId {
    pub const fn new(id: u64) -> Self { Self(id) }
    pub const fn as_u64(self) -> u64 { self.0 }
}

pub struct Placement {
    pub memory_kind: MemoryKind,
    pub device: Option<DeviceId>,
    pub cpu_affinity: Option<CpuDomainId>,
}
```

Set `cpu_affinity: None` in `default_placement()` and every unrelated existing
literal. Add runnable examples for both `CpuDomainId` methods and update the
`Placement` example.

- [ ] **Step 4: Verify GREEN and compile fallout**

Run:

```bash
cargo test -p tenferro-tensor
cargo check --workspace --all-targets
```

Expected: all tensor tests pass and every placement literal has an explicit
affinity value.

- [ ] **Step 5: Commit**

```bash
git add crates docs
git commit -m "feat(tensor): add CPU affinity placement metadata"
```

### Task 2: Define the object-safe executor contract

**Files:**

- Create: `crates/tenferro-cpu/src/domain_executor.rs`
- Create: `crates/tenferro-cpu/src/domain_executor/tests.rs`
- Modify: `crates/tenferro-cpu/src/lib.rs`

- [ ] **Step 1: Write failing object-safety and borrowed-job tests**

The tests must instantiate `&dyn CpuDomainExecutor`, capture a borrowed stack
value in `install`, and run indexed outer jobs without heap-owned tensor input:

```rust
#[test]
fn executor_is_object_safe_and_accepts_borrowed_jobs() {
    let executor = InlineExecutor::new();
    let object: &dyn CpuDomainExecutor = &executor;
    let input = 41usize;
    let output = Cell::new(0usize);
    object.install(&mut scoped_job(|| output.set(input + 1))).unwrap();
    assert_eq!(output.get(), 42);
}

#[test]
fn outer_submission_is_synchronous_and_indexed() {
    let executor = InlineExecutor::new();
    let seen = [AtomicUsize::new(0), AtomicUsize::new(0)];
    executor
        .submit(&indexed_jobs(2, |index| {
            seen[index].fetch_add(1, Ordering::Relaxed);
            Ok(())
        }))
        .unwrap();
    assert_eq!(seen.map(|value| value.load(Ordering::Relaxed)), [1, 1]);
}
```

- [ ] **Step 2: Verify RED**

Run `cargo test -p tenferro-cpu domain_executor::tests --lib`.

Expected: compile failure because the executor module and traits do not exist.

- [ ] **Step 3: Implement exact executor values**

Implement `ScopedCpuJob`, `ScopedCpuJobs`, `CpuDomainExecutor`,
`CpuDomainExecutorCapabilities`, `CpuInnerParallelism`,
`CpuExecutorReentrancy`, `CpuExecutorAffinity`, `CpuExecutorShutdown`, and
`CpuDomainExecutorError`. Keep trait methods object-safe and public types
documented with runnable examples. Implement crate-private stack helpers whose
closures retain their own operation result so executor errors do not erase
tensor errors.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
cargo test -p tenferro-cpu domain_executor::tests --lib
cargo test -p tenferro-cpu --doc domain_executor
```

Expected: borrowed values remain valid, every logical job executes once, and
all doctests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/tenferro-cpu/src/domain_executor.rs crates/tenferro-cpu/src/domain_executor crates/tenferro-cpu/src/lib.rs
git commit -m "feat(cpu): define object-safe domain executor"
```

### Task 3: Adapt managed `CpuContext` to the executor contract

**Files:**

- Modify: `crates/tenferro-cpu/src/context.rs`
- Modify: `crates/tenferro-cpu/src/context/tests.rs`
- Modify: `crates/tenferro-cpu/src/engine.rs`
- Modify: `crates/tenferro-cpu/src/engine/tests.rs`

- [ ] **Step 1: Write failing managed-adapter tests**

Cover sequential, inner install, outer indexed submission, worker affinity
classification, and no leakage into a different ambient Rayon pool:

```rust
#[test]
fn managed_context_reports_verified_rayon_capabilities() {
    let ctx = CpuContext::with_threads(2).unwrap();
    let caps = CpuDomainExecutor::capabilities(&ctx);
    assert_eq!(caps.worker_count.get(), 2);
    assert!(caps.outer_parallelism);
    assert_eq!(caps.inner_parallelism, CpuInnerParallelism::Rayon);
    assert_eq!(caps.shutdown, CpuExecutorShutdown::TenferroOwned);
}
```

- [ ] **Step 2: Verify RED**

Run `cargo test -p tenferro-cpu managed_context_reports_verified_rayon_capabilities`.

Expected: compile failure because `CpuContext` does not implement the trait.

- [ ] **Step 3: Implement the Rayon adapter**

Implement `CpuDomainExecutor for CpuContext`. `submit` enters the owned pool
once and runs `0..jobs.len()` with Rayon `try_for_each`; `install` reuses the
current matching worker scope or enters the pool once. A single-thread context
runs directly. Preserve `ExecutionScopeState` registration and the public
backend re-entry panic contract.

Change `CpuEngine` to retain `Arc<dyn CpuDomainExecutor>` through its resource
domain in Task 4, but keep the current `Arc<CpuContext>` constructor available
internally until that task lands.

- [ ] **Step 4: Verify GREEN and baseline parity**

Run:

```bash
cargo test -p tenferro-cpu context::tests --lib
cargo test -p tenferro-cpu engine::tests --lib
cargo test -p tenferro-cpu tests::context --lib
```

Expected: all existing context/pinning/re-entry tests and new capability tests
pass.

- [ ] **Step 5: Commit**

```bash
git add crates/tenferro-cpu/src/context.rs crates/tenferro-cpu/src/context crates/tenferro-cpu/src/engine.rs crates/tenferro-cpu/src/engine
git commit -m "refactor(cpu): adapt managed contexts to domain executors"
```

### Task 4: Add resource-domain and external-domain descriptors

**Files:**

- Create: `crates/tenferro-cpu/src/resource_domain.rs`
- Create: `crates/tenferro-cpu/src/resource_domain/tests.rs`
- Modify: `crates/tenferro-cpu/src/lib.rs`
- Modify: `crates/tenferro-cpu/src/placement.rs`

- [ ] **Step 1: Write failing descriptor tests**

Test empty sets, zero/mismatched worker budgets, exact/advisory guarantees,
public diagnostics, and executor lifetime:

```rust
#[test]
fn external_domain_retains_executor_owner() {
    let drops = Arc::new(AtomicUsize::new(0));
    let executor = Arc::new(DropCountingExecutor::new(Arc::clone(&drops), 2));
    let domain = ExternalCpuDomain::new(
        CpuDomainId::new(7),
        node_placement(0, &[0, 1]),
        executor,
        NonZeroUsize::new(2).unwrap(),
        CpuPlacementGuarantee::ExactDeclared,
    )
    .unwrap();
    assert_eq!(drops.load(Ordering::Relaxed), 0);
    drop(domain);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
}
```

- [ ] **Step 2: Verify RED**

Run `cargo test -p tenferro-cpu resource_domain::tests --lib`.

Expected: compile failure because the resource-domain types do not exist.

- [ ] **Step 3: Implement descriptors and typed validation**

Implement `CpuPlacementGuarantee::{ExactDeclared, AdvisoryDeclared}`,
`CpuDomainOwnership::{Managed, ExternalManaged}`,
`ExternalCpuDomainError`, private `CpuResourceDomain`, and public
`ExternalCpuDomain` getters. Validate nonempty CPU sets, nonzero executor
workers, and `thread_budget <= worker_count` in `ExternalCpuDomain::new`.
Backend-wide process-cpuset/default/duplicate checks stay in Task 5.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
cargo test -p tenferro-cpu resource_domain::tests --lib
cargo test -p tenferro-cpu --doc resource_domain
```

- [ ] **Step 5: Commit**

```bash
git add crates/tenferro-cpu/src/resource_domain.rs crates/tenferro-cpu/src/resource_domain crates/tenferro-cpu/src/lib.rs crates/tenferro-cpu/src/placement.rs
git commit -m "feat(cpu): define managed and external resource domains"
```

### Task 5: Build the placement-indexed ExternalManaged registry

**Files:**

- Modify: `crates/tenferro-cpu/src/backend.rs`
- Modify: `crates/tenferro-cpu/src/backend/tests.rs`
- Modify: `crates/tenferro-cpu/src/engine.rs`
- Modify: `crates/tenferro-cpu/src/engine/tests.rs`
- Modify: `crates/tenferro-cpu/src/placement.rs`
- Modify: `crates/tenferro-cpu/src/placement/tests.rs`
- Modify: `crates/tenferro-cpu/src/arbiter/tests.rs`

- [ ] **Step 1: Write failing registry and arbitration tests**

Add fixture tests proving one coordinator selects two retained executors,
disjoint domains overlap in time, overlapping domains serialize, and no
managed pool is constructed for an unregistered placement:

```rust
#[test]
fn external_registry_routes_without_reconstructing_executors() {
    let node0_runs = Arc::new(AtomicUsize::new(0));
    let node1_runs = Arc::new(AtomicUsize::new(0));
    let backend = external_backend_fixture(&node0_runs, &node1_runs).unwrap();
    let node1 = backend.for_placement(CpuPlacement::NumaNode(NumaNodeId::new(1))).unwrap();
    assert_eq!(node1.execution_info().execution_mode(), CpuExecutionMode::ExternalManaged);
    node1.install(|| {});
    assert_eq!(node0_runs.load(Ordering::Relaxed), 0);
    assert_eq!(node1_runs.load(Ordering::Relaxed), 1);
}
```

- [ ] **Step 2: Verify RED**

Run `cargo test -p tenferro-cpu external_registry_routes_without_reconstructing_executors`.

Expected: compile failure because the constructor and execution mode do not
exist.

- [ ] **Step 3: Implement registry construction and selection**

Add `CpuExecutionMode::ExternalManaged` and
`CpuBackend::from_external_managed_domains(default_domain, domains)`. Validate:

```text
domains not empty
unique CpuDomainId
unique placement identity
all declared CPUs inside topology.allowed_cpus()
default domain exists
AllAllowed exact domain equals topology.allowed_cpus()
```

Store external engines directly in the shared coordinator. `Auto` selects the
declared default. Explicit unregistered placement returns a typed error. Do not
call `CpuEngine::new` for any external domain. Continue to use the global
`ResourceArbiter` exact CPU-set request, including advisory declarations.

- [ ] **Step 4: Verify GREEN and old-mode parity**

Run:

```bash
cargo test -p tenferro-cpu backend::tests --lib
cargo test -p tenferro-cpu placement::tests --lib
cargo test -p tenferro-cpu arbiter::tests --lib
```

Expected: new external tests pass and Managed, Compatibility,
ProviderDefaultExclusive, fairness, poison, panic release, and re-entry tests
remain green.

- [ ] **Step 5: Commit**

```bash
git add crates/tenferro-cpu/src/backend.rs crates/tenferro-cpu/src/backend crates/tenferro-cpu/src/engine.rs crates/tenferro-cpu/src/engine crates/tenferro-cpu/src/placement.rs crates/tenferro-cpu/src/placement crates/tenferro-cpu/src/arbiter
git commit -m "feat(cpu): register externally managed NUMA domains"
```

### Task 6: Replace the phase-1 provider staging context

**Files:**

- Modify: `crates/tenferro-cpu/src/provider.rs`
- Modify: `crates/tenferro-cpu/src/provider/tests.rs`
- Modify: `crates/tenferro-cpu/src/dot_runtime.rs`
- Modify: `crates/tenferro-cpu/src/dot_runtime/tests.rs`
- Modify: `crates/tenferro-cpu/src/exec_session.rs`
- Modify: `crates/tenferro-cpu/src/backend.rs`
- Modify: `crates/tenferro-cpu/tests/provider_boundary_allocation_tests/main.rs`

- [ ] **Step 1: Write failing execution-context tests**

Require all provider traits to accept `&CpuExecutionContext`, expose exactly
three modes, and prevent direct lease/resource access:

```rust
#[test]
fn provider_context_reports_domain_policy_without_resource_ownership() {
    let fixture = execution_context_fixture(ParallelMode::Outer, 4);
    assert_eq!(fixture.context.parallel_mode(), ParallelMode::Outer);
    assert_eq!(fixture.context.thread_budget().get(), 4);
    assert_eq!(fixture.context.domain_id(), CpuDomainId::new(9));
}
```

Update the source-contract test to reject `CpuProviderContext` and
`CpuKernelParallelism` after migration.

- [ ] **Step 2: Verify RED**

Run `cargo test -p tenferro-cpu provider_context_reports_domain_policy_without_resource_ownership`.

Expected: compile failure because the new context and mode do not exist.

- [ ] **Step 3: Implement the atomic type migration**

Define `ParallelMode` and public `CpuExecutionContext` accessors. Change every
provider trait and built-in implementation to take the new context. Delete
`CpuProviderContext` and `CpuKernelParallelism` in the same commit. Construct
one context below `run_backend_session_cached`; direct provider/composite calls
borrow it and never call `CpuBackend::install`.

Change `CpuProviderBundle::with_provider_bundle` installation to return a typed
`Result` so domain/provider validation can be performed before installation.
Update all in-repository callers directly.

- [ ] **Step 4: Verify GREEN and allocation parity**

Run:

```bash
cargo test -p tenferro-cpu provider::tests --lib
cargo test -p tenferro-cpu dot_runtime::tests --lib
cargo test -p tenferro-cpu --test provider_boundary_allocation_tests
```

Expected: provider behavior is unchanged and warmed request construction and
dispatch still allocate zero.

- [ ] **Step 5: Commit**

```bash
git add crates/tenferro-cpu
git commit -m "refactor(cpu): pass one execution context to providers"
```

### Task 7: Classify provider count and placement capabilities

> **Status (2026-07-21): Task 7a implemented; Task 7b remains required.** The
> conservative trait, classification table, typed bundle/domain validation,
> capability-aware dispatch, and pre-mutation rejection are implemented.
> Built-in BLAS/TBLIS remain `GlobalOrUncontrolled` because no adapter yet
> applies and restores MKL or macOS 15 Accelerate local control per call.
> OpenBLAS is different: its `_local` entry point performs process-global
> set-and-restore, so wiring it can never establish strict per-call control.
> Task 7b must add scoped guards only for genuinely local mechanisms and may
> classify OpenBLAS global control solely for exclusive compatibility and
> diagnostics. The legacy BLAS
> `Auto`/`ProviderDefaultExclusive` compatibility path remains process-wide
> exclusive and must not be described as strict count or placement control.
> Review hardening snapshots each slot descriptor exactly once at bundle
> construction, validates every lazy managed NUMA domain, and validates an
> initial ExternalManaged bundle before returning the backend. The public
> `from_external_managed_domains_with_provider_bundle` route preserves a way
> to initialize a compiled-BLAS backend with caller-controlled providers even
> though the uncontrolled standard BLAS bundle is rejected. This bundle covers
> `dot_general`; linalg capability injection remains Task 7b work.

**Files:**

- Create: `crates/tenferro-cpu/src/provider_capability.rs`
- Create: `crates/tenferro-cpu/src/provider_capability/tests.rs`
- Modify: `crates/tenferro-cpu/src/provider.rs`
- Modify: `crates/tenferro-cpu/src/dot_runtime.rs`
- Modify: `crates/tenferro-cpu/src/backend.rs`
- Modify: `crates/tenferro-cpu/tests/integration/provider_feature_contract.rs`

- [ ] **Step 1: Write failing table-driven capability tests**

Use an injected symbol/probe fixture, not host-library assumptions:

```rust
#[test]
fn openblas_global_set_restore_never_claims_per_call_count() {
    let caps = classify_openblas(&FakeOpenBlasProbe::pthread_with_global_guard());
    assert_eq!(caps.thread_count, CpuThreadCountControl::GlobalOrUncontrolled);
    assert_eq!(caps.placement, CpuPlacementControl::ExternalWorkers);
    assert!(!caps.worker_local_sequential);
}

#[test]
fn strict_subdomain_rejects_external_workers_above_one_thread() {
    let error = validate_provider_for_domain(
        external_worker_caps(),
        &strict_subdomain_fixture(NonZeroUsize::new(2).unwrap()),
        &process_allowed_fixture(&[0, 1, 2, 3]),
    )
    .unwrap_err();
    assert!(matches!(error, CpuProviderDomainError::PlacementNotEnforceable { .. }));
}
```

Cover MKL, pthread/OpenMP OpenBLAS, macOS 15/older Accelerate, ArmPL `_mp`,
serial ArmPL/NVPL, and unknown/injected BLAS.

- [ ] **Step 2: Verify RED**

Run `cargo test -p tenferro-cpu provider_capability::tests --lib`.

Expected: compile failure because provider capabilities do not exist.

- [ ] **Step 3: Implement construction-time classification**

Implement `CpuProviderExecutionCapabilities`, `CpuThreadCountControl`, and
`CpuPlacementControl`; add required `execution_capabilities()` methods to
provider traits. Explicit compile features may supply known mechanisms, but
each provider slot is queried exactly once while its immutable bundle is built
and that descriptor is stored. Validation and hot-path dispatch never re-query
the provider. Unknown injected providers stay conservative unless constructed
with an explicit descriptor.

Validate strict/advisory domain compatibility before output mutation. Budget
one external calls are allowed inline; strict subdomain budget greater than one
with external workers is rejected unless the domain equals the process-allowed
set.

`BinaryClampToOne` means the adapter selects single-threaded mode for every
finite resource-domain budget; it never selects auto. An adapter that cannot
make this guarantee remains `GlobalOrUncontrolled`.

- [ ] **Step 4: Verify GREEN across feature contracts**

Run:

```bash
cargo test -p tenferro-cpu provider_capability::tests --lib
cargo test -p tenferro-cpu --test integration provider_feature_contract
cargo check -p tenferro-cpu --no-default-features --features cpu-faer
cargo check -p tenferro-cpu --no-default-features --features cpu-blas,provider-inject
```

Expected: every table entry is explicit and no unknown provider claims
thread-local count or exact placement.

- [ ] **Step 5: Commit**

```bash
git add crates/tenferro-cpu
git commit -m "feat(cpu): classify provider count and placement control"
```

### Task 8: Route grouped fan-out through the selected executor

**Files:**

- Modify: `crates/tenferro-cpu/src/dot_runtime.rs`
- Modify: `crates/tenferro-cpu/src/dot_runtime/tests.rs`
- Modify: `crates/tenferro-cpu/src/gemm/mod.rs`
- Modify: `crates/tenferro-cpu/src/gemm/tests.rs`
- Modify: `crates/tenferro-cpu/tests/provider_boundary_allocation_tests/main.rs`

- [ ] **Step 1: Write failing single-fan-out tests**

Instrument a fake executor and provider to prove:

```rust
#[test]
fn outer_mode_submits_once_and_forces_sequential_children() {
    let fixture = grouped_fixture(ParallelMode::Outer, 4, 8);
    fixture.execute().unwrap();
    assert_eq!(fixture.executor.submit_calls(), 1);
    assert_eq!(fixture.executor.install_calls(), 0);
    assert_eq!(fixture.provider.observed_modes(), vec![ParallelMode::Sequential; 8]);
}

#[test]
fn inner_mode_installs_once_without_outer_submission() {
    let fixture = grouped_fixture(ParallelMode::Inner, 4, 8);
    fixture.execute().unwrap();
    assert_eq!(fixture.executor.submit_calls(), 0);
    assert_eq!(fixture.executor.install_calls(), 1);
}
```

- [ ] **Step 2: Verify RED**

Run `cargo test -p tenferro-cpu outer_mode_submits_once_and_forces_sequential_children`.

Expected: failure because grouped scheduling still uses ambient `par_iter` and
the old context.

- [ ] **Step 3: Implement executor-owned fan-out**

Replace direct ambient Rayon selection in `execute_grouped` with
`CpuExecutionContext::submit`. The indexed job object performs the existing
prevalidated disjoint output writes. Each provider call receives
`context.sequential_child()`. `Inner` enters one checked executor install and
runs the outer loop sequentially. `Sequential` performs neither executor call.

Because `CpuDomainExecutor` is a public safe boundary, engine-owned fan-out
must audit every job as `UNCLAIMED`, `RUNNING`, or `COMPLETE`, reject duplicate
and out-of-range indices even when an executor discards the per-job error, and
reject a successful submission that omitted or did not complete a job. Pack
this required O(job_count) audit into two bits per job with four inline
`AtomicUsize` words: groups through `2 * usize::BITS` jobs do not spill, while
larger groups may allocate one `SmallVec` spill. This is a bounded small-group
property, not an unbounded zero-allocation guarantee. A submit error remains
authoritative over post-submit audit or provider errors.

Retain the current normative grouped and strided-batched policy. Do not adopt
#1426 threshold changes in this task.

- [ ] **Step 4: Verify GREEN and no allocation regression**

Run:

```bash
cargo test -p tenferro-cpu dot_runtime::tests --lib
cargo test -p tenferro-cpu gemm::tests --lib
cargo test -p tenferro-cpu --test provider_boundary_allocation_tests
```

The focused audit tests cover 8, 9, exactly `2 * usize::BITS`, one job beyond
that inline threshold, and a substantially larger group. They prove inline
versus spilled storage at each boundary, adjacent packed-state updates without
clobbering, duplicate-while-running exclusion before provider mutation,
ignored `len`/`usize::MAX` indices, missing and panic-interrupted jobs, exact
provider-error preservation, and submit-error precedence. The existing public
allocation probe must remain at or below its fixed-main baseline; it does not
turn the bounded grouped-state property into an unbounded promise.

- [ ] **Step 5: Commit**

```bash
git add crates/tenferro-cpu/src/dot_runtime.rs crates/tenferro-cpu/src/dot_runtime crates/tenferro-cpu/src/gemm crates/tenferro-cpu/tests/provider_boundary_allocation_tests
git commit -m "refactor(cpu): make domain executors own grouped fan-out"
```

### Task 9: Resolve CPU affinity and tag outputs

**Files:**

- Create: `crates/tenferro-cpu/src/affinity_policy.rs`
- Create: `crates/tenferro-cpu/src/affinity_policy/tests.rs`
- Modify: `crates/tenferro-cpu/src/backend.rs`
- Modify: `crates/tenferro-cpu/src/backend/tests.rs`
- Modify: `crates/tenferro-cpu/src/lib.rs`

- [ ] **Step 1: Write failing resolver tests**

```rust
#[test]
fn dominant_input_bytes_is_deterministic_and_keeps_inputs_in_place() {
    let inputs = [
        affinity_input(2, CpuDomainId::new(9)),
        affinity_input(8, CpuDomainId::new(4)),
        affinity_input(6, CpuDomainId::new(9)),
    ];
    let selected = resolve_cpu_affinity(
        CpuAffinityPolicy::DominantInputBytes,
        &inputs,
        CpuDomainId::new(1),
    )
    .unwrap();
    assert_eq!(selected.domain, CpuDomainId::new(9));
    assert_eq!(inputs[1].domain, Some(CpuDomainId::new(4)));
}
```

Also test stable-ID tie breaking, unknown affinity, explicit override, and
`RequireSingleDomain` rejection.

- [ ] **Step 2: Verify RED**

Run `cargo test -p tenferro-cpu affinity_policy::tests --lib`.

Expected: compile failure because the resolver does not exist.

- [ ] **Step 3: Implement pure resolution and output tagging**

Implement `CpuAffinityPolicy::{DominantInputBytes, RequireSingleDomain}` and a
pure resolver over `(Option<CpuDomainId>, logical_bytes)` entries using checked
byte counts. Explicit placement wins. Unknown inputs do not contribute. No
input is copied or retagged.

CPU outputs allocated by a selected domain receive
`placement.cpu_affinity = Some(domain_id)`. Preserve device and allocation
domain metadata independently.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
cargo test -p tenferro-cpu affinity_policy::tests --lib
cargo test -p tenferro-cpu backend::tests --lib
```

- [ ] **Step 5: Commit**

```bash
git add crates/tenferro-cpu
git commit -m "feat(cpu): resolve and report CPU affinity domains"
```

### Task 10: Add the placement-bound eager session bridge

**Files:**

- Modify: `crates/tenferro-ad/src/eager.rs`
- Modify: `crates/tenferro-ad/src/eager/tests.rs`
- Modify: `crates/tenferro-ad/src/eager_backend.rs`
- Modify: `crates/tenferro-ad/src/lib.rs`
- Modify: `crates/tenferro-ad/tests/integration/eager_tensor.rs`

- [ ] **Step 1: Write failing public behavior tests**

```rust
#[test]
fn placement_bound_session_reuses_runtime_identity_and_one_backend_entry() {
    let counters = SessionCounters::default();
    let runtime = counted_external_cpu_runtime(counters.clone());
    let mut socket0 = runtime.on_cpu(CpuPlacement::NumaNode(NumaNodeId::new(0))).unwrap();
    assert_eq!(socket0.runtime_id(), runtime.id());
    let output = socket0
        .with_eager_session(|session| run_add_session(session))
        .unwrap();
    assert_eq!(output.as_slice::<f64>().unwrap(), &[3.0]);
    assert_eq!(counters.backend_entries(), 1);
}
```

Also prove the context holds no permit while idle, requires mutable use instead
of a second mutex, rejects CUDA/WebGPU runtimes with a typed error, and retains
the selected external executor owner through the call.

- [ ] **Step 2: Verify RED**

Run `cargo test -p tenferro-ad placement_bound_session_reuses_runtime_identity_and_one_backend_entry`.

Expected: compile failure because `on_cpu` does not exist.

- [ ] **Step 3: Implement the bridge without recursive runtime locking**

Add:

```rust
pub struct CpuPlacementBoundEager {
    runtime: Arc<EagerRuntime>,
    backend: CpuBackend,
}

impl EagerRuntime {
    pub fn on_cpu(
        self: &Arc<Self>,
        placement: CpuPlacement,
    ) -> Result<CpuPlacementBoundEager>;
}

impl CpuPlacementBoundEager {
    pub fn runtime_id(&self) -> ContextId;
    pub fn placement(&self) -> CpuPlacement;
    pub fn with_eager_session<R: Send>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> Result<R> + Send,
    ) -> Result<R>;
}
```

`on_cpu` briefly locks the original backend, clones the CPU coordinator,
resolves placement, and releases the lock. The mutable context owns the cheap
selected handle directly, not another mutex. `with_eager_session` enters its
backend once and passes the existing `BackendSession` trait object. It does not
call ordinary `EagerTensor` operations inside the locked runtime.

- [ ] **Step 4: Verify GREEN and eager parity**

Run:

```bash
cargo test -p tenferro-ad eager::tests --lib
cargo test -p tenferro-ad --test integration eager_tensor::
cargo test -p tenferro-runtime graph::executor::tests
```

- [ ] **Step 5: Commit**

```bash
git add crates/tenferro-ad crates/tenferro-runtime
git commit -m "feat(ad): add placement-bound CPU eager sessions"
```

### Task 11: Documentation, benchmarks, and phase evidence

**Files:**

- Modify: `crates/tenferro-cpu/benches/numa_execution.rs`
- Modify: `docs/design/execution-engine-provider-architecture.md`
- Modify: `docs/design/cpu-backend-execution.md`
- Modify: `docs/guides/cpu-execution.md`
- Modify: `docs/guides/parallelism-and-caching.md`
- Create: `docs/worklogs/2026-07-21-phase-2-cpu-domain-executors.md`
- Modify: `docs/superpowers/specs/2026-07-21-phase-2-cpu-domain-executor-design.md`

- [ ] **Step 1: Add failing documentation/source contract checks**

Extend docs checks so the rendered parallelism guide must name Managed,
ExternalManaged, exact/advisory placement, caller affinity/shutdown ownership,
the one-fan-out rule, and external-BLAS limits. Add a source-contract test that
rejects ambient `par_iter` in the engine composite and the removed phase-1
staging names.

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 scripts/check-docs-site.py
cargo test -p tenferro-cpu --test integration backend_capability_contracts
```

Expected: the new checks fail until docs and source are updated.

- [ ] **Step 3: Update normative/rendered docs and benchmark cases**

Document exact final signatures and ownership. Extend the NUMA benchmark with
managed/external `submit`, `install`, disjoint-domain overlap, and selected
placement diagnostics at 1, 2, and 4 threads. Keep multi-NUMA claims opt-in and
hardware-reported.

Write the worklog with source/issue/design inputs, rejected alternatives,
TDD evidence, feature matrices, allocation results, benchmark manifests,
repository review, and remaining risks.

- [ ] **Step 4: Run focused and workspace verification**

Run:

```bash
cargo fmt --all --check
cargo test -p tenferro-tensor
cargo test -p tenferro-cpu
cargo test -p tenferro-runtime
cargo test -p tenferro-ad --test integration eager_tensor::
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
python3 scripts/repository-rules-review.py --base origin/main --head HEAD --output-json /tmp/phase2-rules-review.json
```

Expected: every command succeeds and the review JSON reports zero unwaived
findings.

- [ ] **Step 5: Run the predeclared performance campaign**

Use the phase-1 interleaved campaign runner with its unchanged current-main
baseline, byte-identical lock, three valid pairs, ±5% sentinel, and +5% case
threshold. Record the complete manifest, raw estimates, allocation probe, and
classification under the phase-2 worklog artifact directory. Do not selectively
rerun valid pairs or replace `INCONCLUSIVE` with a pass.

- [ ] **Step 6: Commit**

```bash
git add crates/tenferro-cpu/benches docs
git commit -m "docs: record phase 2 CPU domain executor evidence"
```

- [ ] **Step 7: Push and update issues**

Push `codex/execution-engine-through-phase9`. Comment on #1436 with commit,
tests, performance classification, and worklog links. Update #1433's phase table
only when every exit criterion is proven; otherwise report the exact pending
gate without calling the phase complete.

## Review follow-up addendum: scoped linalg operation entry

This addendum closes the Phase 2 single-entry review seam without claiming
Phase 2D, Phase 2E, or a performance-gate PASS. It is deliberately limited to
the borrowed CPU linalg entry boundary and the managed Cholesky bypass found by
task-local review.

**Files and ownership seam:**

- Modify `crates/tenferro-linalg/src/cpu/backend.rs`: every public CPU linalg
  `_read` method owns exactly one `CpuBackend::with_linalg_pool` call. Helpers
  named `*_entered` and `managed_cholesky` consume the resulting
  `&CpuExecutionContext` and `&mut BufferPool`; they must not call back into a
  public backend method or reacquire the executor/pool.
- Modify `crates/tenferro-cpu/src/provider.rs` only when the materialization
  seam itself changes. `CpuExecutionContext::with_materialized_tensor_read` is
  the sole scoped bridge from `TensorRead` to a temporary compact host tensor,
  and it returns ordinary-path temporaries to the same `BufferPool`.
- Create `crates/tenferro-linalg/src/cpu/tests/single_entry.rs`: fake-executor
  call counts, managed-domain scope observations, error/panic recovery, and
  two-input nested materialization behavior.
- Modify `crates/tenferro-linalg/src/cpu/tests/mod.rs`: register the new test
  module with `mod single_entry;`.
- Modify `crates/tenferro-linalg/src/cpu/tests/managed_cholesky.rs`: the complete
  fake shared-allocation domain and its read/write/allocation observations.
- Modify
  `crates/tenferro-linalg/tests/integration/cpu_linalg_source_contract.rs`:
  path-sensitive source guards for one entry, non-entering helpers, and the
  private materialization token.

The nine guarded borrowed entry points are `triangular_solve_read`,
`solve_read`, `svd_read`, `qr_read`, `eigh_read`, `cholesky_read`, `lu_read`,
`full_piv_lu_read`, and `eig_read`. The two-input paths nest
`with_materialized_tensor_read` twice; the other seven use one scoped
materialization or a provider-native view fast path after entry.

- [x] **RED: expose the managed `cholesky_read` bypass.** Add nonzero and
  zero-size shared-allocation-domain tests. The nonzero reproducer must observe
  input map, output allocation, and output map/write outside the executor entry
  in the old implementation. The zero-size reproducer must observe zero
  installs in the old implementation. Both require `install = 1`, `submit = 0`,
  correct output, and a successful next operation.
- [x] **RED: make the source contract path-sensitive.** In addition to counting
  the lexical `with_linalg_pool` occurrence, require it to precede managed
  storage dispatch, reject a fallible/early-return prefix, and require the
  managed helper to accept entered context/pool arguments without accepting a
  mutable backend or calling `with_linalg_pool`.
- [x] **GREEN: move managed work below the sole entry.** Select the provider
  inside the entered closure, then perform managed input validation/map,
  zero-size handling, provider factorization, shared-domain allocation, dtype
  adaptation, output map/write, and wrapping through the non-entering helper.
  The owned `cholesky` surface uses the same helper so owned/read parity does
  not retain a second bypass.
- [x] **GREEN: preserve scoped materialization and recovery.** Keep all nine
  `_read` paths on `with_materialized_tensor_read`, preserve temporary reclaim
  on success and ordinary error, and retain the existing documented panic
  recovery/next-operation contract without adding pointer-identity promises.
- [x] **Verify the feature matrix.** Run the complete linalg unit and source
  contract suites for faer, BLAS/LAPACK, and their combined build:

  ```bash
  cargo test -p tenferro-linalg
  RUSTFLAGS='-l dylib=openblas -l dylib=lapack' \
    cargo test -p tenferro-linalg --no-default-features --features cpu-blas
  RUSTFLAGS='-l dylib=openblas -l dylib=lapack' \
    cargo test -p tenferro-linalg --features cpu-blas
  ```

  Also rerun `cargo test -p tenferro-cpu`, linalg and CPU doctests,
  `cargo clippy` for the changed feature combinations, `cargo fmt --all
  --check`, and the focused docs/source-contract checks. Record any
  environment-limited injected-provider link result as a limitation, not as a
  substituted PASS.
