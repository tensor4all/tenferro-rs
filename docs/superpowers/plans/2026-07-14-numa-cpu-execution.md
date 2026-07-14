# NUMA-Aware CPU Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add managed NUMA placement for faer/native CPU graphs, shared cloneable CPU engine coordination, exclusive unmanaged BLAS execution, and rendered user documentation.

**Architecture:** `CpuBackend` becomes a cloneable handle over shared topology, engine, and arbitration state. A placement-bound handle selects a pinned NUMA or lazy all-allowed Rayon engine for faer/native execution; BLAS-backed handles reject explicit placement and run under one exclusive provider reservation, entering the all-allowed native pool only for native segments. Runtime execution keeps one backend session across supported backend, Host, and FFI instructions.

**Tech Stack:** Rust, Rayon custom worker spawning, Linux `sched_getaffinity`/`sched_setaffinity`, Linux sysfs NUMA topology, `thiserror`, fixture-based unit tests, Cargo doctests, MkDocs user guides, Criterion benchmarks.

**Design:** `docs/superpowers/specs/2026-07-14-numa-cpu-execution-design.md`

---

## File map

New CPU modules:

- `crates/tenferro-cpu/src/topology.rs`: CPU/node IDs, canonical CPU sets, topology resolution, Linux discovery, and public diagnostics.
- `crates/tenferro-cpu/src/topology/tests.rs`: parser and injected-topology fixtures.
- `crates/tenferro-cpu/src/placement.rs`: placement requests, resolved/provider-exclusive modes, capability checks, and typed errors.
- `crates/tenferro-cpu/src/placement/tests.rs`: provider matrix and non-fallback tests.
- `crates/tenferro-cpu/src/engine.rs`: pinned Rayon engine, engine-owned buffers/cache, and execution instrumentation.
- `crates/tenferro-cpu/src/engine/tests.rs`: worker affinity and engine resource tests.
- `crates/tenferro-cpu/src/arbiter.rs`: overlap-aware permits and provider-exclusive reservation.
- `crates/tenferro-cpu/src/arbiter/tests.rs`: concurrency, ordering, and unwind release tests.
- `crates/tenferro-cpu/benches/numa_execution.rs`: opt-in placement/concurrency benchmark and metadata.

Primary modified CPU files:

- `crates/tenferro-cpu/src/affinity.rs`: expose CPU masks and current-thread set/restore helpers instead of count-only discovery.
- `crates/tenferro-cpu/src/context.rs`: add pinned-pool construction while retaining existing unpinned compatibility construction.
- `crates/tenferro-cpu/src/backend.rs`: shared backend state, placement-bound handles, aggregate resource APIs, and session routing.
- `crates/tenferro-cpu/src/exec_session.rs`: engine-owned state and provider/native context transitions.
- `crates/tenferro-cpu/src/lib.rs`: modules and public exports.
- `crates/tenferro-cpu/Cargo.toml`: NUMA benchmark target if explicit registration is required.

Runtime/session files:

- `crates/tenferro-tensor/src/backend.rs`: extend backend-session capabilities needed by Host execution without exposing sessions to users.
- `crates/tenferro-runtime/src/exec.rs`: one session loop for supported backend/Host/FFI instructions.
- `crates/tenferro-runtime/src/segment.rs`: run fused and boundary segments inside the graph session when supported.
- `crates/tenferro-runtime/src/graph/executor/tests.rs`: whole-graph session/pool-entry instrumentation.

Documentation:

- `docs/guides/cpu-execution.md`: rendered placement and mixed-provider guide.
- `docs/getting-started/index.md` and `docs/index.md`: discovery links and concise provider warning.
- `docs/design/cpu-backend-execution.md`: durable ownership/session architecture.
- `docs/worklogs/2026-07-14-numa-cpu-execution.md`: implementation record, evidence, and residual risks.

## Task 1: Add canonical topology and placement contracts

**Files:**

- Create: `crates/tenferro-cpu/src/topology.rs`
- Create: `crates/tenferro-cpu/src/topology/tests.rs`
- Create: `crates/tenferro-cpu/src/placement.rs`
- Create: `crates/tenferro-cpu/src/placement/tests.rs`
- Modify: `crates/tenferro-cpu/src/lib.rs:44-82`

- [ ] **Step 1: Write failing topology and capability tests**

Cover sorted/deduplicated CPU sets, sparse OS node IDs, affinity intersection,
empty-node removal, overlap rejection, provider capability, and non-fallback:

```rust
#[test]
fn topology_intersects_nodes_with_allowed_cpus_without_renumbering() {
    let allowed = CpuSet::new([8, 9, 10, 11, 12, 13, 14, 15]).unwrap();
    let topology = CpuTopology::from_discovered(
        allowed.clone(),
        [
            (NumaNodeId::new(2), CpuSet::new([0, 8, 9, 10, 11]).unwrap()),
            (NumaNodeId::new(7), CpuSet::new([12, 13, 14, 15, 16]).unwrap()),
            (NumaNodeId::new(9), CpuSet::new([20, 21]).unwrap()),
        ],
    ).unwrap();

    assert_eq!(topology.allowed_cpus(), &allowed);
    assert_eq!(topology.node_ids(), vec![NumaNodeId::new(2), NumaNodeId::new(7)]);
    assert_eq!(topology.node(NumaNodeId::new(2)).unwrap().cpus().as_slice(), &[8, 9, 10, 11]);
}

#[test]
fn explicit_external_provider_placement_never_falls_back() {
    let topology = two_node_fixture();
    assert!(matches!(
        resolve_placement(CpuBackendKind::Blas, CpuPlacement::NumaNode(NumaNodeId::new(0)), &topology),
        Err(CpuPlacementError::ExternalProviderAffinityUnmanaged { .. })
    ));
    assert!(matches!(
        resolve_placement(CpuBackendKind::Blas, CpuPlacement::AllAllowed, &topology),
        Err(CpuPlacementError::ExternalProviderAffinityUnmanaged { .. })
    ));
    assert_eq!(
        resolve_placement(CpuBackendKind::Blas, CpuPlacement::Auto, &topology).unwrap(),
        ResolvedCpuExecution::ProviderDefaultExclusive,
    );
}
```

- [ ] **Step 2: Run the focused tests and confirm missing APIs fail**

Run: `cargo test -p tenferro-cpu topology placement --release`

Expected: FAIL because the modules and types do not exist.

- [ ] **Step 3: Implement minimal public value types and pure resolution**

Define `CpuId`, `NumaNodeId`, `CpuSet`, `CpuNode`, `CpuTopology`,
`CpuPlacement`, `ResolvedCpuPlacement`, internal `ResolvedCpuExecution`, and
`CpuPlacementError`. Use sorted `Vec<CpuId>` storage, checked non-empty sets,
binary-search membership, linear merge intersection/overlap, and public
`# Examples` for every public item. Do not put provider details below
`CpuBackendKind::{Faer, Blas}` into the public enum surface.

- [ ] **Step 4: Run focused tests and doctests**

Run: `cargo test -p tenferro-cpu topology placement --release && cargo test -p tenferro-cpu --doc --release`

Expected: PASS.

- [ ] **Step 5: Commit the contract**

```bash
git add crates/tenferro-cpu/src/{topology.rs,topology/tests.rs,placement.rs,placement/tests.rs,lib.rs}
git commit -m "feat(cpu): define NUMA placement contracts"
```

## Task 2: Discover process CPU sets and Linux NUMA topology

**Files:**

- Modify: `crates/tenferro-cpu/src/affinity.rs`
- Modify: `crates/tenferro-cpu/src/affinity/tests.rs`
- Modify: `crates/tenferro-cpu/src/topology.rs`
- Modify: `crates/tenferro-cpu/src/topology/tests.rs`

- [ ] **Step 1: Add failing parsers and injected discovery tests**

```rust
#[test]
fn linux_cpu_list_parser_handles_ranges_gaps_and_whitespace() {
    assert_eq!(
        parse_linux_cpu_list("0-3,8,10-11\n").unwrap().as_usize_vec(),
        vec![0, 1, 2, 3, 8, 10, 11],
    );
}

#[test]
fn discovery_falls_back_to_all_allowed_when_node_files_are_unavailable() {
    let source = FixtureTopologySource::unavailable(CpuSet::new([4, 5]).unwrap());
    let topology = discover_from(&source).unwrap();
    assert!(topology.nodes().is_empty());
    assert_eq!(topology.allowed_cpus().as_usize_vec(), vec![4, 5]);
}
```

- [ ] **Step 2: Confirm the tests fail**

Run: `cargo test -p tenferro-cpu affinity topology --release`

Expected: FAIL for missing mask/parser/discovery functions.

- [ ] **Step 3: Implement mask discovery and the narrow OS source**

Refactor the Linux `sched_getaffinity` loop to return `CpuSet`; retain
`process_cpu_affinity_count()` as `.len()`. Parse
`/sys/devices/system/node/node*/cpulist` through a private `TopologySource`
boundary and pure fixture path. Unsupported OSes return one all-allowed domain.
Reject malformed lists, empty allowed masks, arithmetic overflow, and overlapping
usable nodes with typed construction errors.

- [ ] **Step 4: Verify platform-neutral and live invariants**

Run: `cargo test -p tenferro-cpu affinity topology --release`

Expected: PASS on single-node and multi-node hosts.

- [ ] **Step 5: Commit discovery**

```bash
git add crates/tenferro-cpu/src/{affinity.rs,affinity/tests.rs,topology.rs,topology/tests.rs}
git commit -m "feat(cpu): discover allowed NUMA domains"
```

## Task 3: Build verified pinned Rayon contexts

**Files:**

- Modify: `crates/tenferro-cpu/src/affinity.rs`
- Modify: `crates/tenferro-cpu/src/context.rs`
- Modify: `crates/tenferro-cpu/src/context/tests.rs`
- Create: `crates/tenferro-cpu/src/engine.rs`
- Create: `crates/tenferro-cpu/src/engine/tests.rs`
- Modify: `crates/tenferro-cpu/src/lib.rs`

- [ ] **Step 1: Write failing construction and observation tests**

```rust
#[test]
fn pinned_context_reports_only_assigned_cpus() {
    let allowed = process_cpu_affinity().unwrap();
    let selected = CpuSet::new(allowed.as_slice().iter().take(2).copied()).unwrap();
    let ctx = CpuContext::with_pinned_cpus(selected.clone(), selected.len()).unwrap();
    let observed = ctx.install(|| {
        (0..4096usize)
            .into_par_iter()
            .map(|_| current_cpu().unwrap())
            .collect::<BTreeSet<_>>()
    });
    assert!(observed.iter().all(|cpu| selected.contains(*cpu)));
}

#[test]
fn pin_failure_aborts_engine_construction() {
    let source = FailingAffinitySetter::new();
    assert!(matches!(
        CpuContext::with_pinned_cpus_using(CpuSet::new([0]).unwrap(), 1, source),
        Err(CpuContextError::WorkerPinning { .. })
    ));
}
```

- [ ] **Step 2: Confirm tests fail**

Run: `cargo test -p tenferro-cpu pinned worker_affinity --release`

Expected: FAIL for missing pinned construction.

- [ ] **Step 3: Implement worker pinning and verification**

Use Rayon's custom `spawn_handler`. Each OS worker sets its affinity before
`ThreadBuilder::run`, verifies the resulting mask contains only its assigned
logical CPU, and reports startup through a channel. Constructor success requires
one successful report per worker. Always use a real one-worker pool for pinned
engines; retain the direct-call one-thread behavior only for legacy unpinned
`CpuContext::with_threads(1)`.

Implement `CpuEngine` with immutable placement/context and a mutex-protected
`EngineResources { buffers, gemm_analysis_cache }`. Default workers equal CPU
set length; an explicit budget uses `min(budget, cpus.len())`.

- [ ] **Step 4: Run focused tests repeatedly**

Run: `cargo test -p tenferro-cpu context engine --release -- --test-threads=1`

Expected: PASS with every live observation inside the selected CPU set.

- [ ] **Step 5: Commit pinned engines**

```bash
git add crates/tenferro-cpu/src/{affinity.rs,context.rs,context/tests.rs,engine.rs,engine/tests.rs,lib.rs}
git commit -m "feat(cpu): build pinned execution engines"
```

## Task 4: Add overlap-aware resource arbitration

**Files:**

- Create: `crates/tenferro-cpu/src/arbiter.rs`
- Create: `crates/tenferro-cpu/src/arbiter/tests.rs`
- Modify: `crates/tenferro-cpu/src/lib.rs`

- [ ] **Step 1: Write failing concurrency and unwind tests**

```rust
#[test]
fn disjoint_domains_run_together_but_all_allowed_waits() {
    let arbiter = Arc::new(ResourceArbiter::new());
    let node0 = arbiter.acquire(CpuSet::new([0, 1]).unwrap()).unwrap();
    let node1 = arbiter.try_acquire(CpuSet::new([2, 3]).unwrap()).unwrap();
    assert!(arbiter.try_acquire(CpuSet::new([0, 1, 2, 3]).unwrap()).is_none());
    drop((node0, node1));
    assert!(arbiter.try_acquire(CpuSet::new([0, 1, 2, 3]).unwrap()).is_some());
}

#[test]
fn panic_releases_provider_exclusive_reservation() {
    let arbiter = ResourceArbiter::new();
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let _permit = arbiter.acquire_provider_exclusive().unwrap();
        panic!("forced");
    }));
    assert!(arbiter.try_acquire_provider_exclusive().is_some());
}
```

- [ ] **Step 2: Confirm tests fail**

Run: `cargo test -p tenferro-cpu arbiter --release`

Expected: FAIL for missing arbiter.

- [ ] **Step 3: Implement fair stable arbitration**

Use one `Mutex<ArbiterState>` plus `Condvar`. Track monotonically increasing
request IDs, active CPU sets, and the provider-exclusive flag. Grant a request
only when it is the oldest compatible waiter; this stable ordering prevents
starvation and multi-resource deadlock. Permit `Drop` removes active state and
notifies all waiters. Poisoning maps to a typed backend failure, never unwraps.

- [ ] **Step 4: Run concurrency tests under repetition**

Run: `for i in $(seq 1 20); do cargo test -p tenferro-cpu arbiter --release || exit 1; done`

Expected: all 20 runs PASS without hangs.

- [ ] **Step 5: Commit arbitration**

```bash
git add crates/tenferro-cpu/src/{arbiter.rs,arbiter/tests.rs,lib.rs}
git commit -m "feat(cpu): arbitrate overlapping CPU domains"
```

## Task 5: Refactor `CpuBackend` into a cloneable placement handle

**Files:**

- Modify: `crates/tenferro-cpu/src/backend.rs`
- Modify: `crates/tenferro-cpu/src/backend/tests.rs`
- Modify: `crates/tenferro-cpu/src/exec_session.rs`
- Modify: `crates/tenferro-cpu/src/tests/cpu_tests/backend_misc.rs`
- Modify: `crates/tenferro-cpu/src/tests/cpu_tests/context.rs`

- [ ] **Step 1: Write failing handle-sharing and capability tests**

```rust
#[test]
fn clones_share_registry_arbiter_and_aggregate_resources() {
    let backend = CpuBackend::from_topology_for_test(two_node_topology(), CpuBackendKind::Faer).unwrap();
    let node0 = backend.for_placement(CpuPlacement::NumaNode(NumaNodeId::new(0))).unwrap();
    let clone = node0.clone();
    assert_eq!(node0.coordinator_id(), clone.coordinator_id());
    assert_eq!(node0.resolved_placement(), clone.resolved_placement());
}

#[test]
fn blas_rejects_explicit_placement_but_auto_is_exclusive() {
    let backend = CpuBackend::from_topology_for_test(two_node_topology(), CpuBackendKind::Blas).unwrap();
    assert!(backend.for_placement(CpuPlacement::AllAllowed).is_err());
    assert_eq!(backend.resolved_execution(), ResolvedCpuExecution::ProviderDefaultExclusive);
}
```

- [ ] **Step 2: Confirm tests fail**

Run: `cargo test -p tenferro-cpu backend::tests::clone placement --release`

Expected: FAIL because `CpuBackend` is not cloneable or placement-aware.

- [ ] **Step 3: Introduce shared state and migrate constructors**

Replace direct `ctx`/`buffers` ownership with
`Arc<CpuBackendState> { topology, engines, all_allowed: OnceLock<_>, arbiter,
kind, thread_budget, buffer_limit }` plus a per-handle requested/resolved
execution field. Add:

```rust
pub fn for_placement(&self, placement: CpuPlacement) -> Result<Self, CpuPlacementError>;
pub fn placement(&self) -> CpuPlacement;
pub fn resolved_placement(&self) -> Option<&ResolvedCpuPlacement>;
pub fn topology(&self) -> &CpuTopology;
pub fn supports_placement(&self, placement: CpuPlacement) -> bool;
```

Keep existing constructors and `kind()` behavior. Aggregate buffer stats and
clear/limit operations across initialized engines without holding more than one
engine lock at a time. `linalg_context()` resolves the current managed engine
and is available only while a session owns it; remove any helper that could
bypass placement arbitration.

- [ ] **Step 4: Route direct operations through one engine permit/session**

Make `with_backend_session_cached` resolve once, acquire one permit, lock one
engine resource owner, enter the selected pool, and construct `CpuExecSession`.
Provider-default execution acquires the exclusive reservation and uses the lazy
all-allowed engine for native delegated methods while provider methods execute
outside Rayon.

- [ ] **Step 5: Run CPU tests for faer and BLAS feature shapes**

Run:

```bash
cargo test -p tenferro-cpu --features cpu-faer --release
cargo test -p tenferro-cpu --no-default-features --features cpu-blas --release
cargo check -p tenferro-cpu --no-default-features --features cpu-faer,cpu-blas
```

Expected: PASS.

- [ ] **Step 6: Commit the backend handle migration**

```bash
git add crates/tenferro-cpu/src/{backend.rs,backend/tests.rs,exec_session.rs,tests/cpu_tests/backend_misc.rs,tests/cpu_tests/context.rs}
git commit -m "refactor(cpu): share NUMA engine coordination"
```

## Task 6: Keep one managed session across graph boundaries

**Files:**

- Modify: `crates/tenferro-tensor/src/backend.rs`
- Modify: `crates/tenferro-runtime/src/exec.rs`
- Modify: `crates/tenferro-runtime/src/segment.rs`
- Modify: `crates/tenferro-runtime/src/graph/executor/tests.rs`
- Modify: `crates/tenferro-cpu/src/exec_session.rs`
- Modify: `crates/tenferro-cpu/src/backend.rs`

- [ ] **Step 1: Add failing whole-graph instrumentation tests**

Create a counting backend/session test proving one session wraps a program with
ordinary backend operations plus supported Host and exec-session FFI boundaries:

```rust
#[test]
fn managed_graph_enters_one_session_across_supported_boundaries() {
    let (mut executor, counters) = counted_executor();
    let program = backend_host_exec_ffi_fixture();
    executor.run(&program).unwrap();
    assert_eq!(counters.session_entries.load(Ordering::SeqCst), 1);
    assert_eq!(counters.pool_entries.load(Ordering::SeqCst), 1);
}
```

Add a provider-mode test with a native/provider/native fixture and assert one
exclusive reservation, two native pool entries, and one provider call outside
Rayon.

- [ ] **Step 2: Confirm current multiple-session behavior fails**

Run: `cargo test -p tenferro-runtime session_entries provider_native_transition --release`

Expected: FAIL with session/pool counts greater than the contract.

- [ ] **Step 3: Extend the internal session surface**

Add the minimum hidden session capabilities needed for Host transfer and
reclamation. Keep the public user surface unchanged. Implement them for
`CpuExecSession`, CUDA, WebGPU, and default backend sessions without exposing a
user-constructed lifetime.

- [ ] **Step 4: Move the program/segment loop inside one session**

Refactor `eval_exec_ir_unsegmented_*` and `eval_exec_segmented_*` so supported
Host and FFI instructions dispatch through the live session. Fused segments use
the same session. Unsupported extension runtimes return a clear internal
capability error or use the documented provider-default transition; they must
not silently recreate a managed CPU session.

- [ ] **Step 5: Verify runtime and backend suites**

Run:

```bash
cargo test -p tenferro-runtime --release
cargo test -p tenferro-cpu --release
cargo test -p tenferro-ad --release
```

Expected: PASS.

- [ ] **Step 6: Commit session lifetime changes**

```bash
git add crates/tenferro-tensor/src/backend.rs crates/tenferro-runtime/src/{exec.rs,segment.rs,graph/executor/tests.rs} crates/tenferro-cpu/src/{exec_session.rs,backend.rs}
git commit -m "refactor(runtime): keep CPU graph sessions engine-scoped"
```

## Task 7: Add diagnostics and rendered documentation

**Files:**

- Create: `docs/guides/cpu-execution.md`
- Modify: `docs/getting-started/index.md`
- Modify: `docs/index.md`
- Create: `docs/design/cpu-backend-execution.md`
- Modify: `crates/tenferro-cpu/src/lib.rs`
- Modify: `crates/tenferro-cpu/src/backend.rs`
- Create: `docs/worklogs/2026-07-14-numa-cpu-execution.md`

- [ ] **Step 1: Add public diagnostic contract tests**

Test that formatted topology includes allowed CPUs, sparse node IDs, resolved
placement/provider-default mode, backend kind, and worker count without relying
on debug formatting as a machine-readable API.

- [ ] **Step 2: Implement typed diagnostics**

Expose an immutable `CpuExecutionInfo` snapshot from `CpuBackend` with public
documented fields/accessors for topology, requested/resolved placement,
backend kind, and worker count. Concrete linked provider identity remains an
optional diagnostic string, not a stable enum.

- [ ] **Step 3: Write the online guide and durable design document**

The guide must contain the agreed matrix and this mixed-provider flow:

```text
provider-default exclusive session
    native segment -> pinned AllAllowed Rayon engine
    BLAS call      -> outside Rayon, provider-owned workers
    native segment -> pinned AllAllowed Rayon engine
```

State explicitly that thread count is not CPU affinity, external providers have
no explicit placement support, and CPU affinity does not imply NUMA-local
allocation. Add a runnable faer placement example backed by a doctest or checked
tutorial source.

- [ ] **Step 4: Verify rendered docs inputs**

Run:

```bash
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
python3 scripts/check-doc-snippets.py --check
python3 scripts/check-api-consistency.py
```

Expected: all commands PASS.

- [ ] **Step 5: Commit docs and diagnostics**

```bash
git add crates/tenferro-cpu/src/{lib.rs,backend.rs} docs/guides/cpu-execution.md docs/getting-started/index.md docs/index.md docs/design/cpu-backend-execution.md docs/worklogs/2026-07-14-numa-cpu-execution.md
git commit -m "docs: explain NUMA CPU execution ownership"
```

## Task 8: Add opt-in NUMA benchmarks and complete local verification

**Files:**

- Create: `crates/tenferro-cpu/benches/numa_execution.rs`
- Modify: `crates/tenferro-cpu/Cargo.toml`
- Modify: `docs/worklogs/2026-07-14-numa-cpu-execution.md`

- [ ] **Step 1: Add benchmark metadata and fixture smoke test**

The benchmark must print process allowed CPUs, topology, requested/resolved
placement, backend kind, worker count, and matrix shape before measuring. It
must skip with an explanatory message when fewer than two usable nodes exist.

- [ ] **Step 2: Compile the benchmark and run a short smoke sample**

Run:

```bash
cargo bench -p tenferro-cpu --bench numa_execution --no-run
cargo bench -p tenferro-cpu --bench numa_execution -- --sample-size 10
```

Expected: compile succeeds; run either records the three configured cases or
reports that multi-NUMA hardware is unavailable without failing.

- [ ] **Step 3: Run the repository pre-push checklist**

Run exactly:

```bash
cargo fmt --all --check
cargo test --workspace --release
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
python3 scripts/repository-rules-review.py --base origin/main --head HEAD --output-json /tmp/repository-rules-review.json
```

Also run the exact clippy command from the current `.github/workflows` clippy
job. Fix every finding or document a verified residual hardware limitation in
the work log.

- [ ] **Step 4: Re-read rules and update the work log evidence**

Re-read `REPOSITORY_RULES.md`; compare every changed file to the CPU threading,
cache ownership, docs, test organization, and performance rules. Record exact
commands/results and any unavailable multi-NUMA live evidence.

- [ ] **Step 5: Commit benchmark and verification record**

```bash
git add crates/tenferro-cpu/benches/numa_execution.rs crates/tenferro-cpu/Cargo.toml docs/worklogs/2026-07-14-numa-cpu-execution.md
git commit -m "bench(cpu): measure NUMA execution modes"
```

## Task 9: Review, publish, and merge the PR

**Files:**

- Review: all files changed from `origin/main...HEAD`

- [ ] **Step 1: Run verification again on committed HEAD**

Repeat formatting, workspace release tests, coverage, docs, exact CI clippy,
and repository rules review on committed `HEAD`. Do not rely on pre-commit
worktree results.

- [ ] **Step 2: Request code review and address findings**

Use `superpowers:requesting-code-review`, inspect the complete diff against the
spec and Issue #1345, and fix verified findings with focused commits. Re-run
affected tests after each fix and the full gate after the last fix.

- [ ] **Step 3: Push and create the PR**

Push `codex/issue-1345-numa-design` and create the PR to `main` with
`gh pr create`. The body links Issue #1345, the design spec, durable design doc,
work log, provider-placement limitation, and exact verification evidence.

- [ ] **Step 4: Enable and verify auto-merge**

Run:

```bash
gh pr merge --auto --squash --delete-branch
gh pr view --json autoMergeRequest,mergeStateStatus,statusCheckRollup
```

Expected: auto-merge is enabled and required checks are present.

- [ ] **Step 5: Monitor CI and review state until merge**

Poll PR checks and unresolved review threads. Use `github:gh-fix-ci` for Actions
failures and `github:gh-address-comments` for actionable review feedback. Make
focused fixes, push, re-run local affected checks, and restore auto-merge if a
push disables it.

- [ ] **Step 6: Verify the merged state**

Run `gh pr view --json state,mergedAt,mergeCommit,url` and verify `state` is
`MERGED`, `mergedAt` is non-null, and the merge commit is reachable from
`origin/main`. Only then is the requested objective complete.
