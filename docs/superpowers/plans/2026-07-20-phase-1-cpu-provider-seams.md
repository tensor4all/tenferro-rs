# Phase 1 CPU Provider Seams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route CPU eager and graph contractions through validated borrowed requests and immutable pluggable GEMM, layout, and general-contraction providers without adding a session entry, allocation, or hot-path lookup.

**Architecture:** `CpuBackend` owns one immutable `CpuProviderBundle`; `CpuExecSession` borrows its engine-owned `DotGeneralRuntime`. The runtime validates once, allocates outputs and temporaries from the existing session pool, and passes write-into requests to direct trait-object slots. Typed `Unsupported` is the only continuation signal; execution errors are terminal.

**Tech Stack:** Rust 2021, `tenferro-cpu`, `tenferro-tensor`, Criterion 0.5, faer, optional BLAS/TBLIS, Rayon, `SmallVec`, repository fast gate.

---

### Task 1: Carry accepted baseline assets and freeze the indexed case

**Files:**

- Modify: `REPOSITORY_RULES.md`
- Modify: `crates/tenferro-ad/Cargo.toml`
- Create: `crates/tenferro-ad/benches/eager_dispatch_baseline.rs`
- Create: `docs/design/execution-engine-provider-architecture.md`
- Create: `docs/worklogs/2026-07-20-eager-main-baseline.md`

- [ ] **Step 1: Restore accepted evidence without PoC source**

Run `git cherry-pick 89c188d0`. Use `apply_patch` to copy only the
`Performance-Gated Experiment Protocol` from `23d152f3:REPOSITORY_RULES.md`
and the accepted architecture into the new design file. Do not restore
`arbiter.rs`, arbiter tests, or ResourceArbiter PoC documents.

- [ ] **Step 2: Add `slice_f64` before candidate code exists**

Import `SliceConfig`. Add this benchmark to both length loops, using
`consume_lazy` and `consume_materialized` respectively:

```rust
let slice = SliceConfig {
    starts: vec![0],
    limits: vec![len],
    strides: vec![1],
};
group.bench_with_input(BenchmarkId::new("slice_f64", len), &len, |b, _| {
    b.iter(|| {
        consume_lazy(
            black_box(&lhs)
                .slice(black_box(slice.clone()))
                .expect("slice should succeed"),
        )
    });
});
```

- [ ] **Step 3: Verify and record the protocol**

Run:

```bash
cargo bench -p tenferro-ad --bench eager_dispatch_baseline --no-run
cargo test -p tenferro-ad --test integration eager_untracked_slice_returns_lazy_view
```

Append the Phase 1 case matrix, three-pair order, affinity/environment controls,
and interval classification from the Phase 1 spec to the baseline worklog.
Record baseline SHA `85855e272b1495611deb601a9ee06f3546772c3c`.

- [ ] **Step 4: Commit the evidence boundary**

```bash
git add REPOSITORY_RULES.md crates/tenferro-ad/Cargo.toml \
  crates/tenferro-ad/benches/eager_dispatch_baseline.rs \
  docs/design/execution-engine-provider-architecture.md \
  docs/worklogs/2026-07-20-eager-main-baseline.md
git commit -m "Carry phase 1 execution baseline"
```

Expected: no ResourceArbiter implementation file is changed.

### Task 2: Define the object-safe provider SPI using TDD

**Files:**

- Create: `crates/tenferro-cpu/src/provider.rs`
- Create: `crates/tenferro-cpu/src/provider/tests.rs`
- Modify: `crates/tenferro-cpu/src/lib.rs`
- Modify: `crates/tenferro-cpu/Cargo.toml`
- Create: `crates/tenferro-cpu/tests/provider_boundary_allocation_tests/main.rs`

- [ ] **Step 1: Write failing object-safety and typed-outcome tests**

```rust
fn assert_object_safe(
    gemm: &dyn CpuGemmProvider,
    layout: &dyn CpuLayoutTransformProvider,
    general: &dyn CpuGeneralContractionProvider,
) {
    let _ = (gemm, layout, general);
}

#[test]
fn unsupported_is_typed() {
    assert!(matches!(
        CpuProviderOutcome::Unsupported(CpuProviderUnsupported::RuntimeUnavailable),
        CpuProviderOutcome::Unsupported(CpuProviderUnsupported::RuntimeUnavailable),
    ));
}
```

Run `cargo test -p tenferro-cpu provider::tests::unsupported_is_typed`.
Expected: compile failure because the SPI is absent.

- [ ] **Step 2: Implement the public enums/context**

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CpuProviderUnsupported {
    DType(DType),
    Rank { lhs: usize, rhs: usize },
    Layout(CpuOperand),
    Conjugation,
    Accumulation,
    StridedBatch,
    Grouped,
    RuntimeUnavailable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuOperand { Lhs, Rhs, Output }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub enum CpuProviderOutcome {
    Executed,
    Unsupported(CpuProviderUnsupported),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuKernelParallelism { Sequential, Inner }
```

`CpuProviderContext` stores private `&CpuContext` and parallelism fields. Its
public methods return the validated nonzero thread budget as `usize` and
`CpuKernelParallelism`. Providers receive no pool/session/permit accessor and
library code introduces no `unwrap`/`expect` for this invariant.

- [ ] **Step 3: Implement exact borrowed request shapes and traits**

Create private-field request structs for GEMM, grouped GEMM, layout, and
dot-general. Each stores `&TensorRead`, a reborrowed `&mut TensorWrite`, and
only slices/scalars/descriptors. Use the signatures from the Phase 1 spec:

```rust
pub trait CpuGemmProvider: fmt::Debug + Send + Sync + 'static {
    fn gemm(&self, cx: &CpuProviderContext<'_>, req: CpuGemmRequest<'_, '_, '_>)
        -> crate::Result<CpuProviderOutcome>;
    fn strided_batched_gemm(
        &self, cx: &CpuProviderContext<'_>, req: CpuGemmRequest<'_, '_, '_>,
    ) -> crate::Result<CpuProviderOutcome>;
    fn grouped_gemm(
        &self, cx: &CpuProviderContext<'_>, req: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> crate::Result<CpuProviderOutcome>;
}
```

Add the two other traits exactly as specified. Constructors are `pub(crate)`;
public accessors borrow fields. Add compiling `/// # Examples` to every public
item.

- [ ] **Step 4: Add the warmed allocation probe**

Copy the counting-allocator harness pattern from `install_allocation_tests`.
After warm-up, construct and dispatch a crate-provided validated test request
10,000 times and assert:

```rust
assert_eq!(stats.allocations, 0);
assert_eq!(stats.bytes, 0);
```

Run:

```bash
cargo test -p tenferro-cpu provider::tests
cargo test -p tenferro-cpu --test provider_boundary_allocation_tests
cargo test -p tenferro-cpu --doc provider
```

- [ ] **Step 5: Commit**

```bash
git add crates/tenferro-cpu
git commit -m "Add borrowed CPU provider contracts"
```

### Task 3: Validate dot-general once and expose allocation-free axis groups

**Files:**

- Create: `crates/tenferro-cpu/src/dot_runtime.rs`
- Create: `crates/tenferro-cpu/src/dot_runtime/tests.rs`
- Modify: `crates/tenferro-cpu/src/lib.rs`

- [ ] **Step 1: Write failing parity/boundary tests**

Compare new validation with `DotGeneralConfig::validate_dims_with_ranks` for
ranks `0,1,2,8,63,64,65,70`, including duplicate, out-of-range, role-overlap,
pair-count, contracted-extent, and batch-extent failures. Assert:

```rust
let groups = validate_axis_groups(4, 4, &config)?;
assert_eq!(groups.contracting_pairs().collect::<Vec<_>>(), vec![(1, 0)]);
assert_eq!(groups.batch_pairs().collect::<Vec<_>>(), vec![(2, 2)]);
assert_eq!(groups.lhs_free_axes().collect::<Vec<_>>(), vec![0, 3]);
assert_eq!(groups.rhs_free_axes().collect::<Vec<_>>(), vec![1, 3]);
```

Run `cargo test -p tenferro-cpu dot_runtime::tests::axis_groups`.
Expected: compile failure.

- [ ] **Step 2: Implement mask/linear validation**

```rust
fn role_mask(axes: &[usize], rank: usize) -> crate::Result<Option<u64>> {
    if rank > 64 {
        for (position, &axis) in axes.iter().enumerate() {
            if axis >= rank {
                return Err(crate::Error::axis_out_of_bounds("dot_general", axis, rank));
            }
            if axes[..position].contains(&axis) {
                return Err(crate::Error::duplicate_axis("dot_general", axis, "axis role"));
            }
        }
        return Ok(None);
    }
    let mut mask = 0_u64;
    for &axis in axes {
        if axis >= rank {
            return Err(crate::Error::axis_out_of_bounds("dot_general", axis, rank));
        }
        let bit = 1_u64 << axis;
        if mask & bit != 0 {
            return Err(crate::Error::duplicate_axis("dot_general", axis, "axis role"));
        }
        mask |= bit;
    }
    Ok(Some(mask))
}
```

Keep ordered config slices authoritative. Implement free-axis iterators as
range filters over masks or slice membership. Validate dtype/placement,
extents, output metadata, checked products/ranges, and accumulation before
constructing `ValidatedDotGeneral`.

- [ ] **Step 3: Verify and commit**

```bash
cargo test -p tenferro-cpu dot_runtime::tests
cargo test -p tenferro-cpu --test integration runtime_error_tests::dot_general
git add crates/tenferro-cpu/src/dot_runtime.rs \
  crates/tenferro-cpu/src/dot_runtime/tests.rs crates/tenferro-cpu/src/lib.rs
git commit -m "Validate borrowed CPU contraction requests"
```

### Task 4: Add immutable bundles and ABA-safe cache identity

**Files:**

- Modify: `crates/tenferro-cpu/src/dot_runtime.rs`
- Modify: `crates/tenferro-cpu/src/dot_runtime/tests.rs`
- Modify: `crates/tenferro-cpu/src/gemm/mod.rs`
- Modify: `crates/tenferro-cpu/src/gemm/tests.rs`
- Modify: `crates/tenferro-cpu/src/backend.rs`
- Modify: `crates/tenferro-cpu/src/backend/tests.rs`

- [ ] **Step 1: Write failing identity tests**

Assert one bundle reuses a cached slot, a distinct bundle clears it, and a
bundle used after the first bundle's last strong reference is dropped clears
it. Run `cargo test -p tenferro-cpu gemm::tests::provider_bundle_identity`.
Expected: compile failure.

- [ ] **Step 2: Bind the cache to `Weak<CpuProviderBundleInner>`**

```rust
pub(crate) fn bind_provider_bundle(&mut self, bundle: &Arc<CpuProviderBundleInner>) {
    let matches = self.provider_bundle
        .as_ref()
        .and_then(Weak::upgrade)
        .is_some_and(|current| Arc::ptr_eq(&current, bundle));
    if !matches {
        self.slots.clear();
        self.provider_bundle = Some(Arc::downgrade(bundle));
    }
}
```

Initialize the weak field in every constructor and count it in logical retained
bytes. `clear()` removes entries but retains the current binding.

- [ ] **Step 3: Implement bundle construction**

Add `CpuProviderBundle`, `CpuProviderBundleBuilder`, and
`GeneralContractionPolicy::{Preferred,Required}`. `build()` rejects missing
GEMM/layout slots. Add consuming `CpuBackend::with_provider_bundle`; clones
share one bundle identity.

- [ ] **Step 4: Verify and commit**

```bash
cargo test -p tenferro-cpu gemm::tests::provider_bundle_identity
cargo test -p tenferro-cpu backend::tests::provider_bundle
git add crates/tenferro-cpu/src/backend.rs crates/tenferro-cpu/src/backend/tests.rs \
  crates/tenferro-cpu/src/dot_runtime.rs crates/tenferro-cpu/src/dot_runtime/tests.rs \
  crates/tenferro-cpu/src/gemm/mod.rs crates/tenferro-cpu/src/gemm/tests.rs
git commit -m "Bind CPU contraction caches to provider bundles"
```

### Task 5: Adapt built-in layout, faer, BLAS, and TBLIS providers

**Files:**

- Modify: `crates/tenferro-cpu/src/provider.rs`
- Modify: `crates/tenferro-cpu/src/provider/tests.rs`
- Modify: `crates/tenferro-cpu/src/gemm/mod.rs`
- Modify: `crates/tenferro-cpu/src/gemm/faer_gemm.rs`
- Modify: `crates/tenferro-cpu/src/gemm/blas_gemm.rs`
- Modify: `crates/tenferro-cpu/src/gemm/tblis_gemm.rs`
- Modify: `crates/tenferro-cpu/src/structural.rs`
- Modify: `crates/tenferro-tensor/src/backend.rs`

- [ ] **Step 1: Write failing direct-adapter tests**

Execute each adapter into preallocated outputs. Cover f32/f64/c32/c64,
conjugation, non-unit strides, strided batches, grouped jobs, and unsupported
dtype/layout. Every `Unsupported` test snapshots output before and after.

- [ ] **Step 2: Implement adapters without resource ownership**

`StridedLayoutTransformProvider` delegates to existing strided copy under the
runtime-supplied native scope. `FaerGemmProvider` and feature-gated
`BlasGemmProvider` convert validated descriptors to existing raw-stride
kernels. Capability misses occur before output mutation; kernel errors remain
`Err`. `TblisGeneralContractionProvider` builds labels from the four role-group
iterators and returns typed `Unsupported` only before its FFI call.

Document `GroupedGemmJob` as a public provider descriptor only if its type is
exposed by a public accessor; keep `GroupedGemmConfig` hidden.

- [ ] **Step 3: Verify feature combinations**

```bash
cargo test -p tenferro-cpu provider::tests
cargo test -p tenferro-cpu --no-default-features --features cpu-faer provider::tests
cargo check -p tenferro-cpu --no-default-features --features cpu-blas,blas-openblas
cargo check -p tenferro-cpu --features cpu-tblis
```

- [ ] **Step 4: Commit**

```bash
git add crates/tenferro-cpu/src/provider.rs crates/tenferro-cpu/src/provider \
  crates/tenferro-cpu/src/gemm crates/tenferro-cpu/src/structural.rs \
  crates/tenferro-tensor/src/backend.rs
git commit -m "Adapt CPU kernels to provider requests"
```

### Task 6: Implement engine-owned routing and preserve single fan-out

**Files:**

- Modify: `crates/tenferro-cpu/src/dot_runtime.rs`
- Modify: `crates/tenferro-cpu/src/dot_runtime/tests.rs`
- Modify: `crates/tenferro-cpu/src/engine.rs`
- Modify: `crates/tenferro-cpu/src/engine/tests.rs`

- [ ] **Step 1: Write failing route-counter tests**

Test general `Executed`, general `Unsupported` to preferred fallback, general
`Err` terminal, required-general `Unsupported`, GEMM `Unsupported` terminal,
grouped outer calls observing `Sequential`, and strided batch observing
`Inner`. Run `cargo test -p tenferro-cpu dot_runtime::tests::route` and confirm
failure.

- [ ] **Step 2: Implement the explicit state machine**

```rust
if let Some(general) = &self.general {
    let request = validated.general_request(lhs, rhs, out, accumulation);
    match general.dot_general(&provider_context, request)? {
        CpuProviderOutcome::Executed => return Ok(()),
        CpuProviderOutcome::Unsupported(reason) => {
            if self.general_policy == GeneralContractionPolicy::Required {
                return Err(required_provider_unsupported("dot_general", reason));
            }
        }
    }
}
self.execute_layout_plus_gemm(session, cache_slot, validated, lhs, rhs, out, accumulation)
```

Allocate outputs/temporaries from the existing session pool before constructing
write requests. Bind cache identity before analysis. Never catch `Err`.

- [ ] **Step 3: Move scheduling above adapters**

Multiple grouped jobs with multiple threads: one engine `ctx.install`, outer
Rayon fan-out, sequential single-job provider calls. One job or one thread:
sequential outer loop with inner provider permission. Strided batch: sequential
outer loop with inner permission. Providers never call `install`.

- [ ] **Step 4: Verify and commit**

```bash
cargo test -p tenferro-cpu dot_runtime::tests
cargo test -p tenferro-cpu tests::grouped_gemm
cargo test -p tenferro-cpu tests::dot_structural_analytic::test_dot_general
git add crates/tenferro-cpu/src/dot_runtime.rs \
  crates/tenferro-cpu/src/dot_runtime/tests.rs \
  crates/tenferro-cpu/src/engine.rs crates/tenferro-cpu/src/engine/tests.rs
git commit -m "Compose CPU dot general providers in the engine"
```

### Task 7: Route backend, eager, and graph sessions through the bundle

**Files:**

- Modify: `crates/tenferro-cpu/src/backend.rs`
- Modify: `crates/tenferro-cpu/src/backend/tests.rs`
- Modify: `crates/tenferro-cpu/src/exec_session.rs`
- Modify: `crates/tenferro-cpu/src/tests/cpu_tests/context.rs`
- Modify: `crates/tenferro-cpu/src/tests/cpu_tests/dot_structural_analytic.rs`
- Modify: `crates/tenferro-runtime/src/graph/executor/tests/preflight.rs`

- [ ] **Step 1: Write failing parity tests**

Install one spy bundle and execute direct backend dot, cached backend session
dot, and compiled graph dot. Assert the same slot is called once per operation
and no second backend/session is constructed.

- [ ] **Step 2: Pass the bundle into `CpuExecSession` once**

Every `TensorDot` and `SessionCachedDot` method delegates to
`DotGeneralRuntime`. Keep `CpuBackendKind` only for un-migrated operation
families. Remove `with_base_dot_general_provider`, TBLIS applicability helpers,
all contraction `match kind` blocks, and mutable `set_dot_general_provider`.
`DotGeneralProvider` may remain only as construction-time standard-bundle
mapping; it must not appear in session state or execution matches.

- [ ] **Step 3: Verify and commit**

```bash
cargo test -p tenferro-cpu
cargo test -p tenferro-runtime graph::executor
cargo test -p tenferro-ad --test integration eager_tensor
git add crates/tenferro-cpu/src crates/tenferro-cpu/tests \
  crates/tenferro-runtime/src/graph/executor/tests/preflight.rs
git commit -m "Route CPU contractions through provider bundles"
```

### Task 8: Enforce no lookup/allocation and remove staging

**Files:**

- Modify: `crates/tenferro-cpu/tests/integration/backend_capability_contracts.rs`
- Modify: `crates/tenferro-cpu/tests/provider_boundary_allocation_tests/main.rs`

- [ ] **Step 1: Add source-contract tests**

Scan steady-state dispatch modules and reject:

```rust
for forbidden in ["HashMap", "TypeId", "dyn Any", "downcast", "provider_name.to_string"] {
    assert!(!dispatch_source.contains(forbidden), "forbidden hot-path token: {forbidden}");
}
```

Also reject `with_base_dot_general_provider` and
`match self.dot_general_provider`.

- [ ] **Step 2: Compare warmed allocation counts**

Measure request dispatch plus tiny eager elementwise, reduction, slice, and dot
cases. Setup stays outside counted loops. Assert candidate allocation count and
bytes do not exceed the baseline for every case.

- [ ] **Step 3: Verify and commit**

```bash
cargo test -p tenferro-cpu --test integration backend_capability_contracts
cargo test -p tenferro-cpu --test provider_boundary_allocation_tests
rg -n 'with_base_dot_general_provider|match self\.dot_general_provider' \
  crates/tenferro-cpu/src/backend.rs crates/tenferro-cpu/src/exec_session.rs
```

Expected: tests pass and `rg` has no matches.

```bash
git add crates/tenferro-cpu/tests
git commit -m "Enforce direct CPU provider dispatch contracts"
```

### Task 9: Run the fixed three-pair non-inferiority campaign

**Files:**

- Create: `docs/worklogs/2026-07-20-phase-1-cpu-provider-seams.md`

- [ ] **Step 1: Record immutable environment**

```bash
git rev-parse origin/main HEAD
rustc -Vv
cargo -V
lscpu
taskset -pc $$
uptime
pgrep -af 'cargo|rustc' || true
```

Select the first allowed CPU once. Set all of `RAYON_NUM_THREADS`,
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and
`VECLIB_MAXIMUM_THREADS` to one.

- [ ] **Step 2: Build and run both immutable binaries**

Build baseline and candidate benchmark binaries in separate temporary
worktrees with one toolchain/profile/feature set. Run complete order `A/B`,
`B/A`, `A/B`, pinned to the chosen CPU. Record affinity, normalized one-minute
load, and active Cargo/rustc processes at both endpoints. Preserve every
Criterion `estimates.json` and `change/estimates.json`.

- [ ] **Step 3: Classify without changing gates**

```text
PASS         all three 95% relative-change upper endpoints <= +5%
FAIL         at least two lower endpoints > +5%
INCONCLUSIVE every other valid interval pattern, or any invalid pair
```

The campaign passes only if every case passes. `FAIL` blocks promotion.
`INCONCLUSIVE` permits only a complete unchanged three-pair rerun.

- [ ] **Step 4: Write and commit the worklog**

Record all cases, intervals, validity observations, correctness/allocation
commands, decisions, and residual risks.

```bash
git add docs/worklogs/2026-07-20-phase-1-cpu-provider-seams.md
git commit -m "Record phase 1 provider verification"
```

### Task 10: Update docs, run final gates, push, and report

**Files:**

- Modify: `docs/design/execution-engine-provider-architecture.md`
- Modify: `docs/guides/cpu-execution.md`
- Modify: `docs/guides/parallelism-and-caching.md`
- Modify: `docs/performance/cpu-benchmarks.md`
- Modify: `docs/superpowers/specs/2026-07-20-phase-1-cpu-provider-seams-design.md`
- Modify: `docs/worklogs/2026-07-20-phase-1-cpu-provider-seams.md`

- [ ] **Step 1: Document landed contracts and examples**

Add runnable default/custom bundle examples. State engine-owned allocation,
session entry, outer fan-out, and fallback; provider write-into behavior; and
current grouped/strided/TBLIS policies. Mark only Phase 1 complete.

- [ ] **Step 2: Run full focused verification**

```bash
cargo fmt --all
cargo test -p tenferro-cpu
cargo test -p tenferro-runtime graph::executor
cargo test -p tenferro-ad --test integration eager_tensor
python3 scripts/check-doc-snippets.py --check
python3 scripts/test-doc-consistency.py
python3 scripts/test-check-docs-site.py
git diff --check
```

- [ ] **Step 3: Commit docs and run policy gates on committed HEAD**

```bash
git add docs
git commit -m "Document pluggable CPU contraction providers"
bash scripts/check-pr-fast.sh --coverage-reviewed \
  --test 'cargo test -p tenferro-cpu provider::tests'
python3 scripts/repository-rules-review.py --base origin/main --head HEAD \
  --output-json /tmp/phase1-repository-rules-review.json
```

Expected: all commands pass and rules review has zero unwaived findings.

- [ ] **Step 4: Push and update issues**

```bash
git push -u origin codex/execution-engine-phase1
```

Post #1434 evidence with branch, commit range, test counts, allocations, every
benchmark interval/classification, worklog, and risks. Update #1433 Phase 1
status only on campaign `PASS`; do not mark complete on `FAIL` or
`INCONCLUSIVE`.
