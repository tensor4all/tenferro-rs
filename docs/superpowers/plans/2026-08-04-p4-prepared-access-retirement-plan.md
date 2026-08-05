# P4 Prepared Access and Retirement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (recommended) or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activate the five P4 obligations from #1560 with one private, validate-once prepared-access hierarchy and one proven/unproven retirement record, without starting P3/P5 or adding recovery/compatibility machinery.

**Architecture:** Extend the private P2 `BackendAllocation` boundary with borrowed byte-mapping hooks. Add private `CheckedLayout`/`CheckedRead`/`CheckedWrite` and enum-authoritative `PreparedRead`/`PreparedWrite` states under `storage/prepared.rs`; typed host guards convert the already checked mapping once and traverse contiguous or precomputed strided layouts. Add `storage/retirement.rs` with one consuming retirement record that drops resources exactly once after proven completion and intentionally retains the complete record when completion is unproven. Existing public tensor APIs remain unchanged until P3.

**Tech Stack:** Rust 2021, `TensorRank`/`TensorScalar`, `TensorLayout` checked arithmetic, private `BackendAllocation`, the existing trybuild borrow suite, nested Cargo library-proof integration tests, TOML ledger v2, Python contract/doc checkers.

---

### Task 1: Create failing P4 proof surfaces and ledger fixtures

**Files:**
- Create: `crates/tenferro-tensor/src/storage/tests/prepared_access.rs`
- Create: `crates/tenferro-tensor/src/storage/tests/retirement.rs`
- Modify: `crates/tenferro-tensor/src/storage/tests/mod.rs`
- Create: `crates/tenferro-tensor/tests/storage_borrow_contract.rs`
- Create: `crates/tenferro-tensor/tests/storage_prepared_validation.rs`
- Create: `crates/tenferro-tensor/tests/storage_provider_event_retirement.rs`
- Create: `crates/tenferro-tensor/tests/storage_traversal_resolution.rs`
- Create: `crates/tenferro-tensor/tests/storage_prepared_access.rs`
- Modify: `scripts/storage-ownership-contracts.toml`
- Modify: `scripts/test-storage-ownership-contracts-v2.py`

- [ ] **Step 1: Add one RED unit proof for each P4 acceptance property.**

Add these test names with assertions for the stated behavior; call the
not-yet-defined private APIs so the tests fail for missing symbols:

```rust
#[test]
fn checked_layout_rejects_invalid_dtype_and_out_of_span_before_mapping() {}
#[test]
fn prepared_contiguous_read_and_write_use_typed_slices() {}
#[test]
fn prepared_strided_iterators_cover_reverse_and_empty_layouts() {}
#[test]
fn provider_resolution_counts_do_not_depend_on_element_count() {}
#[test]
fn proven_retirement_releases_binding_root_and_context_once() {}
#[test]
fn unproven_retirement_keeps_binding_root_and_context_alive() {}
#[test]
fn pre_admission_rejection_returns_the_unchanged_prepared_package() {}
```

Do not add production code in this step.

- [ ] **Step 2: Add five integration wrappers bound to real private proofs.**

Each wrapper must run Cargo from `env!("CARGO_MANIFEST_DIR")`, pass the exact
library-test filter for its private module, and assert `status.success()`:

```rust
let cargo = std::env::var_os("CARGO").expect("CARGO is set by Cargo");
let status = std::process::Command::new(cargo)
    .args(["test", "-p", "tenferro-tensor", "--lib", "storage::tests::prepared_access", "--quiet"])
    .current_dir(env!("CARGO_MANIFEST_DIR"))
    .status()
    .expect("nested private proof starts");
assert!(status.success());
```

Use the corresponding private filter for each artifact. These wrappers are
test-only bindings, not a production runner mode.

- [ ] **Step 3: Keep all five P4 rows deferred while RED.**

Extend the v2 checker tests to require the existing five P4 obligation IDs and
their new tracked paths, but do not change their state until Task 6.

Run and observe the intended failure:

```bash
cargo test -p tenferro-tensor --test storage_prepared_access --quiet
```

- [ ] **Step 4: Commit the RED proof surfaces.**

```bash
git add crates/tenferro-tensor/src/storage/tests crates/tenferro-tensor/tests scripts/storage-ownership-contracts.toml scripts/test-storage-ownership-contracts-v2.py
git commit -m "test(storage): add p4 prepared access red proofs"
```

### Task 2: Add provider byte mappings at the private root boundary

**Files:**
- Create: `crates/tenferro-tensor/src/storage/prepared.rs`
- Modify: `crates/tenferro-tensor/src/storage/mod.rs`
- Modify: `crates/tenferro-tensor/src/storage/root.rs`
- Modify: `crates/tenferro-tensor/src/storage/tests/root_claims.rs`

- [ ] **Step 1: Define borrowed mapping guards and typed errors.**

Add private `AccessError` variants for invalid layout, dtype/length/alignment,
unsupported mapping, provider mapping failure, and completion-unproven.
`ProviderReadMapping<'a>` and `ProviderWriteMapping<'a>` are boxed borrowed
guard objects exposing only `bytes()` and `bytes_mut()`. Their constructors
retain the provider guard until the prepared access is dropped.

- [ ] **Step 2: Extend the private unsafe trait with exactly one-time hooks.**

Add these private methods, with rustdoc explaining that the caller has already
validated the span and dtype:

```rust
fn map_read(&self, span: RootBoundSpan, dtype: DType)
    -> Result<ProviderReadMapping<'_>, AccessError>;
fn map_write(&self, span: RootBoundSpan, dtype: DType)
    -> Result<ProviderWriteMapping<'_>, AccessError>;
```

Update the test `CountingAllocation` with a mutex-backed byte vector and
atomic mapping counters. Do not add Arc cloning or a runtime recovery path.

- [ ] **Step 3: Expose only a root-borrowed provider reference.**

Add private root helpers returning `&dyn BackendAllocation` through the existing
`RootResourcePin`. Preparation must not call `Arc::clone`, `Arc::get_mut`, or a
second root identity check.

- [ ] **Step 4: Run the focused RED test and commit.**

```bash
cargo test -p tenferro-tensor --lib storage::tests::root_claims --quiet
git add crates/tenferro-tensor/src/storage
git commit -m "feat(storage): add p4 borrowed provider mappings"
```

### Task 3: Implement checked descriptors and enum-authoritative preparation

**Files:**
- Modify: `crates/tenferro-tensor/src/storage/prepared.rs`
- Modify: `crates/tenferro-tensor/src/storage/span.rs`
- Modify: `crates/tenferro-tensor/src/storage/identity.rs`

- [ ] **Step 1: Define rank-preserving checked state.**

Implement private `CheckedStrided<R>`, `CheckedLayout<R>`,
`WriteInjectivityProof`, `CheckedDescriptor<R>`,
`CheckedInjectiveDescriptor<R>`, `CheckedRead<'a, R>`, and
`CheckedWrite<'a, R>`. Store `R::Shape` and `R::Strides`, not an erased rank.

- [ ] **Step 2: Implement one checked constructor.**

Use one path to validate rank/shape/stride/offset arithmetic, dtype size and
alignment, reachable byte range within the exact `RootBoundSpan`, and mutable
injectivity. Return typed failure before provider mapping. Do not add a
post-construction corruption hook or repeated validation helper.

- [ ] **Step 3: Implement `PreparedRead`/`PreparedWrite`.**

Use nested `Host(Contiguous|Strided)` and `Device` enum variants. Preparation
consumes checked state, calls the borrowed root mapping hook once, and publishes
one variant. Device variants retain checked state plus opaque provider state but
no host guard/pointer/iterator. Mapping failure returns the unchanged checked
pairing.

- [ ] **Step 4: Verify and commit.**

```bash
cargo test -p tenferro-tensor --lib storage::tests::prepared_access::checked_layout_rejects_invalid_dtype_and_out_of_span_before_mapping --quiet
cargo test -p tenferro-tensor --test storage_prepared_validation --quiet
git add crates/tenferro-tensor/src/storage/prepared.rs crates/tenferro-tensor/src/storage/span.rs crates/tenferro-tensor/src/storage/identity.rs
git commit -m "feat(storage): add p4 checked prepared states"
```

### Task 4: Implement typed contiguous and incremental strided traversal

**Files:**
- Modify: `crates/tenferro-tensor/src/storage/prepared.rs`
- Modify: `crates/tenferro-tensor/src/storage/tests/prepared_access.rs`

- [ ] **Step 1: Add contiguous accessors.**

Implement `as_slice`/`iter_contiguous` and mutable counterparts. Extract the
checked range once; use standard typed slice iterators in the loop.

- [ ] **Step 2: Add the precomputed strided cursor.**

Implement `StrideCursor<R>` and immutable/mutable iterators. `next()` checks
exhaustion, accesses the proven typed offset, decrements the count, and updates
the carry. Keep unsafe conversion adjacent to `// SAFETY:`/`// INVARIANT:`
comments naming the constructor's bounds and injectivity proof.

- [ ] **Step 3: Reuse the existing public borrow fixtures.**

The prepared types are private to the storage module, so an external trybuild
crate cannot name them without adding a public test-only surface. The existing
`storage_compile_contract` suite continues to prove public view borrow
restrictions, while the private `CheckedRead`/`CheckedWrite` signatures and
their nested Cargo proof wrappers prove the P4 owner-borrow lifetime directly.
Do not duplicate the same restrictions as external fixtures:

```bash
cargo test -p tenferro-tensor --test storage_compile_contract --quiet
cargo test -p tenferro-tensor --test storage_prepared_access --quiet
```

Assert values for empty, singleton, reverse, and noncontiguous layouts. Commit
the prepared module and private unit proofs.

### Task 5: Implement one-shot retirement and pre-admission recovery

**Files:**
- Create: `crates/tenferro-tensor/src/storage/retirement.rs`
- Modify: `crates/tenferro-tensor/src/storage/mod.rs`
- Modify: `crates/tenferro-tensor/src/storage/tests/retirement.rs`

- [ ] **Step 1: Define one consuming retirement record.**

Use private event, binding, and provider-context traits and this shape:

```rust
pub(crate) enum RetirementOutcome {
    Completed,
    Failed(RetirementError),
    CompletionUnproven(RetirementError),
}

pub(crate) struct RetirementRecord {
    event: Box<dyn ProviderEvent>,
    bindings: Box<[Box<dyn ProviderRetirementBinding>]>,
    roots: Box<[RootResourcePin]>,
    provider: Box<dyn ProviderContext>,
}
```

`finish(self)` releases all fields exactly once on proven completion/failure.
On unproven completion it permanently retains the complete private record and
returns diagnostics without an owner. No retry, extraction, quarantine, or
boolean lifecycle table is added.

- [ ] **Step 2: Define exact pre-admission recovery.**

`PreparedPackage::admit` consumes a package and returns
`Err((Self, AdmissionError))` when rejection is known before enqueue. Once
admission may have happened, only the retirement record owns resources.

- [ ] **Step 3: Run lifecycle proofs and commit.**

Use fake binding/root/context drop counters to prove exactly-one release after
proven completion and zero release after unproven completion. Also prove user
handle drop only detaches observation:

```bash
cargo test -p tenferro-tensor --test storage_provider_event_retirement --quiet
cargo test -p tenferro-tensor --lib storage::tests::retirement --quiet
git add crates/tenferro-tensor/src/storage/retirement.rs crates/tenferro-tensor/src/storage/tests/retirement.rs crates/tenferro-tensor/src/storage/mod.rs
git commit -m "feat(storage): add p4 retirement record"
```

### Task 6: Prove constant resolution counts and activate P4 rows

**Files:**
- Modify: `crates/tenferro-tensor/src/storage/tests/prepared_access.rs`
- Modify: `crates/tenferro-tensor/src/storage/tests/retirement.rs`
- Modify: `scripts/storage-ownership-contracts.toml`
- Modify: `scripts/test-storage-ownership-contracts-v2.py`
- Modify: `docs/design/storage-ownership-contracts.md`
- Create: `docs/worklogs/2026-08-04-issue-1560-p4-prepared-access-retirement.md`

- [ ] **Step 1: Prove element-count independence.**

Prepare one contiguous and one strided access over the fake provider at one and
4096 elements. Assert mapping/provider-dispatch counters are equal and add a
source contract that iterator `next()` contains no provider/storage/bounds/
dtype/mapping calls.

- [ ] **Step 2: Update design and worklog.**

Record exact private type mapping, provider boundary, intentional unproven
retention, selected commands, and that P3/P5/P9 remain deferred. Do not rewrite
future-phase requirements into P4.

- [ ] **Step 3: Promote all five P4 rows together.**

Change only each P4 tagged state from deferred to active. Preserve obligation,
artifact, command, graph, and cohort values byte-for-byte. Extend checker tests
so partial P4 activation fails and every active artifact is tracked.

- [ ] **Step 4: Run the complete GREEN gate.**

```bash
cargo fmt --all --check
git diff --check
cargo test -p tenferro-tensor --all-targets --quiet
cargo clippy -p tenferro-tensor --all-targets -- -D warnings
cargo llvm-cov -p tenferro-tensor --lib --summary-only
python3 scripts/test-storage-ownership-contracts-v2.py
python3 scripts/check-storage-ownership-contracts.py
python3 scripts/check-storage-design-docs.py
```

Expected: all P2/P4 tests pass, new storage files exceed 90% line coverage,
and the manifest remains nonterminal because P3/P5/P6+ are deferred.

- [ ] **Step 5: Commit activation and evidence.**

```bash
git add crates/tenferro-tensor/src/storage/tests scripts/storage-ownership-contracts.toml scripts/test-storage-ownership-contracts-v2.py docs/design/storage-ownership-contracts.md docs/worklogs/2026-08-04-issue-1560-p4-prepared-access-retirement.md
git commit -m "feat(storage): activate p4 prepared access retirement"
```

### Task 7: Exact-head review checkpoint

**Files:**
- Read: `REPOSITORY_RULES.md`
- Read: `docs/design/storage-ownership-contracts.md`
- Read: `docs/worklogs/2026-08-04-issue-1560-p4-prepared-access-retirement.md`

- [ ] **Step 1: Run exact-head rules and cleanliness checks.**

```bash
python3 scripts/repository-rules-review.py --base origin/main --head "$(git rev-parse HEAD)" --output-json /tmp/p4-rules-review.json
git status --short --branch
```

The JSON must report `verdict: pass`, zero block findings, and a clean tracked
tree. Fix only concrete findings and rerun exact-head checks.

- [ ] **Step 2: Obtain independent specification and quality approvals.**

Both reviewers inspect the exact final HEAD and confirm only P4 is active, all
five artifacts execute real private proofs, provider work is element-count
independent, and no prohibited machinery was introduced.

- [ ] **Step 3: Stop before P5.**

Mark P4 complete in the working plan and report its exact candidate commit. Do
not activate or implement P5 until the next phase is explicitly selected.
