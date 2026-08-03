# P5 AllocationGroup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task with verification checkpoints.

**Goal:** Activate P5 by adding one private `AllocationGroup` representation with local descriptor slots, N-way disjoint mutable children, and structural copy-free extraction.

**Architecture:** `storage/group.rs` owns the group tables and the only multi-child proof boundary. It stores non-owning rank-erased descriptor metadata beside move-only P4 `OwnedStorage` entries. All child lifetimes are bounded by the exclusive group borrow; provider mapping and asynchronous retirement remain P4 responsibilities.

**Tech Stack:** Rust private storage module, `TensorLayout` checked arithmetic, existing P4 root/prepared types, crate-private unit tests, integration artifact wrapper, trybuild/public borrow contract where applicable, TOML ledger and Python contract checkers.

---

## File map

- Create `crates/tenferro-tensor/src/storage/group.rs`: group tables, descriptor metadata, typed errors, slot resolution, central disjointness proof, and child wrappers.
- Modify `crates/tenferro-tensor/src/storage/mod.rs`: register the private group module and test-only re-exports.
- Modify `crates/tenferro-tensor/src/storage/tests/mod.rs`: register the group proof tests.
- Create `crates/tenferro-tensor/src/storage/tests/group.rs`: private RED/GREEN tests using fake byte allocations and retained counters.
- Create `crates/tenferro-tensor/tests/storage_allocation_group.rs`: canonical P5 artifact wrapper invoking `storage::tests::group`.
- Modify `scripts/storage-ownership-contracts.toml`: promote only `p5-allocation-group` to `state = { kind = "active" }`.
- Modify `scripts/test-storage-ownership-contracts-v2.py`: add only `p5-allocation-group` to `ACTIVE_IDS` and keep all later rows deferred.
- Modify `docs/design/storage-ownership-contracts.md`: record the implemented G2 representation, proof order, lifetime/extraction behavior, and active P5 artifact.
- Create `docs/worklogs/2026-08-04-issue-1561-p5-allocation-group.md`: exact candidate, scope, tests, and residual limits.
- If the existing UI harness can express the private borrow contract without exposing new public API, add one focused fixture under `crates/tenferro-tensor/tests/ui/storage/`; otherwise record the private-harness lifetime proof in the artifact and do not create a misleading public fixture.

## Task 1: Add the failing P5 artifact and minimal private test names

**Files:**
- Create `crates/tenferro-tensor/tests/storage_allocation_group.rs`.
- Modify `crates/tenferro-tensor/src/storage/tests/mod.rs`.
- Create `crates/tenferro-tensor/src/storage/tests/group.rs`.

- [ ] **Step 1: Write the RED artifact wrapper.**

```rust
#[test]
fn p5_allocation_group_artifact_runs_real_proofs() {
    let cargo = std::env::var_os("CARGO").expect("cargo sets CARGO for tests");
    let status = std::process::Command::new(cargo)
        .args([
            "test", "-p", "tenferro-tensor", "--lib",
            "storage::tests::group", "--quiet",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .expect("launch private P5 allocation-group proof tests");
    assert!(status.success(), "private P5 allocation-group proof tests failed");
}
```

Register `mod group;` in `storage/tests/mod.rs`. In `group.rs`, add one
test that calls the not-yet-defined `AllocationGroup::new()` and names the
expected N-way behavior; keep the fake allocation helper local to this module.

- [ ] **Step 2: Run the artifact to verify the intended RED failure.**

Run:

```bash
cargo test -p tenferro-tensor --test storage_allocation_group --quiet
```

Expected: compilation fails because the private `group` module and
`AllocationGroup` implementation do not yet exist. Fix only test typos before
continuing; do not stub a passing group.

- [ ] **Step 3: Commit the RED test harness.**

```bash
git add crates/tenferro-tensor/tests/storage_allocation_group.rs \
  crates/tenferro-tensor/src/storage/tests/mod.rs \
  crates/tenferro-tensor/src/storage/tests/group.rs
git commit -m "test(storage): add p5 allocation group red harness"
```

## Task 2: Implement the group tables and construction-time metadata

**Files:**
- Create `crates/tenferro-tensor/src/storage/group.rs`.
- Modify `crates/tenferro-tensor/src/storage/mod.rs`.
- Modify `crates/tenferro-tensor/src/storage/tests/group.rs`.

- [ ] **Step 1: Add the minimal typed data model.**

Define `AllocationSlot`, `DescriptorSlot`, `AllocationGroup`,
`DescriptorRecord`, `DescriptorInput`, and typed `GroupError`. Store
`Option<OwnedStorage>` and append-only descriptor records. Add constructors
that consume an owner and validate its root-bound span, dtype size/alignment,
rank-erased shape/stride arithmetic, and storage/provider metadata once. Retain
the conservative reachable byte envelope and `Option<WriteInjectivityProof>`;
never store a provider `Arc` or a write-authorizing slot.

- [ ] **Step 2: Add RED tests for construction rejection and retained facts.**

Cover malformed dtype/byte length, out-of-root layout, alignment,
invalid allocation/descriptor slots, and a valid compact, empty, scalar, and
reverse descriptor. Assert failures leave the owner/group available and that
retained metadata is read without a second provider mapping call.

- [ ] **Step 3: Run focused tests and fix only implementation failures.**

```bash
cargo test -p tenferro-tensor --lib storage::tests::group --quiet
```

Expected: construction tests pass; view/split/extraction tests remain absent or
fail because those operations are not implemented.

- [ ] **Step 4: Commit construction.**

```bash
git add crates/tenferro-tensor/src/storage/group.rs \
  crates/tenferro-tensor/src/storage/mod.rs \
  crates/tenferro-tensor/src/storage/tests/group.rs
git commit -m "feat(storage): add p5 group descriptor tables"
```

## Task 3: Implement borrowed read/write resolution and structural extraction

**Files:**
- Modify `crates/tenferro-tensor/src/storage/group.rs`.
- Modify `crates/tenferro-tensor/src/storage/tests/group.rs`.

- [ ] **Step 1: Add RED tests for borrowed resolution.**

Test aliasing `view` calls, one `view_mut`, invalid/vacant slots, and that a
mutable child prevents group/root access until the child is dropped. Use the
existing P4 typed preparation only after local metadata resolution; do not map
or enqueue in slot lookup.

- [ ] **Step 2: Implement `view` and `view_mut`.**

Resolve only the local slot and occupancy, combine the retained descriptor
metadata with the requested typed borrow, and return non-`Clone` children
whose lifetime is tied to `&self` or `&mut self`. Keep P4 validation errors
distinct from group slot errors.

- [ ] **Step 3: Add RED tests for extraction.**

Assert `try_extract` succeeds only for the sole descriptor of an allocation,
vacates entries without renumbering, rejects aliased descriptors unchanged,
and `into_owner` returns the unchanged group on invalid selection while
explicitly discarding unselected owners on success. Check provider drop counts
exactly once.

- [ ] **Step 4: Implement `try_extract` and consuming `into_owner`.**

Count local descriptor references structurally, replace selected entries with
`None`, and move the existing `OwnedStorage`. Never copy bytes, create a
replacement owner, reuse a slot, or keep an extracted-state boolean.

- [ ] **Step 5: Run focused extraction/borrow tests and commit.**

```bash
cargo test -p tenferro-tensor --lib storage::tests::group --quiet
git add crates/tenferro-tensor/src/storage/group.rs \
  crates/tenferro-tensor/src/storage/tests/group.rs
git commit -m "feat(storage): add group borrow and extraction paths"
```

## Task 4: Implement the single central N-way disjointness proof

**Files:**
- Modify `crates/tenferro-tensor/src/storage/group.rs`.
- Modify `crates/tenferro-tensor/src/storage/tests/group.rs`.
- Optionally create one focused UI fixture only if it compiles against the
  actual crate-private proof harness without exposing new public API.

- [ ] **Step 1: Add RED cases for N-way proof behavior.**

Test N=0, N=1, N>2, permutation independence, duplicate slots,
empty/scalar and reverse-stride descriptors, missing injectivity proof,
positive overlap, conservative interleaved rejection, and invalid/vacant
slots. Add provider and validation counters and assert `split_mut` does not
increment them. Each error must leave the group unchanged.

- [ ] **Step 2: Implement the ordered proof.**

Resolve all slots; reject duplicates; reuse retained injectivity where present;
prove missing injectivity once; partition by allocation slot; compare checked
reachable byte envelopes with empty ranges disjoint; then construct children.
Use one adjacent private `unsafe` block only after all proofs pass, with a
SAFETY comment stating the distinct allocation entries and exclusive group
borrow that justify each reference. Do not enumerate arbitrary strided
elements, map providers, enqueue work, or repeat P4 checks.

- [ ] **Step 3: Verify the borrow-contract boundary.**

Run the private lifetime test and, when a truthful fixture is possible, the
focused UI test. The expected compile contract is that no group/root method can
be called while the returned mutable child vector is alive; no public API or
compatibility alias may be introduced solely to make this fixture compile.

- [ ] **Step 4: Run the full focused artifact and commit.**

```bash
cargo test -p tenferro-tensor --test storage_allocation_group --quiet
git add crates/tenferro-tensor/src/storage/group.rs \
  crates/tenferro-tensor/src/storage/tests/group.rs \
  crates/tenferro-tensor/tests/ui/storage
git commit -m "feat(storage): add p5 n-way disjoint split proof"
```

## Task 5: Activate the P5 ledger and update normative design/worklog

**Files:**
- Modify `scripts/storage-ownership-contracts.toml`.
- Modify `scripts/test-storage-ownership-contracts-v2.py`.
- Modify `docs/design/storage-ownership-contracts.md`.
- Create `docs/worklogs/2026-08-04-issue-1561-p5-allocation-group.md`.

- [ ] **Step 1: Promote exactly one ledger row.**

Change only `p5-allocation-group.state` to `{ kind = "active" }` and add
only `p5-allocation-group` to `ACTIVE_IDS`. Keep P3, P6, P7, P8, P9, P10,
P11, P12, P13-A, and P13-B deferred; do not change artifact identity or
command argv.

- [ ] **Step 2: Update the design document.**

Replace the P5-deferred wording with the implemented group table, central proof
order, borrow lifetimes, extraction semantics, and active artifact. Preserve
the graph and all proportional-safety non-goals.

- [ ] **Step 3: Write the exact-head worklog.**

Record the candidate commit, changed files, one active P5 row, test command,
normal checks, coverage for `group.rs`, Miri if available, and the fact that
P3/P9 and later phases remain deferred.

- [ ] **Step 4: Run manifest/doc checks and commit evidence.**

```bash
python3 scripts/check-storage-ownership-contracts.py
python3 scripts/check-storage-design-docs.py
python3 scripts/test-storage-ownership-contracts-v2.py
git diff --check
git add scripts/storage-ownership-contracts.toml \
  scripts/test-storage-ownership-contracts-v2.py \
  docs/design/storage-ownership-contracts.md \
  docs/worklogs/2026-08-04-issue-1561-p5-allocation-group.md
git commit -m "docs(storage): activate p5 allocation group evidence"
```

## Task 6: Exact-head verification and phase handoff

**Files:** No source changes expected; update the P5 worklog only if command
output requires an evidence correction.

- [ ] **Step 1: Run formatting and focused/full Rust gates.**

```bash
cargo fmt --all --check
git diff --check
cargo test -p tenferro-tensor --all-targets --quiet
cargo clippy -p tenferro-tensor --all-targets -- -D warnings
cargo llvm-cov -p tenferro-tensor --lib --summary-only
cargo +nightly miri test -p tenferro-tensor --lib storage::tests::group --quiet
```

Expected: all commands exit 0; new `group.rs` line coverage is at least 90%.

- [ ] **Step 2: Run the exact active artifact and contract checks.**

```bash
cargo test -p tenferro-tensor --test storage_allocation_group --quiet
python3 scripts/test-storage-ownership-contracts-v2.py
python3 scripts/check-storage-ownership-contracts.py
python3 scripts/check-storage-design-docs.py
python3 scripts/repository-rules-review.py --dry-run \
  --llm-skipped-reason "P5 private group proof boundary is issue-authorized; source review performed independently." \
  --base origin/main --head "$(git rev-parse HEAD)" \
  --output-json /tmp/p5-rules-review.json
```

Expected: deterministic rules verdict is `pass`; the only warning is the
explicitly recorded skipped external LLM review.

- [ ] **Step 3: Confirm the worktree and phase boundary.**

```bash
git status --short --branch
git log -1 --oneline
```

The tracked worktree must be clean, the active ledger set must contain exactly
the P0/P1/P2/P4/P5 rows, and no P3/P9 or later code/ledger transition may have
started.

- [ ] **Step 4: Commit any final evidence correction and report the handoff.**

If only the worklog changes, commit it with:

```bash
git add docs/worklogs/2026-08-04-issue-1561-p5-allocation-group.md
git commit -m "docs(storage): bind p5 verification evidence"
```

Report P5 completion and stop. The next phase requires an explicit selection;
the overall #1555 goal remains active until P13-B independently audits and
closes it.

