# Issue #1561 P5 allocation-group worklog

Date: 2026-08-04

## Authority and scope

This phase follows #1555 and the authoritative child issue #1561. P5 is the
selected next phase. P3/P9 and every later phase remain deferred; this work
does not imply the atomic cutover or close #1555.

The proportional-safety boundary is preserved. The implementation adds no
compatibility alias, recovery or retry path, quarantine state, cryptographic
evidence, provider bridge, global registry, slot-generation scheme, hidden
copy, or repeated validation. The public tensor API is not migrated in this
private kernel phase.

The implementation candidate is `7cdb22a3`; the boundary-coverage tests are
in `b84ed483`. The ledger, design-document, and this evidence update follow
those source commits and are verified at the resulting exact `HEAD`.

## Implementation

- `AllocationGroup` owns append-only allocation and descriptor tables. Each
  allocation entry retains one `OwnedStorage`; each descriptor retains checked
  rank-erased layout metadata, dtype/element metadata, root identity/span,
  provider kind, and an optional conservative reachable byte envelope.
- `DescriptorSlot` and `AllocationSlot` are opaque group-local indices. A
  slot carries no ownership or provider authority, and vacated entries are
  never rebound or renumbered.
- `view` and `view_mut` resolve through the borrowed group and return
  non-cloneable borrowed children. A mutable view proves write injectivity
  once when its descriptor has no retained proof.
- `split_mut` resolves all slots once, proves missing injectivity and pairwise
  conservative envelope disjointness, then returns N children. It does not map,
  enqueue, or repeat construction-time layout/range/storage/provider checks.
- `try_extract` succeeds only for the sole local descriptor of an allocation;
  it structurally vacates the descriptor and allocation slots and moves the
  owner without copying. `into_owner` consumes the group and preserves the
  unchanged group on failure.

## Proof artifacts

The canonical ledger artifact is
`crates/tenferro-tensor/tests/storage_allocation_group.rs`. It delegates to the
private `storage::tests::group` harness, which covers:

- N=0, singleton, and N>2 splits, permutation independence, empty and
  reverse-stride layouts, conservative overlap rejection, duplicate slots,
  and cross-root children;
- construction-time dtype/rank/span/alignment/layout checks, retained
  metadata, one-time mapping counters, and mutable injectivity;
- aliased versus sole-owner extraction, vacant-slot behavior, unchanged
  failures, and checked shape/envelope overflow without panic.

The group API's returned child lifetime and `PhantomData` borrow contract keep
the root group inaccessible while mutable children live. This private module
has no public compatibility surface; the existing public compile-contract
fixture remains the baseline for the later API cutover rather than being
invented as a P5-only artifact.

## Verification

The exact P5 evidence checks are:

- `cargo fmt --all --check` and `git diff --check`;
- `cargo test -p tenferro-tensor --all-targets --quiet`;
- `cargo test -p tenferro-tensor --test storage_allocation_group --quiet`
  (12 private proof tests);
- `cargo clippy -p tenferro-tensor --all-targets -- -D warnings`;
- `cargo llvm-cov -p tenferro-tensor --lib --summary-only` (`group.rs`:
  90.23% lines; `prepared.rs`: 91.58% lines; `root.rs`: 97.25% lines);
- `cargo +nightly miri test -p tenferro-tensor --lib
  storage::tests::group --quiet` (12 tests passed);
- `python3 scripts/check-storage-ownership-contracts.py`;
  `python3 scripts/check-storage-design-docs.py`; and
  `python3 scripts/test-storage-ownership-contracts-v2.py` (24/24 passed).

Only `p5-allocation-group` is promoted by this phase. P3/P9 and later rows
remain deferred, so no phase beyond P5 is resumed implicitly.
