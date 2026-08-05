# P5 AllocationGroup and N-Way Disjoint Views Design

Date: 2026-08-04<br>
Authority: #1555, #1561, `docs/design/storage-ownership-contracts.md` (G2/G5),
and `scripts/storage-ownership-contracts.toml`

## Scope and phase gate

This design activates only P5 (`p5-allocation-group`) after P4. It adds the
private storage-kernel representation for one physical owner with multiple
logical descriptors, one N-way disjoint mutable split, and copy-free
structural extraction. P3, P9, P6, and every later phase remain deferred.

The phase preserves the proportional-safety amendment. It does not add a
provider bridge, compatibility alias, global liveness registry, generation or
tombstone state, slot reuse, hidden copy/materialization, quarantine, retry,
recovery, cryptographic evidence, or repeated construction/map/enqueue
validation.

## Design choices

### Alternatives considered

1. A rank-parameterized `AllocationGroup<R>` would make typed child creation
   straightforward, but it cannot represent descriptors of different ranks in
   one group and therefore does not satisfy the G2 model.
2. Extending `TypedTensorViewMutPair` would preserve existing callers but would
   create a second split implementation and a compatibility surface explicitly
   removed by #1561.
3. The selected design is one rank-erased group-local table with one central
   proof boundary. Typed access is reconstructed only after a slot borrow has
   selected the retained metadata; no provider or allocation fact is checked a
   second time.

### Group representation

```rust
pub(crate) struct AllocationGroup {
    allocations: Vec<Option<OwnedStorage>>,
    descriptors: Vec<Option<DescriptorRecord>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DescriptorSlot(u32);
```

`AllocationSlot` is an internal append-only index for `allocations`. A
`DescriptorSlot` is meaningful only when resolved through the group that owns
it. Neither slot carries a root identity, provider handle, range authority, or
write capability. Moving an owner out leaves its allocation entry vacant;
descriptor entries are never reused.

`DescriptorRecord` retains the construction-time facts needed by G2:

- allocation slot and exact root-bound byte span;
- dtype and element size;
- rank-erased shape, strides, offset, element count, and compact/strided form;
- placement, storage, and provider metadata;
- an optional `WriteInjectivityProof` when construction already proved it;
- the conservative reachable byte envelope used by the split proof.

The record metadata is non-owning and non-authoritative. Physical lifetime is
provided only by the occupied `OwnedStorage` entry and its P4 root pin.

## Operations

The private API has one canonical path for each operation:

```rust
impl AllocationGroup {
    pub(crate) fn new() -> Self;
    pub(crate) fn insert_owner(&mut self, owner: OwnedStorage)
        -> Result<AllocationSlot, GroupError>;
    pub(crate) fn insert_descriptor<T, R>(
        &mut self,
        allocation: AllocationSlot,
        descriptor: DescriptorInput<R>,
    ) -> Result<DescriptorSlot, GroupError>;
    pub(crate) fn view<T, R>(&self, slot: DescriptorSlot)
        -> Result<GroupReadView<'_, T, R>, GroupError>;
    pub(crate) fn view_mut<T, R>(&mut self, slot: DescriptorSlot)
        -> Result<GroupWriteView<'_, T, R>, GroupError>;
    pub(crate) fn split_mut<T, R>(&mut self, slots: &[DescriptorSlot])
        -> Result<Vec<GroupWriteView<'_, T, R>>, DisjointViewError>;
    pub(crate) fn try_extract(
        &mut self,
        slot: DescriptorSlot,
    ) -> Result<OwnedStorage, ExtractError>;
    pub(crate) fn into_owner(
        self,
        slot: DescriptorSlot,
    ) -> Result<OwnedStorage, (Self, ExtractError)>;
}
```

The concrete implementation may keep constructors narrower while the module
is private; these operation semantics are normative.

`view` resolves an occupied local slot under `&self` and returns a read child
whose lifetime is bounded by the group borrow. `view_mut` resolves one slot
under `&mut self` and returns one exclusive child. Child wrappers are not
`Clone` and expose no owner extraction.

`split_mut` accepts an empty list and returns an empty vector. It accepts one
slot without needing a pairwise comparison and handles more than two slots in
the same proof. Duplicate slots, invalid/vacant slots, and mixed rank/dtype
requests are typed errors; the group is unchanged on every error.

`try_extract` succeeds only when the selected descriptor is the sole remaining
descriptor referring to its allocation slot. It vacates the descriptor and
allocation entries and moves the existing owner; it never copies bytes. An
aliased allocation returns a typed error without changing the group.

`into_owner` consumes the group, moves the selected owner, and explicitly
discards all other descriptor metadata and owners. A failure returns the
unchanged group with its error.

## Central disjointness proof

Only the private group proof module may construct multiple mutable children.
Its fixed order is:

1. Resolve each requested `DescriptorSlot` and reject invalid or vacant slots.
2. Reject duplicate allocation/descriptor selections that would create an
   aliasing mutable child.
3. Read retained layout, span, storage, provider, and placement metadata.
4. If a record lacks a retained injectivity proof, prove its logical addresses
   injective once; do not enumerate arbitrary strided elements merely to gain
   acceptance.
5. Partition selections by allocation slot and compare conservative reachable
   byte envelopes. Empty envelopes are disjoint.
6. Reject positive envelope overlap or any layout that is not provably
   disjoint with `PairwiseOverlap` or `NotProvablyDisjoint`.
7. Construct the N non-cloneable children from the already-proven disjoint
   allocation references.

The only unsafe operation in this phase is the final construction of multiple
   borrowed child references from distinct allocation entries after the proof.
It remains adjacent to the proof and private to `storage/group.rs`; no raw
pointer or `Arc` is exposed. A child lifetime cannot outlive the exclusive
`AllocationGroup` borrow, so the group root and all other group operations are
inaccessible while children exist.

`split_mut` does not map storage, enqueue work, or repeat provider/range/layout
validation. If child preparation is needed, it consumes the P4 checked
capability built from the retained descriptor and follows P4's provider
mapping rules.

## Errors and unchanged-state behavior

The group uses typed errors for invalid/vacant slots, invalid descriptor input,
wrong dtype/rank, missing allocation, non-injective layout, duplicate slot,
pairwise overlap, and not-provably-disjoint envelopes. Construction errors
return the owner or group according to the consuming operation; operational
validation errors leave the group unchanged. No error path creates a partial
child vector visible to the caller.

## Proof artifacts and tests

The phase has one ledger artifact:

```text
crates/tenferro-tensor/tests/storage_allocation_group.rs
```

The test harness invokes the private library module and covers:

- construction-time dtype, span, layout, alignment, storage, and provider
  rejection;
- aliasing read views and direct borrowed slot resolution;
- `split_mut` with N=0, N=1, and N>2;
- empty, scalar, compact, reverse-stride, overlap, and conservative
  not-provably-disjoint layouts;
- missing injectivity proof and retained-proof reuse;
- permutation-independent results and unchanged validation/map/enqueue
  counters;
- structural extraction uniqueness, vacant slots, unchanged failures, and
  explicit consuming discard;
- compile-fail proof that group/root access is unavailable while mutable
  children are alive.

The existing public `TypedTensorViewMutPair` tests remain as baseline coverage
until the owning public API removal phase; P5 does not add a compatibility
alias or route group behavior through that pair implementation.

## Exit criteria

P5 is complete only when the single artifact is active in the v2 ledger, the
design document records immutable aliasing, conservative disjointness, N-way
borrow lifetimes, and extraction, and the exact artifact command plus normal
format/test/clippy and contract checks pass on one clean commit. P6 and the
atomic P3/P9 cutover remain deferred.
