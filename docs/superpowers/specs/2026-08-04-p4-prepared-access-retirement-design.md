# P4 Prepared Access and Retirement Design

Date: 2026-08-04  
Authority: #1555, #1560, `docs/design/storage-ownership-contracts.md`

## Scope and non-goals

This phase activates only the five P4 obligations in the storage ownership
ledger. It builds the private prepared-access and retirement kernel on top of
the P2 root claim vocabulary. It does not migrate the existing public
`TypedTensor`/`TypedTensorView` APIs, introduce `AllocationGroup`, or start the
atomic P3/P9 cutover. Those changes remain owned by their later phases.

The implementation preserves the proportional-safety amendment: no
compatibility or provider bridge, quarantine/poison/retry/recovery registry,
destructor recovery, cancellation protocol, cryptographic evidence, or
repeated static validation is added. Validation required for memory safety,
aliasing, provider mapping, and retirement remains at the construction or
provider boundary.

## Design

### Checked access state

`storage/prepared.rs` adds private, enum-authoritative state:

```rust
enum CheckedLayout<R: TensorRank> {
    Contiguous { element_range: Range<usize> },
    Strided(CheckedStrided<R>),
}

struct CheckedDescriptor<R: TensorRank> {
    span: RootBoundSpan,
    layout: CheckedLayout<R>,
    dtype: DType,
    element_size: usize,
}

struct CheckedInjectiveDescriptor<R: TensorRank> {
    descriptor: CheckedDescriptor<R>,
    proof: WriteInjectivityProof,
}

struct CheckedRead<'a, R: TensorRank> {
    owner: StorageRef<'a>,
    descriptor: CheckedDescriptor<R>,
}

struct CheckedWrite<'a, R: TensorRank> {
    owner: StorageMut<'a>,
    descriptor: CheckedInjectiveDescriptor<R>,
}
```

The sole checked constructor receives a borrowed owner and a proposed layout.
It validates shape/stride/offset arithmetic, dtype size and alignment, exact
root-span containment, storage/provider compatibility, and mutable
injectivity. It publishes the checked value only after all checks pass. The
prepared transition consumes that value and never repeats those checks.

The layout uses the repository `TensorRank` abstraction so fixed-rank and
dynamic-rank paths retain their rank parameter. The strided plan stores shape,
signed strides, initial element offset, element count, and an incremental carry
cursor initialized before iteration. It does not retain a storage or provider
receiver.

### Provider mapping and typed traversal

The P2 `BackendAllocation` boundary gains private provider mapping hooks that
return borrowed, type-erased byte mappings. The provider receives the already
checked root-bound span and dtype once. A mapping guard owns any provider map
state for its borrow and exposes bytes only to the private typed conversion
helpers. The helpers check the retained byte length/alignment and create the
typed guard once; the only unsafe conversion is adjacent to that invariant.
No provider `Arc` is cloned during preparation.

The public existing `HostReadGuard`/`HostWriteGuard` APIs are not changed in
this phase. P4's private guards support both typed read slices and mutable
typed slices because strided mutable traversal must yield `&mut T`; providers
may implement the mapping guard with an interior-mutable backend lock or a
native borrowed map. The fake provider in the proof tests uses a mutex-backed
byte allocation and atomic counters.

The prepared hierarchy is:

```rust
enum PreparedRead<'a, T: TensorScalar, R: TensorRank> {
    Host(PreparedHostRead<'a, T, R>),
    Device(PreparedDeviceRead<'a, T, R>),
}

enum PreparedWrite<'a, T: TensorScalar, R: TensorRank> {
    Host(PreparedHostWrite<'a, T, R>),
    Device(PreparedDeviceWrite<'a, T, R>),
}

enum PreparedHostRead<'a, T, R: TensorRank> {
    Contiguous(PreparedContiguousRead<'a, T, R>),
    Strided(PreparedStridedRead<'a, T, R>),
}

enum PreparedHostWrite<'a, T, R: TensorRank> {
    Contiguous(PreparedContiguousWrite<'a, T, R>),
    Strided(PreparedStridedWrite<'a, T, R>),
}
```

The device variants retain the checked capability/layout and an opaque
provider-prepared token, but no host guard, pointer, slice, or iterator. Their
binding methods accept only the prepared value and do not accept a replacement
range, key, provider, or access mode.

The contiguous host variants expose `as_slice*` and `iter_contiguous*` after a
single range extraction. The strided variants expose iterators whose `next`
performs only exhaustion, typed access, and precomputed stride/carry updates.
The mutable iterator is constructed only from `CheckedInjectiveDescriptor` and
owns the sole mutable guard borrow. Empty, singleton, reverse, noncontiguous,
overflow, out-of-span, misaligned, wrong-dtype, and non-injective layouts are
rejected before a prepared object exists.

### Retirement

`storage/retirement.rs` adds a small private state boundary. A detached
admission consumes prepared provider bindings into one `RetirementRecord` that
owns the event, bindings, `RootResourcePin`s, and provider context. A user
handle may detach observation, but never releases the record early.

The record has no public lifecycle booleans. `finish` consumes it and has three
typed outcomes:

- proven completion: release binding/event/root/context resources exactly once;
- provider failure after completion is proven: release the same resources and
  return a typed failure;
- completion unproven: retain the complete private record permanently and
  return diagnostics without an owner or recovery operation.

Pre-admission rejection occurs before a record is created and returns the
unchanged prepared package. No asynchronous provider is simulated as
recoverable after possible admission.

## Ledger artifacts

Each deferred row gets a tracked integration artifact that invokes the real
private proof module through Cargo's library test harness:

| Obligation | Artifact | Private proof |
|---|---|---|
| `p4-production-borrow-contract` | `tests/storage_borrow_contract.rs` | checked private borrow lifetime and unchanged failure carrier |
| `p4-access-retirement` | `tests/storage_prepared_validation.rs` | invalid layouts rejected before preparation |
| `p4-provider-release-lifecycle` | `tests/storage_provider_event_retirement.rs` | proven/unproven exactly-once retention |
| `p4-traversal-resolution-counts` | `tests/storage_traversal_resolution.rs` | fake-provider counts independent of element count |
| `p4-prepared-access-api` | `tests/storage_prepared_access.rs` | enum API, typed contiguous/strided traversal, no replacement inputs |
| Existing UI borrow cases | `tests/ui/storage/fail/*.rs` | public owner/view/write borrow restrictions; private P4 types are exercised by the nested library harness |

The integration artifacts are test-only bindings to private proofs, not a
second production authority or a runtime repeated-validation path. The P4
ledger rows become active only after all five artifacts, source contracts, and
their exact commands exist.

## Verification

The RED phase first adds one failing proof per acceptance property, then the
minimal implementation is made GREEN. Required checks are:

```text
cargo fmt --all --check
git diff --check
cargo test -p tenferro-tensor --all-targets --quiet
cargo clippy -p tenferro-tensor --all-targets -- -D warnings
python3 scripts/test-storage-ownership-contracts-v2.py
python3 scripts/check-storage-ownership-contracts.py
python3 scripts/check-storage-design-docs.py
```

Storage coverage must remain above 90% lines for the new checked and retirement
modules. The exact candidate HEAD is reviewed by the repository-rules checker
and by independent specification and quality reviewers before any later phase
starts.
