# P7 CUDA root/prepared cutover

## Scope

This specification is the next bounded step for Issue #1555/#1563. It covers
the CUDA provider only: one physical CUDA allocation owner must be retained by
the tensor storage root, and device preparation must retain the provider state
that the later binding consumes. WebGPU/Metal remains a separate P8 step. The
P7 ledger row stays `deferred` until its implementation, artifact, and exact
feature-gated command all pass.

The existing domain and byte-length work is a prerequisite, not a substitute
for this cutover. This work starts from the current code in which backend
`TypedTensor` values keep a `StorageBuffer::Backend` owner while their
`AllocationGroup` is absent, and in which `PreparedDeviceToken` is only a
zero-sized marker.

## Decision

Use a direct full cutover to the root/group owner model. Do not keep a
provider-buffer field beside a root owner, and do not add a wrapper that
mirrors or forwards ownership between the two representations.

### One owner

`TypedTensor<T, R>` and both tensor views will use the same ownership shape for
host and backend storage:

```rust
struct TypedTensor<T, R: TensorRank = DynRank> {
    storage: OwnedStorage,
    layout: TensorLayout<R>,
    placement: Placement,
    _scalar: PhantomData<T>,
}
```

For a CUDA tensor, the `RootResourcePin::Backend` value contains the provider
owner itself (the CubeCL buffer and its mandatory domain/identity metadata).
The group/descriptor records the root span and checked layout; it does not
own a second buffer and it does not clone a provider handle. If the existing
`AllocationGroup` representation is required for multi-descriptor tensors,
the owner is moved into its one allocation slot and the tensor keeps only the
corresponding descriptor slot. There is no `buffer` plus `group` split after
the cutover.

The root boundary receives a narrow object-safe provider-allocation capability
for the provider crates to implement. It carries root extent, provider kind,
allocation identity, host-map hooks when supported, and provider preparation
hooks. It does not expose a safe pointer, a raw handle, a transfer operation,
or a recovery/compatibility path. The capability is private to storage
ownership and is used only through `StorageRef`/`StorageMut` and the prepared
transitions. The pre-existing `BackendStorage` inspection methods may remain
at the construction/diagnostic edge, but they are not an additional physical
owner in a tensor.

Construction consumes the provider owner into the root once. Views reborrow
the root and retain no cloneable owner projection. Explicit duplication is a
new same-placement allocation and copy, with the destination becoming the new
root; no hidden transfer or fallback to host is introduced.

### Prepared device state

`PreparedDeviceRead` and `PreparedDeviceWrite` become the only device
preparation payloads. Each retains:

- the existing checked capability and `CheckedLayout`;
- an opaque provider-prepared state produced by the root's provider hook; and
- the scalar/rank marker needed for typed dispatch.

The provider state is lifetime-bound to the checked root borrow and is
non-`Clone`. Its provider-specific binding entry point is a narrow hidden
boundary consumed by the CUDA dispatch crate; it accepts the exact prepared
object and no replacement provider, request, key, range, or pointer. Binding
does not revalidate layout/domain/length and does not perform a second
preparation pass. A failed pre-admission bind returns the exact unchanged
prepared object.

Device preparation never creates a host mapping or a host-visible pointer.
The CUDA provider may retain the provider-native launch state needed by one
binding and may expose raw interop only inside an owner/binding-scoped unsafe
callback. No safe unscoped handle or pointer accessor is added.

### CUDA boundary

The CUDA implementation supplies the root capability for `CubeclBuffer` and
uses its existing mandatory allocation domain, byte length, device ordinal,
and allocation identity. Provider preparation checks the already-retained
descriptor/provider relationship once at the transition boundary, then
stores the resulting provider state. CUDA launch code consumes that state
directly; it does not recover the buffer from `TypedTensor::buffer()`, ask for
a provider argument, or reconstruct a raw handle from metadata.

The owner remains live through binding and any detached retirement record. A
post-admission failure follows the existing retirement contract and never
returns an owner before completion is proven. This cutover does not introduce
quarantine, retry, repeated validation, cryptographic identity, or a new
recovery state machine.

## Rejected alternatives

1. **Dual owner plus forwarding wrapper.** Rejected because it leaves two
   physical ownership paths, makes drop/lifetime auditing ambiguous, and would
   be compatibility machinery rather than a cutover.
2. **Callback-only provider access with `StorageBuffer` still owning.**
   Rejected because prepared access would not be rooted in the same owner as
   the descriptor/group and would preserve the current split authority.
3. **Generic host/device transfer or recovery layer.** Rejected as outside
   P7/P8 and contrary to the proportional-safety constraints.

## Implementation sequence

1. Add the narrow root/provider capability and CUDA implementation without
   changing the public raw-interoperability surface.
2. Change CUDA tensor construction so the provider owner is consumed into the
   root/group and all typed/view metadata routes through the descriptor and
   root capabilities. Remove direct CUDA ownership reads from `TypedTensor`.
3. Replace the zero-sized device token with the provider-prepared state and
   make the CUDA binding path consume the exact prepared payload.
4. Migrate the remaining CUDA dispatch sites and detached-retirement paths to
   the prepared payload. Keep host preparation and CPU paths behaviorally
   unchanged.
5. Add the P7 contract artifact and activate the ledger row only after the
   exact CUDA command passes on the final commit. The artifact must prove one
   owner/drop path, provider state retention, no host access for device
   preparation, explicit same-placement duplication, and the absence of safe
   unscoped raw access.

P8 applies the same boundary to WebGPU/Metal only after this CUDA slice is
complete; it is not started by this specification.

## Non-goals and safety limits

- no compatibility adapter, shadow owner, migration/quarantine registry, or
  fallback storage path;
- no hidden host materialization or provider transfer;
- no repeated validation in binding/enqueue;
- no public safe raw pointer/handle API;
- no changes to the P7/P8 ledger status before the required artifact exists;
- no activation of P10 or later phases.

## Verification

During implementation, run the focused tensor/storage unit and compile-fail
tests after each owner/prepared slice, then run the CUDA provider tests and the
existing CUDA GPU/linalg/AD suites. Before activating P7, run:

```text
cargo fmt --all -- --check
cargo check --workspace --quiet
cargo test -p tenferro-gpu --features cuda --test storage_provider_cuda
python3 scripts/check-storage-ownership-contracts.py
python3 scripts/run-storage-ownership-contracts.py --diagnostics-json
```

The parent issue remains open until P8, the later selected phases, and the
independent P13-B closure audit all pass on one exact final commit.
