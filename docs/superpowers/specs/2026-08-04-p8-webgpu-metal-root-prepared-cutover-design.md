# P8 WebGPU and Apple/Metal Root/Prepared Cutover Design

Date: 2026-08-04

Status: design complete; P8 ledger row remains deferred

Authority: #1555, #1564, `docs/design/storage-ownership-contracts.md`
(G1/G3/G5), and `scripts/storage-ownership-contracts.toml`

## Scope

P8 migrates WebGPU device-local storage and Apple host-visible Metal storage to
the same scalar-independent root, move-only span claim, checked descriptor,
prepared device access, and retirement contracts already used by the common
storage kernel and CUDA. It covers allocations, explicit upload/download,
outputs, workspaces, kernels, mappings, empty tensors, runtime execution, FFT
and linalg consumers, and Apple CPU/Metal access transitions.

P8 does not normalize the final public provider namespaces or performance
surface; P10 owns that work. It does not activate P10, P11, P12, or P13. The P8
row remains deferred until its real artifact exists and the exact feature-gated
command passes.

## Rejected designs

1. **Typed `BackendStorage<T>` retained behind the root.** This preserves a
   second scalar-specific storage family and makes reinterpretation/provider
   preparation depend on the descriptor's original scalar.
2. **A root beside `WebGpuBuffer` or an Apple CPU owner beside a Metal owner.**
   This creates parallel physical ownership and ambiguous release authority.
3. **Mapping or downloading on every CPU/Metal endpoint switch.** This is a
   hidden transfer and defeats Apple shared allocation semantics.
4. **Rechecking key/range/layout at every binding or element.** These facts are
   retained by the descriptor and prepared payload; repeating them adds cost
   without preventing a reachable failure.
5. **Recovery registries for broken provider contracts.** Supported in-tree
   providers are trusted unsafe implementations. Normal mapping, admission,
   and completion failures are typed; unsafe-contract violations are not a
   second runtime threat model.

## Final provider allocation boundary

### Cross-crate ingress

The storage kernel keeps one scalar-independent unsafe provider contract. P8
uses the reserved cross-crate visibility promotion rather than preserving the
typed bridge:

```rust
#[doc(hidden)]
pub unsafe trait BackendAllocation:
    core::fmt::Debug + Send + Sync + 'static
{
    fn root_extent(&self) -> RootResourceExtent;
    fn provider_kind(&self) -> BackendId;
    fn capabilities(&self) -> ProviderCapabilities;

    fn prepare_device_access(
        &self,
        request: DeviceAccessRequest<'_>,
    ) -> Result<Box<dyn PreparedDeviceAccess>, DeviceAccessError>;

    fn map_read(
        &self,
        span: RootBoundSpan,
        dtype: DType,
    ) -> Result<ProviderReadMapping<'_>, AccessError>;

    fn map_write(
        &self,
        span: RootBoundSpan,
        dtype: DType,
    ) -> Result<ProviderWriteMapping<'_>, AccessError>;
}
```

The trait is public only because provider crates are separate Rust crates. It is
`#[doc(hidden)]`, unsafe to implement, and not a second tensor API. The safe
root importer consumes one boxed implementation and validates its root extent
before publishing one `OwnedStorage`. There is no generic
`BackendStorage<T>`, `StorageBuffer::Backend`, or provider-to-legacy conversion
in the final P8 path.

The unsafe implementation promises unique ownership of one provider root,
truthful stable extent/domain/local identity, valid `Send`/`Sync`, valid
mapping lifetimes, and exactly-once provider destruction. The storage core does
not recover from a violation of that contract.

### WebGPU allocation

```rust
struct WebGpuAllocation {
    handle: cubecl_runtime::server::Handle,
    extent: RootResourceExtent,
    runtime: WebGpuRuntimeIdentity,
    visibility: WebGpuVisibility,
}

enum WebGpuVisibility {
    DeviceLocal,
    AppleShared {
        resource: ManagedResource<WgpuResource>,
        domain: Arc<AppleDomainState>,
    },
}
```

The type is scalar-independent and non-`Clone`. The root owns it once. Provider
capacity, including WebGPU minimum/padded allocation size, is distinct from the
exact logical root extent. An empty tensor has a zero-byte logical span; any
provider placeholder capacity is never exposed or dereferenced as tensor data.

`AllocationDomainId` is mandatory for both variants. `AllocationId` is derived
from the actual provider allocation namespace. Device ordinal and runtime
identity are endpoint metadata, not ownership proof.

## Descriptor construction and preparation

Tensor construction consumes `WebGpuAllocation` into an allocation-group slot
and inserts one descriptor after validating dtype encoding, checked byte
range, alignment, layout, exact root containment, placement, and provider
compatibility. These facts remain in `DescriptorRecord`.

Device preparation produces one non-cloneable opaque state:

```rust
struct WebGpuPreparedAccess {
    handle: cubecl_runtime::server::Handle,
    shape: Box<[usize]>,
    strides: Box<[usize]>,
    access: WebGpuAccessMode,
    reservation: Option<GpuAccessToken>,
}
```

The provider may retain the minimum CubeCL handle/reservation required by one
binding. Such a handle is a private lifetime/use lease and cannot mint an
owner, root claim, `StorageMut`, or public write API. The checked descriptor and
root borrow remain in `PreparedDeviceRead` or `PreparedDeviceWrite`.

Preparation performs only access-time work:

- acquire the provider's GPU reservation or reject an active host mapping;
- convert the already-checked shape/stride representation to CubeCL binding
  metadata;
- retain provider state until binding/retirement.

It does not compare a replacement domain/key/range, recompute bounds or layout,
map host bytes, or clone a provider context `Arc`. Binding consumes the exact
prepared payload and accepts no tensor, provider, request, key, range, or
handle argument. Existing kernel, structural, FFT, and linalg launch sites use
this single path.

An unadmitted bind failure returns the unchanged prepared payload. Once an
enqueue-capable call is reached, G3 owns all failure and retirement behavior.

## WebGPU mapping rules

### Device-local allocations

`map_read` and `map_write` return `UnsupportedCapability`. They do not trigger a
download, staging allocation, CPU fallback, or queue-wide copy.

### Apple host-visible allocations

The same `WebGpuAllocation` root is visible through CPU and Metal endpoints.
A shared root borrow may create a span-scoped host read mapping; only an
exclusive root/group borrow may create a span-scoped host write mapping. The
provider may map the whole underlying WebGPU resource privately, but the guard
exposes only the checked requested byte range and requested valid scalar
representation.

Host mapping follows the existing CubeCL exclusion protocol:

- host mapping rejects an active GPU reservation;
- GPU preparation rejects an active host mapping;
- read mappings may coexist only when the provider permits them;
- write mapping remains exclusive through the Rust capability and provider
  guard;
- guard drop performs the required unmap/flush transition.

The provider's whole-resource exclusion may conservatively serialize disjoint
spans. It is an implementation limit, not a second public ownership model.

## Apple CPU/Metal endpoint transitions

`AppleContext` owns provider/runtime configuration, not tensor storage. A tensor
created in the context has one root, one allocation key/domain, and one owner
across CPU and Metal operations.

- `upload_tensor` explicitly allocates one Apple-shared destination and copies
  host bytes once.
- A CPU operation acquires a host mapping/guard on that same root.
- A Metal operation ends incompatible mappings, performs required
  synchronization, and acquires GPU access on that same root.
- Returning to CPU waits/unmaps as required, then maps the same root.
- `download_tensor` explicitly allocates a separate host destination and copies
  bytes once.

CPU↔Metal endpoint changes record synchronization/map transitions but zero
transfer bytes and zero new tensor allocations. The allocation key, exact root
span, managed state, and allocation identity remain unchanged.

`AppleTransferStats` counts only explicit upload/download bytes. Separate test
counters record mapping, synchronization, preparation, binding, and allocation
without reclassifying them as transfers.

## Reinterpretation and dtype encoding

C32↔F32 and C64↔F64 reinterpretation replaces descriptor metadata over the same
root. It preserves WebGPU/Metal runtime identity, allocation domain/key, exact
span, visibility mode, managed resource state, and transfer/allocation
counters. No host round trip occurs.

Provider encoding is validated at root/descriptor construction for supported
scalar pairs. Bool remains its provider-defined byte representation and is not
admitted to the sealed complex/real reinterpretation pairs.

## Outputs, workspaces, and extension interop

Every tensor output is created by consuming a newly allocated provider handle
into a root immediately. Output completion does not rely on
`Handle::can_mut()` as public ownership proof and does not pass through a
cloneable `WebGpuBuffer<T>`.

Provider workspaces remain provider-private and are retained by the prepared
operation/event retirement record for every enqueue that uses them. They are
not wrapped as public tensor owners unless they are actual tensor outputs.

The flat `webgpu_interop` surface that returns/clones raw CubeCL handles is not
a public API. Workspace extension crates use a hidden, session-scoped prepared
binding boundary. Any unavoidable in-tree unsafe handle projection is bounded
by the execution-session lifetime, documented with synchronization and
post-retirement invalidity requirements, and cannot be returned from the
callback. WebGPU/Metal does not gain a safe unleased raw-handle API.

## Retirement

After admission, a WebGPU retirement record retains:

- all consumed execution owners/groups;
- prepared bindings and GPU reservations;
- the exact `WgpuSubmission`/event-domain token;
- root holds and provider runtime context.

A successful event wait proves retirement and releases the record exactly once
before publishing `Completed` or `RetiredFailed`. If exact stream completion
submission fails, a successful whole-client synchronization may prove
retirement while preserving the original typed error. If both proofs fail,
the result is `CompletionUnproven`: no owner is returned and the complete
private record is retained permanently.

A best-effort `Drop` path may report diagnostics, but it may not release a
record whose completion is unproven. There is no quarantine state, retry API,
or global retirement registry.

## State table

| Transition | Authority | Access-time work | Failure result |
|---|---|---|---|
| provider handle → root | consumed owner | validate root descriptor once | typed error before publication |
| checked shared descriptor → device prepared read | shared borrow | GPU reservation/binding metadata | unchanged checked capability |
| checked exclusive descriptor → device prepared write | exclusive borrow | exclusive GPU reservation/binding metadata | unchanged checked capability |
| Apple shared root → host read/write | shared/exclusive borrow | wait/map exact span | typed mapping error, no transfer |
| device-local root → host map | matching borrow | none | `UnsupportedCapability` |
| prepared payload → binding | prepared capability | consume opaque state | unchanged payload before enqueue |
| enqueue may have occurred | task ownership | event retirement | no immediate owner return |
| retirement proven | worker/reaper | release bindings/event/roots/context once | completed or retired-failed owner result |
| retirement unproven | private permanent owner | none proven | diagnostics only; no owner |

## Artifact and verification contract

P8 creates the artifact fixed by the reconciled Issue comment and v2 ledger:

```text
crates/tenferro-gpu/tests/storage_provider_webgpu.rs
cargo test -p tenferro-gpu --features webgpu --test storage_provider_webgpu
```

The artifact includes the common provider contract and covers:

- scalar-independent one-owner construction and exactly-once release;
- device-local mapping rejection without transfer/fallback;
- Apple shared read/write guards and host/GPU exclusion;
- device write followed by host read;
- CPU→Metal→CPU identity preservation and zero transfer bytes;
- empty, offset, contiguous, strided, and sealed reinterpret descriptors;
- prepared-once counts independent of element count;
- binding with no replacement provider/key/range request;
- immediate owner/handle drop after enqueue;
- proven success/failure retirement and completion-unproven ownerlessness;
- outputs/workspaces retained through retirement;
- source/API inventory proving typed buffers, direct handle clones at launch
  sites, optional domains, provider bridges, and Apple dual owners are absent.

Metal-specific assertions execute only on a real Apple/Metal lane locally; P11
must run them in required mode. Ordinary unsupported environments produce a
structured skip from the test harness, but a structured skip cannot activate
P8 by itself.

P8 updates `docs/design/gpu-backend-design.md`, the device guide, Apple
examples/tutorials, and session-scoped unsafe interop rustdoc. It distinguishes
mapping/synchronization transitions from transfers.

## Explicit non-goals

- no provider compatibility bridge or second ownership family;
- no CPU fallback, hidden host round trip, transfer, or materialization;
- no public safe raw handle/pointer;
- no per-element provider/storage resolution;
- no repeated descriptor identity/range/layout checks after preparation;
- no cancellation, quarantine, poison/retry, malicious-provider, digest,
  nonce, or attestation machinery;
- no P10 namespace redesign or P11 hardware claim in this phase.
