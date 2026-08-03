# P3+P9 Atomic Host Ownership and Submission Cutover

Date: 2026-08-04

## Authority and scope

This design implements the next graph unit after P5: the atomic cohort of
#1559 (P3 host ownership) and #1565 (P9 submission). It follows the merged
#1598 contract, G2/G3/G4/G7, and the proportional-safety amendments on #1555.
P6 and later remain deferred until this cohort is complete.

The implementation preserves Rust aliasing/lifetime soundness, numerical
correctness, explicit device behavior, and provider retirement. It does not
add compatibility aliases, provider bridges, COW, hidden copies, global
liveness/generation registries, quarantine/poison/retry/cancellation
protocols, cryptographic evidence, or repeated static validation.

## Chosen architecture

### One move-only owner and borrowed views

`TypedTensor<T, R>` and the dtype-erased `Tensor` remain the stable public
names, but become move-only owners. Their storage is private and owner-scoped;
the public `Buffer<T>`, `BackendBuffer<T>`, `TensorOwnedView`, and pair-only
mutable split surface are removed. `duplicate()` is the explicit copy boundary
and returns a fresh allocation identity.

`TypedTensorView<'a, T, R>` and `TypedTensorViewMut<'a, T, R>` contain only a
borrowed descriptor and the group borrow. `as_view()`/`as_view_mut()` retain
static rank, perform no allocation or refcount/layout clone, and expose typed
access through the P4 prepared-access hierarchy. Dynamic rank remains the
explicit `DynRank` case. Mutable access is derived only from `&mut` and is
non-cloneable.

Host construction imports a `Vec<T>` once into the private root/provider
boundary and publishes one checked group descriptor. Backend allocations are
not generalized or bridged by this phase; CUDA/WebGPU/Metal migration remains
owned by P7/P8, while their existing private provider paths are not exposed as
new public compatibility APIs.

### Dtype-erased values and AD retention

`TensorValue` becomes one owning bundle: an `AllocationGroup` plus local
descriptor slots and output/retained-input metadata. It is not cloneable.
Read-only eager/AD handles may share an immutable record that owns this bundle;
the handle is a lookup/read capability and never a write authority. A new
mutable result is a new bundle or an exclusive borrow of its owner. Lazy view
metadata appends a descriptor to the same group instead of storing an `Arc` to
an independently cloneable tensor.

The eager value record and traced checkpoint containers retain direct bundles
and descriptor slots. Retention does not clone storage or allocate a materialized
cache. `take_grad` and owner extraction use local descriptor uniqueness; no
process-global liveness table participates.

### Detached and borrowed runtime submission

`ExecutionInputs` owns one `AllocationGroup` and local bindings. Detached
submission consumes it. All validation, preparation, and worker admission that
can fail before enqueue occurs before ownership transfer; such failures return
the exact unchanged package. Once enqueue may have occurred, the in-flight
record owns the bundle and P4 retirement resources. A proven completion or
proven retired failure may return an owned result bundle. If completion is not
proven, the typed diagnostic returns no owner and the private retirement record
retains bindings, event, roots, and provider context permanently.

Dropping an execution handle detaches observation and never cancels or releases
in-flight storage. A borrowed submission is synchronous-only and is accepted
only by a provider that proves no work survives return or unwind; asynchronous
providers reject it before admission. No panic-catching or recovery state is
introduced for the borrowed path.

## Error and state rules

- Ordinary constructor/descriptor validation returns typed errors before a
  public child capability is constructed.
- Mutable split/extraction failures leave the owner bundle unchanged.
- Pre-admission detached failures return the unchanged `ExecutionInputs`.
- Post-admission failures return owners only after retirement is proven;
  completion-unproven diagnostics return no owner.
- There is one owner for every physical span. Repeated or aliasing logical
  outputs retain descriptors, not additional owners.

## Proof artifacts

The phase activates the four P3 rows and the P9 row atomically:

- `storage_compile_contract.rs`: owner/view borrow and non-Clone compile
  contracts, including no mutable root access while a child lives;
- `storage_static_rank.rs`: ordinary owner/view/reborrow/traversal rank
  preservation;
- `storage_as_view_allocation.rs`: warmed allocation/refcount/provider/layout
  counters remain unchanged across `as_view*`;
- `storage_auto_traits.rs`: the intended `Send`/`Sync` surface;
- the existing P9 command in `storage_compile_contract.rs`, extended with
  consuming submission, unchanged pre-admission failure, borrowed-provider
  rejection, detach, aliasing output, extraction, and AD retention cases.

The phase is complete only when all five rows pass on one exact clean commit,
the public old owner/value paths are physically absent, and P6 remains deferred.
