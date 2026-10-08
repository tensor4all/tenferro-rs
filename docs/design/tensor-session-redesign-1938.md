# Tensor/Session Redesign (#1938): Target Architecture

Status: **accepted target architecture; implementation in progress**. This is the
whole-architecture decision record for [#1938](https://github.com/tensor4all/tenferro-rs/issues/1938),
not proof that the proposed Rust surface or resource protocol already works.
Implementation supplies the focused correctness evidence below. No design can guarantee that implementation will
never expose a problem. In particular, changing an ownership or execution
contract must be brought back to this record, not hidden in an adapter.

Source anchor: `e55c0e17af903866effaac2c8bbc3ca23e126e05`, the #1929 handoff,
which already contains main `3fd19ee07`. The older
`fix/1665-eager-extension-unification` checkout is **not** the source baseline.
This proposal reuses #1929; it does not reopen its reduced completion contract.
API names not present at that anchor are proposed names, not runnable examples.

## 1. Critique incorporated

The previous draft asserted completeness before resolving these contradictions.

| Finding | Correction |
|---|---|
| It described the old preset-only `Tensor` enum, missing current external-scalar storage. | Preserve the opaque erased boundary and its external payload; separate storage from numerical dispatch (D1, D7). |
| It treated `DynRank` as heap-only and narrowed #1917 to static rank. | Preserve small-rank inline metadata, including dynamic rank and rank-changing real views (D3). |
| `HostTensor` was both a core type and an alias to a higher-crate type. | One public tensor family in `tenferro-tensor`; core keeps metadata/scalar facilities, with no reverse dependency (D10). |
| `TypeId + *mut ()` was called type-safe without a lifetime/provenance contract. | Use a lifetime-bound opaque native-session token; unsafe construction and backend-leaf recovery remain explicit (D7). |
| A concrete same-domain handle was claimed to solve A2, despite no lock/permit proof. | Withdraw that claim and the extra scope. Reuse borrowed sessions; explicitly defer A2 (D12). |
| Host pooling/custom deallocation, borrowed group retention, mapping lifetime and safe uninitialized output were unspecified. | Define the allocation/access state transitions, not just smaller structs (D2–D5, D11). |
| Typed admission was requested but the entry signature was left infallible. | Canonical entry returns a typed admission result; rejection precedes the callback (D6). |
| The draft mixed mandatory GPU protocol work with blanket GPU deferral, and treated Rayon handoff as removable bookkeeping. | Separate required safety/protocol changes from optional tuning and restore the issue's measurement scope (§5). |

Evidence at the source anchor:

- `crates/tenferro-tensor/src/types.rs`: `TensorCore`, `TensorPayload::{Native,
  External}` and opaque `Tensor`; do not infer its structure from an old enum.
- `crates/tenferro-tensor-core/src/lib.rs`: `ShapeVec = SmallVec<[usize; 8]>`
  and `StrideVec = SmallVec<[isize; 8]>`; `rank.rs` uses these for `DynRank`.
- `crates/tenferro-tensor/src/storage/root.rs`: `HostAllocation` has a weak
  recycler; recycling must not take execution-admission locks.
- `crates/tenferro-cpu-basic/src/pooled_uninit_output.rs`: the checkout borrows
  the whole pool; `buffer_pool.rs` already owns shared pool state.
- `crates/tenferro-cpu/src/lib.rs`: `FaerParallelismExt` itself calls
  `with_cpu_exec_session`; it is not evidence that the downcast is gone.
- `crates/tenferro-ad/src/eager.rs`: `with_execution_session` locks the eager
  backend before session entry. `docs/design/explicit-session-boundary.md`
  records A2's unresolved mutex/permit/`Send` problem.

## 2. Ownership model and invariants

| Owner | Owns | May borrow/retain |
|---|---|---|
| Ordinary tensor | One allocation authority and one shape | Recycler/deallocator or provider lifetime handle when needed |
| View | No allocation ownership; one logical layout and access authority | Owner, mapping guard, or checked retained-region guard |
| Backend/environment | Provider resources, execution policy, pools | Allocation-domain resources required by its providers |
| Session | Temporary entered authority and ordering | Environment, scratch leases, prepared operations |
| Prepared operation/program | Validated plan, validity keys, deliberate retained constants/factors | Per-run inputs/outputs only during execution |
| Group/AD retained value | Moved owners or explicit retained-region claims | Completion records and saved-value lifetimes |

These lifetimes may refer to each other. An output retaining a recycler or a GPU
allocation retaining its provider is necessary, not evidence of a layering bug.
The goal is to remove *redundant descriptors and mandatory execution/group
state*, not all cross-lifetime references.

Preserve numerical/AD semantics, validation and typed errors, unique mutable
access, explicit device movement, provider exclusion, and completion-dependent
retirement. An `Arc` reference count alone is neither permission to write nor a
reason to reject a proven-disjoint write. Auto-traits follow actual payload and
borrow semantics; arbitrary host `T` is not unconditionally `Send`/`Sync`.

Known invalid inputs, unsupported routes and declared nesting violations fail
before output writes. A provider failure after execution starts is different:
no general transactional rollback is promised. An `_into` destination remains
memory-safe but may contain partial results on such a failure; preserve any
stronger guarantee already documented by the particular operation.

## 3. Architecture decisions

### D1. One family, three actual representations

```text
TypedTensor<T, R = DynRank, D = Dynamic>
TypedTensorView<'a, T, R = DynRank, D = Dynamic>
TypedTensorViewMut<'a, T, R = DynRank, D = Dynamic>
```

Keep `Dynamic` as default, but make it compact too. `Host` is not a small escape
hatch beside the old universal descriptor. A sealed representation mechanism
selects owned/read/write payloads; it does not dispatch numerical operations.
Use GATs only for storage and lifetime-bearing access, not a generic operation
catalog. Shape belongs to the outer tensor once, never inside each storage arm.

| Representation | Storage meaning |
|---|---|
| `Host` | Stable CPU-accessible initialized elements with exclusive owning authority; plain, pooled or explicitly managed storage |
| `Gpu` | Provider-owned device allocation authority, identity and retirement state; not proof of a particular ordinal/provider |
| `Dynamic` | Runtime union of those two compact payloads, with shape outside the union |

Owning tensors are compact column-major; arbitrary strides belong to views.
Backend-erased device handles remain defined below backend leaf crates, in
`tenferro-tensor`, so `Dynamic` does not depend on `tenferro-gpu`. A device
handle may use a vtable; choosing a tagged `Dynamic` union does **not** eliminate
provider vtables or justify duplicating their ownership metadata.

**Scalar bounds:** plain host ownership/borrowing needs no `Copy`, `HostScalar`,
`TensorScalar`, or arithmetic trait. `Clone`/host compaction require `T: Clone`;
scalar erasure requires the appropriate type identity, and managed execution
requires supported dtype/operation capabilities. Device construction is bounded
at its methods, not imposed on the whole `Dynamic<T>` type. Thus a downstream
non-preset scalar can use `TypedTensor<MyScalar>` as host storage without
claiming GPU or numerical support. Do not add an empty public `HostScalar` trait.

`Tensor` stays opaque and supports both preset values and the existing external
host payload path. External storage erasure is not admission to preset kernels.
Existing registered external-scalar adapters remain explicit; they must not be
deleted merely because the compact default uses a tagged storage union. Runtime
owning erasure keeps its required `Send + Sync + 'static` bounds; a local host
container with non-Send elements need not be eligible for that erasure.

**Implemented mapping (#1938 ownership phase).** `TypedTensor<T, R, D>` carries
shape and placement once on the tensor, and `D::Storage<T, R>` is the owned
payload: `Host` is the plain/pooled `Vec<T>`, `Gpu` is the group-backed root
(provider authority, identity and retirement), and `Dynamic` is the runtime
union. An owner's layout is derived from its shape; it is not stored twice.
`Gpu` denotes the *group-backed* payload rather than a proven device ordinal, so
a host-managed group root is group-backed and its host access stays fallible.
Only the statically `Host` representation exposes infallible host access,
`Clone`, `Index`/`IndexMut` and owning host mappings; representation narrowing
(`into_host`/`into_gpu`) is checked and returns the unchanged owner on failure.
Host-representation constructors are spelled `from_host_vec_col_major` /
`from_host_vec_row_major` so that `TypedTensor::from_vec_col_major` stays
unambiguous for the default `Dynamic` representation.

**View parameterization status: implemented as representation markers.** Both
view types now carry `D` (`TypedTensorView<'a, T, R = DynRank, D = Dynamic>`,
same for the mutable view), and `Host` owners produce `Host`-marked views. The
buffer field stays the concrete union (`TensorStorageRef`/`TensorStorageRefMut`)
rather than a `D::Buffer` associated type, and that is a deliberate, measured
decision: a generic-associated buffer projection is invariant in `'a`, so
`TypedTensorView<'a, ..>` stops being covariant and `TensorRead<'long>` no longer
shortens to `TensorRead<'short>`, which breaks the whole borrowed-read surface
(`TensorBackendOps::add_read`/`sub_read` unifying two reads, views reborrowed out
of longer borrows, downstream backends). Two attempts to do the split that way
compiled their own bodies and then produced a wall of `lifetime may not live long
enough` errors outside the view. Keeping the concrete buffer means the ranked
view impls could simply become `D`-generic with no body changes at all.

What `D` therefore guarantees today, and what it does not:

- A `Host`-marked view is only ever built by `TypedTensorView::from_host_slice` /
  `TypedTensorViewMut::from_host_slice` and by
  `TypedTensor<_, _, Host>::as_view`/`as_view_mut`, so it borrows a host slice and
  carries no retained region in practice. Its `as_host_slice` /
  `as_host_slice_mut` are infallible (documented with the construction
  invariant). `Gpu`/`Dynamic` views use the union buffer and the runtime
  representation checks.
- The union `root: Option<GroupReadView>`/`Option<GroupWriteView>` field still
  exists on the view type, so D3's "never carries an optional full group
  descriptor" holds for a host view at runtime but not as a type-level absence of
  the slot. Making the slot structurally absent needs either the GAT split (cost:
  lifetime covariance, as above) or a descriptor-free host arm in a concrete
  buffer enum (cost: the public `TensorStorageRef` gains the rank parameter and
  its match sites change). Neither is required for the behaviour, so both are
  recorded as the open item rather than adopted.
### D2. Allocation ownership, recycling, groups and extraction

For a plain host tensor, adopt a `Vec<T>` directly; no group, `Arc` or extra
metadata box is required. Host payload variants distinguish:

- Plain vector: normal element destruction and vector deallocation.
- Pooled vector: vector plus the existing weak return target. Final release
  returns it once if the pool lives, otherwise destroys it normally. No session
  admission or execution-resource lock is taken in the drop/recycle path.
- Managed host allocation: explicit deallocator/root handle for supported
  custom, pinned or retained-subregion storage. Indirection is allowed here
  because the allocation requires it, not for every ordinary host tensor.

Do not reduce pooled output to a bare vector and lose reuse. Do not reinterpret
an imported allocation as `Vec<T>` unless capacity, alignment, allocator and
original destruction layout are proven compatible. Complex/real owning
reinterpretation, if retained, preserves the original allocation's deallocator;
a view reinterpretation alone never changes ownership or drop type.

GPU payloads retain provider/domain identity and completion state independently
of groups. Dropping the backend before its output must not invalidate storage.
An ordinary tensor does not embed a group, but a managed payload may retain a
root or an exclusive region claim when that is the actual source of its storage.

**Group transitions** (semantic operations; not a new mandatory public facade):

1. Promotion consumes an owner into a group. It may allocate retention metadata,
   never copy tensor elements just to attach the owner. Reuse existing provider
   roots rather than installing a second deallocator.
2. A borrowed binding keeps the source lifetime. It cannot become an owning or
   asynchronous retained binding just by calling `bind(view)`. Async/detached
   execution must take a retained claim or finish before the borrow ends.
3. Independently retained read regions hold the required root pin and access
   claim; they prevent conflicting mutation until released. Existing checked
   disjoint owning-region claims remain valid; root sharing is not write access.
4. Extraction into an ordinary owner requires compact layout, correct dtype/
   representation, an exclusive claim and no conflicting outstanding access or
   pending completion. Reject invalid extraction without consuming the group's
   remaining ownership. A compact subregion may retain a managed root/claim;
   it must not pretend to own/free the entire backing vector.
5. Extracting a noncompact region requires an explicitly requested same-placement
   copy, not a silent conversion to an owning strided tensor.

A single retained AD value need not create a multi-binding group. Use groups
where multiple bindings/claims require them, not as a replacement mandatory
wrapper around every retained scalar tensor.

### D3. Views and small-rank metadata

A view carries access authority and one validated layout. Host access uses a
bounded pointer/capability with a Rust lifetime; it never carries an optional
full group descriptor. Provider access borrows allocation identity/capability,
not a copied root descriptor. Group-derived and mapped views borrow the guard
that supplies their authority.

Reuse `Rank<N>` arrays and the existing inline capacity of eight for `DynRank`
shape/stride storage initially. Construct transformed metadata directly in its
final container. Common small dynamic-rank transforms, including a rank-2
complex view becoming a rank-3 real view, must not allocate metadata. The #1917
target covers immutable **and mutable** views; it is not narrowed to static
rank. Arbitrary rank may spill. Do not introduce a custom compact-metadata
framework or promise a universal byte size before integrated measurements.

An owner stores shape only; `as_view` computes compact strides in O(rank).
An accessor cannot keep returning a reference to a nonexistent stored layout:
return a layout value or borrow shape plus a compact-layout tag. Static-to-erased
operation inputs borrow shape/stride slices or use that compact tag; they do not
first allocate an owned dynamic-rank view (D7).

For mutable views, validate reachable bounds, alignment and injectivity, using
checked signed offsets. Do not form a full backing `&mut [T]` when disjoint
logical views have overlapping envelopes. Preserve the currently checked split
cases; this redesign does not promise a solver for arbitrary interleaved-region
disjointness. A new even/odd split API needs its own proof and focused test.
Mutable views are not `Clone`/`Copy`; immutable reborrows constrain further
mutation in the usual Rust way. Empty, negative-stride, singleton and ZST cases
must preserve bounds, reference validity and correct element destruction.

Rank-changing operations do not rely on unstable `[T; N + 1]` expressions.
Return `DynRank` where the output rank is computed at runtime (including the
ordinary real-view API); an independently requested checked rank conversion is
available. Fixed-rank APIs may preserve a rank only when the signature proves it.

### D4. Construction, host usability and initialization

| Surface | Contract |
|---|---|
| `from_vec_col_major(shape, data)` | Host or default-host adoption; validate shape/length/overflow in O(rank), no data copy |
| `from_vec_row_major(shape, data)` | Explicit host import/reorder; initially `T: Clone`, with documented O(elements) copy and allocation |
| `as_slice()` on Host | No copying: compact owners expose elements; views reject noncontiguous layouts |
| `Clone` on `Host` owners | Deep copy to a plain host allocation for `T: Clone`; never retains a shared mutable owner |
| `Index`/`IndexMut` on Host owners/views | Usual out-of-bounds panic contract, alongside checked `get`; no device mapping or dispatch |
| `to_col_major()` on Host views | Explicit sequential data-container copy for `T: Clone`, including noncompact views |
| Session allocation/materialization | Uses the entered backend, its pool and policy; same-placement output |

Default `Dynamic` tensors do not implement `Clone` or infallible indexing that
could silently download/map a device value. Specialize/check host access first.
Immutable views have no `IndexMut`; mutable access still requires exclusivity.

Host clone/import/compaction are a narrowly specified **container-copy** boundary
for #1903, usable without execution registration. They do not add a host numeric
library. Reuse generic strided traversal where available; no independent GEMM,
reduction, parallel pool or provider dispatch is introduced. Managed tensor
materialization still uses a borrowed session and the existing optimized copy
implementation. Update the old rule prohibiting all data-moving convenience
methods to name this specific exception rather than silently violating it.

There is no safe `empty(shape) -> TypedTensor<T>` exposing uninitialized `T`.
Internal full-overwrite destinations are `MaybeUninit`/uninitialized leases;
only complete initialization yields a tensor (D11). Zero allocation is a
separate initialized path with dtype-appropriate zero semantics.

Constructor examples use `[2, 3]`: column-major `[1,4,2,5,3,6]` and row-major
`[1,2,3,4,5,6]` both describe rows `[1,2,3]`, `[4,5,6]`. Assert logical entries,
including `[1,0] == 4` and `[0,2] == 3`. Flat data does not reveal intended order.

### D5. Conversion, transfer and mapping are separate

| Operation | Meaning |
|---|---|
| `Host/Gpu -> into_dynamic()` | Move/re-tag; no shape, rank or data change |
| `Dynamic -> into_host()` / `into_gpu()` | Checked representation narrowing; no target device argument or movement; failure retains the source owner |
| `DynRank -> into_rank::<N>()` / fixed rank `-> into_dyn()` | Rank-only metadata conversion; keep representation, allocation and dtype |
| Dtype erasure / checked typed recovery | Keep allocation and rank policy explicit; native erasure only for native scalar support, external erasure through its existing explicit adapter |
| Session/provider `upload` / `download` | Explicit transfer in **both** directions, with typed errors and destination execution authority |
| Session/provider `map_read` / `map_write` | Return an owning mapping guard; borrowing `guard.as_view()` / `as_view_mut()` yields a Host view |

There is one representation conversion per direction, including corresponding
borrowed-view conversions. No synonyms such as both `try_into_host` and
`into_host`. For shape/rank/dtype conversions that consume an owner, failure
must return it with the typed cause rather than silently destroy the input.

Representation narrowing must not perform a hidden GPU-ordinal check: `Gpu`
does not encode an ordinal. Execution validates provider/device/allocation-domain
compatibility. Mixed-representation operands use an explicit borrowed widening
to `Dynamic`, not hidden materialization or transfer.

Mapping may synchronize and can fail or be unsupported. Read mapping borrows
shared access; write mapping requires an exclusive owner/claim borrow. A temporary guard must
not be dropped while its returned view is alive: the API returns the guard,
not a view borrowing a guard created inside `as_host_view(&self)`. Mutable
mapping excludes device reads/writes until publication. Unmap/finish makes host
writes visible under the provider's ordering contract; drop preserves safety
and retention even if explicit finish reports an error. No implicit download
fallback. A mapped device owner remains `Gpu`; the mapping guard is not a
representation conversion.

### D6. Canonical session entry is fallible

Keep explicit session propagation. The proposed boundary shape is:

```text
with_backend_session<R: Send>(
    &mut self,
    f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
) -> Result<R, SessionEntryError>
```

The trait method is `Self: Sized` (as a host entry, not an object-safe operation).
The callback may itself return a typed operation result; that result is not
flattened or disguised as admission failure. Eager/runtime wrappers preserve
its source chain. This changes the canonical API, not a parallel `try_*` shim.

Resource-permit contention/reentry, executor-entry failure, resource-lock poison
and incompatible declared context return typed errors before calling `f`. Use
nonblocking permit acquisition with a typed busy result; existing eager-owner
mutex serialization may still wait before admission. Do not hold a permit while
waiting for that owner mutex. Guard
cleanup on error/unwind releases permits and resources. Do not silently skip
admission, rerun `f`, or retry a partially executed operation through a fallback.
Provider failure after enqueue remains an execution/completion error, not a
claim that nothing ran.

Public operations and registered extensions borrow the session. Independent
user-owned tensor kernels need no session; they may use their own scheduling.
Calling managed APIs from them still requires the explicit boundary. Preserve
named compiled/eager/backward entry points and supported native-context regions;
never nest an owner fallback inside a borrowed session.

Admission guards track tenferro-known reentry before a blocking lock where
needed. Keep the proven eager order (backend-owner lock before admission and
resource access); do not acquire a domain permit and then wait for that owner
lock. Existing `enter_or_reuse` reuses entered authority through delegation;
it is not permission for an arbitrary caller to open another public session.

**Implemented mapping (#1938 session phase).** `BackendSessionHost::with_backend_session`
and `with_backend_session_cached` return `Result<R, SessionEntryError>`; there is
no infallible sibling. `SessionEntryError` (in `tenferro-tensor`) has five
variants, all reported before the callback runs: `Reentered` (a CPU execution or
portable session guard is already active on this thread), `Contended` (a
resource admission cannot wait for, today a caller-managed CPU domain already
executing), `IncompatibleContext` (a backend witness that does not match the
active execution scope), `ResourcePoisoned` (poisoned arbiter state) and
`Executor` (typed executor-entry failure). Operation families carry it as their
own source-preserving variant: `tenferro_tensor::Error::SessionEntry`,
`tenferro_runtime::Error::SessionEntry` and `tenferro_einsum::Error::SessionEntry`,
so a caller writes `backend.with_backend_session(|s| op(s))??`. The portable
`with_session_entry_guard` used by CUDA/WebGPU now rejects nesting in every build
profile rather than through a debug assertion. `CpuBackend::install` became
`Result`-returning for the same reason. Best-effort recycling
(`TensorBuffer::reclaim_buffer`, runtime last-use reclaim) drops the tensor
instead of pooling it when no session can be admitted.

**Deviation: cross-thread contention waits.** The text above asks for
nonblocking permit acquisition with a typed busy result. The implementation keeps
the arbiter's FIFO wait for a CPU permit held by *another* thread, by maintainer
decision on this branch: every default `CpuBackend` requests the whole allowed
CPU set on the process-global arbiter, so a busy error would make any two
concurrently used backends (including parallel test threads) fail instead of
serializing. The lock-order rule is what makes waiting deadlock-free: the eager
owner lock is taken before admission, and no permit holder waits on that lock.
Only non-waitable states are typed errors: same-thread reentry, a busy
caller-managed domain (it has no queue), a scope-witness mismatch, poisoned
arbiter state and executor-entry failure. Request-id exhaustion stays a wait
(ids restart once every permit drains). The design council recorded
nonblocking `Busy` as one valid policy, not the only safe one.

**Lock-order enforcement (#1946 F1).** The rule above was stated but not
checked: a thread could hold a session of one eager runtime (or a CPU execution
scope) and block on another runtime's owner lock. The eager owner lock is now
taken only when this thread holds no session, portable session guard or CPU
permit; otherwise entry fails with `Reentered` before any wait. Inside a shared
execution scope the owner lock is tried, not awaited, and a busy owner is
`Contended`. Cross-thread permit waits keep the FIFO policy above.

**Unwind reuse.** The engine-resources lock is still recovered after a callback
unwinds: `BufferPoolLoan` restores in-flight pool accounting on unwind, so the
next session sees a consistent pool, while pool introspection keeps reporting the
poison. Arbiter poisoning is not recovered; its lists cannot be trusted.

### D7. Compact erased operation boundary and bounded native access

**One numerical implementation:** keep object-safe `BackendSession` operations,
but change their input/output descriptors together with storage. Thin typed
facades are permitted and necessary for output typing; a second generic kernel
SPI is not required. This is a design choice, not proof that generics always
increase code size or lack capability value.

- `TensorRead`/`TensorWrite` borrow data capabilities and a `LayoutRef`: either
  compact shape or shape/stride/offset references. Dtype dispatch occurs once
  per operation, not per element. Erasing a static-rank borrow does not allocate
  a `DynRank` owner or clone an `AllocationGroup`.
- The erased owning result is a compact `Tensor`. Same-dtype typed facades
  recover `T` and return `Host`, `Gpu` or `Dynamic` as promised, without copying
  data or reconstructing group state. A same-representation specialization
  validates session compatibility before executing.
- Result **rank and dtype are operation-specific**: general tensordot returns
  `DynRank`; comparisons return Bool; complex absolute value is real; linalg
  results follow their declared result types. Do not cast every result back to
  the input `T, R` and discover mismatches after `_into` writes.
- General einsum may remain on the erased surface. Unsupported external scalars
  fail typed dispatch; independently storing such scalars does not implement
  their kernels.

**Native access decision:** no universal `TypeId` capability registry. Replace
publicly implementable raw-pointer/marker pairs with a single opaque
`NativeSessionRef<'s>` returned by an object-safe session method. It represents
one exclusively borrowed native session, not a catalog of arbitrary services.

The token has private fields: backend-leaf marker, pointer, lifetime tied to the
`&mut self` borrow, and thread-affinity marker. It is not `Clone`, `Copy`,
`Send`, or `Sync`. Its backend-leaf construction is **unsafe**: the implementer
must prove marker/type correspondence, alignment, live exclusive access and
borrow duration. A marker names a lifetime family; it does not prove that a
borrowed session is `'static` or suitable for `Any`.

Keep a small safe visitor in each backend leaf for recovering its concrete
session/services. It checks the exact marker and performs the audited recovery
inside that leaf; the recovered reference cannot outlive the token's borrow.
Safe custom sessions may return `None` or forward a borrowed token from a
standard delegate; they cannot fabricate one by returning a matching `TypeId`.
This reduces the unsafe implementation boundary; it does **not** eliminate it.
Higher-level operation crates use safe visitors, not raw casts. Existing visitor
names may be retained with this new contract; renaming them adds no safety.

A wrapper changing operation dispatch must not use a delegated native token to
silently bypass its override. Native resources and selected algorithms/providers
are distinct: pass the selected provider and effective execution context into
standard algorithms explicitly (D8). Thread policy cannot be reset by extracting
a delegate's capability.

**Implemented mapping (#1938 session phase).** `BackendSession::session_type_id`
and the `unsafe fn session_data_mut` raw-pointer pair are gone. Safe code could
implement both, so a custom session could claim a leaf's marker and hand the
leaf visitor a forged pointer. They are replaced by
`BackendSession::native_session(&mut self) -> Option<NativeSessionRef<'_>>`,
defaulting to `None`. `NativeSessionRef` (in `tenferro-tensor`) has private
fields (marker `TypeId`, pointer, `&'s mut` borrow marker, `*mut ()`
thread-affinity marker), no `Clone`/`Send`/`Sync`, and one `unsafe` constructor
whose contract is marker/type correspondence. Each leaf marker is crate-private,
so only CPU, CUDA and WebGPU sessions create tokens carrying their marker, and
their safe visitors (`with_cpu_exec_session`, `with_cuda_exec_session`,
`with_webgpu_exec_session`, names retained) recover the session with a marker
check plus one audited cast. Custom sessions get `None` by default, so a
dispatch-overriding wrapper does not expose its delegate's native services
unless it deliberately forwards the delegate's token. Compile-fail doctests pin
the forged-construction (E0133), escaped-borrow, `Send` (E0277) and `Clone`
(E0599) rejections.

On the compact boundary, the input half already held from the ownership phase:
any `TypedTensor<T, R, D>` view erases through `TypedTensorView::into_tensor_read`
without allocating small-rank metadata or promoting host storage. The typed
facade (`TypedTensorSessionOpsExt`) stays on the `Dynamic` representation and
returns operation-specific result types: complex `abs` now returns
`TypedTensor<T::Real>` instead of executing and then failing the recovery to
`T`; `compare` already returned `TypedTensor<bool>`. A `Host` owner reaches the
facade through the zero-copy `into_dynamic()` rather than through a second,
representation-parameterized facade family.

### D8. Parallelism, delegation and placement

Reuse the existing execution context and pool, not a second scheduler. Separate
three facts: selected implementation's parallelism declaration, active outer
fan-out, and whether inner tenferro work is permitted. A sequential child may
forbid inner parallelism while still having `outer_active = true`; collapsing
those facts into `Sequential` would admit an unsafe OpenMP delegate.

| Implementation declaration | Under active outer fan-out |
|---|---|
| Sequential, concurrent-call safe | Allowed |
| Same tenferro pool, concurrent-call safe | Allowed same-pool nesting, unless the child policy requests sequential work |
| Independent runtime, potentially internally parallel (default for external BLAS/LAPACK) | Rejected before dispatch |

Outside outer fan-out, external single calls and whole-batch vendor calls are
allowed. Merely running on a Rayon worker is not fan-out. Unknown external
thread counts remain unknown; user/provider declarations of sequential,
concurrent-call capability are explicit configuration, not deductions from
environment variables. No thread setters, second pool or external-worker budget
claim. Completed Rayon packing may precede external provider execution.

Provider selection precedes checking this table. An algorithm's reachable
delegate contributes its contract; a top-level einsum declaration cannot hide
an independent-runtime GEMM inside outer batching. Custom standard-algorithm
reuse passes the custom GEMM provider to that algorithm, not merely its native
CPU delegate. Built-in default operations and custom provider calls share the
same validation and kernel implementation.

**Placement:** remove promises about placement/budgets of uncontrolled external
workers. Retain `CpuSet`/topology where actually needed for tenferro-owned
worker affinity and existing resource exclusion; do not delete a CPU-set type
merely because external placement is unprovable. Simplify public
`CpuPlacementGuarantee` and admission modes accordingly. Resource/domain
identity and permit conflicts are independent of placement promises and survive
that simplification. A supported managed-affinity contract cannot vanish as an
incidental consequence of this change.

**Implemented mapping (#1938 session phase).** The three facts are separate:
the implementation's declaration is its `CpuProviderExecutionCapabilities`
(`accepts_mode(Outer)` is "sequential, concurrent-call safe"); active outer
fan-out is `CpuExecutionContext::is_outer_fan_out_lane`, set only by
`submit_outer` and the public `with_outer_lanes`, never inferred from running on
a Rayon worker; whether inner work is permitted is the lane's `Sequential`
mode, which `enter_or_reuse` does not widen inside a lane. Rejection happens at
submission: `submit_outer` takes an `OuterFanOutChecked` proof produced by
checking every reachable delegate, so an independent-runtime GEMM fails before
any lane runs. A contraction reached from a lane checks its general, GEMM and
layout slots before dispatch whatever the capability policy, which makes the
check transitive for algorithms run per item. Packed LU/solve batching now
fans out through `with_outer_lanes` instead of a bare `rayon::scope`.
A custom `CpuGemmProvider` installed with `with_provider_bundle` is what
standard tensordot and einsum reach (tested); a wrapping session can override
an operation and opt in to forwarding its delegate's native token (tested).

Placement: the exact/advisory `CpuPlacementGuarantee` declaration and the
`PlacementNotEnforceable` rejection were removed. Cooperative-domain bundle
validation checks the thread count only; provider-created threads are
documented as unmanaged. Managed CPU-set pinning (`TenferroDomainVerified`),
the arbiter's CPU-set exclusion, `AllAllowed` identity validation (now
unconditional) and the caller-managed executor rule are unchanged.

### D9. Overridable batch strategy and thresholds

Use backend defaults, a scoped session override and an optional per-operation
policy. Precedence: per-operation > scoped override > backend default. Restore
the prior policy on normal return, error and unwind; no process-global mutation.

Strategies: Auto; sequential batch/sequential items; outer parallel
batch/sequential items; sequential batch/provider-managed items; whole-batch
vendor call. Auto thresholds distinguish per-item work, total batch work and
chunk granularity; all are overridable. Keep existing justified defaults until
the bounded tuning pass, rather than adopting a universal `MNK < 400` cutoff.

Resolve the effective policy with provider, dtype, layouts, output and delegated
contracts before writes. Forced strategy overrides a heuristic, never a safety
constraint or missing provider route. Report a typed error, not silent fallback.
Auto may choose a supported route before execution. Policy/provider changes
invalidate a prepared execution strategy (D11).

Use lane-local scratch and disjoint outputs for outer batching; sequential
items cannot accidentally invoke inner provider parallelism. Allow same-pool
nested work under its separate explicit policy. Whole-batch vendor dispatch has
no enclosing tenferro fan-out. Connect ordinary compatible batched contractions
to existing MKL batch bindings, move the grouped cutoff into the policy, and
keep vendor-specific strided-batch binding work optional. All-batch elementwise
contractions are classified before generic GEMM lowering, not lowered to a loop
of 1x1 GEMMs.

**Superseded by #2004.** The host `CpuBatchPolicy` / `CpuBatchStrategy` /
`CpuBatchThresholds` vocabulary described here was removed once cpueinsum gained
its grouped plan form: lane selection and vendor batching now belong to
cpueinsum and tlinalg, and tenferro passes one parallelism token (the selected
pool with a budget, or `Sequential`).



**`Auto` inside an entered session (#1898 follow-up).** A multi-threaded
session no longer runs every `Auto` batch serially. When the entered Inner
context owns more than one Rayon thread and the provider may run inside a lane
(faer; BLAS is excluded by its declared scheduling), `Auto` fans a strided batch
or a grouped job list out over the context's own lanes, each lane Sequential.
The gate is a cost model, not the item-count thresholds alone: by default one
GEMM item costs about 50 ns plus one nanosecond per 16 multiply-adds, and a
lane needs at least 8 us of estimated work, so short batches keep one provider
call. The three parameters are `CpuBatchThresholds` fields
(`lane_item_overhead_ns`, `lane_muladds_per_ns`, `lane_min_work_ns`), so they
follow the same backend default < scoped < per-operation precedence as the
item thresholds (#1946 F3). Allocating and caller-owned destinations take the
same decision (#1946 F2).
Strided batches split into one contiguous batch chunk per lane over disjoint
output storage. Grouped jobs run one contiguous chunk per lane in one provider
call when the nonempty jobs' output starts increase (which, with the
validator's pairwise disjointness, makes chunk ranges disjoint), and otherwise
one call per job. Outside a session the existing executor fan-out and its
thresholds are unchanged. Independently, faer runs products below 2^20 real
multiply-adds (complex elements weigh four) sequentially inside any parallel
context, so a multi-threaded backend is not slower than one thread on small
GEMMs.

### D10. Crate placement: one public tensor family, no reverse edge

| Crate | Target responsibility |
|---|---|
| `tenferro-tensor-core` | Rank/layout validation, scalar tags/identity, backend-free scalar metadata; no second public tensor family |
| `tenferro-tensor` | Canonical generic owner/views and Host/Gpu/Dynamic payload contracts, erasure, groups, backend/session SPI |
| `tenferro-cpu-basic` | Existing CPU pool and checkout implementation |
| `tenferro-cpu`, `tenferro-gpu` | Backend resources, execution, native visitors and provider integration |
| Operation families, runtime, AD | Existing semantic/algorithm/graph ownership; use the shared execution path |

This deliberately changes the old host-container split. Move/merge HostTensor
and external host-container functionality into the canonical family in
`tenferro-tensor`; migrate callers instead of retaining a core-owned alias that
would need a reverse dependency. Core scalar traits that currently mention
HostTensor or core Tensor must lose those storage-conversion methods; equivalent
storage-facing methods belong in `tenferro-tensor`. Keep pure scalar tags and
validation below it. External erased storage/adapters are migrated, not removed.

#1946 F8 completes this: `HostTensor` / `HostTensorView` are removed with no
alias, `define_scalar_set!` members, `DefaultScalars` and `ErasedHostTensor`
carry `TypedTensor<T, DynRank, Host>`, and `tenferro_tensor::core` re-exports
metadata only.

No new crate, no blanket ban on legitimate existing dependencies between
operation families/runtime. The table assigns ownership, not a fictitious linear
Cargo dependency chain. Public tensor data use needs the tensor crate but no
backend construction, session or AD registration.

### D11. Output/scratch protocol, plan lifetimes and GPU ordering

The operation sequence is:

```text
validate logical inputs/output and select implementation
 -> resolve route, policy, each required packing, workspace and alias rules
 -> acquire disjoint output/scratch leases
 -> execute and establish output initialization/completion
 -> publish output; release/recycle scratch under its lifetime contract
```

Replace the pool-borrowing uninitialized output guard with an owning checkout
lease tied to the existing shared pool state. It carries the allocation and its
single-use accounting token, not `&mut BufferPool` for its whole lifetime.
Checkout/drop lock pool bookkeeping briefly; never keep that lock across kernel
execution, user code or provider waits. Thus output and multiple scratch buffers
can coexist. Reuse current pool counters and weak recycler rather than a new
allocator framework. Successful initialized output transfers its return target
once; error/unwind discards or returns storage without publishing uninitialized
`T`, leaking accounting, or reading stale values. General non-Copy container
cloning separately tracks initialized elements for destructor safety.

Determine which operand actually needs packing. Temporary packed inputs are
internal buffer/layout views, not public grouped tensors. `_into` uses the same
output-writing implementation as allocating execution, and never allocates a
full temporary result merely to copy it into the destination. Overwrite does
not read old output; accumulation/destructive operations retain distinct
contracts. Partial numerical failure is not retroactively treated as validation
failure with unchanged output (see §2).

Prepared operations retain useful validated route/layout data with validity keys
covering operation/configuration (including axes and scalar flags), dtype,
shape/strides, provider/configuration generation, device/domain and batch policy. Do not cache borrowed pointers beyond execution. Reprepare
when those facts change. Address-sensitive preparation binds separately to a
live allocation/access capability. Existing prepared binary einsum metadata
must not be thrown away before backend GEMM analysis. Ordinary eager execution
has a cheap uncached path. Runtime/compiler caches remain runtime-owned;
provider plans and pools have explicit owners, bounded defaults, configure/
clear/stats controls. Sessions borrow them; they do not reset them on exit.

GPU views preserve allocation identity/address capability and ordering validity.
Reusable addresses are conditional on provider generation, stream and access
rules, not merely pointer stability. Zero/initialization commands participate
in CubeCL write publication; do not delete synchronization without a replacement
ordering proof. Async scratch/output/root pins survive host callback return
until completion. A borrowed device buffer is either retained under the
existing claim protocol or execution completes before its Rust borrow ends;
returning from an enqueue is not permission for conflicting host/device access.
Mappings, views and early backend destruction obey the same retirement protocol.

**Implemented mapping (#1938 session phase).** The uninitialized output is the
owning `PooledUninitOutput` lease (pool handle plus single-use token), so
operand packing draws on the session pool while the destination is checked out.
The allocated-dot uninit path therefore takes the direct plan or canonical
packing into the uninit destination, with direct and canonical plans cached
under separate kinds. The canonical fallback packs only an operand that needs
it: a permuted view that is already compact column-major and unconjugated is
borrowed. `_into` avoids full temporaries on these paths:
- all-batch overwrite products are written in place;
- the N-ary einsum's last contraction step writes into the destination when its
  result labels are the output labels;
- the elementwise fallback reclaims its staged result;
- reductions write uninitialized storage.

GPU raw `DeviceBytes` need no retirement event: they drop on the capturing
thread and CubeCL keeps one pool per stream. The cross-thread cuTENSOR
workspace keeps event retirement. Residuals:
- a prepared einsum GEMM analysis slot (needs a slot-allocation contract);
- skipping the LU for an empty solve right-hand side;
- GPU address memoization keyed on pointer stability.

### D12. AD/runtime integration and the A2 decision

Reuse #1929's borrowed `EagerSession`, runtime identity checks and compiled
regions, together with existing numerical/AD tests. Ordinary synchronous calls
construct no trace or retention group. Traced/AD calls add saved-value lifetime
and semantic recording while calling the same numerical implementations.
Backward reuses a session per compatible execution region; gradient accumulation
borrows that session. Cross-provider/owned-only paths remain explicit top-level
regions, never hidden nested fallback.

**A2 is deferred, not solved.** Do not add `with_evaluation_scope` or replace the
eager backend ownership with two speculative same-domain handles in this target.
The handoff's mutex/permit/`Send` conflict is real. A second handle by itself
proves neither lock ordering nor single ownership of pools/caches. Contention
cannot be made safe by running the same callback plainly after failed admission.
Use the existing borrowed session at named boundaries; keep A2's audit allowlist
at zero. If the final bounded performance pass identifies evaluation-wide entry
as a dominant residual, record A2 for a separately specified ownership change.
That follow-up needs a concrete lock/permit graph and cleanup/reentry tests; no
measured win from the reverted experiment is claimed here.

Preserve callback-local `no_grad`/`capture_trace`: thread-local guards do not
cross worker dispatch automatically. Any migration of the retained tensor-owned
`solve` must explicitly preserve its calling-thread mode semantics or document
and test a revised public mode contract; renaming it is not enough. Preserve
consuming in-place FFT's ownership guarantee. Saved values and factors survive
dropped forward handles; unrelated live leaves must not become a mandatory cost
of ordinary tensors. #1803 residual tuning remains separate from AD correctness.

**Implemented mapping (#1938 session phase).** Backward and JVP stage residuals,
bindings and the seed in one backend session. Gradient duplication and storage
share one session. Untracked eager results are retained without an allocation
group, and the weak value/gradient registries sweep dead entries at their
growth point. The owned and borrowed eager dispatch tables are one table.
Revised mode contract (maintainer decision): `with_execution_session`,
`with_eager_session` and the extension execution context carry the calling
thread's `no_grad`/`capture_trace` depths into the callback for its duration,
also when the managed executor runs it on a worker; a guard started inside the
callback stays local to it, and the worker's counters are restored on return
or unwind. Before this, an outer guard reached per-op execution but not a
session callback, so binary einsum (a session fast path) and N-ary einsum
disagreed under `no_grad`. With the contract in place the expanded eager
einsum runs its whole program in one borrowed session
(`EagerSession::apply_standard_op`, `EagerSession::backend_session`). A2
remains deferred.

## 4. Focused implementation evidence

These are correctness checks for the affected boundaries, not a new benchmark
campaign or a demand to execute every backend before selecting the design.
Reuse existing tests where they prove the stated property.

| Boundary | Evidence needed with its implementation |
|---|---|
| Generic data storage | Downstream non-preset scalar, fixed/dynamic rank, no registration; a non-Copy drop-counted type; Host clone/index/compaction |
| Metadata/erasure | Small-rank dynamic and static views, real reinterpretation, operation input erasure; no metadata allocation in the #1917 cases; arbitrary-rank spill works |
| Ownership/pooling | Final-owner recycle exactly once, pool/backend dropped first, custom deallocation, group promotion/extraction, disjoint claims and rejected aliasing |
| Views/mapping | Bounds/alignment/negative/empty layouts, mutable injectivity, compile-fail escaped borrows/guards and duplicate mutable access; real-device mapping where supported |
| Uninitialized output | Coexisting scratch/output, initialization proof, error/unwind accounting, no stale read, numerical output and `_into` sentinel preservation on validation failure |
| Session/native bridge | Typed admission before callback, reentry without deadlock, unwind reuse; native token cannot be forged in safe code or outlive its borrow; delegate preserves context |
| CPU execution | Sequential children, same-pool nested Rayon, external single/batch with no fan-out, rejected incompatible nesting before dispatch through delegates |
| Customization | One public delegation/composition test and one custom GEMM-through-standard-tensordot test; one real external-provider smoke with recorded startup config |
| Prepared/GPU | Changed-layout/provider/policy invalidation, resident views without hidden transfer, output/scratch retirement and device ordering |
| AD/runtime | Existing numerical/oracle and residual tests, eager/runtime identity, callback modes, retained solve/FFT contracts |

Add one small public example per #1938 workload family and the independent
packed-view user-kernel example. Examples accompany implementation, compile and
run, and assert actual values/reconstruction, not only shapes. Feature-gated GPU
examples require real-device execution for device-success claims. Each example
explains defaults versus specialization and the required session propagation.
A proposed-signature sketch in this record is not a passed public doctest.

## 5. Delivery and performance scope

One working branch reusing `e55c0e17a`; one final tenferro-rs PR after integration
and a bounded performance/bottleneck session. Milestones are not PR boundaries.
Keep compilation and focused safety/numerical checks during implementation.
Begin with storage, borrowed-erased dispatch and session output **together** in a
small vertical slice, then migrate GPU/retention, operation families and runtime/
AD. Do not establish a Host-only path whose adapters recreate old groups later.
Remove obsolete APIs and update callers/docs as their replacement lands; do not
mechanically delete legitimate low-level entry helpers inside named boundaries.

Use existing tenferro-benchmark #107 and strided benchmark #41 infrastructure.
Pin the #1929 handoff as the available post-migration source baseline; if a later
integrated-main baseline is available, record that exact revision. An intermediate
merge to main is not required. The handoff records benchmark branches that are
local-only and historical results that do not measure the final eager migration;
do not silently call those comparable baseline measurements.

Choose one or two metrics per representative family: packed small matrices;
changing-shape/layout contractions and prepared replay; batched factorization/
solve; resident GPU chains; resource churn/retention; AD execution. Include a
large compute and a bandwidth control. Record matching inputs/layouts, provider,
revision and timing scope. Measure overhead with explicitly configured and
verified **1T** backend/provider operation; separately label requested
multi-thread throughput experiments. Count allocations/transfers separately
from latency when needed. GPU performance claims require actual GPU evidence.

Retain #1917's common-small-rank zero-metadata-allocation target for both view
variants. Small construction should approach its matched data-ownership baseline,
not a portable byte/latency budget. Do **not** require multi-thread session entry
to approach inline 1T latency: Rayon handoff and tenferro bookkeeping are different
costs, as PR #1941 establishes. Missing hardware, noise or historical runs are
reported rather than replaced with invented improvements or a new platform.

Required integration work includes compact CPU/GPU access/retirement correctness,
fallible entry, parallelism/delegation checks, batch-policy overrides and usable
managed execution. Independent vendor-library lifetime tuning (#1924), microkernel
quality (#1900), A2 and #1803 residual optimization are not blanket blockers.
GPU pointer/zero-path optimizations touching D11 must obey D11, but every GPU
performance leaf need not be closed. Keep unsupported LU/pseudoinverse/WebGPU
capabilities from the handoff explicitly unsupported unless separately fixed.

Close on the accepted integrated architecture, focused correctness/required CI,
and the bounded performance pass with recorded fixes and explicit deferrals.
Do not promise a speedup from this document or require universal thresholds.

## 6. Document/rule migration and remaining limits

This design does not silently override implemented docs today. During migration,
update affected records on the implementation branch:

- `docs/design/tensor.md`, `storage-ownership-contracts.md` and
  `docs/storage-ownership.md`: compact payloads, groups, mapping, allocation
  initialization and container-copy boundary.
- `docs/design/exec-session.md`, `explicit-session-boundary.md` and
  `session-oriented-concrete-apis.md`: fallible entry, native visitors, compact
  inputs/outputs, A2's explicit deferral.
- `docs/architecture/tenferro-crates.md`: concrete ownership moves in D10 and
  actual dependency edges. Core must not refer back to tensor-owned types.
- `REPOSITORY_RULES.md`, `PERFORMANCE_TIPS.md` and their routing/checker inputs:
  deliberate exceptions/changes for host containers, placement, constructors
  and execution participation; retain the session-entry audit and typed safety
  rules. Historical `docs/plans/` and freeze evidence stay historical, not edited
  into a false claim that the new representation was previously verified.
- Public rustdoc, tutorials, shipped compute-skill mirrors and design index:
  migrate when their corresponding implementation exists; no legacy spelling
  is retained solely for source compatibility.

The Rust storage projections, compact borrowed-erasure lifetime signatures,
native-token unsafe construction/recovery and allocation leases still need
focused implementation evidence. They are named validation obligations, not
claims that tests have already passed or invitations to silently change
semantics. No mandatory pre-design benchmark matrix, multi-agent review, new
crate or universal capability registry is introduced.

Review verification: a standalone standard-library-only Rust model passed three
checks (default storage of a non-Copy/non-Send scalar, borrowed fixed-rank layout
erasure, object-safe native-token forwarding). Three negative compile checks
rejected an escaped token, sending a token, and overlapping mutable token
borrows. These establish only those Rust type/lifetime shapes, not the actual
backend's unsafe recovery, mapping, pool accounting, numerical behavior or GPU
ordering. No production implementation, performance result or real-device test
is claimed by this revision.
