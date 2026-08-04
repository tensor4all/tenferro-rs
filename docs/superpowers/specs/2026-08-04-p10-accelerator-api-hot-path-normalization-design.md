# P10 Accelerator API and Storage Hot-Path Normalization Design

Date: 2026-08-04

Status: design complete; P10 ledger rows remain deferred

Authority: #1555, #1566, `docs/design/storage-ownership-contracts.md`
(G1/G4/G6), and `scripts/storage-ownership-contracts.toml`

## Scope

P10 is the final public/API and storage-hot-path normalization after both CUDA
and WebGPU/Apple-Metal use the shared ownership contract. It removes legacy
exports and implicit transfer behavior, gives each provider one coherent
namespace, preserves real provider differences, and proves that final
contiguous and strided traversal contains no per-element storage/provider
work.

P10 does not add a universal accelerator object, emulate unsupported provider
features, or collect final hardware/documentation evidence. It must finish all
product code, checker scripts, benchmark harnesses, and public docs required to
construct the later P13-A candidate.

## Public provider organization

The canonical modules are:

```rust
pub mod cuda {
    pub use /* CudaBackend, CudaRuntime, CudaDeviceId, CudaDeviceInfo,
               discovery, explicit transfer functions, scoped unsafe interop */;
}

pub mod webgpu {
    pub use /* WebGpuBackend, WebGpuRuntime, runtime identity/discovery,
               explicit transfer functions */;
}

pub mod apple {
    pub use /* AppleContext, AppleTransferStats */;
}
```

Provider types keep provider-specific selectors and capabilities. CUDA retains
`CudaDeviceId`; WebGPU retains its native adapter/device selector; Apple keeps
`AppleContext`. P10 does not force them into one lossy device-ID enum.

Root-level exports are limited to genuinely provider-neutral traits/types.
Provider-specific flat aliases, deprecated re-exports, ambiguous
`gpu_available`, fixed-device constructors, and fixed engine IDs are removed.
Examples import through the provider module they use.

Every runtime engine registration requires a caller-selected `EngineId` and
retains distinct engine, allocation-domain, event-domain, and provider-device
identities. Discovery returns concrete provider/device information and never
silently selects ordinal zero.

## Explicit copy and transfer vocabulary

The existing provider-neutral transfer trait remains the common vocabulary but
has no default implementation:

```rust
pub trait TensorDeviceTransfer {
    fn upload_host_tensor(
        &mut self,
        tensor: TensorRead<'_>,
    ) -> Result<Tensor>;

    fn download_to_host(
        &mut self,
        tensor: TensorRead<'_>,
    ) -> Result<Tensor>;
}
```

Using `TensorRead` permits owners and immutable views without creating a second
owner. Every implementing provider must define both methods or explicitly
return a typed unsupported error. CPU identity behavior is implemented
explicitly; it is not inherited from a shallow-clone default.

The final vocabulary is:

- `duplicate` — same-placement fresh allocation and byte-preserving copy;
- `upload_host_tensor` — explicit host-to-provider destination allocation and
  transfer;
- `download_to_host` — explicit provider-to-host destination allocation and
  transfer;
- `map`/host guard — synchronization and borrowed access, not a transfer;
- Apple CPU/Metal endpoint switch — synchronization/map transition, not a
  transfer;
- reinterpretation/view/slice/reshape/transpose — descriptor-only, not a copy;
- numeric cast — a new computed output, not reinterpretation.

Every successful duplicate/upload/download has a fresh allocation identity and
reason-classified allocation/copy counters. No transfer method returns the
source unchanged, uses hidden CPU staging, falls back to another provider, or
materializes an unsupported layout without an explicit named operation.

## Canonical tensor method distribution

- Read-only methods live on `TypedTensorView<T, R>`.
- `TypedTensor<T, R>` and `TypedTensorViewMut<T, R>` delegate through O(1)
  reborrows.
- Mutable operations live on `TypedTensorViewMut<T, R>` and require exclusive
  borrow.
- Consuming conversion/reinterpretation/extraction lives on owners/groups.
- Explicit duplication reads a view and returns a new owner.
- The dtype-erased family mirrors these capability classes without exposing an
  owner projection.

There is no common sized deref target, fake DST, `DerefMut`, COW owner,
`ArcTensor`, or public `&mut OwnedStorage`. `storage_public_api.rs` owns the
canonical method/export inventory and compile contracts.

`as_view()`/`as_view_mut()` preserve `R` and perform no heap allocation,
refcount/provider clone, dynamic layout clone, provider resolution,
synchronization, transfer, or materialization. Explicit dynamic-rank conversion
is the only ordinary rank-erasure boundary.

## Raw interop

Safe generic code has no `device_ptr`, raw handle, or downcast that can escape
a binding lifetime. CUDA-specific interop may remain under `cuda::interop`, but
it is explicitly unsafe and scoped to a live prepared binding/session. Its
rustdoc states synchronization duties, binding lifetime, and post-retirement
invalidity.

WebGPU/Metal does not imitate CUDA pointer interop. Its in-tree extension
boundary remains opaque and session/prepared-binding scoped; no public CubeCL
`Handle` clone or owner-import/export API is exposed.

## Prepared and traversal hot paths

### One resolution boundary

Descriptor validation occurs before `PreparedRead`/`PreparedWrite` is
constructed. Provider preparation and binding occur once per traversal or
launch. The resulting contiguous/strided/device enum variant is the state
authority; no parallel `is_checked`, `is_contiguous`, `is_mapped`, or
`is_writable` booleans exist.

Contiguous host loops use a typed slice or `core::slice::Iter`/`IterMut`.
Strided loops initialize one checked cursor and then perform only:

1. typed load/store at the current proven offset;
2. decrement of remaining elements;
3. incremental stride/carry update;
4. loop termination.

No inner loop performs provider lookup, storage downcast, descriptor-slot
resolution, bounds/range/alignment validation, allocation, synchronization,
flat-index coordinate decoding, or full-rank metadata clone.

Device kernels bind the opaque provider-prepared state once. Kernel launch
maps the output/update domain to the launch domain and does not hide an
unbounded tensor loop in one GPU worker.

## `p10-api-normalization`

Artifact:

```text
crates/tenferro-tensor/tests/storage_public_api.rs
cargo test -p tenferro-tensor --test storage_public_api
```

The test combines compile-pass/compile-fail and rustdoc/public-export inventory
for:

- canonical provider namespaces and constructor/discovery signatures;
- no flat/deprecated aliases or fixed engine IDs;
- no `Clone` on owner/capability types;
- no mutable owner projection or safe raw-handle escape;
- no transfer defaults;
- fresh allocation identity for duplicate/upload/download;
- static-rank owner/view/view-mut parity;
- explicit errors for unsupported provider operations;
- Apple mapping/synchronization distinguished from transfer.

## `p10-element-hot-path-structure`

Artifact and command:

```text
scripts/check-storage-element-hot-path.py
python3 scripts/check-storage-element-hot-path.py
```

The checker has a fixed manifest of canonical final functions rather than a
repository-wide keyword ban. It uses a bracket/comment/string-aware Rust token
scan to inspect:

- contiguous slice/iterator entry points;
- `PreparedStridedIter::next` and mutable counterpart;
- provider bind/launch loops;
- representative CPU and GPU element loops.

It rejects calls to storage/provider resolution, validation, synchronization,
allocation, formatting, coordinate decoding, or owner/refcount operations
inside the marked loop bodies. It also invokes focused runtime counter tests
that compare different element counts and require constant
prepare/map/bind/dispatch counts. This source check is supplemental to runtime,
codegen, and benchmark evidence; it is not a standalone soundness proof.

## `p10-storage-traversal-performance`

Artifacts and command:

```text
docs/testing/storage-traversal-performance.md
python3 scripts/verify-storage-traversal-performance.py \
  --baseline-obligation p1-element-access-baseline \
  --baseline-report docs/testing/storage-element-access-baseline.json \
  --report docs/testing/storage-traversal-performance.md
```

The Markdown report contains exactly one fenced JSON record with schema
`tenferro.storage-traversal-performance.v1`. It records:

- measured source commit and repository-relative benchmark path;
- baseline obligation, report path, and baseline measured commit;
- rustc/cargo versions, target, CPU model, OS, affinity, thread environment,
  features, sample size, warm-up, and measurement duration;
- medians/confidence intervals for contiguous read/write, dynamic contiguous,
  fixed-rank contiguous, representative positive-stride, transpose/reverse
  stride, and empty traversal;
- setup measurements separately from inner-loop measurements;
- result `pass` or `inconclusive` with a concrete reason.

A comparable environment requires the same architecture, compilation target,
single-thread environment, benchmark profile/features, and CPU model family.
On a comparable environment:

- prepared contiguous read/write inner-loop medians must be no more than 10%
  slower than the corresponding direct-slice baseline;
- dynamic contiguous iteration must be no more than 15% slower than its
  baseline;
- incremental strided traversal must not regress relative to the baseline
  logical-order strided case and must show no element-count-dependent provider
  resolution;
- setup allocation/resolution counters must match the separate structural
  contracts.

An incompatible environment or statistically overlapping noisy result is
`inconclusive`, not pass. P10 activation and P13 closure reject inconclusive
performance evidence. No checksum or attestation is required; exact Git
commit, tracked path, and recorded environment are sufficient.

## `p10-static-rank-codegen`

Artifacts and command:

```text
docs/testing/storage-static-rank-codegen.md
python3 scripts/check-storage-static-rank-codegen.py \
  --report docs/testing/storage-static-rank-codegen.md
```

The checker builds a fixed-rank contiguous read reduction and mutable scale
probe in release mode with assembly emission. The report contains one fenced
JSON record with schema `tenferro.storage-static-rank-codegen.v1`, measured
commit, target/rustc/flags, probe symbols, and assembly observations.

Acceptance requires the element loop to be slice-equivalent: no call in the
loop to storage/provider lookup, allocation, validation, dynamic-rank dispatch,
or coordinate decoding. Vectorization is recorded but is not mandatory across
all targets. A missing/ambiguous symbol, incompatible optimizer output, or
unclassified call is inconclusive and blocks activation rather than being
silently accepted.

## Documentation ownership

P10 updates provider quickstarts, capability matrix, namespace rustdoc,
explicit copy/transfer semantics, Apple shared access, synchronization, and
CUDA unsafe interop. Every changed public `Result` API has concrete `# Errors`
and a runnable behavior assertion. Final source-blind product documentation is
owned by P12, but all public names and checker scripts must be stable before
P13-A freezes a candidate.

## Exit and non-goals

P10 is complete only when all four P10 obligations are active and pass, CUDA
and P8 provider contracts remain green, obsolete exports/defaults/raw-safe
paths are physically absent, and performance/codegen evidence is conclusive.

P10 adds no universal accelerator trait beyond the existing capability traits,
no compatibility aliases, no hidden transfer/fallback, no repeated static
validation, no per-element provider work, and no security/attestation or
provider-recovery machinery.
