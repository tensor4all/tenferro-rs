# Issue #1555 P7/P8 owner-borrowing migration checkpoint

Date: 2026-08-04

## Scope and authority

This checkpoint starts only the selected P7/P8 migration boundary after the
completed P1, P2, P4, P5, and P6 work. Parent issue #1555 and child issues
#1563/#1564 remain authoritative. P10 and later phases are not activated by
this change, and the P7/P8 hardware evidence artifacts remain deferred.

The proportional-safety boundary is deliberate: no compatibility bridge,
recovery or quarantine path, cryptographic machinery, or repeated validation
was added. Provider runtime caches and event identities remain provider-owned;
this change only removes tensor-level owner cloning and makes write authority
follow Rust borrowing.

## Design

- `StorageBuffer::Backend` owns one `Box<dyn BackendStorage<T>>`; it is no
  longer cloneable through an `Arc` handle.
- `TensorStorageRef` and `TensorStorageRefMut` borrow the backend owner. A
  mutable backend mapping requires `&mut self`, so host-visible writes cannot
  be manufactured from a read-only view.
- CUDA and WebGPU providers construct owned backend buffers directly. CUDA
  view binding validates the borrowed view without constructing a temporary
  tensor by cloning the allocation owner.
- The concrete `CubeclBuffer` and `WebGpuBuffer` owners are scalar-independent;
  dtype appears only in the `BackendStorage<T>` implementation and borrowed
  tensor/view descriptor, not in the owning provider resource type.
- Provider owners now require a non-optional `AllocationDomainId`; the public
  metadata trait still reports `Some(id)`, while Apple managed-resource state
  remains a separate optional endpoint marker rather than an identity fallback.
- Runtime and CPU/FFT/linalg managed-buffer test providers now use mutable
  mapping receivers and allocate a fresh same-domain output where an old test
  path previously aliased an owner.
- Alias tests that depended on constructing two owning tensors from one
  cloneable handle were removed or rewritten because the safe owner API now
  prevents that construction; borrowed identity checks remain at provider
  boundaries.

## Files and durable contract

- `crates/tenferro-tensor/src/types.rs`
- `crates/tenferro-tensor/src/backend.rs`
- `crates/tenferro-gpu/src/cubecl/{dispatch.rs,mod.rs}`
- `crates/tenferro-gpu/src/webgpu/{mod.rs,memory.rs}`
- CPU, FFT, linalg, runtime provider/test adapters and contract tests
- `docs/design/gpu-backend-design.md`

The GPU design document now records that provider allocations are single
tensor-level owners, views borrow them, and internal provider `Arc` state is
not an alternate tensor ownership path.

## Verification

- `cargo fmt --all -- --check` — passed.
- `cargo check --workspace` — passed.
- `cargo test --workspace --no-run` — passed.
- `cargo test -p tenferro-tensor --lib` — 256 passed.
- `cargo test -p tenferro-runtime --test integration` — 122 passed.
- `cargo test -p tenferro-gpu --features cuda,webgpu --test integration public_surface_contract` — passed after updating the owner-borrowing contract assertions.
- `cargo test -p tenferro-gpu --features cuda,webgpu --no-run` — passed.
- `cargo test -p tenferro-gpu --features cuda,webgpu --lib domain_and_distinguish_allocations` — 2 passed.
- `cargo test -p tenferro-gpu --features cuda,webgpu --test integration public_surface_contract` — 20 passed.
- The owner source contract rejects `Option<AllocationDomainId>` in both provider owner structs.
- Full GPU integration reached 89/90; the sole failure is the pre-existing
  `session_contract` trybuild fixture drift for removed CUDA symbols (the
  compiler now emits a similar-name help line), unrelated to allocation-domain
  changes and intentionally not blessed here.

This is a migration checkpoint, not P7/P8 completion: concrete provider root,
claim, prepared-access, and hardware evidence obligations remain for the
selected provider phases. #1555 remains open.
