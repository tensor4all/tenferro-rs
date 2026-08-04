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
- Provider owners now retain physical `byte_len`; typed element counts are
  derived only at borrowed tensor/view boundaries. Allocation constructors
  pass provider byte sizes, including complex and bool representations.
- Runtime and CPU/FFT/linalg managed-buffer test providers now use mutable
  mapping receivers and allocate a fresh same-domain output where an old test
  path previously aliased an owner.
- Alias tests that depended on constructing two owning tensors from one
  cloneable handle were removed or rewritten because the safe owner API now
  prevents that construction; borrowed identity checks remain at provider
  boundaries.
- The top-level `device_ptr(&CudaRuntime, &Tensor)` escape hatch is no longer
  exported. Crate-internal tests use the existing typed interop helper; the
  linalg-facing interop surface remains a separately tracked migration item.
- `DeviceByteBuffer::ptr()` is no longer a public accessor; workspace callers
  borrow its pointer through `with_ptr`, while the owning linalg workspace
  retains the allocation for the complete operation.
- The public CUDA interop functions `typed_device_ptr` and `raw_cuda_stream`
  are replaced by `with_typed_device_ptr` and `with_raw_cuda_stream` callback
  boundaries with non-returning callbacks. Private provider/linalg helpers
  still keep raw values only for the immediate FFI operation and remain a later
  prepared-binding migration.

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
- `cargo test -p tenferro-gpu --features cuda,webgpu --test integration public_surface_contract` — 21 passed, including the byte-length owner contract.
- `cargo test -p tenferro-gpu --features cuda,webgpu --no-run` — passed.
- `cargo test -p tenferro-gpu --features cuda,webgpu --lib domain_and_distinguish_allocations` — 2 passed.
- `cargo test -p tenferro-gpu --features cuda,webgpu --lib --quiet` — 93 passed, 117 ignored.
- `cargo test -p tenferro-gpu --features cuda --test integration --quiet` — 68 passed on the available NVIDIA A100.
- `cargo test -p tenferro-gpu --features cuda --test storage_provider_cuda_progress --quiet` — 2 passed on the available NVIDIA A100.
- `cargo check -p tenferro-linalg --features cuda --quiet` and `cargo test -p tenferro-linalg --features cuda --test integration --no-run --quiet` — passed after making solve RHS duplication explicit.
- `cargo test -p tenferro-linalg --features cuda --test integration --quiet` — 129 passed, 18 ignored.
- `cargo test -p tenferro-linalg --features cuda --test integration -- --ignored` — 18 passed on the available NVIDIA A100.
- `python3 scripts/run-storage-ownership-contracts.py --receipt-out /tmp/tenferro-storage-ownership-receipt.json --diagnostics-json` — all 19 currently active P0/P1/P2/P3/P4/P5/P6/P9 obligations passed; P7/P8 remain deferred.
- `cargo test -p tenferro-gpu --features cuda,webgpu --test integration public_surface_contract` — 20 passed.
- `cargo test -p tenferro-gpu --features cuda,webgpu --test integration cubecl_launch_contract` — 37 passed, including the scoped workspace and stream-pointer contracts.
- `cargo test -p tenferro-ad --features cuda --quiet` — 333 passed, 1 ignored; the two CUDA eager AD tests now pass after routing backend-owned copies through the runtime session.
- `cargo test -p tenferro-ad --features cuda --test integration -- --ignored` — the CUDA f32 fusion chain passed on the available NVIDIA A100.
- After the AD fix, exact `HEAD` `b5fc7d472de34f793456d25edd3fe37e90e905e3` passed `cargo check --workspace --quiet`, both storage design/ledger checkers, the 24-case v2 contract suite, and the active ownership runner (19 obligations; receipt `/tmp/tenferro-storage-ownership-receipt-b5fc7d47.json`).
- The owner source contract rejects `Option<AllocationDomainId>` in both provider owner structs.
- Full GPU integration reached 92/93; the sole failing test is the pre-existing
  `session_contract` trybuild check, with two fixture mismatches for removed CUDA
  symbols (the compiler now emits similar-name help lines), unrelated to
  allocation-domain changes and intentionally not blessed here.

This is a migration checkpoint, not P7/P8 completion: concrete provider root,
claim, prepared-access, and hardware evidence obligations remain for the
selected provider phases. #1555 remains open.
