# webgpu feature: migration state and the remaining decisions

Status: 54 errors -> 9 with `--features webgpu` (10 with
`--features cuda,webgpu`). Everything mechanical is done; the nine left each need a
decision, which is why this is a separate record from the A100 note.

Verification: `cargo check -j 4 -p tenferro-gpu --features webgpu`.

## Done (mechanical, from the 0.11 APIs)

- the runtime generic is gone from the binding types: `Client<WgpuRuntime>`,
  `BufferArg<WgpuRuntime>`, `TensorBinding<WgpuRuntime>`,
  `TensorHandle::<WgpuRuntime>`, `ComplexTensorHandle::<WgpuRuntime>`;
- `launch_unchecked::<T, WgpuRuntime>` / `launch::<T, WgpuRuntime>` are
  single-generic;
- the type queries: `f32::as_type_native_unchecked().storage_type()` is
  `f32::elem_type_native()` and `StorageType` is `ElemType`, with the `Scalar`
  trait in scope;
- `WgpuDevice::DefaultDevice` / `DiscreteGpu(n)` are
  `WgpuDevice::new(WgpuDeviceKind::{DefaultDevice, DiscreteGpu(n)})`, and a client
  comes from `Device::Wgpu(device).client()` (`WgpuRuntime::client` is gone);
- `cubecl_common::future` is `cubecl::future`; `cubecl::Runtime` is gone;
- the kernel scalar bounds the macro now needs (`+ Scalar`) on
  `pack_lhs/rhs_dot_general`, `launch_materialization`;
- `MatmulSetupError` no longer implements `std::error::Error`, so the two
  `dot_general` sites wrap its message in `std::io::Error::other`.

## The nine, by decision

**Apple/Metal host-visible memory (6 sites, fork-only APIs).**
`init_device_for_graphics_api` + `PrimaryMemoryMode` (imports),
`RuntimeOptions.primary_memory` (`apple.rs:174`), `WgpuMappedReadGuard` /
`WgpuMappedWriteGuard` and `WgpuResource::map_read` / `map_write`
(`mod.rs:150,169,305,331`). Upstream 0.11 has none of them; they are the
host-visible buffer mapping the goal's Apple/Metal note keeps blocked without a
real Metal device and an agreed upstream ownership contract. Options: port them
upstream (needs that contract and the user's approval), or scope the Apple path
out of the feature's build.

**CUDA interop inside the webgpu module (3 sites, fork-only methods plus a
scoping question).**
`mod.rs:116-123` and `event_domain.rs:239` reach into the CUDA server
(`get_resource::<cubecl_cuda::ffi_interop::CudaServer>`, `with_server::<…>`) to
take an allocation id and to submit a stream completion. Two methods they need
do not exist on the ported interop surface:
`CudaServer::submit_stream_completion(stream_id)` and
`GpuResource::allocation_id()`. `mod.rs:123` then mismatches types
(`ManagedResource<WgpuResource>` expected, `ManagedResource<GpuResource>` found),
which is the same path's scoping problem: the wgpu buffer is being resolved
through the CUDA server. With `--features webgpu` alone `cubecl_cuda` is not
linked at all, so these paths also need `#[cfg(feature = "cuda")]` gating with a
defined fallback. Options: port the two methods upstream and gate the paths, or
move the CUDA-dependent part behind its own feature.

**cubek matmul strategy (1 site).**
`gemm.rs:6,463,528` import `cubek_matmul::launch::Strategy` and ask for
`Strategy::Naive`. In 0.11 the public enum is `cubek_matmul::strategy::Strategy`
with `Tiled`/`MultiLevel`/`Auto` — no `Naive`. Which arm replaces it is a choice
about which kernel architecture the Metal/wgpu path requests, not a rename.
