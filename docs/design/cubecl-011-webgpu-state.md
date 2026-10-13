# webgpu feature: migration state and the remaining decisions

Status: 54 errors -> 7 with `--features webgpu` (7 with
`--features cuda,webgpu`). Everything on the tenferro side is done; the seven left
each need an API only the host-visible port provides, which is why this is a
separate record from the A100 note.

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
  `dot_general` sites wrap its message in `std::io::Error::other`;
- the webgpu module's server-typed interop names the WGPU server, not the CUDA
  one: `with_server::<WgpuServer<AutoCompiler>, _>` (`event_domain.rs`) and
  `get_resource::<WgpuServer<AutoCompiler>>` (`mod.rs:116`). Both are the
  WebGPU runtime's own server, and the resolved `ManagedResource<WgpuResource>`
  is the type the buffer field already declared (#2114);
- `cubek_matmul::strategy::Strategy` replaces the now-private
  `launch::Strategy`; the dot-general path asks for `MultiLevel(Naive)` (see
  below).

## The seven, by decision

**Apple/Metal host-visible memory (7 sites, fork-only APIs).**
`init_device_for_graphics_api` + `PrimaryMemoryMode` (imports),
`RuntimeOptions.primary_memory` (`apple.rs:173`), `WgpuMappedReadGuard` /
`WgpuMappedWriteGuard` and `WgpuResource::map_read` / `map_write`
(`mod.rs:149,168,304,330`), plus `WgpuResource::allocation_id` (`mod.rs:118`),
which the WGPU server typing un-masked. Upstream 0.11 has none of them; they are
the host-visible buffer mapping the goal's Apple/Metal note keeps blocked without
a real Metal device and an agreed upstream ownership contract, and the
`t4a/hostvisible` port carries all seven. Options: take that port upstream (needs
the ownership contract and the user's approval), or scope the Apple path out of
the feature's build.

**cubek matmul strategy (decided).**
`gemm.rs` asks for `Strategy::MultiLevel(multi_level::Strategy::Naive)`. 0.11
moved the fork's `Naive` kernel under the `multi-level` family, so this is the
same simple, always-available routine — one output element per unit, no
accelerator requirement — and keeps the webgpu path's kernel choice and
numerics. The workspace pin enables `multi-level` only, so `Tiled` is not
compiled in at all; `Auto` would try `SimpleCyclicCmma` first and fall back to
`SimpleUnit`, silently requesting a different kernel. `cubek_matmul::launch`
keeps `Strategy` private, so the import must come from
`cubek_matmul::strategy`.
