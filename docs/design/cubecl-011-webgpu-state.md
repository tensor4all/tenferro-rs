# webgpu feature: migration state and the remaining decisions

Status: the tenferro side is done and the seven APIs it was missing now exist on
the 0.11 line. Against the rebased ports, `cargo check -j 4 -p tenferro-gpu
--features webgpu` and `--features cuda,webgpu` are both at 0 errors (54 -> 0),
and the Apple/Metal tests pass on a real Metal device. The pins still name the
staged pre-rebase working trees (`../cubecl-ports-combined` at `9f097e00`, base
`a6321cd4`), so repinning to the `6bba5fdf`-rebased ports waits on the publish
step of the agreed order (#2115, #2116).

Verification: `cargo check -j 4 -p tenferro-gpu --features webgpu` and
`--features cuda,webgpu`; the Metal tests in the evidence section below.

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
  one: `with_server::<WgpuServer<AutoCompiler>, _>`
  (`webgpu/event_domain.rs:240`) and `get_resource::<WgpuServer<AutoCompiler>>`
  (`webgpu/mod.rs:116`). Both are the WebGPU runtime's own server, and the
  resolved `ManagedResource<WgpuResource>` is the type the buffer field already
  declared (#2114);
- `cubek_matmul::strategy::Strategy` replaces the now-private
  `launch::Strategy`; the dot-general path asks for `MultiLevel(Naive)` (see
  below).

## The seven sites, now provided by the port

The Mac port branch is `tensor4all/cubecl` `feat/tenferro-upstream-host-visible`,
three feature-scoped commits on upstream `main` `6bba5fdf` with no `t4a-`
packaging:

| commit | content |
| --- | --- |
| `08a8531b` | completion for a queued submission: `WgpuServer::submit_stream_completion`, `WgpuStream::seal_submission` |
| `0f1ab363` | guarded host-visible primary memory: `PrimaryMemoryMode`, `RuntimeOptions.primary_memory`, `init_device_for_graphics_api`, `WgpuMappedRead/WriteGuard`, `WgpuResource::{map_read, map_write, allocation_id}`, `HostAccessError` |
| `3098a155` | runtime-generic `Client::with_server`, which is what makes a device server reachable from outside the crate |

`WgpuServer` and `WgpuResource` are already public at the crate root
(`pub use compute::*`), so exporting the wgpu interop surface needed only the
runtime-side accessor.

The ownership contract the guards encode: a launch, a staged write and a read
copy each take a `GpuAccessToken` on the buffers they touch, a recorded graph
takes them until the submission that flushes it, and the token is released by
that submission. Mapping the host then either waits for the GPU access to clear
or fails with `HostAccessError`.

One part of that contract was missing upstream: `flush` on a stream whose task
queue is empty but which has staged `write_buffer` copies did nothing, because
`submit_tasks` early-returns on `tasks_count == 0`. The reservation therefore
outlived the flush and the host could not map the buffer. The port seals the
staged writes in that case. `apple_context` fails on Metal without it
(`GpuAccessInProgress` on `with_host_read`); a Linux compile cannot see it.

## One more rebase-time change: `Scope::child` is `branch_child`

0.11 renamed the empty child scope for if/else to `cubecl_ir::Scope::branch_child`
(`func_child` is not the equivalent: it is the root of a new non-inlined
function and installs a terminate flag). The two sites in
`crates/tenferro-gpu/src/cubecl/fusion/codegen.rs:68,80` are the CUDA fusion
codegen, so only `--features cuda` catches them: `--features cuda,webgpu` is at
0 errors once they are renamed. `branch_child` does not exist on the pinned
0.10-line revision, so this rename belongs to the repin, not before it.

## Metal evidence (measured)

Apple M5 Max, Metal 4, against the rebased ports
(`local/ports-011-combined` = the three commits above plus the CUDA-interop port
and the three arithmetic fixes), from the scratch verification worktree
`tenferro-rs/.worktrees/verify-ports-011`:

| test binary | result |
| --- | --- |
| `apple_context` | 4 passed |
| `apple_cpu` | 2 passed |
| `webgpu_metal_fft` | 1 passed (numerical + capability matrix) |
| `storage_provider_webgpu` | 3 passed |
| `storage_provider_webgpu_progress` | 2 passed |

On the cubecl side, `cargo test -p cubecl-wgpu --features msl` is 866 passed,
1 failed, 19 ignored; the failure is
`tests_msl::bf16_ty::unary::test_normalize`, which reproduces identically on
unmodified `6bba5fdf` (bf16 pipeline creation on AGXMetalG17X fails with
`XPC_ERROR_CONNECTION_INTERRUPTED`), so it is a local driver issue and not a
port regression. Compile and lint matrix on the port branch: default, `msl`,
`spirv`, `msl,exclusive-memory-only`, `wasm32-unknown-unknown`, `clippy`, and
`cargo fmt --all --check`, all clean.

## cubek matmul strategy (decided)

`gemm.rs` asks for `Strategy::MultiLevel(multi_level::Strategy::Naive)`. 0.11
moved the fork's `Naive` kernel under the `multi-level` family, so this is the
same simple, always-available routine — one output element per unit, no
accelerator requirement — and keeps the webgpu path's kernel choice and
numerics. The workspace pin enables `multi-level` only, so `Tiled` is not
compiled in at all; `Auto` would try `SimpleCyclicCmma` first and fall back to
`SimpleUnit`, silently requesting a different kernel. `cubek_matmul::launch`
keeps `Strategy` private, so the import must come from
`cubek_matmul::strategy`.
