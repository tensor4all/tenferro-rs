# U2, part 1: preparing and launching the kernel catalogue

Date: 2026-10-23. Issue: [tenferro-rs#2208](https://github.com/tensor4all/tenferro-rs/issues/2208)
unit U2. Design: [cuda-seam-u1-2208.md](../design/cuda-seam-u1-2208.md) §2 (the
`KernelSource` seam) and the parent design §4.

## What landed

- `t4a-cuda-kernels` is a new dependency of the `cuda-next` feature, pinned by revision
  because the crate is `publish = false`
  (`5019c713c14c1f5d59ea2cf03bbfff08c40f167e`, which exposes its kernel sources for
  NVRTC — [t4a-cuda-kernels#17](https://github.com/tensor4all/t4a-cuda-kernels/pull/17)).
- `cuda_next::KernelModules`: prepares every translation unit once per runtime through
  `Runtime::compile` (NVRTC, `OutputKind::Cubin`, `sm_<major><minor>` from the device's
  compute capability), binds an entry on first use, and launches it on the runtime's
  stream. `layout_arg` hands a `CLayout` to the launch as the by-value parameter the ABI
  documents, and `device_buffer` exposes the runtime allocation behind a tensor for a
  device-pointer argument.
- The architecture is real, not virtual: the entries use double atomics and other sm_60+
  intrinsics, and NVRTC rejects them when compiled for `compute_XX`.

## Evidence

`tests/seam_launch.rs` on the local A100 (driver 580.126.09, CUDA 12.6):

- `the_catalogue_prepares_every_module_once`: eleven modules prepared, and a second
  preparation performs no NVRTC compilation (the runtime's artifact cache).
- `copy_exact_reads_back_the_same_bytes` — a 1024-element f64 copy round-trips.
- `materialize_strided_gathers_a_transposed_view` — a permuted `CLayout` (144-byte
  by-value parameter) gathers the documented view, compared against the host reference.
- `fill_zero_and_contiguous_to_view_cover_the_movement_family` — `fill_zero`, the
  inverse-permutation store of `contiguous_to_view` (asserted per element) and
  `fill_zero_view` over a strided layout.

## Observations

- A cold `KernelModules::new` compiles all eleven modules eagerly (~90 s on this host).
  That is the correct starting point but not the prepare contract: U3 selects only the
  artifacts the caller's plan needs, keys them by the artifact identity and reports
  cold/warm counters. Until then the eager compile is recorded as a limitation.
- The movement family is the first dispatch slice. The remaining U2 families
  (elementwise, comparison, conversion, gather/scatter) reuse this launcher unchanged;
  what they add is the plan-to-argument marshalling per entry.
