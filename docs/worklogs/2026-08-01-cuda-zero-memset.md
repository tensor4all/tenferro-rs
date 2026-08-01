# CUDA zero allocation byte memset

## Summary

The existing `CudaBackend::zeros::<f64>` path launched a generic CubeCL fill
kernel. A downstream A100 comparison reported about a 5.6% warmed median
regression versus the previous host-zero upload path. This change keeps zero
construction device-native but enqueues `cudaMemsetAsync` on CubeCL's current
CUDA stream instead.

## Context read

- `AGENTS.md`, `REPOSITORY_RULES.md`, and
  `docs/design/gpu-backend-design.md`
- `crates/tenferro-gpu/src/cubecl/{mod,runtime,memory,interop,gemm}.rs`
- CubeCL's pinned `ComputeClient::get_resource`, `CudaServer::raw_stream`, and
  managed-allocation implementation
- cudarc 0.19.8 `memset_d8_async` and `ValidAsZeroBits` contracts

## Decisions

- Reuse the existing CubeCL raw pointer/current-stream interop; no new public
  API, dependency, synchronization, or transfer boundary is needed.
- Keep `CudaZeroScalar` sealed and add cudarc's `ValidAsZeroBits` as its private
  proof that byte zeroing produces a valid scalar. The only admitted type
  remains `f64`, for which all-zero bits are positive zero.
- Return an empty allocation before context, pointer, or stream lookup.
- Retain checked byte-length overflow with operation name `zeros`. Remove the
  obsolete CubeCL workgroup-count limit because byte memset has no launch grid.

The output tensor's CubeCL handle retains the allocation, and the memset is
submitted to the same current stream used by subsequent CubeCL work. A host
zero buffer, a new CUDA abstraction, and an internal synchronization were
rejected because each would either restore the measured cost or broaden the
change without improving the established interop contract.

## Verification

- `cargo test -p tenferro-gpu --test integration cuda_zeros_uses_stream_ordered_byte_memset`
- `cargo check -p tenferro-gpu --features cuda`
- `cargo test -p tenferro-gpu --features cuda --no-run`
- `cargo fmt --all --check`
- `git diff --check`

The existing ignored CUDA runtime test covers lengths 0, 1, 17, and 4097,
exact positive-zero bits after download, placement metadata, and overflow.
Runtime execution and the downstream paired latency comparison remain an A100
hardware gate.
