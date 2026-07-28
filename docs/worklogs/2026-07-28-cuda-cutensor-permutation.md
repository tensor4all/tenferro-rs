# CUDA cuTENSOR Permutation

Issue: tensor4all/tenferro-rs#1506

## Summary

- Added `cutensorCreatePermutation` / `cutensorPermute` to the CUDA cuTENSOR
  dlopen FFI.
- Routed CUDA `F32`, `F64`, `C32`, and `C64` `transpose` and nonnegative-stride
  view canonicalization through cuTENSOR permutation.
- Added a backend-owned cuTENSOR permutation plan cache in the CUDA extension
  cache, with bounded entries, retained-byte accounting, clear participation,
  stats, and bound configuration.
- Recorded the NVIDIA library no-silent-fallback policy in
  `REPOSITORY_RULES.md` and the GPU backend design doc.

cuTENSOR 2.5 rejects negative-stride tensor descriptors. CUDA negative-stride
view canonicalization remains on the existing native CubeCL structural copy
kernel because that layout is outside the vendor permutation descriptor model;
this is not a fallback for missing cuTENSOR on supported descriptors.

## A100 Focused Benchmark

Machine: NVIDIA A100 80GB PCIe. Command:

```bash
CUBECL_DEBUG_LOG=0 CUDA_PATH=/usr/local/cuda \
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcutensor/12:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} \
GPU_BENCH_DEVICE=0 BENCH_RUNS=5 BENCH_WARMUPS=2 \
BENCH_OUTPUT=/tmp/gpu-permutation-1506-rust.jsonl \
./target/release/benchmark_gpu_permutation
```

The benchmark used the `tenferro-benchmark` Rust GPU permutation runner with
`extern/tenferro-rs` symlinked to this branch. The tenferro columns allocate a
fresh device output per public API call; the direct `cutensor` column reuses a
destination. The measured public-API-plus-allocation overhead stayed within
3.9% of direct destination-reuse cuTENSOR on every row.

| pattern | tenferro transpose / cuTENSOR | tenferro to_contiguous / cuTENSOR |
| --- | ---: | ---: |
| `transpose_2d_32768_16384` | 1.021x | 1.028x |
| `transpose_3d_1024_1024_512_201` | 1.011x | 1.013x |
| `transpose_3d_1024_1024_512_102` | 1.039x | 1.039x |
| `rotation_6d_64_32_32_32_16_16` | 1.033x | 1.020x |
| `reverse_23d_128x2` | 1.017x | 1.018x |
| `reverse_18d_3` | 1.022x | 1.034x |
| `cyclic_18d_3` | 1.012x | 1.011x |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | n/a | 0.982x |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 1.034x | 0.999x |

## Verification

- RED: `cargo test -p tenferro-gpu --test integration cutensor -- --nocapture`
  failed before implementation on missing FFI/module/policy contracts.
- `cargo test -p tenferro-gpu --test integration cutensor -- --nocapture`
- `cargo check -p tenferro-gpu --features cuda`
- `cargo test -p tenferro-gpu --features cuda cutensor_loader_missing_library_is_typed_io_error -- --nocapture`
- `CUBECL_DEBUG_LOG=0 CUDA_PATH=/usr/local/cuda LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcutensor/12:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} CARGO_BUILD_JOBS=64 cargo test -p tenferro-gpu --features cuda cuda_cutensor_permutation_transpose_and_to_contiguous_match_cpu -- --ignored --nocapture`
- `CUBECL_DEBUG_LOG=0 CUDA_PATH=/usr/local/cuda LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcutensor/12:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} CARGO_BUILD_JOBS=64 cargo test -p tenferro-gpu --features cuda structural -- --ignored --nocapture`
