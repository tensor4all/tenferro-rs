# CUDA 12.4 runtime compatibility with full CUDA 12.8 capabilities

## Goal

tenferro's CUDA backend must run its baseline CUDA functionality on systems
whose NVIDIA driver and NVRTC are CUDA 12.4 compatible. When both the driver
and NVRTC are CUDA 12.8 or newer, CubeCL must retain every capability supported
by the GPU hardware, including the CUDA 12.8 tensor-map extensions.

This is a runtime compatibility change, not a downgrade of the generated Rust
bindings. The workspace and the tensor4all CubeCL fork continue to compile
against cudarc's `cuda-12080` binding set so that one binary contains both the
12.4-compatible path and the 12.8-only path.

## CubeCL runtime capability model

CubeCL currently derives CUDA capabilities from cudarc's compile-time
`CUDA_VERSION`. That value describes the selected Rust bindings, not the
driver or NVRTC libraries loaded on the target machine. With 12.8 bindings,
this causes a CUDA 12.4 machine to advertise 12.8-only features and can make it
generate PTX newer than the installed driver accepts.

The fork will query two versions while initializing a CUDA device:

- `cuDriverGetVersion` for the driver API level;
- `nvrtcVersion` for the compiler that generates PTX at runtime.

The effective runtime API level is the lower of those two versions. CUDA
12.8-only compiler and tensor-map capabilities are advertised only when the
effective level is at least 12.8. In particular, this applies to fast-tanh
compiler support, `Tma::Im2colWide`, `Tma::SwizzleAtomicity`, and the associated
FP4 tensor-map data types and atomic swizzle modes. Existing architecture checks
remain in force, so a CUDA 12.8 environment does not advertise a hardware
feature that its GPU architecture lacks.

The server must also guard 12.8-only tensor-map encoding at the call boundary.
Capability selection should normally prevent an unsupported request, but the
guard ensures that a manually constructed request on CUDA 12.4 returns a
controlled unsupported error instead of asking cudarc to resolve a symbol that
the older driver does not export.

cudarc's dynamic driver bindings resolve individual functions lazily. Keeping
the 12.8 binding set is therefore compatible with a 12.4 driver as long as
12.8-only symbols are neither advertised nor called on the older runtime.
NVRTC itself determines the emitted PTX version, so loading NVRTC 12.4 on a
12.4 driver produces PTX accepted by that driver.

## tenferro and RunPod alignment

tenferro will keep `cuda-12080` in its workspace cudarc feature contract and
will document CUDA 12.4 as the minimum supported runtime. Documentation must
distinguish the 12.8 binding set from the 12.4 runtime floor and state that
CUDA 12.8-only features require both a 12.8-or-newer driver and NVRTC.

RunPod configuration will again permit hosts reporting CUDA 12.4. The external
runner will choose an NVRTC/toolkit runtime that does not exceed the assigned
host driver:

- CUDA 12.4 hosts load the pinned CUDA 12.4 runtime;
- CUDA 12.8-or-newer hosts load the pinned CUDA 12.8 runtime.

The Rust test archive remains compiled with the 12.8 cudarc bindings. Kernel
PTX is generated only on the GPU runner, so the loaded NVRTC version, rather
than the archive build host, controls PTX compatibility. The workflow will log
the driver and loaded NVRTC versions before testing and reject combinations
below the 12.4 floor or combinations in which NVRTC is newer than the driver.

## CubeK impact

The CubeK revision pinned by tenferro does not directly select
`Im2colWide`, `SwizzleAtomicity`, or the CUDA 12.8 atomic swizzle modes in its
`cubek-matmul` and `cubek-std` crates. tenferro currently enables those CubeK
crates only for its WebGPU GEMM backend; its CUDA backend does not depend on
CubeK. No CubeK source change is required for this compatibility work.

CubeK remains covered by dependency and feature-tree contract checks so that a
future CUDA use cannot silently introduce a second CUDA runtime floor.

## Verification

The change will be verified at four levels:

1. Pure CubeCL tests cover the driver/NVRTC version matrix, including 12.4,
   12.6, 12.8, newer-driver/older-NVRTC, and older-driver/newer-NVRTC cases.
2. CubeCL source-contract tests require 12.8 bindings and runtime gating at the
   12.8-only symbol boundary.
3. tenferro CI contract tests require a 12.4 runtime floor, preserve 12.8
   bindings, and verify adaptive RunPod runtime selection.
4. Trusted GPU runs exercise an RTX 4090 host with a CUDA 12.4 driver and a
   CUDA 12.8-or-newer host. The first must complete the ordinary CUDA suite
   without `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`; the second must retain the
   full capability set supported by its GPU.

## Non-goals

- Supporting a CUDA runtime older than 12.4.
- Emulating CUDA 12.8-only tensor-map operations on CUDA 12.4.
- Changing tenferro's public tensor API or CUDA operation semantics.
- Changing CubeK's WebGPU kernel selection.
