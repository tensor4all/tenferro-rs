# CubeK Complex GEMM API Design

## Context

The first WebGPU `dot_general` implementation in tenferro lowers `F32` and
`C32` contractions through CubeK real matmul. That implementation proves the
WebGPU provider split and planner shape, but it keeps complex GEMM decomposition
inside tenferro. That is the wrong long-term ownership boundary.

Complex matrix multiplication is a matmul-provider concern. Tenferro should
normalize tensor contraction metadata and call a backend matmul provider. CubeK
should own complex GEMM semantics, including optional conjugation, temporary
real buffers, and the real-GEMM lowering used before native complex kernels
exist.

This design supersedes the tenferro-local complex decomposition direction in
`docs/superpowers/plans/2026-06-13-webgpu-complex-gemm.md` for new work.

## Goals

- Add a CubeK-owned `C32` complex GEMM entry point with `lhs_conj` and
  `rhs_conj` semantic attributes.
- Keep the existing CubeK real `launch_ref` API compatible for downstream
  users.
- Make tenferro `WebGpuBackend::dot_general` and
  `WebGpuBackend::dot_general_with_conj` call the CubeK complex GEMM API for
  `C32` inputs instead of owning the complex lowering.
- Preserve the CUDA performance contract: no changes to CUDA cuTENSOR,
  cuBLAS, cuSOLVER, workspace allocation, buffer pools, or algorithm selection.
- Avoid hidden CPU transfers or CPU fallback. All WebGPU complex GEMM work stays
  on the WebGPU provider.

## Non-Goals

- Do not add `C64` WebGPU GEMM before CubeK and CubeCL-WGPU have a supported
  `F64` path.
- Do not make general WebGPU elementwise coverage a dependency of
  `dot_general_with_conj`. Elementwise `conj` should be implemented later, but
  complex GEMM conjugation belongs to the GEMM API.
- Do not add a vague public `gpu` feature or enable a GPU provider by default.
- Do not vendor CubeK source into tenferro.

## Repository Boundary

Clone and patch CubeK as a sibling project:

```text
~/tensor4all/cubek
```

Start from the CubeK release series already used by tenferro:

```text
cubek-matmul 0.2.0
cubek-std 0.2.0
cubecl 0.10.0
```

The preferred long-term distribution is tensor4all-owned CubeK crates published
from that branch. During development, tenferro may use a git dependency or
`[patch.crates-io]` pointing at the tensor4all CubeK fork. The committed
tenferro dependency must be deliberate and documented; a local path dependency
is not acceptable in a PR.

## CubeK API Shape

Add an additive complex GEMM API rather than changing the meaning of real
`launch_ref`.

```rust
pub struct ComplexMatmulOptions {
    pub lhs_conj: bool,
    pub rhs_conj: bool,
}

pub fn launch_c32_ref<R: Runtime>(
    strategy: &Strategy,
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    options: ComplexMatmulOptions,
) -> Result<(), MatmulSetupError>;
```

The exact function name can be adjusted to match CubeK naming conventions, but
the public contract must make these points explicit:

- The input and output buffers are interleaved `C32` values.
- Shapes follow the existing CubeK batched matmul convention:
  `[batch..., M, K]`, `[batch..., K, N]`, `[batch..., M, N]`.
- `lhs_conj` and `rhs_conj` are complex-only semantic attributes.
- Real GEMM remains available through the existing real API.
- Unsupported dtype or layout combinations return explicit setup errors.

Do not add `conj` flags to real `launch_ref` as if real GEMM had meaningful
conjugation semantics.

## CubeK Initial Lowering

The first CubeK implementation can lower `C32` GEMM to four real `F32` GEMMs:

```text
A = Ar + i Ai
B = Br + i Bi

si = lhs_conj ? -1 : +1
sj = rhs_conj ? -1 : +1

real = dot(Ar, Br) - si * sj * dot(Ai, Bi)
imag = sj * dot(Ar, Bi) + si * dot(Ai, Br)
```

CubeK owns the temporary `F32` buffers, extraction kernels, compose kernel, and
future replacement by native complex kernels or fused epilogues. The lowering
must launch over tensor-sized output domains and must not use single-worker
tensor loops.

The conjugation signs are operation attributes. Passing them as compile-time
kernel options is acceptable because they are not runtime shape or stride
metadata.

## Tenferro Integration

Tenferro keeps the `DotGeneralConfig` planner. Its responsibilities are:

- validate ranks, paired batch dims, and paired contracting dims;
- compute `M`, `K`, `N`, batch shape, and tenferro output shape;
- expose CubeK-compatible bindings for `[batch..., M, K]`,
  `[batch..., K, N]`, and `[batch..., M, N]`;
- pack same-device WebGPU scratch tensors only when layout metadata cannot
  represent the required CubeK view;
- dispatch `F32` to CubeK real GEMM;
- dispatch `C32` to CubeK complex GEMM with `lhs_conj` and `rhs_conj`;
- return explicit unsupported errors for `F64`, `C64`, and zero-contracting
  cases until support is verified.

`WebGpuBackend` must override `TensorDot::dot_general_with_conj`. For `F32`,
conjugation is identity and should reuse the real path. For `C32`, it forwards
the flags to CubeK. It must not implement this by calling general WebGPU
`conj` first.

## Error Handling

CubeK errors should distinguish setup failures from runtime launch failures
where the current CubeK error model allows it. Tenferro maps CubeK errors into
`Error::BackendFailure` with operation names that mention WebGPU dot/general
matmul and include enough detail to diagnose unsupported dtype, layout, or
runtime problems.

No WebGPU complex GEMM path may silently download tensors, upload tensors, or
fall back to CPU.

## Tests

CubeK should gain provider-level tests for:

- `C32` rank-2 GEMM without conjugation;
- `C32` batched GEMM without conjugation;
- `lhs_conj`, `rhs_conj`, and both-conjugated `C32` GEMM;
- layout cases matching existing real matmul coverage where feasible;
- explicit unsupported dtype or shape errors.

Tenferro should gain WebGPU tests for:

- `C32` `dot_general_with_conj` with `lhs_conj`;
- `C32` `dot_general_with_conj` with `rhs_conj`;
- batched `C32` `dot_general_with_conj`;
- `F32` `dot_general_with_conj` routing as identity;
- CUDA source contract tests proving the CUDA cuTENSOR path is not rewired to
  CubeK.

Tests that require a WebGPU adapter should use the existing WebGPU availability
guard. CPU reference calculations are allowed in tests, not in backend
execution.

## Documentation

Update tenferro developer docs to state that WebGPU complex GEMM semantics are
owned by CubeK. User docs should describe observable support only:

- WebGPU supports explicit transfer plus experimental `F32` and `C32`
  `dot_general`/einsum coverage.
- Conjugated complex `dot_general` is supported for `C32` once this design is
  implemented.
- `C64`, `F64`, broad elementwise, reductions, indexing, and linalg remain
  explicit unsupported WebGPU paths unless separately implemented.

CubeK docs should document the complex GEMM API as a real matmul extension, not
as a tenferro-specific helper.

## Open Follow-Up

General WebGPU elementwise coverage remains a separate issue. `conj` is a basic
operation and should be implemented for WebGPU tensors, but complex GEMM
conjugation should not depend on that path.
