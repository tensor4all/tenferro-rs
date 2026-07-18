# CUDA 12.4 runtime compatibility

## Session summary

Expanded tenferro's RunPod CUDA host floor from 12.8 down to 12.4 without
removing CUDA 12.8-only CubeCL capabilities from newer environments.

## Context and evidence

- RunPod runs 29628588468 and 29628961173 used RTX 4090 cards with driver
  550.127.05, which advertised CUDA 12.4. The workflow loaded NVRTC 12.8 and
  failed with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`.
- Run 29627753274 succeeded on an RTX 4090 with driver 610.43.02.
- cudarc's dynamic driver bindings resolve individual symbols lazily. Its CUDA
  12.8 binding set can therefore run on a 12.4 driver if 12.8-only symbols are
  not called.
- CubeCL had used cudarc's compile-time `CUDA_VERSION` to advertise fast tanh,
  `Im2colWide`, and tensor-map swizzle atomicity. That value described the
  binding set rather than the loaded driver and NVRTC.
- The CubeK revision pinned by tenferro is enabled only by the WebGPU feature.
  Its `cubek-matmul` and `cubek-std` sources do not directly select the CUDA
  12.8-only CubeCL tensor-map features.

## Chosen design

- Keep cudarc `cuda-12080` bindings so a single binary contains the full CUDA
  12.8 API surface.
- In the tensor4all CubeCL fork, query `cuDriverGetVersion` and `nvrtcVersion`.
  Advertise CUDA 12.8-only capabilities only when both are at least 12.8, and
  guard the corresponding tensor-map symbol calls before lazy resolution.
- Add CUDA 12.4 back to RunPod `allowedCudaVersions`. Select NVRTC 12.4 for a
  host driver below 12.8 and NVRTC 12.8 for a driver at or above 12.8.
- Keep the hosted Rust archive on 12.8 bindings. CubeCL generates PTX on the
  GPU worker, so the worker's selected NVRTC controls PTX compatibility.

## Alternatives rejected

- Changing tenferro to `cuda-12040` bindings would permanently compile out
  CubeCL's CUDA 12.8-only paths, including on CUDA 12.8 and newer systems.
- Loading NVRTC 12.8 on every host and filtering only after startup reproduces
  the unsupported-PTX failure on CUDA 12.4 drivers.
- Keeping the RunPod floor at 12.8 avoids the mismatch but unnecessarily
  removes otherwise compatible GPU cards from the scheduling pool.

## Verification

- CubeCL runtime-version matrix tests: CUDA 12.4, 12.6, 12.8, CUDA 13 with
  NVRTC 12.8, and mismatched driver/NVRTC pairs.
- CubeCL cudarc feature and 12.8 symbol-guard source contracts.
- `cargo check -p t4a-cubecl-cuda`.
- CubeCL fork commit `792c5a722e9ccb0aa62b14529ddb088b5aeb546b`
  pushed and fetched through tenferro's remote git dependency pin.
- `cargo check -p tenferro-gpu --features cuda` using the remote pin without a
  local path override.
- 51 tenferro RunPod, workflow, artifact, and client contract tests.
- CubeK CUDA dependency-path contract.

## Remaining acceptance work

- Run trusted RunPod GPU validation once on a CUDA 12.4 host and once on a CUDA
  12.8-or-newer host. The former must avoid unsupported PTX; the latter must
  retain the full hardware-supported CubeCL capability set.
