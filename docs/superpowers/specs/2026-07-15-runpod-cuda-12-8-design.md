# RunPod CUDA 12.8 alignment

## Problem

The trusted post-merge RunPod run selected an NVIDIA GeForce RTX 5090 and
failed 62 CUDA tests because CubeCL passed `sm_120a` to a CUDA 12.4 NVRTC,
which rejected the Blackwell architecture. The workspace already pins cudarc
to the CUDA 12.8 binding set and documents CUDA 12.8 as the current CubeCL API
floor, so the CI runtime is below the repository contract.

## Design

- Set the shared RunPod workflow CUDA toolkit/runtime version to 12.8 for both
  the GitHub-hosted test archive build and the external RunPod test runner.
- Keep the existing CubeCL, cudarc, cuTENSOR, and OpenXLA dependency versions.
  cuTENSOR remains on its CUDA 12 redistributable, while OpenXLA continues to
  use its separately pinned CUDA 12.9 `ptxas` package.
- Preserve the existing GPU price tiers, including the RTX 5090. CUDA 12.8 is
  the first toolkit release with compiler support for `sm_120`/`sm_120a`.
- Log the loaded NVRTC version before tests so future runtime/toolkit drift is
  visible in the trusted run.
- Add source-contract coverage that requires the workflow runtime version to
  match the workspace cudarc CUDA binding floor.

The existing content-addressed keys include the CUDA runtime version and the
workflow content, so this change intentionally creates fresh archive and
runtime caches without a cache schema migration.

## Failure handling and verification

The workflow continues to fail before CUDA tests if exact CUDA 12.8 packages
cannot be installed. The trusted cleanup job remains unconditional and must
delete the pod after either success or failure.

Verification proceeds from the narrow CI contract tests and actionlint to the
repository-required local checks. The final acceptance test is a trusted
post-merge RunPod run that records CUDA/NVRTC 12.8 or newer, completes the CUDA
archive tests and OpenXLA PJRT tests, emits `RUNPOD_TENFERRO_GPU_TEST_OK`, and
deletes the selected pod.

