# RunPod CUDA Driver Filter

## Session summary

Fixed intermittent RunPod GPU CI failures caused by assigning CUDA 12.8 PTX
workloads to hosts whose drivers only support CUDA 12.4.

## Context read

- Failed RunPod run 29628588468: driver 550.127.05 advertised CUDA 12.4 while
  the workflow loaded NVRTC 12.8, causing
  `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`
- Successful same-main-revision run 29627753274: driver 610.43.02
- RunPod `POST /pods` OpenAPI contract for `allowedCudaVersions`
- `scripts/ci/runpod_client.py`, `runpod_contract.py`, and their contract tests

## Chosen design

- Request only RunPod hosts advertising CUDA 12.8, 12.9, or 13.0 through
  `allowedCudaVersions`, matching the workflow's CUDA 12.8 compiler floor.
- Validate configured CUDA versions against the live OpenAPI schema alongside
  GPU type IDs before creating a paid pod.
- Keep the runtime NVRTC version check as defense in depth.

## Rejected alternatives

- Changing only the container image cannot upgrade the host NVIDIA driver.
- Checking only loaded NVRTC detects an old userspace toolkit but cannot stop
  CUDA 12.8 PTX from reaching a CUDA 12.4 driver.

## Verification

- Focused RunPod client, contract, and workflow contract unit tests
- Local CI-only PR gate
- `actionlint` and Python bytecode compilation
- Live RunPod OpenAPI schema validation

## Residual risk

- A paid RunPod GPU run is still required to validate scheduling behavior
  end-to-end.
