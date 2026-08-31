# CUDA library compatibility documentation

## Summary

Implemented issue #1738 by adding one user-facing compatibility matrix for the
CUDA driver/runtime, NVRTC, cuTENSOR, cuBLAS, cuSOLVER, and cuFFT. README,
`llms.txt`, the documentation landing page, backend selection, and
troubleshooting now route readers to that matrix.

## Sources reviewed

- CUDA library loaders in `tenferro-gpu`, `tenferro-linalg`, and `tenferro-fft`
- `.github/workflows/ci-cache-publish.yml`
- `.github/workflows/runpod-gpu-test.yml`
- `scripts/ci/install_cuda_runtime_tree.sh`
- Existing GPU, backend-selection, and troubleshooting guides
- NVIDIA CUDA compatibility documentation

## Decisions

- Separate tenferro's supported runtime contract from loader soname probing and
  CI-tested package versions. A loadable soname is not presented as a support
  guarantee.
- Keep CUDA 12.4 as the baseline and CUDA 12.8 as the full-capability tier.
- Record exact CI evidence only where it is pinned: cuTENSOR `2.6.0.4`. For
  cuBLAS and cuSOLVER, name the NVIDIA package families because CI does not pin
  their patch versions. State explicitly that cuFFT is not separately pinned.
- Keep the matrix in the existing Devices and GPU guide instead of creating a
  new page, so current README, docs sidebar, and `llms.txt` routes remain the
  shortest path.

## Verification

- `bash scripts/check-pr-fast.sh`
- `bash scripts/build_docs_site.sh`
- `python3 scripts/check-docs-site.py --root-dir .`
- Source-blind review of the rendered Devices and GPU page; its one finding, a
  missing blank line that prevented the compatibility heading from rendering,
  was fixed and the heading/TOC anchor was rechecked.

## Remaining limits

The matrix reports the tested package families rather than promising every
vendor patch release. Future CUDA CI version changes must update the table.
