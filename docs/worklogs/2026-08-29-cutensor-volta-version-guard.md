# 2026-08-29 cuTENSOR Volta version guard

## Scope

PR #1733 reported all-zero cuTENSOR contraction results on a V100 even though
the provider calls returned success. This follow-up prevents tenferro from
using a cuTENSOR release on an architecture that release does not support.

## Context read

- PR #1733's V100 result and version boundary
- `REPOSITORY_RULES.md` and `docs/design/gpu-backend-design.md`
- the CUDA device discovery, runtime metadata, lazy cuTENSOR loader, contraction,
  and permutation paths
- NVIDIA's cuTENSOR 2.2 and 2.3 support and release documentation

## Evidence and decision

The cuTENSOR 2.2 documentation lists SM 7.0 as supported and deprecates that
support. PR #1733 nevertheless reports that 118 of 120 output-mode
permutations returned all-zero results on a V100 with cuTENSOR 2.2 and newer,
while every provider status reported success. The cuTENSOR 2.3 release notes
then remove SM 7.0, and its support table starts at SM 7.5.

The guard rejects version 2.2.0 and newer on compute capability 7.0. This is
stricter than NVIDIA's 2.2 support table because silent wrong results violate
the backend contract. cuTENSOR 2.1.x remains the documented Volta path.

The loader resolves `cutensorGetVersion`, checks the pure version/device rule,
and returns a typed unsupported error before `cutensorCreate`. Turing and newer
devices, plus cuTENSOR 2.1.x on Volta, keep the existing path. No native CubeCL
fallback was added.

Rejecting only cuTENSOR 2.3 would match NVIDIA's published support table but
would leave the reproduced 2.2 wrong-result path enabled. Falling back to a
native CubeCL contraction would violate the accepted vendor-provider contract.

## Verification

- Pure unit coverage pins cuTENSOR 2.1.x acceptance on SM 7.0, cuTENSOR 2.2.0
  rejection on SM 7.0, and cuTENSOR 2.2.0 acceptance on SM 7.5.
- GPU execution remains hardware-gated. A V100 with both cuTENSOR 2.2.x and
  2.3.x is still needed to verify the complete loader and contraction path.
  Do not relax the guard without a numerical matrix that covers every output
  mode on cuTENSOR 2.1.x and the candidate newer release.

## References

- [NVIDIA cuTENSOR 2.2 support table](https://docs.nvidia.com/cuda/cutensor/2.2.0/index.html)
- [NVIDIA cuTENSOR 2.3 release notes](https://docs.nvidia.com/cuda/cutensor/2.3.0/release_notes.html)
