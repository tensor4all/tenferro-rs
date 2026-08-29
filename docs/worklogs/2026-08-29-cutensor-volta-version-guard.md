# 2026-08-29 cuTENSOR Volta version guard

## Session summary

PR #1733 reported all-zero cuTENSOR contraction results on a V100 even though
provider calls returned success. This PR rejects cuTENSOR 2.2 and newer on SM
7.0 before handle creation and keeps cuTENSOR 2.1.x as the Volta path.

## Context read

- PR #1733's V100 version bisect
- the device metadata, lazy loader, contraction, and permutation paths
- `REPOSITORY_RULES.md`, the GPU design and guide, and NVIDIA's cuTENSOR 2.2
  and 2.3 documentation

## Evidence and decision

NVIDIA lists SM 7.0 as supported but deprecated in cuTENSOR 2.2 and removes it
in 2.3. The reproduced 2.2 wrong-result path makes the empirical boundary the
safer contract. The existing lazy loader is the single boundary shared by
contraction and permutation, so the guard belongs there. No native fallback is
added.

## Verification

- Unit coverage pins both sides of the 2.2 boundary on SM 7.0 and acceptance on
  SM 7.5.
- On an A800 (SM 8.0), CUDA 12.1 and cuTENSOR 2.3.1 passed targeted
  `dot_general` and permutation tests through the real provider.
- A V100 remains necessary for end-to-end rejection coverage.

## References

- [NVIDIA cuTENSOR 2.2 support table](https://docs.nvidia.com/cuda/cutensor/2.2.0/index.html)
- [NVIDIA cuTENSOR 2.3 release notes](https://docs.nvidia.com/cuda/cutensor/2.3.0/release_notes.html)
