# 2026-08-01 Revert CUDA zero allocation (#1586)

## Scope

Revert the public `CudaBackend::zeros` API added by #1583 because CubeCL does
not yet publish a kernel output write as a new managed binding cursor. The
allocate-then-fill sequence can therefore return an output whose initialization
is not ordered before a later first use from another thread and stream.

The revert removes `CudaZeroScalar`, `CudaBackend::zeros`, their tests and
exports, and the two operation-aware helper generalizations that had no other
callers. The remaining five source and test files match the parent of #1583;
unrelated CUDA allocation and kernel paths are unchanged.

## Context reviewed

- issue #1586 and the rejected raw-memset follow-up #1585;
- CubeCL issue #16 and the pinned CubeCL binding-cursor behavior summarized in
  #1586;
- #1583's complete diff and all current repository references to the removed
  API and generalized helpers;
- `AGENTS.md`, `CONTRIBUTING.md`, `REPOSITORY_RULES.md`, and the shared Rust,
  performance, documentation, and test rules.

## Decision

Remove the unsafe-to-publish public constructor until CubeCL provides
scheduler-managed binding write-version publication covering both ordinary
kernel outputs and external writes. A future raw-memset path additionally needs
external write registration and stream enqueue to be atomic. Do not add a
Tenferro-wide lock, synchronize before returning, reconstruct CubeCL allocation
internals, or modify adjacent CUDA operations in this leaf.

## Verification

- repository search contains no `CudaZeroScalar`, `CudaBackend::zeros`, or
  caller of the removed helper generalizations;
- the focused CubeCL checked-allocation source contract passes;
- the required local PR gate, repository-rules review, default build, and CUDA
  build are recorded in the PR.

Hardware concurrency validation is not applicable to the removal. The systemic
cursor contract remains tracked by CubeCL #16.
