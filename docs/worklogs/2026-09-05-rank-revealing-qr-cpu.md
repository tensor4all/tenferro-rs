# Rank-revealing QR CPU phase

## Scope

Implements issue #1754's fixed-four-output RRQR contract for concrete, typed,
eager, and traced surfaces, with CPU-faer and CPU-BLAS/LAPACK providers. CUDA
remains an explicit unsupported backend until the native provider phase.

## Context reviewed

- `docs/design/rank-revealing-qr.md`
- `docs/design/dynamic-symbolic-shapes.md`
- existing QR, full-pivot LU, Householder QR, extension metadata, and AD support
  implementations
- faer 0.24 column-pivoted QR and LAPACK `xGEQP3` contracts

## Decisions

- RRQR is a distinct fixed-arity extension op; ordinary QR stays a two-output
  hot path.
- Rank and permutation are I64 tensors, so traced and batched execution never
  embeds data-dependent host metadata.
- Q/R keep full thin shapes. No dynamic compact shape or fixed-rank mask is
  introduced.
- Provider permutations are normalized to the documented zero-based gather
  convention.
- RRQR differentiation is explicitly unsupported in every AD dispatch and
  manifest entry for this phase.

## Verification

Focused tests cover interspersed dependent columns, zero/batched matrices, all
four floating/complex dtypes, invalid tolerances, eager runtime ownership,
traced four-output execution, and unsupported AD manifest entries. Default
integration tests, doctests, clippy, CPU-BLAS compilation, and repository-rule
checks are run before PR creation.

## Remaining phase

A native CUDA column-pivoted Householder implementation is required because
CUDA 12.4 cuSOLVER has no `GEQP3`. It must not download matrix payloads or fall
back to CPU. tensor4all integration follows the merged CUDA phase.
