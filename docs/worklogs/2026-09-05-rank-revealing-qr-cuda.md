# Rank-revealing QR CUDA phase

## Scope

Adds native CUDA execution for issue #1754's fixed-four-output RRQR operation.
The implementation uses same-device column-pivoted Householder steps because
CUDA 12.4 cuSOLVER does not provide `GEQP3`.

## Implementation

- CubeCL kernels compute trailing column norms, deterministic lowest-index
  pivots, device column/permutation swaps, Householder reflectors and updates,
  Q initialization, R extraction, and prefix numerical rank.
- The host controls the bounded `min(m,n)` launch sequence; tensor-sized work
  is parallel over rows, columns, and trailing batches.
- Q, R, permutation, rank, norms, pivots, and reflector state remain on the
  active CUDA runtime. Only the bounded provider-status vector is downloaded;
  the caller may later read rank metadata explicitly.
- Complex reflector reductions use a one-element same-runtime imaginary-unit
  constant. Gauge processing reuses the existing device QR gauge path and
  touches only Q/R.
- CUDA borrowed reads canonicalize on device and dispatch to the same owner
  implementation. There is no CPU fallback.

## Verification

CUDA feature compilation and A100 hardware tests cover F64 interspersed
rank-deficiency with reconstruction and orthogonality, F32 batched/scaled rank,
and C32/C64 reconstruction, permutation, rank, and positive-diagonal gauge.
A source-contract test rejects payload host access, CPU fallback, and
single-worker launches.

## Remaining

After this phase merges, tensor4all-rs updates its tenferro pin and replaces its
local resident QR rank heuristic with this RRQR operation. RRQR AD remains
explicitly unsupported as approved by the design.
