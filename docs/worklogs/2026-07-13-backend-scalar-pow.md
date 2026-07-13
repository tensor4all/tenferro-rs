# Backend scalar `pow` support

## Summary

Issue #1371 adds rank-0 operand support to direct CPU and CUDA `pow`
execution. General broadcasting remains owned by the runtime/eager/traced
layers; the backend path accepts only equal shapes or exactly one rank-0
operand and never materializes a dense broadcasted scalar.

## Context and corrected scope

The investigation began from the earlier #1358 report that CUDA rejected a
case accepted by CPU. At that revision, both direct backends rejected it.
Public `TensorOpsExt::pow`, typed, eager-AD, and traced APIs already performed
explicit NumPy-style broadcasting and needed no production change. The accepted
#1371 contract therefore deliberately extends both direct backends together.

The user-facing precedent was checked against the official JAX `jnp.power`,
PyTorch `torch.pow`, and NumPy `power` documentation: each supports scalar and
broadcast-compatible operands. This change adopts only the rank-0 backend
subset needed to avoid CPU/CUDA divergence.

## Implementation

- CPU owned-buffer and read-view paths read the scalar once and map directly
  over the non-scalar operand using the existing pooled kernels.
- CUDA float paths use a scalar-indexed `powf` kernel through the existing
  scalar binary launch boundary.
- CUDA integer paths use a scalar-indexed checked kernel, retaining the device
  error flag and `NegativeIntegerExponent` contract.
- Equal-shape behavior and error precedence are unchanged. Unequal non-scalar
  shapes still return `ShapeMismatch`.

## Verification evidence

The CPU regression first failed with `ShapeMismatch { op: "pow", lhs: [3],
rhs: [] }`; after implementation, all 243 CPU library tests passed. The CUDA
source contract first failed because `pow` lacked a scalar launcher. The
focused A100 test then passed for F32, F64, I32, and I64 in both operand
positions, including empty outputs, integer negative-exponent errors,
non-scalar shape rejection, and floating NaN/infinity/signed-zero classes.
The CUDA feature build also completed with `--no-run`.

## Residual scope

This does not add general raw-backend broadcasting, mixed-dtype power, or CUDA
complex power. Public broadcasting remains an upper-layer responsibility.
