# Floating-Point Domain Semantics

## Context

Issue #1353 reports that integer remainder by zero returns a typed
`DivisionByZero` error while floating-point remainder by zero produces `NaN`.
The behavior is intentional at the arithmetic level, but tenferro does not yet
state the distinction as a public numeric contract.

## Decision

Tenferro preserves IEEE-style value propagation for floating-point elementwise
operations wherever the backend can reasonably do so. Domain-edge results such
as `NaN`, positive or negative infinity, and signed zero remain tensor values;
they are not converted into typed domain errors merely because an input lies
outside the real-valued mathematical domain.

This policy applies to `F32` and `F64` scalar and elementwise numeric
operations, including division, remainder, square root, logarithms, reciprocal
square root, and similar analytic operations. It does not require a preflight
scan for zero, non-finite, or otherwise exceptional values.

Integer operations retain their existing structured domain checks. Integer
division and remainder by zero return `DivisionByZero`, and integer power with
a negative exponent returns its existing typed domain error. These checks are
needed because the corresponding integer operations do not have IEEE special
values and may otherwise panic, trap, or become backend-dependent.

## Boundary

IEEE value propagation does not turn structural failures into numeric values.
Invalid shapes, axes, dtypes, indices, layouts, device placement, backend
capabilities, and operation configurations continue to return typed errors.
Linear-algebra configuration errors and decomposition failures are also outside
this policy.

Complex operations follow the same principle where their implementation is
defined in terms of IEEE floating-point components. Operation-specific complex
behavior remains governed by the relevant operation contract.

When Rust, IEEE 754, NumPy, and JAX differ on a corner case, tenferro must state
the selected behavior explicitly. CPU/CUDA parity is required for the selected
result classification and for signed-zero behavior where bit-level parity is
part of the operation contract.

## Issue #1353 Contract

For floating-point division and remainder:

- a nonzero finite value divided by signed zero produces the corresponding
  signed infinity;
- zero divided by zero produces `NaN`;
- a floating-point remainder with a zero divisor produces `NaN`;
- `NaN` inputs propagate according to the underlying operation contract;
- signed-zero results follow the operation's IEEE semantics.

For integer division and remainder, a zero divisor continues to return
`DivisionByZero`.

No CPU or CUDA zero-divisor scan is added for floating-point division or
remainder.

## Documentation And Verification

The implementation change should:

1. Add the general floating-point domain policy to
   `docs/spec/tensor-semantics.md`.
2. Reference it from the elementwise primitive contract where useful, without
   duplicating the full policy.
3. Add focused CPU tests for `F32` and `F64` zero divisors, `NaN`, infinity,
   and signed zero.
4. Add ignored CUDA parity tests over the same cases.
5. Check result classes with `is_nan()` and `is_infinite()`, and use sign-bit or
   `to_bits()` checks only where signed-zero behavior is contractual.

If the current arithmetic kernels already satisfy this contract, production
code should remain unchanged. A neighborhood scan should inspect `div`, `rem`,
`sqrt`, `log`, `log1p`, and `rsqrt` for CPU/CUDA special-value parity. Any
independent discrepancy found there should become a focused follow-up issue
rather than expanding #1353 without a clear bound.

## Close Criteria

Issue #1353 can be closed when the active specification states the policy and
CPU/CUDA-focused tests preserve the documented division and remainder behavior.
The closing PR should use `Closes #1353` and describe the issue as a missing
numeric contract rather than changing floating-point zero divisors into errors.
