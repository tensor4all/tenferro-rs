# Integer Tensor Semantics

This note records the core integer arithmetic contract for the CPU and CUDA
tensor backends. It applies to erased `Tensor` dispatch, typed concrete tensor
helpers, eager execution, and traced execution once those layers route to the
same backend primitives.

## Scope

The first integer scope covers `I32` and `I64` core primitive operations in the
standard dense tensor path:

- arithmetic elementwise ops: add, sub, mul, neg, abs, div, rem, pow;
- ordering elementwise ops: maximum and minimum;
- integer reductions: `reduce_sum`, `reduce_prod`, `reduce_max`, and
  `reduce_min`;
- CPU and CUDA capability descriptor rows for the operations that are actually
  implemented.

Unsupported dtypes or operations must remain explicit in the capability table
and return structured backend or dtype errors. Capability documentation follows
implementation; a row must not be promoted to supported before the CPU and CUDA
paths are both covered by parity tests when CUDA support is claimed.

## Wrapping Arithmetic

Integer overflow is defined as two's-complement wrapping for CPU and CUDA. This
keeps debug and release CPU builds aligned and matches CUDA integer arithmetic
for the supported operations.

CPU integer kernels must therefore use explicit wrapping helpers for user data:

- add, sub, mul, neg, abs, and pow use the corresponding wrapping operation;
- `reduce_sum` folds with wrapping add from zero;
- `reduce_prod` folds with wrapping multiply from one;
- integer div follows truncation toward zero and treats `MIN / -1` as the
  wrapped minimum value;
- integer rem has the sign of the dividend and treats `MIN % -1` as zero.

Bare Rust integer operators are acceptable only inside helpers whose contract is
already the wrapping behavior, or for code that is not user-data arithmetic.

## Domain Errors

Integer division and remainder by zero must fail with a typed structured error,
not a panic or backend string failure. Integer pow accepts only non-negative
exponents; negative exponents must also return a typed structured error.

These checks must happen before CPU scalar arithmetic or CUDA kernel launches
that could otherwise trap, panic, or produce backend-dependent behavior.

## Ordering

Integer `maximum`, `minimum`, `reduce_max`, and `reduce_min` use the ordinary
signed total order for `I32` and `I64`. They do not participate in wrapping
arithmetic because they do not overflow.

## Verification

Every implementation step that marks an integer operation supported must include
focused CPU tests and, for CUDA rows, ignored CUDA parity tests that compare
against CPU on edge values. Edge coverage should include the minimum and maximum
representable values, scalar broadcast or read-view paths when those are
separate dispatch surfaces, reduction axes, zero divisors, negative exponents,
and `MIN / -1` or `MIN % -1` where relevant.
