# Faer Linalg Safety Dispatch Design

**Date:** 2026-05-02

**Status:** Proposed dispatch spec

## Issues

Primary:

- #768: BUG: debug_assert_eq! used before transmute, UB in release builds
- #781: BUG: SVD/EIGH/eig decomposition failure panics via unwrap_or_else

Related if the same files are already being edited:

- #803: BUG: faer_gemm beta-scaling loop has no guard against zero stride
- #805: BUG: check_singular_diagonal panics on rank<2 tensors
- #807: BUG: batched linalg helpers use assert!/assert_eq! for validation
- #814: BUG: assert!/panic! validation in CPU indexing/linalg/BLAS helpers

## Goal

Remove release-mode undefined-behavior risk and library panics from faer-backed
linalg paths by converting invariant checks and decomposition failures into
explicit contracts or `Result` errors.

## Scope

This dispatch covers faer-backed CPU linalg and tightly adjacent helpers in
`tenferro-tensor`.

It does not cover:

- adding F32/C32 linalg support (#629, #797),
- GPU linalg performance (#770),
- provider-inject ABI work (#760),
- adding new AD rules for linalg (#772, #777),
- changing the one-CPU-backend feature contract.

## Acceptance Specification

### Complex slice reinterpretation

The conversion helpers that reinterpret `num_complex::Complex64` slices as faer
complex slices must not rely only on `debug_assert_eq!` before unsafe pointer
reinterpretation.

Acceptable outcomes:

- use always-on assertions for layout invariants that are compile-time facts in
  supported builds, or
- mark the helpers `unsafe` and document exact safety requirements, while
  ensuring every call site satisfies them.

The preferred outcome is always-on layout validation with a short comment
explaining why the representation is expected to match.

### Decomposition failure

SVD, EIGH, and eig paths must return `Err` on decomposition failure instead of
panicking through `unwrap_or_else`.

The error should preserve operation context. Use existing error variants where
possible rather than introducing a new public error type.

### Validation panics

If nearby linalg validation currently uses `assert!` on user-controlled shapes
or ranks, convert only the narrow cases listed above or directly touched by the
same call path. Do not broaden into a full linalg validation refactor.

## Design

Keep the faer implementation structure intact. This dispatch should be a safety
and error-propagation patch, not a linalg architecture change.

For decomposition calls:

1. replace panic adapters with `map_err(...)` or equivalent,
2. return through the existing `Result` path,
3. update callers only as needed to propagate `Result`.

For unsafe slice conversions:

1. keep the helper small,
2. make the invariant check visible in release builds or documented as an
   unsafe contract,
3. avoid duplicating conversion logic across f64 and complex paths.

## Testing

Required tests:

- a failure-mode test for at least one faer decomposition path that used to
  panic and now returns `Err`,
- a rank or shape validation test if any related validation panic is changed,
- existing linalg success tests must still pass.

Run at least:

```bash
cargo test -p tenferro-tensor linalg
cargo fmt --all --check
```

If focused filters are too coarse, run the relevant tenferro-tensor linalg test
target or the crate tests.

## Dispatch Prompt

```text
Implement the faer linalg safety dispatch from
docs/plans/2026-05-02-faer-linalg-safety-design.md.

Limit the patch to faer-backed CPU linalg safety and error propagation. Remove
release-only debug assertions before unsafe reinterpretation, return Err instead
of panicking on SVD/EIGH/eig decomposition failures, and add focused regression
tests. Do not add F32/C32 linalg, provider-inject support, GPU linalg work, or
new linalg AD rules.
```

## Review Checklist

- No decomposition failure path uses `panic!` or `unwrap_or_else(|_| panic!)`.
- Unsafe slice reinterpretation has an always-enforced invariant or documented
  unsafe contract.
- Error messages name the failing operation.
- Tests cover an error path, not only success paths.
- The patch does not change linalg public semantics beyond panic-to-error.

## Stop Conditions

Stop and report if:

- faer APIs do not expose recoverable failure information for a target op,
- converting a panic to `Err` requires changing public trait signatures,
- a layout invariant cannot be justified for all supported platforms.
