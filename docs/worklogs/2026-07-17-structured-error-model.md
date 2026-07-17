# Structured Error Model

Date: 2026-07-17

## Scope

This work implements the approved structured error model from
`docs/superpowers/specs/2026-07-17-structured-error-model-design.md` and the
ordered plan in `docs/superpowers/plans/2026-07-17-structured-error-model.md`.
The branch starts from `origin/main` at `1ec2062e` and contains only the
approved design and plan commits before implementation.

## Classification policy

The migration applies one policy at every crate boundary:

1. Caller-controlled tensor relationships use shared typed validation payloads
   (`ShapeMismatch`, rank, axis, dtype, or `InvalidArgument`).
2. Unsupported operations and dtypes use `Unsupported`, with a typed local
   source when the domain owns a more specific reason.
3. Numerical failure (singularity, non-convergence, division by zero, and
   related numeric-domain failures) uses `NumericalFailure` and retains the
   typed local source.
4. Typed in-workspace backend/kernel failures use `BackendSource`; vendor
   status text with no typed source is the only remaining `BackendFailure` text
   case.
5. Typed file, stream, serialization, or dynamic-library failures use
   `ErrorKind::Io` and retain their source. They are not backend failures just
   because they occurred during backend execution.
6. Missing, uninitialized, poisoned, or invalid executor/cache/device state
   uses `ErrorKind::RuntimeState`, retaining a typed source when available.
   Impossible invariants remain `Internal`.

`ErrorPhase` is orthogonal to the category: eager and traced paths use the
same validation payload and classification for the same known input, while
the phase records whether the fact was discovered during graph construction,
compilation, or execution.

## Checkpoints

- `109a8d11`: shared tensor validation payloads and boxed shape mismatch.
- `13bcf73d`: tensor outer source-chain model and public error docs.
- `772d9c4e`: CPU structured validation and typed backend sources.
- `89d3ab42`: CUDA structured errors, including typed unsupported-dtype source.
- `1a9b8b4f`: runtime and default-feature extension migration.

Further checkpoints will record the linalg, optional extension, audit, and
release-gate work below.

## Verification notes

The CPU all-target test suite ran 311 unit and 43 integration tests before the
benchmark failed because the local environment cannot resolve the requested
NUMA/affinity placement:

```text
faer AllAllowed placement should resolve: ManagedAffinityUnavailable {
    requested: AllAllowed,
    backend: Faer,
}
```

This is an environment limitation, not a passing claim for the complete
all-target command. The exact optional WebGPU dependency evidence will be
recorded after the isolated clean-`origin/main` comparison.

## Residual risks

Hardware-dependent CUDA/WebGPU/PJRT execution and hosted affinity/NUMA
benchmarks require their documented environments. No dependency revision or
lint allowlist is added to hide those limitations.
