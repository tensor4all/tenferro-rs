# Borrowed Linalg Solve Design

## Goal

Resolve #1267 by allowing `solve` and `triangular_solve` to consume owned tensors or borrowed tensor views through `TensorRead`, while preserving the existing owned API and backend capability boundaries.

## Selected API

Add two methods to `LinalgBackend`:

```rust
fn solve_read(
    &mut self,
    a: TensorRead<'_>,
    b: TensorRead<'_>,
) -> tenferro_tensor::Result<Tensor>;

fn triangular_solve_read(
    &mut self,
    a: TensorRead<'_>,
    b: TensorRead<'_>,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> tenferro_tensor::Result<Tensor>;
```

The default implementations return typed `Error::Unsupported`, matching the other `LinalgBackend::*_read` methods. They do not silently materialize inputs because materialization policy belongs to the implementing backend.

The existing owned `solve` and `triangular_solve` methods remain unchanged.

## CPU Execution

`CpuBackend` overrides both methods. It validates both read targets as host-accessible tensors, checks dtype pairs consistently with the owned path, and dispatches according to the configured linalg provider.

For Faer, supported strided views use provider view entry points where the existing linalg layout contract permits them. For BLAS/LAPACK, and for Faer layouts outside that contract, the CPU backend explicitly canonicalizes each operand with its pooled materialization path before calling the existing owned implementation. Offset and transposed views must therefore match the owned path without introducing a trait-level hidden fallback.

Both operands are resolved before numerical execution. Invalid dtype pairs, incompatible shapes, singular systems, non-host placements, and provider failures retain the existing typed error vocabulary.

## Alternatives Rejected

1. **Default trait canonicalization.** Rejected because `LinalgBackend` does not own a universal materialization policy, and a hidden fallback would weaken backend capability reporting.
2. **`TensorView`-only signatures.** Rejected because #1420 standardized the linalg borrowed-input family on `TensorRead`, which also accepts owned `&Tensor` values.
3. **CUDA borrowed-view overrides in this issue.** Rejected because #1267 only requires CPU providers; unsupported backends should keep the explicit default error rather than adding unrequested canonicalization or transfer behavior.

## Documentation

Each new trait method includes a compiling example using `TensorRead::from_view`. Error documentation names validation, numerical, unsupported-boundary, and typed provider failures. Existing owned examples and methods remain valid.

## Verification

- A backend that relies on the defaults returns typed `Unsupported` for both methods.
- `TensorRead::from_tensor` is accepted by the CPU overrides.
- Offset/strided view operands produce the same results as equivalent owned operands for `solve` and `triangular_solve`.
- Faer and BLAS/LAPACK feature configurations compile and run their focused tests.
- Dtype mismatch, shape mismatch, singular-system, and non-host placement behavior stays typed and panic-free.
- Existing linalg unit, integration, and doctest suites remain green.

## Scope Boundary

This change adds no concrete extension trait, eager/traced operation, new provider, device transfer, or GPU implementation. Those remain separate issues.
