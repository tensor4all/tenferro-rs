# Session-Scoped BLAS-1 Primitives

## Status

Accepted design for issue #1713. Implementation requires an independent design-review verdict before code changes.

## Goal

Backend-session callers implement allocation-conscious Krylov iterations over `TensorRead` and caller-owned output storage without host loops, hidden transfers, full-vector algebra temporaries, or solver policy in tenferro.

## Session contract

Add object-safe default methods directly to `BackendSession`:

```rust
fn vdot_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> Result<Tensor>;
fn norm_squared_read(&mut self, input: TensorRead<'_>) -> Result<Tensor>;
fn axpby_read_into_accum(
    &mut self,
    alpha: ContractionScalar,
    x: TensorRead<'_>,
    beta: ContractionScalar,
    y: TensorWrite<'_>,
) -> Result<()>;
```

Defaults return typed `Unsupported`. They never transfer, construct a CPU backend, or fall back. CPU backend/session implementations override all three. No new operation-family crate or solver API is introduced.

`ContractionScalar` is reused rather than adding another scalar enum. `alpha` and `beta` must have the exact tensor dtype; callers represent real coefficients for complex vectors as complex values with zero imaginary part. Supported tensor dtypes are F32, F64, C32, and C64.

## Semantics and validation

Shared backend-neutral validation runs before allocation or destination mutation:

- `vdot_read`: identical dtype, shape, element count, and compatible placement; result is a rank-0 tensor of the input dtype; computes `sum(conj(lhs) * rhs)`.
- `norm_squared_read`: supported dtype and placement; result is rank-0 F32 for F32/C32 and F64 for F64/C64; computes the real nonnegative sum of squared magnitudes without `sqrt`.
- `axpby_read_into_accum`: exact x/y shape, dtype, placement, scalar dtype, compact/injective destination, and conservative x/y non-overlap. All failures occur before mutation. It computes `y <- alpha*x + beta*y` in one destination pass.

Any x/y storage overlap is rejected with the existing typed destination-overlap validation error. Supporting exact alias is unnecessary and would weaken the simple mutation proof. Empty tensors preserve the same contracts: dot/norm return zero and AXPBY is a validated no-op.

## CPU implementation

### VDOT

Reuse the existing optimized dot-general provider. Contract every axis, allocate one rank-0 output through the session buffer pool, and set `DotGeneralAccumulation::lhs_conj = true`. This consumes compact or strided reads through the provider's existing same-placement planning/materialization boundary and creates no full-vector conjugation temporary.

### Norm squared

Use `strided_kernel::reduce` directly over the input view with a squared-magnitude map and real addition identity. Dispatch dtype once. Execute under the owning `CpuExecutionContext`; no `conj(x)*x` tensor is created.

### Fused AXPBY

No current strided-rs primitive expresses read-modify-write over one destination. Issue #1713 is the accepted benchmark-backed exception: implement one compact destination pass in tenferro-cpu, with the rationale and `// INVARIANT:` mutation/alias proof at the kernel. Compact x is borrowed; a noncompact same-placement x view is explicitly materialized exactly once and reused. Noncompact y is rejected rather than hidden copy-out.

The loop executes inside the repository CPU context. One-thread sessions use a serial zip loop; multi-thread sessions use Rayon only inside `CpuExecutionContext::with_native_parallelism`, with disjoint compact destination chunks. It allocates no full-size temporary.

## Tests and benchmark

- CPU F32/F64/C32/C64 compact values for all operations against scalar references; complex VDOT conjugates only lhs.
- Strided lhs/rhs VDOT and norm consume provider/strided views; strided x AXPBY records exactly one materialization; compact paths record none.
- Invalid dtype/shape/placement/scalar/overlap/noncompact-y cases prove destination bytes unchanged.
- A fake unsupported session returns typed `Unsupported` and records no transfer/fallback.
- Allocation instrumentation proves norm and fused AXPBY allocate no full-size temporary; VDOT allocates only rank-0 output plus provider-bounded bookkeeping.
- A downstream-style CG microfixture uses `copy_read_into`, VDOT, norm-squared, and AXPBY without manual element loops.
- A Criterion benchmark compares fused AXPBY with the prior composed/manual reference at representative lengths and one/multiple threads. Run it in release mode and record medians, allocation behavior, and the provider/reduction numerical-order note in the worklog.

## Numerical order

VDOT follows the selected dot provider's reduction order. Norm-squared follows strided-kernel reduction order. Results are tested with dtype-appropriate tolerances rather than bitwise equality. AXPBY is elementwise and deterministic for a fixed scalar implementation.

## Non-goals

- No CG/BiCGStab/multi-shift solver, stencil, shift, or convergence policy.
- No additional BLAS-1 catalog.
- No implicit transfer, host scalar extraction, or CPU fallback.
- No n-ary linear combination.
- No GPU implementation claim; unsupported sessions remain explicit until a backend overrides the methods.

## Verification

Run tensor contract tests/doctests, CPU faer and BLAS feature tests, allocation/alias/placement tests, the downstream microfixture, release benchmark, clippy, modified-file coverage review, and combined PR gates.
