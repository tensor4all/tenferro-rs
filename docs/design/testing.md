# Testing Strategy

## Overview

Tests are split into two layers:

1. **Unit tests** — inside the tenferro-rs workspace. Run via `cargo test` in seconds. No external data required.
2. **Benchmark / integration tests** — in the `tensor4all/benchmark_einsum` repository. Uses the einsum_benchmark dataset (pkl format).

## Unit Tests (per crate)

### tenferro-algebra

- Semiring axioms (associativity, distributivity, zero element, identity element)
- `Standard` algebra with f64 / Complex64

### tenferro-device

- `ComputeDevice` Display formatting
- Error type construction and display

### tenferro-tensor

- `Tensor<T>` creation, shape/strides accessors
- View operations (permute, reshape, broadcast) — shape correctness
- `contiguous()` data layout
- Error cases (shape mismatch, etc.)

### tenferro-prims

- Each primitive on small tensors (~2x3), compared against hand-computed values
  - GEMM, reduce, elementwise, trace, anti-trace, permute (copy)

### tenferro-einsum

- **Parser**: `"ij,jk->ik"` to `Subscripts` conversion
- **Basic patterns** (expected values hard-coded):
  - Matrix multiply: `ij,jk->ik`
  - Trace: `ii->`
  - Outer product: `i,j->ij`
  - Batch matmul: `bij,bjk->bik`
  - Transpose: `ij->ji`
  - Contraction: `ijk,ikl->ijl`
- **AD (rrule/frule)**: finite-difference gradient check

### tenferro-linalg

Test case parameters (shape, dtype, symm) are managed in a JSON file:
[`tenferro-linalg/tests/data/linalg_cases.json`](../../tenferro-linalg/tests/data/linalg_cases.json).
Test input matrices are randomly generated for each case.

**Random generation convention:**

- Real and imaginary parts drawn from uniform distribution `[-1, 1]`
- When `symm: true`, symmetrize via `A + A'` (Hermitian for complex)

**JSON schema:**

```json
{
  "svd": [
    {"shape": [3, 2], "dtype": "f64"},
    {"shape": [3, 2], "dtype": "c64"},
    ...
  ],
  "eigen": [
    {"shape": [4, 4], "dtype": "f64", "symm": true}
  ]
}
```

`symm` defaults to `false` when omitted.

#### Forward (decomposition correctness)

Due to phase/sign freedom, tests verify **reconstruction and properties**, not decomposition outputs directly.
BLAS/LAPACK do not specify sign/phase conventions, so reference data cannot be used.

| Operation | Reconstruction test | Property test |
|-----------|-------------------|---------------|
| SVD | `‖A − U·diag(S)·Vt‖ < ε` | `U'U ≈ I`, `V'V ≈ I`, `S ≥ 0` descending |
| QR | `‖A − Q·R‖ < ε` | `Q'Q ≈ I`, R is upper triangular |
| LU | `‖P·A − L·U‖ < ε` | L is unit lower triangular, U is upper triangular |
| Eigen (symmetric) | `‖A − U·diag(E)·U'‖ < ε` | `U'U ≈ I` |

All tests run automatically for each (shape, dtype) case in the JSON file.

#### AD (rrule): finite-difference gradient check

Ported from [BackwardsLinalg.jl](https://github.com/GiggleLiu/BackwardsLinalg.jl).
Source dump: `/tmp/BackwardsLinalg_dump.txt`

**Gradient check method:**

```
gradient_check(f, A; η=1e-5):
    g = analytic_gradient(f, A)          // computed via rrule
    dy_expect = η * sum(|g|²)            // expected change (first-order)
    dy = f(A) - f(A - η·g)              // actual change
    assert |dy - dy_expect| < rtol * |dy_expect| + atol
```

Tolerances: `rtol = 1e-2`, `atol = 1e-8` (same as BackwardsLinalg.jl).

**Scalar test functions and cotangent isolation:**

The gradient check requires a scalar function `f: Matrix → Scalar` to differentiate.
The choice of `f` determines which cotangent paths of the rrule are exercised:

- If `f` depends only on U (e.g., via `U[:,1]`), then dS = 0 and dV = 0,
  so only the dU branch of `svd_back` is tested.
- If `f` depends on multiple outputs, multiple cotangent branches are tested jointly.

Each cotangent branch should be tested in isolation first, then jointly,
to ensure individual branches are correct before testing their combination.

For each (shape, dtype) case in the JSON, all cotangent patterns are tested automatically:
- SVD: dU only, dV only, dS only, joint dU+dV
- QR: joint dQ+dR
- Eigen: dE only, dU only

**Scalar test functions per cotangent pattern (ported from BackwardsLinalg.jl):**

Reference: [GiggleLiu/BackwardsLinalg.jl](https://github.com/GiggleLiu/BackwardsLinalg.jl)

| Operation | Cotangent | Scalar test function | Rationale |
|-----------|-----------|---------------------|-----------|
| SVD | dU only | `real(ψ'Hψ)`, ψ=U[:,1] | Depends only on U → isolates dU |
| | dV only | `real(ψ'Hψ)`, ψ=V[:,1] | Depends only on V → isolates dV |
| | dS only | `sum(S)` | Depends only on S → isolates dS |
| | joint dU+dV | `real(conj(U[1,1])·V[1,1])` | Depends on U and V → tests joint path |
| QR | joint dQ+dR | `real(v'·op·v + v2'·op2·v2)`, v=Q[:,1], v2=R[2,:] | Both Q and R contribute |
| LQ | joint dL+dQ | same structure as QR | Both L and Q contribute |
| Eigen | dE only | `sum(E)` | Depends only on eigenvalues |
| | dU only | `real(v'·op·v)`, v=U[:,1] | Depends only on eigenvectors |

Here `H` and `op` are random Hermitian (or symmetric) matrices, generated independently of the test input `A`.

**Known gaps** (to be addressed in tenferro-rs):

- Degenerate singular/eigenvalues (stress test for `η` regularization)
- LU rrule (not in BackwardsLinalg.jl)
- frule (JVP) — BackwardsLinalg.jl only covers rrule

### chainrules-core / chainrules

- Tape: leaf registration, pullback execution
- TrackedTensor: tracking propagation
- DualTensor: tangent propagation
- Gradients: accumulate / get

## Benchmark / Integration Tests (`tensor4all/benchmark_einsum`)

- Use the [einsum_benchmark](https://benchmark.einsum.org/) dataset (pkl format, 168 problems) as-is
- Additional test cases can be added in the same pkl format
- tenferro-rs computation routines are placed in this repository
- Verification: compare `sum()` of computed result against `instance.result_sum`
- Also usable for performance regression testing (leveraging `../strided-rs-benchmark-suite` JSON metadata)
