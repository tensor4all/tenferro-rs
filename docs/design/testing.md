# Testing Strategy

## Overview

Tests are split into two layers:

1. **Unit tests** — inside the tenferro-rs workspace. Run via `cargo test` in seconds. No external data required.
2. **Benchmark / integration tests** — external performance and compatibility gates run after correctness work is green.

For the current prims/linalg architecture, correctness work is intentionally
driven first. Performance verification is still required before merge, but it
is run as the final phase after the protocol changes compile and pass
functional tests.

## Performance Gates

The primary `einsum` regression gate is the sibling repository:

- `../tenferro-einsum-benchmark`

This benchmark suite is used to confirm that the protocol split does not
degrade the established `einsum` lowering path. In particular, the redesign
must preserve the expected CPU/GPU lowering shape:

- CPU: `permute view -> MakeContiguous -> BatchedGemm`
- GPU: `Contract` fast path when available, otherwise the same explicit
  structural/materialization path

`tenferro-tensor` and `tenferro-linalg` may also add crate-local microbenchmarks
for scalar or linalg-heavy paths, but `../tenferro-einsum-benchmark` remains
the top-level performance gate for contraction behavior.

## Unit Tests (per crate)

### tenferro-algebra

- Semiring axioms (associativity, distributivity, zero element, identity element)
- `Standard<f64>` and `Standard<Complex64>` algebra

### tenferro-tensor

- `Tensor<T>` creation, shape/strides accessors
- View operations (permute, reshape, broadcast) — shape correctness
- `contiguous()` data layout
- Error cases (shape mismatch, etc.)

### tenferro-internal-ops / tenferro-tensor

- Graph op payload, lowering, and shape metadata tests live with
  `tenferro-internal-ops`.
- Runtime tensor execution tests live with `tenferro-tensor`.
  - GEMM, reductions, elementwise ops, trace, anti-trace, and structural ops
    are checked on small tensors against hand-computed values.

### tenferro-einsum

Test cases are ported from [omeinsum-rs](https://github.com/tensor4all/omeinsum-rs) (`tests/`).
omeinsum-rs uses integer index labels (`&[0,1], &[1,2] -> &[0,2]`);
tenferro-einsum uses string subscripts (`"ij,jk->ik"`).
The translation is mechanical: same tensor data and expected values, different API calls.

#### Parser

`Subscripts::parse("ij,jk->ik")` — string to internal representation.
No omeinsum-rs equivalent (omeinsum-rs skips parsing, uses integer labels directly).
Write these tests from scratch.

#### Unary operations

Port from `tests/unary_ops.rs`. All use hand-computed expected values.

| Pattern | omeinsum-rs test | Notes |
|---------|-----------------|-------|
| Trace `ii->` | `test_trace_2x2`, `test_trace_3x3`, `test_trace_5x5` | |
| Diagonal `ii->i` | `test_diag_extract_2x2`, `test_diag_extract_3x3` | |
| Sum `ij->` | `test_sum_all` | |
| Sum axis `ij->j` | `test_sum_axis0`, `test_sum_axis1` | |
| Transpose `ij->ji` | `test_transpose_2x2`, `test_transpose_2x3` | |
| 3D permutation `ijk->kji` | `test_3d_permutation_full`, `test_3d_permutation_partial` | |
| Identity `ij->ij` | `test_identity_2d`, `test_identity_3d` | |
| Embed diagonal `i->ii` | `test_duplicate_vector_to_diagonal` | |
| Broadcast | `test_repeat_*` (4 tests) | |

#### Binary operations

Port from `tests/binary_rules.rs`. All use hand-computed expected values with explicit `size_dict`.

| Pattern | omeinsum-rs test | Notes |
|---------|-----------------|-------|
| Matmul `ij,jk->ik` | `test_matmul` | i=2, j=3, k=4 |
| Matmul transposed variants | `test_matmul_transposed_output/a/b`, `test_matmul_both_transposed` | All 4 combos |
| Dot product `i,i->` | `test_dot_product` | |
| Outer product `i,j->ij` | `test_outer_product` | |
| Hadamard `ij,ij->ij` | `test_hadamard_product` | |
| Batched matmul `bij,bjk->bik` | `test_batched_matmul` | b=2 |
| Vector-matrix `j,jk->k` | `test_vector_matrix` | |
| Matrix-vector `ij,j->i` | `test_matrix_vector` | |
| Scalar-tensor `,ij->ij` | `test_scalar_tensor`, `test_tensor_scalar` | |
| Diagonal contract `ii,ij->j` | `test_diagonal_contract` | |
| Multi-edge `ijk,jkl->il` | `test_multi_edge_contraction` | |
| 8D contraction | `test_8d_contraction` | All dims=2 |

#### N-ary operations and optimizer

Port from `tests/einsum_core.rs` and `tests/optimizer.rs`.

| Pattern | omeinsum-rs test | Notes |
|---------|-----------------|-------|
| 3-matrix chain `ij,jk,kl->il` | `test_3_matrix_chain` | |
| Star `ia,ib,ic->abc` | `test_star_contraction` | Hub variable |
| Cycle `ij,jk,ki->` | `test_tensor_network_cycle` | |
| 4-tensor cycle `ij,jk,kl,li->` | `test_cyclic_contraction` | |
| 5-tensor star | `test_5_tensor_star_contraction` | |
| `ContractionTree::optimize` | `test_greedy_*`, `test_treesa_*` | Greedy and TreeSA |
| Optimized vs pairwise | `test_optimized_vs_pairwise` | Results must match |

#### AD (einsum_rrule / einsum_frule)

Port from `tests/backward.rs`. Uses hand-computed expected gradients for small cases,
plus finite-difference verification from `tests/showcase.rs`.

| Pattern | omeinsum-rs test | Notes |
|---------|-----------------|-------|
| Matmul grad (all 4 transpose combos) | `test_backward_matmul_*` | f32, f64, Complex64 |
| Matmul with identity | `test_backward_matmul_identity` | dA = B^T, dB = A^T |
| Rectangular matmul grad | `test_backward_matmul_rectangular` | 2x3 * 3x2 |
| Trace grad | `test_backward_complex_trace` | Gradient = identity diagonal |
| Sum grad | `test_backward_complex_sum` | Gradient = all ones |
| Transpose grad | `test_backward_complex_transpose` | |
| 3-tensor chain grad | `test_backward_3tensor_chain` | Full chain rule |
| Finite-diff verification | `test_einsum_gradient_verification` (showcase.rs) | Central differences |

### tenferro-ext-tropical

Port from `tests/tropical.rs` and tropical-related tests in other files.

| Pattern | omeinsum-rs test | Notes |
|---------|-----------------|-------|
| MaxPlus/MinPlus associativity | `test_maxplus_associativity` | Semiring axioms |
| Distributivity | `test_tropical_distributivity` | |
| Identity elements | `test_tropical_identity`, `test_tropical_zeros_ones` | |
| Idempotent addition | `test_tropical_idempotent_addition` | a + a == a |
| MaxPlus matmul | `test_tropical_matmul_maxplus` (integration.rs) | Hand-computed |
| MinPlus matmul | `test_tropical_matmul_minplus` (integration.rs) | Hand-computed |
| MaxPlus chain | `test_tropical_chain` (integration.rs) | |
| Tropical unary ops | `test_tropical_unary_*` (unary_ops.rs) | trace, sum, row/col max |
| Tropical backward | `test_backward_tropical_matmul` (backward.rs) | Sparse gradients via argmax |
| Tropical argmax tie-break | new test | Verify that when multiple elements share the max, the gradient flows to the smallest-index element. Must produce identical results on CPU and GPU backends. |
| Shortest path (MinPlus) | `test_minplus_shortest_path` | Bellman-Ford step |
| Viterbi (MaxMul) | `test_viterbi_example` | |

### tenferro-linalg

The current linalg test suite is implemented directly in
[`crates/tenferro-linalg/tests/linalg_tests.rs`](../../crates/tenferro-linalg/tests/linalg_tests.rs).
It is a handwritten test matrix, not a generated JSON-driven harness.

The suite combines:

- Small deterministic fixtures for reconstruction/property tests and error paths
- Shared finite-difference helpers (`check_rrule_fd`, `check_frule_fd`, etc.)
- Targeted branch-coverage tests for tall/wide, batched, and rank-deficient cases
- Dtype coverage across `f64`, `f32`, `Complex64`, and `Complex32`

Test inputs are intentionally deterministic so failures are reproducible.
Some cases use fixed literals; others use helper-generated well-conditioned
or general matrices defined in the test file.

#### Crate-local benchmarks

`tenferro-linalg` also has a crate-local benchmark entry point:

- [`crates/tenferro-linalg/benches/linalg_benchmarks.rs`](../../crates/tenferro-linalg/benches/linalg_benchmarks.rs)

Run with:

```bash
cargo bench -p tenferro-linalg --bench linalg_benchmarks
```

The benchmark set includes forward kernels (`svd`, `qr`, `solve`,
`matrix_exp`) and representative AD rules (`svd_rrule`, `solve_rrule`)
across small/medium square, tall, wide, and batched-small shapes.

#### Forward (decomposition correctness)

Due to phase/sign freedom, tests verify **reconstruction and properties**, not decomposition outputs directly.
BLAS/LAPACK do not specify sign/phase conventions, so reference data cannot be used.

| Operation | Reconstruction test | Property test |
|-----------|-------------------|---------------|
| SVD | `‖A − U·diag(S)·Vt‖ < ε` | `U'U ≈ I`, `V'V ≈ I`, `S ≥ 0` descending |
| QR | `‖A − Q·R‖ < ε` | `Q'Q ≈ I`, R is upper triangular |
| LU | `‖P·A − L·U‖ < ε` | L is unit lower triangular, U is upper triangular |
| Eigen (symmetric) | `‖A − U·diag(E)·U'‖ < ε` | `U'U ≈ I` |
| Lstsq | `A'(Ax − b) ≈ 0` | `‖Ax − b‖` is minimized |

Forward coverage is provided by explicit per-operation tests, with separate
batched and dtype-specific checks where relevant.

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

The current handwritten suite covers the following cotangent patterns:
- SVD: dU only, dV only, dS only, joint dU+dV
- QR: joint dQ+dR
- LU: dL only, dU only, joint dL+dU
- Eigen: dE only, dU only
- Lstsq: dA only (fix b), db only (fix A)

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
| LU | dL only | `real(v'·op·v)`, v=L[:,1] | Depends only on L → isolates dL |
| | dU only | `real(v'·op·v)`, v=U[1,:] | Depends only on U → isolates dU |
| | joint dL+dU | `real(conj(L[1,1])·U[1,1])` | Both L and U contribute |
| Eigen | dE only | `sum(E)` | Depends only on eigenvalues |
| | dU only | `real(v'·op·v)`, v=U[:,1] | Depends only on eigenvectors |
| Lstsq | dA only | `x'·op·x`, x=A\b, fix b | Isolates A cotangent |
| | db only | `x'·op·x`, x=A\b, fix A | Isolates b cotangent |

Here `H` and `op` are random Hermitian (or symmetric) matrices, generated independently of the test input `A`.

**Known gaps:**

- Exact repeated-eigenvalue AD stress tests for general `eig` are not included.
  Current stress coverage focuses on SVD and symmetric/Hermitian `eigen`,
  where the implementation has explicit denominator regularization.

---

## AD Test Matrix

Coverage targets for reverse-mode (rrule/VJP), forward-mode (frule/JVP), and
Hessian-vector product (hvp) across all differentiable operations.

**Test ownership:**

- Unit tests for each rule live in the crate that owns the rule:
  - `crates/tenferro-einsum/tests/` — einsum AD tests
  - `crates/tenferro-linalg/tests/` — linalg AD tests
  - `chainrules/tests/` — tape/tracking infrastructure tests
- Workspace-level integration tests (in `tests/` at the workspace root) cover
  cross-crate AD scenarios: e.g., an einsum followed by an SVD inside a single
  tape, or C-API roundtrip correctness for AD.

### Einsum AD

| Operation | rrule | frule | hvp | Tropical-specific | Notes |
|-----------|-------|-------|-----|-------------------|-------|
| Matmul `ij,jk->ik` (Standard) | planned | planned | planned | — | Finite-diff + hand-computed |
| Trace `ii->` (Standard) | planned | planned | — | — | Gradient = identity diagonal |
| Sum `ij->` (Standard) | planned | planned | — | — | Gradient = all-ones |
| Transpose `ij->ji` (Standard) | planned | planned | — | — | |
| 3-tensor chain (Standard) | planned | planned | planned | — | Full chain rule |
| MaxPlus matmul (tropical) | planned | — | — | argmax route | Sparse gradient via argmax; GPU requires custom kernel |
| MinPlus matmul (tropical) | planned | — | — | argmax route | Same kernel requirement as MaxPlus |
| MaxPlus chain (tropical) | planned | — | — | argmax route | Gradient sparsity increases with chain length |

Notes:
- frule for tropical einsum is not planned: tropical algebra has no
  meaningful JVP (the `max` operation is not differentiable in the usual sense).
- hvp for tropical einsum is not planned for the same reason.
- argmax route testing may require a custom kernel infrastructure separate from
  cuTENSOR/hipTensor; CPU-only tests can run with the reference kernel.

#### Error path: `ModeNotSupported`

These tests verify the explicit error contract for unsupported AD modes
(issue #68). They live in `extension/tenferro-ext-tropical/tests/` and must not depend
on a full AD tape.

| Test | Expected result |
|------|----------------|
| Call tropical einsum frule (`MaxPlus`) | `Err(AutodiffError::ModeNotSupported { mode: "frule", .. })` |
| Call tropical einsum frule (`MinPlus`) | `Err(AutodiffError::ModeNotSupported { mode: "frule", .. })` |
| Call tropical einsum frule (`MaxMul`) | `Err(AutodiffError::ModeNotSupported { mode: "frule", .. })` |
| Call tropical einsum hvp (`MaxPlus`) | `Err(AutodiffError::ModeNotSupported { mode: "hvp", .. })` |
| Call tropical einsum hvp (`MinPlus`) | `Err(AutodiffError::ModeNotSupported { mode: "hvp", .. })` |

Example test structure:

```rust
#[test]
fn tropical_frule_returns_mode_not_supported() {
    let result = einsum_frule(/* tropical MaxPlus ctx */, "ij,jk->ik", &primals, &tangents);
    match result {
        Err(AutodiffError::ModeNotSupported { ref mode, .. }) => {
            assert_eq!(mode, "frule");
        }
        other => panic!("expected ModeNotSupported, got {other:?}"),
    }
}
```

### Linalg AD

All 14 rrule and 14 frule functions are implemented and tested with
finite-difference verification. AD formulas sourced from PyTorch autograd
and Mathieu (2019).

| Operation | rrule | frule | FD status | Notes |
|-----------|-------|-------|-----------|-------|
| SVD | done | done | pass | Per-cotangent-branch FD checks (dU, dS, dVt) |
| QR | done | done | pass | Full-rank and wide-case FD coverage |
| LU | done | done | pass | Square, wide, and tall pullback/pushforward coverage |
| Eigen (symmetric) | done | done | pass | dE only, dU only |
| Eig (general) | done | done | pass | Complex output |
| Cholesky | done | done | pass | |
| `solve` | done | done | pass | dA and db branches |
| `lstsq` | done | done | pass | Includes residual-term pullback |
| `inv` | done | done | pass | |
| `det` | done | done | pass | |
| `slogdet` | done | done | pass | |
| `pinv` | done | done | pass | SVD-based |
| `matrix_exp` | done | done | pass | Pade[13/13] scaling-and-squaring |
| `norm` | done | done | pass | Fro, Nuclear, Spectral |
| `solve_triangular` | — | — | — | Forward-only utility, no AD rules |

Notes:
- hvp for linalg operations is not planned. Second-order differentiation
  through linalg (e.g., SVD Hessians) is mathematically complex and deferred.
- All linalg AD tests use central finite-difference verification
  (`eps = 1e-6`, `atol = 1e-4`).
- `tenferro-linalg` AD rules depend only on `chainrules-core` (not the full
  `tidu` engine); test infrastructure calls `svd_rrule` etc. directly
  without requiring a tape.

---

### chainrules-core / chainrules / tidu

- `chainrules-core`: shared AD traits and rule interfaces
- `chainrules`: engine-independent scalar rules and scalar forward/reverse helpers
- `tidu`: tape, tracked values, dual values, and pullback execution

## Benchmark Tests (`tensor4all/benchmark_einsum`)

Performance benchmarks for einsum, using instances selected from
[einsum_benchmark](https://benchmark.einsum.org/) (same selection as
[strided-rs-benchmark-suite](https://github.com/tensor4all/strided-rs-benchmark-suite)).

**Data stored**: metadata only (shapes, format strings, contraction paths) in JSON.
No tensor data — tensors are generated at benchmark time (zero-filled or random).
Correctness is verified by unit tests (see tenferro-einsum section above), not here.

The repository contains tenferro-rs benchmark runner code for performance regression testing.
