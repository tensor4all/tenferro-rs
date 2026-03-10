# Tensor Prims Protocol Families

`tenferro-prims` is the execution substrate for tensor operations that are
valid over a semiring or over standard scalar arithmetic. The redesign splits
the old monolithic `TensorPrims<A>` surface into smaller protocol families so
`einsum`, tropical algebra, scalar math, and linalg can depend on only the
capabilities they actually need.

## Layering

```
TensorStructural (tenferro-tensor views)
    permute, reshape, broadcast, diagonal, narrow
        │
        ▼
TensorSemiringCore<Alg>
    BatchedGemm, ReduceAdd, Trace, AntiTrace, AntiDiag, MakeContiguous
        │
        ├── required by tenferro-einsum
        └── sufficient for tropical + minimal AD support

TensorSemiringFastPath<Alg>
    Contract, ElementwiseBinary { Add, Mul }
        │
        └── optional performance paths, never required for correctness

TensorScalarPrims<Alg>
    pointwise unary/binary ops and scalar reductions for standard arithmetic

TensorAnalyticPrims<Alg>
    exp/log/trig/pow/variance-style vocabulary
```

The key design rule is:

- `tenferro-einsum` may depend on `TensorSemiringCore` only.
- `TensorSemiringFastPath` is optional and must not change semantics.
- Structural view operations stay in `tenferro-tensor`, not in prims.
- Scalar and analytic vocabulary must not leak into semiring-only backends.

## Execute Contract

All prim families keep the BLAS/cuTENSOR-style contract:

```text
output <- alpha * op(inputs) + beta * output
```

This contract is preserved even for the new family traits so existing high
performance lowering strategies remain valid.

## Why `Permute` Leaves Prims

`Tensor::permute()` is already a zero-copy structural view. Keeping an eager
`Permute` execution primitive in the required core would force semiring-only
backends to implement a materializing reorder that is not needed for
correctness. The redesign therefore treats:

- `permute` as a `tenferro-tensor` view operation
- `MakeContiguous` as the execution primitive that materializes a view when
  needed

This is especially important for `einsum`, where the intended lowering is:

```text
permute view -> MakeContiguous -> BatchedGemm
```

## Minimal Semiring Core

`TensorSemiringCore<Alg>` is intentionally small:

| Operation | Why it is in core |
|-----------|-------------------|
| `BatchedGemm` | fundamental contraction primitive |
| `ReduceAdd` | semiring addition reduction needed by einsum lowering |
| `Trace` | semiring-valid diagonal contraction |
| `AntiTrace` | AD adjoint of trace |
| `AntiDiag` | AD adjoint of diagonal extraction/embedding |
| `MakeContiguous` | explicit view materialization boundary |

Notably absent from core:

- `Contract`
- `ElementwiseBinary { Add, Mul }`
- `ReduceMul`
- `Maximum`, `Minimum`, `Div`, `Exp`, `Log`, and other ordered/analytic ops
- linalg factorizations and solves

Those belong in `TensorSemiringFastPath`, `TensorScalarPrims`,
`TensorAnalyticPrims`, or `tenferro-linalg-prims`.

## Fast Paths

`TensorSemiringFastPath<Alg>` holds operations that are semiring-valid but
optional:

- `Contract`
- `ElementwiseBinary { Add, Mul }`

The public descriptor is generalized, but backend implementations are expected
to re-specialize at `plan()` time so hot loops stay as efficient as the old
specialized kernels.

## Current Implementation Status

The repository is in the middle of the protocol migration.

- `TensorSemiringCore`
- `TensorSemiringFastPath`
- `TensorScalarPrims`
- `TensorAnalyticPrims`

already exist as public traits in `tenferro-prims`.

Today, their implementations are provided through blanket adapters over the
legacy `TensorPrims<A>` surface. That compatibility layer exists only as a
migration mechanism; new high-level code should target the family traits.

The same migration note applies to the current scalar and analytic families:
their public vocabulary is broader than the set of operations actually wired to
legacy execution today.

## Explicit Debt

Two points are intentionally documented as current debt rather than final
design:

- legacy `PrimDescriptor::Permute` still exists in `tenferro-prims`
- the scalar and analytic families still rely on blanket adapters over the old
  `TensorPrims<A>` contract

The `Permute` cleanup remains follow-up work under `#441`.

## Relationship to Linalg

`tenferro-prims` is not responsible for structured factorizations like QR, SVD,
LU, Cholesky, or eigendecomposition. Those live in
[linalg-prims.md](./linalg-prims.md).

The split is intentional:

- `tenferro-prims` owns semiring/scalar execution substrate
- `tenferro-linalg-prims` owns backend-facing linalg kernel contracts
- `tenferro-linalg` owns public APIs and composite lowering

## Performance Invariants

The redesign keeps three invariants:

1. `einsum` keeps its existing lowering shape: structural views plus explicit
   materialization and GEMM.
2. Generalized descriptors must specialize during planning, not per element.
3. Optional fast paths are never required for correctness and may be absent on
   semiring-only or tropical backends.
