# Binary Contraction Pipeline

This document describes the current binary step used inside an N-ary einsum
contraction tree. It is implemented through `tenferro-einsum` planning and the
`TensorBackend::dot_general` execution surface in `tenferro-tensor`.

See [einsum.md](./einsum.md) for the N-ary API and
[tensor-prims.md](./tensor-prims.md) for the current tensor backend protocol.

---

## Step Inputs

Each pairwise step receives:

- left operand labels,
- right operand labels,
- output labels for this intermediate or final result,
- label sizes from the tree's size dictionary.

Repeated labels are normalized before ordinary pairwise contraction. Strict
binary lowering only accepts non-repeated labels; repeated-label cases fall back
to the general path.

## Label Classification

For a binary contraction, labels are classified into:

| Class | Meaning |
| --- | --- |
| left free | appears only in the left operand and survives |
| right free | appears only in the right operand and survives |
| batch | appears in both operands and survives |
| contraction | appears in both operands and does not survive |
| pre-reduced | appears in one operand and does not survive |

Pre-reduced labels are reduced before the `dot_general` step.

## Column-Major Layout

tenferro uses column-major storage. The canonical `DotGeneral` output order is:

```text
lhs_free..., rhs_free..., batch...
```

The strict binary lowering tracks GEMM-like metadata with compute dimensions on
the left and batch dimensions on the right:

```text
lhs matrix dims: lhs_free..., contract...
rhs matrix dims: contract..., rhs_free...
output dims:     lhs_free..., rhs_free..., batch...
```

Batch-last placement means each batch slice occupies a contiguous block in
column-major layout, which is the convention used by the CPU GEMM path.

## Strict Binary Lowering

`crates/tenferro-einsum/src/planning/strict_binary.rs` detects dense binary
contractions that can be represented as one canonical `DotGeneral` plus a final
output transpose if needed.

The strict plan records:

- left and right free labels,
- contraction labels,
- batch labels,
- operand permutations,
- `m`, `k`, and `n`,
- canonical output dimensions,
- final output permutation.

It returns `None` for patterns it should not own, including repeated labels.
That is not an error; the general path handles those semantics.

## General Path

The eager executor and graph builder use the general path when strict binary
lowering does not apply:

1. Extract diagonals for repeated input labels.
2. Reduce labels that no longer survive.
3. Broadcast and multiply for pure outer/Hadamard-like cases.
4. Use `dot_general` for ordinary contractions.
5. Embed repeated output labels with `embed_diagonal`.
6. Transpose to the requested output label order.

This path favors correctness and broad semantics. Backend-specific performance
comes from the lower `TensorBackend` implementation of `dot_general`, reduction,
transpose, and diagonal operations.

## CPU Execution

CPU execution is handled by `CpuBackend`:

- elementwise/reduction/structural work uses strided-kernel and dedicated CPU
  implementations,
- `dot_general` uses the selected CPU provider (`cpu-faer`, `cpu-blas`, or
  an optional external general-contraction provider for supported contractions),
- faer-backed work runs inside the `CpuContext` Rayon pool through
  `CpuExecSession`, with `Par::Seq` for one-thread contexts and explicit
  `Par::rayon(n)` from the configured degree for multi-thread contexts,
- BLAS/LAPACK provider threading remains provider-owned; the external TBLIS
  example clamps TBLIS to one thread for each provider call.

External general-contraction providers are additive to the faer/BLAS providers.
`CpuBackendKind` still selects the complete base provider, while
`CpuProviderBundleBuilder` controls whether `dot_general` attempts an external
provider first. Unsupported shapes and unavailable optional runtimes fall back
to the selected base provider unless the provider is required.

## CubeCL/CUDA Execution

When a compiled program is evaluated with `CudaBackend`, the same graph-level
ops run on GPU if supported:

- `dot_general` can route through CubeCL/CUDA and cuTENSOR/cuBLAS-backed paths,
- structural and reduction steps use CubeCL kernels where implemented,
- unsupported dtypes or operations return `BackendFailure`.

There is no current separate direct `CudaBackend` contraction pipeline. Future
GPU contraction improvements should extend `CudaBackend` and preserve the
explicit placement policy described in [gpu-backend-design.md](./gpu-backend-design.md).

## Planner Safety

Contraction planning must not rely on debug-only assertions for semantic
validity. Missing label sizes, invalid explicit paths, duplicate live operands,
and rank/shape mismatches are normal `Result` errors so debug and release builds
behave the same.
