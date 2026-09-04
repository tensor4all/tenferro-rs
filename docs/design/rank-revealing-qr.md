# Rank-Revealing QR

**Issue:** [#1754](https://github.com/tensor4all/tenferro-rs/issues/1754)

**Status:** approved for implementation. Independent `reviewer-flash` design review returned **APPROVED** with zero blocking findings after the threshold, zero-rank, tie, and AD-dispatch contracts were made explicit.

## Purpose

Adaptive scientific algorithms need a numerically rank-revealing orthogonal
factorization without downloading matrix payloads or moving execution to a
fallback backend. The existing `qr` operation is non-pivoted and returns only
thin `(Q, R)` factors. Rank decisions made downstream therefore duplicate
numerical policy and cannot compact arbitrary interspersed dependent columns.

This design adds column-pivoted rank-revealing QR (RRQR) as a distinct linalg
operation. It is not an SRC-specific primitive and does not change tensor
network storage.

## Why this is a distinct operation

Ordinary QR has two outputs and is differentiable on its full-rank domain.
RRQR additionally selects a discrete permutation and numerical rank. Making
ordinary QR always compute those outputs would add work and storage to its hot
path; making `QrOptions` change output arity would complicate every concrete,
eager, traced, extension, and AD wrapper.

The canonical APIs therefore remain conceptually separate:

```text
qr(A, options)              -> (Q, R)
rank_revealing_qr(A, opts)  -> (Q, R, permutation, rank)
```

They may share provider internals. API compatibility is not a design goal, but
semantic and performance separation is.

## Mathematical contract

For input `A` with shape `[m, n, batch...]`, let `k = min(m, n)`. RRQR returns:

```text
Q                [m, k, batch...]    input dtype
R                [k, n, batch...]    input dtype
column_permutation [n, batch...]     I64
rank             [batch...]          I64 (rank-0 for an unbatched matrix)
```

For each batch member:

```text
A[:, column_permutation] = Q R
Q^H Q = I
```

`column_permutation[j]` is the zero-based original input column placed at
factor column `j`. Providers normalize their native pivot representation to
this convention before returning.

Q and R retain their full thin shapes. The operation never returns a
data-dependent extent. A caller that needs compact factors reads the bounded
rank metadata explicitly and performs same-placement slicing. Fixed maximum
rank plus masks is not used.

## Options and rank semantics

```rust
pub struct RankRevealingQrOptions {
    pub gauge: QrGauge,
    pub rtol: f64,
    pub atol: f64,
}
```

Both tolerances must be finite and non-negative. For each batch member define

```text
threshold = max(atol, rtol * abs(R[0,0]))
rank = length of the leading diagonal prefix for which abs(R[i,i]) > threshold
```

There is no hidden scale floor. An empty diagonal or an all-zero matrix has
rank zero. With `atol = 0`, the rule is scale invariant for every finite,
nonzero scaling that remains representable. The strict comparison is part of
the contract. Any non-finite input or computed diagonal is a typed numerical
failure rather than a rank result. `QrGauge` affects Q/R phases but not
permutation or rank.

Column-pivoted QR makes `abs(R[0,0])` the largest input-column norm in exact
arithmetic. Provider roundoff can change a pivot tie or a rank exactly at the
threshold; cross-provider rank tests therefore use matrices separated from the
threshold by a documented tolerance margin, while each provider must implement
the same formula and deterministic lowest-column-index tie rule.

## Traced and eager representation

Rank and permutation are tensor outputs, never Rust `usize` or `Vec<usize>`.
This is required for batched execution and traced graphs.

The extension operation has fixed arity four and statically/symbolically
inferable output extents. Traced construction therefore needs no new dynamic
shape representation. Eager outputs stay in the input runtime. Direct and typed
surfaces use the same tensor-valued metadata contract; typed Q/R retain scalar
type while metadata uses `TypedTensor<i64>`.

`DynamicTruncate` is not embedded in RRQR. Its upper-bound extent is currently
not accepted by every compiled target, and hidden truncation would make output
shape data-dependent. Downstream eager callers may explicitly synchronize the
rank scalar and slice Q/R on their current placement.

## Backend contract

A hidden `LinalgBackend` hook returns the four ordered tensor outputs. Its
default is an explicit unsupported error. No backend may implement RRQR by
silently downloading A, constructing a CPU backend, or uploading factors.

### CPU-faer

Use faer's column-pivoted Householder QR implementation and the owning
`CpuExecutionContext`/buffer pool. Normalize faer's permutation to the public
zero-based gather convention and compute rank from R using the shared policy.

### CPU-BLAS/LAPACK

Use `xGEQP3` plus the matching `xORGQR`/`xUNGQR`. Workspace, dimensions, JPVT
conversion, and batch offsets are checked before FFI. LAPACK's one-based JPVT
is converted to the public zero-based convention.

### CUDA

CUDA 12.4 cuSOLVER does not expose `GEQP3`; the current tenferro binding only
has `GEQRF`. CUDA RRQR therefore requires a same-device column-pivoted
Householder implementation behind the linalg backend seam.

The implementation uses bounded host launch control (`k` factor steps) but no
payload or pivot readback. Device kernels/provider calls perform trailing
column norm updates, deterministic argmax with lowest-index tie breaking,
column swaps, reflector construction/application, permutation updates, and rank
reduction. Work is parallel over rows, trailing columns, and batches; no single
logical worker may loop over an unconstrained tensor-sized domain. All scratch
is session-owned and all operations use the active stream.

Only provider status and, at an explicit caller boundary, final rank metadata
may be synchronized to the host. CPU fallback is prohibited.

## AD semantics

Pivot selection and numerical rank are discontinuous. The initial delivery
marks every RRQR output unsupported for differentiation while preserving traced
primal execution. This is explicit in the linalg AD support manifest.

A later accepted AD phase may differentiate Q/R on regions with stable pivot
ordering by treating permutation as a fixed residual. It requires Torch or
finite-difference oracle coverage for real and complex dtypes. Permutation and
rank remain nondifferentiable metadata outputs. Rank-deficient and pivot-tie
points remain outside the differentiable domain.

## Validation and errors

Reject before provider execution:

- input tensor rank (number of dimensions) below two;
- unsupported dtype;
- invalid `rtol`;
- incompatible placement/runtime;
- dimensions or batch products that overflow provider integer/address bounds.

Non-finite numerical input behavior is explicit and provider-consistent; it
must not produce an apparently valid permutation/rank. Provider failures retain
their typed source chain.

## Testing

CPU-faer, CPU-BLAS/LAPACK, and CUDA tests cover F32/F64/C32/C64; tall, square,
wide, zero-size, zero matrix, duplicate and interspersed dependent columns,
scaled and ill-conditioned inputs, batches, and invalid options. Assertions
check permutation validity, reconstruction of `A[:, permutation]`, Q
orthogonality, rank, gauge, dtype, shape, and placement.

Traced tests compile and execute symbolic matrix and batch extents and verify
all four output metadata contracts. Missing backend support returns a typed
error. Source-contract tests reject hidden upload/download and CPU fallback in
CUDA execution.

## Delivery

1. Merge this design after independent review.
2. Add the fixed-arity operation, result types, metadata, direct/eager/traced
   surfaces, CPU providers, and exhaustive unsupported-AD dispatch/manifest
   entries; keep differentiation unsupported.
3. Add native CUDA execution and hardware/source-contract tests.
4. Update tensor4all-rs to consume RRQR, explicitly read only rank metadata,
   remove its local resident rank heuristic, and run the SRC CUDA matrix.

Each phase is independently reviewed and mergeable. The tenferro PR merges
before tensor4all updates its pinned revision.
