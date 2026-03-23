# Integer/Bool Metadata Substrate Design

**Date:** 2026-03-23

**Problem**

tenferro still collapses LU metadata to host vectors too early. In particular,
`LuTensorResult` and `LuTensorExResult` currently expose host `Vec<i32>` pivots,
which forces `det`, `slogdet`, and `lu_solve` to rebuild sign/permutation data
on the host. That blocks a clean GPU-generic path.

PyTorch does not have this problem because LU metadata remains tensor-native:

- `linalg_lu_factor` returns `(LU, pivots)`
- `linalg_lu_factor_ex` returns `(LU, pivots, info)`
- `det/slogdet` derive permutation parity from `pivots` tensor operations
- `lu_solve` accepts pivot tensors directly

References:

- [torch/linalg/__init__.py](/home/shinaoka/tensor4all/pytorch/torch/linalg/__init__.py#L2406)
- [LinearAlgebra.cpp](/home/shinaoka/tensor4all/pytorch/aten/src/ATen/native/LinearAlgebra.cpp#L356)
- [BatchLinearAlgebra.cpp](/home/shinaoka/tensor4all/pytorch/aten/src/ATen/native/BatchLinearAlgebra.cpp#L2021)

## Decision

We will align with PyTorch by introducing a dedicated integer/bool metadata
substrate before changing LU public APIs.

The design has four layers:

1. `tenferro-device`
   - low-level runtime support for `i32` and bool-like metadata tensors
   - pointwise, select, and reduction kernels needed for pivot/info workflows

2. `tenferro-prims`
   - a dedicated metadata family rather than stretching the existing numeric
     scalar family across integer/bool semantics
   - CPU and CUDA implementations over the same descriptor set

3. `tenferro-linalg-prims`
   - LU backend contracts return tensor metadata:
     - `pivots: Tensor<i32>`
     - `info: Tensor<i32>`

4. `tenferro-linalg`
   - `det`, `slogdet`, `lu_solve`, and LU public results compose metadata
     tensors directly

## Why not a LU-specific parity helper first?

That would remove the immediate `det/slogdet` host loop, but it would not align
with PyTorch's model and would leave `lu_solve`, `lu_unpack`, LDL pivots, and
future rank/mask paths on an incompatible substrate. The user goal here is
low-level substrate alignment, not another local workaround.

## Why a dedicated metadata family instead of extending `TensorScalarPrims`?

PyTorch can use one dynamic ATen surface for all dtypes because dtype is a
runtime property. tenferro is statically typed and its scalar family currently
assumes numeric operations that do not map cleanly onto integer/bool metadata.
A dedicated metadata family keeps the supported op set explicit and avoids
overloading float/complex semantics.

## Initial metadata op set

The first tranche should cover only the closed set needed by LU-based paths:

- `iota/arange`
- `eq`, `ne`
- `where`
- `sum`
- `all`, `any`
- broadcast/expand/contiguous over metadata tensors

This is intentionally broader than a parity-only helper, but narrower than a
full ATen integer algebra.

## Public surface target

Public LU APIs should eventually match PyTorch as closely as Rust ergonomics
allow:

- `lu_factor -> (LU, pivots: Tensor<i32>)`
- `lu_factor_ex -> (LU, pivots: Tensor<i32>, info: Tensor<i32>)`
- `lu_solve(LU, pivots, B)`
- `lu` should move toward `(P, L, U)` semantics instead of `Option<Vec<usize>>`

The canonical pivot semantics should match PyTorch:

- `torch.int32`
- shape `(*, n)` or `(*, k)` as applicable
- 1-indexed pivot steps

## Non-goals for this tranche

- full integer numeric algebra
- full bool public tensor ergonomics if an internal bool-like representation is
  sufficient for the first tranche
- immediate cleanup of every AD helper that still depends on host slices

## Success criteria

- `det`, `slogdet`, and `lu_solve` no longer build host sign/permutation data
  from backend pivot vectors
- LU backend metadata stays tensor-native on CUDA
- public LU surfaces no longer force a host `Vec` contract
- the added substrate is reusable for LDL pivots, rank/mask-style workflows,
  and future pivot-aware linalg paths
