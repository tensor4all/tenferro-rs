# Routing `tenferro-linalg` GEMM Paths Through `tenferro-prims`

## Summary

Issue #245 should be treated as a focused GEMM-path cleanup, not as a full
backend rewrite.

`tenferro-prims` already owns the shared tensor primitive for batched matrix
multiplication via `PrimDescriptor::BatchedGemm`. The remaining goal is to make
`tenferro-linalg` reuse that primitive wherever the internal operation is
semantically a batched GEMM.

This document fixes the intended migration boundary before runtime changes
begin.

## Current State

Today `tenferro-linalg` still has internal CPU-centric GEMM helpers:

- `backend_mat_mul_nn`
- `backend_mat_mul`
- `complex_mat_mul_nn`

These helpers are used in:

- `matrix_exp`
- many AD rule implementations (`rrule` / `frule`)
- some complex eigendecomposition helper paths

The mathematical operation is often still “matrix multiply”, but the current
entry points are slice-based and call the local `LinalgBackend::mat_mul` path
instead of going through `TensorPrims`.

## Goal

The first real implementation step for #245 is:

- route tensor-expressible GEMM work through `tenferro-prims`
- keep decomposition and solve kernels out of scope
- avoid changing numerical semantics

This should improve consistency across layers and make the later GPU path
straightforward, because `TensorPrims` is already the intended cross-device
primitive execution layer.

## Scope

In scope:

- internal GEMM-style operations in `tenferro-linalg`
- helpers whose only semantic job is batched matrix multiply
- removal of duplicated local GEMM code once equivalent `prims` paths exist

Out of scope:

- `qr`, `svd`, `lu`, `eig`, `solve`, and other factorization/solver kernels
- the broader tensor-level backend redesign tracked in issue #246
- changing AD formulas

## Recommended Migration Shape

The migration should happen in two phases.

### Phase 1: Establish the boundary

Because the repository is currently in API-skeleton mode, this phase should not
rewrite runtime behavior yet.

Instead, lock in the intended boundary:

- `TensorPrims` remains the owner of GEMM-style tensor multiplication
- `tenferro-linalg` stops treating backend-local GEMM as a permanent interface
- new implementation work should target an internal tensor-based GEMM bridge,
  not expand the existing slice-only helpers

### Phase 2: Replace local GEMM dispatch

Once runtime implementation work is allowed:

1. Introduce an internal helper that expresses the operation in terms of
   `Tensor<T>` plus `PrimDescriptor::BatchedGemm`.
2. Migrate `backend_mat_mul_nn` call sites that already operate on logical
   dense matrices.
3. Migrate `backend_mat_mul` call sites used in AD rules, preserving existing
   shapes and transpose conventions.
4. Remove the duplicated helper once all supported cases flow through
   `TensorPrims`.

## Relationship to `complex_mat_mul_nn`

`complex_mat_mul_nn` should not be treated as the first migration target.

It exists inside general `eig` AD helper paths and is entangled with the older
CPU-slice representation used there today. It should be revisited only after
the tensor-level backend work in issue #246 has clarified the broader complex
linalg boundary.

In other words:

- migrate the ordinary `backend_mat_mul*` helpers first
- defer `complex_mat_mul_nn` until the surrounding `eig` path is less CPU-local

## Acceptance Criteria for the First Implementation Pass

The first implementation pass for #245 should be considered complete when:

- the non-complex internal GEMM helpers used by `matrix_exp` and standard AD
  paths route through `tenferro-prims::BatchedGemm`
- no new code is added that deepens the slice-only GEMM path
- existing linalg and AD tests remain green

## Relationship to PyTorch

This is consistent with the PyTorch-inspired layering already chosen for the
repo.

- shared tensor primitives stay in the primitive layer
- factorization and solve kernels stay in the linalg layer
- higher-level linalg code is free to compose lower-level primitives where the
  math is just GEMM

That is the same broad separation of concerns that motivates keeping #245 and
#246 as separate issues.
