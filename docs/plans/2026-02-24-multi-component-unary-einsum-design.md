# Multi-Component Unary Einsum Design

**Date**: 2026-02-24
**Issue**: [#206](https://github.com/tensor4all/tenferro-rs/issues/206)

## Summary

Implement multi-component unary einsum semantics so repeated-label patterns
are handled correctly and parity gaps with OMEinsum.jl are closed.

## Scope

- `tenferro-prims`: multi-component Trace/AntiTrace/AntiDiag execution
- `tenferro-einsum`: pipeline decomposition for all unary patterns
  (including simultaneous repeated labels in input and output)
- `tenferro-tensor`: diagonal view via stride trick
- Tests: unignore existing parity tests + add new coverage

## A. tenferro-prims — Multi-Component Execution

### Problem

`execute_trace`, `execute_anti_trace`, `execute_anti_diag` use a single loop
variable `d` shared across all pairs. For `iijj->` this produces
`sum_d A[d,d,d,d]` instead of the correct `sum_{i,j} A[i,i,j,j]`.

### Solution

**Plan time** (`build_plan`): Apply union-find over `paired` edges to compute
connected components.

```
paired = [(0,1), (2,3)]   # for iijj->
  → Component 0: axes {0,1}, dim = D_i
  → Component 1: axes {2,3}, dim = D_j
```

Store per-plan:
- `comp_dims: Vec<usize>` — dimension of each component
- `axis_to_comp: Vec<Option<usize>>` — axis → component index

**Execution**: Replace single `d` loop with Cartesian product over component
dimensions.

```
// Before (broken):
for d in 0..diag_dim:
    all paired axes = d

// After (correct):
for (d_0, d_1, ...) in cartesian_product(comp_dims):
    component t's axes = d_t
```

Same fix pattern for Trace, AntiTrace, and AntiDiag. AntiDiag additionally
handles **generative components** (no input axis constrains the component)
by looping independently over the component dimension.

## B. tenferro-einsum — Pipeline Decomposition

Following OMEinsum.jl's proven architecture, unary einsum is decomposed into
a pipeline of stages. Each stage maps to an existing prim or view operation.

```
Input tensor
  │
  ├─ Stage 1: Diag (diagonal view via stride trick)
  │   Repeated input label → unique: ii->i, aabb->ab
  │   Zero-copy view, not a prim call.
  │
  ├─ Stage 2: Trace / Reduce
  │   Labels in input but not output: contract or sum.
  │   Trace for paired diagonal axes, Reduce for free axes.
  │
  ├─ Stage 3: Permute
  │   Reorder axes to match output label order.
  │
  ├─ Stage 4: Duplicate (AntiDiag) / Repeat (AntiTrace)
  │   Output labels repeated → diagonal embedding (AntiDiag).
  │   Output labels absent from input → broadcast (AntiTrace).
  │
  └─ Output tensor
```

### Example: `iij->jj`

1. Diag: not needed (`i` is not in output)
2. Trace: `iij->j` (paired `(i0,i1)`, free `j`)
3. Permute: not needed
4. Duplicate: `j->jj` (AntiDiag)

### Example: `iijj->`

1. Diag: not needed
2. Trace: `iijj->` (two components, multi-component fix)
3. Permute: not needed
4. Duplicate: not needed

### Example: `->iii`

1. Diag: not needed
2. Trace/Reduce: not needed
3. Permute: not needed
4. Duplicate: `->iii` (AntiDiag with generative component, size from `size_dict`)

`execute_single_tensor_einsum` classifies labels, determines which stages are
needed, and executes only those stages sequentially with intermediate tensors.

## C. Tensor-Level Diagonal View

Add `Tensor<T>::diagonal(axis1, axis2) -> Tensor<T>` as a zero-copy view.

Implementation: combine strides of the two axes into one:
`new_stride = stride[axis1] + stride[axis2]`, reducing rank by one.
The resulting tensor shares the same data buffer.

Used by Stage 1 of the pipeline. For labels repeated more than twice
(e.g., `iii`), apply iteratively: `diagonal(0,1)` then `diagonal(0,1)` again.

## D. Correspondence with OMEinsum.jl

| OMEinsum.jl | tenferro-rs | Notes |
|-------------|-------------|-------|
| `Tr` | `Trace` prim | Multi-component fix needed |
| `Diag` | `Tensor::diagonal()` view | Stride trick, zero-copy |
| `Sum` | `Reduce` prim | Already works |
| `Permutedims` | `Permute` prim | Already works |
| `Duplicate` | `AntiDiag` prim | Generative component fix needed |
| `Repeat` | `AntiTrace` prim | Already works |

## E. Tests

### Unignore

- `einsum_multi_pair_trace_iijj` — `iijj->` multi-component trace
- `einsum_size_dict_scalar_to_diagonal_and_superdiagonal` — `->ii`, `->iii`

### New prim-level tests

- Multi-component Trace: 2+ independent paired components
- Multi-component AntiDiag: generative component (no input anchor)
- Multi-component AntiTrace: broadcast with multiple components

### New einsum-level tests

- `iij->jj` — pipeline: Trace then AntiDiag
- `ii->ii` — pipeline: Diag then AntiDiag
- `ij->iijj` — pipeline: AntiDiag for both labels

## F. Existing Behavior

Single-pair cases (`ii->`, `ii->i`, `i->ii`) produce exactly one component,
so the Cartesian product degenerates to a single loop variable.
Existing tests verify no regression.
