# Tropical Unary Einsum AD Design

**Date**: 2026-02-24
**Issue**: [#211](https://github.com/tensor4all/tenferro-rs/issues/211)

## Summary

Unify `tenferro-tropical` AD to support both unary and binary tropical
einsum backward. Refactor the 2-operand-only forward+backward pipeline into
an N-ary design that dispatches by operand count.

## Background

`tropical_einsum_rrule` and `tracked_tropical_einsum` reject
`operands.len() != 2`. omeinsum-rs supports unary tropical backward via a
simple argmax scatter: each output stores the linear index of the winning
input element; the backward scatters `cotangent[out]` to
`grad[winner_pos]`.

## Design

### A. Forward — `tropical_forward_with_argmax`

Generalize from `(a: &Tensor<T>, b: &Tensor<T>)` to
`operands: &[&Tensor<T>]`.

**Index analysis** (subscript-driven, not hardcoded to 2 inputs):
- Output modes: from `subs.output`
- Contracted modes: labels appearing in any input but not in output
- Output shape: resolve each output mode from the first operand that
  contains it

**Iteration** (unchanged pattern):
1. For each output position, build `mode_values` map from output indices.
2. Loop over Cartesian product of contracted dimensions.
3. Compute product of all operands at resolved indices.
4. Track argmax (which contracted index produced the tropical winner).
5. Store `tracker.indices[out_flat] = best_k`.

**Unary specifics**: When there is one operand and no contracted indices
(e.g., `ij->ij`), the loop body is a single iteration with `k_flat = 0`,
degenerating to direct copy. The argmax is trivially 0.

### B. Backward — `tropical_backward`

Generalize from `(a, b)` to `operands: &[&Tensor<T>]`, returning
`Vec<Tensor<T::Inner>>` (one gradient per operand).

For each output position:
1. Look up `k_winner` from tracker.
2. Resolve contracted indices to winner values.
3. Dispatch by operand count:

**Unary (1 operand)**: `grad[winner_input_pos] += cotangent[out_idx]`.
No `mul_backward` — tropical addition selects the winner; there is no
second factor.

**Binary (2 operands)**: Same as current — build A/B indices at the
winner, apply `T::mul_backward_a` / `T::mul_backward_b`, accumulate
into `da` / `db`.

**N-ary (3+ operands)**: Return error (future work).

### C. `TropicalEinsumReverseRule`

Simplify stored fields:
- Remove `batch`, `free_a`, `free_b` (only needed for binary index
  classification, now done inline)
- Keep `contracted: Vec<u32>` (used by backward)
- `primals: Vec<Tensor<T>>` (already a Vec, works for any operand count)

### D. `tropical_einsum_rrule`

- Remove `operands.len() != 2` guard
- Accept 1 or 2 operands; reject 0 or 3+
- Subscript analysis: contracted = labels in any `subs.inputs[i]` but
  not in `subs.output`
- Call unified `tropical_forward_with_argmax` and `tropical_backward`

### E. `tracked_tropical_einsum`

Same changes as rrule:
- Accept 1 or 2 operands
- Simplified subscript analysis
- Unified forward+backward

### F. Correspondence with omeinsum-rs

| omeinsum-rs | tenferro-tropical | Notes |
|-------------|-------------------|-------|
| `tropical_unary_backward` | `tropical_backward` (unary branch) | Scatter cotangent to winner |
| `execute_unary_with_argmax` | `tropical_forward_with_argmax` (1 operand) | Argmax over summed indices |
| `cost_and_gradient_unary` | `tropical_einsum_rrule` (dispatch) | Operand-count dispatch |
| `A::is_better()` | `product.inner() == new_sum.inner()` | Winner detection |

### G. Tests

Unary patterns (both `tropical_einsum_rrule` and `tracked_tropical_einsum`):
- `ii->` — trace (max of diagonal)
- `ij->` — full contraction (global max)
- `ij->i` — row-wise max
- `ij->j` — column-wise max

Verify:
- Forward value matches manual computation
- Gradient is nonzero only at winning positions
- Gradient accumulates correctly when multiple outputs share a winner
- Existing binary tests remain green (no regression)
