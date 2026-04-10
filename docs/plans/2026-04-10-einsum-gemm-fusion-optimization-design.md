# Einsum GEMM Fusion Optimization

> **IMPORTANT**: Do NOT auto-implement this design. An agent must discuss the
> plan with a human reviewer and get explicit approval before writing any code.
> The pseudo-code below is illustrative — the actual implementation must be
> verified against the current codebase state.

## Problem

In N-ary einsum chains, the CPU backend sometimes triggers unnecessary physical
transposes (memory copies) before GEMM execution. Two independent issues cause this:

### Issue A: `try_fuse_dims` does not sort by stride

`try_fuse_dims` in `tenferro-tensor/src/cpu/gemm/mod.rs` checks whether a group
of dimensions can be fused into a single (total_size, base_stride) pair for
strided GEMM. It checks contiguity **in the given dim order**, but the dim order
in `DotGeneralConfig` may not match the physical stride order.

strided-rs's equivalent `try_fuse_group` sorts dimensions by stride before
checking, which is strictly more permissive.

**Failing pattern — cyclic trace `ij,jk,ki->`:**

```
Step 1: A[i,j] * B[j,k] -> [i,k] col-major, strides [1, I]
Step 2: [i,k] * C[k,i] -> scalar, contract i,k

  rhs C[k,i] col-major strides [1, K]
  rhs_contract = [1, 0]  (i at dim 1, k at dim 0 — label order from LHS)

  try_fuse_dims(shapes=[I, K], strides=[K, 1])
    base = K, K*I != 1  -> FAIL

  With stride sort: sorted strides [1, K], shapes [K, I]
    base = 1, 1*K = K   -> SUCCESS, fuse to (KI, 1)
```

Other failing patterns include batched cyclic traces (`bij,bjk,bki->b`) and
4-matrix cyclic products (`ij,jk,kl,li->`).

### Issue B: BLAS backend does not use transpose flags

The BLAS backend requires `a_rs == 1 && b_rs == 1 && c_rs == 1` (unit stride
for the M-group of each operand). When this fails, it falls back to
`canonical_gemm_layout` which physically transposes both operands.

Standard BLAS `dgemm`/`sgemm` supports `transA`/`transB` flags that handle
non-unit leading strides. strided-rs uses these flags to avoid copies.

**Failing pattern — `ij,kj->ik` (contract on trailing dim of RHS):**

```
rhs B[k,j] col-major strides [1, K]
rhs_contract = [1] (j at dim 1) -> b_rs = K != 1 -> BLAS FAIL

With trans flag: treat B as j x k transposed -> transB = 'T', ldb = K
```

**Failing pattern in einsum chain — `ij,jk,ik->`:**

```
Step 2: lhs = [i,k], contract i -> lhs_free = [1](k) -> a_rs = I != 1
With trans flag: transA = 'T', lda = I
```

## Proposed Changes

### Fix 1: Stride-sorted `try_fuse_dims`

In `tenferro-tensor/src/cpu/gemm/mod.rs`, modify `try_fuse_dims` to sort
dimension (shape, stride) pairs by ascending absolute stride before checking
contiguity. This matches strided-rs's `try_fuse_group` behavior.

```rust
fn try_fuse_dims(shapes: &[usize], strides: &[isize]) -> Option<(usize, isize)> {
    if shapes.is_empty() {
        return Some((1, 0));
    }
    if shapes.len() == 1 {
        return Some((shapes[0], strides[0]));
    }
    // Sort by ascending absolute stride
    let mut pairs: Vec<(usize, isize)> = shapes.iter().copied()
        .zip(strides.iter().copied()).collect();
    pairs.sort_by_key(|&(_, s)| s.unsigned_abs());

    let base_stride = pairs[0].1;
    let mut expected = base_stride;
    for &(dim, stride) in &pairs {
        if stride != expected {
            return None;
        }
        expected = stride.checked_mul(dim as isize)?;
    }
    Some((shapes.iter().product(), base_stride))
}
```

**Files:** `tenferro-tensor/src/cpu/gemm/mod.rs` (lines 172-188)

### Fix 2: BLAS transpose flags

In the BLAS backend path (`typed_blas_gemm`), instead of requiring `a_rs == 1`,
determine whether each operand should use `transA`/`transB` based on which
stride is unit:

- `a_rs == 1`: no transpose needed (column-major M-leading)
- `a_cs == 1` (contract stride is 1): use `transA = 'T'`
- neither is 1: fall back to `canonical_gemm_layout`

This requires `a_cs` to be computed in the BLAS path (currently `#[cfg(feature = "cpu-faer")]` only).

**Files:**
- `tenferro-tensor/src/cpu/gemm/mod.rs`: remove `cfg(feature = "cpu-faer")` gates on `a_cs`, `b_cs`, `c_cs` in `analyse_gemm` and `GemmDims`
- `tenferro-tensor/src/cpu/gemm/blas_gemm.rs`: accept strides and use `CblasTrans`/`CblasNoTrans` accordingly

## Verification

### Correctness
- All existing `cargo test --workspace --release` must pass.
- Oracle replay tests cover the numeric correctness of these paths.

### New tests
- Unit test for `try_fuse_dims` with reversed-stride inputs (the cyclic pattern).
- Unit test for BLAS `dot_general` with `ij,kj->ik` pattern (non-unit `b_rs`).
- Integration test: `einsum("ij,jk,ki->", A, B, C)` verifies no regression.

### Performance
- No new allocations in the fast path (when fuse succeeds).
- BLAS trans-flag path avoids `canonical_gemm_layout` fallback (one fewer copy).
