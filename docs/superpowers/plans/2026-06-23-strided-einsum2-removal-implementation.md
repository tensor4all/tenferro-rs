# Strided-Einsum2 Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `strided-einsum2` from `tenferro-cpu` while preserving the current Faer `dot_general` GEMM preparation algorithm.

**Architecture:** Add a `tenferro-cpu/src/gemm/faer_prepared.rs` internal module that owns the dot-general-specific strided GEMM preparation previously provided by `strided-einsum2`. The module computes canonical dot-general axis groups, copies only non-fusable operands into pooled col-major temporaries, and dispatches batched Faer GEMM through the existing `FaerGemm` trait.

**Tech Stack:** Rust 2021, `tenferro-cpu`, Faer, `strided-kernel` copy/view helpers, Cargo feature tests.

---

### File Structure

- Create: `crates/tenferro-cpu/src/gemm/faer_prepared.rs`
  - Owns Faer-only dot-general preparation, operand-local copy decisions, batch offset iteration, and test-only dispatch/copy counters.
- Modify: `crates/tenferro-cpu/src/gemm/mod.rs`
  - Wires the new prepared Faer path into `dot_general_faer_cached`, `dot_general_faer_with_conj_cached`, and `dot_general_faer_read_cached`; removes `strided_dot`.
- Delete: `crates/tenferro-cpu/src/gemm/strided_dot.rs`
  - Removes the `strided-einsum2` adapter.
- Modify: `crates/tenferro-cpu/src/gemm/tests.rs`
  - Replaces `strided_dot` dispatch assertions with prepared Faer dispatch/copy assertions.
- Modify: `Cargo.toml`
  - Removes workspace dependency `strided-einsum2`.
- Modify: `crates/tenferro-cpu/Cargo.toml`
  - Removes optional dependency and feature forwarding entries for `strided-einsum2`.
- Modify: `crates/tenferro-cpu/tests/provider_feature_contract.rs`
  - Removes provider feature expectations for `strided-einsum2`.
- Modify: `docs/worklogs/2026-06-22-v0.1-publish-readiness.md`
  - Removes `strided-einsum2` from publish prerequisites.

### Task 1: Lock the Faer Prepared Path Contract With Failing Tests

**Files:**
- Modify: `crates/tenferro-cpu/src/gemm/tests.rs`

- [ ] **Step 1: Replace the old strided-dot dispatch test import**

Change the Faer test imports to reference the planned prepared Faer test helpers:

```rust
#[cfg(feature = "cpu-faer")]
use super::{dot_general_faer_cached, dot_general_faer_read_cached, faer_prepared};
```

- [ ] **Step 2: Replace the transposed-view direct test**

Replace `faer_read_transposed_view_uses_strided_dot_without_materializing_input` with:

```rust
#[cfg(feature = "cpu-faer")]
#[test]
fn faer_read_transposed_view_uses_prepared_gemm_without_operand_copies() {
    let lhs_source =
        TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let lhs_view = lhs_source.as_view().transpose_view([1, 0]).unwrap();
    let rhs = TypedTensor::<f64>::from_vec_col_major(
        vec![3, 2],
        vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0],
    )
    .unwrap();
    let rhs = Tensor::F64(rhs);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let mut buffers = BufferPool::new();
    let mut cache = GemmAnalysisCache::default();
    let ctx = CpuContext::with_threads(1).unwrap();

    faer_prepared::test_reset_stats();
    let out = dot_general_faer_read_cached(
        &mut buffers,
        &mut cache,
        Some(0),
        &ctx,
        TensorRead::from_view(TensorView::F64(lhs_view)),
        TensorRead::from_tensor(&rhs),
        &config,
    )
    .unwrap()
    .expect("same-dtype f64 inputs should be handled directly");

    let stats = faer_prepared::test_stats();
    assert_eq!(stats.dispatches, 1);
    assert_eq!(stats.lhs_copies, 0);
    assert_eq!(stats.rhs_copies, 0);
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[50.0, 122.0, 68.0, 167.0]);
}
```

- [ ] **Step 3: Add a one-sided copy test**

Append this test near the Faer GEMM tests:

```rust
#[cfg(feature = "cpu-faer")]
#[test]
fn faer_noncanonical_contract_copies_only_nonfusable_rhs_operand() {
    let lhs =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let rhs = TypedTensor::<f64>::from_vec_col_major(
        vec![3, 2],
        vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0],
    )
    .unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![0, 1],
        rhs_contracting_dims: vec![1, 0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let mut buffers = BufferPool::new();
    let mut cache = GemmAnalysisCache::default();
    let ctx = CpuContext::with_threads(1).unwrap();

    faer_prepared::test_reset_stats();
    let out = dot_general_faer_cached(&mut buffers, &mut cache, Some(1), &ctx, &lhs, &rhs, &config)
        .unwrap();

    let stats = faer_prepared::test_stats();
    assert_eq!(stats.dispatches, 1);
    assert_eq!(stats.lhs_copies, 0);
    assert_eq!(stats.rhs_copies, 1);
    assert_eq!(out.shape(), &[] as &[usize]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[212.0]);
}
```

- [ ] **Step 4: Run tests and verify RED**

Run:

```bash
cargo test -p tenferro-cpu --features cpu-faer faer_read_transposed_view_uses_prepared_gemm_without_operand_copies -- --exact
```

Expected: FAIL to compile because `faer_prepared` is not yet defined or exposed to the test module.

Run:

```bash
cargo test -p tenferro-cpu --features cpu-faer faer_noncanonical_contract_copies_only_nonfusable_rhs_operand -- --exact
```

Expected: FAIL to compile for the same missing `faer_prepared` module.

### Task 2: Implement the Prepared Faer Dot-General Module

**Files:**
- Create: `crates/tenferro-cpu/src/gemm/faer_prepared.rs`
- Modify: `crates/tenferro-cpu/src/gemm/mod.rs`

- [ ] **Step 1: Add module wiring**

In `crates/tenferro-cpu/src/gemm/mod.rs`, replace:

```rust
#[cfg(feature = "cpu-faer")]
mod strided_dot;
```

with:

```rust
#[cfg(feature = "cpu-faer")]
mod faer_prepared;
```

- [ ] **Step 2: Add the prepared module skeleton**

Create `crates/tenferro-cpu/src/gemm/faer_prepared.rs` with these public-to-parent entry points:

```rust
use num_traits::{One, Zero};
use smallvec::SmallVec;
use std::sync::atomic::{AtomicUsize, Ordering};
use strided_kernel::{copy_into_col_major, StridedView, StridedViewMut};
use tenferro_tensor::{col_major_strides, Buffer, DotGeneralConfig, TypedTensor};

use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::{default_placement, Error};

use super::{checked_product, validate_dot_general, FaerGemm, TypedTensorRead};

#[cfg(test)]
static DISPATCH_COUNT: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static LHS_COPY_COUNT: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static RHS_COPY_COUNT: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PreparedFaerStats {
    pub dispatches: usize,
    pub lhs_copies: usize,
    pub rhs_copies: usize,
}

#[cfg(test)]
pub(super) fn test_reset_stats() {
    DISPATCH_COUNT.store(0, Ordering::SeqCst);
    LHS_COPY_COUNT.store(0, Ordering::SeqCst);
    RHS_COPY_COUNT.store(0, Ordering::SeqCst);
}

#[cfg(test)]
pub(super) fn test_stats() -> PreparedFaerStats {
    PreparedFaerStats {
        dispatches: DISPATCH_COUNT.load(Ordering::SeqCst),
        lhs_copies: LHS_COPY_COUNT.load(Ordering::SeqCst),
        rhs_copies: RHS_COPY_COUNT.load(Ordering::SeqCst),
    }
}
```

- [ ] **Step 3: Implement axis grouping and fusable checks**

Add helpers equivalent to the dot-general subset of `strided-einsum2`:

```rust
fn free_axes(rank: usize, contracting: &[usize], batch: &[usize]) -> SmallVec<[usize; 8]> {
    (0..rank)
        .filter(|axis| !contracting.contains(axis) && !batch.contains(axis))
        .collect()
}

fn try_fuse_col_major_group(dims: &[usize], strides: &[isize]) -> Option<(usize, isize)> {
    if dims.len() != strides.len() {
        return None;
    }
    let total = checked_product(dims)?;
    if dims.is_empty() {
        return Some((1, 0));
    }

    let mut base_stride = None;
    let mut expected_stride = None;
    for (&dim, &stride) in dims.iter().zip(strides.iter()) {
        if dim <= 1 {
            continue;
        }
        if stride == 0 {
            return None;
        }
        if let Some(expected) = expected_stride {
            if stride != expected {
                return None;
            }
        } else {
            base_stride = Some(stride);
        }
        let dim = isize::try_from(dim).ok()?;
        expected_stride = Some(stride.checked_mul(dim)?);
    }

    let stride = base_stride.unwrap_or_else(|| {
        strides
            .iter()
            .copied()
            .min_by_key(|stride| stride.unsigned_abs())
            .unwrap_or(0)
    });
    Some((total, stride))
}
```

- [ ] **Step 4: Implement operand preparation**

Add a `PreparedOperand<T>` that either references the original host data or owns a pooled col-major temporary:

```rust
struct PreparedOperand<T> {
    ptr: *const T,
    row_stride: isize,
    col_stride: isize,
    batch_strides: SmallVec<[isize; 8]>,
    conj: bool,
    buffer: Option<Vec<T>>,
}
```

Implement `prepare_operand` with this behavior:

```rust
fn prepare_operand<R, T>(
    buffers: &mut BufferPool,
    read: &R,
    dims: SmallVec<[usize; 8]>,
    strides: SmallVec<[isize; 8]>,
    n_group1: usize,
    n_group2: usize,
    conj: bool,
    copy_counter: &AtomicUsize,
) -> crate::Result<PreparedOperand<T>>
where
    R: TypedTensorRead<T>,
    T: PoolScalar + Copy + Clone + Send + Sync + 'static,
{
    let Some(data) = read.host_data_opt()? else {
        return Err(Error::backend_failure(
            "dot_general",
            "CPU dot_general requires host-backed inputs",
        ));
    };
    let n_inner = n_group1 + n_group2;
    let view = StridedView::new(data, &dims, &strides, read.offset())
        .map_err(|err| Error::backend_failure("dot_general", err))?;
    let group1 = try_fuse_col_major_group(&dims[..n_group1], &strides[..n_group1]);
    let group2 = try_fuse_col_major_group(&dims[n_group1..n_inner], &strides[n_group1..n_inner]);

    if let (Some((_, row_stride)), Some((_, col_stride))) = (group1, group2) {
        return Ok(PreparedOperand {
            ptr: view.ptr(),
            row_stride,
            col_stride,
            batch_strides: strides[n_inner..].iter().copied().collect(),
            conj,
            buffer: None,
        });
    }

    copy_counter.fetch_add(1, Ordering::SeqCst);
    let len = checked_product(&dims)
        .ok_or_else(|| Error::backend_failure("dot_general", "operand element count overflow"))?;
    let mut buffer = unsafe { T::pool_acquire(buffers, len) };
    let out_strides = col_major_strides(&dims)?;
    let mut out = StridedViewMut::new(&mut buffer, &dims, &out_strides, 0)
        .map_err(|err| Error::backend_failure("dot_general", err))?;
    copy_into_col_major(&mut out, &view)
        .map_err(|err| Error::backend_failure("dot_general", err))?;
    let ptr = buffer.as_ptr();
    let row_stride = if checked_product(&dims[..n_group1]).unwrap_or(0) == 0 {
        0
    } else {
        1
    };
    let col_stride = checked_product(&dims[..n_group1])
        .ok_or_else(|| Error::backend_failure("dot_general", "operand row count overflow"))?
        as isize;
    Ok(PreparedOperand {
        ptr,
        row_stride,
        col_stride,
        batch_strides: out_strides[n_inner..].iter().copied().collect(),
        conj,
        buffer: Some(buffer),
    })
}
```

- [ ] **Step 5: Implement batch iteration and GEMM dispatch**

Add a helper that attempts fused batch stepping first and otherwise iterates a `SmallVec<[usize; 8]>` multi-index:

```rust
fn try_fuse_batch_group(dims: &[usize], strides: &[isize]) -> Option<(usize, isize)> {
    match dims.len() {
        0 => Some((1, 0)),
        1 => Some((dims[0], strides[0])),
        _ => {
            if dims.len() != strides.len() {
                return None;
            }
            for (&dim, &stride) in dims.iter().zip(strides.iter()) {
                if dim > 1 && stride == 0 {
                    return None;
                }
            }
            let mut base_idx = None;
            let mut base_abs = usize::MAX;
            for (idx, (&dim, &stride)) in dims.iter().zip(strides.iter()).enumerate() {
                if dim <= 1 {
                    continue;
                }
                let abs = stride.unsigned_abs();
                if abs < base_abs {
                    base_abs = abs;
                    base_idx = Some(idx);
                }
            }
            let Some(base) = base_idx else {
                let stride = strides
                    .iter()
                    .copied()
                    .min_by_key(|stride| stride.unsigned_abs())
                    .unwrap_or(0);
                return Some((checked_product(dims)?, stride));
            };
            let mut used = smallvec::smallvec![false; dims.len()];
            used[base] = true;
            let mut expected_abs = base_abs.checked_mul(dims[base])?;
            let non_singleton = dims.iter().filter(|&&dim| dim > 1).count();
            for _ in 1..non_singleton {
                let mut next = None;
                for idx in 0..dims.len() {
                    if used[idx] || dims[idx] <= 1 {
                        continue;
                    }
                    if strides[idx].unsigned_abs() == expected_abs {
                        next = Some(idx);
                        break;
                    }
                }
                let idx = next?;
                used[idx] = true;
                expected_abs = expected_abs.checked_mul(dims[idx])?;
            }
            Some((checked_product(dims)?, strides[base]))
        }
    }
}

fn offset_for_index(index: &[usize], strides: &[isize]) -> Option<isize> {
    index
        .iter()
        .zip(strides.iter())
        .try_fold(0isize, |acc, (&idx, &stride)| {
            let idx = isize::try_from(idx).ok()?;
            acc.checked_add(idx.checked_mul(stride)?)
        })
}

fn advance_index(index: &mut [usize], dims: &[usize]) {
    for axis in (0..dims.len()).rev() {
        index[axis] += 1;
        if index[axis] < dims[axis] {
            break;
        }
        index[axis] = 0;
    }
}

fn for_each_batch(
    batch_dims: &[usize],
    a_batch_strides: &[isize],
    b_batch_strides: &[isize],
    c_batch_strides: &[isize],
    mut f: impl FnMut(isize, isize, isize) -> crate::Result<()>,
) -> crate::Result<()> {
    if let (Some((total, a_step)), Some((_, b_step)), Some((_, c_step))) = (
        try_fuse_batch_group(batch_dims, a_batch_strides),
        try_fuse_batch_group(batch_dims, b_batch_strides),
        try_fuse_batch_group(batch_dims, c_batch_strides),
    ) {
        let mut a_off = 0isize;
        let mut b_off = 0isize;
        let mut c_off = 0isize;
        for _ in 0..total {
            f(a_off, b_off, c_off)?;
            a_off = a_off.checked_add(a_step).ok_or_else(|| {
                Error::backend_failure("dot_general", "lhs batch offset overflow")
            })?;
            b_off = b_off.checked_add(b_step).ok_or_else(|| {
                Error::backend_failure("dot_general", "rhs batch offset overflow")
            })?;
            c_off = c_off.checked_add(c_step).ok_or_else(|| {
                Error::backend_failure("dot_general", "output batch offset overflow")
            })?;
        }
        return Ok(());
    }

    let total = checked_product(batch_dims)
        .ok_or_else(|| Error::backend_failure("dot_general", "batch element count overflow"))?;
    let mut index = smallvec::smallvec![0usize; batch_dims.len()];
    for _ in 0..total {
        let a_off = offset_for_index(&index, a_batch_strides)
            .ok_or_else(|| Error::backend_failure("dot_general", "lhs batch offset overflow"))?;
        let b_off = offset_for_index(&index, b_batch_strides)
            .ok_or_else(|| Error::backend_failure("dot_general", "rhs batch offset overflow"))?;
        let c_off = offset_for_index(&index, c_batch_strides)
            .ok_or_else(|| Error::backend_failure("dot_general", "output batch offset overflow"))?;
        f(a_off, b_off, c_off)?;
        advance_index(&mut index, batch_dims);
    }
    Ok(())
}
```

- [ ] **Step 6: Implement `dot_general_prepared_faer`**

Add `DotGeneralPreparedPlan` before the public-to-parent entry point:

```rust
struct DotGeneralPreparedPlan {
    lhs_dims: SmallVec<[usize; 8]>,
    lhs_strides: SmallVec<[isize; 8]>,
    rhs_dims: SmallVec<[usize; 8]>,
    rhs_strides: SmallVec<[isize; 8]>,
    batch_dims: SmallVec<[usize; 8]>,
    out_shape: SmallVec<[usize; 8]>,
    out_len: usize,
    output_row_stride: isize,
    output_col_stride: isize,
    output_batch_strides: SmallVec<[isize; 8]>,
    lhs_n_group1: usize,
    lhs_n_group2: usize,
    rhs_n_group1: usize,
    rhs_n_group2: usize,
    m: usize,
    n: usize,
    k: usize,
    batch_total: usize,
}
```

Implement `DotGeneralPreparedPlan::new` by computing:

```rust
let lhs_free = free_axes(lhs.shape().len(), &config.lhs_contracting_dims, &config.lhs_batch_dims);
let rhs_free = free_axes(rhs.shape().len(), &config.rhs_contracting_dims, &config.rhs_batch_dims);
let lhs_dims = lhs_free
    .iter()
    .chain(config.lhs_contracting_dims.iter())
    .chain(config.lhs_batch_dims.iter())
    .map(|&axis| lhs.shape()[axis])
    .collect::<SmallVec<[usize; 8]>>();
let rhs_dims = config
    .rhs_contracting_dims
    .iter()
    .chain(rhs_free.iter())
    .chain(config.rhs_batch_dims.iter())
    .map(|&axis| rhs.shape()[axis])
    .collect::<SmallVec<[usize; 8]>>();
let lhs_source_strides = lhs.strides()?;
let rhs_source_strides = rhs.strides()?;
let lhs_strides = lhs_free
    .iter()
    .chain(config.lhs_contracting_dims.iter())
    .chain(config.lhs_batch_dims.iter())
    .map(|&axis| lhs_source_strides[axis])
    .collect::<SmallVec<[isize; 8]>>();
let rhs_strides = config
    .rhs_contracting_dims
    .iter()
    .chain(rhs_free.iter())
    .chain(config.rhs_batch_dims.iter())
    .map(|&axis| rhs_source_strides[axis])
    .collect::<SmallVec<[isize; 8]>>();
```

Then compute `m`, `n`, `k`, `batch_total`, `out_shape`, and output strides
with `checked_product` and `col_major_strides`; return `Error::backend_failure`
on overflow.

Expose this entry point to `gemm/mod.rs`:

```rust
#[allow(clippy::too_many_arguments)]
pub(super) fn dot_general_prepared_faer<L, R, T>(
    buffers: &mut BufferPool,
    ctx: &crate::CpuContext,
    lhs: &L,
    rhs: &R,
    config: &DotGeneralConfig,
    lhs_conj: bool,
    rhs_conj: bool,
) -> crate::Result<TypedTensor<T>>
where
    L: TypedTensorRead<T>,
    R: TypedTensorRead<T>,
    T: FaerGemm + PoolScalar + Copy + Clone + Zero + One + PartialEq + 'static,
{
    validate_dot_general(lhs, rhs, config)?;
    let plan = DotGeneralPreparedPlan::new(lhs, rhs, config)?;
    if plan.m == 0 || plan.n == 0 || plan.k == 0 || plan.batch_total == 0 {
        let data = T::pool_acquire_zeroed(buffers, plan.out_len);
        return TypedTensor::from_buffer_col_major(
            plan.out_shape.into_vec(),
            Buffer::Host(data),
            default_placement(),
        );
    }

    let lhs_prepared = prepare_operand(
        buffers,
        lhs,
        plan.lhs_dims.clone(),
        plan.lhs_strides.clone(),
        plan.lhs_n_group1,
        plan.lhs_n_group2,
        lhs_conj,
        &LHS_COPY_COUNT,
    )?;
    let rhs_prepared = prepare_operand(
        buffers,
        rhs,
        plan.rhs_dims.clone(),
        plan.rhs_strides.clone(),
        plan.rhs_n_group1,
        plan.rhs_n_group2,
        rhs_conj,
        &RHS_COPY_COUNT,
    )?;
    let mut out_data = unsafe { T::pool_acquire(buffers, plan.out_len) };
    let c_ptr = out_data.as_mut_ptr();

    for_each_batch(
        &plan.batch_dims,
        &lhs_prepared.batch_strides,
        &rhs_prepared.batch_strides,
        &plan.output_batch_strides,
        |a_batch_off, b_batch_off, c_batch_off| {
            unsafe {
                T::strided_gemm_with_conj(
                    ctx,
                    T::one(),
                    lhs_prepared.ptr.offset(a_batch_off),
                    plan.m,
                    plan.k,
                    lhs_prepared.row_stride,
                    lhs_prepared.col_stride,
                    lhs_prepared.conj,
                    rhs_prepared.ptr.offset(b_batch_off),
                    plan.n,
                    rhs_prepared.row_stride,
                    rhs_prepared.col_stride,
                    rhs_prepared.conj,
                    T::zero(),
                    c_ptr.offset(c_batch_off),
                    plan.output_row_stride,
                    plan.output_col_stride,
                );
            }
            Ok(())
        },
    )?;

    DISPATCH_COUNT.fetch_add(1, Ordering::SeqCst);
    TypedTensor::from_buffer_col_major(
        plan.out_shape.into_vec(),
        Buffer::Host(out_data),
        default_placement(),
    )
}
```

- [ ] **Step 7: Run focused tests and verify GREEN for the new module**

Run:

```bash
cargo test -p tenferro-cpu --features cpu-faer faer_read_transposed_view_uses_prepared_gemm_without_operand_copies -- --exact
cargo test -p tenferro-cpu --features cpu-faer faer_noncanonical_contract_copies_only_nonfusable_rhs_operand -- --exact
```

Expected: both PASS.

### Task 3: Wire Prepared Faer Into Existing Dot-General Entry Points

**Files:**
- Modify: `crates/tenferro-cpu/src/gemm/mod.rs`

- [ ] **Step 1: Remove `strided_einsum2` bounds from Faer functions**

In `dot_general_faer_cached` and `dot_general_faer_with_conj_cached`, replace bounds containing:

```rust
+ strided_einsum2::ScalarBase
strided_einsum2::backend::FaerBackend: strided_einsum2::Backend<T>,
```

with the existing tenferro-local bounds:

```rust
T: FaerGemm + PoolScalar + Copy + Clone + Zero + One + PartialEq + 'static,
```

- [ ] **Step 2: Replace non-conjugated early return**

In `dot_general_faer_with_conj_cached`, replace the `strided_dot::dot_general_strided_with_backend` early return with:

```rust
faer_prepared::dot_general_prepared_faer(
    buffers, ctx, lhs, rhs, config, lhs_conj, rhs_conj,
)
```

Remove the old direct-only decision as the deciding path for non-conjugated
Faer. The prepared path is the optimized path for both owned and read inputs.

- [ ] **Step 3: Replace read-direct dispatch macro bodies**

In `dot_general_faer_read_cached`, replace each `strided_dot::dot_general_strided_with_backend` call with:

```rust
faer_prepared::dot_general_prepared_faer(
    buffers, ctx, a, b, config, false, false,
)
.map(|result| Some(crate::Tensor::$wrap(result)))
```

- [ ] **Step 4: Run focused CPU tests**

Run:

```bash
cargo test -p tenferro-cpu --features cpu-faer dot_general -- --nocapture
```

Expected: all matching tests PASS.

### Task 4: Remove the Dependency Surface

**Files:**
- Delete: `crates/tenferro-cpu/src/gemm/strided_dot.rs`
- Modify: `Cargo.toml`
- Modify: `crates/tenferro-cpu/Cargo.toml`
- Modify: `crates/tenferro-cpu/tests/provider_feature_contract.rs`
- Modify: `docs/worklogs/2026-06-22-v0.1-publish-readiness.md`

- [ ] **Step 1: Delete the old adapter**

Remove `crates/tenferro-cpu/src/gemm/strided_dot.rs`.

- [ ] **Step 2: Remove workspace dependency**

Delete this line from root `Cargo.toml`:

```toml
strided-einsum2 = { git = "https://github.com/tensor4all/strided-rs", rev = "71bdd913158a87437e51f4f9b69cba4cac6f5082", version = "0.1.0", default-features = false }
```

- [ ] **Step 3: Remove tenferro-cpu feature forwarding**

Update `crates/tenferro-cpu/Cargo.toml` so:

```toml
cpu-faer = ["dep:faer"]
```

and remove all `dep:strided-einsum2` and `strided-einsum2/...` entries from BLAS provider features. Delete the dependency entry:

```toml
strided-einsum2 = { workspace = true, optional = true }
```

- [ ] **Step 4: Update provider feature contract expectations**

Remove these required values from `crates/tenferro-cpu/tests/provider_feature_contract.rs`:

```rust
"dep:strided-einsum2",
"strided-einsum2/blas-openblas",
"strided-einsum2/blas-accelerate",
"strided-einsum2/blas-mkl",
```

- [ ] **Step 5: Update publish readiness documentation**

Delete the `strided-einsum2` prerequisite bullet from `docs/worklogs/2026-06-22-v0.1-publish-readiness.md`.

- [ ] **Step 6: Prove no dependency references remain**

Run:

```bash
rg "strided-einsum2|strided_einsum2|strided_dot" Cargo.toml crates/tenferro-cpu docs/worklogs/2026-06-22-v0.1-publish-readiness.md
```

Expected: no output.

### Task 5: Final Verification and Commits

**Files:**
- All files touched above.

- [ ] **Step 1: Format**

Run:

```bash
cargo fmt --all --check
```

Expected: PASS. If it fails, run `cargo fmt --all`, then rerun the check.

- [ ] **Step 2: Run focused tests**

Run:

```bash
cargo test -p tenferro-cpu --features cpu-faer
```

Expected: PASS.

- [ ] **Step 3: Run publish package check for tenferro-cpu**

Run:

```bash
cargo package -p tenferro-cpu --allow-dirty
```

Expected: it must not fail because of `strided-einsum2`. If it fails on unpublished tenferro or `t4a-*` dependencies, confirm the error no longer names `strided-einsum2`.

- [ ] **Step 4: Commit implementation**

Commit after tests pass:

```bash
git add Cargo.toml crates/tenferro-cpu docs/worklogs/2026-06-22-v0.1-publish-readiness.md
git commit -m "Remove strided-einsum2 from tenferro-cpu"
```
