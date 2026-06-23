use num_traits::{One, Zero};
use smallvec::SmallVec;
#[cfg(test)]
use std::cell::Cell;
use strided_kernel::{copy_into_col_major, StridedView, StridedViewMut};
use tenferro_tensor::{Buffer, DotGeneralConfig, TypedTensor};

use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::{default_placement, Error};

use super::{checked_product, validate_dot_general, FaerGemm, TypedTensorRead};

#[cfg(test)]
thread_local! {
    static DISPATCH_COUNT: Cell<usize> = const { Cell::new(0) };
    static LHS_COPY_COUNT: Cell<usize> = const { Cell::new(0) };
    static RHS_COPY_COUNT: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PreparedFaerStats {
    pub dispatches: usize,
    pub lhs_copies: usize,
    pub rhs_copies: usize,
}

#[cfg(test)]
pub(super) fn test_reset_stats() {
    DISPATCH_COUNT.with(|count| count.set(0));
    LHS_COPY_COUNT.with(|count| count.set(0));
    RHS_COPY_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
pub(super) fn test_stats() -> PreparedFaerStats {
    PreparedFaerStats {
        dispatches: DISPATCH_COUNT.with(Cell::get),
        lhs_copies: LHS_COPY_COUNT.with(Cell::get),
        rhs_copies: RHS_COPY_COUNT.with(Cell::get),
    }
}

#[derive(Clone, Copy)]
enum OperandRole {
    Lhs,
    Rhs,
}

#[cfg(test)]
fn record_dispatch() {
    DISPATCH_COUNT.with(|count| count.set(count.get() + 1));
}

#[cfg(not(test))]
fn record_dispatch() {}

#[cfg(test)]
fn record_copy(role: OperandRole) {
    match role {
        OperandRole::Lhs => LHS_COPY_COUNT.with(|count| count.set(count.get() + 1)),
        OperandRole::Rhs => RHS_COPY_COUNT.with(|count| count.set(count.get() + 1)),
    }
}

#[cfg(not(test))]
fn record_copy(_role: OperandRole) {}

struct PreparedOperand<T> {
    ptr: *const T,
    row_stride: isize,
    col_stride: isize,
    batch_strides: SmallVec<[isize; 8]>,
    conj: bool,
    _buffer: Option<Vec<T>>,
}

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

fn free_axes(rank: usize, contracting: &[usize], batch: &[usize]) -> SmallVec<[usize; 8]> {
    (0..rank)
        .filter(|axis| !contracting.contains(axis) && !batch.contains(axis))
        .collect()
}

fn checked_product_or_error(dims: &[usize], message: &'static str) -> crate::Result<usize> {
    checked_product(dims).ok_or_else(|| Error::backend_failure("dot_general", message))
}

fn col_major_strides_small(shape: &[usize]) -> crate::Result<SmallVec<[isize; 8]>> {
    let mut strides = SmallVec::with_capacity(shape.len());
    let mut stride = 1isize;
    for &extent in shape {
        strides.push(stride);
        let extent = isize::try_from(extent).map_err(|_| {
            Error::backend_failure("dot_general", "shape extent does not fit in isize")
        })?;
        stride = stride
            .checked_mul(extent)
            .ok_or_else(|| Error::backend_failure("dot_general", "column-major stride overflow"))?;
    }
    Ok(strides)
}

fn collect_shape_and_strides(
    shape: &[usize],
    strides: &[isize],
    axes: impl Iterator<Item = usize>,
) -> (SmallVec<[usize; 8]>, SmallVec<[isize; 8]>) {
    let mut dims = SmallVec::new();
    let mut permuted_strides = SmallVec::new();
    for axis in axes {
        dims.push(shape[axis]);
        permuted_strides.push(strides[axis]);
    }
    (dims, permuted_strides)
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

            let mut used: SmallVec<[bool; 8]> = smallvec::smallvec![false; dims.len()];
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

    let total = checked_product_or_error(batch_dims, "batch element count overflow")?;
    let mut index: SmallVec<[usize; 8]> = smallvec::smallvec![0usize; batch_dims.len()];
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

impl DotGeneralPreparedPlan {
    fn new<L, R, T>(lhs: &L, rhs: &R, config: &DotGeneralConfig) -> crate::Result<Self>
    where
        L: TypedTensorRead<T>,
        R: TypedTensorRead<T>,
    {
        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();
        let lhs_free = free_axes(
            lhs_shape.len(),
            &config.lhs_contracting_dims,
            &config.lhs_batch_dims,
        );
        let rhs_free = free_axes(
            rhs_shape.len(),
            &config.rhs_contracting_dims,
            &config.rhs_batch_dims,
        );

        let lhs_source_strides = lhs.strides()?;
        let rhs_source_strides = rhs.strides()?;

        let lhs_axes = lhs_free
            .iter()
            .copied()
            .chain(config.lhs_contracting_dims.iter().copied())
            .chain(config.lhs_batch_dims.iter().copied());
        let (lhs_dims, lhs_strides) =
            collect_shape_and_strides(lhs_shape, &lhs_source_strides, lhs_axes);

        let rhs_axes = config
            .rhs_contracting_dims
            .iter()
            .copied()
            .chain(rhs_free.iter().copied())
            .chain(config.rhs_batch_dims.iter().copied());
        let (rhs_dims, rhs_strides) =
            collect_shape_and_strides(rhs_shape, &rhs_source_strides, rhs_axes);

        let lhs_free_len = lhs_free.len();
        let lhs_contract_len = config.lhs_contracting_dims.len();
        let lhs_contract_end = lhs_free_len + lhs_contract_len;
        let rhs_contract_len = config.rhs_contracting_dims.len();
        let rhs_free_len = rhs_free.len();
        let rhs_free_end = rhs_contract_len + rhs_free_len;

        let batch_dims: SmallVec<[usize; 8]> =
            lhs_dims[lhs_contract_end..].iter().copied().collect();

        let m =
            checked_product_or_error(&lhs_dims[..lhs_free_len], "lhs free element count overflow")?;
        let n = checked_product_or_error(
            &rhs_dims[rhs_contract_len..rhs_free_end],
            "rhs free element count overflow",
        )?;
        let k = checked_product_or_error(
            &lhs_dims[lhs_free_len..lhs_contract_end],
            "contract element count overflow",
        )?;
        let batch_total = checked_product_or_error(&batch_dims, "batch element count overflow")?;

        let mut out_shape = SmallVec::<[usize; 8]>::new();
        out_shape.extend_from_slice(&lhs_dims[..lhs_free_len]);
        out_shape.extend_from_slice(&rhs_dims[rhs_contract_len..rhs_free_end]);
        out_shape.extend_from_slice(&batch_dims);
        let out_len = checked_product_or_error(&out_shape, "output element count overflow")?;
        let out_strides = col_major_strides_small(&out_shape)?;
        let output_row_stride =
            try_fuse_col_major_group(&out_shape[..lhs_free_len], &out_strides[..lhs_free_len])
                .ok_or_else(|| Error::backend_failure("dot_general", "output row fuse failed"))?
                .1;
        let output_col_stride = try_fuse_col_major_group(
            &out_shape[lhs_free_len..lhs_free_len + rhs_free_len],
            &out_strides[lhs_free_len..lhs_free_len + rhs_free_len],
        )
        .ok_or_else(|| Error::backend_failure("dot_general", "output column fuse failed"))?
        .1;
        let output_batch_strides = out_strides[lhs_free_len + rhs_free_len..]
            .iter()
            .copied()
            .collect();

        Ok(Self {
            lhs_dims,
            lhs_strides,
            rhs_dims,
            rhs_strides,
            batch_dims,
            out_shape,
            out_len,
            output_row_stride,
            output_col_stride,
            output_batch_strides,
            lhs_n_group1: lhs_free_len,
            lhs_n_group2: lhs_contract_len,
            rhs_n_group1: rhs_contract_len,
            rhs_n_group2: rhs_free_len,
            m,
            n,
            k,
            batch_total,
        })
    }
}

fn prepare_operand<R, T>(
    buffers: &mut BufferPool,
    read: &R,
    dims: SmallVec<[usize; 8]>,
    strides: SmallVec<[isize; 8]>,
    n_group1: usize,
    n_group2: usize,
    conj: bool,
    role: OperandRole,
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
            _buffer: None,
        });
    }

    record_copy(role);
    let len = checked_product_or_error(&dims, "operand element count overflow")?;
    // SAFETY: copy_into_col_major overwrites every element before the buffer is read.
    let mut buffer = unsafe { T::pool_acquire(buffers, len) };
    let out_strides = col_major_strides_small(&dims)?;
    let mut out = StridedViewMut::new(&mut buffer, &dims, &out_strides, 0)
        .map_err(|err| Error::backend_failure("dot_general", err))?;
    copy_into_col_major(&mut out, &view)
        .map_err(|err| Error::backend_failure("dot_general", err))?;
    let ptr = buffer.as_ptr();
    let row_extent = checked_product_or_error(&dims[..n_group1], "operand row count overflow")?;
    let row_stride = if row_extent == 0 { 0 } else { 1 };
    let col_stride = isize::try_from(row_extent)
        .map_err(|_| Error::backend_failure("dot_general", "operand row count overflow"))?;

    Ok(PreparedOperand {
        ptr,
        row_stride,
        col_stride,
        batch_strides: out_strides[n_inner..].iter().copied().collect(),
        conj,
        _buffer: Some(buffer),
    })
}

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
    T: FaerGemm + PoolScalar + Copy + Clone + Zero + One + PartialEq + Send + Sync + 'static,
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

    let mut lhs_prepared = prepare_operand(
        buffers,
        lhs,
        plan.lhs_dims.clone(),
        plan.lhs_strides.clone(),
        plan.lhs_n_group1,
        plan.lhs_n_group2,
        lhs_conj,
        OperandRole::Lhs,
    )?;
    let mut rhs_prepared = prepare_operand(
        buffers,
        rhs,
        plan.rhs_dims.clone(),
        plan.rhs_strides.clone(),
        plan.rhs_n_group1,
        plan.rhs_n_group2,
        rhs_conj,
        OperandRole::Rhs,
    )?;
    // SAFETY: each per-batch GEMM uses beta=0 and overwrites the output slice.
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

    if let Some(buffer) = lhs_prepared._buffer.take() {
        T::pool_release(buffers, buffer);
    }
    if let Some(buffer) = rhs_prepared._buffer.take() {
        T::pool_release(buffers, buffer);
    }

    record_dispatch();
    TypedTensor::from_buffer_col_major(
        plan.out_shape.into_vec(),
        Buffer::Host(out_data),
        default_placement(),
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn col_major_strides_small_matches_tensor_strides_without_spilling() {
        let shape = [2, 3, 4, 5];

        let strides = super::col_major_strides_small(&shape).unwrap();

        assert_eq!(strides.as_slice(), &[1, 2, 6, 24]);
        assert_eq!(
            strides.as_slice(),
            tenferro_tensor::col_major_strides(&shape)
                .unwrap()
                .as_slice()
        );
        assert!(!strides.spilled());
    }
}
