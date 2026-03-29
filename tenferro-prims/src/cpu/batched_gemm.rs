use strided_view::{StridedView, StridedViewMut};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};

use crate::infra::typed_dispatch::{
    cast_scalar_value, cast_strided_view, cast_strided_view_mut, dispatch_standard_scalar_type,
};

use super::context::CpuContext;
use super::gemm_support::{advance_batch_offset, batch_iteration_count, batch_offset};
use super::layout_fusion::try_fuse_group_in_order;
use super::reduction::scale_output;

#[cfg(feature = "gemm-faer")]
use super::gemm_support::FaerGemm;

#[cfg(feature = "gemm-blas")]
use super::gemm_support::BlasGemm;

/// Fallback for the OpenBLAS backend, which requires contiguous column-major data.
/// Packs strided A, B, C into scratch buffers, calls contiguous gemm, unpacks C.
#[cfg(feature = "gemm-blas")]
pub(super) fn execute_batched_gemm_contiguous<T: Scalar + 'static>(
    ctx: &mut CpuContext,
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
    batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
    gemm_fn: fn(T, &[T], &[T], T, &mut [T], usize, usize, usize) -> Result<()>,
) -> Result<()> {
    let a = inputs[0];
    let b = inputs[1];
    let nb = batch_dims.len();
    let a_strides = a.strides();
    let b_strides = b.strides();
    let c_strides = output.strides();
    let a_row = a_strides[0];
    let a_col = a_strides[1];
    let a_batch = &a_strides[2..];
    let b_row = b_strides[0];
    let b_col = b_strides[1];
    let b_batch = &b_strides[2..];
    let c_row = c_strides[0];
    let c_col = c_strides[1];
    let c_batch = &c_strides[2..];

    let a_ptr = a.ptr();
    let b_ptr = b.ptr();
    let c_ptr = output.as_mut_ptr();

    debug_assert!(
        batch_dims
            .iter()
            .enumerate()
            .all(|(i, &d)| d <= 1 || (a_batch[i] != 0 && b_batch[i] != 0 && c_batch[i] != 0)),
        "batch GEMM stride must be non-zero for batch dims > 1"
    );

    if nb == 0
        && a_row == 1
        && a_col == m as isize
        && b_row == 1
        && b_col == k as isize
        && c_row == 1
        && c_col == m as isize
    {
        let a_mat = unsafe { std::slice::from_raw_parts(a_ptr, m * k) };
        let b_mat = unsafe { std::slice::from_raw_parts(b_ptr, k * n) };
        let c_mat = unsafe { std::slice::from_raw_parts_mut(c_ptr, m * n) };
        return gemm_fn(alpha, a_mat, b_mat, beta, c_mat, m, n, k);
    }

    let mut a_mat = ctx.take_scratch::<T>(m * k)?;
    let mut b_mat = ctx.take_scratch::<T>(k * n)?;
    let mut c_mat = ctx.take_scratch::<T>(m * n)?;

    let do_batch = |a_off: isize,
                    b_off: isize,
                    c_off: isize,
                    a_mat: &mut [T],
                    b_mat: &mut [T],
                    c_mat: &mut [T]|
     -> Result<()> {
        for kk in 0..k {
            for i in 0..m {
                let src_off = a_off + i as isize * a_row + kk as isize * a_col;
                a_mat[i + kk * m] = unsafe { *a_ptr.offset(src_off) };
            }
        }

        for j in 0..n {
            for kk in 0..k {
                let src_off = b_off + kk as isize * b_row + j as isize * b_col;
                b_mat[kk + j * k] = unsafe { *b_ptr.offset(src_off) };
            }
        }

        if beta == T::zero() {
            c_mat.fill(T::zero());
        } else {
            for j in 0..n {
                for i in 0..m {
                    let src_off = c_off + i as isize * c_row + j as isize * c_col;
                    c_mat[i + j * m] = unsafe { *c_ptr.offset(src_off) };
                }
            }
        }

        gemm_fn(alpha, a_mat, b_mat, beta, c_mat, m, n, k)?;

        for j in 0..n {
            for i in 0..m {
                let dst_off = c_off + i as isize * c_row + j as isize * c_col;
                unsafe {
                    *c_ptr.offset(dst_off) = c_mat[i + j * m];
                }
            }
        }
        Ok(())
    };

    let mut result = Ok(());

    if nb == 0 {
        result = do_batch(0, 0, 0, &mut a_mat, &mut b_mat, &mut c_mat);
    } else {
        let a_fused = try_fuse_group_in_order(batch_dims, a_batch);
        let b_fused = try_fuse_group_in_order(batch_dims, b_batch);
        let c_fused = try_fuse_group_in_order(batch_dims, c_batch);
        let total = batch_iteration_count(batch_dims)?;

        if let (Some((_, a_step)), Some((_, b_step)), Some((_, c_step))) =
            (a_fused, b_fused, c_fused)
        {
            let mut a_off = 0isize;
            let mut b_off = 0isize;
            let mut c_off = 0isize;
            for _ in 0..total {
                if let Err(e) = do_batch(a_off, b_off, c_off, &mut a_mat, &mut b_mat, &mut c_mat) {
                    result = Err(e);
                    break;
                }
                match (
                    advance_batch_offset(a_off, a_step),
                    advance_batch_offset(b_off, b_step),
                    advance_batch_offset(c_off, c_step),
                ) {
                    (Ok(next_a), Ok(next_b), Ok(next_c)) => {
                        a_off = next_a;
                        b_off = next_b;
                        c_off = next_c;
                    }
                    _ => {
                        result = Err(Error::InvalidArgument("batch offset overflow".into()));
                        break;
                    }
                }
            }
        } else {
            let mut idx = vec![0usize; nb];
            for flat in 0..total {
                let a_off = batch_offset(flat, &idx, a_fused, a_batch)?;
                let b_off = batch_offset(flat, &idx, b_fused, b_batch)?;
                let c_off = batch_offset(flat, &idx, c_fused, c_batch)?;
                if let Err(e) = do_batch(a_off, b_off, c_off, &mut a_mat, &mut b_mat, &mut c_mat) {
                    result = Err(e);
                    break;
                }

                if flat + 1 < total {
                    for ax in 0..nb {
                        let next = idx[ax] + 1;
                        if next < batch_dims[ax] {
                            idx[ax] = next;
                            break;
                        } else {
                            idx[ax] = 0;
                        }
                    }
                }
            }
        }
    }

    ctx.put_scratch(a_mat);
    ctx.put_scratch(b_mat);
    ctx.put_scratch(c_mat);
    result
}

/// Strided batched GEMM via faer — zero allocation, zero copy.
#[cfg(feature = "gemm-faer")]
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_batched_gemm_strided<T: FaerGemm>(
    ctx: &CpuContext,
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
    batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let a = inputs[0];
    let b = inputs[1];
    let nb = batch_dims.len();
    let a_strides = a.strides();
    let b_strides = b.strides();
    let c_strides = output.strides();
    let a_row = a_strides[0];
    let a_col = a_strides[1];
    let a_batch = &a_strides[2..];
    let b_row = b_strides[0];
    let b_col = b_strides[1];
    let b_batch = &b_strides[2..];
    let c_row = c_strides[0];
    let c_col = c_strides[1];
    let c_batch = &c_strides[2..];

    let a_ptr = a.ptr();
    let b_ptr = b.ptr();
    let c_ptr = output.as_mut_ptr();
    let a_addr = a_ptr as usize;
    let b_addr = b_ptr as usize;
    let c_addr = c_ptr as usize;

    ctx.install(move || {
        let a_ptr = a_addr as *const T;
        let b_ptr = b_addr as *const T;
        let c_ptr = c_addr as *mut T;
        if nb == 0 {
            unsafe {
                T::strided_gemm(
                    alpha, a_ptr, m, k, a_row, a_col, b_ptr, n, b_row, b_col, beta, c_ptr, c_row,
                    c_col,
                );
            }
        } else {
            let a_fused = try_fuse_group_in_order(batch_dims, a_batch);
            let b_fused = try_fuse_group_in_order(batch_dims, b_batch);
            let c_fused = try_fuse_group_in_order(batch_dims, c_batch);
            let total = batch_iteration_count(batch_dims)?;

            if let (Some((_, a_step)), Some((_, b_step)), Some((_, c_step))) =
                (a_fused, b_fused, c_fused)
            {
                let mut a_off = 0isize;
                let mut b_off = 0isize;
                let mut c_off = 0isize;
                for _ in 0..total {
                    unsafe {
                        T::strided_gemm(
                            alpha,
                            a_ptr.offset(a_off),
                            m,
                            k,
                            a_row,
                            a_col,
                            b_ptr.offset(b_off),
                            n,
                            b_row,
                            b_col,
                            beta,
                            c_ptr.offset(c_off),
                            c_row,
                            c_col,
                        );
                    }
                    a_off = advance_batch_offset(a_off, a_step)?;
                    b_off = advance_batch_offset(b_off, b_step)?;
                    c_off = advance_batch_offset(c_off, c_step)?;
                }
            } else {
                let mut idx = vec![0usize; nb];
                for flat in 0..total {
                    let a_off = batch_offset(flat, &idx, a_fused, a_batch)?;
                    let b_off = batch_offset(flat, &idx, b_fused, b_batch)?;
                    let c_off = batch_offset(flat, &idx, c_fused, c_batch)?;
                    unsafe {
                        T::strided_gemm(
                            alpha,
                            a_ptr.offset(a_off),
                            m,
                            k,
                            a_row,
                            a_col,
                            b_ptr.offset(b_off),
                            n,
                            b_row,
                            b_col,
                            beta,
                            c_ptr.offset(c_off),
                            c_row,
                            c_col,
                        );
                    }

                    if flat + 1 < total {
                        for ax in 0..nb {
                            let next = idx[ax] + 1;
                            if next < batch_dims[ax] {
                                idx[ax] = next;
                                break;
                            } else {
                                idx[ax] = 0;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_batched_gemm<T: Scalar + 'static>(
    _ctx: &mut CpuContext,
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
    batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    if inputs.iter().any(|view| view.dims().contains(&0)) || output.dims().contains(&0) {
        scale_output(output, beta);
        return Ok(());
    }

    dispatch_standard_scalar_type!(T, Concrete, {
        let a = cast_strided_view!(inputs[0], T, Concrete);
        let b = cast_strided_view!(inputs[1], T, Concrete);
        let out = cast_strided_view_mut!(output, T, Concrete);
        let alpha = cast_scalar_value!(alpha, T, Concrete);
        let beta = cast_scalar_value!(beta, T, Concrete);

        #[cfg(feature = "gemm-faer")]
        {
            return execute_batched_gemm_strided(
                _ctx,
                alpha,
                &[a, b],
                beta,
                out,
                batch_dims,
                m,
                n,
                k,
            );
        }
        #[cfg(all(feature = "gemm-blas", not(feature = "gemm-faer")))]
        {
            return execute_batched_gemm_contiguous(
                _ctx,
                alpha,
                &[a, b],
                beta,
                out,
                batch_dims,
                m,
                n,
                k,
                <Concrete as BlasGemm>::contiguous_gemm,
            );
        }
    });

    Err(Error::InvalidArgument(format!(
        "BatchedGemm supports only f32, f64, Complex32, and Complex64 (got {})",
        std::any::type_name::<T>()
    )))
}
