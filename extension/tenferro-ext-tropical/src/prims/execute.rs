use strided_view::{StridedView, StridedViewMut};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tropical_gemm::{TropicalMaxMul, TropicalMaxPlus, TropicalMinPlus, TropicalSemiring};

use super::plan::TropicalPlan;
use super::view::{for_each_index, scale_output, unflatten_index};
use crate::scalar::{MaxMul, MaxPlus, MinPlus};

/// Trait for dispatching to SIMD-optimized tropical GEMM via tropical-gemm crate.
///
/// Maps tenferro's tropical scalar types to tropical-gemm's types and calls
/// `tropical_matmul_strided_batched` for SIMD-accelerated computation.
/// Generic over the inner scalar type (f32 or f64).
pub(crate) trait TropicalGemmDispatch: Scalar {
    /// The inner floating-point type (f32 or f64).
    type Inner: Copy + Default;
    /// Extract the inner scalar value.
    fn inner_value(&self) -> Self::Inner;
    /// Wrap an inner scalar value back into the tropical type.
    fn from_inner(v: Self::Inner) -> Self;
    /// Execute SIMD-optimized batched GEMM.
    /// Input: row-major buffers packed per batch element.
    /// Returns: row-major result buffer.
    fn dispatch_gemm(
        a: &[Self::Inner],
        b: &[Self::Inner],
        batch_size: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<Self::Inner>;
}

macro_rules! impl_tropical_gemm_dispatch {
    ($tropical_ty:ident, $inner:ty, $gemm_ty:ty) => {
        impl TropicalGemmDispatch for $tropical_ty<$inner> {
            type Inner = $inner;

            #[inline]
            fn inner_value(&self) -> $inner {
                self.0
            }

            #[inline]
            fn from_inner(v: $inner) -> Self {
                $tropical_ty(v)
            }

            fn dispatch_gemm(
                a: &[$inner],
                b: &[$inner],
                batch_size: usize,
                m: usize,
                k: usize,
                n: usize,
            ) -> Vec<$inner> {
                if batch_size <= 1 {
                    let result = tropical_gemm::tropical_matmul::<$gemm_ty>(a, m, k, b, n);
                    result.iter().map(|v| v.value()).collect()
                } else {
                    let result = tropical_gemm::tropical_matmul_strided_batched::<$gemm_ty>(
                        a, b, batch_size, m, k, n,
                    );
                    result.iter().map(|v| v.value()).collect()
                }
            }
        }
    };
}

impl_tropical_gemm_dispatch!(MaxPlus, f64, TropicalMaxPlus<f64>);
impl_tropical_gemm_dispatch!(MinPlus, f64, TropicalMinPlus<f64>);
impl_tropical_gemm_dispatch!(MaxMul, f64, TropicalMaxMul<f64>);
impl_tropical_gemm_dispatch!(MaxPlus, f32, TropicalMaxPlus<f32>);
impl_tropical_gemm_dispatch!(MinPlus, f32, TropicalMinPlus<f32>);
impl_tropical_gemm_dispatch!(MaxMul, f32, TropicalMaxMul<f32>);

pub(crate) fn execute_batched_gemm_optimized<T: Scalar + TropicalGemmDispatch>(
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
    let batch_size: usize = batch_dims.iter().product::<usize>().max(1);

    let a_total = batch_size * m * k;
    let b_total = batch_size * k * n;
    let mut a_buf: Vec<T::Inner> = Vec::with_capacity(a_total);
    let mut b_buf: Vec<T::Inner> = Vec::with_capacity(b_total);

    for batch_flat in 0..batch_size {
        let batch_idx = unflatten_index(batch_flat, batch_dims);
        for i in 0..m {
            for j in 0..k {
                let mut idx = vec![i, j];
                idx.extend_from_slice(&batch_idx);
                a_buf.push(a.get(&idx).inner_value());
            }
        }
    }
    for batch_flat in 0..batch_size {
        let batch_idx = unflatten_index(batch_flat, batch_dims);
        for i in 0..k {
            for j in 0..n {
                let mut idx = vec![i, j];
                idx.extend_from_slice(&batch_idx);
                b_buf.push(b.get(&idx).inner_value());
            }
        }
    }

    let result = T::dispatch_gemm(&a_buf, &b_buf, batch_size, m, k, n);

    for batch_flat in 0..batch_size {
        let batch_idx = unflatten_index(batch_flat, batch_dims);
        for i in 0..m {
            for j in 0..n {
                let mut c_idx = vec![i, j];
                c_idx.extend_from_slice(&batch_idx);
                let flat = batch_flat * m * n + i * n + j;
                let val = T::from_inner(result[flat]);
                let old = if beta == T::zero() {
                    T::zero()
                } else {
                    beta * output.get(&c_idx)
                };
                output.set(&c_idx, alpha * val + old);
            }
        }
    }
    Ok(())
}

pub(crate) fn execute_batched_gemm_fallback<T: Scalar>(
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
    let batch_size: usize = if batch_dims.is_empty() {
        1
    } else {
        batch_dims.iter().product()
    };

    for batch_flat in 0..batch_size {
        let batch_idx = unflatten_index(batch_flat, batch_dims);
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for kk in 0..k {
                    let mut a_idx = vec![i, kk];
                    a_idx.extend_from_slice(&batch_idx);
                    let mut b_idx = vec![kk, j];
                    b_idx.extend_from_slice(&batch_idx);
                    sum = sum + a.get(&a_idx) * b.get(&b_idx);
                }
                let mut c_idx = vec![i, j];
                c_idx.extend_from_slice(&batch_idx);
                let old = if beta == T::zero() {
                    T::zero()
                } else {
                    beta * output.get(&c_idx)
                };
                output.set(&c_idx, alpha * sum + old);
            }
        }
    }
    Ok(())
}

fn execute_reduce_sum<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    reduced_axes: &[usize],
) -> Result<()> {
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&ax| in_dims[ax]).collect();
    let reduced_total: usize = reduced_dims.iter().product();

    for_each_index(&out_dims, |out_idx| {
        let mut sum = T::zero();
        for red_flat in 0..reduced_total {
            let red_idx = unflatten_index(red_flat, &reduced_dims);
            let mut in_idx = Vec::with_capacity(in_dims.len());
            let mut out_pos = 0;
            let mut red_pos = 0;
            for ax in 0..in_dims.len() {
                if red_pos < reduced_axes.len() && reduced_axes[red_pos] == ax {
                    in_idx.push(red_idx[red_pos]);
                    red_pos += 1;
                } else {
                    in_idx.push(out_idx[out_pos]);
                    out_pos += 1;
                }
            }
            sum = sum + input.get(&in_idx);
        }
        let old = if beta == T::zero() {
            T::zero()
        } else {
            beta * output.get(out_idx)
        };
        output.set(out_idx, alpha * sum + old);
    });
    Ok(())
}

pub(crate) fn execute_trace<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    paired_axes: &[(usize, usize)],
    free_axes: &[usize],
) -> Result<()> {
    if paired_axes.is_empty() {
        return Err(Error::InvalidArgument(
            "trace requires at least one paired axis".into(),
        ));
    }

    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    for &(ax1, ax2) in paired_axes {
        if ax1 >= in_dims.len() || ax2 >= in_dims.len() {
            return Err(Error::InvalidArgument(
                "trace paired axis out of bounds".into(),
            ));
        }
    }
    for &ax in free_axes {
        if ax >= in_dims.len() {
            return Err(Error::InvalidArgument(
                "trace free axis out of bounds".into(),
            ));
        }
    }
    if out_dims.len() != free_axes.len() {
        return Err(Error::InvalidArgument(
            "trace output rank does not match free axes".into(),
        ));
    }

    let diag_dim = in_dims[paired_axes[0].0];

    for_each_index(&out_dims, |out_idx| {
        let mut sum = T::zero();
        for d in 0..diag_dim {
            let mut in_idx = vec![0; in_dims.len()];
            for (out_pos, &in_ax) in free_axes.iter().enumerate() {
                in_idx[in_ax] = out_idx[out_pos];
            }
            for &(ax1, ax2) in paired_axes {
                in_idx[ax1] = d;
                in_idx[ax2] = d;
            }
            sum = sum + input.get(&in_idx);
        }
        let old = if beta == T::zero() {
            T::zero()
        } else {
            beta * output.get(out_idx)
        };
        output.set(out_idx, alpha * sum + old);
    });
    Ok(())
}

pub(crate) fn execute_anti_trace<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    paired_axes: &[(usize, usize)],
    free_axes: &[usize],
) -> Result<()> {
    if paired_axes.is_empty() {
        return Err(Error::InvalidArgument(
            "anti-trace requires at least one paired axis".into(),
        ));
    }

    scale_output(output, beta);

    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    for &(ax1, ax2) in paired_axes {
        if ax1 >= out_dims.len() || ax2 >= out_dims.len() {
            return Err(Error::InvalidArgument(
                "anti-trace paired axis out of bounds".into(),
            ));
        }
    }
    for &ax in free_axes {
        if ax >= out_dims.len() {
            return Err(Error::InvalidArgument(
                "anti-trace free axis out of bounds".into(),
            ));
        }
    }
    if in_dims.len() != free_axes.len() {
        return Err(Error::InvalidArgument(
            "anti-trace input rank does not match free axes".into(),
        ));
    }
    let diag_dim = out_dims[paired_axes[0].0];

    for_each_index(&in_dims, |in_idx| {
        let val = alpha * input.get(in_idx);
        for d in 0..diag_dim {
            let mut out_idx = vec![0; out_dims.len()];
            for (in_pos, &out_ax) in free_axes.iter().enumerate() {
                out_idx[out_ax] = in_idx[in_pos];
            }
            for &(ax1, ax2) in paired_axes {
                out_idx[ax1] = d;
                out_idx[ax2] = d;
            }
            let old = output.get(&out_idx);
            output.set(&out_idx, old + val);
        }
    });
    Ok(())
}

pub(crate) fn execute_anti_diag<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
    paired_axes: &[(usize, usize)],
    free_axes: &[usize],
) -> Result<()> {
    if paired_axes.is_empty() {
        return Err(Error::InvalidArgument(
            "anti-diag requires at least one paired axis".into(),
        ));
    }

    scale_output(output, beta);

    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    for &(ax1, ax2) in paired_axes {
        if ax1 >= out_dims.len() || ax2 >= out_dims.len() {
            return Err(Error::InvalidArgument(
                "anti-diag paired axis out of bounds".into(),
            ));
        }
    }
    for &ax in free_axes {
        if ax >= out_dims.len() {
            return Err(Error::InvalidArgument(
                "anti-diag free axis out of bounds".into(),
            ));
        }
    }
    if in_dims.len() != free_axes.len() {
        return Err(Error::InvalidArgument(
            "anti-diag input rank does not match free axes".into(),
        ));
    }

    for_each_index(&in_dims, |in_idx| {
        let val = alpha * input.get(in_idx);
        let mut out_idx = vec![0; out_dims.len()];
        for (in_pos, &out_ax) in free_axes.iter().enumerate() {
            out_idx[out_ax] = in_idx[in_pos];
        }
        for &(ax1, ax2) in paired_axes {
            out_idx[ax2] = out_idx[ax1];
        }
        let old = output.get(&out_idx);
        output.set(&out_idx, old + val);
    });
    Ok(())
}

pub(crate) fn tropical_execute<T: Scalar>(
    plan: &TropicalPlan<T>,
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
) -> Result<()> {
    match plan {
        TropicalPlan::BatchedGemm {
            batch_dims,
            m,
            n,
            k,
            ..
        } => {
            if inputs.len() != 2 {
                return Err(Error::InvalidArgument(
                    "BatchedGemm execute requires 2 input tensors".into(),
                ));
            }
            execute_batched_gemm_fallback(alpha, inputs, beta, output, batch_dims, *m, *n, *k)
        }
        TropicalPlan::Reduce { reduced_axes, .. } => {
            if inputs.len() != 1 {
                return Err(Error::InvalidArgument(
                    "ReduceAdd execute requires 1 input tensor".into(),
                ));
            }
            execute_reduce_sum(alpha, inputs[0], beta, output, reduced_axes)
        }
        TropicalPlan::Trace {
            paired_axes,
            free_axes,
            ..
        } => {
            if inputs.len() != 1 {
                return Err(Error::InvalidArgument(
                    "Trace execute requires 1 input tensor".into(),
                ));
            }
            execute_trace(alpha, inputs[0], beta, output, paired_axes, free_axes)
        }
        TropicalPlan::AntiTrace {
            paired_axes,
            free_axes,
            ..
        } => {
            if inputs.len() != 1 {
                return Err(Error::InvalidArgument(
                    "AntiTrace execute requires 1 input tensor".into(),
                ));
            }
            execute_anti_trace(alpha, inputs[0], beta, output, paired_axes, free_axes)
        }
        TropicalPlan::AntiDiag {
            paired_axes,
            free_axes,
            ..
        } => {
            if inputs.len() != 1 {
                return Err(Error::InvalidArgument(
                    "AntiDiag execute requires 1 input tensor".into(),
                ));
            }
            execute_anti_diag(alpha, inputs[0], beta, output, paired_axes, free_axes)
        }
        TropicalPlan::MakeContiguous { .. } => {
            if inputs.len() != 1 {
                return Err(Error::InvalidArgument(
                    "MakeContiguous execute requires 1 input tensor".into(),
                ));
            }
            execute_make_contiguous(alpha, inputs[0], beta, output)
        }
    }
}

pub(crate) fn execute_make_contiguous<T: Scalar>(
    alpha: T,
    input: &StridedView<T>,
    beta: T,
    output: &mut StridedViewMut<T>,
) -> Result<()> {
    if alpha == T::one() && beta == T::zero() {
        strided_perm::copy_into(output, input).map_err(|e| Error::StrideError(e.to_string()))?;
    } else {
        let dims = output.dims().to_vec();
        for_each_index(&dims, |idx| {
            let val = alpha * input.get(idx);
            if beta == T::zero() {
                output.set(idx, val);
            } else {
                output.set(idx, val + beta * output.get(idx));
            }
        });
    }
    Ok(())
}
