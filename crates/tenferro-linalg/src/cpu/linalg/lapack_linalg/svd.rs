use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_tensor::TypedTensor;
use tlinalg_blas::svd::{SvdMode, SvdOutputs};
use tlinalg_blas::Op;

use super::helpers::{
    checked_product, has_zero_dim, matrix_with_batch_shape, pooled_output, provider,
    release_scratch, split_core_and_batch_result, tensor_from_vec_with_template,
    vector_with_batch_shape, LapackLinalg,
};

/// The pooled `(values, U, Vt)` buffers one batched SVD fills.
type SvdBuffers<T> = (Vec<<T as LapackLinalg>::RealScalar>, Vec<T>, Vec<T>);

/// `(U, S, Vt)` with the spectrum in the real scalar, as the faer route returns them.
pub(crate) type TypedSvd<T> = (
    TypedTensor<T>,
    TypedTensor<<T as LapackLinalg>::RealScalar>,
    TypedTensor<T>,
);

/// Run one batched SVD in `mode`, returning `(values, U, Vt)` in pooled buffers.
fn svd_buffers<T: LapackLinalg>(
    buffers: &mut BufferPool,
    op: Op,
    mode: SvdMode,
    input: &TypedTensor<T>,
    shapes: [&[usize]; 3],
) -> tenferro_tensor::Result<SvdBuffers<T>> {
    let op_name = op.as_str();
    let [s_shape, u_shape, vt_shape] = shapes;
    let mut s = pooled_output::<T::RealScalar>(buffers, op_name, "singular values", s_shape)?;
    let mut u = pooled_output::<T>(buffers, op_name, "left singular vectors", u_shape)?;
    let mut vt = pooled_output::<T>(buffers, op_name, "right singular vectors", vt_shape)?;
    let view = input.as_view();
    T::svd(
        buffers,
        op,
        mode,
        super::super::raw_view(op_name, &view)?,
        SvdOutputs {
            s: &mut s,
            u: &mut u,
            vt: &mut vt,
        },
    )
    .map_err(provider(op))?;
    Ok((s, u, vt))
}

pub(crate) fn svd<T: LapackLinalg>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<TypedSvd<T>> {
    let (matrix_shape, batch_shape) = split_core_and_batch_result(input, 2, "svd")?;
    let (m, n) = (matrix_shape[0], matrix_shape[1]);
    let k = m.min(n);
    let u_shape = matrix_with_batch_shape(m, k, batch_shape);
    let s_shape = vector_with_batch_shape(k, batch_shape);
    let vt_shape = matrix_with_batch_shape(k, n, batch_shape);
    if has_zero_dim(input.shape()) {
        return Ok((
            tensor_from_vec_with_template(u_shape, Vec::new(), input)?,
            tensor_from_vec_with_template(s_shape, Vec::new(), input)?,
            tensor_from_vec_with_template(vt_shape, Vec::new(), input)?,
        ));
    }
    let (s, u, vt) = svd_buffers(
        buffers,
        Op::Svd,
        SvdMode::Thin,
        input,
        [&s_shape, &u_shape, &vt_shape],
    )?;
    Ok((
        tensor_from_vec_with_template(u_shape, u, input)?,
        tensor_from_vec_with_template(s_shape, s, input)?,
        tensor_from_vec_with_template(vt_shape, vt, input)?,
    ))
}

pub(crate) fn svd_full<T: LapackLinalg>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<TypedSvd<T>> {
    let (matrix_shape, batch_shape) = split_core_and_batch_result(input, 2, "svd_full")?;
    let (m, n) = (matrix_shape[0], matrix_shape[1]);
    let u_shape = matrix_with_batch_shape(m, m, batch_shape);
    let s_shape = vector_with_batch_shape(m.min(n), batch_shape);
    let vt_shape = matrix_with_batch_shape(n, n, batch_shape);
    if has_zero_dim(input.shape()) {
        return empty_full_svd_outputs("svd_full", m, n, batch_shape, input);
    }
    let (s, u, vt) = svd_buffers(
        buffers,
        Op::SvdFull,
        SvdMode::Full,
        input,
        [&s_shape, &u_shape, &vt_shape],
    )?;
    Ok((
        tensor_from_vec_with_template(u_shape, u, input)?,
        tensor_from_vec_with_template(s_shape, s, input)?,
        tensor_from_vec_with_template(vt_shape, vt, input)?,
    ))
}

/// Full-SVD factors for an input with an empty core dimension.
///
/// The full variant keeps its `m x m` and `n x n` output shapes even when the other core
/// dimension is zero, so the factor for the non-empty dimension is the identity rather than an
/// empty tensor. This mirrors the faer provider so one public call has one shape and unitarity
/// contract.
fn empty_full_svd_outputs<T: LapackLinalg, U>(
    op: &'static str,
    m: usize,
    n: usize,
    batch_shape: &[usize],
    template: &TypedTensor<U>,
) -> tenferro_tensor::Result<TypedSvd<T>> {
    let blocks = checked_product(op, "batch shape", batch_shape)?;
    Ok((
        tensor_from_vec_with_template(
            matrix_with_batch_shape(m, m, batch_shape),
            identity_blocks::<T>(op, m, blocks)?,
            template,
        )?,
        tensor_from_vec_with_template(
            vector_with_batch_shape(m.min(n), batch_shape),
            Vec::new(),
            template,
        )?,
        tensor_from_vec_with_template(
            matrix_with_batch_shape(n, n, batch_shape),
            identity_blocks::<T>(op, n, blocks)?,
            template,
        )?,
    ))
}

/// `blocks` column-major `dim x dim` identity matrices laid out back to back.
fn identity_blocks<T: LapackLinalg>(
    op: &'static str,
    dim: usize,
    blocks: usize,
) -> tenferro_tensor::Result<Vec<T>> {
    let per_block = checked_product(op, "identity block", &[dim, dim])?;
    let len = checked_product(op, "identity stack", &[per_block, blocks])?;
    let mut data = vec![T::default(); len];
    for block in 0..blocks {
        let base = block * per_block;
        for index in 0..dim {
            data[base + index + index * dim] = T::unit();
        }
    }
    Ok(data)
}

pub(crate) fn svd_values<T: LapackLinalg>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<TypedTensor<T::RealScalar>> {
    let (matrix_shape, batch_shape) = split_core_and_batch_result(input, 2, "svd_values")?;
    let (m, n) = (matrix_shape[0], matrix_shape[1]);
    let s_shape = vector_with_batch_shape(m.min(n), batch_shape);
    if has_zero_dim(input.shape()) {
        return tensor_from_vec_with_template(s_shape, Vec::new(), input);
    }
    let (s, u, vt) = svd_buffers(
        buffers,
        Op::SvdValues,
        SvdMode::Values,
        input,
        [&s_shape, &[0], &[0]],
    )?;
    release_scratch(buffers, u);
    release_scratch(buffers, vt);
    tensor_from_vec_with_template(s_shape, s, input)
}
