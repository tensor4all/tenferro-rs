use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use strided_kernel::{col_major_strides, copy_into, map_into, Identity, StridedView};

use crate::{
    buffer_pool::{BufferPool, PoolScalar},
    types::{flat_to_multi, Tensor, TensorRank, TypedTensor, TypedTensorView},
    DType,
};

use super::{
    cpu_backend_buffer_error, tensor_from_array, typed_array_uninit, typed_array_uninit_from_pool,
    typed_view, typed_view_from_view,
};

fn with_local_pool<T>(f: impl FnOnce(&mut BufferPool) -> T) -> T {
    let mut buffers = BufferPool::new();
    f(&mut buffers)
}

fn validate_rank(op: &'static str, expected: usize, actual: usize) -> crate::Result<()> {
    if expected != actual {
        return Err(crate::Error::RankMismatch {
            op,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_axis(op: &'static str, axis: usize, rank: usize) -> crate::Result<()> {
    if axis >= rank {
        return Err(crate::Error::AxisOutOfBounds { op, axis, rank });
    }
    Ok(())
}

fn validate_axes_distinct(op: &'static str, axis_a: usize, axis_b: usize) -> crate::Result<()> {
    if axis_a == axis_b {
        return Err(crate::Error::DuplicateAxis {
            op,
            axis: axis_a,
            role: "axes",
        });
    }
    Ok(())
}

fn validate_permutation(op: &'static str, perm: &[usize], rank: usize) -> crate::Result<()> {
    validate_rank(op, rank, perm.len())?;
    let mut seen = vec![false; rank];
    for &axis in perm {
        validate_axis(op, axis, rank)?;
        if seen[axis] {
            return Err(crate::Error::DuplicateAxis {
                op,
                axis,
                role: "perm",
            });
        }
        seen[axis] = true;
    }
    Ok(())
}

macro_rules! dispatch_tensor_unary_result {
    ($input:expr, |$tensor:ident| $body:expr) => {
        match $input {
            Tensor::F32($tensor) => Ok(Tensor::F32($body?)),
            Tensor::F64($tensor) => Ok(Tensor::F64($body?)),
            Tensor::I32($tensor) => Ok(Tensor::I32($body?)),
            Tensor::I64($tensor) => Ok(Tensor::I64($body?)),
            Tensor::Bool($tensor) => Ok(Tensor::Bool($body?)),
            Tensor::C32($tensor) => Ok(Tensor::C32($body?)),
            Tensor::C64($tensor) => Ok(Tensor::C64($body?)),
        }
    };
}

macro_rules! dispatch_tensor_unary_with_bool_special_result {
    ($input:expr, |$tensor:ident| $body:expr, bool |$bool_tensor:ident| $bool_body:expr) => {
        match $input {
            Tensor::F32($tensor) => Ok(Tensor::F32($body?)),
            Tensor::F64($tensor) => Ok(Tensor::F64($body?)),
            Tensor::I32($tensor) => Ok(Tensor::I32($body?)),
            Tensor::I64($tensor) => Ok(Tensor::I64($body?)),
            Tensor::Bool($bool_tensor) => Ok(Tensor::Bool($bool_body?)),
            Tensor::C32($tensor) => Ok(Tensor::C32($body?)),
            Tensor::C64($tensor) => Ok(Tensor::C64($body?)),
        }
    };
}

fn host_view<'a, T: Copy>(
    op: &'static str,
    tensor: &'a TypedTensor<T>,
) -> crate::Result<StridedView<'a, T, Identity>> {
    match &tensor.buffer {
        crate::Buffer::Host(data) => {
            let strides = col_major_strides(tensor.shape());
            StridedView::new(data, tensor.shape(), &strides, 0)
                .map_err(|err| crate::Error::backend_failure(op, err))
        }
        crate::Buffer::Backend(_) => Err(cpu_backend_buffer_error(op)),
    }
}

fn copy_view_to_array<T: Copy + Clone>(
    op: &'static str,
    mut out: strided_kernel::StridedArray<T>,
    src: &StridedView<'_, T>,
) -> crate::Result<TypedTensor<T>> {
    copy_into(&mut out.view_mut(), src).map_err(|err| crate::Error::backend_failure(op, err))?;
    Ok(tensor_from_array(out))
}

fn zeroed_tensor_from_pool<T>(buffers: &mut BufferPool, shape: Vec<usize>) -> TypedTensor<T>
where
    T: Zero + Clone + PoolScalar + 'static,
{
    filled_tensor_from_pool(buffers, shape, T::zero())
}

fn filled_tensor_from_pool<T>(
    buffers: &mut BufferPool,
    shape: Vec<usize>,
    fill: T,
) -> TypedTensor<T>
where
    T: Copy + Clone + PoolScalar + 'static,
{
    let len = shape.iter().product();
    // SAFETY: every element is initialized with `fill` before returning.
    let mut data = unsafe { T::pool_acquire(buffers, len) };
    data.fill(fill);
    TypedTensor::from_vec_col_major(shape, data)
}

fn clone_host_tensor_from_pool<T>(
    buffers: &mut BufferPool,
    op: &'static str,
    tensor: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + PoolScalar + 'static,
{
    let input = match &tensor.buffer {
        crate::Buffer::Host(data) => data,
        crate::Buffer::Backend(_) => return Err(cpu_backend_buffer_error(op)),
    };
    // SAFETY: copy_from_slice initializes every element before returning.
    let mut data = unsafe { T::pool_acquire(buffers, input.len()) };
    data.copy_from_slice(input);
    Ok(TypedTensor::from_buffer_col_major(
        tensor.shape().to_vec(),
        crate::Buffer::Host(data),
        tensor.placement.clone(),
    ))
}

pub fn transpose(input: &Tensor, perm: &[usize]) -> crate::Result<Tensor> {
    with_local_pool(|buffers| transpose_with_pool(buffers, input, perm))
}

pub(crate) fn transpose_with_pool(
    buffers: &mut BufferPool,
    input: &Tensor,
    perm: &[usize],
) -> crate::Result<Tensor> {
    dispatch_tensor_unary_result!(input, |t| typed_transpose_with_pool(buffers, t, perm))
}

pub fn reshape(input: &Tensor, shape: &[usize]) -> crate::Result<Tensor> {
    dispatch_tensor_unary_result!(input, |t| typed_reshape(t, shape))
}

pub fn broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> crate::Result<Tensor> {
    with_local_pool(|buffers| broadcast_in_dim_with_pool(buffers, input, shape, dims))
}

pub(crate) fn broadcast_in_dim_with_pool(
    buffers: &mut BufferPool,
    input: &Tensor,
    shape: &[usize],
    dims: &[usize],
) -> crate::Result<Tensor> {
    dispatch_tensor_unary_result!(input, |t| typed_broadcast_in_dim_with_pool(
        buffers, t, shape, dims
    ))
}

pub fn convert(input: &Tensor, to: DType) -> crate::Result<Tensor> {
    with_local_pool(|buffers| convert_with_pool(buffers, input, to))
}

pub(crate) fn convert_with_pool(
    buffers: &mut BufferPool,
    input: &Tensor,
    to: DType,
) -> crate::Result<Tensor> {
    macro_rules! converted {
        ($variant:ident, $tensor:expr, $map:expr) => {
            Ok(Tensor::$variant(typed_convert_with_pool(
                buffers, $tensor, $map,
            )?))
        };
    }

    match (input, to) {
        (Tensor::F32(t), DType::F32) => Ok(Tensor::F32(t.clone())),
        (Tensor::F32(t), DType::F64) => converted!(F64, t, |x| x as f64),
        (Tensor::F32(t), DType::I32) => converted!(I32, t, |x| x as i32),
        (Tensor::F32(t), DType::I64) => converted!(I64, t, |x| x as i64),
        (Tensor::F32(t), DType::Bool) => converted!(Bool, t, |x| x != 0.0),
        (Tensor::F32(t), DType::C32) => converted!(C32, t, |x| Complex32::new(x, 0.0)),
        (Tensor::F32(t), DType::C64) => {
            converted!(C64, t, |x| Complex64::new(x as f64, 0.0))
        }
        (Tensor::F64(t), DType::F32) => converted!(F32, t, |x| x as f32),
        (Tensor::F64(t), DType::F64) => Ok(Tensor::F64(t.clone())),
        (Tensor::F64(t), DType::I32) => converted!(I32, t, |x| x as i32),
        (Tensor::F64(t), DType::I64) => converted!(I64, t, |x| x as i64),
        (Tensor::F64(t), DType::Bool) => converted!(Bool, t, |x| x != 0.0),
        (Tensor::F64(t), DType::C32) => {
            converted!(C32, t, |x| Complex32::new(x as f32, 0.0))
        }
        (Tensor::F64(t), DType::C64) => converted!(C64, t, |x| Complex64::new(x, 0.0)),
        (Tensor::I32(t), DType::F32) => converted!(F32, t, |x| x as f32),
        (Tensor::I32(t), DType::F64) => converted!(F64, t, |x| x as f64),
        (Tensor::I32(t), DType::I32) => Ok(Tensor::I32(t.clone())),
        (Tensor::I32(t), DType::I64) => converted!(I64, t, |x| x as i64),
        (Tensor::I32(t), DType::Bool) => converted!(Bool, t, |x| x != 0),
        (Tensor::I32(t), DType::C32) => {
            converted!(C32, t, |x| Complex32::new(x as f32, 0.0))
        }
        (Tensor::I32(t), DType::C64) => {
            converted!(C64, t, |x| Complex64::new(x as f64, 0.0))
        }
        (Tensor::I64(t), DType::F32) => converted!(F32, t, |x| x as f32),
        (Tensor::I64(t), DType::F64) => converted!(F64, t, |x| x as f64),
        (Tensor::I64(t), DType::I32) => converted!(I32, t, |x| x as i32),
        (Tensor::I64(t), DType::I64) => Ok(Tensor::I64(t.clone())),
        (Tensor::I64(t), DType::Bool) => converted!(Bool, t, |x| x != 0),
        (Tensor::I64(t), DType::C32) => {
            converted!(C32, t, |x| Complex32::new(x as f32, 0.0))
        }
        (Tensor::I64(t), DType::C64) => {
            converted!(C64, t, |x| Complex64::new(x as f64, 0.0))
        }
        (Tensor::Bool(t), DType::F32) => converted!(F32, t, |x| if x { 1.0 } else { 0.0 }),
        (Tensor::Bool(t), DType::F64) => converted!(F64, t, |x| if x { 1.0 } else { 0.0 }),
        (Tensor::Bool(t), DType::I32) => converted!(I32, t, |x| if x { 1 } else { 0 }),
        (Tensor::Bool(t), DType::I64) => converted!(I64, t, |x| if x { 1 } else { 0 }),
        (Tensor::Bool(t), DType::Bool) => Ok(Tensor::Bool(t.clone())),
        (Tensor::Bool(t), DType::C32) => {
            converted!(C32, t, |x| Complex32::new(if x { 1.0 } else { 0.0 }, 0.0))
        }
        (Tensor::Bool(t), DType::C64) => {
            converted!(C64, t, |x| Complex64::new(if x { 1.0 } else { 0.0 }, 0.0))
        }
        (Tensor::C32(t), DType::F32) => converted!(F32, t, |z| z.re),
        (Tensor::C32(t), DType::F64) => converted!(F64, t, |z| z.re as f64),
        (Tensor::C32(t), DType::I32) => converted!(I32, t, |z| z.re as i32),
        (Tensor::C32(t), DType::I64) => converted!(I64, t, |z| z.re as i64),
        (Tensor::C32(t), DType::Bool) => converted!(Bool, t, |z| z.re != 0.0 || z.im != 0.0),
        (Tensor::C32(t), DType::C32) => Ok(Tensor::C32(t.clone())),
        (Tensor::C32(t), DType::C64) => {
            converted!(C64, t, |z| Complex64::new(z.re as f64, z.im as f64))
        }
        (Tensor::C64(t), DType::F32) => converted!(F32, t, |z| z.re as f32),
        (Tensor::C64(t), DType::F64) => converted!(F64, t, |z| z.re),
        (Tensor::C64(t), DType::I32) => converted!(I32, t, |z| z.re as i32),
        (Tensor::C64(t), DType::I64) => converted!(I64, t, |z| z.re as i64),
        (Tensor::C64(t), DType::Bool) => converted!(Bool, t, |z| z.re != 0.0 || z.im != 0.0),
        (Tensor::C64(t), DType::C32) => {
            converted!(C32, t, |z| Complex32::new(z.re as f32, z.im as f32))
        }
        (Tensor::C64(t), DType::C64) => Ok(Tensor::C64(t.clone())),
    }
}

pub fn extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor> {
    with_local_pool(|buffers| extract_diagonal_with_pool(buffers, input, axis_a, axis_b))
}

pub(crate) fn extract_diagonal_with_pool(
    buffers: &mut BufferPool,
    input: &Tensor,
    axis_a: usize,
    axis_b: usize,
) -> crate::Result<Tensor> {
    dispatch_tensor_unary_result!(input, |t| typed_extract_diagonal_with_pool(
        buffers, t, axis_a, axis_b
    ))
}

pub fn embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor> {
    with_local_pool(|buffers| embed_diagonal_with_pool(buffers, input, axis_a, axis_b))
}

pub(crate) fn embed_diagonal_with_pool(
    buffers: &mut BufferPool,
    input: &Tensor,
    axis_a: usize,
    axis_b: usize,
) -> crate::Result<Tensor> {
    dispatch_tensor_unary_with_bool_special_result!(
        input,
        |t| typed_embed_diagonal_with_pool(buffers, t, axis_a, axis_b),
        bool | t
            | typed_embed_diagonal_impl(t, axis_a, axis_b, |shape| {
                filled_tensor_from_pool(buffers, shape, false)
            })
    )
}

pub fn tril(input: &Tensor, k: i64) -> crate::Result<Tensor> {
    with_local_pool(|buffers| tril_with_pool(buffers, input, k))
}

pub(crate) fn tril_with_pool(
    buffers: &mut BufferPool,
    input: &Tensor,
    k: i64,
) -> crate::Result<Tensor> {
    dispatch_tensor_unary_with_bool_special_result!(
        input,
        |t| typed_tril_with_pool(buffers, t, k),
        bool | t | typed_triangular_mask_with_fill_pool(buffers, t, k, false, false)
    )
}

pub fn triu(input: &Tensor, k: i64) -> crate::Result<Tensor> {
    with_local_pool(|buffers| triu_with_pool(buffers, input, k))
}

pub(crate) fn triu_with_pool(
    buffers: &mut BufferPool,
    input: &Tensor,
    k: i64,
) -> crate::Result<Tensor> {
    dispatch_tensor_unary_with_bool_special_result!(
        input,
        |t| typed_triu_with_pool(buffers, t, k),
        bool | t | typed_triangular_mask_with_fill_pool(buffers, t, k, true, false)
    )
}

pub fn typed_transpose<T: Copy + Clone>(
    tensor: &TypedTensor<T>,
    perm: &[usize],
) -> crate::Result<TypedTensor<T>> {
    validate_permutation("transpose", perm, tensor.shape().len())?;
    let src = host_view("transpose", tensor)?;
    let permuted = src
        .permute(perm)
        .map_err(|err| crate::Error::backend_failure("transpose", err))?;
    // SAFETY: copy_into overwrites every output element.
    let out = unsafe { typed_array_uninit(permuted.dims()) };
    copy_view_to_array("transpose", out, &permuted)
}

fn typed_transpose_view_impl<T, R>(
    view: &TypedTensorView<'_, T, R>,
    perm: &[usize],
    make_out: impl FnOnce(&[usize]) -> strided_kernel::StridedArray<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + 'static,
    R: TensorRank,
{
    validate_permutation("transpose", perm, view.shape().len())?;
    let src = typed_view_from_view("transpose", view)?;
    let permuted = src
        .permute(perm)
        .map_err(|err| crate::Error::backend_failure("transpose", err))?;
    // SAFETY: copy_into overwrites every output element.
    let out = make_out(permuted.dims());
    copy_view_to_array("transpose", out, &permuted)
}

pub(crate) fn typed_transpose_with_pool<T>(
    buffers: &mut BufferPool,
    tensor: &TypedTensor<T>,
    perm: &[usize],
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + PoolScalar + 'static,
{
    typed_transpose_view_with_pool(buffers, &tensor.as_view(), perm)
}

pub(crate) fn typed_transpose_view_with_pool<T, R>(
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T, R>,
    perm: &[usize],
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + PoolScalar + 'static,
    R: TensorRank,
{
    typed_transpose_view_impl(view, perm, |shape| unsafe {
        typed_array_uninit_from_pool(buffers, shape)
    })
}

pub fn typed_reshape<T: Clone + 'static>(
    tensor: &TypedTensor<T>,
    shape: &[usize],
) -> crate::Result<TypedTensor<T>> {
    let old_n: usize = tensor.shape().iter().product();
    let new_n: usize = shape.iter().product();
    if old_n != new_n {
        return Err(crate::Error::ShapeMismatch {
            op: "reshape",
            lhs: tensor.shape().to_vec(),
            rhs: shape.to_vec(),
        });
    }
    Ok(TypedTensor::from_buffer_col_major(
        shape.to_vec(),
        tensor.buffer.clone(),
        tensor.placement.clone(),
    ))
}

pub fn typed_broadcast_in_dim<T: Copy + Clone>(
    tensor: &TypedTensor<T>,
    shape: &[usize],
    dims: &[usize],
) -> crate::Result<TypedTensor<T>> {
    typed_broadcast_in_dim_impl(tensor, shape, dims, |shape| unsafe {
        typed_array_uninit(shape)
    })
}

pub(crate) fn typed_broadcast_in_dim_with_pool<T>(
    buffers: &mut BufferPool,
    tensor: &TypedTensor<T>,
    shape: &[usize],
    dims: &[usize],
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + PoolScalar,
{
    typed_broadcast_in_dim_impl(tensor, shape, dims, |shape| unsafe {
        typed_array_uninit_from_pool(buffers, shape)
    })
}

fn typed_broadcast_in_dim_impl<T>(
    tensor: &TypedTensor<T>,
    shape: &[usize],
    dims: &[usize],
    make_out: impl FnOnce(&[usize]) -> strided_kernel::StridedArray<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone,
{
    validate_rank("broadcast_in_dim", tensor.shape().len(), dims.len())?;
    let mut seen = vec![false; shape.len()];
    let mut base_dims = vec![1usize; shape.len()];
    let mut base_strides = vec![0isize; shape.len()];
    let source_strides = col_major_strides(tensor.shape());
    for (src_axis, &dst_axis) in dims.iter().enumerate() {
        validate_axis("broadcast_in_dim", dst_axis, shape.len())?;
        if seen[dst_axis] {
            return Err(crate::Error::DuplicateAxis {
                op: "broadcast_in_dim",
                axis: dst_axis,
                role: "dims",
            });
        }
        seen[dst_axis] = true;
        let source_dim = tensor.shape()[src_axis];
        let target_dim = shape[dst_axis];
        if source_dim != target_dim && source_dim != 1 {
            return Err(crate::Error::ShapeMismatch {
                op: "broadcast_in_dim",
                lhs: tensor.shape().to_vec(),
                rhs: shape.to_vec(),
            });
        }
        base_dims[dst_axis] = source_dim;
        base_strides[dst_axis] = source_strides[src_axis];
    }
    let base: StridedView<'_, T, Identity> = match &tensor.buffer {
        crate::Buffer::Host(data) => StridedView::new(data, &base_dims, &base_strides, 0)
            .map_err(|err| crate::Error::backend_failure("broadcast_in_dim", err))?,
        crate::Buffer::Backend(_) => return Err(cpu_backend_buffer_error("broadcast_in_dim")),
    };
    let broadcast: StridedView<'_, T, Identity> = base
        .broadcast(shape)
        .map_err(|err| crate::Error::backend_failure("broadcast_in_dim", err))?;
    // SAFETY: copy_into overwrites every output element.
    let mut out = make_out(shape);
    copy_into(&mut out.view_mut(), &broadcast)
        .map_err(|err| crate::Error::backend_failure("broadcast_in_dim", err))?;
    Ok(tensor_from_array(out))
}

fn typed_convert_with_pool<S, T>(
    buffers: &mut BufferPool,
    tensor: &TypedTensor<S>,
    f: impl Fn(S) -> T,
) -> crate::Result<TypedTensor<T>>
where
    S: Copy,
    T: Copy + Clone + PoolScalar,
{
    // SAFETY: map_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, tensor.shape()) };
    map_into(&mut out.view_mut(), &typed_view("convert", tensor)?, f)
        .map_err(|err| crate::Error::backend_failure("convert", err))?;
    Ok(tensor_from_array(out))
}

pub fn typed_extract_diagonal<T: Copy + Clone>(
    tensor: &TypedTensor<T>,
    axis_a: usize,
    axis_b: usize,
) -> crate::Result<TypedTensor<T>> {
    validate_axis("extract_diagonal", axis_a, tensor.shape().len())?;
    validate_axis("extract_diagonal", axis_b, tensor.shape().len())?;
    validate_axes_distinct("extract_diagonal", axis_a, axis_b)?;

    let diag = host_view("extract_diagonal", tensor)?
        .diagonal_view(&[(axis_a, axis_b)])
        .map_err(|err| crate::Error::backend_failure("extract_diagonal", err))?;
    // SAFETY: copy_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit(diag.dims()) };
    copy_into(&mut out.view_mut(), &diag)
        .map_err(|err| crate::Error::backend_failure("extract_diagonal", err))?;
    Ok(tensor_from_array(out))
}

pub(crate) fn typed_extract_diagonal_with_pool<T>(
    buffers: &mut BufferPool,
    tensor: &TypedTensor<T>,
    axis_a: usize,
    axis_b: usize,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + PoolScalar,
{
    validate_axis("extract_diagonal", axis_a, tensor.shape().len())?;
    validate_axis("extract_diagonal", axis_b, tensor.shape().len())?;
    validate_axes_distinct("extract_diagonal", axis_a, axis_b)?;

    let diag = host_view("extract_diagonal", tensor)?
        .diagonal_view(&[(axis_a, axis_b)])
        .map_err(|err| crate::Error::backend_failure("extract_diagonal", err))?;
    // SAFETY: copy_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, diag.dims()) };
    copy_into(&mut out.view_mut(), &diag)
        .map_err(|err| crate::Error::backend_failure("extract_diagonal", err))?;
    Ok(tensor_from_array(out))
}

pub fn typed_embed_diagonal<T: Copy + Zero + Clone>(
    tensor: &TypedTensor<T>,
    axis_a: usize,
    axis_b: usize,
) -> crate::Result<TypedTensor<T>> {
    typed_embed_diagonal_impl(tensor, axis_a, axis_b, |shape| TypedTensor::zeros(shape))
}

pub(crate) fn typed_embed_diagonal_with_pool<T>(
    buffers: &mut BufferPool,
    tensor: &TypedTensor<T>,
    axis_a: usize,
    axis_b: usize,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Zero + Clone + PoolScalar + 'static,
{
    typed_embed_diagonal_impl(tensor, axis_a, axis_b, |shape| {
        zeroed_tensor_from_pool(buffers, shape)
    })
}

fn typed_embed_diagonal_impl<T>(
    tensor: &TypedTensor<T>,
    axis_a: usize,
    axis_b: usize,
    make_zeroed: impl FnOnce(Vec<usize>) -> TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone,
{
    validate_axis("embed_diagonal", axis_a, tensor.shape().len())?;
    if axis_b > tensor.shape().len() {
        return Err(crate::Error::AxisOutOfBounds {
            op: "embed_diagonal",
            axis: axis_b,
            rank: tensor.shape().len(),
        });
    }

    let n = tensor.shape()[axis_a];
    let mut out_shape = tensor.shape().to_vec();
    out_shape.insert(axis_b, n);
    let mut out = make_zeroed(out_shape);

    let in_rank = tensor.shape().len();
    let out_rank = out.shape().len();
    let mut in_idx = vec![0usize; in_rank];
    let mut out_idx = vec![0usize; out_rank];

    let input_data = match &tensor.buffer {
        crate::Buffer::Host(data) => data,
        crate::Buffer::Backend(_) => return Err(cpu_backend_buffer_error("embed_diagonal")),
    };

    for flat in 0..tensor.n_elements() {
        flat_to_multi(flat, tensor.shape(), &mut in_idx);
        let diag_val = in_idx[axis_a];
        let mut src_axis = 0usize;
        for out_axis in 0..out_rank {
            if out_axis == axis_b {
                out_idx[out_axis] = diag_val;
            } else {
                out_idx[out_axis] = in_idx[src_axis];
                src_axis += 1;
            }
        }
        *out.get_mut(&out_idx) = input_data[flat];
    }
    Ok(out)
}

pub fn typed_tril<T: Copy + Zero + Clone>(
    tensor: &TypedTensor<T>,
    k: i64,
) -> crate::Result<TypedTensor<T>> {
    typed_triangular_mask(tensor, k, false)
}

pub(crate) fn typed_tril_with_pool<T>(
    buffers: &mut BufferPool,
    tensor: &TypedTensor<T>,
    k: i64,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Zero + Clone + PoolScalar + 'static,
{
    typed_triangular_mask_with_fill_pool(buffers, tensor, k, false, T::zero())
}

pub fn typed_triu<T: Copy + Zero + Clone>(
    tensor: &TypedTensor<T>,
    k: i64,
) -> crate::Result<TypedTensor<T>> {
    typed_triangular_mask(tensor, k, true)
}

pub(crate) fn typed_triu_with_pool<T>(
    buffers: &mut BufferPool,
    tensor: &TypedTensor<T>,
    k: i64,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Zero + Clone + PoolScalar + 'static,
{
    typed_triangular_mask_with_fill_pool(buffers, tensor, k, true, T::zero())
}

fn typed_triangular_mask<T: Copy + Zero + Clone>(
    tensor: &TypedTensor<T>,
    k: i64,
    upper: bool,
) -> crate::Result<TypedTensor<T>> {
    if tensor.shape().len() < 2 {
        return Err(crate::Error::RankMismatch {
            op: if upper { "triu" } else { "tril" },
            expected: 2,
            actual: tensor.shape().len(),
        });
    }

    let rows = tensor.shape()[0];
    let cols = tensor.shape()[1];
    if tensor.shape().contains(&0) {
        return Ok(tensor.clone());
    }

    let batch_count: usize = tensor.shape()[2..].iter().product();
    let block_size = rows * cols;
    let mut out = tensor.clone();
    let data = match &mut out.buffer {
        crate::Buffer::Host(data) => data,
        crate::Buffer::Backend(_) => {
            return Err(cpu_backend_buffer_error(if upper {
                "triu"
            } else {
                "tril"
            }))
        }
    };

    for batch_idx in 0..batch_count {
        let base = batch_idx * block_size;
        for col in 0..cols {
            let boundary = col as i128 - k as i128;
            for row in 0..rows {
                let row = row as i128;
                let keep = if upper {
                    row <= boundary
                } else {
                    row >= boundary
                };
                if !keep {
                    data[base + row as usize + col * rows] = T::zero();
                }
            }
        }
    }

    Ok(out)
}

fn typed_triangular_mask_with_fill_pool<T>(
    buffers: &mut BufferPool,
    tensor: &TypedTensor<T>,
    k: i64,
    upper: bool,
    fill: T,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + PoolScalar + 'static,
{
    if tensor.shape().len() < 2 {
        return Err(crate::Error::RankMismatch {
            op: if upper { "triu" } else { "tril" },
            expected: 2,
            actual: tensor.shape().len(),
        });
    }

    let rows = tensor.shape()[0];
    let cols = tensor.shape()[1];
    if tensor.shape().contains(&0) {
        return Ok(tensor.clone());
    }

    let batch_count: usize = tensor.shape()[2..].iter().product();
    let block_size = rows * cols;
    let mut out =
        clone_host_tensor_from_pool(buffers, if upper { "triu" } else { "tril" }, tensor)?;
    let data = match &mut out.buffer {
        crate::Buffer::Host(data) => data,
        crate::Buffer::Backend(_) => unreachable!("clone_host_tensor_from_pool returns host data"),
    };

    for batch_idx in 0..batch_count {
        let base = batch_idx * block_size;
        for col in 0..cols {
            let boundary = col as i128 - k as i128;
            for row in 0..rows {
                let row = row as i128;
                let keep = if upper {
                    row <= boundary
                } else {
                    row >= boundary
                };
                if !keep {
                    data[base + row as usize + col * rows] = fill;
                }
            }
        }
    }

    Ok(out)
}
