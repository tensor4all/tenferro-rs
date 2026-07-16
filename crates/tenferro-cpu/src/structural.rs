use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use std::sync::Arc;
use strided_kernel::{
    col_major_strides, copy_into, map_into, Identity, StridedView, StridedViewMut,
};

use crate::{
    buffer_pool::{BufferPool, PoolScalar},
    flat_to_multi,
};
use tenferro_tensor::{
    DType, MemoryKind, Placement, Tensor, TensorRank, TensorRead, TensorView, TypedTensor,
    TypedTensorView, TypedTensorViewMut,
};

#[cfg(test)]
use super::typed_array_uninit;
use super::{
    cpu_backend_buffer_error, tensor_from_array, typed_array_uninit_from_pool, typed_host_data,
    typed_view, typed_view_from_view,
};

#[cfg(test)]
fn with_test_pool<T>(f: impl FnOnce(&mut BufferPool) -> T) -> T {
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

fn checked_shape_product(
    op: &'static str,
    role: &'static str,
    shape: &[usize],
) -> crate::Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op,
                message: format!("{role} element count overflows usize"),
            })
    })
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

macro_rules! dispatch_tensor_view_unary_result {
    ($input:expr, |$view:ident| $body:expr) => {
        match $input {
            TensorView::F32($view) => Ok(Tensor::F32($body?)),
            TensorView::F64($view) => Ok(Tensor::F64($body?)),
            TensorView::I32($view) => Ok(Tensor::I32($body?)),
            TensorView::I64($view) => Ok(Tensor::I64($body?)),
            TensorView::Bool($view) => Ok(Tensor::Bool($body?)),
            TensorView::C32($view) => Ok(Tensor::C32($body?)),
            TensorView::C64($view) => Ok(Tensor::C64($body?)),
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
    match tensor.buffer() {
        crate::Buffer::Host(data) => {
            let strides = col_major_strides(tensor.shape());
            StridedView::new(data.as_slice(), tensor.shape(), &strides, 0)
                .map_err(|err| crate::Error::backend_failure(op, err))
        }
        crate::Buffer::Backend(_) => Err(cpu_backend_buffer_error(op)),
    }
}

fn copy_view_to_array<T: Copy + Clone + Send + Sync + 'static>(
    op: &'static str,
    mut out: strided_kernel::StridedArray<T>,
    src: &StridedView<'_, T>,
    placement: &tenferro_tensor::Placement,
) -> crate::Result<TypedTensor<T>> {
    copy_into(&mut out.view_mut(), src).map_err(|err| crate::Error::backend_failure(op, err))?;
    tensor_from_array_with_placement(op, out, placement)
}

fn tensor_from_array_with_placement<T: 'static>(
    op: &'static str,
    out: strided_kernel::StridedArray<T>,
    placement: &tenferro_tensor::Placement,
) -> crate::Result<TypedTensor<T>> {
    let shape = out.dims().to_vec();
    TypedTensor::from_buffer_col_major(
        shape,
        crate::Buffer::Host(out.into_data()),
        placement.clone(),
    )
    .map_err(|err| crate::Error::backend_failure(op, err))
}

pub(crate) fn typed_materialize_view_with_pool<T, R>(
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T, R>,
    op: &'static str,
) -> crate::Result<TypedTensor<T, R>>
where
    T: Copy + Clone + PoolScalar + 'static,
    R: TensorRank,
{
    if view.backend_buffer().is_some() {
        return Err(cpu_backend_buffer_error(op));
    }
    validate_cpu_host_placement(op, "source", view.placement())?;
    let src: StridedView<'_, T, Identity> = StridedView::new(
        view.host_storage()?,
        view.shape(),
        view.strides(),
        view.offset(),
    )
    .map_err(|err| crate::Error::backend_failure(op, err))?;
    // INVARIANT: validated equal-shaped copy_into overwrites every logical output element.
    // SAFETY: copy_into overwrites every logical output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, view.shape()) }?;
    copy_into(&mut out.view_mut(), &src).map_err(|err| crate::Error::backend_failure(op, err))?;
    let shape = R::shape_from_vec(view.shape().to_vec().into())
        .map_err(|err| crate::Error::backend_failure(op, err))?;
    TypedTensor::from_buffer_col_major(
        shape,
        crate::Buffer::Host(out.into_data()),
        view.placement().clone(),
    )
}

pub(crate) fn typed_copy_view_into<T, R>(
    src: &TypedTensorView<'_, T, R>,
    dst: &mut TypedTensorViewMut<'_, T, R>,
    op: &'static str,
) -> crate::Result<()>
where
    T: Copy + Send + Sync + 'static,
    R: TensorRank,
{
    if src.shape() != dst.shape() {
        return Err(crate::Error::ShapeMismatch {
            op,
            lhs: src.shape().to_vec(),
            rhs: dst.shape().to_vec(),
        });
    }
    if let (Some(src_buffer), Some(dst_buffer)) = (src.backend_buffer(), dst.backend_buffer()) {
        if Arc::ptr_eq(src_buffer, dst_buffer) {
            return Err(crate::Error::InvalidConfig {
                op,
                message: "CPU copy source and destination allocations must not alias".into(),
            });
        }
    }
    if src.backend_buffer().is_some() || dst.backend_buffer().is_some() {
        return Err(cpu_backend_buffer_error(op));
    }
    validate_cpu_host_placement(op, "source", src.placement())?;
    validate_cpu_host_placement(op, "destination", dst.placement())?;

    let src_view: StridedView<'_, T, Identity> = StridedView::new(
        src.host_storage()?,
        src.shape(),
        src.strides(),
        src.offset(),
    )
    .map_err(|err| crate::Error::backend_failure(op, err))?;
    let dst_shape = dst.shape().to_vec();
    let dst_strides = dst.strides().to_vec();
    let dst_offset = dst.offset();
    let mut dst_view = StridedViewMut::new(
        dst.host_storage_mut()?,
        &dst_shape,
        &dst_strides,
        dst_offset,
    )
    .map_err(|err| crate::Error::backend_failure(op, err))?;
    copy_into(&mut dst_view, &src_view).map_err(|err| crate::Error::backend_failure(op, err))
}

pub(crate) fn validate_cpu_host_placement(
    op: &'static str,
    role: &'static str,
    placement: &Placement,
) -> crate::Result<()> {
    if matches!(
        placement.memory_kind,
        MemoryKind::PinnedHost | MemoryKind::UnpinnedHost
    ) {
        return Ok(());
    }
    Err(crate::Error::backend_failure(
        op,
        format!(
            "CPU backend requires {role} host placement for {op}, got {:?}",
            placement.memory_kind
        ),
    ))
}

fn zeroed_tensor_from_pool<T>(
    buffers: &mut BufferPool,
    op: &'static str,
    shape: Vec<usize>,
) -> crate::Result<TypedTensor<T>>
where
    T: Zero + Clone + PoolScalar + 'static,
{
    filled_tensor_from_pool(buffers, op, shape, T::zero())
}

fn filled_tensor_from_pool<T>(
    buffers: &mut BufferPool,
    op: &'static str,
    shape: Vec<usize>,
    fill: T,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + PoolScalar + 'static,
{
    let len = checked_shape_product(op, "output shape", &shape)?;
    // SAFETY: every pooled element is initialized with `fill` before returning.
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
    let input = match tensor.buffer() {
        crate::Buffer::Host(data) => data.as_slice(),
        crate::Buffer::Backend(_) => return Err(cpu_backend_buffer_error(op)),
    };
    // SAFETY: copy_from_slice initializes every pooled element before returning.
    let mut data = unsafe { T::pool_acquire(buffers, input.len()) };
    data.copy_from_slice(input);
    TypedTensor::from_buffer_col_major(
        tensor.shape().to_vec(),
        crate::Buffer::Host(data),
        tensor.placement().clone(),
    )
}

#[cfg(test)]
pub(crate) fn transpose(input: &Tensor, perm: &[usize]) -> crate::Result<Tensor> {
    with_test_pool(|buffers| transpose_with_pool(buffers, input, perm))
}

pub(crate) fn transpose_with_pool(
    buffers: &mut BufferPool,
    input: &Tensor,
    perm: &[usize],
) -> crate::Result<Tensor> {
    dispatch_tensor_unary_result!(input, |t| typed_transpose_with_pool(buffers, t, perm))
}

pub(crate) fn transpose_read_with_pool(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
    perm: &[usize],
) -> crate::Result<Tensor> {
    match input {
        TensorRead::Tensor(input) => transpose_with_pool(buffers, input, perm),
        TensorRead::View(input) => dispatch_tensor_view_unary_result!(input, |view| {
            typed_transpose_view_with_pool(buffers, &view, perm)
        }),
    }
}

pub(crate) fn reshape(input: &Tensor, shape: &[usize]) -> crate::Result<Tensor> {
    dispatch_tensor_unary_result!(input, |t| typed_reshape(t, shape))
}

pub(crate) fn reshape_read_with_pool(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
    shape: &[usize],
) -> crate::Result<Tensor> {
    match input {
        TensorRead::Tensor(input) => reshape(input, shape),
        TensorRead::View(input) => dispatch_tensor_view_unary_result!(input, |view| {
            typed_reshape_view_with_pool(buffers, &view, shape)
        }),
    }
}

#[cfg(test)]
pub(crate) fn broadcast_in_dim(
    input: &Tensor,
    shape: &[usize],
    dims: &[usize],
) -> crate::Result<Tensor> {
    with_test_pool(|buffers| broadcast_in_dim_with_pool(buffers, input, shape, dims))
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

pub(crate) fn broadcast_in_dim_read_with_pool(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
    shape: &[usize],
    dims: &[usize],
) -> crate::Result<Tensor> {
    match input {
        TensorRead::Tensor(input) => broadcast_in_dim_with_pool(buffers, input, shape, dims),
        TensorRead::View(input) => dispatch_tensor_view_unary_result!(input, |view| {
            typed_broadcast_in_dim_view_with_pool(buffers, &view, shape, dims)
        }),
    }
}

/// Convert a tensor to another dtype using checked dtype conversion.
///
/// Use `TensorStructural::cast` when an explicit lossy dtype projection is
/// intended.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBackend;
/// use tenferro_tensor::{DType, Tensor, TensorStructural};
///
/// let mut backend = CpuBackend::new();
/// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
/// let y = backend.convert(&x, DType::F64).unwrap();
/// assert_eq!(y.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
/// ```
///
/// # Errors
///
/// Returns an error when the requested conversion is outside tenferro's checked
/// dtype-promotion lattice.
#[cfg(test)]
pub(crate) fn convert(input: &Tensor, to: DType) -> crate::Result<Tensor> {
    with_test_pool(|buffers| convert_with_pool(buffers, input, to))
}

#[cfg(test)]
pub(crate) fn convert_with_pool(
    buffers: &mut BufferPool,
    input: &Tensor,
    to: DType,
) -> crate::Result<Tensor> {
    tenferro_tensor::validate::validate_convert_dtype("convert", input.dtype(), to)?;
    cast_with_pool(buffers, input, to)
}

pub(crate) fn cast_with_pool(
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
        (Tensor::F32(t), DType::I32) => {
            validate_real_values_cast_to_i32(t, |x| x as f64)?;
            converted!(I32, t, |x| x as i32)
        }
        (Tensor::F32(t), DType::I64) => {
            validate_real_values_cast_to_i64(t, |x| x as f64)?;
            converted!(I64, t, |x| x as i64)
        }
        (Tensor::F32(t), DType::Bool) => converted!(Bool, t, |x| x != 0.0),
        (Tensor::F32(t), DType::C32) => converted!(C32, t, |x| Complex32::new(x, 0.0)),
        (Tensor::F32(t), DType::C64) => {
            converted!(C64, t, |x| Complex64::new(x as f64, 0.0))
        }
        (Tensor::F64(t), DType::F32) => converted!(F32, t, |x| x as f32),
        (Tensor::F64(t), DType::F64) => Ok(Tensor::F64(t.clone())),
        (Tensor::F64(t), DType::I32) => {
            validate_real_values_cast_to_i32(t, |x| x)?;
            converted!(I32, t, |x| x as i32)
        }
        (Tensor::F64(t), DType::I64) => {
            validate_real_values_cast_to_i64(t, |x| x)?;
            converted!(I64, t, |x| x as i64)
        }
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
        (Tensor::C32(t), DType::I32) => {
            validate_real_values_cast_to_i32(t, |z| z.re as f64)?;
            converted!(I32, t, |z| z.re as i32)
        }
        (Tensor::C32(t), DType::I64) => {
            validate_real_values_cast_to_i64(t, |z| z.re as f64)?;
            converted!(I64, t, |z| z.re as i64)
        }
        (Tensor::C32(t), DType::Bool) => converted!(Bool, t, |z| z.re != 0.0 || z.im != 0.0),
        (Tensor::C32(t), DType::C32) => Ok(Tensor::C32(t.clone())),
        (Tensor::C32(t), DType::C64) => {
            converted!(C64, t, |z| Complex64::new(z.re as f64, z.im as f64))
        }
        (Tensor::C64(t), DType::F32) => converted!(F32, t, |z| z.re as f32),
        (Tensor::C64(t), DType::F64) => converted!(F64, t, |z| z.re),
        (Tensor::C64(t), DType::I32) => {
            validate_real_values_cast_to_i32(t, |z| z.re)?;
            converted!(I32, t, |z| z.re as i32)
        }
        (Tensor::C64(t), DType::I64) => {
            validate_real_values_cast_to_i64(t, |z| z.re)?;
            converted!(I64, t, |z| z.re as i64)
        }
        (Tensor::C64(t), DType::Bool) => converted!(Bool, t, |z| z.re != 0.0 || z.im != 0.0),
        (Tensor::C64(t), DType::C32) => {
            converted!(C32, t, |z| Complex32::new(z.re as f32, z.im as f32))
        }
        (Tensor::C64(t), DType::C64) => Ok(Tensor::C64(t.clone())),
    }
}

fn validate_real_values_cast_to_i32<S: Copy>(
    tensor: &TypedTensor<S>,
    real: impl Fn(S) -> f64,
) -> crate::Result<()> {
    for &value in typed_host_data("cast", tensor)? {
        validate_real_cast_to_i32(real(value))?;
    }
    Ok(())
}

fn validate_real_values_cast_to_i64<S: Copy>(
    tensor: &TypedTensor<S>,
    real: impl Fn(S) -> f64,
) -> crate::Result<()> {
    for &value in typed_host_data("cast", tensor)? {
        validate_real_cast_to_i64(real(value))?;
    }
    Ok(())
}

fn validate_real_cast_to_i32(value: f64) -> crate::Result<()> {
    if !value.is_finite() {
        return Err(invalid_cast_value(format!(
            "real value must be finite when casting to i32, got {value}"
        )));
    }
    if value < i32::MIN as f64 || value > i32::MAX as f64 {
        return Err(invalid_cast_value(format!(
            "real value {value} is out of i32 range"
        )));
    }
    Ok(())
}

fn validate_real_cast_to_i64(value: f64) -> crate::Result<()> {
    const I64_MIN_F64: f64 = -9_223_372_036_854_775_808.0;
    const I64_MAX_EXCLUSIVE_F64: f64 = 9_223_372_036_854_775_808.0;

    if !value.is_finite() {
        return Err(invalid_cast_value(format!(
            "real value must be finite when casting to i64, got {value}"
        )));
    }
    if !(I64_MIN_F64..I64_MAX_EXCLUSIVE_F64).contains(&value) {
        return Err(invalid_cast_value(format!(
            "real value {value} is out of i64 range"
        )));
    }
    Ok(())
}

fn invalid_cast_value(message: String) -> crate::Error {
    crate::Error::InvalidConfig {
        op: "cast",
        message,
    }
}

#[cfg(test)]
pub(crate) fn extract_diagonal(
    input: &Tensor,
    axis_a: usize,
    axis_b: usize,
) -> crate::Result<Tensor> {
    with_test_pool(|buffers| extract_diagonal_with_pool(buffers, input, axis_a, axis_b))
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

#[cfg(test)]
pub(crate) fn embed_diagonal(
    input: &Tensor,
    axis_a: usize,
    axis_b: usize,
) -> crate::Result<Tensor> {
    with_test_pool(|buffers| embed_diagonal_with_pool(buffers, input, axis_a, axis_b))
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
                filled_tensor_from_pool(buffers, "embed_diagonal", shape, false)
            })
    )
}

#[cfg(test)]
pub(crate) fn tril(input: &Tensor, k: i64) -> crate::Result<Tensor> {
    with_test_pool(|buffers| tril_with_pool(buffers, input, k))
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

#[cfg(test)]
pub(crate) fn triu(input: &Tensor, k: i64) -> crate::Result<Tensor> {
    with_test_pool(|buffers| triu_with_pool(buffers, input, k))
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

#[cfg(test)]
pub(crate) fn typed_transpose<T: Copy + Clone + Send + Sync + 'static>(
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
    copy_view_to_array("transpose", out, &permuted, tensor.placement())
}

fn typed_transpose_view_impl<T, R>(
    view: &TypedTensorView<'_, T, R>,
    perm: &[usize],
    make_out: impl FnOnce(&[usize]) -> crate::Result<strided_kernel::StridedArray<T>>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Send + Sync + 'static,
    R: TensorRank,
{
    validate_permutation("transpose", perm, view.shape().len())?;
    let src = typed_view_from_view("transpose", view)?;
    let permuted = src
        .permute(perm)
        .map_err(|err| crate::Error::backend_failure("transpose", err))?;
    checked_shape_product("transpose", "output shape", permuted.dims())?;
    // SAFETY: copy_into overwrites every output element.
    let out = make_out(permuted.dims())?;
    copy_view_to_array("transpose", out, &permuted, view.placement())
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
        // INVARIANT: validated transpose copy overwrites every pooled output element.
        // SAFETY: transpose materialization copies every output element before returning.
        typed_array_uninit_from_pool(buffers, shape)
    })
}

pub fn typed_reshape<T: Clone + 'static>(
    tensor: &TypedTensor<T>,
    shape: &[usize],
) -> crate::Result<TypedTensor<T>> {
    let old_n = checked_shape_product("reshape", "input shape", tensor.shape())?;
    let new_n = checked_shape_product("reshape", "output shape", shape)?;
    if old_n != new_n {
        return Err(crate::Error::ShapeMismatch {
            op: "reshape",
            lhs: tensor.shape().to_vec(),
            rhs: shape.to_vec(),
        });
    }
    TypedTensor::from_buffer_col_major(
        shape.to_vec(),
        tensor.buffer().clone(),
        tensor.placement().clone(),
    )
}

pub(crate) fn typed_reshape_view_with_pool<T, R>(
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T, R>,
    shape: &[usize],
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + PoolScalar + 'static,
    R: TensorRank,
{
    let old_n = checked_shape_product("reshape", "input shape", view.shape())?;
    let new_n = checked_shape_product("reshape", "output shape", shape)?;
    if old_n != new_n {
        return Err(crate::Error::ShapeMismatch {
            op: "reshape",
            lhs: view.shape().to_vec(),
            rhs: shape.to_vec(),
        });
    }

    let src = typed_view_from_view("reshape", view)?;
    // INVARIANT: equal element counts let the source-shaped copy target fully overwrite the
    // pooled compact output before the same storage is attached to `shape`.
    // SAFETY: copy_into overwrites every pooled output element before returning.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, shape) }?;
    let copy_strides = col_major_strides(view.shape());
    let mut copy_target = StridedViewMut::new(out.data_mut(), view.shape(), &copy_strides, 0)
        .map_err(|err| crate::Error::backend_failure("reshape", err))?;
    copy_into(&mut copy_target, &src)
        .map_err(|err| crate::Error::backend_failure("reshape", err))?;
    TypedTensor::from_buffer_col_major(
        shape.to_vec(),
        crate::Buffer::Host(out.into_data()),
        view.placement().clone(),
    )
}

#[cfg(test)]
pub(crate) fn typed_broadcast_in_dim<T: Copy + Clone + Send + Sync + 'static>(
    tensor: &TypedTensor<T>,
    shape: &[usize],
    dims: &[usize],
) -> crate::Result<TypedTensor<T>> {
    typed_broadcast_in_dim_view_impl(&tensor.as_view(), shape, dims, |shape| unsafe {
        // SAFETY: broadcast materialization writes every output element before returning.
        Ok(typed_array_uninit(shape))
    })
}

pub(crate) fn typed_broadcast_in_dim_with_pool<T>(
    buffers: &mut BufferPool,
    tensor: &TypedTensor<T>,
    shape: &[usize],
    dims: &[usize],
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + PoolScalar + 'static,
{
    typed_broadcast_in_dim_view_with_pool(buffers, &tensor.as_view(), shape, dims)
}

pub(crate) fn typed_broadcast_in_dim_view_with_pool<T, R>(
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T, R>,
    shape: &[usize],
    dims: &[usize],
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + PoolScalar + 'static,
    R: TensorRank,
{
    typed_broadcast_in_dim_view_impl(view, shape, dims, |shape| unsafe {
        // INVARIANT: validated broadcast copy overwrites every pooled output element.
        // SAFETY: broadcast materialization writes every output element before returning.
        typed_array_uninit_from_pool(buffers, shape)
    })
}

fn typed_broadcast_in_dim_view_impl<T, R>(
    view: &TypedTensorView<'_, T, R>,
    shape: &[usize],
    dims: &[usize],
    make_out: impl FnOnce(&[usize]) -> crate::Result<strided_kernel::StridedArray<T>>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Send + Sync + 'static,
    R: TensorRank,
{
    validate_rank("broadcast_in_dim", view.shape().len(), dims.len())?;
    let mut seen = vec![false; shape.len()];
    let mut base_dims = vec![1usize; shape.len()];
    let mut base_strides = vec![0isize; shape.len()];
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
        let source_dim = view.shape()[src_axis];
        let target_dim = shape[dst_axis];
        if source_dim != target_dim && source_dim != 1 {
            return Err(crate::Error::ShapeMismatch {
                op: "broadcast_in_dim",
                lhs: view.shape().to_vec(),
                rhs: shape.to_vec(),
            });
        }
        base_dims[dst_axis] = source_dim;
        base_strides[dst_axis] = view.strides()[src_axis];
    }
    if view.backend_buffer().is_some() {
        return Err(cpu_backend_buffer_error("broadcast_in_dim"));
    }
    let base: StridedView<'_, T, Identity> = StridedView::new(
        view.host_storage()?,
        &base_dims,
        &base_strides,
        view.offset(),
    )
    .map_err(|err| crate::Error::backend_failure("broadcast_in_dim", err))?;
    let broadcast: StridedView<'_, T, Identity> = base
        .broadcast(shape)
        .map_err(|err| crate::Error::backend_failure("broadcast_in_dim", err))?;
    checked_shape_product("broadcast_in_dim", "output shape", shape)?;
    // SAFETY: copy_into overwrites every output element.
    let mut out = make_out(shape)?;
    copy_into(&mut out.view_mut(), &broadcast)
        .map_err(|err| crate::Error::backend_failure("broadcast_in_dim", err))?;
    tensor_from_array_with_placement("broadcast_in_dim", out, view.placement())
}

fn typed_convert_with_pool<S, T>(
    buffers: &mut BufferPool,
    tensor: &TypedTensor<S>,
    f: impl Fn(S) -> T + Sync,
) -> crate::Result<TypedTensor<T>>
where
    S: Copy + Send + Sync,
    T: Copy + Clone + PoolScalar,
{
    // SAFETY: map_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, tensor.shape()) }?;
    map_into(&mut out.view_mut(), &typed_view("convert", tensor)?, f)
        .map_err(|err| crate::Error::backend_failure("convert", err))?;
    Ok(tensor_from_array(out))
}

#[cfg(test)]
pub(crate) fn typed_extract_diagonal<T: Copy + Clone + Send + Sync>(
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
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, diag.dims()) }?;
    copy_into(&mut out.view_mut(), &diag)
        .map_err(|err| crate::Error::backend_failure("extract_diagonal", err))?;
    Ok(tensor_from_array(out))
}

#[cfg(test)]
pub(crate) fn typed_embed_diagonal<T: Copy + Zero + Clone>(
    tensor: &TypedTensor<T>,
    axis_a: usize,
    axis_b: usize,
) -> crate::Result<TypedTensor<T>> {
    typed_embed_diagonal_impl(tensor, axis_a, axis_b, TypedTensor::zeros)
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
        zeroed_tensor_from_pool(buffers, "embed_diagonal", shape)
    })
}

fn typed_embed_diagonal_impl<T>(
    tensor: &TypedTensor<T>,
    axis_a: usize,
    axis_b: usize,
    make_zeroed: impl FnOnce(Vec<usize>) -> crate::Result<TypedTensor<T>>,
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
    let mut out = make_zeroed(out_shape)?;

    let in_rank = tensor.shape().len();
    let out_rank = out.shape().len();
    let mut in_idx = vec![0usize; in_rank];
    let mut out_idx = vec![0usize; out_rank];

    let input_data = match tensor.buffer() {
        crate::Buffer::Host(data) => data.as_slice(),
        crate::Buffer::Backend(_) => return Err(cpu_backend_buffer_error("embed_diagonal")),
    };

    // Intentionally sequential: embed_diagonal writes a sparse diagonal subset
    // into a zeroed output and has no current strided-kernel parallel primitive.
    for (flat, value) in input_data
        .iter()
        .copied()
        .enumerate()
        .take(tensor.n_elements())
    {
        flat_to_multi(flat, tensor.shape(), &mut in_idx);
        let diag_val = in_idx[axis_a];
        let mut src_axis = 0usize;
        for (out_axis, out_slot) in out_idx.iter_mut().enumerate().take(out_rank) {
            if out_axis == axis_b {
                *out_slot = diag_val;
            } else {
                *out_slot = in_idx[src_axis];
                src_axis += 1;
            }
        }
        *out.get_mut(&out_idx)? = value;
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) fn typed_tril<T: Copy + Zero + Clone>(
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

#[cfg(test)]
pub(crate) fn typed_triu<T: Copy + Zero + Clone>(
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

#[cfg(test)]
fn typed_triangular_mask<T: Copy + Zero + Clone>(
    tensor: &TypedTensor<T>,
    k: i64,
    upper: bool,
) -> crate::Result<TypedTensor<T>> {
    let op = if upper { "triu" } else { "tril" };
    if tensor.shape().len() < 2 {
        return Err(crate::Error::RankMismatch {
            op,
            expected: 2,
            actual: tensor.shape().len(),
        });
    }

    let rows = tensor.shape()[0];
    let cols = tensor.shape()[1];
    if tensor.shape().contains(&0) {
        return Ok(tensor.clone());
    }

    let (batch_count, block_size) = checked_triangular_extent(op, tensor.shape(), rows, cols)?;
    let mut out = tensor.clone();
    let data = out.host_data_mut()?;

    // Intentionally sequential: triangular masks are index-dependent in the
    // innermost matrix plane and remain a dedicated CPU-kernel exception.
    for batch_idx in 0..batch_count {
        for col in 0..cols {
            let boundary = col as i128 - k as i128;
            for row in 0..rows {
                let row_idx = row;
                let row = row_idx as i128;
                let keep = if upper {
                    row <= boundary
                } else {
                    row >= boundary
                };
                if !keep {
                    let offset =
                        checked_triangular_offset(op, batch_idx, block_size, col, rows, row_idx)?;
                    data[offset] = T::zero();
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
    let op = if upper { "triu" } else { "tril" };
    if tensor.shape().len() < 2 {
        return Err(crate::Error::RankMismatch {
            op,
            expected: 2,
            actual: tensor.shape().len(),
        });
    }

    let rows = tensor.shape()[0];
    let cols = tensor.shape()[1];
    if tensor.shape().contains(&0) {
        return Ok(tensor.clone());
    }

    let (batch_count, block_size) = checked_triangular_extent(op, tensor.shape(), rows, cols)?;
    let mut out = clone_host_tensor_from_pool(buffers, op, tensor)?;
    let data = out.host_data_mut()?;

    // Intentionally sequential: triangular masks are index-dependent in the
    // innermost matrix plane and remain a dedicated CPU-kernel exception.
    for batch_idx in 0..batch_count {
        for col in 0..cols {
            let boundary = col as i128 - k as i128;
            for row in 0..rows {
                let row_idx = row;
                let row = row_idx as i128;
                let keep = if upper {
                    row <= boundary
                } else {
                    row >= boundary
                };
                if !keep {
                    let offset =
                        checked_triangular_offset(op, batch_idx, block_size, col, rows, row_idx)?;
                    data[offset] = fill;
                }
            }
        }
    }

    Ok(out)
}

fn checked_triangular_extent(
    op: &'static str,
    shape: &[usize],
    rows: usize,
    cols: usize,
) -> crate::Result<(usize, usize)> {
    let batch_count = shape[2..].iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op,
                message: format!("batch extent overflows usize: {acc} * {dim}"),
            })
    })?;
    let block_size = rows
        .checked_mul(cols)
        .ok_or_else(|| crate::Error::InvalidConfig {
            op,
            message: format!("matrix block size overflows usize: {rows} * {cols}"),
        })?;
    Ok((batch_count, block_size))
}

fn checked_triangular_offset(
    op: &'static str,
    batch_idx: usize,
    block_size: usize,
    col: usize,
    rows: usize,
    row_idx: usize,
) -> crate::Result<usize> {
    let base = batch_idx
        .checked_mul(block_size)
        .ok_or_else(|| crate::Error::InvalidConfig {
            op,
            message: format!("batch offset overflows usize: {batch_idx} * {block_size}"),
        })?;
    let col_offset = col
        .checked_mul(rows)
        .ok_or_else(|| crate::Error::InvalidConfig {
            op,
            message: format!("column offset overflows usize: {col} * {rows}"),
        })?;
    base.checked_add(col_offset)
        .and_then(|offset| offset.checked_add(row_idx))
        .ok_or_else(|| crate::Error::InvalidConfig {
            op,
            message: "triangular mask offset overflows usize".to_string(),
        })
}
