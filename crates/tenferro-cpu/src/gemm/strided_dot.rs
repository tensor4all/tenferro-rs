use num_traits::{One, Zero};
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use tenferro_tensor::{Buffer, DotGeneralConfig as TensorDotGeneralConfig, TypedTensor};

use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::{default_placement, Error};

use super::TypedTensorRead;

#[cfg(test)]
static DISPATCH_COUNT: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
pub(super) fn test_dispatch_count() -> usize {
    DISPATCH_COUNT.load(Ordering::SeqCst)
}

fn map_strided_error(err: impl std::fmt::Display) -> Error {
    Error::backend_failure("dot_general", err)
}

fn to_strided_config(config: &TensorDotGeneralConfig) -> strided_einsum2::DotGeneralConfig<'_> {
    strided_einsum2::DotGeneralConfig {
        lhs_contracting_dims: config.lhs_contracting_dims.as_slice(),
        rhs_contracting_dims: config.rhs_contracting_dims.as_slice(),
        lhs_batch_dims: config.lhs_batch_dims.as_slice(),
        rhs_batch_dims: config.rhs_batch_dims.as_slice(),
    }
}

fn checked_product(shape: &[usize]) -> crate::Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| Error::backend_failure("dot_general", "output element count overflow"))
}

fn as_strided_view<'a, R, T>(read: &'a R) -> crate::Result<strided_einsum2::StridedView<'a, T>>
where
    R: TypedTensorRead<T>,
    T: 'static,
{
    let Some(data) = read.host_data_opt() else {
        return Err(Error::backend_failure(
            "dot_general",
            "CPU dot_general requires host-backed inputs",
        ));
    };
    strided_einsum2::StridedView::new(data, read.shape(), read.strides().as_slice(), read.offset())
        .map_err(map_strided_error)
}

pub(crate) fn dot_general_strided_with_backend<L, R, T, B>(
    buffers: &mut BufferPool,
    lhs: &L,
    rhs: &R,
    config: &TensorDotGeneralConfig,
) -> crate::Result<TypedTensor<T>>
where
    L: TypedTensorRead<T>,
    R: TypedTensorRead<T>,
    T: PoolScalar + Copy + Clone + Zero + One + PartialEq + strided_einsum2::ScalarBase + 'static,
    B: strided_einsum2::Backend<T>,
{
    #[cfg(test)]
    DISPATCH_COUNT.fetch_add(1, Ordering::SeqCst);

    super::validate_dot_general(lhs, rhs, config)?;

    let lhs_view = as_strided_view(lhs)?;
    let rhs_view = as_strided_view(rhs)?;

    let strided_config = to_strided_config(config);
    let out_shape = strided_config
        .expected_output_shape(lhs.shape(), rhs.shape())
        .map_err(map_strided_error)?;
    let out_n = checked_product(&out_shape)?;

    // SAFETY: beta=0 strided dot fully overwrites outputs or explicitly zero-fills k=0.
    let mut out_data = unsafe { T::pool_acquire(buffers, out_n) };
    let out_strides = strided_einsum2::col_major_strides(&out_shape);
    let out_view = strided_einsum2::StridedViewMut::new(&mut out_data, &out_shape, &out_strides, 0)
        .map_err(map_strided_error)?;

    strided_einsum2::dot_general_with_backend_into::<T, B>(
        out_view,
        &lhs_view,
        &rhs_view,
        &strided_config,
        T::one(),
        T::zero(),
    )
    .map_err(map_strided_error)?;

    Ok(TypedTensor::from_buffer_col_major(
        out_shape,
        Buffer::Host(out_data),
        default_placement(),
    ))
}
