use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::TypedTensor;

fn checked_shape_product(op: &'static str, shape: &[usize]) -> crate::Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            crate::Error::backend_failure(op, format!("shape product overflow for {shape:?}"))
        })
}

pub(crate) fn pooled_uninit_tensor<T>(
    buffers: &mut BufferPool,
    shape: Vec<usize>,
) -> crate::Result<TypedTensor<T>>
where
    T: Clone + PoolScalar,
{
    let len = checked_shape_product("cpu_pooled_output", &shape)?;
    // SAFETY: callers use this helper only for pooled outputs that are fully overwritten.
    let data = unsafe { T::pool_acquire(buffers, len) };
    TypedTensor::from_vec_col_major(shape, data)
}
