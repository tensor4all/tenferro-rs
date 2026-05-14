use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::TypedTensor;

pub(crate) fn pooled_uninit_tensor<T>(buffers: &mut BufferPool, shape: Vec<usize>) -> TypedTensor<T>
where
    T: Clone + PoolScalar,
{
    let len = shape.iter().product();
    // SAFETY: callers use this helper only for outputs that are fully written
    // before any read.
    let data = unsafe { T::pool_acquire(buffers, len) };
    TypedTensor::from_vec(shape, data)
}
