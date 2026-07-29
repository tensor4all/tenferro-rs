use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::TypedTensor;

fn checked_shape_product(op: &'static str, shape: &[usize]) -> crate::Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            crate::Error::invalid_argument(
                op,
                "shape",
                format!("shape product overflow for {shape:?}"),
            )
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

pub(crate) fn pooled_zeroed_tensor<T>(
    buffers: &mut BufferPool,
    shape: Vec<usize>,
) -> crate::Result<TypedTensor<T>>
where
    T: Clone + PoolScalar,
{
    let len = checked_shape_product("cpu_pooled_output", &shape)?;
    let data = T::pool_acquire_zeroed(buffers, len);
    TypedTensor::from_vec_col_major(shape, data)
}

#[cfg(test)]
mod tests {
    use super::{pooled_uninit_tensor, pooled_zeroed_tensor};
    use crate::buffer_pool::BufferPool;
    use tenferro_tensor::{ErrorKind, ValidationKind};

    #[test]
    fn pooled_uninit_tensor_preserves_shape_for_full_overwrite_callers() {
        let mut buffers = BufferPool::new();
        let output = pooled_uninit_tensor::<f64>(&mut buffers, vec![2, 3]).unwrap();

        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn pooled_uninit_tensor_reports_shape_product_overflow() {
        let mut buffers = BufferPool::new();
        let error = pooled_uninit_tensor::<f64>(&mut buffers, vec![usize::MAX, 2])
            .expect_err("an overflowing output shape must be rejected before allocation");

        assert_eq!(
            error.kind(),
            ErrorKind::Validation(ValidationKind::InvalidArgument)
        );
        assert!(error.to_string().contains("shape product overflow"));
    }

    #[test]
    fn pooled_zeroed_tensor_is_valid_before_kernel_handoff() {
        let mut buffers = BufferPool::new();
        let output = pooled_zeroed_tensor::<bool>(&mut buffers, vec![2, 3]).unwrap();

        assert_eq!(output.as_slice().unwrap(), &[false; 6]);
    }
}
