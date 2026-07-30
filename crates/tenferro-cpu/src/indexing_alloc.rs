use crate::buffer_pool::{BufferPool, PoolScalar};
#[cfg(test)]
pub(crate) use crate::PooledUninitOutput;
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

#[cfg(test)]
mod tests {
    use super::{pooled_uninit_tensor, PooledUninitOutput};
    use crate::buffer_pool::{BufferPool, PoolScalar};
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
    fn pooled_uninit_output_reuses_stale_bool_storage_without_reading_it() {
        let mut buffers = BufferPool::new();
        <bool as PoolScalar>::pool_release(&mut buffers, vec![true; 6]);
        let mut output = PooledUninitOutput::<bool>::new(&mut buffers, vec![2, 3]).unwrap();
        output.as_uninit_slice_mut().iter_mut().for_each(|value| {
            value.write(false);
        });
        let output = unsafe { output.assume_init() }.unwrap();

        assert_eq!(output.as_slice().unwrap(), &[false; 6]);
    }

    #[test]
    fn pooled_uninit_output_error_drop_clears_in_flight_accounting() {
        let mut buffers = BufferPool::new();
        <bool as PoolScalar>::pool_release(&mut buffers, Vec::with_capacity(6));

        let output = PooledUninitOutput::<bool>::new(&mut buffers, vec![2, 3]).unwrap();
        drop(output);
        buffers.clear_in_flight_retained();

        assert_eq!(buffers.stats().buffers, 0);
        assert_eq!(buffers.stats().capacity_bytes, 0);
    }
}
