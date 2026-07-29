use std::mem::{ManuallyDrop, MaybeUninit};

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

pub(crate) struct PooledUninitOutput<T> {
    shape: Vec<usize>,
    data: Vec<MaybeUninit<T>>,
}

impl<T> PooledUninitOutput<T>
where
    T: Clone + PoolScalar,
{
    pub(crate) fn new(buffers: &mut BufferPool, shape: Vec<usize>) -> crate::Result<Self> {
        let len = checked_shape_product("cpu_pooled_output", &shape)?;
        // INVARIANT: this owner exposes only MaybeUninit storage until a
        // full-overwrite caller proves initialization through assume_init.
        Ok(Self {
            shape,
            data: T::pool_acquire_uninit(buffers, len),
        })
    }

    pub(crate) fn as_uninit_bytes_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        // INVARIANT: MaybeUninit<T> may be viewed as size_of::<T>() uninitialized
        // bytes; no initialized T reference is constructed at this boundary.
        // SAFETY: MaybeUninit<T> and its byte range share the same allocation,
        // and the returned lifetime remains bounded by this exclusive borrow.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr().cast::<MaybeUninit<u8>>(),
                std::mem::size_of_val(self.data.as_slice()),
            )
        }
    }

    /// Convert the owner after a full-overwrite kernel succeeds.
    ///
    /// # Safety
    ///
    /// Every element in `self.data` must contain a valid initialized `T`.
    pub(crate) unsafe fn assume_init(self) -> crate::Result<TypedTensor<T>> {
        let Self { shape, data } = self;
        let mut data = ManuallyDrop::new(data);
        // INVARIANT: the caller's successful full-overwrite replay initialized
        // every element, and MaybeUninit<T> has the same allocation layout as T.
        // SAFETY: the caller guarantees every element is initialized, while
        // ManuallyDrop transfers the unchanged allocation exactly once.
        let initialized = unsafe {
            Vec::from_raw_parts(data.as_mut_ptr().cast::<T>(), data.len(), data.capacity())
        };
        TypedTensor::from_vec_col_major(shape, initialized)
    }
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
        output.data.iter_mut().for_each(|value| {
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
