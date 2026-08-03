use std::alloc::Layout;
use std::mem::{ManuallyDrop, MaybeUninit};

use strided_kernel::StridedViewMut;
use tenferro_tensor::{validate::checked_shape_product, TensorRank, TensorScalar, TypedTensor};

use crate::buffer_pool::{BufferPool, PoolScalar, UninitCheckoutToken};
use crate::{Error, Result};

fn checked_compact_strides(shape: &[usize]) -> Result<Vec<isize>> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1isize;
    for &dim in shape {
        strides.push(stride);
        let dim = isize::try_from(dim).map_err(|_| {
            Error::invalid_argument("pooled_uninit_output", "shape", "dimension exceeds isize")
        })?;
        stride = stride.checked_mul(dim).ok_or_else(|| {
            Error::invalid_argument("pooled_uninit_output", "shape", "compact stride overflow")
        })?;
    }
    Ok(strides)
}

#[derive(Debug)]
/// Owns a pooled full-overwrite destination until initialization is proven.
///
/// This type is public because `tenferro-cpu` is a sibling crate and must use
/// the same canonical owner; `pub(crate)` would prevent that cross-crate
/// ownership. Before the unsafe completion handoff, callers receive only
/// `MaybeUninit` storage.
pub struct PooledUninitOutput<'pool, T: PoolScalar> {
    pool: &'pool mut BufferPool,
    shape: Vec<usize>,
    strides: Vec<isize>,
    data: Vec<std::mem::MaybeUninit<T>>,
    checkout: Option<UninitCheckoutToken>,
    byte_len: usize,
}

impl<'pool, T: PoolScalar> PooledUninitOutput<'pool, T> {
    /// Creates a compact, pooled full-overwrite destination.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_internal_cpu_kernels::{buffer_pool::BufferPool, PooledUninitOutput};
    /// let mut pool = BufferPool::new();
    /// let output = PooledUninitOutput::<f32>::new(&mut pool, vec![2, 3]).unwrap();
    /// drop(output);
    /// ```
    /// # Errors
    /// Returns `Error::Validation` for shape-product, layout, or stride
    /// validation failures, or `Error::BackendSource` if allocation
    /// reservation fails. No pool accounting is changed before validation.
    pub fn new(pool: &'pool mut BufferPool, shape: Vec<usize>) -> Result<Self> {
        let len = checked_shape_product("pooled_uninit_output", "shape", &shape)?;
        let layout = Layout::array::<T>(len).map_err(|_| {
            Error::invalid_argument(
                "pooled_uninit_output",
                "shape",
                "element byte layout exceeds allocation limits",
            )
        })?;
        let byte_len = layout.size();
        let strides = checked_compact_strides(&shape)?;
        let (data, checkout) =
            <T as crate::buffer_pool::private::Sealed>::pool_acquire_uninit_tracked(pool, len)?;
        debug_assert!(match checkout {
            UninitCheckoutToken::Fresh { actual_capacity }
            | UninitCheckoutToken::Reused { actual_capacity } => {
                data.capacity() == actual_capacity
            }
        });
        Ok(Self {
            pool,
            shape,
            strides,
            data,
            checkout: Some(checkout),
            byte_len,
        })
    }

    /// Borrows the destination as `MaybeUninit` elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_internal_cpu_kernels::{buffer_pool::BufferPool, PooledUninitOutput};
    /// let mut pool = BufferPool::new();
    /// let mut output = PooledUninitOutput::<i32>::new(&mut pool, vec![1]).unwrap();
    /// output.as_uninit_slice_mut()[0].write(7);
    /// ```
    ///
    pub fn as_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
        self.data.as_mut_slice()
    }

    /// Borrows the validated compact destination for typed uninitialized kernels.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_internal_cpu_kernels::{buffer_pool::BufferPool, PooledUninitOutput};
    /// let mut pool = BufferPool::new();
    /// let mut output = PooledUninitOutput::<f32>::new(&mut pool, vec![2]).unwrap();
    /// let view = output.as_uninit_view_mut().unwrap();
    /// assert_eq!(view.dims(), &[2]);
    /// ```
    /// # Errors
    /// Returns `Error::BackendSource` if validated strided-view metadata is
    /// rejected.
    pub fn as_uninit_view_mut(&mut self) -> Result<StridedViewMut<'_, MaybeUninit<T>>> {
        let shape = &self.shape;
        let strides = &self.strides;
        let data = &mut self.data;
        StridedViewMut::new(data, shape, strides, 0)
            .map_err(|err| Error::backend_source("pooled_uninit_output", err))
    }

    #[cfg(test)]
    pub(crate) fn token_is_reused_with_capacity(&self, capacity: usize) -> bool {
        matches!(self.checkout, Some(UninitCheckoutToken::Reused { actual_capacity }) if actual_capacity == capacity)
    }

    #[cfg(test)]
    pub(crate) fn token_is_fresh(&self) -> bool {
        matches!(
            self.checkout,
            Some(UninitCheckoutToken::Fresh { actual_capacity })
                if self.data.capacity() == actual_capacity
        )
    }

    /// Returns the backing bytes for erased uninitialized kernel descriptors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_internal_cpu_kernels::{buffer_pool::BufferPool, PooledUninitOutput};
    /// let mut pool = BufferPool::new();
    /// let mut output = PooledUninitOutput::<i32>::new(&mut pool, vec![1]).unwrap();
    /// assert_eq!(output.as_uninit_bytes_mut().len(), 4);
    /// ```
    pub fn as_uninit_bytes_mut(&mut self) -> &mut [std::mem::MaybeUninit<u8>] {
        let data = self.as_uninit_slice_mut();
        // SAFETY: this creates only MaybeUninit byte elements over the same allocation.
        unsafe {
            std::slice::from_raw_parts_mut(
                data.as_mut_ptr().cast::<std::mem::MaybeUninit<u8>>(),
                self.byte_len,
            )
        }
    }

    /// Converts the allocation after a successful full-overwrite kernel.
    ///
    /// # Safety
    /// Every logical element must have been initialized by the completed kernel.
    /// The kernel must have completed all validation before writing, and must
    /// not retain any destination view after returning.
    /// Successful completion transfers success accounting to the owning
    /// `BufferPoolLoan` context; direct internal callers must keep that context alive.
    /// Completes the handoff as a dynamic-rank typed tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_internal_cpu_kernels::{buffer_pool::BufferPool, PooledUninitOutput};
    /// let mut pool = BufferPool::new();
    /// let mut output = PooledUninitOutput::<i32>::new(&mut pool, vec![1]).unwrap();
    /// output.as_uninit_slice_mut()[0].write(7);
    /// // SAFETY: the preceding write initializes every logical destination element.
    /// // SAFETY: the example writes every logical destination element before completion.
    /// let tensor = unsafe { output.assume_init() }.unwrap();
    /// assert_eq!(tensor.as_slice().unwrap(), &[7]);
    /// ```
    ///
    /// # Errors
    /// Returns `Error::Validation` if tensor construction rejects the shape or
    /// data length. Dynamic-rank conversion itself is infallible, so this
    /// method does not produce a rank-conversion `Error::BackendSource`.
    pub unsafe fn assume_init(self) -> Result<TypedTensor<T>>
    where
        T: TensorScalar,
    {
        // SAFETY: this method has the same full-initialization precondition.
        unsafe { self.assume_init_as::<tenferro_tensor::DynRank>() }
    }

    /// Completes the handoff with an explicitly selected tensor rank.
    ///
    /// # Safety
    /// Every logical element must have been initialized by the completed kernel.
    /// The destination must be fully initialized before this method is called;
    /// otherwise reading or dropping the resulting tensor is undefined behavior.
    ///
    /// # Errors
    /// Returns `Error::BackendSource` if rank conversion rejects the shape, or
    /// `Error::Validation` if tensor construction rejects the shape or length.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_internal_cpu_kernels::{buffer_pool::BufferPool, PooledUninitOutput};
    /// use tenferro_tensor::Rank;
    /// let mut pool = BufferPool::new();
    /// let mut output = PooledUninitOutput::<i32>::new(&mut pool, vec![1]).unwrap();
    /// output.as_uninit_slice_mut()[0].write(7);
    /// // SAFETY: the preceding write initializes every logical destination element.
    /// let tensor = unsafe { output.assume_init_as::<Rank<1>>() }.unwrap();
    /// assert_eq!(tensor.as_slice().unwrap(), &[7]);
    /// ```
    pub unsafe fn assume_init_as<R: TensorRank>(mut self) -> Result<TypedTensor<T, R>>
    where
        T: TensorScalar,
    {
        let shape = R::shape_from_vec(std::mem::take(&mut self.shape).into())
            .map_err(|err| Error::backend_source("pooled_uninit_output", err))?;
        let data = std::mem::take(&mut self.data);
        let mut data = ManuallyDrop::new(data);
        // SAFETY: caller proves initialization and MaybeUninit<T> has identical layout.
        let data = unsafe {
            Vec::from_raw_parts(data.as_mut_ptr().cast::<T>(), data.len(), data.capacity())
        };
        let tensor = TypedTensor::from_vec_col_major(shape, data)?;
        self.checkout = None;
        Ok(tensor)
    }
}

impl<T: PoolScalar> Drop for PooledUninitOutput<'_, T> {
    fn drop(&mut self) {
        let Some(checkout) = self.checkout.take() else {
            return;
        };
        let data = std::mem::take(&mut self.data);
        <T as crate::buffer_pool::private::Sealed>::pool_discard_uninit(
            &mut *self.pool,
            data,
            checkout,
        );
    }
}

#[cfg(test)]
mod tests;
