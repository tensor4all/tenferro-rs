use std::alloc::Layout;
use std::mem::{ManuallyDrop, MaybeUninit};

use strided_basic::StridedViewMut;
use tenferro_tensor::{validate::checked_shape_product, TensorRank, TypedTensor};

use crate::buffer_pool::{BufferPool, PoolScalar, UninitCheckoutToken};
use crate::{Error, Result};

fn checked_compact_strides(shape: &[usize]) -> Result<tenferro_tensor::StrideVec> {
    // Inline for common ranks: the output lease is taken on every allocated op.
    let mut strides = tenferro_tensor::StrideVec::with_capacity(shape.len());
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
/// `MaybeUninit` storage. Each checkout owns a shared pool handle, so output
/// and scratch leases can coexist without holding a mutable pool borrow.
///
/// # Examples
///
/// ```
/// use tenferro_cpu_basic::{buffer_pool::BufferPool, PooledUninitOutput};
/// let pool = BufferPool::new();
/// let mut output = PooledUninitOutput::<i32>::new(&pool, vec![2])?;
/// for slot in output.as_uninit_slice_mut() {
///     slot.write(3);
/// }
/// // SAFETY: the loop initializes every logical destination element.
/// let tensor = unsafe { output.assume_init_recycled() }?;
/// assert_eq!(tensor.as_slice()?, &[3, 3]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub struct PooledUninitOutput<T: PoolScalar> {
    pool: BufferPool,
    shape: Vec<usize>,
    strides: tenferro_tensor::StrideVec,
    data: Vec<std::mem::MaybeUninit<T>>,
    checkout: Option<UninitCheckoutToken>,
    byte_len: usize,
}

impl<T: PoolScalar> PooledUninitOutput<T> {
    /// Creates a compact, pooled full-overwrite destination.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu_basic::{buffer_pool::BufferPool, PooledUninitOutput};
    /// let mut pool = BufferPool::new();
    /// let output = PooledUninitOutput::<f32>::new(&pool, vec![2, 3]).unwrap();
    /// drop(output);
    /// ```
    /// # Errors
    /// Returns `Error::Validation` for shape-product, layout, or stride
    /// validation failures, or `Error::BackendSource` if allocation
    /// reservation fails. No pool accounting is changed before validation.
    pub fn new(pool: &BufferPool, shape: Vec<usize>) -> Result<Self> {
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
        let mut pool = pool.checkout_handle();
        let (data, checkout) =
            <T as crate::buffer_pool::private::Sealed>::pool_acquire_uninit_tracked(
                &mut pool, len,
            )?;
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
    /// use tenferro_cpu_basic::{buffer_pool::BufferPool, PooledUninitOutput};
    /// let mut pool = BufferPool::new();
    /// let mut output = PooledUninitOutput::<i32>::new(&pool, vec![1]).unwrap();
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
    /// use tenferro_cpu_basic::{buffer_pool::BufferPool, PooledUninitOutput};
    /// let mut pool = BufferPool::new();
    /// let mut output = PooledUninitOutput::<f32>::new(&pool, vec![2]).unwrap();
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
    /// use tenferro_cpu_basic::{buffer_pool::BufferPool, PooledUninitOutput};
    /// let mut pool = BufferPool::new();
    /// let mut output = PooledUninitOutput::<i32>::new(&pool, vec![1]).unwrap();
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
    /// Successful completion transfers the checkout's return target to the
    /// initialized tensor; the pool lock is held only for brief bookkeeping.
    /// Finish a fully initialized ranked output while retaining its pool recycler.
    ///
    /// # Safety
    /// Every logical element must have been initialized and no writer may retain
    /// the output view.
    ///
    /// # Errors
    /// Returns `Error::Validation` with `ValidationError::RankMismatch` if the
    /// checked output metadata cannot be represented by the rank `R`.
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu_basic::{buffer_pool::BufferPool, PooledUninitOutput};
    /// use tenferro_tensor::Rank;
    /// let mut pool = BufferPool::new();
    /// let mut output = PooledUninitOutput::<i32>::new(&pool, vec![1]).unwrap();
    /// output.as_uninit_slice_mut()[0].write(7);
    /// // SAFETY: the preceding write initializes every logical destination element.
    /// let tensor = unsafe { output.assume_init_as_recycled::<Rank<1>>() }.unwrap();
    /// assert_eq!(tensor.as_slice().unwrap(), &[7]);
    /// ```
    pub unsafe fn assume_init_as_recycled<R: TensorRank>(self) -> Result<TypedTensor<T, R>> {
        // SAFETY: the caller guarantees complete initialization.
        unsafe { self.finish() }
    }

    /// Transfer a fully initialized output with automatic original-pool return.
    /// The pool is retained weakly; dropping the pool first frees later outputs
    /// normally. Dropping an output never acquires a CPU session lock.
    ///
    /// # Safety
    /// Every logical element must be initialized and no kernel may retain a
    /// destination view.
    ///
    /// # Errors
    /// Returns [`Error::Validation`] if the output shape and initialized buffer
    /// length disagree, or [`Error::RuntimeState`] if recycler attachment cannot
    /// obtain the host allocation. Shape conversion failures retain their typed
    /// source as [`Error::BackendSource`].
    ///
    /// # Examples
    /// ```
    /// use tenferro_cpu_basic::{buffer_pool::BufferPool, PooledUninitOutput};
    /// let mut pool = BufferPool::new();
    /// let mut output = PooledUninitOutput::<f64>::new(&pool, vec![1])?;
    /// output.as_uninit_slice_mut()[0].write(2.0);
    /// // SAFETY: the only element has been initialized.
    /// let tensor = unsafe { output.assume_init_recycled() }?;
    /// assert_eq!(tensor.as_slice()?, &[2.0]);
    /// drop(tensor);
    /// assert_eq!(pool.len(), 1);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub unsafe fn assume_init_recycled(self) -> Result<TypedTensor<T>> {
        // SAFETY: the caller guarantees complete initialization.
        unsafe { self.finish() }
    }

    unsafe fn finish<R: TensorRank>(mut self) -> Result<TypedTensor<T, R>> {
        let shape = R::shape_from_vec(std::mem::take(&mut self.shape).into())
            .map_err(|err| Error::backend_source("pooled_uninit_output", err))?;
        let data = std::mem::take(&mut self.data);
        let mut data = ManuallyDrop::new(data);
        // SAFETY: caller proves initialization and MaybeUninit<T> has identical layout.
        let data = unsafe {
            Vec::from_raw_parts(data.as_mut_ptr().cast::<T>(), data.len(), data.capacity())
        };
        // INVARIANT: only successful completion takes the checkout, and this
        // method consumes self, so a live output always owns its token here.
        let Some(checkout) = self.checkout.take() else {
            unreachable!("a live pooled output owns its checkout");
        };
        let recycler = <T as crate::buffer_pool::private::Sealed>::pool_finish_recycled(
            &mut self.pool,
            checkout,
        );
        TypedTensor::from_vec_col_major_with_recycler(shape, data, recycler)
    }
}

impl<T: PoolScalar> Drop for PooledUninitOutput<T> {
    /// Returns an abandoned destination's buffer to the pool instead of freeing it.
    ///
    /// An output that is dropped without being finished — an operation that failed or that
    /// deliberately abandons a destination — leaves storage that is still the pool's, so it goes
    /// back for the next call rather than to the C allocator. `pool_release` also settles the
    /// buffer's in-flight count.
    fn drop(&mut self) {
        if self.checkout.is_none() {
            return;
        }
        let data = std::mem::take(&mut self.data);
        let mut data = ManuallyDrop::new(data);
        // SAFETY: `MaybeUninit<T>` and `T` have the same layout, so this is the same allocation
        // with the same capacity. The elements are still uninitialized, and the pool only ever
        // hands its storage back out as `MaybeUninit` or zero fills it before use, so no
        // uninitialized `T` is read. Every pooled scalar is `Copy`, so the buffer has no drop
        // glue to run over them.
        let buf = unsafe {
            Vec::from_raw_parts(data.as_mut_ptr().cast::<T>(), data.len(), data.capacity())
        };
        <T as PoolScalar>::pool_release(&mut self.pool, buf);
    }
}

#[cfg(test)]
mod tests;
