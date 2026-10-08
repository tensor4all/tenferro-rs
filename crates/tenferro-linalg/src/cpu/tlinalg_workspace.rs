//! The `tlinalg-blas` scratch contract over tenferro's buffer pool.
//!
//! The LAPACK provider obtains every reusable buffer through [`tlinalg_blas::Workspace`] (and
//! [`tlinalg_blas::IndexWorkspace`] for integer work buffers). The faer provider owns its scratch
//! and takes none. This is the host side of that contract: the pool, its retention and its accounting stay
//! exactly as they are, so the extraction does not introduce a second allocator.
//!
//! # Abandonment
//!
//! A buffer that is dropped rather than released stays checked out to the caller, which is what the
//! pool's own capacity path does today: it takes an allocation and discards the checkout token. The
//! host clears outstanding accounting at the end of a session, and replenishes on unwind
//! (`CpuBackend::with_execution_resources`), so nothing here has to be tidy on an error path.

#![cfg(feature = "blas")]

use core::mem::MaybeUninit;

use num_complex::{Complex32, Complex64};
use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tlinalg_blas::{IndexWorkspace, Workspace};

/// tenferro's implementation of the extracted scratch contract.
pub(crate) struct TlinalgWorkspace<'a> {
    pool: &'a mut BufferPool,
}

impl<'a> TlinalgWorkspace<'a> {
    /// Wrap the session's pool for one call.
    pub(crate) fn new(pool: &'a mut BufferPool) -> Self {
        Self { pool }
    }
}

macro_rules! impl_workspace {
    ($scalar:ty) => {
        impl Workspace<$scalar> for TlinalgWorkspace<'_> {
            fn acquire_zeroed(&mut self, len: usize) -> Vec<$scalar> {
                <$scalar as PoolScalar>::pool_acquire_zeroed(self.pool, len)
            }

            fn acquire_capacity(&mut self, cap: usize) -> Vec<$scalar> {
                self.pool.acquire_with_capacity::<$scalar>(cap)
            }

            fn acquire_uninit(&mut self, len: usize) -> Vec<MaybeUninit<$scalar>> {
                // The pool's capacity path hands back a length-zero vector over a recycled
                // allocation whose contents are unspecified, which is exactly an uninitialized
                // buffer. `MaybeUninit` is a transparent wrapper, so this is a view change over the
                // same allocation, not a second one.
                let buffer = self.pool.acquire_with_capacity::<$scalar>(len);
                // SAFETY: `Vec<$scalar>` and `Vec<MaybeUninit<$scalar>>` have identical layout, and
                // the vector has length zero, so no initialized `$scalar` is dropped or observed as
                // initialized.
                let mut uninit: Vec<MaybeUninit<$scalar>> = unsafe {
                    core::mem::transmute::<Vec<$scalar>, Vec<MaybeUninit<$scalar>>>(buffer)
                };
                uninit.resize_with(len, MaybeUninit::uninit);
                uninit
            }

            fn release(&mut self, buf: Vec<$scalar>) {
                <$scalar as PoolScalar>::pool_release(self.pool, buf)
            }
        }
    };
}

impl_workspace!(f32);
impl_workspace!(f64);
impl_workspace!(Complex32);
impl_workspace!(Complex64);

impl IndexWorkspace for TlinalgWorkspace<'_> {
    fn acquire_zeroed_index(&mut self, len: usize) -> Vec<i32> {
        <i32 as PoolScalar>::pool_acquire_zeroed(self.pool, len)
    }

    fn release_index(&mut self, buf: Vec<i32>) {
        <i32 as PoolScalar>::pool_release(self.pool, buf)
    }
}
