//! Scalar-agnostic entry points over the ordinary CPU numerical bodies.
//!
//! The preset scalar types reach the ordinary CPU kernels through the typed
//! pool, which hands out the destination buffer. An external scalar type cannot
//! use that pool, so these entry points take a caller-provided destination and
//! the caller's own arithmetic instead, and run the same elementwise and
//! reduction bodies the preset path uses.
//!
//! Nothing here inspects a dtype tag: the element type and the arithmetic are
//! the caller's. Support for the preset scalars is therefore not a precondition
//! for using the ordinary CPU numerical path, and no set-specific numerical
//! specialization is introduced.

use strided_kernel::{reduce, zip_map2_into, StridedView, StridedViewMut};
use tenferro_tensor::col_major_strides;
use tenferro_tensor_core::HostTensor;

fn strides_for(shape: &[usize]) -> crate::Result<Vec<isize>> {
    col_major_strides(shape)
}

fn require_same_shape(op: &'static str, lhs: &[usize], rhs: &[usize]) -> crate::Result<()> {
    if lhs == rhs {
        Ok(())
    } else {
        Err(crate::Error::shape_mismatch(op, lhs.to_vec(), rhs.to_vec()))
    }
}

/// Apply a caller-supplied binary operation elementwise into a caller-owned
/// destination.
///
/// The destination, the two operands, and the arithmetic are all the caller's.
/// The traversal is the same `zip_map2_into` body the preset scalar types use
/// through the typed pool; only the destination's origin differs.
///
/// # Examples
///
/// ```rust
/// use tenferro_internal_cpu_kernels::scalar_ops::scalar_binary_into;
/// use tenferro_tensor_core::HostTensor;
///
/// let lhs = HostTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
/// let rhs = HostTensor::from_vec_col_major(vec![2], vec![10.0_f64, 20.0])?;
/// let mut out = HostTensor::from_vec_col_major(vec![2], vec![0.0_f64, 0.0])?;
/// scalar_binary_into("add", &mut out, &lhs, &rhs, |a, b| a + b)?;
/// assert_eq!(out.as_slice(), &[11.0, 22.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns an error when the three shapes are not identical, or when the
/// underlying strided traversal rejects the views.
pub fn scalar_binary_into<T, F>(
    op: &'static str,
    destination: &mut HostTensor<T>,
    lhs: &HostTensor<T>,
    rhs: &HostTensor<T>,
    f: F,
) -> crate::Result<()>
where
    T: Copy + Send + Sync,
    F: Fn(T, T) -> T + Copy + Sync,
{
    require_same_shape(op, destination.shape(), lhs.shape())?;
    require_same_shape(op, lhs.shape(), rhs.shape())?;

    let strides = strides_for(lhs.shape())?;
    let destination_shape = destination.shape().to_vec();
    let mut destination_view: StridedViewMut<'_, T> =
        StridedViewMut::new(destination.as_mut_slice(), &destination_shape, &strides, 0)
            .map_err(|err| crate::Error::backend_source(op, err))?;
    let lhs_view: StridedView<'_, T> = StridedView::new(lhs.as_slice(), lhs.shape(), &strides, 0)
        .map_err(|err| crate::Error::backend_source(op, err))?;
    let rhs_view: StridedView<'_, T> = StridedView::new(rhs.as_slice(), rhs.shape(), &strides, 0)
        .map_err(|err| crate::Error::backend_source(op, err))?;

    zip_map2_into(&mut destination_view, &lhs_view, &rhs_view, f)
        .map_err(|err| crate::Error::backend_source(op, err))
}

/// Fold every element of a caller-owned tensor with a caller-supplied
/// associative operation, starting from `init`.
///
/// The accumulation order is the backend's and is not part of the contract. The
/// arithmetic is the caller's, so an extended-precision scalar keeps its low
/// components exactly as a preset scalar keeps its own.
///
/// # Examples
///
/// ```rust
/// use tenferro_internal_cpu_kernels::scalar_ops::scalar_fold;
/// use tenferro_tensor_core::HostTensor;
///
/// let values = HostTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
/// let total = scalar_fold("sum", &values, 0.0_f64, |a, b| a + b)?;
/// assert_eq!(total, 6.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns an error when the strided reduction rejects the view.
pub fn scalar_fold<T, F>(
    op: &'static str,
    source: &HostTensor<T>,
    init: T,
    f: F,
) -> crate::Result<T>
where
    T: Copy + Send + Sync,
    F: Fn(T, T) -> T + Copy + Sync,
{
    let strides = strides_for(source.shape())?;
    let view: StridedView<'_, T> = StridedView::new(source.as_slice(), source.shape(), &strides, 0)
        .map_err(|err| crate::Error::backend_source(op, err))?;
    reduce(&view, |element| element, f, init).map_err(|err| crate::Error::backend_source(op, err))
}

#[cfg(test)]
mod tests;
