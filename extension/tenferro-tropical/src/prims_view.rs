use strided_view::{StridedView, StridedViewMut};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

/// Convert a CPU tensor to an immutable strided view.
pub(crate) fn tensor_to_view<T: Scalar>(t: &Tensor<T>) -> Result<StridedView<'_, T>> {
    let data = t
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;
    StridedView::new(data, t.dims(), t.strides(), t.offset())
        .map_err(|e| Error::StrideError(format!("{e}")))
}

/// Convert a CPU tensor to a mutable strided view.
pub(crate) fn tensor_to_view_mut<T: Scalar>(t: &mut Tensor<T>) -> Result<StridedViewMut<'_, T>> {
    let dims = t.dims().to_vec();
    let strides = t.strides().to_vec();
    let offset = t.offset();
    let data = t
        .buffer_mut()
        .as_mut_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;
    StridedViewMut::new(data, &dims, &strides, offset)
        .map_err(|e| Error::StrideError(format!("{e}")))
}

/// Iterate over all index combinations for the given dimensions (column-major order).
pub(crate) fn for_each_index(dims: &[usize], mut f: impl FnMut(&[usize])) {
    let ndim = dims.len();
    if ndim == 0 {
        f(&[]);
        return;
    }
    let total: usize = dims.iter().product();
    if total == 0 {
        return;
    }
    let mut index = vec![0usize; ndim];
    for _ in 0..total {
        f(&index);
        for d in 0..ndim {
            index[d] += 1;
            if index[d] < dims[d] {
                break;
            }
            index[d] = 0;
        }
    }
}

/// Unflatten a linear index to multi-dimensional indices (column-major).
pub(crate) fn unflatten_index(flat: usize, dims: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; dims.len()];
    let mut remainder = flat;
    for d in 0..dims.len() {
        indices[d] = remainder % dims[d];
        remainder /= dims[d];
    }
    indices
}

/// Find the position of a mode label in a mode list.
pub(crate) fn mode_position(modes: &[u32], label: u32) -> Result<usize> {
    modes
        .iter()
        .position(|&m| m == label)
        .ok_or_else(|| Error::InvalidArgument(format!("mode label {label} not found")))
}

/// Scale all elements of the output by `beta`, or zero them if `beta == 0`.
pub(crate) fn scale_output<T: Scalar>(output: &mut StridedViewMut<T>, beta: T) {
    let dims = output.dims().to_vec();
    if beta == T::zero() {
        for_each_index(&dims, |idx| {
            output.set(idx, T::zero());
        });
    } else if beta != T::one() {
        for_each_index(&dims, |idx| {
            let old = output.get(idx);
            output.set(idx, beta * old);
        });
    }
}
