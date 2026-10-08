#[cfg(feature = "native")]
pub mod faer_linalg;

#[cfg(feature = "blas")]
#[cfg_attr(feature = "native", allow(dead_code, unused_imports))]
pub mod lapack_linalg;

mod rank_revealing_qr;

use strided_view::RawStridedRef;
use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar, PooledUninitOutput};
use tenferro_tensor::{TypedTensor, TypedTensorView};

/// Describe a host view to a linalg provider without copying it.
///
/// Both providers read their operands as borrowed strided descriptors. The view's shape, strides
/// and offset pass through unchanged; `RawStridedRef::new` re-validates that every reachable offset
/// lies inside the borrowed storage, so a malformed view fails closed instead of reading out of
/// range. A negative stride is not a supported provider layout.
pub(crate) fn raw_view<'v, T: 'static>(
    op: &'static str,
    view: &'v TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<RawStridedRef<'v, T>> {
    if view.strides().iter().any(|&stride| stride < 0) {
        return Err(tenferro_tensor::Error::unsupported(
            op,
            "a negative stride is not a supported linalg input layout",
        ));
    }
    RawStridedRef::new(
        view.host_storage()?,
        view.shape(),
        view.strides(),
        view.offset(),
    )
    .map_err(|error| tenferro_tensor::Error::invalid_argument(op, "layout", error.to_string()))
}

#[cfg(feature = "native")]
pub(crate) use faer_linalg as faer;
#[cfg(feature = "blas")]
pub(crate) use lapack_linalg as blas;

/// A view the providers can read in place, or a compact pooled copy of it.
///
/// The providers take non-negative strides only. A borrowed operand with a negative stride (a
/// reversed slice, say) is gathered once into a compact column-major tensor, which is the copy the
/// pre-extraction routes made for every such operand anyway; any other layout is read in place.
pub(crate) fn provider_readable<T: Copy + Clone + PoolScalar + 'static>(
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
    op: &'static str,
) -> tenferro_tensor::Result<Option<TypedTensor<T>>> {
    if view.strides().iter().all(|&stride| stride >= 0) {
        return Ok(None);
    }
    output_from_rhs_view(buffers, view, op).map(Some)
}

pub(crate) fn output_from_rhs_view<T: Copy + Clone + PoolScalar + 'static>(
    buffers: &mut BufferPool,
    rhs: &TypedTensorView<'_, T>,
    op: &'static str,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let rank = rhs.shape().len();
    if !matches!(rank, 1 | 2) {
        return Err(tenferro_tensor::Error::rank_mismatch(op, 2, rank));
    }
    let mut output = PooledUninitOutput::<T>::new(buffers, rhs.shape().to_vec())?;
    let data = output.as_uninit_slice_mut();

    // PooledUninitOutput::new validates the compact shape product before
    // allocating `data`. Thus the indices below are in bounds and the
    // column-major arithmetic for the compact destination cannot
    // overflow: `row + col * rows < rows * cols`.
    if rank == 1 {
        let rows = rhs.shape()[0];
        for (row, slot) in data.iter_mut().enumerate().take(rows) {
            let value = rhs.get(&[row]).ok_or_else(|| {
                tenferro_tensor::Error::runtime_state(op, "RHS view is not host-addressable")
            })?;
            slot.write(*value);
        }
    } else {
        let rows = rhs.shape()[0];
        let cols = rhs.shape()[1];
        for col in 0..cols {
            for row in 0..rows {
                let index = row + col * rows;
                let value = rhs.get(&[row, col]).ok_or_else(|| {
                    tenferro_tensor::Error::runtime_state(op, "RHS view is not host-addressable")
                })?;
                data[index].write(*value);
            }
        }
    }

    // SAFETY: every element of the compact destination was initialized from
    // the corresponding logical RHS element above.
    let mut output = unsafe { output.assume_init()? };
    output.set_placement(rhs.placement().clone());
    Ok(output)
}
