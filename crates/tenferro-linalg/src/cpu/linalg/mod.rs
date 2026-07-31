#[cfg(feature = "cpu-faer")]
pub mod faer_linalg;

#[cfg(feature = "cpu-blas")]
#[cfg_attr(feature = "cpu-faer", allow(dead_code, unused_imports))]
pub mod lapack_linalg;

use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar, PooledUninitOutput};
use tenferro_tensor::{TypedTensor, TypedTensorView};

#[cfg(feature = "cpu-faer")]
pub(crate) use faer_linalg as faer;
#[cfg(feature = "cpu-blas")]
pub(crate) use lapack_linalg as blas;

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
