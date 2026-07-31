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
    let element_count = tenferro_tensor::validate::checked_shape_product(op, "rhs", rhs.shape())?;
    let mut output = PooledUninitOutput::<T>::new(buffers, rhs.shape().to_vec())?;
    let data = output.as_uninit_slice_mut();
    match rhs.shape() {
        [rows] => {
            for (row, slot) in data.iter_mut().enumerate().take(*rows) {
                let value = rhs.get(&[row]).ok_or_else(|| {
                    tenferro_tensor::Error::runtime_state(op, "RHS view is not host-addressable")
                })?;
                slot.write(*value);
            }
        }
        [rows, cols] => {
            for col in 0..*cols {
                let col_offset = col.checked_mul(*rows).ok_or_else(|| {
                    tenferro_tensor::Error::invalid_argument(
                        op,
                        "rhs",
                        "RHS compact index overflows usize",
                    )
                })?;
                for row in 0..*rows {
                    let index = col_offset.checked_add(row).ok_or_else(|| {
                        tenferro_tensor::Error::invalid_argument(
                            op,
                            "rhs",
                            "RHS compact index overflows usize",
                        )
                    })?;
                    let value = rhs.get(&[row, col]).ok_or_else(|| {
                        tenferro_tensor::Error::runtime_state(
                            op,
                            "RHS view is not host-addressable",
                        )
                    })?;
                    data[index].write(*value);
                }
            }
        }
        _ => unreachable!("rank was validated above"),
    }
    debug_assert_eq!(element_count, data.len());
    // SAFETY: every element of the compact destination was initialized from
    // the corresponding logical RHS element above.
    let mut output = unsafe { output.assume_init()? };
    output.set_placement(rhs.placement().clone());
    Ok(output)
}
