use tenferro_algebra::{Scalar, Standard};
use tenferro_device::{Error, Result};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{validate_shape_count, CpuBackend, CpuContext, SortPrimsDescriptor, TensorSortPrims};

/// CPU execution plan for the sort protocol family.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::CpuSortPlan;
/// let _ = std::mem::size_of::<CpuSortPlan>();
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuSortPlan {
    /// Sort elements along an axis.
    Sort {
        /// Axis along which to sort.
        axis: usize,
        /// If true, sort in descending order.
        descending: bool,
        /// If true, use a stable sort.
        stable: bool,
    },
    /// Compute the sort permutation along an axis.
    Argsort {
        /// Axis along which to compute the permutation.
        axis: usize,
        /// If true, sort in descending order.
        descending: bool,
        /// If true, use a stable sort.
        stable: bool,
    },
    /// Select top-k elements along an axis.
    Topk {
        /// Axis along which to select top-k.
        axis: usize,
        /// Number of elements to select.
        k: usize,
        /// If true, select the k largest.
        largest: bool,
        /// If true, the returned k elements are sorted.
        sorted: bool,
    },
}

/// Sort one axis-slice, returning (sorted_values, original_indices).
fn sort_slice<T: Copy + PartialOrd>(
    slice: &[T],
    descending: bool,
    stable: bool,
) -> (Vec<T>, Vec<i64>) {
    let mut indexed: Vec<(T, i64)> = slice.iter().copied().zip(0i64..).collect();
    let cmp = |a: &(T, i64), b: &(T, i64)| {
        let ord = a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal);
        if descending {
            ord.reverse()
        } else {
            ord
        }
    };
    if stable {
        indexed.sort_by(cmp);
    } else {
        indexed.sort_unstable_by(cmp);
    }
    let values: Vec<T> = indexed.iter().map(|(v, _)| *v).collect();
    let indices: Vec<i64> = indexed.iter().map(|(_, i)| *i).collect();
    (values, indices)
}

/// Execute sort or argsort along the given axis.
fn execute_sort_along_axis<T: Scalar + PartialOrd>(
    input: &Tensor<T>,
    values_out: &mut Tensor<T>,
    indices_out: &mut Tensor<i64>,
    axis: usize,
    descending: bool,
    stable: bool,
    write_values: bool,
) -> Result<()> {
    let src = input.contiguous(MemoryOrder::ColumnMajor);
    let src_data = src
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;

    let shape = src.dims();
    let ndim = shape.len();
    if axis >= ndim {
        return Err(Error::InvalidArgument(format!(
            "sort axis {axis} >= ndim {ndim}"
        )));
    }

    let axis_size = shape[axis];
    let pre_size: usize = shape[..axis].iter().product();
    let post_size: usize = shape[axis + 1..].iter().product();

    let total: usize = shape.iter().product();
    let mut val_data = vec![T::zero(); total];
    let mut idx_data = vec![0i64; total];

    // Temporary buffer for one axis slice
    let mut slice_buf = vec![T::zero(); axis_size];

    for post in 0..post_size {
        for pre in 0..pre_size {
            // Extract axis slice from column-major data
            for a in 0..axis_size {
                let flat = pre + a * pre_size + post * pre_size * axis_size;
                slice_buf[a] = src_data[flat];
            }

            let (sorted_vals, sorted_idx) = sort_slice(&slice_buf, descending, stable);

            // Write back
            for a in 0..axis_size {
                let flat = pre + a * pre_size + post * pre_size * axis_size;
                if write_values {
                    val_data[flat] = sorted_vals[a];
                }
                idx_data[flat] = sorted_idx[a];
            }
        }
    }

    if write_values {
        *values_out = Tensor::from_slice(&val_data, shape, MemoryOrder::ColumnMajor)
            .map_err(|e| Error::InvalidArgument(format!("sort values output: {e}")))?;
    }
    *indices_out = Tensor::from_slice(&idx_data, shape, MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("sort indices output: {e}")))?;

    Ok(())
}

/// Execute topk along the given axis.
fn execute_topk<T: Scalar + PartialOrd>(
    input: &Tensor<T>,
    values_out: &mut Tensor<T>,
    indices_out: &mut Tensor<i64>,
    axis: usize,
    k: usize,
    largest: bool,
    sorted: bool,
) -> Result<()> {
    let src = input.contiguous(MemoryOrder::ColumnMajor);
    let src_data = src
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;

    let shape = src.dims();
    let ndim = shape.len();
    if axis >= ndim {
        return Err(Error::InvalidArgument(format!(
            "topk axis {axis} >= ndim {ndim}"
        )));
    }

    let axis_size = shape[axis];
    if k > axis_size {
        return Err(Error::InvalidArgument(format!(
            "topk k={k} > axis size {axis_size}"
        )));
    }

    let pre_size: usize = shape[..axis].iter().product();
    let post_size: usize = shape[axis + 1..].iter().product();

    // Output shape: same as input but with axis_size replaced by k
    let mut out_shape = shape.to_vec();
    out_shape[axis] = k;
    let out_total: usize = out_shape.iter().product();

    let mut val_data = vec![T::zero(); out_total];
    let mut idx_data = vec![0i64; out_total];

    let mut slice_buf = vec![T::zero(); axis_size];

    for post in 0..post_size {
        for pre in 0..pre_size {
            // Extract axis slice
            for a in 0..axis_size {
                let flat = pre + a * pre_size + post * pre_size * axis_size;
                slice_buf[a] = src_data[flat];
            }

            // Sort descending if largest, ascending if smallest
            let (sorted_vals, sorted_idx) = sort_slice(&slice_buf, largest, sorted);

            // Take first k
            for a in 0..k {
                let out_flat = pre + a * pre_size + post * pre_size * k;
                val_data[out_flat] = sorted_vals[a];
                idx_data[out_flat] = sorted_idx[a];
            }
        }
    }

    *values_out = Tensor::from_slice(&val_data, &out_shape, MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("topk values output: {e}")))?;
    *indices_out = Tensor::from_slice(&idx_data, &out_shape, MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("topk indices output: {e}")))?;

    Ok(())
}

impl<S: Scalar + PartialOrd + 'static> TensorSortPrims<Standard<S>> for CpuBackend {
    type Plan = CpuSortPlan;
    type Context = CpuContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &SortPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        validate_shape_count(shapes, 1, "SortPrims")?;
        match desc {
            SortPrimsDescriptor::Sort {
                axis,
                descending,
                stable,
            } => Ok(CpuSortPlan::Sort {
                axis: *axis,
                descending: *descending,
                stable: *stable,
            }),
            SortPrimsDescriptor::Argsort {
                axis,
                descending,
                stable,
            } => Ok(CpuSortPlan::Argsort {
                axis: *axis,
                descending: *descending,
                stable: *stable,
            }),
            SortPrimsDescriptor::Topk {
                axis,
                k,
                largest,
                sorted,
            } => Ok(CpuSortPlan::Topk {
                axis: *axis,
                k: *k,
                largest: *largest,
                sorted: *sorted,
            }),
        }
    }

    fn execute(
        _ctx: &mut Self::Context,
        plan: &Self::Plan,
        input: &Tensor<S>,
        values_out: &mut Tensor<S>,
        indices_out: &mut Tensor<i64>,
    ) -> Result<()> {
        match plan {
            CpuSortPlan::Sort {
                axis,
                descending,
                stable,
            } => execute_sort_along_axis(
                input,
                values_out,
                indices_out,
                *axis,
                *descending,
                *stable,
                true,
            ),
            CpuSortPlan::Argsort {
                axis,
                descending,
                stable,
            } => execute_sort_along_axis(
                input,
                values_out,
                indices_out,
                *axis,
                *descending,
                *stable,
                false,
            ),
            CpuSortPlan::Topk {
                axis,
                k,
                largest,
                sorted,
            } => execute_topk(input, values_out, indices_out, *axis, *k, *largest, *sorted),
        }
    }

    fn has_sort_support(_desc: &SortPrimsDescriptor) -> bool {
        true
    }
}
