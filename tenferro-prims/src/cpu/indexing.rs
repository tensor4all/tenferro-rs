use tenferro_algebra::{Scalar, Standard};
use tenferro_device::{Error, Result};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{
    validate_execute_inputs, validate_shape_count, CpuBackend, CpuContext, IndexingPrimsDescriptor,
    ScatterReduction, TensorIndexingPrims,
};

/// CPU execution plan for the indexing protocol family.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::CpuIndexingPlan;
/// let _ = std::mem::size_of::<CpuIndexingPlan>();
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuIndexingPlan {
    /// Select slices along an axis by index.
    IndexSelect {
        /// Axis along which to select.
        axis: usize,
    },
    /// Gather elements along an axis using an index tensor.
    Gather {
        /// Axis along which to gather.
        axis: usize,
    },
    /// Scatter source values along an axis into the output.
    Scatter {
        /// Axis along which to scatter.
        axis: usize,
        /// Reduction mode.
        reduction: ScatterReduction,
    },
    /// Put source values at index positions.
    IndexPut {
        /// Whether to accumulate (add) instead of overwriting.
        accumulate: bool,
    },
}

/// Execute `IndexSelect`: for each index i, copy the i-th slice along `axis`.
fn execute_index_select<T: Scalar>(
    source: &Tensor<T>,
    indices: &Tensor<i64>,
    output: &mut Tensor<T>,
    axis: usize,
) -> Result<()> {
    let src = source.contiguous(MemoryOrder::ColumnMajor);
    let src_data = src
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;
    let idx = indices.contiguous(MemoryOrder::ColumnMajor);
    let idx_data = idx
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;

    let src_shape = src.dims();
    let ndim = src_shape.len();
    if axis >= ndim {
        return Err(Error::InvalidArgument(format!(
            "index_select axis {axis} >= ndim {ndim}"
        )));
    }

    let num_indices = idx_data.len();
    let axis_size = src_shape[axis];

    // Compute the stride product for dimensions before and after axis
    let pre_size: usize = src_shape[..axis].iter().product();
    let post_size: usize = src_shape[axis + 1..].iter().product();

    // Build output shape: replace src_shape[axis] with num_indices
    let mut out_shape = src_shape.to_vec();
    out_shape[axis] = num_indices;
    let out_total: usize = out_shape.iter().product();

    // Allocate output data
    let mut out_data = vec![T::zero(); out_total];

    // Column-major: axis `axis` has stride = product of dims before it
    // For each combination of (pre, idx_pos, post):
    //   out[pre + idx_pos * pre_size + post * pre_size * num_indices]
    //     = src[pre + indices[idx_pos] * pre_size + post * pre_size * axis_size]
    for post in 0..post_size {
        for (idx_pos, &idx_val) in idx_data.iter().enumerate() {
            let idx_usize = idx_val as usize;
            if idx_usize >= axis_size {
                return Err(Error::InvalidArgument(format!(
                    "index_select: index {idx_val} out of bounds for axis {axis} with size {axis_size}"
                )));
            }
            let src_offset = idx_usize * pre_size + post * pre_size * axis_size;
            let out_offset = idx_pos * pre_size + post * pre_size * num_indices;
            out_data[out_offset..out_offset + pre_size]
                .copy_from_slice(&src_data[src_offset..src_offset + pre_size]);
        }
    }

    *output = Tensor::from_slice(&out_data, &out_shape, MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("index_select output: {e}")))?;
    Ok(())
}

/// Execute `Gather`: for each position in the output, use the index tensor to
/// select an element along `axis`.
fn execute_gather<T: Scalar>(
    source: &Tensor<T>,
    indices: &Tensor<i64>,
    output: &mut Tensor<T>,
    axis: usize,
) -> Result<()> {
    let src = source.contiguous(MemoryOrder::ColumnMajor);
    let src_data = src
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;
    let idx = indices.contiguous(MemoryOrder::ColumnMajor);
    let idx_data = idx
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;

    let src_shape = src.dims();
    let ndim = src_shape.len();
    if axis >= ndim {
        return Err(Error::InvalidArgument(format!(
            "gather axis {axis} >= ndim {ndim}"
        )));
    }

    let out_shape = idx.dims().to_vec();
    if out_shape.len() != ndim {
        return Err(Error::InvalidArgument(format!(
            "gather: index tensor rank {} != source rank {ndim}",
            out_shape.len()
        )));
    }

    let axis_size = src_shape[axis];
    let total: usize = out_shape.iter().product();
    let mut out_data = vec![T::zero(); total];

    // Column-major multi-index iteration
    let mut multi_idx = vec![0usize; ndim];
    for flat in 0..total {
        // The index tensor gives us the position along axis
        let idx_val = idx_data[flat];
        let idx_usize = idx_val as usize;
        if idx_usize >= axis_size {
            return Err(Error::InvalidArgument(format!(
                "gather: index {idx_val} out of bounds for axis {axis} with size {axis_size}"
            )));
        }

        // Compute source flat index: same multi-index but with axis replaced by idx_val
        let mut src_flat = 0usize;
        let mut stride = 1usize;
        for d in 0..ndim {
            let coord = if d == axis { idx_usize } else { multi_idx[d] };
            src_flat += coord * stride;
            stride *= src_shape[d];
        }

        out_data[flat] = src_data[src_flat];

        // Increment multi-index (column-major order)
        for d in 0..ndim {
            multi_idx[d] += 1;
            if multi_idx[d] < out_shape[d] {
                break;
            }
            multi_idx[d] = 0;
        }
    }

    *output = Tensor::from_slice(&out_data, &out_shape, MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("gather output: {e}")))?;
    Ok(())
}

/// Execute `Scatter`: place source values into output at indexed positions
/// along `axis`.
fn execute_scatter<T: Scalar>(
    source: &Tensor<T>,
    indices: &Tensor<i64>,
    output: &mut Tensor<T>,
    axis: usize,
    reduction: ScatterReduction,
) -> Result<()> {
    let src = source.contiguous(MemoryOrder::ColumnMajor);
    let src_data = src
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;
    let idx = indices.contiguous(MemoryOrder::ColumnMajor);
    let idx_data = idx
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;

    let out_shape = output.dims().to_vec();
    let ndim = out_shape.len();
    if axis >= ndim {
        return Err(Error::InvalidArgument(format!(
            "scatter axis {axis} >= ndim {ndim}"
        )));
    }

    let src_shape = src.dims();
    if src_shape.len() != ndim {
        return Err(Error::InvalidArgument(format!(
            "scatter: source rank {} != output rank {ndim}",
            src_shape.len()
        )));
    }

    let axis_size = out_shape[axis];
    let total: usize = src_shape.iter().product();

    // Make output contiguous so we can write into it
    let out_contig = output.contiguous(MemoryOrder::ColumnMajor);
    let mut out_data = out_contig
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?
        .to_vec();

    // Column-major multi-index iteration over source shape
    let mut multi_idx = vec![0usize; ndim];
    for flat in 0..total {
        let idx_val = idx_data[flat];
        let idx_usize = idx_val as usize;
        if idx_usize >= axis_size {
            return Err(Error::InvalidArgument(format!(
                "scatter: index {idx_val} out of bounds for axis {axis} with size {axis_size}"
            )));
        }

        // Compute output flat index: same multi-index but with axis replaced by idx_val
        let mut out_flat = 0usize;
        let mut stride = 1usize;
        for d in 0..ndim {
            let coord = if d == axis { idx_usize } else { multi_idx[d] };
            out_flat += coord * stride;
            stride *= out_shape[d];
        }

        match reduction {
            ScatterReduction::None => {
                out_data[out_flat] = src_data[flat];
            }
            ScatterReduction::Add => {
                out_data[out_flat] = out_data[out_flat] + src_data[flat];
            }
        }

        // Increment multi-index (column-major order)
        for d in 0..ndim {
            multi_idx[d] += 1;
            if multi_idx[d] < src_shape[d] {
                break;
            }
            multi_idx[d] = 0;
        }
    }

    *output = Tensor::from_slice(&out_data, &out_shape, MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("scatter output: {e}")))?;
    Ok(())
}

/// Execute `IndexPut`: place values at the 1-D index positions.
fn execute_index_put<T: Scalar>(
    values: &Tensor<T>,
    indices: &Tensor<i64>,
    output: &mut Tensor<T>,
    accumulate: bool,
) -> Result<()> {
    let vals = values.contiguous(MemoryOrder::ColumnMajor);
    let vals_data = vals
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;
    let idx = indices.contiguous(MemoryOrder::ColumnMajor);
    let idx_data = idx
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;

    let out_shape = output.dims().to_vec();
    let out_total: usize = out_shape.iter().product();
    let out_contig = output.contiguous(MemoryOrder::ColumnMajor);
    let mut out_data = out_contig
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?
        .to_vec();

    if idx_data.len() != vals_data.len() {
        return Err(Error::InvalidArgument(format!(
            "index_put: indices length {} != values length {}",
            idx_data.len(),
            vals_data.len()
        )));
    }

    for (i, &idx_val) in idx_data.iter().enumerate() {
        let idx_usize = idx_val as usize;
        if idx_usize >= out_total {
            return Err(Error::InvalidArgument(format!(
                "index_put: index {idx_val} out of bounds for output of size {out_total}"
            )));
        }
        if accumulate {
            out_data[idx_usize] = out_data[idx_usize] + vals_data[i];
        } else {
            out_data[idx_usize] = vals_data[i];
        }
    }

    *output = Tensor::from_slice(&out_data, &out_shape, MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("index_put output: {e}")))?;
    Ok(())
}

impl<S: Scalar + 'static> TensorIndexingPrims<Standard<S>> for CpuBackend {
    type Plan = CpuIndexingPlan;
    type Context = CpuContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &IndexingPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        validate_shape_count(shapes, 3, "IndexingPrims")?;
        match desc {
            IndexingPrimsDescriptor::IndexSelect { axis } => {
                Ok(CpuIndexingPlan::IndexSelect { axis: *axis })
            }
            IndexingPrimsDescriptor::Gather { axis } => Ok(CpuIndexingPlan::Gather { axis: *axis }),
            IndexingPrimsDescriptor::Scatter { axis, reduction } => Ok(CpuIndexingPlan::Scatter {
                axis: *axis,
                reduction: *reduction,
            }),
            IndexingPrimsDescriptor::IndexPut { accumulate } => Ok(CpuIndexingPlan::IndexPut {
                accumulate: *accumulate,
            }),
        }
    }

    fn execute(
        _ctx: &mut Self::Context,
        plan: &Self::Plan,
        inputs: &[&Tensor<S>],
        indices: &Tensor<i64>,
        output: &mut Tensor<S>,
    ) -> Result<()> {
        validate_execute_inputs(inputs, 1, "IndexingPrims")?;
        match plan {
            CpuIndexingPlan::IndexSelect { axis } => {
                execute_index_select(inputs[0], indices, output, *axis)
            }
            CpuIndexingPlan::Gather { axis } => execute_gather(inputs[0], indices, output, *axis),
            CpuIndexingPlan::Scatter { axis, reduction } => {
                execute_scatter(inputs[0], indices, output, *axis, *reduction)
            }
            CpuIndexingPlan::IndexPut { accumulate } => {
                execute_index_put(inputs[0], indices, output, *accumulate)
            }
        }
    }

    fn has_indexing_support(_desc: IndexingPrimsDescriptor) -> bool {
        true
    }
}
