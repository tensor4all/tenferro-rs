use std::mem::size_of_val;

use smallvec::SmallVec;
use strided_kernel::{
    col_major_strides, ErasedConcatenatePlan, ErasedDynamicSlicePlan, ErasedDynamicUpdateSlicePlan,
    ErasedGatherPlan, ErasedPadPlan, ErasedRawStridedPtr, ErasedRawStridedRef, ErasedReversePlan,
    ErasedScatterPlan, ErasedSlicePlan, ExecContext, GatherSpec, KernelDType, ScatterSpec,
};

use super::indexed_plan_cache::{IndexedPlanCache, IndexedPlanFamily, IndexedPlanKey};
use super::typed_host_data;
use super::PooledUninitOutput;
use crate::buffer_pool::{BufferPool, PoolScalar};
use tenferro_tensor::TensorScalar;
use tenferro_tensor::{DType, GatherConfig, PadConfig, ScatterConfig, SliceConfig};
use tenferro_tensor::{Tensor, TypedTensor};

type InlineStrides = SmallVec<[isize; 8]>;

fn inline_col_major_strides(op: &'static str, dims: &[usize]) -> crate::Result<InlineStrides> {
    let mut strides = InlineStrides::from_elem(1, dims.len());
    for axis in 1..dims.len() {
        if strides[axis - 1] == 0 {
            strides[axis] = 0;
            continue;
        }
        let prior_dim = isize::try_from(dims[axis - 1]).map_err(|_| {
            crate::Error::invalid_argument(
                op,
                "shape",
                format!(
                    "dimension {} does not fit in isize while computing column-major strides",
                    dims[axis - 1]
                ),
            )
        })?;
        strides[axis] = strides[axis - 1].checked_mul(prior_dim).ok_or_else(|| {
            crate::Error::invalid_argument(
                op,
                "shape",
                format!("column-major stride overflows isize for shape {dims:?}"),
            )
        })?;
    }
    Ok(strides)
}

// Indexing families delegate bulk traversal to strided-kernel erased plans.
// Backend entrypoints inject the CpuContext-derived execution policy.

trait TensorAsTyped<T> {
    fn as_typed(&self) -> Option<&TypedTensor<T>>;
}

macro_rules! impl_tensor_as_typed {
    ($(($ty:ty, $variant:ident)),+ $(,)?) => {
        $(
            impl TensorAsTyped<$ty> for Tensor {
                fn as_typed(&self) -> Option<&TypedTensor<$ty>> {
                    match self {
                        Tensor::$variant(tensor) => Some(tensor),
                        _ => None,
                    }
                }
            }
        )+
    };
}

impl_tensor_as_typed!(
    (f32, F32),
    (f64, F64),
    (i32, I32),
    (i64, I64),
    (bool, Bool),
    (num_complex::Complex<f32>, C32),
    (num_complex::Complex<f64>, C64),
);

fn kernel_dtype(dtype: DType) -> KernelDType {
    match dtype {
        DType::F32 => KernelDType::F32,
        DType::F64 => KernelDType::F64,
        DType::I32 => KernelDType::I32,
        DType::I64 => KernelDType::I64,
        DType::Bool => KernelDType::Bool,
        DType::C32 => KernelDType::C32,
        DType::C64 => KernelDType::C64,
    }
}

fn typed_bytes<T>(data: &[T]) -> &[u8] {
    // SAFETY: `data` is an aligned typed slice. The returned byte slice has
    // the same lifetime and exact byte length, and is read-only.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) }
}

fn gather_spec(config: &GatherConfig) -> GatherSpec {
    GatherSpec {
        offset_dims: config.offset_dims.clone(),
        collapsed_slice_dims: config.collapsed_slice_dims.clone(),
        start_index_map: config.start_index_map.clone(),
        index_vector_dim: config.index_vector_dim,
        slice_sizes: config.slice_sizes.clone(),
    }
}

fn scatter_spec(config: &ScatterConfig) -> ScatterSpec {
    ScatterSpec {
        update_window_dims: config.update_window_dims.clone(),
        inserted_window_dims: config.inserted_window_dims.clone(),
        scatter_dims_to_operand_dims: config.scatter_dims_to_operand_dims.clone(),
        index_vector_dim: config.index_vector_dim,
    }
}

fn indexed_plan_key(
    family: IndexedPlanFamily,
    dtype: KernelDType,
    index_dtype: KernelDType,
    dims: &[&[usize]],
    strides: &[&[isize]],
    config: &[&[usize]],
) -> IndexedPlanKey {
    IndexedPlanKey::from_slices(family, dtype, index_dtype, dims, strides, config)
}

macro_rules! dispatch_tensor_unary_result {
    ($input:expr, |$tensor:ident| $body:expr) => {
        match $input {
            Tensor::F32($tensor) => Ok(Tensor::F32($body?)),
            Tensor::F64($tensor) => Ok(Tensor::F64($body?)),
            Tensor::I32($tensor) => Ok(Tensor::I32($body?)),
            Tensor::I64($tensor) => Ok(Tensor::I64($body?)),
            Tensor::Bool($tensor) => Ok(Tensor::Bool($body?)),
            Tensor::C32($tensor) => Ok(Tensor::C32($body?)),
            Tensor::C64($tensor) => Ok(Tensor::C64($body?)),
        }
    };
}

macro_rules! dispatch_same_dtype_result {
    ($op:literal, $lhs:expr, $rhs:expr, |$lhs_t:ident, $rhs_t:ident| $body:expr) => {
        match ($lhs, $rhs) {
            (Tensor::F32($lhs_t), Tensor::F32($rhs_t)) => Ok(Tensor::F32($body?)),
            (Tensor::F64($lhs_t), Tensor::F64($rhs_t)) => Ok(Tensor::F64($body?)),
            (Tensor::I32($lhs_t), Tensor::I32($rhs_t)) => Ok(Tensor::I32($body?)),
            (Tensor::I64($lhs_t), Tensor::I64($rhs_t)) => Ok(Tensor::I64($body?)),
            (Tensor::Bool($lhs_t), Tensor::Bool($rhs_t)) => Ok(Tensor::Bool($body?)),
            (Tensor::C32($lhs_t), Tensor::C32($rhs_t)) => Ok(Tensor::C32($body?)),
            (Tensor::C64($lhs_t), Tensor::C64($rhs_t)) => Ok(Tensor::C64($body?)),
            _ => Err(crate::Error::dtype_mismatch(
                $op,
                $lhs.dtype(),
                $rhs.dtype(),
            )),
        }
    };
}

macro_rules! dispatch_same_dtype_without_bool_result {
    ($op:literal, $lhs:expr, $rhs:expr, $bool_message:literal, |$lhs_t:ident, $rhs_t:ident| $body:expr) => {
        match ($lhs, $rhs) {
            (Tensor::F32($lhs_t), Tensor::F32($rhs_t)) => Ok(Tensor::F32($body?)),
            (Tensor::F64($lhs_t), Tensor::F64($rhs_t)) => Ok(Tensor::F64($body?)),
            (Tensor::I32($lhs_t), Tensor::I32($rhs_t)) => Ok(Tensor::I32($body?)),
            (Tensor::I64($lhs_t), Tensor::I64($rhs_t)) => Ok(Tensor::I64($body?)),
            (Tensor::C32($lhs_t), Tensor::C32($rhs_t)) => Ok(Tensor::C32($body?)),
            (Tensor::C64($lhs_t), Tensor::C64($rhs_t)) => Ok(Tensor::C64($body?)),
            (Tensor::Bool(_), Tensor::Bool(_)) => {
                Err(crate::Error::unsupported($op, $bool_message))
            }
            _ => Err(crate::Error::dtype_mismatch(
                $op,
                $lhs.dtype(),
                $rhs.dtype(),
            )),
        }
    };
}

#[cfg(test)]
fn with_test_pool<T>(f: impl FnOnce(&mut BufferPool) -> T) -> T {
    let mut buffers = BufferPool::new();
    f(&mut buffers)
}

#[cfg(test)]
pub(crate) fn gather(
    operand: &Tensor,
    start_indices: &Tensor,
    config: &GatherConfig,
) -> crate::Result<Tensor> {
    with_test_pool(|buffers| {
        let mut cache = IndexedPlanCache::default();
        gather_with_pool(
            buffers,
            &mut cache,
            &ExecContext::serial(),
            operand,
            start_indices,
            config,
        )
    })
}

pub(crate) fn gather_with_pool(
    buffers: &mut BufferPool,
    cache: &mut IndexedPlanCache,
    exec_context: &ExecContext,
    operand: &Tensor,
    start_indices: &Tensor,
    config: &GatherConfig,
) -> crate::Result<Tensor> {
    let start_indices = try_index_tensor(start_indices)?;
    dispatch_tensor_unary_result!(operand, |t| typed_gather(
        buffers,
        cache,
        exec_context,
        t,
        &start_indices,
        config
    ))
}

#[cfg(test)]
pub(crate) fn scatter(
    operand: &Tensor,
    scatter_indices: &Tensor,
    updates: &Tensor,
    config: &ScatterConfig,
) -> crate::Result<Tensor> {
    with_test_pool(|buffers| {
        let mut cache = IndexedPlanCache::default();
        scatter_with_pool(
            buffers,
            &mut cache,
            &ExecContext::serial(),
            operand,
            scatter_indices,
            updates,
            config,
        )
    })
}

pub(crate) fn scatter_with_pool(
    buffers: &mut BufferPool,
    cache: &mut IndexedPlanCache,
    exec_context: &ExecContext,
    operand: &Tensor,
    scatter_indices: &Tensor,
    updates: &Tensor,
    config: &ScatterConfig,
) -> crate::Result<Tensor> {
    let scatter_indices = try_index_tensor(scatter_indices)?;
    dispatch_same_dtype_without_bool_result!(
        "scatter",
        operand,
        updates,
        "Bool data tensors are not supported by additive scatter; supported data dtypes: F32/F64/I32/I64/C32/C64",
        |op, upd| typed_scatter(
            buffers,
            cache,
            exec_context,
            op,
            &scatter_indices,
            upd,
            config
        )
    )
}

pub(crate) fn try_slice_with_pool(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    input: &Tensor,
    config: &SliceConfig,
) -> crate::Result<Tensor> {
    dispatch_tensor_unary_result!(input, |t| typed_slice(buffers, exec_context, t, config))
}

#[cfg(test)]
pub(crate) fn dynamic_slice(
    input: &Tensor,
    starts: &Tensor,
    slice_sizes: &[usize],
) -> crate::Result<Tensor> {
    with_test_pool(|buffers| {
        let mut cache = IndexedPlanCache::default();
        dynamic_slice_with_pool(
            buffers,
            &mut cache,
            &ExecContext::serial(),
            input,
            starts,
            slice_sizes,
        )
    })
}

pub(crate) fn dynamic_slice_with_pool(
    buffers: &mut BufferPool,
    cache: &mut IndexedPlanCache,
    exec_context: &ExecContext,
    input: &Tensor,
    starts: &Tensor,
    slice_sizes: &[usize],
) -> crate::Result<Tensor> {
    let starts = try_index_tensor(starts)?;
    dispatch_tensor_unary_result!(input, |t| typed_dynamic_slice(
        buffers,
        cache,
        exec_context,
        t,
        &starts,
        slice_sizes
    ))
}

/// Return `operand` with `update` written at dynamic `starts`.
///
/// Starts are clamped so the whole update window fits inside the operand,
/// matching `dynamic_slice` behavior.
///
/// # Examples
///
/// ```
/// use tenferro_cpu as cpu;
/// use tenferro_tensor::{Tensor, TypedTensor};
///
/// let operand = Tensor::F64(TypedTensor::from_vec_col_major(vec![5], vec![0.0; 5])?);
/// let update = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![3.0, 4.0])?);
/// let starts = Tensor::I64(TypedTensor::from_vec_col_major(vec![1], vec![4])?);
///
/// let out = cpu::dynamic_update_slice(&operand, &update, &starts).unwrap();
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[0.0, 0.0, 0.0, 3.0, 4.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[cfg(test)]
pub(crate) fn dynamic_update_slice(
    operand: &Tensor,
    update: &Tensor,
    starts: &Tensor,
) -> crate::Result<Tensor> {
    with_test_pool(|buffers| {
        let mut cache = IndexedPlanCache::default();
        dynamic_update_slice_with_pool(
            buffers,
            &mut cache,
            &ExecContext::serial(),
            operand,
            update,
            starts,
        )
    })
}

pub(crate) fn dynamic_update_slice_with_pool(
    buffers: &mut BufferPool,
    cache: &mut IndexedPlanCache,
    exec_context: &ExecContext,
    operand: &Tensor,
    update: &Tensor,
    starts: &Tensor,
) -> crate::Result<Tensor> {
    let starts = try_index_tensor(starts)?;
    dispatch_same_dtype_result!("dynamic_update_slice", operand, update, |op, upd| {
        typed_dynamic_update_slice(buffers, cache, exec_context, op, upd, &starts)
    })
}

#[cfg(test)]
pub(crate) fn pad(input: &Tensor, config: &PadConfig) -> crate::Result<Tensor> {
    try_pad(input, config)
}

#[cfg(test)]
fn try_pad(input: &Tensor, config: &PadConfig) -> crate::Result<Tensor> {
    with_test_pool(|buffers| try_pad_with_pool(buffers, &ExecContext::serial(), input, config))
}

pub(crate) fn try_pad_with_pool(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    input: &Tensor,
    config: &PadConfig,
) -> crate::Result<Tensor> {
    dispatch_tensor_unary_result!(input, |t| typed_pad(buffers, exec_context, t, config))
}

pub(crate) fn try_concatenate_with_pool(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    inputs: &[&Tensor],
    axis: usize,
) -> crate::Result<Tensor> {
    let first = inputs.first().copied().ok_or_else(|| {
        crate::Error::invalid_argument(
            "concatenate",
            "configuration",
            "concatenate requires at least one input",
        )
    })?;
    dispatch_tensor_unary_result!(first, |t| typed_concatenate_from_dyn_inputs(
        buffers,
        exec_context,
        t,
        inputs,
        axis
    ))
}

pub(crate) fn reverse_with_pool(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    input: &Tensor,
    axes: &[usize],
) -> crate::Result<Tensor> {
    dispatch_tensor_unary_result!(input, |t| typed_reverse(buffers, exec_context, t, axes))
}

fn typed_slice<T: Copy + Clone + PoolScalar + TensorScalar>(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    input: &TypedTensor<T>,
    config: &SliceConfig,
) -> crate::Result<TypedTensor<T>> {
    let input_shape = input.shape();
    let rank = input_shape.len();
    if config.starts.len() != rank {
        return Err(crate::Error::rank_mismatch(
            "slice",
            rank,
            config.starts.len(),
        ));
    }
    if config.limits.len() != rank {
        return Err(crate::Error::rank_mismatch(
            "slice",
            rank,
            config.limits.len(),
        ));
    }
    if config.strides.len() != rank {
        return Err(crate::Error::rank_mismatch(
            "slice",
            rank,
            config.strides.len(),
        ));
    }

    let out_shape: Vec<usize> = input
        .shape()
        .iter()
        .enumerate()
        .map(|(axis, &dim)| {
            let start = config.starts[axis];
            let limit = config.limits[axis];
            let stride = config.strides[axis];
            if start > limit {
                return Err(crate::Error::invalid_argument(
                    "slice",
                    "configuration",
                    format!("start exceeds limit on axis {axis}"),
                ));
            }
            if limit > dim {
                return Err(crate::Error::invalid_argument(
                    "slice",
                    "configuration",
                    format!("limit {limit} on axis {axis} exceeds dimension size {dim}"),
                ));
            }
            if stride == 0 {
                return Err(crate::Error::invalid_argument(
                    "slice",
                    "configuration",
                    format!("stride must be positive on axis {axis}"),
                ));
            }
            let span = limit - start;
            Ok(span.div_ceil(stride))
        })
        .collect::<crate::Result<Vec<_>>>()?;

    let dtype = kernel_dtype(T::dtype());
    let input_strides = inline_col_major_strides("slice", input_shape)?;
    let out_strides = inline_col_major_strides("slice", &out_shape)?;
    let plan = ErasedSlicePlan::compile(
        dtype,
        input_shape,
        &input_strides,
        &out_shape,
        &out_strides,
        &config.starts,
        &config.limits,
        &config.strides,
    )
    .map_err(|err| crate::Error::backend_source("slice", err))?;

    let mut out = PooledUninitOutput::new(buffers, out_shape.clone())?;
    let input_ref = unsafe {
        // SAFETY: this operation supplies initialized typed bytes with
        // alignment and dtype matching the descriptor; validated dimensions,
        // strides, and offset keep every reachable source element in bounds
        // for the retained borrow.
        crate::erased_raw_strided_ref(
            dtype,
            typed_bytes(typed_host_data("slice", input)?),
            input_shape,
            &input_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source("slice", err))?;
    let input_ptr = ErasedRawStridedPtr::from_ref(&input_ref);
    let mut out_ref = unsafe {
        // SAFETY: the output guard exclusively owns aligned storage whose
        // byte layout agrees with dtype; validated dimensions, strides, and
        // offset keep every destination in bounds, and the following kernel
        // overwrites every reachable element before typed exposure.
        crate::erased_raw_strided_uninit_mut(
            dtype,
            out.as_uninit_bytes_mut(),
            &out_shape,
            &out_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source("slice", err))?;
    plan.execute_uninit(exec_context, &mut out_ref, &input_ptr)
        .map_err(|err| crate::Error::backend_source("slice", err))?;

    // INVARIANT: the compact output has no unreachable storage, so successful
    // static-slice replay initializes every element.
    // SAFETY: every element contains a valid T after successful replay.
    unsafe { out.assume_init() }
}

fn typed_concatenate_from_dyn_inputs<T>(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    _first: &TypedTensor<T>,
    inputs: &[&Tensor],
    axis: usize,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + PoolScalar + TensorScalar,
    Tensor: TensorAsTyped<T>,
{
    let first_dtype = inputs
        .first()
        .ok_or_else(|| {
            crate::Error::invalid_argument(
                "concatenate",
                "configuration",
                "concatenate requires at least one input",
            )
        })?
        .dtype();
    let typed_inputs = collect_typed_inputs(first_dtype, inputs)?;
    typed_concatenate(buffers, exec_context, &typed_inputs, axis)
}

fn collect_typed_inputs<'a, T>(
    first_dtype: crate::DType,
    inputs: &[&'a Tensor],
) -> crate::Result<Vec<&'a TypedTensor<T>>>
where
    Tensor: TensorAsTyped<T>,
{
    inputs
        .iter()
        .map(|tensor| {
            TensorAsTyped::<T>::as_typed(*tensor).ok_or_else(|| {
                crate::Error::dtype_mismatch("concatenate", first_dtype, tensor.dtype())
            })
        })
        .collect()
}

fn typed_concatenate<T: Copy + Clone + PoolScalar + TensorScalar>(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    inputs: &[&TypedTensor<T>],
    axis: usize,
) -> crate::Result<TypedTensor<T>> {
    let first = inputs.first().copied().ok_or_else(|| {
        crate::Error::invalid_argument(
            "concatenate",
            "configuration",
            "concatenate requires at least one input",
        )
    })?;
    let first_shape = first.shape();
    let rank = first_shape.len();
    if axis >= rank {
        return Err(crate::Error::axis_out_of_bounds("concatenate", axis, rank));
    }

    let mut out_shape = first_shape.to_vec();
    let mut axis_extent = 0usize;
    for input in inputs {
        let input_shape = input.shape();
        if input_shape.len() != rank {
            return Err(crate::Error::rank_mismatch(
                "concatenate",
                rank,
                input_shape.len(),
            ));
        }
        for dim in 0..rank {
            if dim == axis {
                axis_extent = axis_extent.checked_add(input_shape[dim]).ok_or_else(|| {
                    crate::Error::invalid_argument(
                        "concatenate",
                        "configuration",
                        "concatenate axis extent overflows usize".to_string(),
                    )
                })?;
            } else if input_shape[dim] != first_shape[dim] {
                return Err(crate::Error::shape_mismatch(
                    "concatenate",
                    first_shape.to_vec(),
                    input_shape.to_vec(),
                ));
            }
        }
    }
    out_shape[axis] = axis_extent;

    let dtype = kernel_dtype(T::dtype());
    let input_dims: SmallVec<[&[usize]; 4]> = inputs.iter().map(|input| input.shape()).collect();
    let input_strides: SmallVec<[InlineStrides; 4]> = input_dims
        .iter()
        .map(|dims| inline_col_major_strides("concatenate", dims))
        .collect::<crate::Result<_>>()?;
    let input_stride_refs: SmallVec<[&[isize]; 4]> =
        input_strides.iter().map(SmallVec::as_slice).collect();
    let out_strides = inline_col_major_strides("concatenate", &out_shape)?;
    let plan = ErasedConcatenatePlan::compile(
        dtype,
        &input_dims,
        &input_stride_refs,
        &out_shape,
        &out_strides,
        axis,
    )
    .map_err(|err| crate::Error::backend_source("concatenate", err))?;

    let mut out = PooledUninitOutput::new(buffers, out_shape.clone())?;
    let input_refs: SmallVec<[ErasedRawStridedRef<'_>; 4]> = inputs
        .iter()
        .zip(input_strides.iter())
        .map(|(input, strides)| {
            unsafe {
                // SAFETY: this operation supplies initialized typed bytes with
                // alignment and dtype matching the descriptor; validated dimensions,
                // strides, and offset keep every reachable source element in bounds
                // for the retained borrow.
                crate::erased_raw_strided_ref(
                    dtype,
                    typed_bytes(typed_host_data("concatenate", input)?),
                    input.shape(),
                    strides,
                    0,
                )
            }
            .map_err(|err| crate::Error::backend_source("concatenate", err))
        })
        .collect::<crate::Result<_>>()?;
    let input_ptrs: SmallVec<[ErasedRawStridedPtr<'_>; 4]> = input_refs
        .iter()
        .map(ErasedRawStridedPtr::from_ref)
        .collect();
    let mut out_ref = unsafe {
        // SAFETY: the output guard exclusively owns aligned storage whose
        // byte layout agrees with dtype; validated dimensions, strides, and
        // offset keep every destination in bounds, and the following kernel
        // overwrites every reachable element before typed exposure.
        crate::erased_raw_strided_uninit_mut(
            dtype,
            out.as_uninit_bytes_mut(),
            &out_shape,
            &out_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source("concatenate", err))?;
    plan.execute_uninit(exec_context, &mut out_ref, &input_ptrs)
        .map_err(|err| crate::Error::backend_source("concatenate", err))?;

    // INVARIANT: the compact output has no unreachable storage, so successful
    // concatenate replay initializes every element.
    // SAFETY: every element contains a valid T after successful replay.
    unsafe { out.assume_init() }
}

fn typed_reverse<T: Copy + Clone + PoolScalar + TensorScalar>(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    input: &TypedTensor<T>,
    axes: &[usize],
) -> crate::Result<TypedTensor<T>> {
    let input_shape = input.shape();
    let rank = input_shape.len();
    for &axis in axes {
        if axis >= rank {
            return Err(crate::Error::axis_out_of_bounds("reverse", axis, rank));
        }
    }

    let dtype = kernel_dtype(T::dtype());
    let input_strides = inline_col_major_strides("reverse", input_shape)?;
    let out_strides = inline_col_major_strides("reverse", input_shape)?;
    let plan = ErasedReversePlan::compile(dtype, input_shape, &input_strides, &out_strides, axes)
        .map_err(|err| crate::Error::backend_source("reverse", err))?;

    let mut out = PooledUninitOutput::new(buffers, input_shape.to_vec())?;
    let input_ref = unsafe {
        // SAFETY: this operation supplies initialized typed bytes with
        // alignment and dtype matching the descriptor; validated dimensions,
        // strides, and offset keep every reachable source element in bounds
        // for the retained borrow.
        crate::erased_raw_strided_ref(
            dtype,
            typed_bytes(typed_host_data("reverse", input)?),
            input_shape,
            &input_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source("reverse", err))?;
    let input_ptr = ErasedRawStridedPtr::from_ref(&input_ref);
    let mut out_ref = unsafe {
        // SAFETY: the output guard exclusively owns aligned storage whose
        // byte layout agrees with dtype; validated dimensions, strides, and
        // offset keep every destination in bounds, and the following kernel
        // overwrites every reachable element before typed exposure.
        crate::erased_raw_strided_uninit_mut(
            dtype,
            out.as_uninit_bytes_mut(),
            input_shape,
            &out_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source("reverse", err))?;
    plan.execute_uninit(exec_context, &mut out_ref, &input_ptr)
        .map_err(|err| crate::Error::backend_source("reverse", err))?;

    // INVARIANT: the compact output has no unreachable storage, so successful
    // reverse replay initializes every element.
    // SAFETY: every element contains a valid T after successful replay.
    unsafe { out.assume_init() }
}

struct IndexTensor {
    shape: Vec<usize>,
    values: Vec<i64>,
}

/// Maximum exact integer representable by f32 (2^24).
const F32_MAX_EXACT_INT: f32 = 16_777_216.0;
/// Maximum exact integer representable by f64 (2^53).
const F64_MAX_EXACT_INT: f64 = 9_007_199_254_740_992.0;

fn f32_index_to_i64(value: f32) -> crate::Result<i64> {
    if !value.is_finite() || value.fract() != 0.0 || value.abs() > F32_MAX_EXACT_INT {
        return Err(crate::Error::invalid_argument(
            "index_tensor",
            "index",
            format!("index value {value} is not an exactly representable i64"),
        ));
    }
    Ok(value as i64)
}

fn f64_index_to_i64(value: f64) -> crate::Result<i64> {
    if !value.is_finite() || value.fract() != 0.0 || value.abs() > F64_MAX_EXACT_INT {
        return Err(crate::Error::invalid_argument(
            "index_tensor",
            "index",
            format!("index value {value} is not an exactly representable i64"),
        ));
    }
    Ok(value as i64)
}

fn try_index_tensor(tensor: &Tensor) -> crate::Result<IndexTensor> {
    match tensor {
        Tensor::I32(t) => Ok(IndexTensor {
            shape: t.shape().to_vec(),
            values: typed_host_data("index_tensor", t)?
                .iter()
                .map(|&value| value as i64)
                .collect(),
        }),
        Tensor::I64(t) => Ok(IndexTensor {
            shape: t.shape().to_vec(),
            values: typed_host_data("index_tensor", t)?.to_vec(),
        }),
        Tensor::F32(t) => {
            let values: crate::Result<Vec<i64>> = typed_host_data("index_tensor", t)?
                .iter()
                .map(|&value| f32_index_to_i64(value))
                .collect();
            Ok(IndexTensor {
                shape: t.shape().to_vec(),
                values: values?,
            })
        }
        Tensor::F64(t) => {
            let values: crate::Result<Vec<i64>> = typed_host_data("index_tensor", t)?
                .iter()
                .map(|&value| f64_index_to_i64(value))
                .collect();
            Ok(IndexTensor {
                shape: t.shape().to_vec(),
                values: values?,
            })
        }
        Tensor::Bool(_) => Err(crate::Error::invalid_argument(
            "index_tensor",
            "configuration",
            "bool index tensors are not supported; supported index dtypes: I32/I64/F32/F64",
        )),
        Tensor::C32(_) | Tensor::C64(_) => Err(crate::Error::invalid_argument(
            "index_tensor",
            "configuration",
            "complex index tensors are not supported; supported index dtypes: I32/I64/F32/F64",
        )),
    }
}

fn checked_product(op: &'static str, role: &'static str, shape: &[usize]) -> crate::Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            crate::Error::invalid_argument(
                op,
                "configuration",
                format!("{role} element count overflows usize"),
            )
        })
    })
}

fn try_index_vector_size(
    op: &'static str,
    shape: &[usize],
    index_vector_dim: usize,
) -> crate::Result<usize> {
    if index_vector_dim > shape.len() {
        return Err(crate::Error::axis_out_of_bounds(
            op,
            index_vector_dim,
            shape.len(),
        ));
    }
    Ok(if index_vector_dim == shape.len() {
        1
    } else {
        shape[index_vector_dim]
    })
}

fn try_index_batch_shape(
    op: &'static str,
    shape: &[usize],
    index_vector_dim: usize,
) -> crate::Result<Vec<usize>> {
    if index_vector_dim > shape.len() {
        return Err(crate::Error::axis_out_of_bounds(
            op,
            index_vector_dim,
            shape.len(),
        ));
    }
    if index_vector_dim == shape.len() {
        return Ok(shape.to_vec());
    }
    Ok(shape
        .iter()
        .enumerate()
        .filter_map(|(axis, &dim)| (axis != index_vector_dim).then_some(dim))
        .collect())
}

fn clamp_window_start(
    op: &'static str,
    start: i64,
    dim_size: usize,
    window_size: usize,
) -> crate::Result<usize> {
    if window_size > dim_size {
        return Err(crate::Error::invalid_argument(
            op,
            "configuration",
            format!("window size {window_size} exceeds dimension size {dim_size}"),
        ));
    }
    let max_start = dim_size.saturating_sub(window_size) as i64;
    Ok(start.clamp(0, max_start) as usize)
}

fn operand_window_dims(rank: usize, collapsed_or_inserted: &[usize]) -> Vec<usize> {
    (0..rank)
        .filter(|dim| !collapsed_or_inserted.contains(dim))
        .collect()
}

fn typed_gather<T: Copy + Clone + PoolScalar + TensorScalar>(
    buffers: &mut BufferPool,
    cache: &mut IndexedPlanCache,
    exec_context: &ExecContext,
    operand: &TypedTensor<T>,
    start_indices: &IndexTensor,
    config: &GatherConfig,
) -> crate::Result<TypedTensor<T>> {
    let operand_shape = operand.shape();
    let rank = operand_shape.len();
    if config.slice_sizes.len() != rank {
        return Err(crate::Error::rank_mismatch(
            "gather",
            rank,
            config.slice_sizes.len(),
        ));
    }

    for &dim in &config.collapsed_slice_dims {
        if dim >= rank {
            return Err(crate::Error::axis_out_of_bounds("gather", dim, rank));
        }
    }
    {
        let mut seen = vec![false; rank];
        for &dim in &config.collapsed_slice_dims {
            if seen[dim] {
                return Err(crate::Error::duplicate_axis(
                    "gather",
                    dim,
                    "collapsed_slice_dims",
                ));
            }
            seen[dim] = true;
        }
    }
    for &dim in &config.collapsed_slice_dims {
        if config.slice_sizes[dim] != 1 {
            return Err(crate::Error::invalid_argument(
                "gather",
                "configuration",
                format!(
                    "collapsed slice dimension {dim} must have slice_size == 1, got {}",
                    config.slice_sizes[dim]
                ),
            ));
        }
    }

    let index_size =
        try_index_vector_size("gather", &start_indices.shape, config.index_vector_dim)?;
    if index_size != config.start_index_map.len() {
        return Err(crate::Error::invalid_argument(
            "gather",
            "configuration",
            format!(
                "start_index_map length {} does not match index vector size {}",
                config.start_index_map.len(),
                index_size
            ),
        ));
    }
    for &operand_dim in &config.start_index_map {
        if operand_dim >= rank {
            return Err(crate::Error::axis_out_of_bounds(
                "gather",
                operand_dim,
                rank,
            ));
        }
    }
    {
        let mut seen = vec![false; rank];
        for &operand_dim in &config.start_index_map {
            if seen[operand_dim] {
                return Err(crate::Error::duplicate_axis(
                    "gather",
                    operand_dim,
                    "start_index_map",
                ));
            }
            seen[operand_dim] = true;
        }
    }

    let window_dims = operand_window_dims(rank, &config.collapsed_slice_dims);
    if config.offset_dims.len() != window_dims.len() {
        return Err(crate::Error::invalid_argument(
            "gather",
            "configuration",
            format!(
                "offset_dims length {} does not match window dims count {}",
                config.offset_dims.len(),
                window_dims.len()
            ),
        ));
    }

    let batch_shape =
        try_index_batch_shape("gather", &start_indices.shape, config.index_vector_dim)?;
    let out_rank = batch_shape.len() + config.offset_dims.len();
    for &out_axis in &config.offset_dims {
        if out_axis >= out_rank {
            return Err(crate::Error::axis_out_of_bounds(
                "gather", out_axis, out_rank,
            ));
        }
    }
    {
        let mut seen = vec![false; out_rank];
        for &out_axis in &config.offset_dims {
            if seen[out_axis] {
                return Err(crate::Error::duplicate_axis(
                    "gather",
                    out_axis,
                    "offset_dims",
                ));
            }
            seen[out_axis] = true;
        }
    }

    let mut out_axis_to_operand_dim = vec![None; out_rank];
    for (offset_axis, &out_axis) in config.offset_dims.iter().enumerate() {
        out_axis_to_operand_dim[out_axis] = Some(window_dims[offset_axis]);
    }

    let mut out_shape = vec![0usize; out_rank];
    let mut batch_axis = 0usize;
    for out_axis in 0..out_rank {
        if let Some(operand_dim) = out_axis_to_operand_dim[out_axis] {
            out_shape[out_axis] = config.slice_sizes[operand_dim];
        } else {
            out_shape[out_axis] = batch_shape[batch_axis];
            batch_axis += 1;
        }
    }

    for (operand_dim, &dim_size) in operand_shape.iter().enumerate() {
        let _ = clamp_window_start("gather", 0, dim_size, config.slice_sizes[operand_dim])?;
    }

    let mut out = PooledUninitOutput::<T>::new(buffers, out_shape.clone())?;

    let dtype = kernel_dtype(T::dtype());
    let operand_strides = col_major_strides(operand_shape);
    let index_strides = col_major_strides(&start_indices.shape);
    let out_strides = col_major_strides(&out_shape);
    let index_dtype = KernelDType::I64;
    let key = indexed_plan_key(
        IndexedPlanFamily::Gather,
        dtype,
        index_dtype,
        &[operand_shape, &start_indices.shape, &out_shape],
        &[&operand_strides, &index_strides, &out_strides],
        &[
            &config.offset_dims,
            &config.collapsed_slice_dims,
            &config.start_index_map,
            std::slice::from_ref(&config.index_vector_dim),
            &config.slice_sizes,
        ],
    );
    let plan = cache
        .gather(key, || {
            ErasedGatherPlan::compile(
                dtype,
                index_dtype,
                operand_shape,
                &operand_strides,
                &start_indices.shape,
                &index_strides,
                &out_shape,
                &out_strides,
                gather_spec(config),
            )
        })
        .map_err(|err| crate::Error::backend_source("gather", err))?;

    let operand_ref = unsafe {
        // SAFETY: this operation supplies initialized typed bytes with
        // alignment and dtype matching the descriptor; validated dimensions,
        // strides, and offset keep every reachable source element in bounds
        // for the retained borrow.
        crate::erased_raw_strided_ref(
            dtype,
            typed_bytes(typed_host_data("gather", operand)?),
            operand_shape,
            &operand_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source("gather", err))?;
    let index_ref = unsafe {
        // SAFETY: this operation supplies initialized typed bytes with
        // alignment and dtype matching the descriptor; validated dimensions,
        // strides, and offset keep every reachable source element in bounds
        // for the retained borrow.
        crate::erased_raw_strided_ref(
            index_dtype,
            typed_bytes(&start_indices.values),
            &start_indices.shape,
            &index_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source("gather", err))?;
    let mut out_ref = unsafe {
        // SAFETY: the output guard exclusively owns aligned storage whose
        // byte layout agrees with dtype; validated dimensions, strides, and
        // offset keep every destination in bounds, and the following kernel
        // overwrites every reachable element before typed exposure.
        crate::erased_raw_strided_uninit_mut(
            dtype,
            out.as_uninit_bytes_mut(),
            &out_shape,
            &out_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source("gather", err))?;
    let operand_ptr = ErasedRawStridedPtr::from_ref(&operand_ref);
    let index_ptr = ErasedRawStridedPtr::from_ref(&index_ref);
    plan.execute_uninit(exec_context, &mut out_ref, &operand_ptr, &index_ptr)
        .map_err(|err| crate::Error::backend_source("gather", err))?;

    // SAFETY: the gather plan writes every logical output element.
    unsafe { out.assume_init() }
}

fn typed_scatter<T>(
    buffers: &mut BufferPool,
    cache: &mut IndexedPlanCache,
    exec_context: &ExecContext,
    operand: &TypedTensor<T>,
    scatter_indices: &IndexTensor,
    updates: &TypedTensor<T>,
    config: &ScatterConfig,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + PoolScalar + TensorScalar,
{
    let operand_shape = operand.shape();
    let updates_shape = updates.shape();
    let op_rank = operand_shape.len();
    for &dim in &config.inserted_window_dims {
        if dim >= op_rank {
            return Err(crate::Error::axis_out_of_bounds("scatter", dim, op_rank));
        }
    }
    {
        let mut seen = vec![false; op_rank];
        for &dim in &config.inserted_window_dims {
            if seen[dim] {
                return Err(crate::Error::duplicate_axis(
                    "scatter",
                    dim,
                    "inserted_window_dims",
                ));
            }
            seen[dim] = true;
        }
    }

    let index_size =
        try_index_vector_size("scatter", &scatter_indices.shape, config.index_vector_dim)?;
    if index_size != config.scatter_dims_to_operand_dims.len() {
        return Err(crate::Error::invalid_argument(
            "scatter",
            "configuration",
            format!(
                "scatter_dims_to_operand_dims length {} does not match index vector size {}",
                config.scatter_dims_to_operand_dims.len(),
                index_size
            ),
        ));
    }
    for &operand_dim in &config.scatter_dims_to_operand_dims {
        if operand_dim >= op_rank {
            return Err(crate::Error::axis_out_of_bounds(
                "scatter",
                operand_dim,
                op_rank,
            ));
        }
    }
    {
        let mut seen = vec![false; op_rank];
        for &operand_dim in &config.scatter_dims_to_operand_dims {
            if seen[operand_dim] {
                return Err(crate::Error::duplicate_axis(
                    "scatter",
                    operand_dim,
                    "scatter_dims_to_operand_dims",
                ));
            }
            seen[operand_dim] = true;
        }
    }

    let batch_shape =
        try_index_batch_shape("scatter", &scatter_indices.shape, config.index_vector_dim)?;
    let window_dims = operand_window_dims(op_rank, &config.inserted_window_dims);
    if config.update_window_dims.len() != window_dims.len() {
        return Err(crate::Error::invalid_argument(
            "scatter",
            "configuration",
            format!(
                "update_window_dims length {} does not match window dims count {}",
                config.update_window_dims.len(),
                window_dims.len()
            ),
        ));
    }

    let update_rank = updates_shape.len();
    let expected_batch_rank = update_rank
        .checked_sub(config.update_window_dims.len())
        .ok_or_else(|| {
            crate::Error::invalid_argument(
                "scatter",
                "configuration",
                format!(
                    "update_window_dims length {} exceeds update rank {}",
                    config.update_window_dims.len(),
                    update_rank
                ),
            )
        })?;
    if expected_batch_rank != batch_shape.len() {
        return Err(crate::Error::invalid_argument(
            "scatter",
            "configuration",
            format!(
                "updates batch rank {} does not match index batch shape length {}",
                expected_batch_rank,
                batch_shape.len()
            ),
        ));
    }

    for &axis in &config.update_window_dims {
        if axis >= update_rank {
            return Err(crate::Error::axis_out_of_bounds(
                "scatter",
                axis,
                update_rank,
            ));
        }
    }
    {
        let mut seen = vec![false; update_rank];
        for &axis in &config.update_window_dims {
            if seen[axis] {
                return Err(crate::Error::duplicate_axis(
                    "scatter",
                    axis,
                    "update_window_dims",
                ));
            }
            seen[axis] = true;
        }
    }

    let mut is_update_window_dim = vec![false; update_rank];
    for &axis in &config.update_window_dims {
        is_update_window_dim[axis] = true;
    }

    {
        let mut batch_axis = 0usize;
        for axis in 0..update_rank {
            if !is_update_window_dim[axis] {
                if updates_shape[axis] != batch_shape[batch_axis] {
                    return Err(crate::Error::invalid_argument("scatter", "configuration", format!(
                            "updates batch dim {} extent {} does not match index batch dim {} extent {}",
                            axis,
                            updates_shape[axis],
                            batch_axis,
                            batch_shape[batch_axis]
                        )));
                }
                batch_axis += 1;
            }
        }
    }

    let mut window_shape = vec![1usize; op_rank];
    let mut window_shape_updates = vec![0usize; config.update_window_dims.len()];
    for (pos, &update_axis) in config.update_window_dims.iter().enumerate() {
        let dim = updates_shape[update_axis];
        window_shape_updates[pos] = dim;
        window_shape[window_dims[pos]] = dim;
    }
    for axis in 0..op_rank {
        let _ = clamp_window_start("scatter", 0, operand_shape[axis], window_shape[axis])?;
    }

    checked_product("scatter", "batch shape", &batch_shape)?;
    checked_product("scatter", "window update shape", &window_shape_updates)?;

    let dtype = kernel_dtype(T::dtype());
    let index_dtype = KernelDType::I64;
    let operand_strides = col_major_strides(operand_shape);
    let index_strides = col_major_strides(&scatter_indices.shape);
    let update_strides = col_major_strides(updates_shape);
    let out_strides = col_major_strides(operand_shape);
    let key = indexed_plan_key(
        IndexedPlanFamily::Scatter,
        dtype,
        index_dtype,
        &[
            operand_shape,
            &scatter_indices.shape,
            updates_shape,
            operand_shape,
        ],
        &[
            &operand_strides,
            &index_strides,
            &update_strides,
            &out_strides,
        ],
        &[
            &config.update_window_dims,
            &config.inserted_window_dims,
            &config.scatter_dims_to_operand_dims,
            std::slice::from_ref(&config.index_vector_dim),
        ],
    );
    let plan = cache
        .scatter(key, || {
            ErasedScatterPlan::compile(
                dtype,
                index_dtype,
                operand_shape,
                &operand_strides,
                &scatter_indices.shape,
                &index_strides,
                updates_shape,
                &update_strides,
                operand_shape,
                &out_strides,
                scatter_spec(config),
            )
        })
        .map_err(|err| crate::Error::backend_source("scatter", err))?;

    // INVARIANT: ErasedScatterPlan first copies the full operand into `out`,
    // then applies every additive update.
    let mut out = PooledUninitOutput::<T>::new(buffers, operand_shape.to_vec())?;
    let operand_ref = unsafe {
        // SAFETY: this operation supplies initialized typed bytes with
        // alignment and dtype matching the descriptor; validated dimensions,
        // strides, and offset keep every reachable source element in bounds
        // for the retained borrow.
        crate::erased_raw_strided_ref(
            dtype,
            typed_bytes(typed_host_data("scatter", operand)?),
            operand_shape,
            &operand_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source("scatter", err))?;
    let index_ref = unsafe {
        // SAFETY: this operation supplies initialized typed bytes with
        // alignment and dtype matching the descriptor; validated dimensions,
        // strides, and offset keep every reachable source element in bounds
        // for the retained borrow.
        crate::erased_raw_strided_ref(
            index_dtype,
            typed_bytes(&scatter_indices.values),
            &scatter_indices.shape,
            &index_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source("scatter", err))?;
    let update_ref = unsafe {
        // SAFETY: this operation supplies initialized typed bytes with
        // alignment and dtype matching the descriptor; validated dimensions,
        // strides, and offset keep every reachable source element in bounds
        // for the retained borrow.
        crate::erased_raw_strided_ref(
            dtype,
            typed_bytes(typed_host_data("scatter", updates)?),
            updates_shape,
            &update_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source("scatter", err))?;
    let mut out_ref = unsafe {
        // SAFETY: the output guard exclusively owns aligned storage whose
        // byte layout agrees with dtype; validated dimensions, strides, and
        // offset keep every destination in bounds, and the following kernel
        // overwrites every reachable element before typed exposure.
        crate::erased_raw_strided_uninit_mut(
            dtype,
            out.as_uninit_bytes_mut(),
            operand_shape,
            &out_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source("scatter", err))?;
    let operand_ptr = ErasedRawStridedPtr::from_ref(&operand_ref);
    let index_ptr = ErasedRawStridedPtr::from_ref(&index_ref);
    let update_ptr = ErasedRawStridedPtr::from_ref(&update_ref);
    plan.execute_uninit(
        exec_context,
        &mut out_ref,
        &operand_ptr,
        &index_ptr,
        &update_ptr,
    )
    .map_err(|err| crate::Error::backend_source("scatter", err))?;

    // SAFETY: the scatter plan first copies the initialized operand and then
    // applies additive updates before returning success.
    unsafe { out.assume_init() }
}

fn typed_dynamic_slice<T: Copy + Clone + PoolScalar + TensorScalar>(
    buffers: &mut BufferPool,
    cache: &mut IndexedPlanCache,
    exec_context: &ExecContext,
    input: &TypedTensor<T>,
    starts: &IndexTensor,
    slice_sizes: &[usize],
) -> crate::Result<TypedTensor<T>> {
    let input_shape = input.shape();
    if slice_sizes.len() != input_shape.len() {
        return Err(crate::Error::rank_mismatch(
            "dynamic_slice",
            input_shape.len(),
            slice_sizes.len(),
        ));
    }
    if starts.shape.len() != 1 {
        return Err(crate::Error::invalid_argument(
            "dynamic_slice",
            "starts",
            "starts must be a rank-1 tensor",
        ));
    }
    if starts.values.len() != input_shape.len() {
        return Err(crate::Error::invalid_argument(
            "dynamic_slice",
            "starts",
            format!(
                "starts length {} must match input rank {}",
                starts.values.len(),
                input_shape.len()
            ),
        ));
    }

    for axis in 0..input_shape.len() {
        let _ = clamp_window_start(
            "dynamic_slice",
            starts.values[axis],
            input_shape[axis],
            slice_sizes[axis],
        )?;
    }

    let out_shape = slice_sizes.to_vec();
    let dtype = kernel_dtype(T::dtype());
    let input_strides = col_major_strides(input_shape);
    let start_strides = col_major_strides(&starts.shape);
    let out_strides = col_major_strides(&out_shape);
    let index_dtype = KernelDType::I64;
    let key = indexed_plan_key(
        IndexedPlanFamily::DynamicSlice,
        dtype,
        index_dtype,
        &[input_shape, &starts.shape, &out_shape],
        &[&input_strides, &start_strides, &out_strides],
        &[slice_sizes],
    );
    let plan = cache
        .dynamic_slice(key, || {
            ErasedDynamicSlicePlan::compile(
                dtype,
                index_dtype,
                input_shape,
                &input_strides,
                &starts.shape,
                &start_strides,
                &out_shape,
                &out_strides,
                slice_sizes,
            )
        })
        .map_err(|err| crate::Error::backend_source("dynamic_slice", err))?;
    // INVARIANT: ErasedDynamicSlicePlan writes every output coordinate exactly once.
    let mut out = PooledUninitOutput::<T>::new(buffers, out_shape.clone())?;
    let input_ref = unsafe {
        // SAFETY: this operation supplies initialized typed bytes with
        // alignment and dtype matching the descriptor; validated dimensions,
        // strides, and offset keep every reachable source element in bounds
        // for the retained borrow.
        crate::erased_raw_strided_ref(
            dtype,
            typed_bytes(typed_host_data("dynamic_slice", input)?),
            input_shape,
            &input_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source("dynamic_slice", err))?;
    let start_ref = unsafe {
        // SAFETY: this operation supplies initialized typed bytes with
        // alignment and dtype matching the descriptor; validated dimensions,
        // strides, and offset keep every reachable source element in bounds
        // for the retained borrow.
        crate::erased_raw_strided_ref(
            index_dtype,
            typed_bytes(&starts.values),
            &starts.shape,
            &start_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source("dynamic_slice", err))?;
    let mut out_ref = unsafe {
        // SAFETY: the output guard exclusively owns aligned storage whose
        // byte layout agrees with dtype; validated dimensions, strides, and
        // offset keep every destination in bounds, and the following kernel
        // overwrites every reachable element before typed exposure.
        crate::erased_raw_strided_uninit_mut(
            dtype,
            out.as_uninit_bytes_mut(),
            &out_shape,
            &out_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source("dynamic_slice", err))?;
    let input_ptr = ErasedRawStridedPtr::from_ref(&input_ref);
    let start_ptr = ErasedRawStridedPtr::from_ref(&start_ref);
    plan.execute_uninit(exec_context, &mut out_ref, &input_ptr, &start_ptr)
        .map_err(|err| crate::Error::backend_source("dynamic_slice", err))?;

    // SAFETY: the dynamic-slice plan writes every logical output element.
    unsafe { out.assume_init() }
}

fn typed_dynamic_update_slice<T: Copy + Clone + PoolScalar + TensorScalar>(
    buffers: &mut BufferPool,
    cache: &mut IndexedPlanCache,
    exec_context: &ExecContext,
    operand: &TypedTensor<T>,
    update: &TypedTensor<T>,
    starts: &IndexTensor,
) -> crate::Result<TypedTensor<T>> {
    let operand_shape = operand.shape();
    let update_shape = update.shape();
    if update_shape.len() != operand_shape.len() {
        return Err(crate::Error::rank_mismatch(
            "dynamic_update_slice",
            operand_shape.len(),
            update_shape.len(),
        ));
    }
    if starts.shape.len() != 1 {
        return Err(crate::Error::invalid_argument(
            "dynamic_update_slice",
            "configuration",
            "starts must be a rank-1 tensor",
        ));
    }
    if starts.values.len() != operand_shape.len() {
        return Err(crate::Error::invalid_argument(
            "dynamic_update_slice",
            "configuration",
            format!(
                "starts length {} must match operand rank {}",
                starts.values.len(),
                operand_shape.len()
            ),
        ));
    }

    for axis in 0..operand_shape.len() {
        let _ = clamp_window_start(
            "dynamic_update_slice",
            starts.values[axis],
            operand_shape[axis],
            update_shape[axis],
        )?;
    }

    let dtype = kernel_dtype(T::dtype());
    let operand_strides = col_major_strides(operand_shape);
    let start_strides = col_major_strides(&starts.shape);
    let update_strides = col_major_strides(update_shape);
    let out_strides = col_major_strides(operand_shape);
    let index_dtype = KernelDType::I64;
    let key = indexed_plan_key(
        IndexedPlanFamily::DynamicUpdateSlice,
        dtype,
        index_dtype,
        &[operand_shape, &starts.shape, update_shape, operand_shape],
        &[
            &operand_strides,
            &start_strides,
            &update_strides,
            &out_strides,
        ],
        &[],
    );
    let plan = cache
        .dynamic_update_slice(key, || {
            ErasedDynamicUpdateSlicePlan::compile(
                dtype,
                index_dtype,
                operand_shape,
                &operand_strides,
                &starts.shape,
                &start_strides,
                update_shape,
                &update_strides,
                operand_shape,
                &out_strides,
            )
        })
        .map_err(|err| crate::Error::backend_source("dynamic_update_slice", err))?;
    // INVARIANT: ErasedDynamicUpdateSlicePlan copies the full operand into
    // `out` before overwriting the update window.
    let mut out = PooledUninitOutput::<T>::new(buffers, operand_shape.to_vec())?;
    let operand_ref = unsafe {
        // SAFETY: this operation supplies initialized typed bytes with
        // alignment and dtype matching the descriptor; validated dimensions,
        // strides, and offset keep every reachable source element in bounds
        // for the retained borrow.
        crate::erased_raw_strided_ref(
            dtype,
            typed_bytes(typed_host_data("dynamic_update_slice", operand)?),
            operand_shape,
            &operand_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source("dynamic_update_slice", err))?;
    let update_ref = unsafe {
        // SAFETY: this operation supplies initialized typed bytes with
        // alignment and dtype matching the descriptor; validated dimensions,
        // strides, and offset keep every reachable source element in bounds
        // for the retained borrow.
        crate::erased_raw_strided_ref(
            dtype,
            typed_bytes(typed_host_data("dynamic_update_slice", update)?),
            update_shape,
            &update_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source("dynamic_update_slice", err))?;
    let start_ref = unsafe {
        // SAFETY: this operation supplies initialized typed bytes with
        // alignment and dtype matching the descriptor; validated dimensions,
        // strides, and offset keep every reachable source element in bounds
        // for the retained borrow.
        crate::erased_raw_strided_ref(
            index_dtype,
            typed_bytes(&starts.values),
            &starts.shape,
            &start_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source("dynamic_update_slice", err))?;
    let mut out_ref = unsafe {
        // SAFETY: the output guard exclusively owns aligned storage whose
        // byte layout agrees with dtype; validated dimensions, strides, and
        // offset keep every destination in bounds, and the following kernel
        // overwrites every reachable element before typed exposure.
        crate::erased_raw_strided_uninit_mut(
            dtype,
            out.as_uninit_bytes_mut(),
            operand_shape,
            &out_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source("dynamic_update_slice", err))?;
    let operand_ptr = ErasedRawStridedPtr::from_ref(&operand_ref);
    let update_ptr = ErasedRawStridedPtr::from_ref(&update_ref);
    let start_ptr = ErasedRawStridedPtr::from_ref(&start_ref);
    plan.execute_uninit(
        exec_context,
        &mut out_ref,
        &operand_ptr,
        &update_ptr,
        &start_ptr,
    )
    .map_err(|err| crate::Error::backend_source("dynamic_update_slice", err))?;

    // SAFETY: the dynamic-update plan first copies the initialized operand and
    // then overwrites the update window before returning success.
    unsafe { out.assume_init() }
}

fn typed_pad<T: Copy + Clone + PoolScalar + TensorScalar>(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    input: &TypedTensor<T>,
    config: &PadConfig,
) -> crate::Result<TypedTensor<T>> {
    let input_shape = input.shape();
    let rank = input_shape.len();
    if config.edge_padding_low.len() != rank {
        return Err(crate::Error::rank_mismatch(
            "pad",
            rank,
            config.edge_padding_low.len(),
        ));
    }
    if config.edge_padding_high.len() != rank {
        return Err(crate::Error::rank_mismatch(
            "pad",
            rank,
            config.edge_padding_high.len(),
        ));
    }
    if config.interior_padding.len() != rank {
        return Err(crate::Error::rank_mismatch(
            "pad",
            rank,
            config.interior_padding.len(),
        ));
    }

    let mut out_shape = Vec::with_capacity(input_shape.len());
    for (axis, &input_extent) in input_shape.iter().enumerate() {
        if config.interior_padding[axis] < 0 {
            return Err(crate::Error::invalid_argument(
                "pad",
                "configuration",
                format!("interior padding must be non-negative on axis {axis}"),
            ));
        }
        let input_extent_i64 = i64::try_from(input_extent).map_err(|_| {
            crate::Error::invalid_argument(
                "pad",
                "configuration",
                format!("input extent on axis {axis} does not fit in i64"),
            )
        })?;
        let spacing = config.interior_padding[axis]
            .checked_add(1)
            .ok_or_else(|| {
                crate::Error::invalid_argument(
                    "pad",
                    "configuration",
                    format!("interior padding overflow on axis {axis}"),
                )
            })?;
        let base = if input_extent == 0 {
            0
        } else {
            input_extent_i64
                .checked_sub(1)
                .and_then(|extent| extent.checked_mul(spacing))
                .and_then(|extent| extent.checked_add(1))
                .ok_or_else(|| {
                    crate::Error::invalid_argument(
                        "pad",
                        "configuration",
                        format!("padded input extent overflow on axis {axis}"),
                    )
                })?
        };
        let dim = config.edge_padding_low[axis]
            .checked_add(config.edge_padding_high[axis])
            .and_then(|edge| edge.checked_add(base))
            .ok_or_else(|| {
                crate::Error::invalid_argument(
                    "pad",
                    "configuration",
                    format!("output dimension overflow on axis {axis}"),
                )
            })?;
        out_shape.push(usize::try_from(dim).map_err(|_| {
            crate::Error::invalid_argument(
                "pad",
                "configuration",
                format!("negative output dimension on axis {axis}"),
            )
        })?);
    }

    let dtype = kernel_dtype(T::dtype());
    let input_strides = inline_col_major_strides("pad", input_shape)?;
    let out_strides = inline_col_major_strides("pad", &out_shape)?;
    let plan = ErasedPadPlan::compile(
        dtype,
        input_shape,
        &input_strides,
        &out_shape,
        &out_strides,
        &config.edge_padding_low,
        &config.edge_padding_high,
        &config.interior_padding,
    )
    .map_err(|err| crate::Error::backend_source("pad", err))?;
    let fill = T::pool_zero();
    let mut out = PooledUninitOutput::new(buffers, out_shape.clone())?;
    let input_ref = unsafe {
        // SAFETY: this operation supplies initialized typed bytes with
        // alignment and dtype matching the descriptor; validated dimensions,
        // strides, and offset keep every reachable source element in bounds
        // for the retained borrow.
        crate::erased_raw_strided_ref(
            dtype,
            typed_bytes(typed_host_data("pad", input)?),
            input_shape,
            &input_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source("pad", err))?;
    let input_ptr = ErasedRawStridedPtr::from_ref(&input_ref);
    let mut out_ref = unsafe {
        // SAFETY: the output guard exclusively owns aligned storage whose
        // byte layout agrees with dtype; validated dimensions, strides, and
        // offset keep every destination in bounds, and the following kernel
        // overwrites every reachable element before typed exposure.
        crate::erased_raw_strided_uninit_mut(
            dtype,
            out.as_uninit_bytes_mut(),
            &out_shape,
            &out_strides,
            0,
        )
    }
    .map_err(|err| crate::Error::backend_source("pad", err))?;
    plan.execute_uninit(
        exec_context,
        &mut out_ref,
        &input_ptr,
        typed_bytes(std::slice::from_ref(&fill)),
    )
    .map_err(|err| crate::Error::backend_source("pad", err))?;

    // INVARIANT: the compact output has no unreachable storage, so successful
    // pad replay initializes every element.
    // SAFETY: every element contains a valid T after successful replay.
    unsafe { out.assume_init() }
}

#[cfg(test)]
mod tests;
