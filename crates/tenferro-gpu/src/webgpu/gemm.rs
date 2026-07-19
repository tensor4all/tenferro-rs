use cubecl::prelude::{CubeElement, CubePrimitive};
use cubecl_wgpu::WgpuRuntime;
use cubek_matmul::{
    definition::MatmulElems,
    launch::{launch_c32_ref, launch_ref, ComplexMatmulOptions, Strategy},
};
use cubek_std::InputBinding;
use num_complex::Complex32;
use smallvec::SmallVec;

use super::{
    alloc_output, comptime_sequence, cube_count_for_len, cube_dim_1d, ensure_resident_on_runtime,
    kernels, typed_tensor_binding_with_layout, unsupported_dtype, unsupported_operation,
    WebGpuBackend,
};
use crate::{col_major_strides, DotGeneralConfig, Error, ShapeMismatch, Tensor, TypedTensor};

const DOT_GENERAL_OP: &str = "webgpu_dot_general";

type AxisVec = SmallVec<[usize; 8]>;
type ShapeVec = SmallVec<[usize; 8]>;

#[derive(Clone, Debug)]
struct DotGeneralPlan {
    lhs_free: AxisVec,
    rhs_free: AxisVec,
    lhs_contract: AxisVec,
    rhs_contract: AxisVec,
    lhs_batch: AxisVec,
    rhs_batch: AxisVec,
    output_shape: Vec<usize>,
    lhs_cubek_shape: Vec<usize>,
    rhs_cubek_shape: Vec<usize>,
    out_cubek_shape: Vec<usize>,
    lhs_cubek_strides: Option<Vec<usize>>,
    rhs_cubek_strides: Option<Vec<usize>>,
    out_cubek_strides: Vec<usize>,
    k: usize,
}

struct PreparedOperand<T> {
    tensor: TypedTensor<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

fn dtype_mismatch(op: &'static str, lhs: &Tensor, rhs: &Tensor) -> Error {
    Error::dtype_mismatch(op, lhs.dtype(), rhs.dtype())
}

fn checked_product(label: &str, dims: &[usize]) -> crate::Result<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            Error::invalid_argument(
                DOT_GENERAL_OP,
                "shape",
                format!("{label} product overflow for dimensions {dims:?}"),
            )
        })
    })
}

fn axes_shape(shape: &[usize], axes: &[usize]) -> ShapeVec {
    axes.iter().map(|&axis| shape[axis]).collect()
}

fn free_axes(rank: usize, contracting: &[usize], batch: &[usize]) -> AxisVec {
    (0..rank)
        .filter(|axis| !contracting.contains(axis) && !batch.contains(axis))
        .collect()
}

fn dense_strides(shape: &[usize]) -> crate::Result<Vec<usize>> {
    col_major_strides(shape)?
        .into_iter()
        .map(|stride| {
            usize::try_from(stride).map_err(|_| {
                Error::Internal(format!(
                    "negative stride is not valid for dense shape {shape:?}"
                ))
            })
        })
        .collect()
}

fn flattened_group_stride(shape: &[usize], strides: &[usize], axes: &[usize]) -> Option<usize> {
    let Some((&first, rest)) = axes.split_first() else {
        return Some(1);
    };
    let mut previous = first;
    for &axis in rest {
        if axis != previous + 1 {
            return None;
        }
        if strides[axis] != strides[previous].checked_mul(shape[previous])? {
            return None;
        }
        previous = axis;
    }
    Some(strides[first])
}

fn input_cubek_strides(
    input_shape: &[usize],
    batch_axes: &[usize],
    row_axes: &[usize],
    col_axes: &[usize],
) -> crate::Result<Option<Vec<usize>>> {
    let dense = dense_strides(input_shape)?;
    let Some(row_stride) = flattened_group_stride(input_shape, &dense, row_axes) else {
        return Ok(None);
    };
    let Some(col_stride) = flattened_group_stride(input_shape, &dense, col_axes) else {
        return Ok(None);
    };

    let mut strides = Vec::with_capacity(batch_axes.len() + 2);
    strides.extend(batch_axes.iter().map(|&axis| dense[axis]));
    strides.push(row_stride);
    strides.push(col_stride);
    Ok(Some(strides))
}

fn dense_layout_strides(shape: &[usize]) -> crate::Result<Vec<usize>> {
    dense_strides(shape)
}

fn output_cubek_strides(m: usize, n: usize, batch_shape: &[usize]) -> crate::Result<Vec<usize>> {
    let mn = m.checked_mul(n).ok_or_else(|| {
        Error::invalid_argument(
            DOT_GENERAL_OP,
            "shape",
            format!("output matrix product overflow for M={m} N={n}"),
        )
    })?;
    let mut next_batch_stride = mn;
    let mut strides = Vec::with_capacity(batch_shape.len() + 2);
    for &dim in batch_shape {
        strides.push(next_batch_stride);
        next_batch_stride = next_batch_stride.checked_mul(dim).ok_or_else(|| {
            Error::invalid_argument(
                DOT_GENERAL_OP,
                "shape",
                format!("output batch stride overflow for batch shape {batch_shape:?}"),
            )
        })?;
    }
    strides.push(1);
    strides.push(m);
    Ok(strides)
}

fn cube_shape(batch_shape: &[usize], rows: usize, cols: usize) -> Vec<usize> {
    let mut shape = Vec::with_capacity(batch_shape.len() + 2);
    shape.extend_from_slice(batch_shape);
    shape.push(rows);
    shape.push(cols);
    shape
}

fn build_dot_general_plan(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    config: &DotGeneralConfig,
) -> crate::Result<DotGeneralPlan> {
    config.validate_dims_with_ranks(lhs_shape.len(), rhs_shape.len())?;

    let lhs_free = free_axes(
        lhs_shape.len(),
        &config.lhs_contracting_dims,
        &config.lhs_batch_dims,
    );
    let rhs_free = free_axes(
        rhs_shape.len(),
        &config.rhs_contracting_dims,
        &config.rhs_batch_dims,
    );
    let lhs_contract: AxisVec = config.lhs_contracting_dims.iter().copied().collect();
    let rhs_contract: AxisVec = config.rhs_contracting_dims.iter().copied().collect();
    let lhs_batch: AxisVec = config.lhs_batch_dims.iter().copied().collect();
    let rhs_batch: AxisVec = config.rhs_batch_dims.iter().copied().collect();

    for (&lhs_axis, &rhs_axis) in lhs_contract.iter().zip(&rhs_contract) {
        let lhs_dim = lhs_shape[lhs_axis];
        let rhs_dim = rhs_shape[rhs_axis];
        if lhs_dim != rhs_dim {
            return Err(Error::validation(
                DOT_GENERAL_OP,
                ShapeMismatch::ContractedDimensions {
                    lhs_axis,
                    lhs_size: lhs_dim,
                    rhs_axis,
                    rhs_size: rhs_dim,
                }
                .into(),
            ));
        }
    }
    for (&lhs_axis, &rhs_axis) in lhs_batch.iter().zip(&rhs_batch) {
        let lhs_dim = lhs_shape[lhs_axis];
        let rhs_dim = rhs_shape[rhs_axis];
        if lhs_dim != rhs_dim {
            return Err(Error::validation(
                DOT_GENERAL_OP,
                ShapeMismatch::IncompatibleShapes {
                    lhs: ShapeVec::from_vec(vec![lhs_dim]),
                    rhs: ShapeVec::from_vec(vec![rhs_dim]),
                }
                .into(),
            ));
        }
    }

    let lhs_free_shape = axes_shape(lhs_shape, &lhs_free);
    let rhs_free_shape = axes_shape(rhs_shape, &rhs_free);
    let contract_shape = axes_shape(lhs_shape, &lhs_contract);
    let batch_shape = axes_shape(lhs_shape, &lhs_batch);
    let m = checked_product("lhs free", &lhs_free_shape)?;
    let n = checked_product("rhs free", &rhs_free_shape)?;
    let k = checked_product("contracting", &contract_shape)?;

    let mut output_shape =
        Vec::with_capacity(lhs_free_shape.len() + rhs_free_shape.len() + batch_shape.len());
    output_shape.extend(lhs_free_shape.iter().copied());
    output_shape.extend(rhs_free_shape.iter().copied());
    output_shape.extend(batch_shape.iter().copied());

    let lhs_cubek_shape = cube_shape(&batch_shape, m, k);
    let rhs_cubek_shape = cube_shape(&batch_shape, k, n);
    let out_cubek_shape = cube_shape(&batch_shape, m, n);
    let lhs_cubek_strides = input_cubek_strides(lhs_shape, &lhs_batch, &lhs_free, &lhs_contract)?;
    let rhs_cubek_strides = input_cubek_strides(rhs_shape, &rhs_batch, &rhs_contract, &rhs_free)?;
    let out_cubek_strides = output_cubek_strides(m, n, &batch_shape)?;

    Ok(DotGeneralPlan {
        lhs_free,
        rhs_free,
        lhs_contract,
        rhs_contract,
        lhs_batch,
        rhs_batch,
        output_shape,
        lhs_cubek_shape,
        rhs_cubek_shape,
        out_cubek_shape,
        lhs_cubek_strides,
        rhs_cubek_strides,
        out_cubek_strides,
        k,
    })
}

fn pack_lhs_operand<T>(
    backend: &WebGpuBackend,
    input: &TypedTensor<T>,
    plan: &DotGeneralPlan,
) -> crate::Result<TypedTensor<T>>
where
    T: CubePrimitive + CubeElement + Clone + Send + Sync + 'static,
{
    let output = alloc_output::<T>(backend.runtime(), &plan.lhs_cubek_shape, DOT_GENERAL_OP)?;
    let len = output.n_elements();
    if len == 0 {
        return Ok(output);
    }

    let output_strides = dense_layout_strides(&plan.lhs_cubek_shape)?;
    let input_strides = dense_layout_strides(input.shape())?;
    let output_binding = typed_tensor_binding_with_layout(
        &output,
        &plan.lhs_cubek_shape,
        &output_strides,
        DOT_GENERAL_OP,
    )?;
    let input_binding =
        typed_tensor_binding_with_layout(input, input.shape(), &input_strides, DOT_GENERAL_OP)?;

    unsafe {
        // SAFETY: The planner validates that free, contracting, and batch axes
        // partition the input rank. Both tensor bindings cover their backing
        // allocations exactly, and the kernel writes one packed output element
        // per guarded launch position.
        kernels::pack_lhs_dot_general::launch_unchecked::<T, WgpuRuntime>(
            backend.runtime().client(),
            cube_count_for_len(len)?,
            cube_dim_1d(),
            output_binding.into_tensor_arg(),
            input_binding.into_tensor_arg(),
            comptime_sequence(&plan.lhs_free),
            comptime_sequence(&plan.lhs_contract),
            comptime_sequence(&plan.lhs_batch),
            input.shape().len(),
            plan.lhs_cubek_shape.len(),
        );
    }

    Ok(output)
}

fn pack_rhs_operand<T>(
    backend: &WebGpuBackend,
    input: &TypedTensor<T>,
    plan: &DotGeneralPlan,
) -> crate::Result<TypedTensor<T>>
where
    T: CubePrimitive + CubeElement + Clone + Send + Sync + 'static,
{
    let output = alloc_output::<T>(backend.runtime(), &plan.rhs_cubek_shape, DOT_GENERAL_OP)?;
    let len = output.n_elements();
    if len == 0 {
        return Ok(output);
    }

    let output_strides = dense_layout_strides(&plan.rhs_cubek_shape)?;
    let input_strides = dense_layout_strides(input.shape())?;
    let output_binding = typed_tensor_binding_with_layout(
        &output,
        &plan.rhs_cubek_shape,
        &output_strides,
        DOT_GENERAL_OP,
    )?;
    let input_binding =
        typed_tensor_binding_with_layout(input, input.shape(), &input_strides, DOT_GENERAL_OP)?;

    unsafe {
        // SAFETY: Same invariant as the lhs pack launch, with rhs axes mapped
        // to CubeK `[batch..., K, N]` order.
        kernels::pack_rhs_dot_general::launch_unchecked::<T, WgpuRuntime>(
            backend.runtime().client(),
            cube_count_for_len(len)?,
            cube_dim_1d(),
            output_binding.into_tensor_arg(),
            input_binding.into_tensor_arg(),
            comptime_sequence(&plan.rhs_contract),
            comptime_sequence(&plan.rhs_free),
            comptime_sequence(&plan.rhs_batch),
            input.shape().len(),
            plan.rhs_cubek_shape.len(),
        );
    }

    Ok(output)
}

fn prepare_lhs_operand<T>(
    backend: &WebGpuBackend,
    input: &TypedTensor<T>,
    plan: &DotGeneralPlan,
) -> crate::Result<PreparedOperand<T>>
where
    T: CubePrimitive + CubeElement + Clone + Send + Sync + 'static,
{
    let tensor = match &plan.lhs_cubek_strides {
        Some(strides) => {
            return Ok(PreparedOperand {
                tensor: input.clone(),
                shape: plan.lhs_cubek_shape.clone(),
                strides: strides.clone(),
            });
        }
        None => pack_lhs_operand(backend, input, plan)?,
    };
    Ok(PreparedOperand {
        tensor,
        shape: plan.lhs_cubek_shape.clone(),
        strides: dense_layout_strides(&plan.lhs_cubek_shape)?,
    })
}

fn prepare_rhs_operand<T>(
    backend: &WebGpuBackend,
    input: &TypedTensor<T>,
    plan: &DotGeneralPlan,
) -> crate::Result<PreparedOperand<T>>
where
    T: CubePrimitive + CubeElement + Clone + Send + Sync + 'static,
{
    let tensor = match &plan.rhs_cubek_strides {
        Some(strides) => {
            return Ok(PreparedOperand {
                tensor: input.clone(),
                shape: plan.rhs_cubek_shape.clone(),
                strides: strides.clone(),
            });
        }
        None => pack_rhs_operand(backend, input, plan)?,
    };
    Ok(PreparedOperand {
        tensor,
        shape: plan.rhs_cubek_shape.clone(),
        strides: dense_layout_strides(&plan.rhs_cubek_shape)?,
    })
}

fn dot_general_f32(
    backend: &WebGpuBackend,
    lhs: &TypedTensor<f32>,
    rhs: &TypedTensor<f32>,
    config: &DotGeneralConfig,
) -> crate::Result<TypedTensor<f32>> {
    let plan = build_dot_general_plan(lhs.shape(), rhs.shape(), config)?;
    ensure_resident_on_runtime(backend.runtime(), lhs, DOT_GENERAL_OP)?;
    ensure_resident_on_runtime(backend.runtime(), rhs, DOT_GENERAL_OP)?;

    let output = alloc_output::<f32>(backend.runtime(), &plan.output_shape, DOT_GENERAL_OP)?;
    if output.n_elements() == 0 {
        return Ok(output);
    }
    if plan.k == 0 {
        return Err(unsupported_operation(
            DOT_GENERAL_OP,
            "CubeK matmul does not support zero-sized contracting dimensions yet",
        ));
    }
    let lhs = prepare_lhs_operand(backend, lhs, &plan)?;
    let rhs = prepare_rhs_operand(backend, rhs, &plan)?;

    let dtype = f32::as_type_native_unchecked().storage_type();
    let lhs_binding = InputBinding::new(
        typed_tensor_binding_with_layout(&lhs.tensor, &lhs.shape, &lhs.strides, DOT_GENERAL_OP)?,
        dtype,
    );
    let rhs_binding = InputBinding::new(
        typed_tensor_binding_with_layout(&rhs.tensor, &rhs.shape, &rhs.strides, DOT_GENERAL_OP)?,
        dtype,
    );
    let output_binding = typed_tensor_binding_with_layout(
        &output,
        &plan.out_cubek_shape,
        &plan.out_cubek_strides,
        DOT_GENERAL_OP,
    )?;
    let mut dtypes = MatmulElems::from_single_dtype(f32::as_type_native_unchecked());

    launch_ref(
        &Strategy::Naive,
        backend.runtime().client(),
        lhs_binding,
        rhs_binding,
        output_binding,
        &mut dtypes,
    )
    .map_err(|err| Error::backend_failure(DOT_GENERAL_OP, err.to_string()))?;

    Ok(output)
}

fn dot_general_c32(
    backend: &WebGpuBackend,
    lhs: &TypedTensor<Complex32>,
    rhs: &TypedTensor<Complex32>,
    config: &DotGeneralConfig,
    lhs_conj: bool,
    rhs_conj: bool,
) -> crate::Result<TypedTensor<Complex32>> {
    let plan = build_dot_general_plan(lhs.shape(), rhs.shape(), config)?;
    ensure_resident_on_runtime(backend.runtime(), lhs, DOT_GENERAL_OP)?;
    ensure_resident_on_runtime(backend.runtime(), rhs, DOT_GENERAL_OP)?;

    let output = alloc_output::<Complex32>(backend.runtime(), &plan.output_shape, DOT_GENERAL_OP)?;
    if output.n_elements() == 0 {
        return Ok(output);
    }
    if plan.k == 0 {
        return Err(unsupported_operation(
            DOT_GENERAL_OP,
            "CubeK matmul does not support zero-sized contracting dimensions yet",
        ));
    }
    let lhs = prepare_lhs_operand(backend, lhs, &plan)?;
    let rhs = prepare_rhs_operand(backend, rhs, &plan)?;

    let dtype = Complex32::as_type_native_unchecked().storage_type();
    let lhs_binding = InputBinding::new(
        typed_tensor_binding_with_layout(&lhs.tensor, &lhs.shape, &lhs.strides, DOT_GENERAL_OP)?,
        dtype,
    );
    let rhs_binding = InputBinding::new(
        typed_tensor_binding_with_layout(&rhs.tensor, &rhs.shape, &rhs.strides, DOT_GENERAL_OP)?,
        dtype,
    );
    let output_binding = typed_tensor_binding_with_layout(
        &output,
        &plan.out_cubek_shape,
        &plan.out_cubek_strides,
        DOT_GENERAL_OP,
    )?;
    let mut dtypes = MatmulElems::from_single_dtype(Complex32::as_type_native_unchecked());

    launch_c32_ref(
        &Strategy::Naive,
        backend.runtime().client(),
        lhs_binding,
        rhs_binding,
        output_binding,
        &mut dtypes,
        ComplexMatmulOptions { lhs_conj, rhs_conj },
    )
    .map_err(|err| Error::backend_failure(DOT_GENERAL_OP, err.to_string()))?;

    Ok(output)
}

pub(super) fn dot_general_with_conj(
    backend: &WebGpuBackend,
    lhs: &Tensor,
    rhs: &Tensor,
    config: &DotGeneralConfig,
    lhs_conj: bool,
    rhs_conj: bool,
) -> crate::Result<Tensor> {
    match (lhs, rhs) {
        (Tensor::F32(lhs), Tensor::F32(rhs)) => {
            dot_general_f32(backend, lhs, rhs, config).map(Tensor::F32)
        }
        (Tensor::F64(_), Tensor::F64(_)) => {
            Err(unsupported_dtype(DOT_GENERAL_OP, crate::DType::F64))
        }
        (Tensor::C32(lhs), Tensor::C32(rhs)) => {
            dot_general_c32(backend, lhs, rhs, config, lhs_conj, rhs_conj).map(Tensor::C32)
        }
        (Tensor::C64(_), Tensor::C64(_)) => {
            Err(unsupported_dtype(DOT_GENERAL_OP, crate::DType::C64))
        }
        _ => Err(dtype_mismatch(DOT_GENERAL_OP, lhs, rhs)),
    }
}

pub(super) fn dot_general(
    backend: &WebGpuBackend,
    lhs: &Tensor,
    rhs: &Tensor,
    config: &DotGeneralConfig,
) -> crate::Result<Tensor> {
    dot_general_with_conj(backend, lhs, rhs, config, false, false)
}
