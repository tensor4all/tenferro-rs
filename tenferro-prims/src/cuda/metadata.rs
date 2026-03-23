use tenferro_algebra::Scalar;
use tenferro_device::{Error, LogicalMemorySpace, Result};
use tenferro_tensor::Tensor;

use crate::{
    validate_rank, validate_shape_eq, CudaBackend, CudaContext, MetadataBinaryOp, MetadataDType,
    MetadataGenerateOp, MetadataPrimsDescriptor, MetadataReductionOp, MetadataTensorMut,
    MetadataTensorRef, MetadataTernaryOp, TensorMetadataPrims,
};

fn ensure_cuda_tensor<T: Scalar>(tensor: &Tensor<T>, device_id: usize, label: &str) -> Result<()> {
    match tensor.logical_memory_space() {
        LogicalMemorySpace::GpuMemory {
            device_id: tensor_device,
        } if tensor_device == device_id => Ok(()),
        LogicalMemorySpace::GpuMemory {
            device_id: tensor_device,
        } => Err(Error::DeviceError(format!(
            "{label} is on CUDA device {tensor_device}, expected device {device_id}"
        ))),
        other => Err(Error::DeviceError(format!(
            "{label} is not resident on CUDA device {device_id}: {other:?}"
        ))),
    }
}

fn tensor_dims_ref<'a>(tensor: &'a MetadataTensorRef<'a>) -> &'a [usize] {
    match tensor {
        MetadataTensorRef::I32(tensor) => tensor.dims(),
        MetadataTensorRef::Bool(tensor) => tensor.dims(),
    }
}

fn tensor_dims_mut<'a>(tensor: &'a MetadataTensorMut<'a>) -> &'a [usize] {
    match tensor {
        MetadataTensorMut::I32(tensor) => tensor.dims(),
        MetadataTensorMut::Bool(tensor) => tensor.dims(),
    }
}

fn tensor_device_ptr_ref<T: Scalar>(tensor: &Tensor<T>, label: &str) -> Result<*const T> {
    tensor
        .buffer()
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError(format!("{label} buffer is not on GPU")))
}

fn tensor_device_ptr_mut<T: Scalar>(tensor: &mut Tensor<T>, label: &str) -> Result<*mut T> {
    tensor
        .buffer()
        .as_device_ptr()
        .map(|ptr| ptr as *mut T)
        .ok_or_else(|| Error::DeviceError(format!("{label} buffer is not on GPU")))
}

fn required_storage_len(
    dims: &[usize],
    strides: &[isize],
    offset: isize,
    label: &str,
) -> Result<usize> {
    if dims.len() != strides.len() {
        return Err(Error::InvalidArgument(format!(
            "{label} rank mismatch: dims={} strides={}",
            dims.len(),
            strides.len()
        )));
    }
    if dims.contains(&0) {
        return Ok(0);
    }

    let mut min_pos = offset;
    let mut max_pos = offset;
    for (axis, (&dim, &stride)) in dims.iter().zip(strides).enumerate() {
        let extent = isize::try_from(dim - 1)
            .ok()
            .and_then(|d| d.checked_mul(stride))
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "{label} extent overflow for dimension {axis} (size={dim}, stride={stride})"
                ))
            })?;
        if extent >= 0 {
            max_pos = max_pos.checked_add(extent).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "{label} maximum offset overflow for dimension {axis}"
                ))
            })?;
        } else {
            min_pos = min_pos.checked_add(extent).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "{label} minimum offset overflow for dimension {axis}"
                ))
            })?;
        }
    }

    if min_pos < 0 {
        return Err(Error::InvalidArgument(format!(
            "{label} accesses negative buffer positions {}..={}",
            min_pos, max_pos
        )));
    }

    let max_pos = usize::try_from(max_pos).map_err(|_| {
        Error::InvalidArgument(format!(
            "{label} maximum position {max_pos} exceeds usize range"
        ))
    })?;
    max_pos
        .checked_add(1)
        .ok_or_else(|| Error::InvalidArgument(format!("{label} storage length overflow")))
}

fn validate_storage_len(
    actual_len: usize,
    dims: &[usize],
    strides: &[isize],
    offset: isize,
    label: &str,
) -> Result<()> {
    let required = required_storage_len(dims, strides, offset, label)?;
    if actual_len < required {
        return Err(Error::InvalidArgument(format!(
            "{label} length mismatch: actual={actual_len} required={required}"
        )));
    }
    Ok(())
}

fn validate_supported_generate(output_dtype: MetadataDType) -> Result<()> {
    if output_dtype == MetadataDType::I32 {
        Ok(())
    } else {
        Err(Error::InvalidArgument(
            "metadata iota currently supports I32 output only".into(),
        ))
    }
}

fn validate_supported_binary(
    op: MetadataBinaryOp,
    lhs_dtype: MetadataDType,
    rhs_dtype: MetadataDType,
    output_dtype: MetadataDType,
) -> Result<()> {
    if !matches!(op, MetadataBinaryOp::Equal | MetadataBinaryOp::NotEqual) {
        return Err(Error::InvalidArgument(format!(
            "metadata binary operation {op:?} is not supported on CudaBackend"
        )));
    }
    match (lhs_dtype, rhs_dtype, output_dtype) {
        (MetadataDType::I32, MetadataDType::I32, MetadataDType::Bool)
        | (MetadataDType::Bool, MetadataDType::Bool, MetadataDType::Bool) => Ok(()),
        _ => Err(Error::InvalidArgument(format!(
            "unsupported metadata binary dtype combination: lhs={lhs_dtype:?} rhs={rhs_dtype:?} dst={output_dtype:?}"
        ))),
    }
}

fn validate_supported_ternary(
    op: MetadataTernaryOp,
    cond_dtype: MetadataDType,
    lhs_dtype: MetadataDType,
    rhs_dtype: MetadataDType,
    output_dtype: MetadataDType,
) -> Result<()> {
    if !matches!(op, MetadataTernaryOp::Where) {
        return Err(Error::InvalidArgument(format!(
            "metadata ternary operation {op:?} is not supported on CudaBackend"
        )));
    }
    match (cond_dtype, lhs_dtype, rhs_dtype, output_dtype) {
        (MetadataDType::Bool, MetadataDType::I32, MetadataDType::I32, MetadataDType::I32)
        | (MetadataDType::Bool, MetadataDType::Bool, MetadataDType::Bool, MetadataDType::Bool) =>
        {
            Ok(())
        }
        _ => Err(Error::InvalidArgument(format!(
            "unsupported metadata ternary dtype combination: cond={cond_dtype:?} lhs={lhs_dtype:?} rhs={rhs_dtype:?} dst={output_dtype:?}"
        ))),
    }
}

fn validate_supported_reduction(
    op: MetadataReductionOp,
    input_dtype: MetadataDType,
    output_dtype: MetadataDType,
) -> Result<()> {
    match (op, input_dtype, output_dtype) {
        (MetadataReductionOp::Sum, MetadataDType::Bool, MetadataDType::I32)
        | (MetadataReductionOp::Sum, MetadataDType::I32, MetadataDType::I32)
        | (MetadataReductionOp::All, MetadataDType::Bool, MetadataDType::Bool)
        | (MetadataReductionOp::Any, MetadataDType::Bool, MetadataDType::Bool) => Ok(()),
        _ => Err(Error::InvalidArgument(format!(
            "unsupported metadata reduction dtype combination: op={op:?} input={input_dtype:?} dst={output_dtype:?}"
        ))),
    }
}

fn validate_metadata_handle_count(
    inputs: &[MetadataTensorRef<'_>],
    expected: usize,
    op_name: &str,
) -> Result<()> {
    if inputs.len() != expected {
        return Err(Error::InvalidArgument(format!(
            "{op_name} expects {expected} input(s) (got {})",
            inputs.len()
        )));
    }
    Ok(())
}

fn plan_metadata_reduction(
    desc: &MetadataPrimsDescriptor,
    input_dims: &[usize],
    output_dims: &[usize],
) -> Result<(Vec<usize>, Vec<usize>)> {
    let MetadataPrimsDescriptor::Reduction {
        modes_a, modes_c, ..
    } = desc
    else {
        return Err(Error::InvalidArgument(
            "expected metadata reduction descriptor".into(),
        ));
    };
    validate_rank(input_dims, modes_a.len(), "CudaMetadataReduction input")?;
    validate_rank(output_dims, modes_c.len(), "CudaMetadataReduction output")?;

    let mut expected_output = Vec::with_capacity(modes_c.len());
    for &mode in modes_c {
        let Some(axis) = modes_a.iter().position(|&candidate| candidate == mode) else {
            return Err(Error::InvalidArgument(format!(
                "CudaMetadataReduction: output mode {mode} not found in input modes {modes_a:?}"
            )));
        };
        expected_output.push(input_dims[axis]);
    }
    validate_shape_eq(
        output_dims,
        &expected_output,
        "CudaMetadataReduction output",
    )?;

    let kept_axes = modes_c
        .iter()
        .map(|mode| {
            modes_a
                .iter()
                .position(|&candidate| candidate == *mode)
                .ok_or_else(|| {
                    Error::InvalidArgument(format!(
                    "CudaMetadataReduction: output mode {mode} not found in input modes {modes_a:?}"
                ))
                })
        })
        .collect::<Result<Vec<_>>>()?;
    let reduced_axes: Vec<usize> = modes_a
        .iter()
        .enumerate()
        .filter(|(_, mode)| !modes_c.contains(mode))
        .map(|(idx, _)| idx)
        .collect();
    Ok((kept_axes, reduced_axes))
}

impl TensorMetadataPrims for CudaBackend {
    type Plan = MetadataPrimsDescriptor;
    type Context = CudaContext;

    fn plan(
        ctx: &mut Self::Context,
        desc: &MetadataPrimsDescriptor,
        inputs: &[MetadataTensorRef<'_>],
        output: MetadataTensorMut<'_>,
    ) -> Result<Self::Plan> {
        match desc {
            MetadataPrimsDescriptor::Generate { op, output_dtype } => {
                validate_supported_generate(*output_dtype)?;
                validate_metadata_handle_count(inputs, 0, "CudaMetadataGenerate")?;
                if !matches!(*op, MetadataGenerateOp::IotaStartZero) {
                    return Err(Error::InvalidArgument(format!(
                        "metadata generate operation {op:?} is not supported on CudaBackend"
                    )));
                }
                match output {
                    MetadataTensorMut::I32(tensor) => {
                        ensure_cuda_tensor(tensor, ctx.device_id(), "CudaMetadataGenerate output")?;
                        validate_storage_len(
                            tensor.buffer().len(),
                            tensor.dims(),
                            tensor.strides(),
                            tensor.offset(),
                            "CudaMetadataGenerate output",
                        )?;
                    }
                    MetadataTensorMut::Bool(_) => unreachable!(),
                }
                Ok(desc.clone())
            }
            MetadataPrimsDescriptor::Binary {
                op,
                lhs_dtype,
                rhs_dtype,
                output_dtype,
            } => {
                validate_supported_binary(*op, *lhs_dtype, *rhs_dtype, *output_dtype)?;
                validate_metadata_handle_count(inputs, 2, "CudaMetadataBinary")?;
                validate_shape_eq(
                    tensor_dims_ref(&inputs[0]),
                    tensor_dims_ref(&inputs[1]),
                    "CudaMetadataBinary input",
                )?;
                validate_shape_eq(
                    tensor_dims_ref(&inputs[0]),
                    tensor_dims_mut(&output),
                    "CudaMetadataBinary output",
                )?;
                match (inputs[0], inputs[1], output) {
                    (
                        MetadataTensorRef::I32(lhs),
                        MetadataTensorRef::I32(rhs),
                        MetadataTensorMut::Bool(dst),
                    ) => {
                        ensure_cuda_tensor(lhs, ctx.device_id(), "CudaMetadataBinary lhs")?;
                        ensure_cuda_tensor(rhs, ctx.device_id(), "CudaMetadataBinary rhs")?;
                        ensure_cuda_tensor(dst, ctx.device_id(), "CudaMetadataBinary dst")?;
                        validate_storage_len(
                            lhs.buffer().len(),
                            lhs.dims(),
                            lhs.strides(),
                            lhs.offset(),
                            "CudaMetadataBinary lhs",
                        )?;
                        validate_storage_len(
                            rhs.buffer().len(),
                            rhs.dims(),
                            rhs.strides(),
                            rhs.offset(),
                            "CudaMetadataBinary rhs",
                        )?;
                        validate_storage_len(
                            dst.buffer().len(),
                            dst.dims(),
                            dst.strides(),
                            dst.offset(),
                            "CudaMetadataBinary dst",
                        )?;
                    }
                    (
                        MetadataTensorRef::Bool(lhs),
                        MetadataTensorRef::Bool(rhs),
                        MetadataTensorMut::Bool(dst),
                    ) => {
                        ensure_cuda_tensor(lhs, ctx.device_id(), "CudaMetadataBinary lhs")?;
                        ensure_cuda_tensor(rhs, ctx.device_id(), "CudaMetadataBinary rhs")?;
                        ensure_cuda_tensor(dst, ctx.device_id(), "CudaMetadataBinary dst")?;
                        validate_storage_len(
                            lhs.buffer().len(),
                            lhs.dims(),
                            lhs.strides(),
                            lhs.offset(),
                            "CudaMetadataBinary lhs",
                        )?;
                        validate_storage_len(
                            rhs.buffer().len(),
                            rhs.dims(),
                            rhs.strides(),
                            rhs.offset(),
                            "CudaMetadataBinary rhs",
                        )?;
                        validate_storage_len(
                            dst.buffer().len(),
                            dst.dims(),
                            dst.strides(),
                            dst.offset(),
                            "CudaMetadataBinary dst",
                        )?;
                    }
                    _ => {
                        return Err(Error::InvalidArgument(
                            "unsupported metadata binary execution dtype combination".into(),
                        ));
                    }
                }
                Ok(desc.clone())
            }
            MetadataPrimsDescriptor::Ternary {
                op,
                cond_dtype,
                lhs_dtype,
                rhs_dtype,
                output_dtype,
            } => {
                validate_supported_ternary(
                    *op,
                    *cond_dtype,
                    *lhs_dtype,
                    *rhs_dtype,
                    *output_dtype,
                )?;
                validate_metadata_handle_count(inputs, 3, "CudaMetadataTernary")?;
                validate_shape_eq(
                    tensor_dims_ref(&inputs[0]),
                    tensor_dims_ref(&inputs[1]),
                    "CudaMetadataTernary input",
                )?;
                validate_shape_eq(
                    tensor_dims_ref(&inputs[0]),
                    tensor_dims_ref(&inputs[2]),
                    "CudaMetadataTernary input",
                )?;
                validate_shape_eq(
                    tensor_dims_ref(&inputs[0]),
                    tensor_dims_mut(&output),
                    "CudaMetadataTernary output",
                )?;
                match (inputs[0], inputs[1], inputs[2], output) {
                    (
                        MetadataTensorRef::Bool(cond),
                        MetadataTensorRef::I32(on_true),
                        MetadataTensorRef::I32(on_false),
                        MetadataTensorMut::I32(dst),
                    ) => {
                        ensure_cuda_tensor(cond, ctx.device_id(), "CudaMetadataTernary cond")?;
                        ensure_cuda_tensor(on_true, ctx.device_id(), "CudaMetadataTernary true")?;
                        ensure_cuda_tensor(on_false, ctx.device_id(), "CudaMetadataTernary false")?;
                        ensure_cuda_tensor(dst, ctx.device_id(), "CudaMetadataTernary dst")?;
                        validate_storage_len(
                            cond.buffer().len(),
                            cond.dims(),
                            cond.strides(),
                            cond.offset(),
                            "CudaMetadataTernary cond",
                        )?;
                        validate_storage_len(
                            on_true.buffer().len(),
                            on_true.dims(),
                            on_true.strides(),
                            on_true.offset(),
                            "CudaMetadataTernary true",
                        )?;
                        validate_storage_len(
                            on_false.buffer().len(),
                            on_false.dims(),
                            on_false.strides(),
                            on_false.offset(),
                            "CudaMetadataTernary false",
                        )?;
                        validate_storage_len(
                            dst.buffer().len(),
                            dst.dims(),
                            dst.strides(),
                            dst.offset(),
                            "CudaMetadataTernary dst",
                        )?;
                    }
                    (
                        MetadataTensorRef::Bool(cond),
                        MetadataTensorRef::Bool(on_true),
                        MetadataTensorRef::Bool(on_false),
                        MetadataTensorMut::Bool(dst),
                    ) => {
                        ensure_cuda_tensor(cond, ctx.device_id(), "CudaMetadataTernary cond")?;
                        ensure_cuda_tensor(on_true, ctx.device_id(), "CudaMetadataTernary true")?;
                        ensure_cuda_tensor(on_false, ctx.device_id(), "CudaMetadataTernary false")?;
                        ensure_cuda_tensor(dst, ctx.device_id(), "CudaMetadataTernary dst")?;
                        validate_storage_len(
                            cond.buffer().len(),
                            cond.dims(),
                            cond.strides(),
                            cond.offset(),
                            "CudaMetadataTernary cond",
                        )?;
                        validate_storage_len(
                            on_true.buffer().len(),
                            on_true.dims(),
                            on_true.strides(),
                            on_true.offset(),
                            "CudaMetadataTernary true",
                        )?;
                        validate_storage_len(
                            on_false.buffer().len(),
                            on_false.dims(),
                            on_false.strides(),
                            on_false.offset(),
                            "CudaMetadataTernary false",
                        )?;
                        validate_storage_len(
                            dst.buffer().len(),
                            dst.dims(),
                            dst.strides(),
                            dst.offset(),
                            "CudaMetadataTernary dst",
                        )?;
                    }
                    _ => {
                        return Err(Error::InvalidArgument(
                            "unsupported metadata ternary execution dtype combination".into(),
                        ));
                    }
                }
                Ok(desc.clone())
            }
            MetadataPrimsDescriptor::Reduction {
                input_dtype,
                output_dtype,
                op,
                ..
            } => {
                validate_supported_reduction(*op, *input_dtype, *output_dtype)?;
                validate_metadata_handle_count(inputs, 1, "CudaMetadataReduction")?;
                let input_dims = tensor_dims_ref(&inputs[0]).to_vec();
                let output_dims = tensor_dims_mut(&output).to_vec();
                let _ = plan_metadata_reduction(desc, &input_dims, &output_dims)?;
                match (inputs[0], output) {
                    (MetadataTensorRef::I32(input), MetadataTensorMut::I32(dst)) => {
                        ensure_cuda_tensor(input, ctx.device_id(), "CudaMetadataReduction input")?;
                        ensure_cuda_tensor(dst, ctx.device_id(), "CudaMetadataReduction dst")?;
                        validate_storage_len(
                            input.buffer().len(),
                            input.dims(),
                            input.strides(),
                            input.offset(),
                            "CudaMetadataReduction input",
                        )?;
                        validate_storage_len(
                            dst.buffer().len(),
                            dst.dims(),
                            dst.strides(),
                            dst.offset(),
                            "CudaMetadataReduction dst",
                        )?;
                    }
                    (MetadataTensorRef::Bool(input), MetadataTensorMut::I32(dst)) => {
                        ensure_cuda_tensor(input, ctx.device_id(), "CudaMetadataReduction input")?;
                        ensure_cuda_tensor(dst, ctx.device_id(), "CudaMetadataReduction dst")?;
                        validate_storage_len(
                            input.buffer().len(),
                            input.dims(),
                            input.strides(),
                            input.offset(),
                            "CudaMetadataReduction input",
                        )?;
                        validate_storage_len(
                            dst.buffer().len(),
                            dst.dims(),
                            dst.strides(),
                            dst.offset(),
                            "CudaMetadataReduction dst",
                        )?;
                    }
                    (MetadataTensorRef::Bool(input), MetadataTensorMut::Bool(dst)) => {
                        ensure_cuda_tensor(input, ctx.device_id(), "CudaMetadataReduction input")?;
                        ensure_cuda_tensor(dst, ctx.device_id(), "CudaMetadataReduction dst")?;
                        validate_storage_len(
                            input.buffer().len(),
                            input.dims(),
                            input.strides(),
                            input.offset(),
                            "CudaMetadataReduction input",
                        )?;
                        validate_storage_len(
                            dst.buffer().len(),
                            dst.dims(),
                            dst.strides(),
                            dst.offset(),
                            "CudaMetadataReduction dst",
                        )?;
                    }
                    _ => {
                        return Err(Error::InvalidArgument(
                            "unsupported metadata reduction execution dtype combination".into(),
                        ));
                    }
                }
                Ok(desc.clone())
            }
        }
    }

    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        inputs: &[MetadataTensorRef<'_>],
        output: MetadataTensorMut<'_>,
    ) -> Result<()> {
        let runtime = ctx.shared_runtime();
        match plan {
            MetadataPrimsDescriptor::Generate {
                op: MetadataGenerateOp::IotaStartZero,
                output_dtype: MetadataDType::I32,
            } => match output {
                MetadataTensorMut::I32(dst) => {
                    let dst_len = dst.buffer().len();
                    let dst_ptr = tensor_device_ptr_mut(dst, "CudaMetadataGenerate output")?;
                    unsafe {
                        runtime.metadata_generate_iota_i32(
                            dst_ptr,
                            dst_len,
                            dst.dims(),
                            dst.strides(),
                            dst.offset(),
                        )?;
                    }
                    Ok(())
                }
                MetadataTensorMut::Bool(_) => Err(Error::InvalidArgument(
                    "metadata iota currently supports I32 output only".into(),
                )),
            },
            MetadataPrimsDescriptor::Binary {
                op,
                lhs_dtype,
                rhs_dtype,
                output_dtype,
            } => {
                let equal = matches!(op, MetadataBinaryOp::Equal);
                match (
                    *lhs_dtype,
                    *rhs_dtype,
                    *output_dtype,
                    inputs[0],
                    inputs[1],
                    output,
                ) {
                    (
                        MetadataDType::I32,
                        MetadataDType::I32,
                        MetadataDType::Bool,
                        MetadataTensorRef::I32(lhs),
                        MetadataTensorRef::I32(rhs),
                        MetadataTensorMut::Bool(dst),
                    ) => {
                        let lhs_len = lhs.buffer().len();
                        let rhs_len = rhs.buffer().len();
                        let dst_len = dst.buffer().len();
                        let lhs_ptr = tensor_device_ptr_ref(lhs, "CudaMetadataBinary lhs")?;
                        let rhs_ptr = tensor_device_ptr_ref(rhs, "CudaMetadataBinary rhs")?;
                        let dst_ptr = tensor_device_ptr_mut(dst, "CudaMetadataBinary dst")?;
                        unsafe {
                            runtime.metadata_binary_i32_bool(
                                equal,
                                lhs_ptr,
                                lhs_len,
                                rhs_ptr,
                                rhs_len,
                                dst_ptr,
                                dst_len,
                                lhs.dims(),
                                lhs.strides(),
                                lhs.offset(),
                                rhs.strides(),
                                rhs.offset(),
                                dst.strides(),
                                dst.offset(),
                            )?;
                        }
                        Ok(())
                    }
                    (
                        MetadataDType::Bool,
                        MetadataDType::Bool,
                        MetadataDType::Bool,
                        MetadataTensorRef::Bool(lhs),
                        MetadataTensorRef::Bool(rhs),
                        MetadataTensorMut::Bool(dst),
                    ) => {
                        let lhs_len = lhs.buffer().len();
                        let rhs_len = rhs.buffer().len();
                        let dst_len = dst.buffer().len();
                        let lhs_ptr = tensor_device_ptr_ref(lhs, "CudaMetadataBinary lhs")?;
                        let rhs_ptr = tensor_device_ptr_ref(rhs, "CudaMetadataBinary rhs")?;
                        let dst_ptr = tensor_device_ptr_mut(dst, "CudaMetadataBinary dst")?;
                        unsafe {
                            runtime.metadata_binary_bool_bool(
                                equal,
                                lhs_ptr,
                                lhs_len,
                                rhs_ptr,
                                rhs_len,
                                dst_ptr,
                                dst_len,
                                lhs.dims(),
                                lhs.strides(),
                                lhs.offset(),
                                rhs.strides(),
                                rhs.offset(),
                                dst.strides(),
                                dst.offset(),
                            )?;
                        }
                        Ok(())
                    }
                    _ => Err(Error::InvalidArgument(
                        "unsupported metadata binary execution dtype combination".into(),
                    )),
                }
            }
            MetadataPrimsDescriptor::Ternary {
                op: MetadataTernaryOp::Where,
                cond_dtype,
                lhs_dtype,
                rhs_dtype,
                output_dtype,
            } => match (
                *cond_dtype,
                *lhs_dtype,
                *rhs_dtype,
                *output_dtype,
                inputs[0],
                inputs[1],
                inputs[2],
                output,
            ) {
                (
                    MetadataDType::Bool,
                    MetadataDType::I32,
                    MetadataDType::I32,
                    MetadataDType::I32,
                    MetadataTensorRef::Bool(cond),
                    MetadataTensorRef::I32(on_true),
                    MetadataTensorRef::I32(on_false),
                    MetadataTensorMut::I32(dst),
                ) => {
                    let cond_len = cond.buffer().len();
                    let true_len = on_true.buffer().len();
                    let false_len = on_false.buffer().len();
                    let dst_len = dst.buffer().len();
                    let cond_ptr = tensor_device_ptr_ref(cond, "CudaMetadataTernary cond")?;
                    let true_ptr = tensor_device_ptr_ref(on_true, "CudaMetadataTernary true")?;
                    let false_ptr = tensor_device_ptr_ref(on_false, "CudaMetadataTernary false")?;
                    let dst_ptr = tensor_device_ptr_mut(dst, "CudaMetadataTernary dst")?;
                    unsafe {
                        runtime.metadata_where_i32(
                            cond_ptr,
                            cond_len,
                            true_ptr,
                            true_len,
                            false_ptr,
                            false_len,
                            dst_ptr,
                            dst_len,
                            cond.dims(),
                            cond.strides(),
                            cond.offset(),
                            on_true.strides(),
                            on_true.offset(),
                            on_false.strides(),
                            on_false.offset(),
                            dst.strides(),
                            dst.offset(),
                        )?;
                    }
                    Ok(())
                }
                (
                    MetadataDType::Bool,
                    MetadataDType::Bool,
                    MetadataDType::Bool,
                    MetadataDType::Bool,
                    MetadataTensorRef::Bool(cond),
                    MetadataTensorRef::Bool(on_true),
                    MetadataTensorRef::Bool(on_false),
                    MetadataTensorMut::Bool(dst),
                ) => {
                    let cond_len = cond.buffer().len();
                    let true_len = on_true.buffer().len();
                    let false_len = on_false.buffer().len();
                    let dst_len = dst.buffer().len();
                    let cond_ptr = tensor_device_ptr_ref(cond, "CudaMetadataTernary cond")?;
                    let true_ptr = tensor_device_ptr_ref(on_true, "CudaMetadataTernary true")?;
                    let false_ptr = tensor_device_ptr_ref(on_false, "CudaMetadataTernary false")?;
                    let dst_ptr = tensor_device_ptr_mut(dst, "CudaMetadataTernary dst")?;
                    unsafe {
                        runtime.metadata_where_bool(
                            cond_ptr,
                            cond_len,
                            true_ptr,
                            true_len,
                            false_ptr,
                            false_len,
                            dst_ptr,
                            dst_len,
                            cond.dims(),
                            cond.strides(),
                            cond.offset(),
                            on_true.strides(),
                            on_true.offset(),
                            on_false.strides(),
                            on_false.offset(),
                            dst.strides(),
                            dst.offset(),
                        )?;
                    }
                    Ok(())
                }
                _ => Err(Error::InvalidArgument(
                    "unsupported metadata ternary execution dtype combination".into(),
                )),
            },
            MetadataPrimsDescriptor::Reduction {
                op,
                input_dtype,
                output_dtype,
                ..
            } => {
                let input_dims = tensor_dims_ref(&inputs[0]).to_vec();
                let output_dims = tensor_dims_mut(&output).to_vec();
                let (kept_axes, reduced_axes) =
                    plan_metadata_reduction(plan, &input_dims, &output_dims)?;
                match (*op, *input_dtype, *output_dtype, inputs[0], output) {
                    (
                        MetadataReductionOp::Sum,
                        MetadataDType::I32,
                        MetadataDType::I32,
                        MetadataTensorRef::I32(input),
                        MetadataTensorMut::I32(dst),
                    ) => {
                        let input_len = input.buffer().len();
                        let dst_len = dst.buffer().len();
                        let input_ptr =
                            tensor_device_ptr_ref(input, "CudaMetadataReduction input")?;
                        let dst_ptr = tensor_device_ptr_mut(dst, "CudaMetadataReduction dst")?;
                        unsafe {
                            runtime.metadata_reduce_sum_i32(
                                input_ptr,
                                input_len,
                                dst_ptr,
                                dst_len,
                                input.dims(),
                                input.strides(),
                                input.offset(),
                                dst.dims(),
                                dst.strides(),
                                dst.offset(),
                                &kept_axes,
                                &reduced_axes,
                            )?;
                        }
                        Ok(())
                    }
                    (
                        MetadataReductionOp::Sum,
                        MetadataDType::Bool,
                        MetadataDType::I32,
                        MetadataTensorRef::Bool(input),
                        MetadataTensorMut::I32(dst),
                    ) => {
                        let input_len = input.buffer().len();
                        let dst_len = dst.buffer().len();
                        let input_ptr =
                            tensor_device_ptr_ref(input, "CudaMetadataReduction input")?;
                        let dst_ptr = tensor_device_ptr_mut(dst, "CudaMetadataReduction dst")?;
                        unsafe {
                            runtime.metadata_reduce_sum_bool(
                                input_ptr,
                                input_len,
                                dst_ptr,
                                dst_len,
                                input.dims(),
                                input.strides(),
                                input.offset(),
                                dst.dims(),
                                dst.strides(),
                                dst.offset(),
                                &kept_axes,
                                &reduced_axes,
                            )?;
                        }
                        Ok(())
                    }
                    (
                        MetadataReductionOp::All,
                        MetadataDType::Bool,
                        MetadataDType::Bool,
                        MetadataTensorRef::Bool(input),
                        MetadataTensorMut::Bool(dst),
                    ) => {
                        let input_len = input.buffer().len();
                        let dst_len = dst.buffer().len();
                        let input_ptr =
                            tensor_device_ptr_ref(input, "CudaMetadataReduction input")?;
                        let dst_ptr = tensor_device_ptr_mut(dst, "CudaMetadataReduction dst")?;
                        unsafe {
                            runtime.metadata_reduce_all_bool(
                                input_ptr,
                                input_len,
                                dst_ptr,
                                dst_len,
                                input.dims(),
                                input.strides(),
                                input.offset(),
                                dst.dims(),
                                dst.strides(),
                                dst.offset(),
                                &kept_axes,
                                &reduced_axes,
                            )?;
                        }
                        Ok(())
                    }
                    (
                        MetadataReductionOp::Any,
                        MetadataDType::Bool,
                        MetadataDType::Bool,
                        MetadataTensorRef::Bool(input),
                        MetadataTensorMut::Bool(dst),
                    ) => {
                        let input_len = input.buffer().len();
                        let dst_len = dst.buffer().len();
                        let input_ptr =
                            tensor_device_ptr_ref(input, "CudaMetadataReduction input")?;
                        let dst_ptr = tensor_device_ptr_mut(dst, "CudaMetadataReduction dst")?;
                        unsafe {
                            runtime.metadata_reduce_any_bool(
                                input_ptr,
                                input_len,
                                dst_ptr,
                                dst_len,
                                input.dims(),
                                input.strides(),
                                input.offset(),
                                dst.dims(),
                                dst.strides(),
                                dst.offset(),
                                &kept_axes,
                                &reduced_axes,
                            )?;
                        }
                        Ok(())
                    }
                    _ => Err(Error::InvalidArgument(
                        "unsupported metadata reduction execution dtype combination".into(),
                    )),
                }
            }
            _ => Err(Error::InvalidArgument(
                "unsupported metadata descriptor on CudaBackend".into(),
            )),
        }
    }

    fn has_metadata_support(desc: MetadataPrimsDescriptor) -> bool {
        match desc {
            MetadataPrimsDescriptor::Generate {
                op: MetadataGenerateOp::IotaStartZero,
                output_dtype: MetadataDType::I32,
            } => true,
            MetadataPrimsDescriptor::Binary {
                op,
                lhs_dtype,
                rhs_dtype,
                output_dtype,
            } => {
                matches!(op, MetadataBinaryOp::Equal | MetadataBinaryOp::NotEqual)
                    && matches!(
                        (lhs_dtype, rhs_dtype, output_dtype),
                        (MetadataDType::I32, MetadataDType::I32, MetadataDType::Bool)
                            | (
                                MetadataDType::Bool,
                                MetadataDType::Bool,
                                MetadataDType::Bool
                            )
                    )
            }
            MetadataPrimsDescriptor::Ternary {
                op: MetadataTernaryOp::Where,
                cond_dtype: MetadataDType::Bool,
                lhs_dtype,
                rhs_dtype,
                output_dtype,
            } => matches!(
                (lhs_dtype, rhs_dtype, output_dtype),
                (MetadataDType::I32, MetadataDType::I32, MetadataDType::I32)
                    | (
                        MetadataDType::Bool,
                        MetadataDType::Bool,
                        MetadataDType::Bool
                    )
            ),
            MetadataPrimsDescriptor::Reduction {
                op,
                input_dtype,
                output_dtype,
                ..
            } => matches!(
                (op, input_dtype, output_dtype),
                (
                    MetadataReductionOp::Sum,
                    MetadataDType::Bool,
                    MetadataDType::I32
                ) | (
                    MetadataReductionOp::Sum,
                    MetadataDType::I32,
                    MetadataDType::I32
                ) | (
                    MetadataReductionOp::All,
                    MetadataDType::Bool,
                    MetadataDType::Bool
                ) | (
                    MetadataReductionOp::Any,
                    MetadataDType::Bool,
                    MetadataDType::Bool
                )
            ),
            _ => false,
        }
    }
}
