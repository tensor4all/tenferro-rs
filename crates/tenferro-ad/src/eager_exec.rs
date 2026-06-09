use num_complex::{Complex32, Complex64};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::DotGeneralConfig;
use tenferro_tensor::{
    BackendSession, DType, PadConfig, SliceConfig, Tensor, TensorBackend, TensorRead, TypedTensor,
};

use crate::error::{Error, Result};
use crate::extension_runtime::ExtensionExecutor;
use crate::scalar_semantics::dynamic_truncate_size;
use crate::shape_infer::promote_dtype_for_binary_op;

enum PromotedTensor<'a> {
    Borrowed(&'a Tensor),
    Owned(Box<Tensor>),
}

enum PromotedTensorRead<'a> {
    Borrowed(Box<TensorRead<'a>>),
    Owned(Box<Tensor>),
}

enum ConcreteTensorRead<'a> {
    Borrowed(&'a Tensor),
    Owned(Box<Tensor>),
}

impl<'a> PromotedTensor<'a> {
    fn tensor(&'a self) -> &'a Tensor {
        match self {
            Self::Borrowed(tensor) => tensor,
            Self::Owned(tensor) => tensor,
        }
    }
}

impl PromotedTensorRead<'_> {
    fn tensor_read(&self) -> TensorRead<'_> {
        match self {
            Self::Borrowed(read) => read.as_ref().clone(),
            Self::Owned(tensor) => TensorRead::from_tensor(tensor.as_ref()),
        }
    }
}

impl<'a> ConcreteTensorRead<'a> {
    fn tensor(&'a self) -> &'a Tensor {
        match self {
            Self::Borrowed(tensor) => tensor,
            Self::Owned(tensor) => tensor,
        }
    }
}

fn promote_to_dtype<'a>(
    exec: &mut dyn BackendSession,
    tensor: &'a Tensor,
    promoted: DType,
) -> Result<PromotedTensor<'a>> {
    if tensor.dtype() == promoted {
        Ok(PromotedTensor::Borrowed(tensor))
    } else {
        Ok(PromotedTensor::Owned(Box::new(
            exec.convert(tensor, promoted).map_err(Error::from)?,
        )))
    }
}

/// If the two tensors have different dtypes, insert Convert ops so they
/// both match the promoted result dtype. Returns the (possibly converted)
/// tensors.
fn promote_binary_to_dtype<'a>(
    exec: &mut dyn BackendSession,
    a: &'a Tensor,
    b: &'a Tensor,
    promoted: DType,
) -> Result<(PromotedTensor<'a>, PromotedTensor<'a>)> {
    let a = promote_to_dtype(exec, a, promoted)?;
    let b = promote_to_dtype(exec, b, promoted)?;
    Ok((a, b))
}

fn promote_binary<'a>(
    exec: &mut dyn BackendSession,
    a: &'a Tensor,
    b: &'a Tensor,
    op: &StdTensorOp,
) -> Result<(PromotedTensor<'a>, PromotedTensor<'a>)> {
    let promoted = promote_dtype_for_binary_op(op, a.dtype(), b.dtype());
    promote_binary_to_dtype(exec, a, b, promoted)
}

fn materialize_tensor_read(input: TensorRead<'_>) -> Tensor {
    input.to_tensor()
}

fn concrete_tensor_read(input: TensorRead<'_>) -> ConcreteTensorRead<'_> {
    match input {
        TensorRead::Tensor(tensor) => ConcreteTensorRead::Borrowed(tensor),
        TensorRead::View(view) => ConcreteTensorRead::Owned(Box::new(view.to_tensor())),
    }
}

fn concrete_tensor_reads<'a>(inputs: &[TensorRead<'a>]) -> Vec<ConcreteTensorRead<'a>> {
    inputs.iter().cloned().map(concrete_tensor_read).collect()
}

fn concrete_promoted_read_to_dtype<'a>(
    exec: &mut dyn BackendSession,
    input: TensorRead<'a>,
    promoted: DType,
) -> Result<ConcreteTensorRead<'a>> {
    if input.dtype() == promoted {
        Ok(concrete_tensor_read(input))
    } else {
        let input = concrete_tensor_read(input);
        Ok(ConcreteTensorRead::Owned(Box::new(
            exec.convert(input.tensor(), promoted)
                .map_err(Error::from)?,
        )))
    }
}

fn promote_read_to_dtype<'a>(
    exec: &mut dyn BackendSession,
    input: TensorRead<'a>,
    promoted: DType,
) -> Result<PromotedTensorRead<'a>> {
    if input.dtype() == promoted {
        Ok(PromotedTensorRead::Borrowed(Box::new(input)))
    } else {
        let input = concrete_tensor_read(input);
        Ok(PromotedTensorRead::Owned(Box::new(
            exec.convert(input.tensor(), promoted)
                .map_err(Error::from)?,
        )))
    }
}

fn promote_binary_reads_to_dtype<'a>(
    exec: &mut dyn BackendSession,
    a: TensorRead<'a>,
    b: TensorRead<'a>,
    promoted: DType,
) -> Result<(PromotedTensorRead<'a>, PromotedTensorRead<'a>)> {
    let a = promote_read_to_dtype(exec, a, promoted)?;
    let b = promote_read_to_dtype(exec, b, promoted)?;
    Ok((a, b))
}

fn promote_binary_reads<'a>(
    exec: &mut dyn BackendSession,
    a: TensorRead<'a>,
    b: TensorRead<'a>,
    op: &StdTensorOp,
) -> Result<(PromotedTensorRead<'a>, PromotedTensorRead<'a>)> {
    let promoted = promote_dtype_for_binary_op(op, a.dtype(), b.dtype());
    promote_binary_reads_to_dtype(exec, a, b, promoted)
}

pub(crate) fn exec_dot_general_with_conj_on_tensor_reads<B: TensorBackend>(
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
    config: &DotGeneralConfig,
    lhs_conj: bool,
    rhs_conj: bool,
    backend: &mut B,
) -> Result<Tensor> {
    let promoted = crate::shape_infer::promote_dtype(lhs.dtype(), rhs.dtype());
    backend.with_backend_session(|exec| {
        let (lhs, rhs) = promote_binary_reads_to_dtype(exec, lhs, rhs, promoted)?;
        exec.dot_general_with_conj_read(
            lhs.tensor_read(),
            rhs.tensor_read(),
            config,
            lhs_conj,
            rhs_conj,
        )
        .map_err(Error::from)
    })
}

/// Execute a single [`StdTensorOp`] on concrete tensors.
///
/// This core helper rejects extension ops because they require a runtime owner
/// with a registered extension executor.
pub(crate) fn exec_op_on_tensors<B: TensorBackend>(
    op: &StdTensorOp,
    inputs: &[&Tensor],
    backend: &mut B,
) -> Result<Vec<Tensor>> {
    if let StdTensorOp::Extension(ext) = op {
        return Err(missing_extension_executor_error(ext.as_ref()));
    }

    exec_standard_op_on_tensors(op, inputs, backend)
}

pub(crate) fn exec_op_on_tensors_with_extension_executor<B: TensorBackend + 'static>(
    op: &StdTensorOp,
    inputs: &[&Tensor],
    backend: &mut B,
    extension_executor: Option<&mut ExtensionExecutor<B>>,
) -> Result<Vec<Tensor>> {
    if let StdTensorOp::Extension(ext) = op {
        let Some(extension_executor) = extension_executor else {
            return Err(missing_extension_executor_error(ext.as_ref()));
        };
        let outputs = extension_executor.execute(backend, ext.as_ref(), inputs);
        return outputs.map_err(|err| extension_error(ext.as_ref(), err));
    }

    exec_standard_op_on_tensors(op, inputs, backend)
}

pub(crate) fn exec_op_on_tensor_reads_with_extension_executor<B: TensorBackend + 'static>(
    op: &StdTensorOp,
    inputs: &[TensorRead<'_>],
    backend: &mut B,
    extension_executor: Option<&mut ExtensionExecutor<B>>,
) -> Result<Vec<Tensor>> {
    if let StdTensorOp::Extension(ext) = op {
        let Some(extension_executor) = extension_executor else {
            return Err(missing_extension_executor_error(ext.as_ref()));
        };
        let input_tensors = concrete_tensor_reads(inputs);
        let input_refs: Vec<&Tensor> = input_tensors
            .iter()
            .map(ConcreteTensorRead::tensor)
            .collect();
        let outputs = extension_executor.execute(backend, ext.as_ref(), &input_refs);
        return outputs.map_err(|err| extension_error(ext.as_ref(), err));
    }

    exec_standard_op_on_tensor_reads(op, inputs, backend)
}

fn extension_error(
    ext: &dyn tenferro_ops::ext_op::ExtensionOp,
    err: tenferro_tensor::Error,
) -> Error {
    Error::TensorRuntime(tenferro_tensor::Error::backend_failure(
        "extension",
        format!("family_id={:?}: {err}", ext.family_id()),
    ))
}

fn missing_extension_executor_error(ext: &dyn tenferro_ops::ext_op::ExtensionOp) -> Error {
    Error::TensorRuntime(tenferro_tensor::Error::InvalidConfig {
        op: "extension",
        message: format!(
            "extension op for family_id {:?} requires an ExtensionExecutor; execute through EagerRuntime or register and pass the extension runtime owner",
            ext.family_id()
        ),
    })
}

fn exec_standard_op_on_tensor_reads<B: TensorBackend>(
    op: &StdTensorOp,
    inputs: &[TensorRead<'_>],
    backend: &mut B,
) -> Result<Vec<Tensor>> {
    backend.with_backend_session(|exec| {
        let result = match op {
            StdTensorOp::Add => {
                let (a, b) = promote_binary_reads(exec, inputs[0].clone(), inputs[1].clone(), op)?;
                vec![exec.add_read(a.tensor_read(), b.tensor_read())?]
            }
            StdTensorOp::Mul => {
                let (a, b) = promote_binary_reads(exec, inputs[0].clone(), inputs[1].clone(), op)?;
                vec![exec.mul_read(a.tensor_read(), b.tensor_read())?]
            }
            StdTensorOp::Neg => vec![exec.neg_read(inputs[0].clone())?],
            StdTensorOp::Div => {
                let (a, b) = promote_binary_reads(exec, inputs[0].clone(), inputs[1].clone(), op)?;
                vec![exec.div_read(a.tensor_read(), b.tensor_read())?]
            }
            StdTensorOp::Exp => vec![exec.exp_read(inputs[0].clone())?],
            StdTensorOp::Log => vec![exec.log_read(inputs[0].clone())?],
            StdTensorOp::Sin => vec![exec.sin_read(inputs[0].clone())?],
            StdTensorOp::Cos => vec![exec.cos_read(inputs[0].clone())?],
            StdTensorOp::Tanh => vec![exec.tanh_read(inputs[0].clone())?],
            StdTensorOp::Sqrt => vec![exec.sqrt_read(inputs[0].clone())?],
            StdTensorOp::Rsqrt => vec![exec.rsqrt_read(inputs[0].clone())?],
            StdTensorOp::Pow => {
                let (a, b) = promote_binary_reads(exec, inputs[0].clone(), inputs[1].clone(), op)?;
                vec![exec.pow_read(a.tensor_read(), b.tensor_read())?]
            }
            StdTensorOp::Abs => vec![exec.abs_read(inputs[0].clone())?],
            StdTensorOp::Sign => vec![exec.sign_read(inputs[0].clone())?],
            StdTensorOp::Conj => vec![exec.conj_read(inputs[0].clone())?],
            StdTensorOp::Maximum => {
                let (a, b) = promote_binary_reads(exec, inputs[0].clone(), inputs[1].clone(), op)?;
                vec![exec.maximum_read(a.tensor_read(), b.tensor_read())?]
            }
            StdTensorOp::Minimum => {
                let (a, b) = promote_binary_reads(exec, inputs[0].clone(), inputs[1].clone(), op)?;
                vec![exec.minimum_read(a.tensor_read(), b.tensor_read())?]
            }
            StdTensorOp::Compare(dir) => {
                let (a, b) = promote_binary_reads(exec, inputs[0].clone(), inputs[1].clone(), op)?;
                vec![exec.compare_read(a.tensor_read(), b.tensor_read(), dir)?]
            }
            StdTensorOp::Transpose { perm } => vec![exec.transpose_read(inputs[0].clone(), perm)?],
            StdTensorOp::ReduceSum { axes, .. } => {
                vec![exec.reduce_sum_read(inputs[0].clone(), axes)?]
            }
            StdTensorOp::DotGeneral { config, .. } => {
                let (a, b) = promote_binary_reads(exec, inputs[0].clone(), inputs[1].clone(), op)?;
                vec![exec.dot_general_read(a.tensor_read(), b.tensor_read(), config)?]
            }
            StdTensorOp::Reshape { to_shape, .. } => {
                let shape = resolve_tensor_read_shape_exprs(inputs, to_shape);
                vec![exec.reshape_read(inputs[0].clone(), &shape)?]
            }
            StdTensorOp::BroadcastInDim { shape, dims } => {
                let shape = resolve_tensor_read_shape_exprs(inputs, shape);
                vec![exec.broadcast_in_dim_read(inputs[0].clone(), &shape, dims)?]
            }
            StdTensorOp::ExtractDiag { axis_a, axis_b } => {
                let input = concrete_tensor_read(inputs[0].clone());
                vec![exec.extract_diagonal(input.tensor(), *axis_a, *axis_b)?]
            }
            StdTensorOp::EmbedDiag { axis_a, axis_b } => {
                let input = concrete_tensor_read(inputs[0].clone());
                vec![exec.embed_diagonal(input.tensor(), *axis_a, *axis_b)?]
            }
            StdTensorOp::Tril { k } => {
                let input = concrete_tensor_read(inputs[0].clone());
                vec![exec.tril(input.tensor(), *k)?]
            }
            StdTensorOp::Triu { k } => {
                let input = concrete_tensor_read(inputs[0].clone());
                vec![exec.triu(input.tensor(), *k)?]
            }
            StdTensorOp::Slice(config) => {
                let input = concrete_tensor_read(inputs[0].clone());
                vec![exec.slice(input.tensor(), config)?]
            }
            StdTensorOp::Pad(config) => {
                let input = concrete_tensor_read(inputs[0].clone());
                vec![exec.pad(input.tensor(), config)?]
            }
            StdTensorOp::Reverse { axes } => {
                let input = concrete_tensor_read(inputs[0].clone());
                vec![exec.reverse(input.tensor(), axes)?]
            }
            StdTensorOp::ReduceProd { axes, .. } => {
                vec![exec.reduce_prod_read(inputs[0].clone(), axes)?]
            }
            StdTensorOp::ReduceMax { axes, .. } => {
                vec![exec.reduce_max_read(inputs[0].clone(), axes)?]
            }
            StdTensorOp::ReduceMin { axes, .. } => {
                vec![exec.reduce_min_read(inputs[0].clone(), axes)?]
            }
            StdTensorOp::Expm1 => vec![exec.expm1_read(inputs[0].clone())?],
            StdTensorOp::Log1p => vec![exec.log1p_read(inputs[0].clone())?],
            StdTensorOp::Convert { to, .. } => {
                let input = concrete_tensor_read(inputs[0].clone());
                vec![exec.convert(input.tensor(), *to)?]
            }
            StdTensorOp::Constant { dtype, bytes } => vec![constant_tensor(*dtype, bytes)],
            StdTensorOp::Select => {
                let value_dtype =
                    crate::shape_infer::promote_dtype(inputs[1].dtype(), inputs[2].dtype());
                let b = promote_read_to_dtype(exec, inputs[1].clone(), value_dtype)?;
                let c = promote_read_to_dtype(exec, inputs[2].clone(), value_dtype)?;
                vec![exec.select_read(inputs[0].clone(), b.tensor_read(), c.tensor_read())?]
            }
            StdTensorOp::Clamp => {
                let value_dtype =
                    crate::shape_infer::promote_dtypes(inputs.iter().map(|t| t.dtype()));
                let input = promote_read_to_dtype(exec, inputs[0].clone(), value_dtype)?;
                let lower = promote_read_to_dtype(exec, inputs[1].clone(), value_dtype)?;
                let upper = promote_read_to_dtype(exec, inputs[2].clone(), value_dtype)?;
                vec![exec.clamp_read(
                    input.tensor_read(),
                    lower.tensor_read(),
                    upper.tensor_read(),
                )?]
            }
            StdTensorOp::Concatenate { axis, .. } => {
                let promoted = crate::shape_infer::promote_dtypes(inputs.iter().map(|t| t.dtype()));
                let mut tensors = Vec::with_capacity(inputs.len());
                for input in inputs {
                    tensors.push(concrete_promoted_read_to_dtype(
                        exec,
                        input.clone(),
                        promoted,
                    )?);
                }
                let refs: Vec<&Tensor> = tensors.iter().map(ConcreteTensorRead::tensor).collect();
                vec![exec.concatenate(&refs, *axis)?]
            }
            StdTensorOp::Gather(config) => {
                let tensors = concrete_tensor_reads(inputs);
                vec![exec.gather(tensors[0].tensor(), tensors[1].tensor(), config)?]
            }
            StdTensorOp::GatherDynamicSliceSizes {
                offset_dims,
                collapsed_slice_dims,
                start_index_map,
                index_vector_dim,
                slice_sizes,
            } => {
                let slice_sizes = resolve_tensor_read_shape_exprs(inputs, slice_sizes);
                let config = tenferro_tensor::GatherConfig {
                    offset_dims: offset_dims.clone(),
                    collapsed_slice_dims: collapsed_slice_dims.clone(),
                    start_index_map: start_index_map.clone(),
                    index_vector_dim: *index_vector_dim,
                    slice_sizes,
                };
                let tensors = concrete_tensor_reads(inputs);
                vec![exec.gather(tensors[0].tensor(), tensors[1].tensor(), &config)?]
            }
            StdTensorOp::Scatter(config) => {
                let operand_dtype =
                    crate::shape_infer::promote_dtype(inputs[0].dtype(), inputs[2].dtype());
                let operand =
                    concrete_promoted_read_to_dtype(exec, inputs[0].clone(), operand_dtype)?;
                let updates =
                    concrete_promoted_read_to_dtype(exec, inputs[2].clone(), operand_dtype)?;
                let indices = concrete_tensor_read(inputs[1].clone());
                vec![exec.scatter(operand.tensor(), indices.tensor(), updates.tensor(), config)?]
            }
            StdTensorOp::DynamicSlice { slice_sizes } => {
                let tensors = concrete_tensor_reads(inputs);
                vec![exec.dynamic_slice(tensors[0].tensor(), tensors[1].tensor(), slice_sizes)?]
            }
            StdTensorOp::DynamicUpdateSlice => {
                let operand_dtype =
                    crate::shape_infer::promote_dtype(inputs[0].dtype(), inputs[1].dtype());
                let operand =
                    concrete_promoted_read_to_dtype(exec, inputs[0].clone(), operand_dtype)?;
                let update =
                    concrete_promoted_read_to_dtype(exec, inputs[1].clone(), operand_dtype)?;
                let starts = concrete_tensor_read(inputs[2].clone());
                vec![exec.dynamic_update_slice(
                    operand.tensor(),
                    update.tensor(),
                    starts.tensor(),
                )?]
            }
            StdTensorOp::ShapeOf { axis } => {
                let input = &inputs[0];
                if *axis >= input.shape().len() {
                    return Err(Error::Internal(format!(
                        "ShapeOf: axis {} out of bounds for rank {}",
                        axis,
                        input.shape().len()
                    )));
                }
                let size = input.shape()[*axis] as f64;
                vec![Tensor::F64(TypedTensor::from_vec_col_major(
                    vec![],
                    vec![size],
                ))]
            }
            StdTensorOp::DynamicTruncate { axis } => {
                let input = &inputs[0];
                if *axis >= input.shape().len() {
                    return Err(Error::Internal(format!(
                        "DynamicTruncate: axis {} out of bounds for rank {}",
                        axis,
                        input.shape().len()
                    )));
                }
                let size_tensor = concrete_tensor_read(inputs[1].clone());
                let axis_extent = input.shape()[*axis];
                let size = dynamic_truncate_size(size_tensor.tensor(), axis_extent)?;
                let rank = input.shape().len();
                let mut limits = input.shape().to_vec();
                limits[*axis] = size;
                let config = SliceConfig {
                    starts: vec![0; rank],
                    limits,
                    strides: vec![1; rank],
                };
                let input = concrete_tensor_read(inputs[0].clone());
                vec![exec.slice(input.tensor(), &config)?]
            }
            StdTensorOp::PadToMatch { axis } => {
                let input = &inputs[0];
                let reference = &inputs[1];
                if *axis >= input.shape().len() {
                    return Err(Error::Internal(format!(
                        "PadToMatch: axis {} out of bounds for rank {}",
                        axis,
                        input.shape().len()
                    )));
                }
                let target_size = reference.shape()[*axis];
                let current_size = input.shape()[*axis];
                if current_size >= target_size {
                    vec![materialize_tensor_read(inputs[0].clone())]
                } else {
                    let rank = input.shape().len();
                    let mut high = vec![0i64; rank];
                    high[*axis] = (target_size - current_size) as i64;
                    let config = PadConfig {
                        edge_padding_low: vec![0i64; rank],
                        edge_padding_high: high,
                        interior_padding: vec![0i64; rank],
                    };
                    let input = concrete_tensor_read(inputs[0].clone());
                    vec![exec.pad(input.tensor(), &config)?]
                }
            }
            StdTensorOp::Extension(_) => {
                unreachable!("Extension is handled before opening an exec session")
            }
        };
        Ok(result)
    })
}

fn exec_standard_op_on_tensors<B: TensorBackend>(
    op: &StdTensorOp,
    inputs: &[&Tensor],
    backend: &mut B,
) -> Result<Vec<Tensor>> {
    backend.with_backend_session(|exec| {
        let result = match op {
            StdTensorOp::Add => {
                let (a, b) = promote_binary(exec, inputs[0], inputs[1], op)?;
                vec![exec.add(a.tensor(), b.tensor())?]
            }
            StdTensorOp::Mul => {
                let (a, b) = promote_binary(exec, inputs[0], inputs[1], op)?;
                vec![exec.mul(a.tensor(), b.tensor())?]
            }
            StdTensorOp::Neg => vec![exec.neg(inputs[0])?],
            StdTensorOp::Div => {
                let (a, b) = promote_binary(exec, inputs[0], inputs[1], op)?;
                vec![exec.div(a.tensor(), b.tensor())?]
            }
            StdTensorOp::Exp => vec![exec.exp(inputs[0])?],
            StdTensorOp::Log => vec![exec.log(inputs[0])?],
            StdTensorOp::Sin => vec![exec.sin(inputs[0])?],
            StdTensorOp::Cos => vec![exec.cos(inputs[0])?],
            StdTensorOp::Tanh => vec![exec.tanh(inputs[0])?],
            StdTensorOp::Sqrt => vec![exec.sqrt(inputs[0])?],
            StdTensorOp::Rsqrt => vec![exec.rsqrt(inputs[0])?],
            StdTensorOp::Pow => {
                let (a, b) = promote_binary(exec, inputs[0], inputs[1], op)?;
                vec![exec.pow(a.tensor(), b.tensor())?]
            }
            StdTensorOp::Abs => vec![exec.abs(inputs[0])?],
            StdTensorOp::Sign => vec![exec.sign(inputs[0])?],
            StdTensorOp::Conj => vec![exec.conj(inputs[0])?],
            StdTensorOp::Maximum => {
                let (a, b) = promote_binary(exec, inputs[0], inputs[1], op)?;
                vec![exec.maximum(a.tensor(), b.tensor())?]
            }
            StdTensorOp::Minimum => {
                let (a, b) = promote_binary(exec, inputs[0], inputs[1], op)?;
                vec![exec.minimum(a.tensor(), b.tensor())?]
            }
            StdTensorOp::Compare(dir) => {
                let (a, b) = promote_binary(exec, inputs[0], inputs[1], op)?;
                vec![exec.compare(a.tensor(), b.tensor(), dir)?]
            }
            StdTensorOp::Transpose { perm } => vec![exec.transpose(inputs[0], perm)?],
            StdTensorOp::ReduceSum { axes, .. } => vec![exec.reduce_sum(inputs[0], axes)?],
            StdTensorOp::DotGeneral { config, .. } => {
                let (a, b) = promote_binary(exec, inputs[0], inputs[1], op)?;
                vec![exec.dot_general(a.tensor(), b.tensor(), config)?]
            }
            StdTensorOp::Reshape { to_shape, .. } => {
                let shape = resolve_tensor_shape_exprs(inputs, to_shape);
                vec![exec.reshape(inputs[0], &shape)?]
            }
            StdTensorOp::BroadcastInDim { shape, dims } => {
                let shape = resolve_tensor_shape_exprs(inputs, shape);
                vec![exec.broadcast_in_dim(inputs[0], &shape, dims)?]
            }
            StdTensorOp::ExtractDiag { axis_a, axis_b } => {
                vec![exec.extract_diagonal(inputs[0], *axis_a, *axis_b)?]
            }
            StdTensorOp::EmbedDiag { axis_a, axis_b } => {
                vec![exec.embed_diagonal(inputs[0], *axis_a, *axis_b)?]
            }
            StdTensorOp::Tril { k } => vec![exec.tril(inputs[0], *k)?],
            StdTensorOp::Triu { k } => vec![exec.triu(inputs[0], *k)?],
            StdTensorOp::Slice(config) => vec![exec.slice(inputs[0], config)?],
            StdTensorOp::Pad(config) => vec![exec.pad(inputs[0], config)?],
            StdTensorOp::Reverse { axes } => vec![exec.reverse(inputs[0], axes)?],
            StdTensorOp::ReduceProd { axes, .. } => vec![exec.reduce_prod(inputs[0], axes)?],
            StdTensorOp::ReduceMax { axes, .. } => vec![exec.reduce_max(inputs[0], axes)?],
            StdTensorOp::ReduceMin { axes, .. } => vec![exec.reduce_min(inputs[0], axes)?],
            StdTensorOp::Expm1 => vec![exec.expm1(inputs[0])?],
            StdTensorOp::Log1p => vec![exec.log1p(inputs[0])?],
            StdTensorOp::Convert { to, .. } => vec![exec.convert(inputs[0], *to)?],
            StdTensorOp::Constant { dtype, bytes } => vec![constant_tensor(*dtype, bytes)],
            StdTensorOp::Select => {
                let value_dtype =
                    crate::shape_infer::promote_dtype(inputs[1].dtype(), inputs[2].dtype());
                let b = promote_to_dtype(exec, inputs[1], value_dtype)?;
                let c = promote_to_dtype(exec, inputs[2], value_dtype)?;
                vec![exec.select(inputs[0], b.tensor(), c.tensor())?]
            }
            StdTensorOp::Clamp => {
                let (a, b) = promote_binary(exec, inputs[0], inputs[1], op)?;
                let (a, c) = promote_binary(exec, a.tensor(), inputs[2], op)?;
                vec![exec.clamp(a.tensor(), b.tensor(), c.tensor())?]
            }
            StdTensorOp::Concatenate { axis, .. } => {
                let promoted = crate::shape_infer::promote_dtypes(inputs.iter().map(|t| t.dtype()));
                let mut promoted_inputs = Vec::with_capacity(inputs.len());
                for t in inputs {
                    promoted_inputs.push(promote_to_dtype(exec, t, promoted)?);
                }
                let promoted_refs: Vec<&Tensor> =
                    promoted_inputs.iter().map(PromotedTensor::tensor).collect();
                vec![exec.concatenate(&promoted_refs, *axis)?]
            }
            StdTensorOp::Gather(config) => {
                vec![exec.gather(inputs[0], inputs[1], config)?]
            }
            StdTensorOp::GatherDynamicSliceSizes {
                offset_dims,
                collapsed_slice_dims,
                start_index_map,
                index_vector_dim,
                slice_sizes,
            } => {
                let slice_sizes = resolve_tensor_shape_exprs(inputs, slice_sizes);
                let config = tenferro_tensor::GatherConfig {
                    offset_dims: offset_dims.clone(),
                    collapsed_slice_dims: collapsed_slice_dims.clone(),
                    start_index_map: start_index_map.clone(),
                    index_vector_dim: *index_vector_dim,
                    slice_sizes,
                };
                vec![exec.gather(inputs[0], inputs[1], &config)?]
            }
            StdTensorOp::Scatter(config) => {
                let (operand, updates) = promote_binary(exec, inputs[0], inputs[2], op)?;
                vec![exec.scatter(operand.tensor(), inputs[1], updates.tensor(), config)?]
            }
            StdTensorOp::DynamicSlice { slice_sizes } => {
                vec![exec.dynamic_slice(inputs[0], inputs[1], slice_sizes)?]
            }
            StdTensorOp::DynamicUpdateSlice => {
                let (operand, update) = promote_binary(exec, inputs[0], inputs[1], op)?;
                vec![exec.dynamic_update_slice(operand.tensor(), update.tensor(), inputs[2])?]
            }
            StdTensorOp::ShapeOf { axis } => {
                let input = inputs[0];
                if *axis >= input.shape().len() {
                    return Err(Error::Internal(format!(
                        "ShapeOf: axis {} out of bounds for rank {}",
                        axis,
                        input.shape().len()
                    )));
                }
                let size = input.shape()[*axis] as f64;
                vec![Tensor::F64(TypedTensor::from_vec_col_major(
                    vec![],
                    vec![size],
                ))]
            }
            StdTensorOp::DynamicTruncate { axis } => {
                let input = inputs[0];
                if *axis >= input.shape().len() {
                    return Err(Error::Internal(format!(
                        "DynamicTruncate: axis {} out of bounds for rank {}",
                        axis,
                        input.shape().len()
                    )));
                }
                let size_tensor = inputs[1];
                let axis_extent = input.shape()[*axis];
                let size = dynamic_truncate_size(size_tensor, axis_extent)?;
                let rank = input.shape().len();
                let mut limits = input.shape().to_vec();
                limits[*axis] = size;
                let config = SliceConfig {
                    starts: vec![0; rank],
                    limits,
                    strides: vec![1; rank],
                };
                vec![exec.slice(input, &config)?]
            }
            StdTensorOp::PadToMatch { axis } => {
                let input = inputs[0];
                let reference = inputs[1];
                if *axis >= input.shape().len() {
                    return Err(Error::Internal(format!(
                        "PadToMatch: axis {} out of bounds for rank {}",
                        axis,
                        input.shape().len()
                    )));
                }
                let target_size = reference.shape()[*axis];
                let current_size = input.shape()[*axis];
                if current_size >= target_size {
                    vec![input.clone()]
                } else {
                    let rank = input.shape().len();
                    let mut high = vec![0i64; rank];
                    high[*axis] = (target_size - current_size) as i64;
                    let config = PadConfig {
                        edge_padding_low: vec![0i64; rank],
                        edge_padding_high: high,
                        interior_padding: vec![0i64; rank],
                    };
                    vec![exec.pad(input, &config)?]
                }
            }
            StdTensorOp::Extension(_) => {
                unreachable!("Extension is handled before opening an exec session")
            }
        };
        Ok(result)
    })
}

fn resolve_tensor_shape_exprs(inputs: &[&Tensor], exprs: &[DimExpr]) -> Vec<usize> {
    let input_shapes: Vec<&[usize]> = inputs.iter().map(|tensor| tensor.shape()).collect();
    DimExpr::eval_all(exprs, &input_shapes)
}

fn resolve_tensor_read_shape_exprs(inputs: &[TensorRead<'_>], exprs: &[DimExpr]) -> Vec<usize> {
    let input_shapes: Vec<&[usize]> = inputs.iter().map(|tensor| tensor.shape()).collect();
    DimExpr::eval_all(exprs, &input_shapes)
}

fn constant_tensor(dtype: DType, bytes: &[u8]) -> Tensor {
    match dtype {
        DType::F64 => Tensor::F64(TypedTensor::from_vec_col_major(
            vec![],
            vec![f64::from_le_bytes(exact_bytes::<8>(dtype, bytes))],
        )),
        DType::F32 => Tensor::F32(TypedTensor::from_vec_col_major(
            vec![],
            vec![f32::from_le_bytes(exact_bytes::<4>(dtype, bytes))],
        )),
        DType::I32 => Tensor::I32(TypedTensor::from_vec_col_major(
            vec![],
            vec![i32::from_le_bytes(exact_bytes::<4>(dtype, bytes))],
        )),
        DType::I64 => Tensor::I64(TypedTensor::from_vec_col_major(
            vec![],
            vec![i64::from_le_bytes(exact_bytes::<8>(dtype, bytes))],
        )),
        DType::Bool => Tensor::Bool(TypedTensor::from_vec_col_major(
            vec![],
            vec![exact_bytes::<1>(dtype, bytes)[0] != 0],
        )),
        DType::C64 => {
            let data = exact_bytes::<16>(dtype, bytes);
            let mut re_bytes = [0u8; 8];
            let mut im_bytes = [0u8; 8];
            re_bytes.copy_from_slice(&data[..8]);
            im_bytes.copy_from_slice(&data[8..]);
            let re = f64::from_le_bytes(re_bytes);
            let im = f64::from_le_bytes(im_bytes);
            Tensor::C64(TypedTensor::from_vec_col_major(
                vec![],
                vec![Complex64::new(re, im)],
            ))
        }
        DType::C32 => {
            let data = exact_bytes::<8>(dtype, bytes);
            let mut re_bytes = [0u8; 4];
            let mut im_bytes = [0u8; 4];
            re_bytes.copy_from_slice(&data[..4]);
            im_bytes.copy_from_slice(&data[4..]);
            let re = f32::from_le_bytes(re_bytes);
            let im = f32::from_le_bytes(im_bytes);
            Tensor::C32(TypedTensor::from_vec_col_major(
                vec![],
                vec![Complex32::new(re, im)],
            ))
        }
    }
}

fn exact_bytes<const N: usize>(dtype: DType, bytes: &[u8]) -> [u8; N] {
    if bytes.len() != N {
        panic!(
            "constant {:?} expected {} bytes, got {}",
            dtype,
            N,
            bytes.len()
        );
    }
    let mut out = [0u8; N];
    out.copy_from_slice(bytes);
    out
}

#[cfg(test)]
mod tests;
