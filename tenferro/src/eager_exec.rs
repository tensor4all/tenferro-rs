use num_complex::{Complex32, Complex64};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::validate::validate_nonsingular_u;
use tenferro_tensor::{
    DType, DotGeneralConfig, PadConfig, SliceConfig, Tensor, TensorBackend, TensorExec, TypedTensor,
};

use crate::einsum_subscripts::to_einsum_subscripts;
use crate::error::{Error, Result};
use crate::scalar_semantics::dynamic_truncate_size;
use crate::shape_infer::promote_dtype_for_binary_op;

/// If the two tensors have different dtypes, insert Convert ops so they
/// both match the promoted result dtype. Returns the (possibly converted)
/// tensors.
fn promote_binary_to_dtype(
    exec: &mut dyn TensorExec,
    a: &Tensor,
    b: &Tensor,
    promoted: DType,
) -> Result<(Tensor, Tensor)> {
    let a = if a.dtype() != promoted {
        exec.convert(a, promoted).map_err(Error::from)?
    } else {
        a.clone()
    };
    let b = if b.dtype() != promoted {
        exec.convert(b, promoted).map_err(Error::from)?
    } else {
        b.clone()
    };
    Ok((a, b))
}

fn promote_binary(
    exec: &mut dyn TensorExec,
    a: &Tensor,
    b: &Tensor,
    op: &StdTensorOp,
) -> Result<(Tensor, Tensor)> {
    let promoted = promote_dtype_for_binary_op(op, a.dtype(), b.dtype());
    promote_binary_to_dtype(exec, a, b, promoted)
}

pub(crate) fn exec_dot_general_with_conj_on_tensors<B: TensorBackend>(
    lhs: &Tensor,
    rhs: &Tensor,
    config: &DotGeneralConfig,
    lhs_conj: bool,
    rhs_conj: bool,
    backend: &mut B,
) -> Result<Tensor> {
    let promoted = crate::shape_infer::promote_dtype(lhs.dtype(), rhs.dtype());
    backend.with_exec_session(|exec| {
        let (lhs, rhs) = promote_binary_to_dtype(exec, lhs, rhs, promoted)?;
        exec.dot_general_with_conj(&lhs, &rhs, config, lhs_conj, rhs_conj)
            .map_err(Error::from)
    })
}

/// Execute a single [`StdTensorOp`] on concrete tensors.
///
/// Most ops produce one output tensor. Multi-output linalg ops return one
/// tensor per output slot.
pub fn exec_op_on_tensors<B: TensorBackend>(
    op: &StdTensorOp,
    inputs: &[&Tensor],
    backend: &mut B,
) -> Result<Vec<Tensor>> {
    if let StdTensorOp::NaryEinsum { subscripts, .. } = op {
        let parsed = to_einsum_subscripts(subscripts);
        return Ok(vec![tenferro_einsum::eager_einsum_subscripts(
            backend, inputs, &parsed,
        )?]);
    }

    if let StdTensorOp::Extension(ext) = op {
        // Per spec Section 8 the eager path MUST NOT open a backend exec
        // session for extension ops; the extension owns its execution
        // model. Route errors through the standard tensor error channel
        // so callers see consistent error types.
        return ext.eager_execute(inputs).map_err(|err| {
            Error::TensorRuntime(tenferro_tensor::Error::BackendFailure {
                op: "extension",
                message: format!("family_id={:?}: {err}", ext.family_id()),
            })
        });
    }

    backend.with_exec_session(|exec| {
        let result = match op {
            StdTensorOp::Add => {
                let (a, b) = promote_binary(exec, inputs[0], inputs[1], op)?;
                vec![exec.add(&a, &b)?]
            }
            StdTensorOp::Mul => {
                let (a, b) = promote_binary(exec, inputs[0], inputs[1], op)?;
                vec![exec.mul(&a, &b)?]
            }
            StdTensorOp::Neg => vec![exec.neg(inputs[0])?],
            StdTensorOp::Div => {
                let (a, b) = promote_binary(exec, inputs[0], inputs[1], op)?;
                vec![exec.div(&a, &b)?]
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
                vec![exec.pow(&a, &b)?]
            }
            StdTensorOp::Abs => vec![exec.abs(inputs[0])?],
            StdTensorOp::Sign => vec![exec.sign(inputs[0])?],
            StdTensorOp::Conj => vec![exec.conj(inputs[0])?],
            StdTensorOp::Maximum => {
                let (a, b) = promote_binary(exec, inputs[0], inputs[1], op)?;
                vec![exec.maximum(&a, &b)?]
            }
            StdTensorOp::Minimum => {
                let (a, b) = promote_binary(exec, inputs[0], inputs[1], op)?;
                vec![exec.minimum(&a, &b)?]
            }
            StdTensorOp::Compare(dir) => {
                let (a, b) = promote_binary(exec, inputs[0], inputs[1], op)?;
                vec![exec.compare(&a, &b, dir)?]
            }
            StdTensorOp::Transpose { perm } => vec![exec.transpose(inputs[0], perm)?],
            StdTensorOp::ReduceSum { axes, .. } => vec![exec.reduce_sum(inputs[0], axes)?],
            StdTensorOp::DotGeneral { config, .. } => {
                let (a, b) = promote_binary(exec, inputs[0], inputs[1], op)?;
                vec![exec.dot_general(&a, &b, config)?]
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
                let (a, b) = promote_binary(exec, inputs[0], inputs[1], op)?;
                let (a, c) = promote_binary(exec, &a, inputs[2], op)?;
                vec![exec.select(&a, &b, &c)?]
            }
            StdTensorOp::Clamp => {
                let (a, b) = promote_binary(exec, inputs[0], inputs[1], op)?;
                let (a, c) = promote_binary(exec, &a, inputs[2], op)?;
                vec![exec.clamp(&a, &b, &c)?]
            }
            StdTensorOp::Concatenate { axis, .. } => {
                let promoted = crate::shape_infer::promote_dtypes(inputs.iter().map(|t| t.dtype()));
                let mut promoted_inputs: Vec<Tensor> = Vec::with_capacity(inputs.len());
                for t in inputs {
                    if t.dtype() != promoted {
                        promoted_inputs.push(exec.convert(t, promoted).map_err(Error::from)?);
                    } else {
                        promoted_inputs.push((*t).clone());
                    }
                }
                let promoted_refs: Vec<&Tensor> = promoted_inputs.iter().collect();
                vec![exec.concatenate(&promoted_refs, *axis)?]
            }
            StdTensorOp::NaryEinsum { .. } => {
                unreachable!("NaryEinsum is handled before opening an exec session")
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
                vec![exec.scatter(&operand, inputs[1], &updates, config)?]
            }
            StdTensorOp::DynamicSlice { slice_sizes } => {
                vec![exec.dynamic_slice(inputs[0], inputs[1], slice_sizes)?]
            }
            StdTensorOp::DynamicUpdateSlice => {
                let (operand, update) = promote_binary(exec, inputs[0], inputs[1], op)?;
                vec![exec.dynamic_update_slice(&operand, &update, inputs[2])?]
            }
            StdTensorOp::Cholesky { .. } => vec![exec.cholesky(inputs[0])?],
            StdTensorOp::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
                ..
            } => {
                let (a, b) = promote_binary(exec, inputs[0], inputs[1], op)?;
                vec![exec.triangular_solve(
                    &a,
                    &b,
                    *left_side,
                    *lower,
                    *transpose_a,
                    *unit_diagonal,
                )?]
            }
            StdTensorOp::Svd { .. } => exec.svd(inputs[0])?,
            StdTensorOp::Qr { .. } => exec.qr(inputs[0])?,
            StdTensorOp::Lu { .. } => exec.lu(inputs[0])?,
            StdTensorOp::FullPivLu { .. } => exec.full_piv_lu(inputs[0])?,
            StdTensorOp::FullPivLuSolve { transpose_a } => {
                let (a, b) = promote_binary(exec, inputs[0], inputs[1], op)?;
                vec![exec.full_piv_lu_solve(&a, &b, *transpose_a)?]
            }
            StdTensorOp::Eigh { .. } => exec.eigh(inputs[0])?,
            StdTensorOp::Eig { .. } => exec.eig(inputs[0])?,
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
            StdTensorOp::ValidateNonsingular { .. } => {
                validate_nonsingular_u(inputs[0])?;
                vec![inputs[0].clone()]
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
        DType::I64 => Tensor::I64(TypedTensor::from_vec_col_major(
            vec![],
            vec![i64::from_le_bytes(exact_bytes::<8>(dtype, bytes))],
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
