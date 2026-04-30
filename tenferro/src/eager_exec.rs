use num_complex::{Complex32, Complex64};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::validate::validate_nonsingular_u;
use tenferro_tensor::{DType, PadConfig, SliceConfig, Tensor, TensorBackend, TypedTensor};

use crate::error::{Error, Result};

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
        return Ok(vec![tenferro_einsum::eager_einsum(
            backend, inputs, subscripts,
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
            StdTensorOp::Add => vec![exec.add(inputs[0], inputs[1])?],
            StdTensorOp::Mul => vec![exec.mul(inputs[0], inputs[1])?],
            StdTensorOp::Neg => vec![exec.neg(inputs[0])?],
            StdTensorOp::Div => vec![exec.div(inputs[0], inputs[1])?],
            StdTensorOp::Exp => vec![exec.exp(inputs[0])?],
            StdTensorOp::Log => vec![exec.log(inputs[0])?],
            StdTensorOp::Sin => vec![exec.sin(inputs[0])?],
            StdTensorOp::Cos => vec![exec.cos(inputs[0])?],
            StdTensorOp::Tanh => vec![exec.tanh(inputs[0])?],
            StdTensorOp::Sqrt => vec![exec.sqrt(inputs[0])?],
            StdTensorOp::Rsqrt => vec![exec.rsqrt(inputs[0])?],
            StdTensorOp::Pow => vec![exec.pow(inputs[0], inputs[1])?],
            StdTensorOp::Abs => vec![exec.abs(inputs[0])?],
            StdTensorOp::Sign => vec![exec.sign(inputs[0])?],
            StdTensorOp::Conj => vec![exec.conj(inputs[0])?],
            StdTensorOp::Maximum => vec![exec.maximum(inputs[0], inputs[1])?],
            StdTensorOp::Minimum => vec![exec.minimum(inputs[0], inputs[1])?],
            StdTensorOp::Compare(dir) => vec![exec.compare(inputs[0], inputs[1], dir)?],
            StdTensorOp::Transpose { perm } => vec![exec.transpose(inputs[0], perm)?],
            StdTensorOp::ReduceSum { axes, .. } => vec![exec.reduce_sum(inputs[0], axes)?],
            StdTensorOp::DotGeneral { config, .. } => {
                vec![exec.dot_general(inputs[0], inputs[1], config)?]
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
            StdTensorOp::Select => vec![exec.select(inputs[0], inputs[1], inputs[2])?],
            StdTensorOp::Clamp => vec![exec.clamp(inputs[0], inputs[1], inputs[2])?],
            StdTensorOp::Concatenate { axis, .. } => {
                vec![exec.concatenate(inputs, *axis)?]
            }
            StdTensorOp::NaryEinsum { .. } => {
                unreachable!("NaryEinsum is handled before opening an exec session")
            }
            StdTensorOp::Gather(config) => vec![exec.gather(inputs[0], inputs[1], config)?],
            StdTensorOp::Scatter(config) => {
                vec![exec.scatter(inputs[0], inputs[1], inputs[2], config)?]
            }
            StdTensorOp::DynamicSlice { slice_sizes } => {
                vec![exec.dynamic_slice(inputs[0], inputs[1], slice_sizes)?]
            }
            StdTensorOp::Cholesky { .. } => vec![exec.cholesky(inputs[0])?],
            StdTensorOp::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
                ..
            } => {
                vec![exec.triangular_solve(
                    inputs[0],
                    inputs[1],
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
                vec![exec.full_piv_lu_solve(inputs[0], inputs[1], *transpose_a)?]
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
                vec![Tensor::F64(TypedTensor::from_vec(vec![], vec![size]))]
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
                let size_f64 = match size_tensor {
                    Tensor::F64(inner) => inner.host_data()[0],
                    Tensor::F32(inner) => inner.host_data()[0] as f64,
                    _ => {
                        return Err(Error::Internal(
                            "DynamicTruncate size must be an f32 or f64 scalar".into(),
                        ))
                    }
                };
                let rounded_size = if size_f64.is_finite() {
                    size_f64.round()
                } else {
                    0.0
                };
                let size = rounded_size.max(0.0).min(axis_extent as f64) as usize;
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
        DType::F64 => Tensor::F64(TypedTensor::from_vec(
            vec![],
            vec![f64::from_le_bytes(exact_bytes::<8>(dtype, bytes))],
        )),
        DType::F32 => Tensor::F32(TypedTensor::from_vec(
            vec![],
            vec![f32::from_le_bytes(exact_bytes::<4>(dtype, bytes))],
        )),
        DType::I64 => Tensor::I64(TypedTensor::from_vec(
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
            Tensor::C64(TypedTensor::from_vec(vec![], vec![Complex64::new(re, im)]))
        }
        DType::C32 => {
            let data = exact_bytes::<8>(dtype, bytes);
            let mut re_bytes = [0u8; 4];
            let mut im_bytes = [0u8; 4];
            re_bytes.copy_from_slice(&data[..4]);
            im_bytes.copy_from_slice(&data[4..]);
            let re = f32::from_le_bytes(re_bytes);
            let im = f32::from_le_bytes(im_bytes);
            Tensor::C32(TypedTensor::from_vec(vec![], vec![Complex32::new(re, im)]))
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
