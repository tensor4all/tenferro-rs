use crate::core::AdMode;
use ::tidu::{NodeId as ChainNodeId, Tape};
use tenferro_prims::TensorTempPoolContext;

use super::*;
use crate::{DynTensor, DynTensorTyped};
use tenferro_device::LogicalMemorySpace;

pub(crate) fn has_forward<S: Scalar>(operands: &[&AdTensor<S>]) -> bool {
    operands.iter().any(|op| op.mode() == AdMode::Forward)
}

pub(crate) fn has_reverse<S: Scalar>(operands: &[&AdTensor<S>]) -> bool {
    operands.iter().any(|op| op.requires_grad())
}

pub(crate) fn has_any_tangent<S: Scalar>(operands: &[&AdTensor<S>]) -> bool {
    operands.iter().any(|op| op.tangent().is_some())
}

pub(crate) fn ensure_dense_linalg_inputs<S: Scalar>(
    op: &'static str,
    operands: &[&AdTensor<S>],
) -> Result<()> {
    if operands.iter().any(|op| !op.is_dense()) {
        return Err(Error::UnsupportedStructuredLinalg { op });
    }

    Ok(())
}

pub(crate) fn derive_reverse_tape_handle<S: Scalar + DynTensorTyped>(
    operands: &[&AdTensor<S>],
) -> Result<Option<Tape<DynTensor>>> {
    let mut tape: Option<Tape<DynTensor>> = None;

    for op in operands {
        if let Some(current) = op.reverse_tape() {
            if let Some(expected) = &tape {
                if !expected.same_tape(&current as &Tape<DynTensor>) {
                    return Err(Error::MixedReverseTape {
                        expected: expected.id() as u64,
                        found: current.id() as u64,
                    });
                }
            } else {
                tape = Some(current);
            }
        }
    }

    if operands.iter().any(|op| op.requires_grad()) {
        let tape = tape.unwrap_or_else(Tape::new);
        for op in operands {
            op.ensure_reverse_leaf_on(&tape)?;
        }
        return Ok(Some(tape));
    }

    Ok(None)
}

#[derive(Clone)]
pub(crate) struct ReverseInputSpec<T: Scalar> {
    pub(crate) node: ChainNodeId,
    pub(crate) layout: StructuredTensor<T>,
}

pub(crate) fn collect_reverse_input_specs<S: Scalar>(
    operands: &[&AdTensor<S>],
) -> Vec<Option<ReverseInputSpec<S>>> {
    operands
        .iter()
        .map(|op| match op.reverse_node_id() {
            Some(node) => Some(ReverseInputSpec {
                node,
                layout: op.structured_primal().clone(),
            }),
            None => None,
        })
        .collect()
}

pub(crate) fn normalize_pullback_shape<T: Scalar>(
    grad: Tensor<T>,
    expected_dims: &[usize],
    op_name: &'static str,
) -> Result<Tensor<T>> {
    if grad.dims() == expected_dims {
        return Ok(grad);
    }

    let expected_len: usize = expected_dims.iter().product();
    if grad.len() != expected_len {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "{op_name} pullback shape mismatch: expected {:?} (len={expected_len}), got {:?} (len={})",
                expected_dims,
                grad.dims(),
                grad.len()
            ),
        });
    }

    grad.reshape(expected_dims)
        .map_err(|e| Error::InvalidAdTensor {
            message: format!(
                "{op_name} pullback reshape failed from {:?} to {:?}: {e}",
                grad.dims(),
                expected_dims
            ),
        })
}

pub(crate) fn normalize_output_tangent_shape<T: Scalar>(
    tangent: Tensor<T>,
    expected_dims: &[usize],
    op_name: &'static str,
) -> Result<Tensor<T>> {
    if tangent.dims() == expected_dims {
        return Ok(tangent);
    }

    let expected_len: usize = expected_dims.iter().product();
    if tangent.len() != expected_len {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "{op_name} tangent shape mismatch: expected {:?} (len={expected_len}), got {:?} (len={})",
                expected_dims,
                tangent.dims(),
                tangent.len()
            ),
        });
    }

    tangent
        .reshape(expected_dims)
        .map_err(|e| Error::InvalidAdTensor {
            message: format!(
                "{op_name} tangent reshape failed from {:?} to {:?}: {e}",
                tangent.dims(),
                expected_dims
            ),
        })
}

pub(crate) fn dense_input_snapshot_in_backend<B, C, T>(
    ctx: &mut C,
    input: &AdTensor<T>,
    needs_tangent: bool,
) -> Result<(Tensor<T>, Option<Tensor<T>>)>
where
    T: EinsumRuntimeValue,
    B: DenseEinsumBackend<T, C>,
    C: TensorTempPoolContext,
{
    let primal = to_dense_in_ctx::<B, _, T>(ctx, input.structured_primal())?
        .contiguous(MemoryOrder::ColumnMajor);
    let tangent = if needs_tangent {
        Some(match input.structured_tangent() {
            Some(tangent) => {
                to_dense_in_ctx::<B, _, T>(ctx, tangent)?.contiguous(MemoryOrder::ColumnMajor)
            }
            None => zero_like(&primal)?,
        })
    } else {
        None
    };
    Ok((primal, tangent))
}

pub(crate) fn dense_input_snapshot_in_runtime<T>(
    op_name: &'static str,
    input: &AdTensor<T>,
    needs_tangent: bool,
) -> Result<(Tensor<T>, Option<Tensor<T>>)>
where
    T: EinsumRuntimeValue,
{
    dispatch_einsum_runtime!(T, op_name, |ctx, Backend| {
        dense_input_snapshot_in_backend::<Backend, _, T>(ctx, input, needs_tangent)
    })
}

pub(crate) fn compress_pullback_like<T>(
    op_name: &'static str,
    grad: Tensor<T>,
    layout: &StructuredTensor<T>,
) -> Result<Tensor<T>>
where
    T: EinsumRuntimeValue,
{
    let dense = normalize_pullback_shape(grad, layout.logical_dims(), op_name)?;
    if layout.is_dense() {
        return Ok(dense);
    }

    dispatch_einsum_runtime!(T, op_name, |ctx, Backend| {
        compress_pullback_like_in_backend::<Backend, _, T>(ctx, op_name, dense.clone(), layout)
    })
}

pub(crate) fn compress_structured_pullback_like<T>(
    op_name: &'static str,
    grad: Tensor<T>,
    layout: &StructuredTensor<T>,
) -> Result<StructuredTensor<T>>
where
    T: EinsumRuntimeValue,
{
    let payload = compress_pullback_like(op_name, grad, layout)?;
    layout.with_payload_like(payload)
}

pub(crate) fn compress_pullback_like_in_backend<B, C, T>(
    ctx: &mut C,
    op_name: &'static str,
    grad: Tensor<T>,
    layout: &StructuredTensor<T>,
) -> Result<Tensor<T>>
where
    T: EinsumRuntimeValue,
    B: DenseEinsumBackend<T, C>,
    C: TensorTempPoolContext,
{
    let dense = normalize_pullback_shape(grad, layout.logical_dims(), op_name)?;
    if layout.is_dense() {
        return Ok(dense);
    }

    compress_dense_to_layout_in_ctx::<B, _, T>(ctx, &dense, layout)
        .map(StructuredTensor::into_payload)
}

pub(crate) fn zero_like<T: Scalar>(tensor: &Tensor<T>) -> Result<Tensor<T>> {
    Ok(Tensor::zeros(
        tensor.dims(),
        tensor.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )?)
}

pub(crate) fn scalar_from_rank0_tensor<T>(tensor: &Tensor<T>, op_name: &'static str) -> Result<T>
where
    T: Scalar + Copy,
{
    if !tensor.dims().is_empty() || tensor.len() != 1 {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "{op_name} expects rank-0 cotangent, got dims {:?}",
                tensor.dims()
            ),
        });
    }
    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let contiguous = if contiguous.logical_memory_space() == LogicalMemorySpace::MainMemory {
        contiguous
    } else {
        contiguous
            .to_memory_space_async(LogicalMemorySpace::MainMemory)
            .map_err(Error::from)?
    };
    let off = contiguous.offset() as usize;
    contiguous
        .buffer()
        .as_slice()
        .and_then(|values| values.get(off))
        .copied()
        .ok_or_else(|| Error::InvalidAdTensor {
            message: format!("{op_name} could not materialize rank-0 cotangent on host memory"),
        })
}

pub(crate) fn broadcast_scalar_like<T>(value: T, template: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Scalar + Copy,
{
    let len = template.len();
    let data = vec![value; len];
    let dense = Tensor::from_slice(&data, template.dims(), MemoryOrder::ColumnMajor)
        .map_err(Error::from)?;
    if template.logical_memory_space() == LogicalMemorySpace::MainMemory {
        Ok(dense)
    } else {
        dense
            .to_memory_space_async(template.logical_memory_space())
            .map_err(Error::from)
    }
}

pub(crate) fn wrap_dense_ad_output<TIn: Scalar, TOut: Scalar>(
    op_name: &'static str,
    inputs: &[&AdTensor<TIn>],
    primal: Tensor<TOut>,
    tangent: Option<Tensor<TOut>>,
) -> Result<AdTensor<TOut>> {
    let tangent = tangent
        .map(|tangent| normalize_output_tangent_shape(tangent, primal.dims(), op_name))
        .transpose()?;
    wrap_structured_ad_output(
        op_name,
        inputs,
        StructuredTensor(tenferro_tensor::StructuredTensor::from_dense(primal)),
        tangent.map(|value| StructuredTensor(tenferro_tensor::StructuredTensor::from_dense(value))),
    )
}

pub(crate) fn wrap_structured_ad_output<TIn: Scalar, TOut: Scalar>(
    op_name: &'static str,
    inputs: &[&AdTensor<TIn>],
    primal: StructuredTensor<TOut>,
    tangent: Option<StructuredTensor<TOut>>,
) -> Result<AdTensor<TOut>> {
    if has_reverse(inputs) {
        return Err(Error::UnsupportedAdOp { op: op_name });
    }

    if has_forward(inputs) {
        let tangent = tangent.ok_or_else(|| Error::InvalidAdTensor {
            message: "forward-mode inputs must provide tangent output".to_string(),
        })?;
        return AdTensor::new_forward(primal, tangent);
    }

    Ok(AdTensor::new_primal(primal))
}

pub(crate) fn wrap_same_type_structured_ad_output<T: Scalar + DynTensorTyped>(
    _op_name: &'static str,
    inputs: &[&AdTensor<T>],
    primal: StructuredTensor<T>,
    tangent: Option<StructuredTensor<T>>,
) -> Result<AdTensor<T>> {
    if has_reverse(inputs) {
        let tape = derive_reverse_tape_handle(inputs)?.ok_or_else(|| Error::InvalidAdTensor {
            message: "reverse-mode output requested but no reverse tape found".to_string(),
        })?;
        return AdTensor::new_reverse_output(primal, &tape, tangent);
    }

    if has_forward(inputs) {
        let tangent = tangent.ok_or_else(|| Error::InvalidAdTensor {
            message: "forward-mode inputs must provide tangent output".to_string(),
        })?;
        return AdTensor::new_forward(primal, tangent);
    }

    Ok(AdTensor::new_primal(primal))
}

pub(crate) fn wrap_same_type_dense_ad_output<T: Scalar + DynTensorTyped>(
    op_name: &'static str,
    inputs: &[&AdTensor<T>],
    primal: Tensor<T>,
    tangent: Option<Tensor<T>>,
) -> Result<AdTensor<T>> {
    let tangent = tangent
        .map(|tangent| normalize_output_tangent_shape(tangent, primal.dims(), op_name))
        .transpose()?;
    wrap_same_type_structured_ad_output(
        op_name,
        inputs,
        StructuredTensor(tenferro_tensor::StructuredTensor::from_dense(primal)),
        tangent.map(|value| StructuredTensor(tenferro_tensor::StructuredTensor::from_dense(value))),
    )
}

pub(crate) fn collect_structured_ad_tangents<'a, S: Scalar>(
    operands: &[&'a AdTensor<S>],
) -> Vec<Option<&'a StructuredTensor<S>>> {
    operands.iter().map(|op| op.structured_tangent()).collect()
}

pub(crate) fn collect_reverse_input_nodes<S: Scalar>(
    operands: &[&AdTensor<S>],
) -> Vec<Option<ChainNodeId>> {
    operands.iter().map(|op| op.reverse_node_id()).collect()
}

pub(crate) fn sum_einsum_tangent_terms<B, C, T>(
    ctx: &mut C,
    subscripts: &str,
    primals: &[&Tensor<T>],
    tangents: &[Option<&Tensor<T>>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Option<Tensor<T>>>
where
    T: EinsumRuntimeValue,
    B: DenseEinsumBackend<T, C>,
    C: TensorTempPoolContext,
{
    let mut out_tangent: Option<Tensor<T>> = None;

    for (k, tangent_opt) in tangents.iter().enumerate() {
        let Some(tangent_k) = tangent_opt else {
            continue;
        };

        let mut term_operands: Vec<&Tensor<T>> = primals.to_vec();
        term_operands[k] = tangent_k;
        let term = tf_einsum::einsum::<Standard<T>, B>(ctx, subscripts, &term_operands, size_dict)
            .map_err(Error::from)?;

        out_tangent = Some(match out_tangent {
            None => term,
            Some(existing) => Tensor::<T>::accumulate_tangent(existing, &term),
        });
    }

    Ok(out_tangent)
}

pub(crate) fn sum_structured_einsum_tangent_terms<B, C, T>(
    ctx: &mut C,
    subscripts: &Subscripts,
    primals: &[&StructuredTensor<T>],
    tangents: &[Option<&StructuredTensor<T>>],
) -> Result<Option<StructuredTensor<T>>>
where
    T: EinsumRuntimeValue,
    B: DenseEinsumBackend<T, C>,
    C: TensorTempPoolContext,
{
    let mut out_tangent: Option<StructuredTensor<T>> = None;

    for (k, tangent_opt) in tangents.iter().enumerate() {
        let Some(tangent_k) = tangent_opt else {
            continue;
        };

        let mut term_operands: Vec<&StructuredTensor<T>> = primals.to_vec();
        term_operands[k] = tangent_k;
        let term = einsum_with_subscripts_in_ctx::<B, _, T>(ctx, subscripts, &term_operands)?;

        out_tangent = Some(match out_tangent {
            None => term,
            Some(existing) => accumulate_structured_tangent(existing, &term)?,
        });
    }

    Ok(out_tangent)
}
