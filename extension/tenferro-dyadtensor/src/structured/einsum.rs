use std::collections::{HashMap, HashSet};

use chainrules_core::Differentiable as _;
use tenferro_algebra::{HasAlgebra, Scalar, Standard};
use tenferro_einsum::{self as tf_einsum, Subscripts};
use tenferro_prims::{CpuBackend, CpuContext, TensorPrims};
use tenferro_tensor::Tensor;

use crate::api::with_runtime_cpu_only;
use crate::{Error, Result};

use super::meta::{plan_axis_classes_for_subscripts, OperandAxisClasses};
use super::StructuredTensor;

impl<T> StructuredTensor<T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    /// Materialize this structured tensor into a dense payload tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Requires default runtime to be configured.
    /// let dense = structured.to_dense()?;
    /// ```
    pub fn to_dense(&self) -> Result<Tensor<T>> {
        if self.is_dense() {
            return Ok(self.payload().clone());
        }

        with_runtime_cpu_only("structured_to_dense", |ctx| to_dense_in_ctx(ctx, self))
    }

    /// Contract/einsum structured operands while preserving compressed metadata.
    ///
    /// `subscripts.inputs` rank must match each operand logical rank.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Requires default runtime to be configured.
    /// let out = StructuredTensor::einsum_with_subscripts(&subs, &[&a, &b])?;
    /// ```
    pub fn einsum_with_subscripts(subscripts: &Subscripts, operands: &[&Self]) -> Result<Self> {
        if operands.is_empty() {
            return Err(Error::InvalidAdTensor {
                message: "structured einsum requires at least one operand".to_string(),
            });
        }

        with_runtime_cpu_only("structured_einsum", |ctx| {
            einsum_with_subscripts_in_ctx(ctx, subscripts, operands)
        })
    }
}

pub(crate) fn to_dense_in_ctx<T>(
    ctx: &mut CpuContext,
    tensor: &StructuredTensor<T>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    if tensor.is_dense() {
        return Ok(tensor.payload().clone());
    }

    let input_labels = usize_vec_to_u32(&(0..tensor.payload().dims().len()).collect::<Vec<_>>())?;
    let output_labels = usize_vec_to_u32(tensor.axis_classes())?;
    let inputs = [input_labels.as_slice()];
    let subs = Subscripts::new(&inputs, &output_labels);
    let out = tf_einsum::einsum_with_subscripts::<Standard<T>, CpuBackend>(
        ctx,
        &subs,
        &[tensor.payload()],
        None,
    )
    .map_err(Error::from)?;
    if out.dims() != tensor.logical_dims() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "structured_to_dense output shape mismatch: expected {:?}, got {:?}",
                tensor.logical_dims(),
                out.dims()
            ),
        });
    }
    Ok(out)
}

pub(crate) fn compress_dense_to_layout_in_ctx<T>(
    ctx: &mut CpuContext,
    dense: &Tensor<T>,
    layout: &StructuredTensor<T>,
) -> Result<StructuredTensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    if dense.dims() != layout.logical_dims() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "structured compression shape mismatch: expected {:?}, got {:?}",
                layout.logical_dims(),
                dense.dims()
            ),
        });
    }
    if layout.is_dense() {
        return Ok(StructuredTensor::from_dense(dense.clone()));
    }

    let input_labels = usize_vec_to_u32(layout.axis_classes())?;
    let output_labels = usize_vec_to_u32(&(0..layout.class_count()).collect::<Vec<_>>())?;
    let inputs = [input_labels.as_slice()];
    let subs = Subscripts::new(&inputs, &output_labels);
    let payload =
        tf_einsum::einsum_with_subscripts::<Standard<T>, CpuBackend>(ctx, &subs, &[dense], None)
            .map_err(Error::from)?;
    layout.with_payload_like(payload)
}

pub(crate) fn einsum_with_subscripts_in_ctx<T>(
    ctx: &mut CpuContext,
    subscripts: &Subscripts,
    operands: &[&StructuredTensor<T>],
) -> Result<StructuredTensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    let operand_meta: Vec<OperandAxisClasses> = operands
        .iter()
        .map(|operand| {
            OperandAxisClasses::new(
                operand.logical_dims().to_vec(),
                operand.axis_classes().to_vec(),
            )
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| Error::InvalidAdTensor {
            message: format!("invalid structured operand metadata: {e}"),
        })?;
    let plan = plan_axis_classes_for_subscripts(&operand_meta, subscripts).map_err(|e| {
        Error::InvalidAdTensor {
            message: format!("failed to plan structured einsum: {e}"),
        }
    })?;

    let mut normalized_payloads: Vec<Tensor<T>> = Vec::with_capacity(operands.len());
    let mut normalized_roots: Vec<Vec<usize>> = Vec::with_capacity(operands.len());

    for (operand_idx, operand) in operands.iter().enumerate() {
        let class_roots = &plan.operand_plans[operand_idx].class_roots;
        if operand.payload().dims().len() != class_roots.len() {
            return Err(Error::InvalidAdTensor {
                message: format!(
                    "operand {} payload rank {} does not match planned local class count {}",
                    operand_idx,
                    operand.payload().dims().len(),
                    class_roots.len()
                ),
            });
        }
        let (normalized, roots) = normalize_payload_for_roots(ctx, operand.payload(), class_roots)?;
        normalized_payloads.push(normalized);
        normalized_roots.push(roots);
    }

    let input_labels_u32: Vec<Vec<u32>> = normalized_roots
        .iter()
        .map(|roots| usize_vec_to_u32(roots))
        .collect::<Result<_>>()?;
    let output_labels_u32 = usize_vec_to_u32(&plan.output_compressed_roots)?;
    let input_refs: Vec<&[u32]> = input_labels_u32.iter().map(Vec::as_slice).collect();
    let payload_refs: Vec<&Tensor<T>> = normalized_payloads.iter().collect();
    let backend_subs = Subscripts::new(&input_refs, &output_labels_u32);

    let compressed_output = tf_einsum::einsum_with_subscripts::<Standard<T>, CpuBackend>(
        ctx,
        &backend_subs,
        &payload_refs,
        None,
    )
    .map_err(Error::from)?;

    StructuredTensor::new(
        plan.output_dims.clone(),
        plan.output_axis_classes.clone(),
        compressed_output,
    )
}

pub(crate) fn accumulate_tangent<T>(
    lhs: StructuredTensor<T>,
    rhs: &StructuredTensor<T>,
) -> Result<StructuredTensor<T>>
where
    T: Scalar,
{
    if lhs.logical_dims() != rhs.logical_dims() || lhs.axis_classes() != rhs.axis_classes() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "structured tangent layout mismatch: lhs dims {:?} classes {:?}, rhs dims {:?} classes {:?}",
                lhs.logical_dims(),
                lhs.axis_classes(),
                rhs.logical_dims(),
                rhs.axis_classes(),
            ),
        });
    }

    let logical_dims = lhs.logical_dims().to_vec();
    let axis_classes = lhs.axis_classes().to_vec();
    let payload = Tensor::<T>::accumulate_tangent(lhs.into_payload(), rhs.payload());
    StructuredTensor::new(logical_dims, axis_classes, payload)
}

pub(crate) fn reverse_subscripts(subscripts: &Subscripts, input_idx: usize) -> Subscripts {
    let mut rev_inputs = vec![subscripts.output.clone()];
    for (idx, input) in subscripts.inputs.iter().enumerate() {
        if idx != input_idx {
            rev_inputs.push(input.clone());
        }
    }
    Subscripts {
        inputs: rev_inputs,
        output: subscripts.inputs[input_idx].clone(),
    }
}

fn unique_ids_first_appearance(ids: &[usize]) -> Vec<usize> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for &id in ids {
        if seen.insert(id) {
            out.push(id);
        }
    }
    out
}

fn first_duplicate_pair(ids: &[usize]) -> Option<(usize, usize)> {
    let mut first_pos: HashMap<usize, usize> = HashMap::new();
    for (pos, &id) in ids.iter().enumerate() {
        if let Some(&first) = first_pos.get(&id) {
            return Some((first, pos));
        }
        first_pos.insert(id, pos);
    }
    None
}

fn normalize_payload_for_roots<T>(
    ctx: &mut CpuContext,
    payload: &Tensor<T>,
    roots: &[usize],
) -> Result<(Tensor<T>, Vec<usize>)>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    if payload.dims().len() != roots.len() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "payload rank {} must match roots length {}",
                payload.dims().len(),
                roots.len()
            ),
        });
    }
    if unique_ids_first_appearance(roots).len() == roots.len() {
        return Ok((payload.clone(), roots.to_vec()));
    }

    let mut current_payload = payload.clone();
    let mut current_roots = roots.to_vec();
    let mut round = 0u32;

    while let Some((pos_a, pos_b)) = first_duplicate_pair(&current_roots) {
        let rank = current_roots.len();
        debug_assert!(
            pos_b < rank,
            "pos_b ({}) must be less than current_roots length ({})",
            pos_b,
            rank
        );
        let base = 1_000_000u32.saturating_add(round.saturating_mul(10_000));
        let mut input_labels: Vec<u32> = (0..rank).map(|i| base + i as u32).collect();
        input_labels[pos_b] = input_labels[pos_a];
        let output_labels: Vec<u32> = input_labels
            .iter()
            .enumerate()
            .filter_map(|(axis, &label)| (axis != pos_b).then_some(label))
            .collect();
        let inputs = [input_labels.as_slice()];
        let subs = Subscripts::new(&inputs, &output_labels);
        current_payload = tf_einsum::einsum_with_subscripts::<Standard<T>, CpuBackend>(
            ctx,
            &subs,
            &[&current_payload],
            None,
        )
        .map_err(Error::from)?;
        current_roots.remove(pos_b);
        round = round.saturating_add(1);
    }

    Ok((current_payload, current_roots))
}

fn usize_vec_to_u32(values: &[usize]) -> Result<Vec<u32>> {
    values
        .iter()
        .map(|&v| {
            u32::try_from(v).map_err(|_| Error::InvalidAdTensor {
                message: format!("label id {} does not fit into u32", v),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests;
