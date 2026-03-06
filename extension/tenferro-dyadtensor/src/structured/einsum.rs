use std::collections::{HashMap, HashSet};

use chainrules_core::Differentiable as _;
use tenferro_algebra::{HasAlgebra, Scalar, Standard};
use tenferro_einsum::{self as tf_einsum, Subscripts};
use tenferro_prims::{CpuBackend, CpuContext, TensorPrims};
use tenferro_tensor::Tensor;

use crate::runtime::{with_default_runtime, RuntimeContext};
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

        with_cpu_runtime("structured_to_dense", |ctx| to_dense_in_ctx(ctx, self))
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

        with_cpu_runtime("structured_einsum", |ctx| {
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

fn with_cpu_runtime<R>(
    op: &'static str,
    f: impl FnOnce(&mut CpuContext) -> Result<R>,
) -> Result<R> {
    with_default_runtime(|runtime| match runtime {
        RuntimeContext::Cpu(ctx) => f(ctx),
        RuntimeContext::Cuda(_) => Err(Error::UnsupportedRuntimeOp {
            op,
            runtime: "cuda",
        }),
        RuntimeContext::Rocm(_) => Err(Error::UnsupportedRuntimeOp {
            op,
            runtime: "rocm",
        }),
    })
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
mod tests {
    use super::*;
    use crate::{set_default_runtime, RuntimeContext};
    use tenferro_prims::{CpuContext, CudaContext};
    use tenferro_tensor::MemoryOrder;

    fn vector(values: &[f64]) -> Tensor<f64> {
        Tensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
    }

    fn matrix(values: &[f64], rows: usize, cols: usize) -> Tensor<f64> {
        Tensor::<f64>::from_slice(values, &[rows, cols], MemoryOrder::ColumnMajor).unwrap()
    }

    fn as_slice(tensor: &Tensor<f64>) -> &[f64] {
        tensor
            .buffer()
            .as_slice()
            .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"))
    }

    #[test]
    fn public_structured_ops_cover_cpu_and_runtime_error_paths() {
        let diag = StructuredTensor::from_diagonal_vector(vector(&[2.0, 3.0]), 2).unwrap();
        let dense_layout = StructuredTensor::from_dense(matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2));
        let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);

        {
            let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
            assert_eq!(dense_layout.to_dense().unwrap().dims(), &[2, 2]);
            assert_eq!(as_slice(&diag.to_dense().unwrap()), &[2.0, 0.0, 0.0, 3.0]);

            let out = StructuredTensor::einsum_with_subscripts(&subs, &[&diag, &diag]).unwrap();
            assert!(out.is_diag());
            assert_eq!(as_slice(out.payload()), &[4.0, 9.0]);

            let err = StructuredTensor::<f64>::einsum_with_subscripts(&subs, &[]).unwrap_err();
            assert!(matches!(err, Error::InvalidAdTensor { .. }));
        }

        {
            let _guard = set_default_runtime(RuntimeContext::Cuda(CudaContext::new()));
            let err = diag.to_dense().unwrap_err();
            assert!(matches!(
                err,
                Error::UnsupportedRuntimeOp {
                    op: "structured_to_dense",
                    runtime: "cuda"
                }
            ));

            let err = StructuredTensor::einsum_with_subscripts(&subs, &[&diag, &diag]).unwrap_err();
            assert!(matches!(
                err,
                Error::UnsupportedRuntimeOp {
                    op: "structured_einsum",
                    runtime: "cuda"
                }
            ));
        }
    }

    #[test]
    fn compression_and_tangent_accumulation_cover_dense_and_diag_paths() {
        let mut ctx = CpuContext::new(1);
        let dense_layout = StructuredTensor::from_dense(matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2));
        let diag_layout = StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap();

        let dense =
            compress_dense_to_layout_in_ctx(&mut ctx, dense_layout.payload(), &dense_layout)
                .unwrap();
        assert!(dense.is_dense());

        let diag = compress_dense_to_layout_in_ctx(
            &mut ctx,
            &matrix(&[5.0, 7.0, 11.0, 13.0], 2, 2),
            &diag_layout,
        )
        .unwrap();
        assert_eq!(as_slice(diag.payload()), &[5.0, 13.0]);

        let err = compress_dense_to_layout_in_ctx(&mut ctx, &vector(&[1.0, 2.0]), &diag_layout)
            .unwrap_err();
        assert!(matches!(err, Error::InvalidAdTensor { .. }));

        let diag_rhs = diag_layout.with_payload_like(vector(&[3.0, 4.0])).unwrap();
        let sum = accumulate_tangent(diag_layout.clone(), &diag_rhs).unwrap();
        assert_eq!(as_slice(sum.payload()), &[4.0, 6.0]);

        let err = accumulate_tangent(diag_layout, &dense_layout).unwrap_err();
        assert!(matches!(err, Error::InvalidAdTensor { .. }));
    }

    #[test]
    fn internal_helpers_cover_normalization_and_error_branches() {
        let mut ctx = CpuContext::new(1);

        assert_eq!(unique_ids_first_appearance(&[3, 3, 1, 3, 1]), vec![3, 1]);
        assert_eq!(first_duplicate_pair(&[4, 5, 4]), Some((0, 2)));
        assert_eq!(first_duplicate_pair(&[4, 5, 6]), None);

        let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
        let rev = reverse_subscripts(&subs, 1);
        assert_eq!(rev.inputs, vec![vec![0, 2], vec![0, 1]]);
        assert_eq!(rev.output, vec![1, 2]);

        let (same_payload, same_roots) =
            normalize_payload_for_roots(&mut ctx, &vector(&[1.0, 2.0]), &[0]).unwrap();
        assert_eq!(same_roots, vec![0]);
        assert_eq!(as_slice(&same_payload), &[1.0, 2.0]);

        let (diag_payload, diag_roots) =
            normalize_payload_for_roots(&mut ctx, &matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2), &[0, 0])
                .unwrap();
        assert_eq!(diag_roots, vec![0]);
        assert_eq!(as_slice(&diag_payload), &[1.0, 4.0]);

        let err = normalize_payload_for_roots(&mut ctx, &vector(&[1.0, 2.0]), &[0, 0]).unwrap_err();
        assert!(matches!(err, Error::InvalidAdTensor { .. }));

        let rank1_subs = Subscripts::new(&[&[0]], &[0]);
        let diag = StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap();
        let err = einsum_with_subscripts_in_ctx(&mut ctx, &rank1_subs, &[&diag]).unwrap_err();
        assert!(matches!(err, Error::InvalidAdTensor { .. }));

        let err = usize_vec_to_u32(&[usize::MAX]).unwrap_err();
        assert!(matches!(err, Error::InvalidAdTensor { .. }));
    }
}
