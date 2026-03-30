use super::super::*;
use tenferro_prims::TensorTempPoolContext;

pub(crate) fn structured_einsum_input_grads_in_backend<B, C, T>(
    ctx: &mut C,
    subscripts: &Subscripts,
    primals: &[StructuredTensor<T>],
    cotangent: &StructuredTensor<T>,
    input_grad_mask: &[bool],
) -> Result<Vec<Option<StructuredTensor<T>>>>
where
    T: EinsumRuntimeValue,
    B: DenseEinsumBackend<T, C>,
    C: TensorTempPoolContext,
{
    let size_dict: std::collections::HashMap<u32, usize> = {
        let mut sd = std::collections::HashMap::new();
        for (i, input_labels) in subscripts.inputs.iter().enumerate() {
            let dims = primals[i].logical_dims();
            for (j, &label) in input_labels.iter().enumerate() {
                sd.entry(label).or_insert(dims[j]);
            }
        }
        sd
    };

    let mut input_grads = vec![None; primals.len()];

    for (k, _primal) in primals.iter().enumerate() {
        if !input_grad_mask[k] {
            continue;
        }

        let mut rev_subs = reverse_subscripts(subscripts, k);
        let mut conj_store: Vec<StructuredTensor<T>> = Vec::new();
        for (idx, operand) in primals.iter().enumerate() {
            if idx != k {
                conj_store.push(operand.conj());
            }
        }

        let all_input_labels: std::collections::HashSet<u32> = rev_subs
            .inputs
            .iter()
            .flat_map(|labels| labels.iter().copied())
            .collect();
        let mut unique_missing = Vec::new();
        {
            let mut seen = std::collections::HashSet::new();
            for &label in &rev_subs.output {
                if !all_input_labels.contains(&label) && seen.insert(label) {
                    unique_missing.push(label);
                }
            }
        }

        let mut delta_tensors: Vec<StructuredTensor<T>> = Vec::new();
        for &label in &unique_missing {
            let dim = size_dict
                .get(&label)
                .copied()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: format!(
                        "einsum structured pullback: missing dimension for label {}",
                        label
                    ),
                })?;
            let mut data = vec![T::zero(); dim * dim];
            for i in 0..dim {
                data[i * dim + i] = T::one();
            }
            let eye = tenferro_tensor::Tensor::from_slice(
                &data,
                &[dim, dim],
                tenferro_tensor::MemoryOrder::ColumnMajor,
            )
            .map_err(Error::from)?;
            delta_tensors.push(StructuredTensor(
                tenferro_tensor::StructuredTensor::from_dense(eye),
            ));
            rev_subs.inputs.push(vec![label, label]);
        }

        let mut rev_operands: Vec<&StructuredTensor<T>> = Vec::with_capacity(primals.len());
        rev_operands.push(cotangent);
        for c in &conj_store {
            rev_operands.push(c);
        }
        for dt in &delta_tensors {
            rev_operands.push(dt);
        }

        let grad = einsum_with_subscripts_in_ctx::<B, _, T>(ctx, &rev_subs, &rev_operands)?;
        input_grads[k] = Some(grad);
    }

    Ok(input_grads)
}

pub(crate) fn structured_einsum_pullback_in_backend<B, C, T>(
    ctx: &mut C,
    subscripts: &Subscripts,
    reverse_nodes: &[Option<NodeId>],
    primals: &[StructuredTensor<T>],
    cotangent: &StructuredTensor<T>,
) -> Result<Vec<(NodeId, StructuredTensor<T>)>>
where
    T: EinsumRuntimeValue,
    B: DenseEinsumBackend<T, C>,
    C: TensorTempPoolContext,
{
    let size_dict: std::collections::HashMap<u32, usize> = {
        let mut sd = std::collections::HashMap::new();
        for (i, input_labels) in subscripts.inputs.iter().enumerate() {
            let dims = primals[i].logical_dims();
            for (j, &label) in input_labels.iter().enumerate() {
                sd.entry(label).or_insert(dims[j]);
            }
        }
        sd
    };

    let mut input_grads = Vec::new();

    for (k, maybe_node) in reverse_nodes.iter().enumerate() {
        let Some(node) = maybe_node else {
            continue;
        };
        let mut rev_subs = reverse_subscripts(subscripts, k);
        let mut conj_store: Vec<StructuredTensor<T>> = Vec::new();
        for (idx, operand) in primals.iter().enumerate() {
            if idx != k {
                conj_store.push(operand.conj());
            }
        }

        let all_input_labels: std::collections::HashSet<u32> = rev_subs
            .inputs
            .iter()
            .flat_map(|labels| labels.iter().copied())
            .collect();
        let mut unique_missing = Vec::new();
        {
            let mut seen = std::collections::HashSet::new();
            for &label in &rev_subs.output {
                if !all_input_labels.contains(&label) && seen.insert(label) {
                    unique_missing.push(label);
                }
            }
        }
        let mut delta_tensors: Vec<StructuredTensor<T>> = Vec::new();
        for &label in &unique_missing {
            let dim = size_dict
                .get(&label)
                .copied()
                .ok_or_else(|| Error::InvalidAdTensor {
                    message: format!(
                        "einsum structured pullback: missing dimension for label {}",
                        label
                    ),
                })?;
            let mut data = vec![T::zero(); dim * dim];
            for i in 0..dim {
                data[i * dim + i] = T::one();
            }
            let eye = tenferro_tensor::Tensor::from_slice(
                &data,
                &[dim, dim],
                tenferro_tensor::MemoryOrder::ColumnMajor,
            )
            .map_err(Error::from)?;
            delta_tensors.push(StructuredTensor(
                tenferro_tensor::StructuredTensor::from_dense(eye),
            ));
            rev_subs.inputs.push(vec![label, label]);
        }

        let mut rev_operands: Vec<&StructuredTensor<T>> = Vec::with_capacity(primals.len());
        rev_operands.push(cotangent);
        for c in &conj_store {
            rev_operands.push(c);
        }
        for dt in &delta_tensors {
            rev_operands.push(dt);
        }
        let grad = einsum_with_subscripts_in_ctx::<B, _, T>(ctx, &rev_subs, &rev_operands)?;
        input_grads.push((*node, grad));
    }

    Ok(input_grads)
}
