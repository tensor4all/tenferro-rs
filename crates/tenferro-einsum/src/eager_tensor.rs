//! EagerTensor einsum extension API.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::mem::size_of;
use std::sync::Arc;

use computegraph::compile::{compile, CompiledProgram, Instruction};
use computegraph::graph::GraphBuilder;
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::{ValueKey, ValueRef};
use tenferro_ad::error::{Error, Result};
use tenferro_ad::extension::apply_eager;
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::ExtensionCacheKey;

use crate::binary_dot::{try_build_exact_output_binary_dot_plan, BinaryDotOperandOrder};
use crate::builder::build_einsum_graph;
use crate::cache::{EINSUM_EAGER_EXPANDED_PROGRAMS_CACHE, EINSUM_EXTENSION_FAMILY_ID};
use crate::extension::{
    ensure_einsum_extension_rule_registered, register_runtime, EinsumExtensionOp,
};
use crate::optimize::{
    default_auto_options, hash_einsum_plan_spec, resolve_plan_spec, EinsumPlanSpec,
};
use crate::{parse_einsum_subscripts, EinsumSubscripts, Subscripts, TensorDotAxes};

/// Execute an einsum eagerly on [`EagerTensor`] values.
///
/// # Examples
///
/// ```
/// use tenferro_ad::{EagerRuntime, EagerTensor};
/// use tenferro_cpu::CpuBackend;
/// use tenferro_einsum::eager_tensor;
/// use tenferro_tensor::Tensor;
///
/// let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new());
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]),
///     runtime.clone(),
/// );
/// let b = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]),
///     runtime,
/// );
/// let out = eager_tensor::einsum(&[&a, &b], "ij,jk->ik")?;
/// assert_eq!(out.data().shape(), &[2, 4]);
/// # Ok::<(), tenferro_ad::error::Error>(())
/// ```
pub fn einsum(inputs: &[&EagerTensor], subscripts: &str) -> Result<EagerTensor> {
    let subscripts = parse_einsum_subscripts(subscripts)
        .map_err(|err| Error::ContractionError(err.to_string()))?;
    einsum_subscripts(inputs, &subscripts)
}

/// Execute an einsum eagerly from integer labels.
///
/// # Examples
///
/// ```
/// use tenferro_ad::{EagerRuntime, EagerTensor};
/// use tenferro_cpu::CpuBackend;
/// use tenferro_einsum::{eager_tensor, parse_einsum_subscripts};
/// use tenferro_tensor::Tensor;
///
/// let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new());
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]),
///     runtime.clone(),
/// );
/// let b = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]),
///     runtime,
/// );
/// let subscripts = parse_einsum_subscripts("ij,jk->ik").unwrap();
/// let out = eager_tensor::einsum_subscripts(&[&a, &b], &subscripts)?;
/// assert_eq!(out.data().shape(), &[2, 4]);
/// # Ok::<(), tenferro_ad::error::Error>(())
/// ```
pub fn einsum_subscripts(
    inputs: &[&EagerTensor],
    subscripts: &EinsumSubscripts,
) -> Result<EagerTensor> {
    if let Some(result) = try_direct_binary_dot_general(inputs, subscripts) {
        return result;
    }

    let output_shape_hint = infer_eager_output_shape(subscripts, inputs)?;
    if let Some(result) = try_expand_eager_einsum(inputs, subscripts)? {
        return Ok(result);
    }

    ensure_einsum_extension_rule_registered().map_err(|err| Error::Internal(err.to_string()))?;
    if let Some(first) = inputs.first() {
        first
            .runtime()
            .register_extension(register_runtime)
            .map_err(|err| Error::Internal(err.to_string()))?;
    }

    let op = Arc::new(EinsumExtensionOp::with_output_shape_hint(
        subscripts.clone(),
        output_shape_hint,
        EinsumPlanSpec::Auto(default_auto_options()),
    ));
    let mut outputs = apply_eager(op, inputs)?;
    outputs
        .pop()
        .ok_or_else(|| Error::Internal("einsum extension produced no eager output".to_string()))
}

fn try_direct_binary_dot_general(
    inputs: &[&EagerTensor],
    subscripts: &EinsumSubscripts,
) -> Option<Result<EagerTensor>> {
    if inputs.len() != 2 || subscripts.inputs.len() != 2 {
        return None;
    }

    let lhs_labels = &subscripts.inputs[0];
    let rhs_labels = &subscripts.inputs[1];
    if lhs_labels.len() != inputs[0].shape().len() || rhs_labels.len() != inputs[1].shape().len() {
        return None;
    }

    if let Some(plan) =
        try_build_exact_output_binary_dot_plan(lhs_labels, rhs_labels, &subscripts.output)
    {
        return Some(match plan.operand_order {
            BinaryDotOperandOrder::Original => inputs[0].dot_general(inputs[1], plan.config),
            BinaryDotOperandOrder::Swapped => inputs[1].dot_general(inputs[0], plan.config),
        });
    }
    None
}

fn try_expand_eager_einsum(
    inputs: &[&EagerTensor],
    subscripts: &EinsumSubscripts,
) -> Result<Option<EagerTensor>> {
    if inputs.len() <= 1 {
        return Ok(None);
    }

    let shapes: Vec<Vec<usize>> = inputs
        .iter()
        .map(|tensor| tensor.shape().to_vec())
        .collect();
    let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
    let subs = Subscripts::from(subscripts);
    let plan_spec = EinsumPlanSpec::Auto(default_auto_options());
    let program = cached_expanded_eager_program(
        inputs[0].runtime(),
        subscripts,
        &subs,
        &plan_spec,
        &shape_refs,
        &shapes,
    )?;
    execute_eager_einsum_program(inputs, &program)
}

struct ExpandedEagerProgram {
    compiled: CompiledProgram<StdTensorOp>,
    input_slots: Vec<(usize, usize)>,
}

fn cached_expanded_eager_program(
    runtime: &Arc<EagerRuntime>,
    subscripts: &EinsumSubscripts,
    subs: &Subscripts,
    plan_spec: &EinsumPlanSpec,
    shape_refs: &[&[usize]],
    shapes: &[Vec<usize>],
) -> Result<Arc<ExpandedEagerProgram>> {
    runtime.with_extension_caches_mut(|caches| {
        let key = expanded_eager_program_cache_key(subscripts, plan_spec, shapes);
        if let Some(cached) = caches.get::<Arc<ExpandedEagerProgram>>(&key) {
            return Ok(Arc::clone(cached));
        }

        let tree = resolve_plan_spec(plan_spec, subs, shape_refs)
            .map_err(|err| Error::ContractionError(err.to_string()))?;
        let program = Arc::new(build_expanded_eager_program(&tree, shapes)?);
        let retained_bytes = expanded_eager_program_retained_bytes(&program);
        caches.put(key, Arc::clone(&program), retained_bytes);
        Ok(program)
    })
}

fn expanded_eager_program_cache_key(
    subscripts: &EinsumSubscripts,
    plan_spec: &EinsumPlanSpec,
    shapes: &[Vec<usize>],
) -> ExtensionCacheKey {
    let mut hasher = DefaultHasher::new();
    subscripts.hash(&mut hasher);
    shapes.hash(&mut hasher);
    hash_einsum_plan_spec(plan_spec, &mut hasher);
    ExtensionCacheKey::new(
        EINSUM_EXTENSION_FAMILY_ID,
        EINSUM_EAGER_EXPANDED_PROGRAMS_CACHE,
        hasher.finish(),
    )
}

fn build_expanded_eager_program(
    tree: &crate::ContractionTree,
    shapes: &[Vec<usize>],
) -> Result<ExpandedEagerProgram> {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut input_vals = Vec::with_capacity(shapes.len());
    for input_idx in 0..shapes.len() {
        let local = builder.add_input(TensorInputKey::User {
            id: input_idx as u64,
        });
        input_vals.push(ValueRef::Local(local));
    }

    let result_ref = build_einsum_graph(&mut builder, tree, &input_vals, shapes)
        .map_err(|err| Error::ContractionError(err.to_string()))?;
    let ValueRef::Local(result_local) = result_ref else {
        return Err(Error::Internal(
            "expanded eager einsum returned an external value".into(),
        ));
    };
    builder.set_outputs(vec![result_local]);
    let graph = Arc::new(builder.build());
    let output_key = graph.values()[result_local].key.clone();
    let view = resolve(vec![graph]);
    let graph = materialize_merge(&view, &[output_key]);
    let compiled = compile(&graph);
    let input_slots = compiled
        .input_slots
        .iter()
        .zip(graph.inputs.iter())
        .map(|(&slot, key)| {
            let ValueKey::Input(TensorInputKey::User { id }) = key else {
                return Err(Error::Internal(format!(
                    "expanded eager einsum saw unexpected input key: {key:?}"
                )));
            };
            Ok((slot, *id as usize))
        })
        .collect::<Result<_>>()?;

    Ok(ExpandedEagerProgram {
        compiled,
        input_slots,
    })
}

fn execute_eager_einsum_program(
    inputs: &[&EagerTensor],
    program: &ExpandedEagerProgram,
) -> Result<Option<EagerTensor>> {
    let mut slots: Vec<Option<EagerTensor>> = vec![None; program.compiled.n_slots];
    for &(slot, input_idx) in &program.input_slots {
        let tensor = inputs.get(input_idx).ok_or_else(|| {
            Error::Internal(format!(
                "expanded eager einsum input {input_idx} is missing"
            ))
        })?;
        slots[slot] = Some((*tensor).clone());
    }

    let mut instruction_idx = 0;
    while instruction_idx < program.compiled.instructions.len() {
        if let Some((output_slot, output)) = try_execute_eager_broadcast_multiply_pattern(
            &program.compiled.instructions,
            instruction_idx,
            &slots,
            &program.compiled.output_slots,
        )? {
            slots[output_slot] = Some(output);
            instruction_idx += 3;
            continue;
        }

        let instr = &program.compiled.instructions[instruction_idx];
        if instr.outputs.len() != 1 {
            return Err(Error::Internal(format!(
                "expanded eager einsum expected single-output op, got {} outputs",
                instr.outputs.len()
            )));
        }
        let input_values: Vec<EagerTensor> = instr
            .inputs
            .iter()
            .map(|&slot| {
                slots
                    .get(slot)
                    .and_then(Option::as_ref)
                    .cloned()
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "expanded eager einsum missing value for slot {slot}"
                        ))
                    })
            })
            .collect::<Result<_>>()?;
        let input_refs: Vec<&EagerTensor> = input_values.iter().collect();
        let output =
            tenferro_ad::eager_tensor::apply_standard_op(instr.operation.clone(), &input_refs)?;
        slots[instr.outputs[0]] = Some(output);
        instruction_idx += 1;
    }

    let [output_slot] = program.compiled.output_slots.as_slice() else {
        return Err(Error::Internal(format!(
            "expanded eager einsum expected one graph output, got {}",
            program.compiled.output_slots.len()
        )));
    };
    slots
        .get_mut(*output_slot)
        .and_then(Option::take)
        .map(Some)
        .ok_or_else(|| Error::Internal("expanded eager einsum output slot is missing".into()))
}

fn expanded_eager_program_retained_bytes(program: &ExpandedEagerProgram) -> usize {
    size_of::<ExpandedEagerProgram>()
        + vec_retained_bytes(&program.input_slots)
        + compiled_program_retained_bytes(&program.compiled)
}

fn compiled_program_retained_bytes(program: &CompiledProgram<StdTensorOp>) -> usize {
    size_of::<CompiledProgram<StdTensorOp>>()
        + vec_retained_bytes(&program.instructions)
        + vec_retained_bytes(&program.input_slots)
        + vec_retained_bytes(&program.output_slots)
        + program
            .instructions
            .iter()
            .map(instruction_retained_bytes)
            .sum::<usize>()
}

fn instruction_retained_bytes(instruction: &Instruction<StdTensorOp>) -> usize {
    size_of::<Instruction<StdTensorOp>>()
        + std_tensor_op_retained_bytes(&instruction.operation)
        + vec_retained_bytes(&instruction.inputs)
        + vec_retained_bytes(&instruction.outputs)
}

fn std_tensor_op_retained_bytes(op: &StdTensorOp) -> usize {
    match op {
        StdTensorOp::DotGeneral { config } => {
            vec_retained_bytes(&config.lhs_contracting_dims)
                + vec_retained_bytes(&config.rhs_contracting_dims)
                + vec_retained_bytes(&config.lhs_batch_dims)
                + vec_retained_bytes(&config.rhs_batch_dims)
        }
        StdTensorOp::Transpose { perm } => vec_retained_bytes(perm),
        StdTensorOp::Reshape { to_shape } => vec_retained_bytes(to_shape),
        StdTensorOp::BroadcastInDim { shape, dims } => {
            vec_retained_bytes(shape) + vec_retained_bytes(dims)
        }
        StdTensorOp::Constant { bytes, .. } => vec_retained_bytes(bytes),
        StdTensorOp::ReduceSum { axes }
        | StdTensorOp::ReduceProd { axes }
        | StdTensorOp::ReduceMax { axes }
        | StdTensorOp::ReduceMin { axes }
        | StdTensorOp::Reverse { axes } => vec_retained_bytes(axes),
        StdTensorOp::DynamicSlice { slice_sizes } => vec_retained_bytes(slice_sizes),
        StdTensorOp::GatherDynamicSliceSizes {
            offset_dims,
            collapsed_slice_dims,
            start_index_map,
            slice_sizes,
            ..
        } => {
            vec_retained_bytes(offset_dims)
                + vec_retained_bytes(collapsed_slice_dims)
                + vec_retained_bytes(start_index_map)
                + vec_retained_bytes(slice_sizes)
        }
        _ => 0,
    }
}

fn vec_retained_bytes<T>(values: &Vec<T>) -> usize {
    values.capacity() * size_of::<T>()
}

fn try_execute_eager_broadcast_multiply_pattern(
    instructions: &[Instruction<StdTensorOp>],
    instruction_idx: usize,
    slots: &[Option<EagerTensor>],
    output_slots: &[usize],
) -> Result<Option<(usize, EagerTensor)>> {
    if instruction_idx + 2 >= instructions.len() {
        return Ok(None);
    }
    let lhs_bc = &instructions[instruction_idx];
    let rhs_bc = &instructions[instruction_idx + 1];
    let multiply = &instructions[instruction_idx + 2];

    let StdTensorOp::BroadcastInDim {
        shape: lhs_shape_exprs,
        dims: lhs_dims,
    } = &lhs_bc.operation
    else {
        return Ok(None);
    };
    let StdTensorOp::BroadcastInDim {
        shape: rhs_shape_exprs,
        dims: rhs_dims,
    } = &rhs_bc.operation
    else {
        return Ok(None);
    };
    if !matches!(multiply.operation, StdTensorOp::Mul)
        || lhs_bc.outputs.len() != 1
        || rhs_bc.outputs.len() != 1
        || multiply.outputs.len() != 1
        || multiply.inputs.len() != 2
        || lhs_bc.inputs.is_empty()
        || rhs_bc.inputs.is_empty()
        || multiply.inputs[0] != lhs_bc.outputs[0]
        || multiply.inputs[1] != rhs_bc.outputs[0]
    {
        return Ok(None);
    }

    let lhs_bc_slot = lhs_bc.outputs[0];
    let rhs_bc_slot = rhs_bc.outputs[0];
    if output_slots.contains(&lhs_bc_slot)
        || output_slots.contains(&rhs_bc_slot)
        || instructions[instruction_idx + 3..]
            .iter()
            .any(|instr| instr.inputs.contains(&lhs_bc_slot) || instr.inputs.contains(&rhs_bc_slot))
    {
        return Ok(None);
    }

    let lhs = slot_tensor(slots, lhs_bc.inputs[0])?;
    let rhs = slot_tensor(slots, rhs_bc.inputs[0])?;
    let lhs_shape = eval_shape_exprs(slots, &lhs_bc.inputs, lhs_shape_exprs)?;
    let rhs_shape = eval_shape_exprs(slots, &rhs_bc.inputs, rhs_shape_exprs)?;
    let Some(output) = tenferro_ad::eager_tensor::try_backend_broadcast_multiply_untracked(
        lhs, &lhs_shape, lhs_dims, rhs, &rhs_shape, rhs_dims,
    )?
    else {
        return Ok(None);
    };

    Ok(Some((multiply.outputs[0], output)))
}

fn eval_shape_exprs(
    slots: &[Option<EagerTensor>],
    input_slots: &[usize],
    shape: &[DimExpr],
) -> Result<Vec<usize>> {
    let inputs = input_slots
        .iter()
        .map(|&slot| slot_tensor(slots, slot))
        .collect::<Result<Vec<_>>>()?;
    let input_shapes = inputs
        .iter()
        .map(|tensor| tensor.shape())
        .collect::<Vec<_>>();
    Ok(DimExpr::eval_all(shape, &input_shapes))
}

fn slot_tensor(slots: &[Option<EagerTensor>], slot: usize) -> Result<&EagerTensor> {
    slots.get(slot).and_then(Option::as_ref).ok_or_else(|| {
        Error::Internal(format!(
            "expanded eager einsum missing value for slot {slot}"
        ))
    })
}

fn infer_eager_output_shape(
    subscripts: &EinsumSubscripts,
    inputs: &[&EagerTensor],
) -> Result<Vec<tenferro_runtime::SymDim>> {
    if inputs.is_empty() {
        return Err(Error::ContractionError(
            "einsum requires at least one input tensor".into(),
        ));
    }
    if subscripts.inputs.len() != inputs.len() {
        return Err(Error::ContractionError(format!(
            "einsum subscripts expect {} inputs, got {}",
            subscripts.inputs.len(),
            inputs.len()
        )));
    }

    let mut label_dims = std::collections::HashMap::new();
    for (labels, tensor) in subscripts.inputs.iter().zip(inputs.iter()) {
        let shape = tensor.shape();
        if labels.len() != shape.len() {
            return Err(Error::ContractionError(format!(
                "einsum input rank mismatch: labels={}, shape={}",
                labels.len(),
                shape.len()
            )));
        }
        for (&label, &dim) in labels.iter().zip(shape.iter()) {
            if let Some(existing) = label_dims.insert(label, dim) {
                if existing != dim {
                    return Err(Error::ContractionError(format!(
                        "einsum label {label} has inconsistent dimensions {existing} and {dim}"
                    )));
                }
            }
        }
    }

    subscripts
        .output
        .iter()
        .map(|label| {
            label_dims
                .get(label)
                .copied()
                .map(tenferro_runtime::SymDim::from)
                .ok_or_else(|| {
                    Error::ContractionError(format!(
                        "einsum output label {label} is missing from input labels"
                    ))
                })
        })
        .collect()
}

/// Execute a NumPy-style tensor contraction on [`EagerTensor`] values.
///
/// This helper lives in the einsum extension namespace because it is
/// contraction sugar over `dot_general`, not a linear algebra facade.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::Tensor;
/// use tenferro_cpu::CpuBackend;
/// use tenferro_ad::{EagerRuntime, EagerTensor};
/// use tenferro_einsum::{eager_tensor, TensorDotAxes};
///
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
/// let lhs = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]),
///     ctx.clone(),
/// );
/// let rhs = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]),
///     ctx,
/// );
/// let out = eager_tensor::tensordot(&lhs, &rhs, TensorDotAxes::Count(1)).unwrap();
///
/// assert_eq!(out.data().shape(), &[2, 4]);
/// ```
pub fn tensordot(
    lhs: &EagerTensor,
    rhs: &EagerTensor,
    axes: TensorDotAxes<'_>,
) -> Result<EagerTensor> {
    let config = crate::tensordot::dot_general_config(axes, lhs.shape().len(), rhs.shape().len())?;
    crate::tensordot::validate_concrete_contract_dims(lhs.shape(), rhs.shape(), &config)?;
    lhs.dot_general(rhs, config)
}

#[cfg(test)]
mod tests;
