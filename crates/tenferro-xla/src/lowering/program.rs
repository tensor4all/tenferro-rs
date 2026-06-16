use std::sync::Arc;

use computegraph::compile::compile;
use computegraph::graph::GraphBuilder;
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::{ValueKey, ValueRef};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::sym_dim::SymDim;
use tenferro_runtime::{GraphInstructionView, GraphOpView, GraphProgram};
use tenferro_tensor::{DType, DotGeneralConfig};

use crate::{Error, Result, StableHloModule};

use super::emit::{format_usize_list, Emitter};
use super::shape::static_output_shape;
use super::types::{format_tensor_type, validate_dtype, TensorType};

#[derive(Clone, Debug)]
struct Value {
    name: String,
    ty: TensorType,
}

pub(crate) fn lower_graph_program(program: &GraphProgram) -> Result<StableHloModule> {
    let view = program.lowering_view();
    if view.input_slots().len() != program.input_specs().len() {
        return Err(Error::InvalidProgram {
            message: format!(
                "input slot count {} does not match input spec count {}",
                view.input_slots().len(),
                program.input_specs().len()
            ),
        });
    }

    let mut slots: Vec<Option<Value>> = vec![None; view.slot_count()];
    let mut args = Vec::with_capacity(program.input_specs().len());
    for (index, input) in program.input_specs().iter().enumerate() {
        validate_dtype(input.dtype(), "program input")?;
        let ty = TensorType::new(input.shape().to_vec(), input.dtype(), "program input")?;
        let value = Value {
            name: format!("%arg{index}"),
            ty: ty.clone(),
        };
        let slot = view.input_slots()[index];
        let Some(slot_ref) = slots.get_mut(slot) else {
            return Err(Error::InvalidProgram {
                message: format!(
                    "input slot {slot} is outside slot table length {}",
                    view.slot_count()
                ),
            });
        };
        *slot_ref = Some(value);
        args.push(format!("%arg{index}: {}", format_tensor_type(&ty)));
    }

    let mut emitter = Emitter::default();
    for inst in view.instructions() {
        lower_instruction(inst, &mut slots, &mut emitter)?;
    }

    let outputs = view
        .output_slots()
        .iter()
        .map(|&slot| slot_value(&slots, slot).cloned())
        .collect::<Result<Vec<_>>>()?;

    let return_types = outputs
        .iter()
        .map(|value| format_tensor_type(&value.ty))
        .collect::<Vec<_>>();
    let signature_return = match return_types.as_slice() {
        [] => "()".to_string(),
        [single] => single.clone(),
        many => format!("({})", many.join(", ")),
    };

    let mut text = String::new();
    text.push_str("module {\n");
    text.push_str(&format!(
        "  func.func @main({}) -> {} {{\n",
        args.join(", "),
        signature_return
    ));
    for line in emitter.finish() {
        text.push_str("    ");
        text.push_str(&line);
        text.push('\n');
    }
    match outputs.as_slice() {
        [] => text.push_str("    return\n"),
        [single] => text.push_str(&format!(
            "    return {} : {}\n",
            single.name,
            format_tensor_type(&single.ty)
        )),
        many => {
            let names = many
                .iter()
                .map(|value| value.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            text.push_str(&format!(
                "    return {names} : {}\n",
                return_types.join(", ")
            ));
        }
    }
    text.push_str("  }\n");
    text.push_str("}\n");
    Ok(StableHloModule::new(text))
}

fn lower_instruction(
    inst: GraphInstructionView<'_>,
    slots: &mut [Option<Value>],
    emitter: &mut Emitter,
) -> Result<()> {
    let input_values = inst
        .input_slots()
        .iter()
        .map(|&slot| slot_value(slots, slot).cloned())
        .collect::<Result<Vec<_>>>()?;
    let input_shapes = input_values
        .iter()
        .map(|value| value.ty.shape.as_slice())
        .collect::<Vec<_>>();

    if let GraphOpView::Extension { op } = inst.op() {
        return lower_extension_instruction(op, inst, &input_values, slots, emitter);
    }

    if inst.output_slots().len() != 1 {
        return Err(Error::UnsupportedOp {
            op: inst.op_name(),
            reason: "multiple outputs are not part of the initial XLA subset",
        });
    }

    validate_dtype(inst.dtype(), "instruction output")?;
    let output_shape = static_output_shape(inst, 0, &input_shapes)?;
    let output_ty = TensorType::new(output_shape, inst.dtype(), "instruction output")?;
    let value = match inst.op() {
        GraphOpView::Constant { dtype, bytes } => {
            lower_constant(dtype, bytes, &output_ty, emitter)?
        }
        GraphOpView::Add => {
            lower_same_type_binary("stablehlo.add", &input_values, &output_ty, emitter)?
        }
        GraphOpView::Multiply => {
            lower_same_type_binary("stablehlo.multiply", &input_values, &output_ty, emitter)?
        }
        GraphOpView::Negate => lower_unary("stablehlo.negate", &input_values, &output_ty, emitter)?,
        GraphOpView::Convert { to } => lower_convert(to, &input_values, &output_ty, emitter)?,
        GraphOpView::Reshape => lower_reshape(&input_values, &output_ty, emitter)?,
        GraphOpView::BroadcastInDim { dims } => {
            lower_broadcast_in_dim(dims, &input_values, &output_ty, emitter)?
        }
        GraphOpView::Transpose { perm } => {
            lower_transpose(perm, &input_values, &output_ty, emitter)?
        }
        GraphOpView::ReduceSum { axes } => {
            lower_reduce_sum(axes, &input_values, &output_ty, emitter)?
        }
        GraphOpView::DotGeneral { config } => {
            lower_dot_general(config, &input_values, &output_ty, emitter)?
        }
        GraphOpView::Extension { .. } => unreachable!("extension instructions are handled first"),
        GraphOpView::Unsupported { name } => {
            return Err(Error::UnsupportedOp {
                op: name,
                reason: "operation is outside the initial StableHLO lowering subset",
            });
        }
    };

    let output_slot = inst.output_slots()[0];
    let Some(slot_ref) = slots.get_mut(output_slot) else {
        return Err(Error::InvalidProgram {
            message: format!(
                "output slot {output_slot} is outside slot table length {}",
                slots.len()
            ),
        });
    };
    *slot_ref = Some(value);
    Ok(())
}

fn lower_extension_instruction(
    op: &dyn tenferro_ops::ext_op::ExtensionOp,
    inst: GraphInstructionView<'_>,
    input_values: &[Value],
    slots: &mut [Option<Value>],
    emitter: &mut Emitter,
) -> Result<()> {
    let input_dtypes = input_values
        .iter()
        .map(|value| value.ty.dtype)
        .collect::<Vec<_>>();
    let input_sym_shapes = input_values
        .iter()
        .map(|value| {
            value
                .ty
                .shape
                .iter()
                .copied()
                .map(SymDim::from)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let input_sym_shape_refs = input_sym_shapes
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();

    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let input_refs = (0..input_values.len())
        .map(|input_idx| {
            let local = builder.add_input(TensorInputKey::User {
                id: input_idx as u64,
            });
            ValueRef::Local(local)
        })
        .collect::<Vec<_>>();

    let Some(output_refs) = op
        .lower_to_standard_ops(
            &mut builder,
            &input_refs,
            &input_dtypes,
            &input_sym_shape_refs,
        )
        .map_err(|err| Error::InvalidProgram {
            message: format!(
                "extension family {:?} standard-op lowering failed: {err}",
                op.family_id()
            ),
        })?
    else {
        return Err(Error::UnsupportedOp {
            op: op.family_id(),
            reason: "extension does not provide a standard-op lowering for exact static shapes",
        });
    };

    if output_refs.len() != inst.output_slots().len() {
        return Err(Error::InvalidProgram {
            message: format!(
                "extension family {:?} standard-op lowering returned {} outputs for {} output slots",
                op.family_id(),
                output_refs.len(),
                inst.output_slots().len()
            ),
        });
    }

    let mut output_locals = Vec::with_capacity(output_refs.len());
    for output in output_refs {
        let ValueRef::Local(local) = output else {
            return Err(Error::InvalidProgram {
                message: format!(
                    "extension family {:?} standard-op lowering returned an external output",
                    op.family_id()
                ),
            });
        };
        output_locals.push(local);
    }
    builder.set_outputs(output_locals.clone());
    let graph = Arc::new(builder.build());
    let output_keys = output_locals
        .iter()
        .map(|&local| graph.values()[local].key.clone())
        .collect::<Vec<_>>();
    let view = resolve(vec![graph]);
    let graph = materialize_merge(&view, &output_keys);
    let compiled = compile(&graph);

    let mut sub_input_indices = Vec::with_capacity(graph.inputs.len());
    let mut sub_input_dtypes = Vec::with_capacity(graph.inputs.len());
    let mut sub_input_shapes = Vec::with_capacity(graph.inputs.len());
    for key in &graph.inputs {
        let ValueKey::Input(TensorInputKey::User { id }) = key else {
            return Err(Error::InvalidProgram {
                message: format!(
                    "extension family {:?} standard-op lowering produced unexpected input key: {key:?}",
                    op.family_id()
                ),
            });
        };
        let input_idx = *id as usize;
        let Some(input_value) = input_values.get(input_idx) else {
            return Err(Error::InvalidProgram {
                message: format!(
                    "extension family {:?} standard-op lowering referenced missing input {input_idx}",
                    op.family_id()
                ),
            });
        };
        sub_input_indices.push(input_idx);
        sub_input_dtypes.push(input_value.ty.dtype);
        sub_input_shapes.push(DimExpr::from_concrete(&input_value.ty.shape));
    }

    let sub_program = tenferro_runtime::extension::compile_std_to_exec(
        &compiled,
        &sub_input_dtypes,
        &sub_input_shapes,
    )
    .map_err(|err| Error::InvalidProgram {
        message: format!(
            "extension family {:?} standard-op lowering produced invalid graph: {err}",
            op.family_id()
        ),
    })?;
    let sub_view = sub_program.lowering_view();
    let mut sub_slots: Vec<Option<Value>> = vec![None; sub_view.slot_count()];
    for (sub_arg_idx, &sub_slot) in sub_view.input_slots().iter().enumerate() {
        let input_idx = sub_input_indices[sub_arg_idx];
        sub_slots[sub_slot] = Some(input_values[input_idx].clone());
    }
    for sub_inst in sub_view.instructions() {
        lower_instruction(sub_inst, &mut sub_slots, emitter)?;
    }

    let sub_outputs = sub_view
        .output_slots()
        .iter()
        .map(|&slot| slot_value(&sub_slots, slot).cloned())
        .collect::<Result<Vec<_>>>()?;
    if sub_outputs.len() != inst.output_slots().len() {
        return Err(Error::InvalidProgram {
            message: format!(
                "extension family {:?} standard-op lowering produced {} executable outputs for {} output slots",
                op.family_id(),
                sub_outputs.len(),
                inst.output_slots().len()
            ),
        });
    }

    let parent_input_shapes = input_values
        .iter()
        .map(|value| value.ty.shape.as_slice())
        .collect::<Vec<_>>();
    for (output_idx, (&parent_slot, value)) in
        inst.output_slots().iter().zip(sub_outputs).enumerate()
    {
        let expected_shape = static_output_shape(inst, output_idx, &parent_input_shapes)?;
        if value.ty.shape != expected_shape {
            return Err(Error::InvalidProgram {
                message: format!(
                    "extension family {:?} standard-op lowering output {output_idx} shape {:?} does not match expected {:?}",
                    op.family_id(),
                    value.ty.shape,
                    expected_shape
                ),
            });
        }
        if output_idx == 0 {
            validate_dtype(inst.dtype(), "extension output")?;
            if value.ty.dtype != inst.dtype() {
                return Err(Error::InvalidProgram {
                    message: format!(
                        "extension family {:?} standard-op lowering output 0 dtype {:?} does not match expected {:?}",
                        op.family_id(),
                        value.ty.dtype,
                        inst.dtype()
                    ),
                });
            }
        }
        let Some(slot_ref) = slots.get_mut(parent_slot) else {
            return Err(Error::InvalidProgram {
                message: format!(
                    "extension output slot {parent_slot} is outside slot table length {}",
                    slots.len()
                ),
            });
        };
        *slot_ref = Some(value);
    }
    Ok(())
}

fn lower_constant(
    dtype: DType,
    bytes: &[u8],
    output_ty: &TensorType,
    emitter: &mut Emitter,
) -> Result<Value> {
    validate_dtype(dtype, "constant")?;
    if !output_ty.shape.is_empty() {
        return Err(Error::InvalidProgram {
            message: "ExecOp::Constant must lower as a scalar tensor".to_string(),
        });
    }
    let literal = match dtype {
        DType::F32 => {
            let bytes: [u8; 4] = bytes.try_into().map_err(|_| Error::InvalidProgram {
                message: format!("F32 constant expected 4 bytes, got {}", bytes.len()),
            })?;
            format_float(f32::from_le_bytes(bytes) as f64)
        }
        DType::F64 => {
            let bytes: [u8; 8] = bytes.try_into().map_err(|_| Error::InvalidProgram {
                message: format!("F64 constant expected 8 bytes, got {}", bytes.len()),
            })?;
            format_float(f64::from_le_bytes(bytes))
        }
        other => {
            return Err(Error::UnsupportedDType {
                dtype: other,
                context: "constant",
            });
        }
    };
    let name = emitter.value();
    emitter.line(format!(
        "{name} = stablehlo.constant dense<{literal}> : {}",
        format_tensor_type(output_ty)
    ));
    Ok(Value {
        name,
        ty: output_ty.clone(),
    })
}

fn lower_same_type_binary(
    op: &'static str,
    inputs: &[Value],
    output_ty: &TensorType,
    emitter: &mut Emitter,
) -> Result<Value> {
    require_input_count(op, inputs, 2)?;
    let name = emitter.value();
    emitter.line(format!(
        "{name} = {op} {}, {} : {}",
        inputs[0].name,
        inputs[1].name,
        format_tensor_type(output_ty)
    ));
    Ok(Value {
        name,
        ty: output_ty.clone(),
    })
}

fn lower_unary(
    op: &'static str,
    inputs: &[Value],
    output_ty: &TensorType,
    emitter: &mut Emitter,
) -> Result<Value> {
    require_input_count(op, inputs, 1)?;
    let name = emitter.value();
    emitter.line(format!(
        "{name} = {op} {} : {}",
        inputs[0].name,
        format_tensor_type(output_ty)
    ));
    Ok(Value {
        name,
        ty: output_ty.clone(),
    })
}

fn lower_convert(
    to: DType,
    inputs: &[Value],
    output_ty: &TensorType,
    emitter: &mut Emitter,
) -> Result<Value> {
    require_input_count("stablehlo.convert", inputs, 1)?;
    validate_dtype(to, "convert target")?;
    let name = emitter.value();
    emitter.line(format!(
        "{name} = stablehlo.convert {} : ({}) -> {}",
        inputs[0].name,
        format_tensor_type(&inputs[0].ty),
        format_tensor_type(output_ty)
    ));
    Ok(Value {
        name,
        ty: output_ty.clone(),
    })
}

fn lower_reshape(inputs: &[Value], output_ty: &TensorType, emitter: &mut Emitter) -> Result<Value> {
    require_input_count("stablehlo.reshape", inputs, 1)?;
    let name = emitter.value();
    emitter.line(format!(
        "{name} = stablehlo.reshape {} : ({}) -> {}",
        inputs[0].name,
        format_tensor_type(&inputs[0].ty),
        format_tensor_type(output_ty)
    ));
    Ok(Value {
        name,
        ty: output_ty.clone(),
    })
}

fn lower_broadcast_in_dim(
    dims: &[usize],
    inputs: &[Value],
    output_ty: &TensorType,
    emitter: &mut Emitter,
) -> Result<Value> {
    require_input_count("stablehlo.broadcast_in_dim", inputs, 1)?;
    let name = emitter.value();
    emitter.line(format!(
        "{name} = stablehlo.broadcast_in_dim {}, dims = {} : ({}) -> {}",
        inputs[0].name,
        format_usize_list(dims),
        format_tensor_type(&inputs[0].ty),
        format_tensor_type(output_ty)
    ));
    Ok(Value {
        name,
        ty: output_ty.clone(),
    })
}

fn lower_transpose(
    perm: &[usize],
    inputs: &[Value],
    output_ty: &TensorType,
    emitter: &mut Emitter,
) -> Result<Value> {
    require_input_count("stablehlo.transpose", inputs, 1)?;
    emit_transpose(&inputs[0], perm, output_ty, emitter)
}

fn lower_reduce_sum(
    axes: &[usize],
    inputs: &[Value],
    output_ty: &TensorType,
    emitter: &mut Emitter,
) -> Result<Value> {
    require_input_count("stablehlo.reduce", inputs, 1)?;
    let init_ty = TensorType::scalar(output_ty.dtype, "reduce init")?;
    let init = emitter.value();
    emitter.line(format!(
        "{init} = stablehlo.constant dense<{}> : {}",
        format_float(0.0),
        format_tensor_type(&init_ty)
    ));
    let name = emitter.value();
    emitter.line(format!(
        "{name} = stablehlo.reduce({} init: {init}) applies stablehlo.add across dimensions = {} : ({}, {}) -> {}",
        inputs[0].name,
        format_usize_list(axes),
        format_tensor_type(&inputs[0].ty),
        format_tensor_type(&init_ty),
        format_tensor_type(output_ty)
    ));
    Ok(Value {
        name,
        ty: output_ty.clone(),
    })
}

fn lower_dot_general(
    config: &DotGeneralConfig,
    inputs: &[Value],
    output_ty: &TensorType,
    emitter: &mut Emitter,
) -> Result<Value> {
    require_input_count("stablehlo.dot_general", inputs, 2)?;
    let lhs = &inputs[0];
    let rhs = &inputs[1];
    let stable_shape = stablehlo_dot_shape(&lhs.ty.shape, &rhs.ty.shape, config)?;
    let stable_ty = TensorType::new(stable_shape, output_ty.dtype, "dot_general output")?;
    let dot = emitter.value();
    let batching = if config.lhs_batch_dims.is_empty() {
        String::new()
    } else {
        format!(
            "batching_dims = {} x {}, ",
            format_usize_list(&config.lhs_batch_dims),
            format_usize_list(&config.rhs_batch_dims)
        )
    };
    emitter.line(format!(
        "{dot} = stablehlo.dot_general {}, {}, {batching}contracting_dims = {} x {}, precision = [DEFAULT, DEFAULT] : ({}, {}) -> {}",
        lhs.name,
        rhs.name,
        format_usize_list(&config.lhs_contracting_dims),
        format_usize_list(&config.rhs_contracting_dims),
        format_tensor_type(&lhs.ty),
        format_tensor_type(&rhs.ty),
        format_tensor_type(&stable_ty)
    ));
    let dot_value = Value {
        name: dot,
        ty: stable_ty,
    };
    if dot_value.ty.shape == output_ty.shape {
        return Ok(dot_value);
    }

    let lhs_free = free_dims(
        lhs.ty.shape.len(),
        &config.lhs_contracting_dims,
        &config.lhs_batch_dims,
    );
    let rhs_free = free_dims(
        rhs.ty.shape.len(),
        &config.rhs_contracting_dims,
        &config.rhs_batch_dims,
    );
    let batch_count = config.lhs_batch_dims.len();
    let mut perm = Vec::with_capacity(dot_value.ty.shape.len());
    perm.extend(batch_count..batch_count + lhs_free.len());
    perm.extend(batch_count + lhs_free.len()..batch_count + lhs_free.len() + rhs_free.len());
    perm.extend(0..batch_count);
    emit_transpose(&dot_value, &perm, output_ty, emitter)
}

fn emit_transpose(
    input: &Value,
    perm: &[usize],
    output_ty: &TensorType,
    emitter: &mut Emitter,
) -> Result<Value> {
    if perm.len() != input.ty.shape.len() || perm.len() != output_ty.shape.len() {
        return Err(Error::InvalidProgram {
            message: format!(
                "transpose permutation length {} does not match input rank {} and output rank {}",
                perm.len(),
                input.ty.shape.len(),
                output_ty.shape.len()
            ),
        });
    }
    let name = emitter.value();
    emitter.line(format!(
        "{name} = stablehlo.transpose {}, dims = {} : ({}) -> {}",
        input.name,
        format_usize_list(perm),
        format_tensor_type(&input.ty),
        format_tensor_type(output_ty)
    ));
    Ok(Value {
        name,
        ty: output_ty.clone(),
    })
}

fn stablehlo_dot_shape(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    config: &DotGeneralConfig,
) -> Result<Vec<usize>> {
    config
        .validate_dims_with_ranks(lhs_shape.len(), rhs_shape.len())
        .map_err(|message| Error::InvalidProgram { message })?;
    let lhs_free = free_dims(
        lhs_shape.len(),
        &config.lhs_contracting_dims,
        &config.lhs_batch_dims,
    );
    let rhs_free = free_dims(
        rhs_shape.len(),
        &config.rhs_contracting_dims,
        &config.rhs_batch_dims,
    );
    let mut shape = Vec::new();
    shape.extend(config.lhs_batch_dims.iter().map(|&axis| lhs_shape[axis]));
    shape.extend(lhs_free.into_iter().map(|axis| lhs_shape[axis]));
    shape.extend(rhs_free.into_iter().map(|axis| rhs_shape[axis]));
    Ok(shape)
}

fn free_dims(rank: usize, contracting: &[usize], batch: &[usize]) -> Vec<usize> {
    (0..rank)
        .filter(|axis| !contracting.contains(axis) && !batch.contains(axis))
        .collect()
}

fn require_input_count(op: &'static str, inputs: &[Value], expected: usize) -> Result<()> {
    if inputs.len() == expected {
        return Ok(());
    }
    Err(Error::InvalidProgram {
        message: format!("{op} expected {expected} inputs, got {}", inputs.len()),
    })
}

fn slot_value(slots: &[Option<Value>], slot: usize) -> Result<&Value> {
    slots
        .get(slot)
        .ok_or_else(|| Error::InvalidProgram {
            message: format!("slot {slot} is outside slot table length {}", slots.len()),
        })?
        .as_ref()
        .ok_or_else(|| Error::InvalidProgram {
            message: format!("slot {slot} has no value"),
        })
}

fn format_float(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.8e}")
    } else if value.is_nan() {
        "0x7ff8000000000000".to_string()
    } else if value.is_sign_negative() {
        "-0x7ff0000000000000".to_string()
    } else {
        "0x7ff0000000000000".to_string()
    }
}

#[cfg(test)]
mod tests;
