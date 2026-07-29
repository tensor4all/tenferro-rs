use std::collections::HashMap;
use std::sync::Arc;

use computegraph::compile::compile;
use computegraph::graph::GraphBuilder;
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::{ValueKey, ValueRef};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::ExtensionStandardLowering;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::sym_dim::SymDim;
use tenferro_runtime::program::{
    CoreSemanticOp, ProgramInputSpec, ProgramValue, SemanticOpRef, SemanticOperationView,
    SemanticProgram, SemanticProgramBuilder,
};
use tenferro_tensor::{DType, DotGeneralConfig};

use crate::{Error, Result, StableHloModule};

use super::emit::{format_usize_list, Emitter};
use super::types::{format_tensor_type, validate_dtype, TensorType};

#[derive(Clone, Debug)]
struct Value {
    name: String,
    ty: TensorType,
}

pub(crate) fn lower_semantic_program(program: &SemanticProgram) -> Result<StableHloModule> {
    let mut inputs = Vec::with_capacity(program.inputs().len());
    let mut args = Vec::with_capacity(program.inputs().len());
    for (index, &input) in program.inputs().iter().enumerate() {
        let ty = semantic_value_type(program, input, "Input", 0, "program input")?;
        let value = Value {
            name: format!("%arg{index}"),
            ty: ty.clone(),
        };
        args.push(format!("%arg{index}: {}", format_tensor_type(&ty)));
        inputs.push(value);
    }

    let mut emitter = Emitter::default();
    let outputs = lower_semantic_operations(program, &inputs, &mut emitter)?;

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

fn lower_semantic_operations(
    program: &SemanticProgram,
    inputs: &[Value],
    emitter: &mut Emitter,
) -> Result<Vec<Value>> {
    if program.inputs().len() != inputs.len() {
        return Err(Error::InvalidProgram {
            message: format!(
                "semantic input count {} does not match supplied value count {}",
                program.inputs().len(),
                inputs.len()
            ),
        });
    }
    let mut values = HashMap::with_capacity(program.inputs().len());
    for (&input, value) in program.inputs().iter().zip(inputs) {
        if values.insert(input, value.clone()).is_some() {
            return Err(Error::InvalidProgram {
                message: "semantic input appears more than once".to_string(),
            });
        }
    }
    for operation in program.operations() {
        lower_semantic_operation(program, operation, &mut values, emitter)?;
    }
    program
        .outputs()
        .iter()
        .map(|output| {
            values
                .get(output)
                .cloned()
                .ok_or_else(|| Error::InvalidProgram {
                    message: "semantic program output is unavailable".to_string(),
                })
        })
        .collect()
}

fn lower_semantic_operation(
    program: &SemanticProgram,
    operation: SemanticOperationView<'_>,
    values: &mut HashMap<ProgramValue, Value>,
    emitter: &mut Emitter,
) -> Result<()> {
    let input_values = operation
        .inputs()
        .iter()
        .map(|input| {
            values
                .get(input)
                .cloned()
                .ok_or_else(|| Error::InvalidProgram {
                    message: "semantic operation input is unavailable".to_string(),
                })
        })
        .collect::<Result<Vec<_>>>()?;
    if let SemanticOpRef::Extension(extension) = operation.op() {
        let outputs =
            lower_extension_operation(program, extension, operation, &input_values, emitter)?;
        for (&output, value) in operation.outputs().iter().zip(outputs) {
            if values.insert(output, value).is_some() {
                return Err(Error::InvalidProgram {
                    message: "semantic value has more than one producer".to_string(),
                });
            }
        }
        return Ok(());
    }

    if operation.outputs().len() != 1 {
        return Err(Error::UnsupportedOp {
            op: semantic_op_name(operation.op()),
            reason: "multiple outputs are not part of the initial XLA subset",
        });
    }

    let output = operation.outputs()[0];
    let op_name = semantic_op_name(operation.op());
    let output_ty = semantic_value_type(program, output, op_name, 0, "instruction output")?;
    let value = match operation.op() {
        SemanticOpRef::Core(CoreSemanticOp::Constant { dtype, bytes }) => {
            lower_constant(*dtype, bytes, &output_ty, emitter)?
        }
        SemanticOpRef::Core(CoreSemanticOp::Add) => {
            lower_same_type_binary("stablehlo.add", &input_values, &output_ty, emitter)?
        }
        SemanticOpRef::Core(CoreSemanticOp::Mul) => {
            lower_same_type_binary("stablehlo.multiply", &input_values, &output_ty, emitter)?
        }
        SemanticOpRef::Core(CoreSemanticOp::Neg) => {
            lower_unary("stablehlo.negate", &input_values, &output_ty, emitter)?
        }
        SemanticOpRef::Core(CoreSemanticOp::Div) => {
            lower_same_type_binary("stablehlo.divide", &input_values, &output_ty, emitter)?
        }
        SemanticOpRef::Core(CoreSemanticOp::Abs) => {
            lower_unary("stablehlo.abs", &input_values, &output_ty, emitter)?
        }
        SemanticOpRef::Core(CoreSemanticOp::Exp) => {
            lower_unary("stablehlo.exponential", &input_values, &output_ty, emitter)?
        }
        SemanticOpRef::Core(CoreSemanticOp::Log) => {
            lower_unary("stablehlo.log", &input_values, &output_ty, emitter)?
        }
        SemanticOpRef::Core(CoreSemanticOp::Sin) => {
            lower_unary("stablehlo.sine", &input_values, &output_ty, emitter)?
        }
        SemanticOpRef::Core(CoreSemanticOp::Cos) => {
            lower_unary("stablehlo.cosine", &input_values, &output_ty, emitter)?
        }
        SemanticOpRef::Core(CoreSemanticOp::Tanh) => {
            lower_unary("stablehlo.tanh", &input_values, &output_ty, emitter)?
        }
        SemanticOpRef::Core(CoreSemanticOp::Sqrt) => {
            lower_unary("stablehlo.sqrt", &input_values, &output_ty, emitter)?
        }
        SemanticOpRef::Core(CoreSemanticOp::Rsqrt) => {
            lower_unary("stablehlo.rsqrt", &input_values, &output_ty, emitter)?
        }
        SemanticOpRef::Core(CoreSemanticOp::Pow) => {
            lower_same_type_binary("stablehlo.power", &input_values, &output_ty, emitter)?
        }
        SemanticOpRef::Core(CoreSemanticOp::Expm1) => lower_unary(
            "stablehlo.exponential_minus_one",
            &input_values,
            &output_ty,
            emitter,
        )?,
        SemanticOpRef::Core(CoreSemanticOp::Log1p) => {
            lower_unary("stablehlo.log_plus_one", &input_values, &output_ty, emitter)?
        }
        SemanticOpRef::Core(CoreSemanticOp::Convert { to, .. }) => {
            lower_convert(*to, &input_values, &output_ty, emitter)?
        }
        SemanticOpRef::Core(CoreSemanticOp::Reshape { .. }) => {
            lower_reshape(&input_values, &output_ty, emitter)?
        }
        SemanticOpRef::Core(CoreSemanticOp::BroadcastInDim { dims, .. }) => {
            lower_broadcast_in_dim(dims, &input_values, &output_ty, emitter)?
        }
        SemanticOpRef::Core(CoreSemanticOp::Transpose { perm }) => {
            lower_transpose(perm, &input_values, &output_ty, emitter)?
        }
        SemanticOpRef::Core(CoreSemanticOp::ReduceSum { axes }) => {
            lower_reduce_sum(axes, &input_values, &output_ty, emitter)?
        }
        SemanticOpRef::Core(CoreSemanticOp::ReduceSumSquares { axes }) => {
            lower_reduce_sum_squares(axes, &input_values, &output_ty, emitter)?
        }
        SemanticOpRef::Core(CoreSemanticOp::DotGeneral { config }) => {
            lower_dot_general(config, &input_values, &output_ty, emitter)?
        }
        SemanticOpRef::Extension(_) => {
            return Err(Error::InvalidProgram {
                message: "extension semantic operation reached builtin lowering arm".to_string(),
            });
        }
        SemanticOpRef::Core(_) => {
            return Err(Error::UnsupportedOp {
                op: op_name,
                reason: "operation is outside the initial StableHLO lowering subset",
            });
        }
        _ => {
            return Err(Error::UnsupportedOp {
                op: op_name,
                reason: "operation is outside the initial StableHLO lowering subset",
            });
        }
    };

    if values.insert(output, value).is_some() {
        return Err(Error::InvalidProgram {
            message: "semantic value has more than one producer".to_string(),
        });
    }
    Ok(())
}

fn lower_extension_operation(
    program: &SemanticProgram,
    op: &dyn tenferro_ops::ext_op::ExtensionOp,
    operation: SemanticOperationView<'_>,
    input_values: &[Value],
    emitter: &mut Emitter,
) -> Result<Vec<Value>> {
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

    let output_refs = match op
        .lower_to_standard_ops(
            &mut builder,
            &input_refs,
            &input_dtypes,
            &input_sym_shape_refs,
        )
        .map_err(|source| Error::ExtensionLowering { source })?
    {
        ExtensionStandardLowering::Lowered(outputs) => outputs,
        ExtensionStandardLowering::Unsupported => {
            return Err(Error::UnsupportedOp {
                op: op.family_id(),
                reason: "extension does not provide a standard-op lowering for exact static shapes",
            });
        }
    };

    if output_refs.len() != operation.outputs().len() {
        return Err(Error::InvalidProgram {
            message: format!(
                "extension family {:?} standard-op lowering returned {} outputs for {} semantic values",
                op.family_id(),
                output_refs.len(),
                operation.outputs().len()
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
        let input_idx = usize::try_from(*id).map_err(|_| Error::InvalidProgram {
            message: format!(
                "extension family {:?} standard-op lowering referenced oversized input id {id}",
                op.family_id()
            ),
        })?;
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

    let sub_semantic =
        build_standard_semantic_subprogram(&compiled, &sub_input_dtypes, &sub_input_shapes)?;
    let sub_inputs = sub_input_indices
        .iter()
        .map(|&input_idx| input_values[input_idx].clone())
        .collect::<Vec<_>>();
    let sub_outputs = lower_semantic_operations(&sub_semantic, &sub_inputs, emitter)?;
    if sub_outputs.len() != operation.outputs().len() {
        return Err(Error::InvalidProgram {
            message: format!(
                "extension family {:?} standard-op lowering produced {} semantic outputs for {} outputs",
                op.family_id(),
                sub_outputs.len(),
                operation.outputs().len()
            ),
        });
    }

    for (output_idx, (&output, value)) in operation.outputs().iter().zip(&sub_outputs).enumerate() {
        let expected = semantic_value_type(
            program,
            output,
            op.family_id(),
            output_idx,
            "extension output",
        )?;
        if value.ty != expected {
            return Err(Error::InvalidProgram {
                message: format!(
                    "extension family {:?} standard-op lowering output {output_idx} type {:?} does not match expected {:?}",
                    op.family_id(),
                    value.ty,
                    expected
                ),
            });
        }
    }
    Ok(sub_outputs)
}

fn build_standard_semantic_subprogram(
    compiled: &computegraph::compile::CompiledProgram<StdTensorOp>,
    input_dtypes: &[DType],
    input_shapes: &[Vec<DimExpr>],
) -> Result<Arc<SemanticProgram>> {
    if compiled.input_slots.len() != input_dtypes.len()
        || compiled.input_slots.len() != input_shapes.len()
    {
        return Err(Error::InvalidProgram {
            message: "extension standard-op input metadata count mismatch".to_string(),
        });
    }
    let mut builder = SemanticProgramBuilder::new();
    let mut values = vec![None; compiled.n_slots];
    for (index, &slot) in compiled.input_slots.iter().enumerate() {
        let value = builder
            .input(ProgramInputSpec::new(
                input_dtypes[index],
                input_shapes[index].clone(),
            ))
            .map_err(semantic_build_error)?;
        let Some(entry) = values.get_mut(slot) else {
            return Err(Error::InvalidProgram {
                message: "extension standard-op input slot is out of bounds".to_string(),
            });
        };
        *entry = Some(value);
    }
    for instruction in &compiled.instructions {
        let inputs = instruction
            .inputs
            .iter()
            .map(|&slot| {
                values
                    .get(slot)
                    .and_then(|value| *value)
                    .ok_or_else(|| Error::InvalidProgram {
                        message: "extension standard-op input is unavailable".to_string(),
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let core = CoreSemanticOp::try_from(&instruction.operation).map_err(|source| {
            Error::InvalidProgram {
                message: format!(
                    "extension standard-op lowering returned a nested extension: {source}"
                ),
            }
        })?;
        let outputs = builder
            .add_op(core, &inputs)
            .map_err(semantic_build_error)?;
        if outputs.len() != instruction.outputs.len() {
            return Err(Error::InvalidProgram {
                message: "extension standard-op output count mismatch".to_string(),
            });
        }
        for (&slot, &output) in instruction.outputs.iter().zip(outputs.iter()) {
            let Some(entry) = values.get_mut(slot) else {
                return Err(Error::InvalidProgram {
                    message: "extension standard-op output slot is out of bounds".to_string(),
                });
            };
            if entry.replace(output).is_some() {
                return Err(Error::InvalidProgram {
                    message: "extension standard-op slot has multiple producers".to_string(),
                });
            }
        }
    }
    let outputs = compiled
        .output_slots
        .iter()
        .map(|&slot| {
            values
                .get(slot)
                .and_then(|value| *value)
                .ok_or_else(|| Error::InvalidProgram {
                    message: "extension standard-op output is unavailable".to_string(),
                })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(builder
        .finish(&outputs)
        .map_err(|source| Error::InvalidProgram {
            message: format!("extension semantic subprogram freeze failed: {source}"),
        })?
        .program)
}

fn semantic_build_error(source: tenferro_runtime::program::ProgramBuildError) -> Error {
    Error::InvalidProgram {
        message: format!("extension semantic subprogram build failed: {source}"),
    }
}

fn semantic_value_type(
    program: &SemanticProgram,
    value: ProgramValue,
    op: &'static str,
    output_index: usize,
    context: &'static str,
) -> Result<TensorType> {
    let metadata = program
        .value_metadata(value)
        .map_err(|source| Error::InvalidProgram {
            message: format!("semantic value metadata is unavailable: {source}"),
        })?;
    validate_dtype(metadata.dtype(), context)?;
    let shape = metadata
        .shape()
        .iter()
        .enumerate()
        .map(|(axis, extent)| match extent {
            ShapeExtent::Exact(DimExpr::Const(value)) => Ok(*value),
            ShapeExtent::Exact(_) => Err(Error::NonStaticShape {
                op,
                output_index,
                axis,
                kind: "symbolic",
            }),
            ShapeExtent::UpperBound(_) => Err(Error::NonStaticShape {
                op,
                output_index,
                axis,
                kind: "an upper bound",
            }),
            ShapeExtent::Unknown => Err(Error::NonStaticShape {
                op,
                output_index,
                axis,
                kind: "unknown",
            }),
        })
        .collect::<Result<Vec<_>>>()?;
    TensorType::new(shape, metadata.dtype(), context)
}

fn semantic_op_name(op: SemanticOpRef<'_>) -> &'static str {
    match op {
        SemanticOpRef::Extension(extension) => extension.family_id(),
        SemanticOpRef::Core(core) => match core {
            CoreSemanticOp::Add => "Add",
            CoreSemanticOp::Sub => "Sub",
            CoreSemanticOp::Mul => "Multiply",
            CoreSemanticOp::Neg => "Negate",
            CoreSemanticOp::Conj => "Conj",
            CoreSemanticOp::DotGeneral { .. } => "DotGeneral",
            CoreSemanticOp::Transpose { .. } => "Transpose",
            CoreSemanticOp::Reshape { .. } => "Reshape",
            CoreSemanticOp::BroadcastInDim { .. } => "BroadcastInDim",
            CoreSemanticOp::Convert { .. } => "Convert",
            CoreSemanticOp::Constant { .. } => "Constant",
            CoreSemanticOp::ReduceSum { .. } => "ReduceSum",
            CoreSemanticOp::ReduceSumSquares { .. } => "ReduceSumSquares",
            CoreSemanticOp::Div => "Divide",
            CoreSemanticOp::Rem => "Remainder",
            CoreSemanticOp::Abs => "Abs",
            CoreSemanticOp::Sign => "Sign",
            CoreSemanticOp::Maximum => "Maximum",
            CoreSemanticOp::Minimum => "Minimum",
            CoreSemanticOp::Compare(_) => "Compare",
            CoreSemanticOp::Select => "Select",
            CoreSemanticOp::Clamp => "Clamp",
            CoreSemanticOp::Exp => "Exp",
            CoreSemanticOp::Log => "Log",
            CoreSemanticOp::Sin => "Sin",
            CoreSemanticOp::Cos => "Cos",
            CoreSemanticOp::Tanh => "Tanh",
            CoreSemanticOp::Sqrt => "Sqrt",
            CoreSemanticOp::Rsqrt => "Rsqrt",
            CoreSemanticOp::Pow => "Pow",
            CoreSemanticOp::Expm1 => "Expm1",
            CoreSemanticOp::Log1p => "Log1p",
            CoreSemanticOp::ExtractDiag { .. } => "ExtractDiag",
            CoreSemanticOp::EmbedDiag { .. } => "EmbedDiag",
            CoreSemanticOp::Tril { .. } => "Tril",
            CoreSemanticOp::Triu { .. } => "Triu",
            CoreSemanticOp::Gather(_) => "Gather",
            CoreSemanticOp::GatherDynamicSliceSizes { .. } => "GatherDynamicSliceSizes",
            CoreSemanticOp::Scatter(_) => "Scatter",
            CoreSemanticOp::Slice(_) => "Slice",
            CoreSemanticOp::DynamicSlice { .. } => "DynamicSlice",
            CoreSemanticOp::DynamicUpdateSlice => "DynamicUpdateSlice",
            CoreSemanticOp::Pad(_) => "Pad",
            CoreSemanticOp::Concatenate { .. } => "Concatenate",
            CoreSemanticOp::Reverse { .. } => "Reverse",
            CoreSemanticOp::ShapeOf { .. } => "ShapeOf",
            CoreSemanticOp::DynamicTruncate { .. } => "DynamicTruncate",
            CoreSemanticOp::PadToMatch { .. } => "PadToMatch",
            CoreSemanticOp::ReduceProd { .. } => "ReduceProd",
            CoreSemanticOp::ReduceMax { .. } => "ReduceMax",
            CoreSemanticOp::ReduceMin { .. } => "ReduceMin",
            _ => "UnknownCoreSemanticOp",
        },
        _ => "UnknownSemanticOp",
    }
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
            format_f32(f32::from_le_bytes(bytes))
        }
        DType::F64 => {
            let bytes: [u8; 8] = bytes.try_into().map_err(|_| Error::InvalidProgram {
                message: format!("F64 constant expected 8 bytes, got {}", bytes.len()),
            })?;
            format_f64(f64::from_le_bytes(bytes))
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

fn lower_reduce_sum_squares(
    axes: &[usize],
    inputs: &[Value],
    output_ty: &TensorType,
    emitter: &mut Emitter,
) -> Result<Value> {
    require_input_count("reduce_sum_squares", inputs, 1)?;
    let squared = lower_same_type_binary(
        "stablehlo.multiply",
        &[inputs[0].clone(), inputs[0].clone()],
        &inputs[0].ty,
        emitter,
    )?;
    lower_reduce_sum(axes, &[squared], output_ty, emitter)
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
    let identity_perm = perm
        .iter()
        .enumerate()
        .all(|(axis, &mapped)| axis == mapped);
    if identity_perm && dot_value.ty.shape == output_ty.shape {
        return Ok(dot_value);
    }
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
    let mut seen = vec![false; input.ty.shape.len()];
    for &axis in perm {
        if axis >= input.ty.shape.len() || seen[axis] {
            return Err(Error::InvalidProgram {
                message: format!(
                    "transpose permutation must be a bijection over rank {}, got {perm:?}",
                    input.ty.shape.len()
                ),
            });
        }
        seen[axis] = true;
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
        .map_err(Error::from)?;
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

fn format_f32(value: f32) -> String {
    if value.is_finite() {
        format!("{value:.8e}")
    } else if value.is_nan() {
        format!("0x{:08x}", value.to_bits())
    } else if value.is_sign_negative() {
        "-0x7f800000".to_string()
    } else {
        "0x7f800000".to_string()
    }
}

fn format_f64(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.8e}")
    } else if value.is_nan() {
        format!("0x{:016x}", value.to_bits())
    } else if value.is_sign_negative() {
        "-0x7ff0000000000000".to_string()
    } else {
        "0x7ff0000000000000".to_string()
    }
}

#[cfg(test)]
mod tests;
