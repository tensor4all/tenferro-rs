use computegraph::compile::CompiledProgram;
use tenferro_ops::semiring_ops::SemiringOps;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::v2::DType;

use super::exec::{ExecInstruction, ExecOp, ExecProgram};
use super::stablehlo::{StableHloInstruction, StableHloOp, StableHloProgram};

pub fn lower_to_stablehlo(prog: &CompiledProgram<StdTensorOp>) -> StableHloProgram {
    let instructions = prog
        .instructions
        .iter()
        .map(|instr| {
            let op = match &instr.op {
                StdTensorOp::Add => StableHloOp::Add,
                StdTensorOp::Mul => StableHloOp::Multiply,
                StdTensorOp::Neg => StableHloOp::Negate,
                StdTensorOp::Conj => StableHloOp::Conj,
                StdTensorOp::DotGeneral(c) => StableHloOp::DotGeneral(c.clone()),
                StdTensorOp::Transpose { perm } => StableHloOp::Transpose { perm: perm.clone() },
                StdTensorOp::Reshape { shape } => StableHloOp::Reshape {
                    shape: shape.clone(),
                },
                StdTensorOp::BroadcastInDim { shape, dims } => StableHloOp::BroadcastInDim {
                    shape: shape.clone(),
                    dims: dims.clone(),
                },
                StdTensorOp::ReduceSum { axes } => StableHloOp::ReduceSum { axes: axes.clone() },
                StdTensorOp::ExtractDiag { axis_a, axis_b } => StableHloOp::ExtractDiag {
                    axis_a: *axis_a,
                    axis_b: *axis_b,
                },
                StdTensorOp::Div => StableHloOp::Divide,
                StdTensorOp::Abs => StableHloOp::Abs,
                StdTensorOp::Sign => StableHloOp::Sign,
                StdTensorOp::Maximum => StableHloOp::Maximum,
                StdTensorOp::Minimum => StableHloOp::Minimum,
                StdTensorOp::Compare(dir) => StableHloOp::Compare(dir.clone()),
                StdTensorOp::Select => StableHloOp::Select,
                StdTensorOp::Clamp => StableHloOp::Clamp,
                StdTensorOp::Exp => StableHloOp::Exp,
                StdTensorOp::Log => StableHloOp::Log,
                StdTensorOp::Sin => StableHloOp::Sin,
                StdTensorOp::Cos => StableHloOp::Cos,
                StdTensorOp::Tanh => StableHloOp::Tanh,
                StdTensorOp::Sqrt => StableHloOp::Sqrt,
                StdTensorOp::Rsqrt => StableHloOp::Rsqrt,
                StdTensorOp::Pow => StableHloOp::Pow,
                StdTensorOp::Expm1 => StableHloOp::Expm1,
                StdTensorOp::Log1p => StableHloOp::Log1p,
                _ => todo!("lower_to_stablehlo: unsupported op {:?}", instr.op),
            };
            StableHloInstruction {
                op,
                input_slots: instr.inputs.clone(),
                output_slots: instr.outputs.clone(),
                dtype: DType::F64,
            }
        })
        .collect();
    StableHloProgram {
        instructions,
        input_slots: prog.input_slots.clone(),
        output_slots: prog.output_slots.clone(),
        n_slots: prog.n_slots,
    }
}

pub fn lower_semiring_to_stablehlo<Op: SemiringOps>(
    _prog: &CompiledProgram<Op>,
) -> StableHloProgram {
    todo!()
}

pub fn compile_to_exec(stablehlo: &StableHloProgram) -> ExecProgram {
    let instructions: Vec<ExecInstruction> = stablehlo
        .instructions
        .iter()
        .enumerate()
        .map(|(idx, instr)| {
            let op = match &instr.op {
                StableHloOp::Add => ExecOp::Add,
                StableHloOp::Multiply => ExecOp::Multiply,
                StableHloOp::Negate => ExecOp::Negate,
                StableHloOp::Conj => ExecOp::Conj,
                StableHloOp::DotGeneral(c) => ExecOp::BatchedGemm(c.clone()),
                StableHloOp::Transpose { perm } => ExecOp::Permute { perm: perm.clone() },
                StableHloOp::Reshape { shape } => ExecOp::Reshape {
                    shape: shape.clone(),
                },
                StableHloOp::BroadcastInDim { shape, dims } => ExecOp::BroadcastInDim {
                    shape: shape.clone(),
                    dims: dims.clone(),
                },
                StableHloOp::ReduceSum { axes } => ExecOp::ReduceSum { axes: axes.clone() },
                StableHloOp::ExtractDiag { axis_a, axis_b } => ExecOp::ExtractDiag {
                    axis_a: *axis_a,
                    axis_b: *axis_b,
                },
                StableHloOp::Divide => ExecOp::Divide,
                StableHloOp::Abs => ExecOp::Abs,
                StableHloOp::Sign => ExecOp::Sign,
                StableHloOp::Maximum => ExecOp::Maximum,
                StableHloOp::Minimum => ExecOp::Minimum,
                StableHloOp::Compare(dir) => ExecOp::Compare(dir.clone()),
                StableHloOp::Select => ExecOp::Select,
                StableHloOp::Clamp => ExecOp::Clamp,
                StableHloOp::Exp => ExecOp::Exp,
                StableHloOp::Log => ExecOp::Log,
                StableHloOp::Sin => ExecOp::Sin,
                StableHloOp::Cos => ExecOp::Cos,
                StableHloOp::Tanh => ExecOp::Tanh,
                StableHloOp::Sqrt => ExecOp::Sqrt,
                StableHloOp::Rsqrt => ExecOp::Rsqrt,
                StableHloOp::Pow => ExecOp::Pow,
                StableHloOp::Expm1 => ExecOp::Expm1,
                StableHloOp::Log1p => ExecOp::Log1p,
                _ => todo!("compile_to_exec: unsupported op {:?}", instr.op),
            };

            let last_use = compute_last_use(
                &instr.input_slots,
                idx,
                &stablehlo.instructions,
                &stablehlo.output_slots,
            );

            ExecInstruction {
                op,
                input_slots: instr.input_slots.clone(),
                output_slots: instr.output_slots.clone(),
                dtype: instr.dtype,
                last_use,
            }
        })
        .collect();

    ExecProgram {
        instructions,
        input_slots: stablehlo.input_slots.clone(),
        output_slots: stablehlo.output_slots.clone(),
        n_slots: stablehlo.n_slots,
    }
}

fn compute_last_use(
    input_slots: &[usize],
    current_idx: usize,
    all_instructions: &[StableHloInstruction],
    output_slots: &[usize],
) -> Vec<bool> {
    input_slots
        .iter()
        .map(|&slot| {
            if output_slots.contains(&slot) {
                return false;
            }
            for later in &all_instructions[current_idx + 1..] {
                if later.input_slots.contains(&slot) {
                    return false;
                }
            }
            true
        })
        .collect()
}
