use computegraph::compile::CompiledProgram;
use computegraph::{GraphOp, Operand};
use tenferro_ops::semiring_ops::SemiringOps;
use tenferro_ops::std_tensor_op::StdTensorOp;

use super::exec::ExecProgram;
use super::stablehlo::StableHloProgram;

// --- Standard algebra path ---

pub fn lower_to_stablehlo(_prog: &CompiledProgram<StdTensorOp>) -> StableHloProgram {
    todo!()
}

// --- Custom algebra path ---
// SemiringOp<T> lowers to the same StableHloOp types but with semiring semantics.
// Not serializable to StableHLO MLIR; always goes through optimizing compiler.

pub fn lower_semiring_to_stablehlo<Op: SemiringOps>(
    _prog: &CompiledProgram<Op>,
) -> StableHloProgram {
    todo!()
}

// --- Optimizing compiler (algebra-agnostic) ---

pub fn compile_to_exec(_stablehlo: &StableHloProgram) -> ExecProgram {
    todo!()
}
