use computegraph::compile::CompiledProgram;
use tenferro_ops::std_tensor_op::StdTensorOp;

use super::exec::ExecProgram;
use super::stablehlo::StableHloProgram;

pub fn lower_to_stablehlo(_prog: &CompiledProgram<StdTensorOp>) -> StableHloProgram {
    todo!()
}

pub fn compile_to_exec(_stablehlo: &StableHloProgram) -> ExecProgram {
    todo!()
}
