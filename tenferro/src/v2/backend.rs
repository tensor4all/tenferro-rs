use super::exec::ExecProgram;
use tenferro_ops::config::DotGeneralConfig;

pub trait SemiringCore {
    type Operand;

    fn batched_gemm(
        &mut self,
        lhs: &Self::Operand,
        rhs: &Self::Operand,
        config: &DotGeneralConfig,
    ) -> Self::Operand;

    fn reduce_sum(&mut self, operand: &Self::Operand, axes: &[usize]) -> Self::Operand;
}

pub trait SemiringFastPath: SemiringCore {
    fn contract(
        &mut self,
        _lhs: &Self::Operand,
        _rhs: &Self::Operand,
        _config: &DotGeneralConfig,
    ) -> Option<Self::Operand> {
        None
    }

    fn elementwise_mul(
        &mut self,
        _lhs: &Self::Operand,
        _rhs: &Self::Operand,
    ) -> Option<Self::Operand> {
        None
    }

    fn elementwise_add(
        &mut self,
        _lhs: &Self::Operand,
        _rhs: &Self::Operand,
    ) -> Option<Self::Operand> {
        None
    }
}

pub fn eval_exec_ir<B: SemiringCore>(
    _backend: &mut B,
    _program: &ExecProgram,
    _inputs: &[B::Operand],
) -> Vec<B::Operand> {
    todo!()
}
