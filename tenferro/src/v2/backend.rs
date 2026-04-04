use super::exec::{ExecOp, ExecProgram};
use computegraph::Operand;
use tenferro_ops::config::DotGeneralConfig;
use tenferro_tensor::v2::Tensor;

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

pub fn eval_exec_ir_host(program: &ExecProgram, inputs: Vec<Tensor>) -> Vec<Tensor> {
    let mut slots: Vec<Option<Tensor>> = vec![None; program.n_slots];
    for (i, tensor) in inputs.into_iter().enumerate() {
        slots[program.input_slots[i]] = Some(tensor);
    }
    for inst in &program.instructions {
        let result = match &inst.op {
            ExecOp::Add => {
                let a = slots[inst.input_slots[0]].as_ref().unwrap();
                let b = slots[inst.input_slots[1]].as_ref().unwrap();
                a.add(b)
            }
            ExecOp::Multiply => {
                let a = slots[inst.input_slots[0]].as_ref().unwrap();
                let b = slots[inst.input_slots[1]].as_ref().unwrap();
                a.multiply(b)
            }
            ExecOp::Negate => slots[inst.input_slots[0]].as_ref().unwrap().neg(),
            ExecOp::Conj => slots[inst.input_slots[0]].as_ref().unwrap().conj(),
            ExecOp::BatchedGemm(config) => {
                let a = slots[inst.input_slots[0]].as_ref().unwrap();
                let b = slots[inst.input_slots[1]].as_ref().unwrap();
                a.dot_general(
                    b,
                    &config.lhs_contracting_dims,
                    &config.rhs_contracting_dims,
                    &config.lhs_batch_dims,
                    &config.rhs_batch_dims,
                )
            }
            ExecOp::Permute { perm } => slots[inst.input_slots[0]]
                .as_ref()
                .unwrap()
                .transpose_perm(perm),
            ExecOp::Reshape { shape } => {
                slots[inst.input_slots[0]].as_ref().unwrap().reshape(shape)
            }
            ExecOp::BroadcastInDim { shape, dims } => slots[inst.input_slots[0]]
                .as_ref()
                .unwrap()
                .broadcast_in_dim(shape, dims),
            ExecOp::ReduceSum { axes } => slots[inst.input_slots[0]]
                .as_ref()
                .unwrap()
                .reduce_sum(axes),
            other => todo!("eval_exec_ir_host: unsupported op {:?}", other),
        };
        slots[inst.output_slots[0]] = Some(result);
    }
    program
        .output_slots
        .iter()
        .map(|&i| slots[i].take().unwrap())
        .collect()
}
