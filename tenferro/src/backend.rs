use super::exec::{ExecOp, ExecProgram};
use super::{indexing, linalg, reduction, standard, structural};
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

    fn add(&mut self, lhs: &Self::Operand, rhs: &Self::Operand) -> Self::Operand;

    fn mul(&mut self, lhs: &Self::Operand, rhs: &Self::Operand) -> Self::Operand;
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

fn get<'a>(slots: &'a [Option<Tensor>], input_slots: &[usize], idx: usize) -> &'a Tensor {
    slots[input_slots[idx]].as_ref().unwrap()
}

pub fn eval_exec_ir<B: SemiringCore<Operand = Tensor>>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
) -> Vec<Tensor> {
    let mut slots: Vec<Option<Tensor>> = vec![None; program.n_slots];
    for (i, tensor) in inputs.into_iter().enumerate() {
        slots[program.input_slots[i]] = Some(tensor);
    }
    for inst in &program.instructions {
        let result = match &inst.op {
            // Structural
            ExecOp::Permute { perm } => {
                structural::permute(get(&slots, &inst.input_slots, 0), perm)
            }
            ExecOp::Reshape { shape } => {
                structural::reshape(get(&slots, &inst.input_slots, 0), shape)
            }
            ExecOp::BroadcastInDim { shape, dims } => {
                structural::broadcast_in_dim(get(&slots, &inst.input_slots, 0), shape, dims)
            }
            ExecOp::ExtractDiag { axis_a, axis_b } => {
                structural::extract_diag(get(&slots, &inst.input_slots, 0), *axis_a, *axis_b)
            }

            // Semiring
            ExecOp::BatchedGemm(config) => backend.batched_gemm(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
                config,
            ),
            ExecOp::ReduceSum { axes } => {
                backend.reduce_sum(get(&slots, &inst.input_slots, 0), axes)
            }
            ExecOp::Add => backend.add(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
            ),
            ExecOp::Multiply => backend.mul(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
            ),

            // Standard
            ExecOp::Negate => standard::neg(get(&slots, &inst.input_slots, 0)),
            ExecOp::Conj => standard::conj(get(&slots, &inst.input_slots, 0)),
            ExecOp::Divide => standard::div(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
            ),
            ExecOp::Abs => standard::abs(get(&slots, &inst.input_slots, 0)),
            ExecOp::Sign => standard::sign(get(&slots, &inst.input_slots, 0)),
            ExecOp::Maximum => standard::maximum(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
            ),
            ExecOp::Minimum => standard::minimum(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
            ),
            ExecOp::Compare(dir) => standard::compare(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
                dir,
            ),
            ExecOp::Select => standard::select(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
                get(&slots, &inst.input_slots, 2),
            ),
            ExecOp::Clamp => standard::clamp(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
                get(&slots, &inst.input_slots, 2),
            ),
            ExecOp::Exp => standard::exp(get(&slots, &inst.input_slots, 0)),
            ExecOp::Log => standard::log(get(&slots, &inst.input_slots, 0)),
            ExecOp::Sin => standard::sin(get(&slots, &inst.input_slots, 0)),
            ExecOp::Cos => standard::cos(get(&slots, &inst.input_slots, 0)),
            ExecOp::Tanh => standard::tanh(get(&slots, &inst.input_slots, 0)),
            ExecOp::Sqrt => standard::sqrt(get(&slots, &inst.input_slots, 0)),
            ExecOp::Rsqrt => standard::rsqrt(get(&slots, &inst.input_slots, 0)),
            ExecOp::Pow => standard::pow(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
            ),
            ExecOp::Expm1 => standard::expm1(get(&slots, &inst.input_slots, 0)),
            ExecOp::Log1p => standard::log1p(get(&slots, &inst.input_slots, 0)),

            // Indexing
            ExecOp::Gather(c) => indexing::gather(get(&slots, &inst.input_slots, 0), c),
            ExecOp::Scatter(c) => indexing::scatter(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
                c,
            ),
            ExecOp::Slice(c) => indexing::slice(get(&slots, &inst.input_slots, 0), c),
            ExecOp::DynamicSlice => indexing::dynamic_slice(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
            ),
            ExecOp::Pad(c) => indexing::pad(get(&slots, &inst.input_slots, 0), c),
            ExecOp::Concatenate { axis } => {
                let inputs: Vec<&Tensor> = inst
                    .input_slots
                    .iter()
                    .map(|&i| slots[i].as_ref().unwrap())
                    .collect();
                indexing::concatenate(&inputs, *axis)
            }
            ExecOp::Reverse { axes } => indexing::reverse(get(&slots, &inst.input_slots, 0), axes),

            // Reductions
            ExecOp::ReduceProd { axes } => {
                reduction::reduce_prod(get(&slots, &inst.input_slots, 0), axes)
            }
            ExecOp::ReduceMax { axes } => {
                reduction::reduce_max(get(&slots, &inst.input_slots, 0), axes)
            }
            ExecOp::ReduceMin { axes } => {
                reduction::reduce_min(get(&slots, &inst.input_slots, 0), axes)
            }

            // Linalg
            ExecOp::Cholesky => linalg::cholesky(get(&slots, &inst.input_slots, 0)),
            ExecOp::CustomCall { target } => {
                let inputs: Vec<&Tensor> = inst
                    .input_slots
                    .iter()
                    .map(|&i| slots[i].as_ref().unwrap())
                    .collect();
                linalg::custom_call(target, &inputs)
            }
        };
        slots[inst.output_slots[0]] = Some(result);
    }
    program
        .output_slots
        .iter()
        .map(|&i| slots[i].take().unwrap())
        .collect()
}
