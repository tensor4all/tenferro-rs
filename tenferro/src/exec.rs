use tenferro_algebra::Semiring;
use tenferro_tensor::cpu::structural::{
    typed_broadcast_in_dim, typed_embed_diagonal, typed_extract_diagonal, typed_reshape,
    typed_transpose,
};
use tenferro_tensor::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SemiringBackend,
    SliceConfig, Tensor, TensorBackend, TypedTensor,
};

#[derive(Clone, Debug)]
pub enum ExecOp {
    Permute {
        perm: Vec<usize>,
    },
    Reshape {
        shape: Vec<usize>,
    },
    BroadcastInDim {
        shape: Vec<usize>,
        dims: Vec<usize>,
    },
    BatchedGemm(DotGeneralConfig),
    ReduceSum {
        axes: Vec<usize>,
    },
    ExtractDiag {
        axis_a: usize,
        axis_b: usize,
    },
    EmbedDiag {
        axis_a: usize,
        axis_b: usize,
    },
    Add,
    Multiply,
    Negate,
    Conj,
    Divide,
    Abs,
    Sign,
    Maximum,
    Minimum,
    Compare(CompareDir),
    Select,
    Clamp,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Sqrt,
    Rsqrt,
    Pow,
    Expm1,
    Log1p,
    Gather(GatherConfig),
    Scatter(ScatterConfig),
    Slice(SliceConfig),
    DynamicSlice {
        slice_sizes: Vec<usize>,
    },
    Pad(PadConfig),
    Concatenate {
        axis: usize,
    },
    Reverse {
        axes: Vec<usize>,
    },
    ReduceProd {
        axes: Vec<usize>,
    },
    ReduceMax {
        axes: Vec<usize>,
    },
    ReduceMin {
        axes: Vec<usize>,
    },
    Cholesky,
    TriangularSolve {
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    },
    CustomCall {
        target: String,
    },
}

#[derive(Clone, Debug)]
pub struct ExecInstruction {
    pub op: ExecOp,
    pub input_slots: Vec<usize>,
    pub output_slots: Vec<usize>,
    pub dtype: tenferro_tensor::DType,
    pub last_use: Vec<bool>,
}

#[derive(Clone, Debug)]
pub struct ExecProgram {
    pub instructions: Vec<ExecInstruction>,
    pub input_slots: Vec<usize>,
    pub output_slots: Vec<usize>,
    pub n_slots: usize,
}

fn get<'a, T>(slots: &'a [Option<T>], input_slots: &[usize], idx: usize) -> &'a T {
    slots[input_slots[idx]].as_ref().expect("missing slot")
}

pub fn eval_exec_ir<B: TensorBackend>(
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
            ExecOp::Permute { perm } => backend.transpose(get(&slots, &inst.input_slots, 0), perm),
            ExecOp::Reshape { shape } => backend.reshape(get(&slots, &inst.input_slots, 0), shape),
            ExecOp::BroadcastInDim { shape, dims } => {
                backend.broadcast_in_dim(get(&slots, &inst.input_slots, 0), shape, dims)
            }
            ExecOp::BatchedGemm(config) => backend.dot_general(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
                config,
            ),
            ExecOp::ReduceSum { axes } => {
                backend.reduce_sum(get(&slots, &inst.input_slots, 0), axes)
            }
            ExecOp::ExtractDiag { axis_a, axis_b } => {
                backend.extract_diagonal(get(&slots, &inst.input_slots, 0), *axis_a, *axis_b)
            }
            ExecOp::EmbedDiag { axis_a, axis_b } => {
                backend.embed_diagonal(get(&slots, &inst.input_slots, 0), *axis_a, *axis_b)
            }
            ExecOp::Add => backend.add(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
            ),
            ExecOp::Multiply => backend.mul(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
            ),
            ExecOp::Negate => backend.neg(get(&slots, &inst.input_slots, 0)),
            ExecOp::Conj => backend.conj(get(&slots, &inst.input_slots, 0)),
            ExecOp::Divide => backend.div(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
            ),
            ExecOp::Abs => backend.abs(get(&slots, &inst.input_slots, 0)),
            ExecOp::Sign => backend.sign(get(&slots, &inst.input_slots, 0)),
            ExecOp::Maximum => backend.maximum(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
            ),
            ExecOp::Minimum => backend.minimum(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
            ),
            ExecOp::Compare(dir) => backend.compare(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
                dir,
            ),
            ExecOp::Select => backend.select(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
                get(&slots, &inst.input_slots, 2),
            ),
            ExecOp::Clamp => backend.clamp(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
                get(&slots, &inst.input_slots, 2),
            ),
            ExecOp::Exp => backend.exp(get(&slots, &inst.input_slots, 0)),
            ExecOp::Log => backend.log(get(&slots, &inst.input_slots, 0)),
            ExecOp::Sin => backend.sin(get(&slots, &inst.input_slots, 0)),
            ExecOp::Cos => backend.cos(get(&slots, &inst.input_slots, 0)),
            ExecOp::Tanh => backend.tanh(get(&slots, &inst.input_slots, 0)),
            ExecOp::Sqrt => backend.sqrt(get(&slots, &inst.input_slots, 0)),
            ExecOp::Rsqrt => backend.rsqrt(get(&slots, &inst.input_slots, 0)),
            ExecOp::Pow => backend.pow(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
            ),
            ExecOp::Expm1 => backend.expm1(get(&slots, &inst.input_slots, 0)),
            ExecOp::Log1p => backend.log1p(get(&slots, &inst.input_slots, 0)),
            ExecOp::Gather(config) => backend.gather(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
                config,
            ),
            ExecOp::Scatter(config) => backend.scatter(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
                get(&slots, &inst.input_slots, 2),
                config,
            ),
            ExecOp::Slice(config) => backend.slice(get(&slots, &inst.input_slots, 0), config),
            ExecOp::DynamicSlice { slice_sizes } => backend.dynamic_slice(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
                slice_sizes,
            ),
            ExecOp::Pad(config) => backend.pad(get(&slots, &inst.input_slots, 0), config),
            ExecOp::Concatenate { axis } => {
                let inputs: Vec<&Tensor> = inst
                    .input_slots
                    .iter()
                    .map(|&slot| slots[slot].as_ref().expect("missing concat slot"))
                    .collect();
                backend.concatenate(&inputs, *axis)
            }
            ExecOp::Reverse { axes } => backend.reverse(get(&slots, &inst.input_slots, 0), axes),
            ExecOp::ReduceProd { axes } => {
                backend.reduce_prod(get(&slots, &inst.input_slots, 0), axes)
            }
            ExecOp::ReduceMax { axes } => {
                backend.reduce_max(get(&slots, &inst.input_slots, 0), axes)
            }
            ExecOp::ReduceMin { axes } => {
                backend.reduce_min(get(&slots, &inst.input_slots, 0), axes)
            }
            ExecOp::Cholesky => backend.cholesky(get(&slots, &inst.input_slots, 0)),
            ExecOp::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            } => backend.triangular_solve(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
                *left_side,
                *lower,
                *transpose_a,
                *unit_diagonal,
            ),
            ExecOp::CustomCall { target } => {
                let results: Vec<Tensor> = match target.as_str() {
                    "svd" => backend.svd(get(&slots, &inst.input_slots, 0)),
                    "qr" => backend.qr(get(&slots, &inst.input_slots, 0)),
                    "eigh" => backend.eigh(get(&slots, &inst.input_slots, 0)),
                    "solve" => vec![backend.solve(
                        get(&slots, &inst.input_slots, 0),
                        get(&slots, &inst.input_slots, 1),
                    )],
                    _ => todo!("custom call target {target}"),
                };
                for (i, tensor) in results.into_iter().enumerate() {
                    slots[inst.output_slots[i]] = Some(tensor);
                }
                continue;
            }
        };
        slots[inst.output_slots[0]] = Some(result);
    }

    program
        .output_slots
        .iter()
        .map(|&slot| slots[slot].take().expect("missing output slot"))
        .collect()
}

pub fn eval_semiring_ir<B, Alg>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<TypedTensor<Alg::Scalar>>,
) -> Vec<TypedTensor<Alg::Scalar>>
where
    Alg: Semiring,
    B: SemiringBackend<Alg>,
{
    let mut slots: Vec<Option<TypedTensor<Alg::Scalar>>> = vec![None; program.n_slots];
    for (i, tensor) in inputs.into_iter().enumerate() {
        slots[program.input_slots[i]] = Some(tensor);
    }

    for inst in &program.instructions {
        let result = match &inst.op {
            ExecOp::Permute { perm } => typed_transpose(get(&slots, &inst.input_slots, 0), perm),
            ExecOp::Reshape { shape } => typed_reshape(get(&slots, &inst.input_slots, 0), shape),
            ExecOp::BroadcastInDim { shape, dims } => {
                typed_broadcast_in_dim(get(&slots, &inst.input_slots, 0), shape, dims)
            }
            ExecOp::BatchedGemm(config) => backend.batched_gemm(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
                config,
            ),
            ExecOp::ReduceSum { axes } => {
                backend.reduce_sum(get(&slots, &inst.input_slots, 0), axes)
            }
            ExecOp::ExtractDiag { axis_a, axis_b } => {
                typed_extract_diagonal(get(&slots, &inst.input_slots, 0), *axis_a, *axis_b)
            }
            ExecOp::EmbedDiag { axis_a, axis_b } => {
                typed_embed_diagonal(get(&slots, &inst.input_slots, 0), *axis_a, *axis_b)
            }
            ExecOp::Add => backend.add(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
            ),
            ExecOp::Multiply => backend.mul(
                get(&slots, &inst.input_slots, 0),
                get(&slots, &inst.input_slots, 1),
            ),
            _ => panic!("non-semiring op in semiring program: {:?}", inst.op),
        };
        slots[inst.output_slots[0]] = Some(result);
    }

    program
        .output_slots
        .iter()
        .map(|&slot| slots[slot].take().expect("missing semiring output"))
        .collect()
}
