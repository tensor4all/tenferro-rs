use super::buffer_pool::BufferPool;
use crate::error::Result;
use num_complex::{Complex32, Complex64};
use std::sync::Arc;
use tenferro_algebra::Semiring;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_tensor::cpu::structural::{
    typed_broadcast_in_dim, typed_embed_diagonal, typed_extract_diagonal, typed_reshape,
    typed_transpose,
};
use tenferro_tensor::Error as TensorError;
use tenferro_tensor::{
    Buffer, CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig,
    SemiringBackend, SliceConfig, Tensor, TensorBackend, TypedTensor,
};

#[derive(Clone, Debug)]
pub enum ExecOp {
    Permute {
        perm: Vec<usize>,
    },
    Reshape {
        shape: Vec<DimExpr>,
    },
    BroadcastInDim {
        shape: Vec<DimExpr>,
        dims: Vec<usize>,
    },
    Convert {
        to: DType,
    },
    Constant {
        dtype: DType,
        bytes: Vec<u8>,
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
    Tril {
        k: i64,
    },
    Triu {
        k: i64,
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

fn get<'a, T>(slots: &'a [Option<T>], input_slots: &[usize], idx: usize) -> Result<&'a T> {
    let slot = input_slots[idx];
    slots[slot]
        .as_ref()
        .ok_or(TensorError::MissingValue { slot }.into())
}

fn resolve_tensor_shape_exprs(
    slots: &[Option<Tensor>],
    input_slots: &[usize],
    exprs: &[DimExpr],
) -> Result<Vec<usize>> {
    let mut input_shapes = Vec::with_capacity(input_slots.len());
    for &slot in input_slots {
        input_shapes.push(
            slots[slot]
                .as_ref()
                .ok_or(TensorError::MissingValue { slot })?
                .shape(),
        );
    }
    Ok(DimExpr::eval_all(exprs, &input_shapes))
}

fn resolve_semiring_shape_exprs<Alg: Semiring>(
    slots: &[Option<TypedTensor<Alg::Scalar>>],
    input_slots: &[usize],
    exprs: &[DimExpr],
) -> Result<Vec<usize>> {
    let mut input_shapes = Vec::with_capacity(input_slots.len());
    for &slot in input_slots {
        input_shapes.push(
            slots[slot]
                .as_ref()
                .ok_or(TensorError::MissingValue { slot })?
                .shape
                .as_slice(),
        );
    }
    Ok(DimExpr::eval_all(exprs, &input_shapes))
}

pub fn eval_exec_ir<B: TensorBackend>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
    pool: &mut BufferPool,
) -> Result<Vec<Tensor>> {
    let mut slots: Vec<Option<Tensor>> = vec![None; program.n_slots];
    for (i, tensor) in inputs.into_iter().enumerate() {
        slots[program.input_slots[i]] = Some(tensor);
    }

    for inst in &program.instructions {
        let result = match &inst.op {
            ExecOp::Permute { perm } => {
                backend.transpose(get(&slots, &inst.input_slots, 0)?, perm)?
            }
            ExecOp::Reshape { shape } => {
                let shape = resolve_tensor_shape_exprs(&slots, &inst.input_slots, shape)?;
                backend.reshape(get(&slots, &inst.input_slots, 0)?, &shape)?
            }
            ExecOp::BroadcastInDim { shape, dims } => {
                let shape = resolve_tensor_shape_exprs(&slots, &inst.input_slots, shape)?;
                backend.broadcast_in_dim(get(&slots, &inst.input_slots, 0)?, &shape, dims)?
            }
            ExecOp::Convert { to } => backend.convert(get(&slots, &inst.input_slots, 0)?, *to)?,
            ExecOp::Constant { dtype, bytes } => constant_tensor(*dtype, bytes),
            ExecOp::BatchedGemm(config) => backend.dot_general(
                get(&slots, &inst.input_slots, 0)?,
                get(&slots, &inst.input_slots, 1)?,
                config,
            )?,
            ExecOp::ReduceSum { axes } => {
                backend.reduce_sum(get(&slots, &inst.input_slots, 0)?, axes)?
            }
            ExecOp::ExtractDiag { axis_a, axis_b } => {
                backend.extract_diagonal(get(&slots, &inst.input_slots, 0)?, *axis_a, *axis_b)?
            }
            ExecOp::EmbedDiag { axis_a, axis_b } => {
                backend.embed_diagonal(get(&slots, &inst.input_slots, 0)?, *axis_a, *axis_b)?
            }
            ExecOp::Tril { k } => backend.tril(get(&slots, &inst.input_slots, 0)?, *k)?,
            ExecOp::Triu { k } => backend.triu(get(&slots, &inst.input_slots, 0)?, *k)?,
            ExecOp::Add => backend.add(
                get(&slots, &inst.input_slots, 0)?,
                get(&slots, &inst.input_slots, 1)?,
            )?,
            ExecOp::Multiply => backend.mul(
                get(&slots, &inst.input_slots, 0)?,
                get(&slots, &inst.input_slots, 1)?,
            )?,
            ExecOp::Negate => backend.neg(get(&slots, &inst.input_slots, 0)?)?,
            ExecOp::Conj => backend.conj(get(&slots, &inst.input_slots, 0)?)?,
            ExecOp::Divide => backend.div(
                get(&slots, &inst.input_slots, 0)?,
                get(&slots, &inst.input_slots, 1)?,
            )?,
            ExecOp::Abs => backend.abs(get(&slots, &inst.input_slots, 0)?)?,
            ExecOp::Sign => backend.sign(get(&slots, &inst.input_slots, 0)?)?,
            ExecOp::Maximum => backend.maximum(
                get(&slots, &inst.input_slots, 0)?,
                get(&slots, &inst.input_slots, 1)?,
            )?,
            ExecOp::Minimum => backend.minimum(
                get(&slots, &inst.input_slots, 0)?,
                get(&slots, &inst.input_slots, 1)?,
            )?,
            ExecOp::Compare(dir) => backend.compare(
                get(&slots, &inst.input_slots, 0)?,
                get(&slots, &inst.input_slots, 1)?,
                dir,
            )?,
            ExecOp::Select => backend.select(
                get(&slots, &inst.input_slots, 0)?,
                get(&slots, &inst.input_slots, 1)?,
                get(&slots, &inst.input_slots, 2)?,
            )?,
            ExecOp::Clamp => backend.clamp(
                get(&slots, &inst.input_slots, 0)?,
                get(&slots, &inst.input_slots, 1)?,
                get(&slots, &inst.input_slots, 2)?,
            )?,
            ExecOp::Exp => backend.exp(get(&slots, &inst.input_slots, 0)?)?,
            ExecOp::Log => backend.log(get(&slots, &inst.input_slots, 0)?)?,
            ExecOp::Sin => backend.sin(get(&slots, &inst.input_slots, 0)?)?,
            ExecOp::Cos => backend.cos(get(&slots, &inst.input_slots, 0)?)?,
            ExecOp::Tanh => backend.tanh(get(&slots, &inst.input_slots, 0)?)?,
            ExecOp::Sqrt => backend.sqrt(get(&slots, &inst.input_slots, 0)?)?,
            ExecOp::Rsqrt => backend.rsqrt(get(&slots, &inst.input_slots, 0)?)?,
            ExecOp::Pow => backend.pow(
                get(&slots, &inst.input_slots, 0)?,
                get(&slots, &inst.input_slots, 1)?,
            )?,
            ExecOp::Expm1 => backend.expm1(get(&slots, &inst.input_slots, 0)?)?,
            ExecOp::Log1p => backend.log1p(get(&slots, &inst.input_slots, 0)?)?,
            ExecOp::Gather(config) => backend.gather(
                get(&slots, &inst.input_slots, 0)?,
                get(&slots, &inst.input_slots, 1)?,
                config,
            )?,
            ExecOp::Scatter(config) => backend.scatter(
                get(&slots, &inst.input_slots, 0)?,
                get(&slots, &inst.input_slots, 1)?,
                get(&slots, &inst.input_slots, 2)?,
                config,
            )?,
            ExecOp::Slice(config) => backend.slice(get(&slots, &inst.input_slots, 0)?, config)?,
            ExecOp::DynamicSlice { slice_sizes } => backend.dynamic_slice(
                get(&slots, &inst.input_slots, 0)?,
                get(&slots, &inst.input_slots, 1)?,
                slice_sizes,
            )?,
            ExecOp::Pad(config) => backend.pad(get(&slots, &inst.input_slots, 0)?, config)?,
            ExecOp::Concatenate { axis } => {
                let mut inputs = Vec::with_capacity(inst.input_slots.len());
                for &slot in &inst.input_slots {
                    inputs.push(
                        slots[slot]
                            .as_ref()
                            .ok_or(TensorError::MissingValue { slot })?,
                    );
                }
                backend.concatenate(&inputs, *axis)?
            }
            ExecOp::Reverse { axes } => {
                backend.reverse(get(&slots, &inst.input_slots, 0)?, axes)?
            }
            ExecOp::ReduceProd { axes } => {
                backend.reduce_prod(get(&slots, &inst.input_slots, 0)?, axes)?
            }
            ExecOp::ReduceMax { axes } => {
                backend.reduce_max(get(&slots, &inst.input_slots, 0)?, axes)?
            }
            ExecOp::ReduceMin { axes } => {
                backend.reduce_min(get(&slots, &inst.input_slots, 0)?, axes)?
            }
            ExecOp::Cholesky => backend.cholesky(get(&slots, &inst.input_slots, 0)?)?,
            ExecOp::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            } => backend.triangular_solve(
                get(&slots, &inst.input_slots, 0)?,
                get(&slots, &inst.input_slots, 1)?,
                *left_side,
                *lower,
                *transpose_a,
                *unit_diagonal,
            )?,
            ExecOp::CustomCall { target } => {
                let results: Vec<Tensor> = match target.as_str() {
                    "lu" => backend.lu(get(&slots, &inst.input_slots, 0)?),
                    "svd" => backend.svd(get(&slots, &inst.input_slots, 0)?),
                    "qr" => backend.qr(get(&slots, &inst.input_slots, 0)?),
                    "eigh" => backend.eigh(get(&slots, &inst.input_slots, 0)?),
                    "eig" => backend.eig(get(&slots, &inst.input_slots, 0)?),
                    _ => todo!("custom call target {target}"),
                }?;
                for (i, tensor) in results.into_iter().enumerate() {
                    slots[inst.output_slots[i]] = Some(tensor);
                }
                reclaim_last_use_inputs(&mut slots, inst, pool);
                continue;
            }
        };
        slots[inst.output_slots[0]] = Some(result);
        reclaim_last_use_inputs(&mut slots, inst, pool);
    }

    program
        .output_slots
        .iter()
        .map(|&slot| {
            slots[slot]
                .take()
                .ok_or(TensorError::MissingValue { slot }.into())
        })
        .collect()
}

fn constant_tensor(dtype: DType, bytes: &[u8]) -> Tensor {
    match dtype {
        DType::F64 => Tensor::F64(TypedTensor::from_vec(
            vec![],
            vec![f64::from_le_bytes(exact_bytes::<8>(dtype, bytes))],
        )),
        DType::F32 => Tensor::F32(TypedTensor::from_vec(
            vec![],
            vec![f32::from_le_bytes(exact_bytes::<4>(dtype, bytes))],
        )),
        DType::C64 => {
            let data = exact_bytes::<16>(dtype, bytes);
            let mut re_bytes = [0u8; 8];
            let mut im_bytes = [0u8; 8];
            re_bytes.copy_from_slice(&data[..8]);
            im_bytes.copy_from_slice(&data[8..]);
            let re = f64::from_le_bytes(re_bytes);
            let im = f64::from_le_bytes(im_bytes);
            Tensor::C64(TypedTensor::from_vec(vec![], vec![Complex64::new(re, im)]))
        }
        DType::C32 => {
            let data = exact_bytes::<8>(dtype, bytes);
            let mut re_bytes = [0u8; 4];
            let mut im_bytes = [0u8; 4];
            re_bytes.copy_from_slice(&data[..4]);
            im_bytes.copy_from_slice(&data[4..]);
            let re = f32::from_le_bytes(re_bytes);
            let im = f32::from_le_bytes(im_bytes);
            Tensor::C32(TypedTensor::from_vec(vec![], vec![Complex32::new(re, im)]))
        }
    }
}

fn exact_bytes<const N: usize>(dtype: DType, bytes: &[u8]) -> [u8; N] {
    if bytes.len() != N {
        panic!(
            "constant {:?} expected {} bytes, got {}",
            dtype,
            N,
            bytes.len()
        );
    }
    let mut out = [0u8; N];
    out.copy_from_slice(bytes);
    out
}

fn reclaim_last_use_inputs(
    slots: &mut [Option<Tensor>],
    inst: &ExecInstruction,
    pool: &mut BufferPool,
) {
    for (i, &is_last) in inst.last_use.iter().enumerate() {
        if is_last {
            if let Some(tensor) = slots[inst.input_slots[i]].take() {
                reclaim_tensor_buffer(tensor, pool);
            }
        }
    }
}

fn reclaim_tensor_buffer(tensor: Tensor, pool: &mut BufferPool) {
    let bytes = match tensor {
        Tensor::F64(t) => extract_host_bytes(t),
        Tensor::F32(t) => extract_host_bytes(t),
        Tensor::C64(t) => extract_host_bytes(t),
        Tensor::C32(t) => extract_host_bytes(t),
    };

    if let Some(buf) = bytes {
        pool.return_buffer(buf);
    }
}

fn extract_host_bytes<T>(typed: TypedTensor<T>) -> Option<Vec<u8>> {
    if !typed.is_contiguous_col_major() {
        return None;
    }
    match typed.buffer {
        Buffer::Host(data) => {
            let data = Arc::try_unwrap(data).ok()?;
            let mut data = std::mem::ManuallyDrop::new(data);
            let ptr = data.as_mut_ptr() as *mut u8;
            let len = data.len() * std::mem::size_of::<T>();
            let cap = data.capacity() * std::mem::size_of::<T>();
            Some(unsafe { Vec::from_raw_parts(ptr, len, cap) })
        }
        Buffer::Backend(_) => None,
    }
}

pub fn eval_semiring_ir<B, Alg>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<TypedTensor<Alg::Scalar>>,
) -> Result<Vec<TypedTensor<Alg::Scalar>>>
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
            ExecOp::Permute { perm } => typed_transpose(get(&slots, &inst.input_slots, 0)?, perm),
            ExecOp::Reshape { shape } => {
                let shape = resolve_semiring_shape_exprs::<Alg>(&slots, &inst.input_slots, shape)?;
                typed_reshape(get(&slots, &inst.input_slots, 0)?, &shape)
            }
            ExecOp::BroadcastInDim { shape, dims } => {
                let shape = resolve_semiring_shape_exprs::<Alg>(&slots, &inst.input_slots, shape)?;
                typed_broadcast_in_dim(get(&slots, &inst.input_slots, 0)?, &shape, dims)
            }
            ExecOp::BatchedGemm(config) => backend.batched_gemm(
                get(&slots, &inst.input_slots, 0)?,
                get(&slots, &inst.input_slots, 1)?,
                config,
            ),
            ExecOp::ReduceSum { axes } => {
                backend.reduce_sum(get(&slots, &inst.input_slots, 0)?, axes)
            }
            ExecOp::ExtractDiag { axis_a, axis_b } => {
                typed_extract_diagonal(get(&slots, &inst.input_slots, 0)?, *axis_a, *axis_b)
            }
            ExecOp::EmbedDiag { axis_a, axis_b } => {
                typed_embed_diagonal(get(&slots, &inst.input_slots, 0)?, *axis_a, *axis_b)
            }
            ExecOp::Add => backend.add(
                get(&slots, &inst.input_slots, 0)?,
                get(&slots, &inst.input_slots, 1)?,
            ),
            ExecOp::Multiply => backend.mul(
                get(&slots, &inst.input_slots, 0)?,
                get(&slots, &inst.input_slots, 1)?,
            ),
            _ => panic!("non-semiring op in semiring program: {:?}", inst.op),
        };
        slots[inst.output_slots[0]] = Some(result?);
    }

    program
        .output_slots
        .iter()
        .map(|&slot| {
            slots[slot]
                .take()
                .ok_or(TensorError::MissingValue { slot }.into())
        })
        .collect()
}
