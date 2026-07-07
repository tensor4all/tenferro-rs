use tenferro_core_ops::PrimitiveOpKind;
use tenferro_tensor::{
    BackendId, DType, OperationCapability, SupportLevel, TensorBackendCapability,
};

use super::CudaBackend;

const U: SupportLevel = SupportLevel::Unsupported;
const F: SupportLevel = SupportLevel::FallbackCopy;
const N: SupportLevel = SupportLevel::Native;

/// Return the CUDA backend operation capability descriptor table.
///
/// # Examples
///
/// ```rust
/// use tenferro_core_ops::PrimitiveOpKind;
/// use tenferro_gpu::cuda_capabilities;
/// use tenferro_tensor::{DType, SupportLevel};
///
/// let add = cuda_capabilities()
///     .iter()
///     .find(|entry| entry.op == PrimitiveOpKind::Add && entry.dtype == DType::I32)
///     .unwrap();
/// assert_eq!(add.result, SupportLevel::Native);
/// ```
#[must_use]
pub const fn cuda_capabilities() -> &'static [OperationCapability] {
    CUDA_CAPABILITIES
}

impl TensorBackendCapability for CudaBackend {
    fn backend_id(&self) -> BackendId {
        BackendId::Cuda
    }

    fn capabilities(&self) -> &'static [OperationCapability] {
        CUDA_CAPABILITIES
    }
}

const fn owned_level(
    op: PrimitiveOpKind,
    dtype: DType,
    output_dtype: DType,
    level: SupportLevel,
) -> OperationCapability {
    OperationCapability {
        backend: BackendId::Cuda,
        op,
        dtype,
        output_dtype,
        result: level,
        read_inputs: if matches!(level, SupportLevel::Native) {
            F
        } else {
            U
        },
        write_output: U,
        strided_output: U,
        accumulation: U,
    }
}

const fn same_dtype(op: PrimitiveOpKind, dtype: DType, level: SupportLevel) -> OperationCapability {
    owned_level(op, dtype, dtype, level)
}

const fn dot_native(dtype: DType) -> OperationCapability {
    OperationCapability {
        backend: BackendId::Cuda,
        op: PrimitiveOpKind::DotGeneral,
        dtype,
        output_dtype: dtype,
        result: N,
        read_inputs: N,
        write_output: N,
        strided_output: N,
        accumulation: N,
    }
}

const CUDA_CAPABILITIES: &[OperationCapability] = &[
    same_dtype(PrimitiveOpKind::Add, DType::F32, N),
    same_dtype(PrimitiveOpKind::Add, DType::F64, N),
    same_dtype(PrimitiveOpKind::Add, DType::I32, N),
    same_dtype(PrimitiveOpKind::Add, DType::I64, N),
    same_dtype(PrimitiveOpKind::Add, DType::C32, N),
    same_dtype(PrimitiveOpKind::Add, DType::C64, N),
    same_dtype(PrimitiveOpKind::Mul, DType::F32, N),
    same_dtype(PrimitiveOpKind::Mul, DType::F64, N),
    same_dtype(PrimitiveOpKind::Mul, DType::I32, N),
    same_dtype(PrimitiveOpKind::Mul, DType::I64, N),
    same_dtype(PrimitiveOpKind::Mul, DType::C32, N),
    same_dtype(PrimitiveOpKind::Mul, DType::C64, N),
    same_dtype(PrimitiveOpKind::Neg, DType::F32, N),
    same_dtype(PrimitiveOpKind::Neg, DType::F64, N),
    same_dtype(PrimitiveOpKind::Neg, DType::I32, U),
    same_dtype(PrimitiveOpKind::Neg, DType::I64, U),
    same_dtype(PrimitiveOpKind::Neg, DType::C32, N),
    same_dtype(PrimitiveOpKind::Neg, DType::C64, N),
    same_dtype(PrimitiveOpKind::Conj, DType::F32, N),
    same_dtype(PrimitiveOpKind::Conj, DType::F64, N),
    same_dtype(PrimitiveOpKind::Conj, DType::C32, N),
    same_dtype(PrimitiveOpKind::Conj, DType::C64, N),
    same_dtype(PrimitiveOpKind::Div, DType::F32, N),
    same_dtype(PrimitiveOpKind::Div, DType::F64, N),
    same_dtype(PrimitiveOpKind::Div, DType::C32, N),
    same_dtype(PrimitiveOpKind::Div, DType::C64, N),
    same_dtype(PrimitiveOpKind::Abs, DType::F32, N),
    same_dtype(PrimitiveOpKind::Abs, DType::F64, N),
    owned_level(PrimitiveOpKind::Abs, DType::C32, DType::F32, U),
    owned_level(PrimitiveOpKind::Abs, DType::C64, DType::F64, U),
    same_dtype(PrimitiveOpKind::Sign, DType::F32, N),
    same_dtype(PrimitiveOpKind::Sign, DType::F64, N),
    same_dtype(PrimitiveOpKind::Maximum, DType::F32, N),
    same_dtype(PrimitiveOpKind::Maximum, DType::F64, N),
    same_dtype(PrimitiveOpKind::Minimum, DType::F32, N),
    same_dtype(PrimitiveOpKind::Minimum, DType::F64, N),
    owned_level(PrimitiveOpKind::Compare, DType::F32, DType::Bool, N),
    owned_level(PrimitiveOpKind::Compare, DType::F64, DType::Bool, N),
    owned_level(PrimitiveOpKind::Compare, DType::I32, DType::Bool, N),
    owned_level(PrimitiveOpKind::Compare, DType::I64, DType::Bool, N),
    owned_level(PrimitiveOpKind::Compare, DType::Bool, DType::Bool, U),
    same_dtype(PrimitiveOpKind::Select, DType::F32, N),
    same_dtype(PrimitiveOpKind::Select, DType::F64, N),
    same_dtype(PrimitiveOpKind::Select, DType::I32, N),
    same_dtype(PrimitiveOpKind::Select, DType::I64, N),
    same_dtype(PrimitiveOpKind::Select, DType::Bool, U),
    same_dtype(PrimitiveOpKind::Select, DType::C32, U),
    same_dtype(PrimitiveOpKind::Select, DType::C64, U),
    same_dtype(PrimitiveOpKind::Clamp, DType::F32, N),
    same_dtype(PrimitiveOpKind::Clamp, DType::F64, N),
    same_dtype(PrimitiveOpKind::Exp, DType::F32, N),
    same_dtype(PrimitiveOpKind::Exp, DType::F64, N),
    same_dtype(PrimitiveOpKind::Exp, DType::C32, U),
    same_dtype(PrimitiveOpKind::Exp, DType::C64, U),
    same_dtype(PrimitiveOpKind::Log, DType::F32, N),
    same_dtype(PrimitiveOpKind::Log, DType::F64, N),
    same_dtype(PrimitiveOpKind::Log, DType::C32, U),
    same_dtype(PrimitiveOpKind::Log, DType::C64, U),
    same_dtype(PrimitiveOpKind::Sin, DType::F32, N),
    same_dtype(PrimitiveOpKind::Sin, DType::F64, N),
    same_dtype(PrimitiveOpKind::Sin, DType::C32, U),
    same_dtype(PrimitiveOpKind::Sin, DType::C64, U),
    same_dtype(PrimitiveOpKind::Cos, DType::F32, N),
    same_dtype(PrimitiveOpKind::Cos, DType::F64, N),
    same_dtype(PrimitiveOpKind::Cos, DType::C32, U),
    same_dtype(PrimitiveOpKind::Cos, DType::C64, U),
    same_dtype(PrimitiveOpKind::Tanh, DType::F32, N),
    same_dtype(PrimitiveOpKind::Tanh, DType::F64, N),
    same_dtype(PrimitiveOpKind::Tanh, DType::C32, U),
    same_dtype(PrimitiveOpKind::Tanh, DType::C64, U),
    same_dtype(PrimitiveOpKind::Sqrt, DType::F32, N),
    same_dtype(PrimitiveOpKind::Sqrt, DType::F64, N),
    same_dtype(PrimitiveOpKind::Sqrt, DType::C32, U),
    same_dtype(PrimitiveOpKind::Sqrt, DType::C64, U),
    same_dtype(PrimitiveOpKind::Rsqrt, DType::F32, N),
    same_dtype(PrimitiveOpKind::Rsqrt, DType::F64, N),
    same_dtype(PrimitiveOpKind::Rsqrt, DType::C32, U),
    same_dtype(PrimitiveOpKind::Rsqrt, DType::C64, U),
    same_dtype(PrimitiveOpKind::Pow, DType::F32, N),
    same_dtype(PrimitiveOpKind::Pow, DType::F64, N),
    same_dtype(PrimitiveOpKind::Pow, DType::C32, U),
    same_dtype(PrimitiveOpKind::Pow, DType::C64, U),
    same_dtype(PrimitiveOpKind::Expm1, DType::F32, N),
    same_dtype(PrimitiveOpKind::Expm1, DType::F64, N),
    same_dtype(PrimitiveOpKind::Expm1, DType::C32, U),
    same_dtype(PrimitiveOpKind::Expm1, DType::C64, U),
    same_dtype(PrimitiveOpKind::Log1p, DType::F32, N),
    same_dtype(PrimitiveOpKind::Log1p, DType::F64, N),
    same_dtype(PrimitiveOpKind::Log1p, DType::C32, U),
    same_dtype(PrimitiveOpKind::Log1p, DType::C64, U),
    same_dtype(PrimitiveOpKind::ReduceSum, DType::F32, N),
    same_dtype(PrimitiveOpKind::ReduceSum, DType::F64, N),
    same_dtype(PrimitiveOpKind::ReduceSum, DType::I32, N),
    same_dtype(PrimitiveOpKind::ReduceSum, DType::I64, N),
    same_dtype(PrimitiveOpKind::ReduceSum, DType::C32, N),
    same_dtype(PrimitiveOpKind::ReduceSum, DType::C64, N),
    same_dtype(PrimitiveOpKind::ReduceProd, DType::F32, N),
    same_dtype(PrimitiveOpKind::ReduceProd, DType::F64, N),
    same_dtype(PrimitiveOpKind::ReduceProd, DType::I32, N),
    same_dtype(PrimitiveOpKind::ReduceProd, DType::I64, N),
    same_dtype(PrimitiveOpKind::ReduceProd, DType::C32, N),
    same_dtype(PrimitiveOpKind::ReduceProd, DType::C64, N),
    same_dtype(PrimitiveOpKind::ReduceMax, DType::F32, N),
    same_dtype(PrimitiveOpKind::ReduceMax, DType::F64, N),
    same_dtype(PrimitiveOpKind::ReduceMin, DType::F32, N),
    same_dtype(PrimitiveOpKind::ReduceMin, DType::F64, N),
    dot_native(DType::F32),
    dot_native(DType::F64),
    dot_native(DType::C32),
    dot_native(DType::C64),
];
