mod cholesky;
mod lu;
mod qr;
mod runtime;
mod scalar_type;
mod solve;
mod solve_triangular;
mod svdvals;
mod thin_svd;
mod wrappers;

use num_complex::{Complex32, Complex64};
use tenferro_algebra::Standard;
use tenferro_device::Result;
use tenferro_prims::{
    AnalyticPrimsDescriptor, AnalyticUnaryOp, ComplexRealPrimsDescriptor, ComplexRealUnaryOp,
    ComplexScalePrimsDescriptor, ScalarBinaryOp, ScalarPrimsDescriptor, ScalarReductionOp,
    ScalarUnaryOp, TensorAnalyticPrims, TensorComplexRealPrims, TensorComplexScalePrims,
    TensorScalarPrims,
};
use tenferro_tensor::Tensor;

use super::TensorLinalgContextFor;
use crate::{
    CholeskyTensorExResult, EigTensorResult, EigenTensorResult, LinalgCapabilityOp,
    LuTensorExResult, LuTensorResult, QrTensorResult, SolveTensorExResult, SvdTensorResult,
    TensorLinalgPrims,
};
pub use scalar_type::{CudaDataType, CudaLinalgScalar};

/// Marker type for the CUDA tensor linalg backend.
///
/// # Examples
///
/// ```ignore
/// let _backend = tenferro_linalg_prims::backend::CudaTensorLinalgBackend;
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct CudaTensorLinalgBackend;

fn unsupported<T, S: CudaLinalgScalar>(op: &str) -> Result<T> {
    let _ = S::cuda_data_type();
    runtime::unsupported(op)
}

fn has_real_det_support_f32() -> bool {
    <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
        ScalarPrimsDescriptor::PointwiseBinary {
            op: ScalarBinaryOp::Mul,
        },
    ) && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
        ScalarPrimsDescriptor::Reduction {
            modes_a: vec![0],
            modes_c: vec![],
            op: ScalarReductionOp::Prod,
        },
    )
}

fn has_real_det_support_f64() -> bool {
    <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
        ScalarPrimsDescriptor::PointwiseBinary {
            op: ScalarBinaryOp::Mul,
        },
    ) && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
        ScalarPrimsDescriptor::Reduction {
            modes_a: vec![0],
            modes_c: vec![],
            op: ScalarReductionOp::Prod,
        },
    )
}

fn has_complex_det_support_c32() -> bool {
    <tenferro_prims::CudaBackend as TensorComplexScalePrims<Complex32>>::has_complex_scale_support(
        ComplexScalePrimsDescriptor::PointwiseMul,
    ) && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<Complex32>>>::has_scalar_support(
        ScalarPrimsDescriptor::Reduction {
            modes_a: vec![0],
            modes_c: vec![],
            op: ScalarReductionOp::Prod,
        },
    )
}

fn has_complex_det_support_c64() -> bool {
    <tenferro_prims::CudaBackend as TensorComplexScalePrims<Complex64>>::has_complex_scale_support(
        ComplexScalePrimsDescriptor::PointwiseMul,
    ) && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<Complex64>>>::has_scalar_support(
        ScalarPrimsDescriptor::Reduction {
            modes_a: vec![0],
            modes_c: vec![],
            op: ScalarReductionOp::Prod,
        },
    )
}

fn has_complex_slogdet_support_c32() -> bool {
    <tenferro_prims::CudaBackend as TensorComplexRealPrims<Complex32>>::has_complex_real_support(
        ComplexRealPrimsDescriptor::PointwiseUnary {
            op: ComplexRealUnaryOp::Abs,
        },
    ) && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
        ScalarPrimsDescriptor::PointwiseUnary {
            op: ScalarUnaryOp::Reciprocal,
        },
    ) && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
        ScalarPrimsDescriptor::PointwiseBinary {
            op: ScalarBinaryOp::Greater,
        },
    ) && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
        ScalarPrimsDescriptor::PointwiseTernary {
            op: tenferro_prims::ScalarTernaryOp::Where,
        },
    ) && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
        ScalarPrimsDescriptor::Reduction {
            modes_a: vec![0],
            modes_c: vec![],
            op: ScalarReductionOp::Sum,
        },
    ) && <tenferro_prims::CudaBackend as TensorAnalyticPrims<Standard<f32>>>::has_analytic_support(
        AnalyticPrimsDescriptor::PointwiseUnary {
            op: AnalyticUnaryOp::Log,
        },
    ) && <tenferro_prims::CudaBackend as TensorComplexScalePrims<Complex32>>::has_complex_scale_support(
        ComplexScalePrimsDescriptor::PointwiseMul,
    ) && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<Complex32>>>::has_scalar_support(
        ScalarPrimsDescriptor::Reduction {
            modes_a: vec![0],
            modes_c: vec![],
            op: ScalarReductionOp::Prod,
        },
    )
}

fn has_complex_slogdet_support_c64() -> bool {
    <tenferro_prims::CudaBackend as TensorComplexRealPrims<Complex64>>::has_complex_real_support(
        ComplexRealPrimsDescriptor::PointwiseUnary {
            op: ComplexRealUnaryOp::Abs,
        },
    ) && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
        ScalarPrimsDescriptor::PointwiseUnary {
            op: ScalarUnaryOp::Reciprocal,
        },
    ) && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
        ScalarPrimsDescriptor::PointwiseBinary {
            op: ScalarBinaryOp::Greater,
        },
    ) && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
        ScalarPrimsDescriptor::PointwiseTernary {
            op: tenferro_prims::ScalarTernaryOp::Where,
        },
    ) && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
        ScalarPrimsDescriptor::Reduction {
            modes_a: vec![0],
            modes_c: vec![],
            op: ScalarReductionOp::Sum,
        },
    ) && <tenferro_prims::CudaBackend as TensorAnalyticPrims<Standard<f64>>>::has_analytic_support(
        AnalyticPrimsDescriptor::PointwiseUnary {
            op: AnalyticUnaryOp::Log,
        },
    ) && <tenferro_prims::CudaBackend as TensorComplexScalePrims<Complex64>>::has_complex_scale_support(
        ComplexScalePrimsDescriptor::PointwiseMul,
    ) && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<Complex64>>>::has_scalar_support(
        ScalarPrimsDescriptor::Reduction {
            modes_a: vec![0],
            modes_c: vec![],
            op: ScalarReductionOp::Prod,
        },
    )
}

fn has_det_support<T: CudaLinalgScalar>() -> bool {
    match T::cuda_data_type() {
        scalar_type::CudaDataType::F32 => lu::has_lu_support::<T>() && has_real_det_support_f32(),
        scalar_type::CudaDataType::F64 => lu::has_lu_support::<T>() && has_real_det_support_f64(),
        scalar_type::CudaDataType::Complex32 => {
            lu::has_lu_support::<T>() && has_complex_det_support_c32()
        }
        scalar_type::CudaDataType::Complex64 => {
            lu::has_lu_support::<T>() && has_complex_det_support_c64()
        }
    }
}

fn has_real_slogdet_support_f32() -> bool {
    has_real_det_support_f32()
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseUnary {
                op: ScalarUnaryOp::Abs,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Greater,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Mul,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Add,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Sub,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::Reduction {
                modes_a: vec![0],
                modes_c: vec![],
                op: ScalarReductionOp::Sum,
            },
        )
        && <tenferro_prims::CudaBackend as TensorAnalyticPrims<Standard<f32>>>::has_analytic_support(
            AnalyticPrimsDescriptor::PointwiseUnary {
                op: AnalyticUnaryOp::Log,
            },
        )
}

fn has_real_slogdet_support_f64() -> bool {
    has_real_det_support_f64()
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseUnary {
                op: ScalarUnaryOp::Abs,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Greater,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Mul,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Add,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Sub,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::Reduction {
                modes_a: vec![0],
                modes_c: vec![],
                op: ScalarReductionOp::Sum,
            },
        )
        && <tenferro_prims::CudaBackend as TensorAnalyticPrims<Standard<f64>>>::has_analytic_support(
            AnalyticPrimsDescriptor::PointwiseUnary {
                op: AnalyticUnaryOp::Log,
            },
        )
}

fn has_slogdet_support<T: CudaLinalgScalar>() -> bool {
    match T::cuda_data_type() {
        scalar_type::CudaDataType::F32 => {
            lu::has_lu_support::<T>() && has_real_slogdet_support_f32()
        }
        scalar_type::CudaDataType::F64 => {
            lu::has_lu_support::<T>() && has_real_slogdet_support_f64()
        }
        scalar_type::CudaDataType::Complex32 => {
            lu::has_lu_support::<T>() && has_complex_slogdet_support_c32()
        }
        scalar_type::CudaDataType::Complex64 => {
            lu::has_lu_support::<T>() && has_complex_slogdet_support_c64()
        }
    }
}

fn has_real_pinv_support_f32() -> bool {
    thin_svd::has_thin_svd_support::<f32>()
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseUnary {
                op: ScalarUnaryOp::Reciprocal,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Greater,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Mul,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Add,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Sub,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::Reduction {
                modes_a: vec![0],
                modes_c: vec![],
                op: ScalarReductionOp::Max,
            },
        )
}

fn has_real_pinv_support_f64() -> bool {
    thin_svd::has_thin_svd_support::<f64>()
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseUnary {
                op: ScalarUnaryOp::Reciprocal,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Greater,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Mul,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Add,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Sub,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::Reduction {
                modes_a: vec![0],
                modes_c: vec![],
                op: ScalarReductionOp::Max,
            },
        )
}

fn has_complex_pinv_support_c32() -> bool {
    thin_svd::has_thin_svd_support::<Complex32>()
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseUnary {
                op: ScalarUnaryOp::Reciprocal,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Greater,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Mul,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Add,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Sub,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::Reduction {
                modes_a: vec![0],
                modes_c: vec![],
                op: ScalarReductionOp::Max,
            },
        )
        && <tenferro_prims::CudaBackend as TensorComplexScalePrims<Complex32>>::has_complex_scale_support(
            ComplexScalePrimsDescriptor::PointwiseMul,
        )
}

fn has_complex_pinv_support_c64() -> bool {
    thin_svd::has_thin_svd_support::<Complex64>()
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseUnary {
                op: ScalarUnaryOp::Reciprocal,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Greater,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Mul,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Add,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Sub,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::Reduction {
                modes_a: vec![0],
                modes_c: vec![],
                op: ScalarReductionOp::Max,
            },
        )
        && <tenferro_prims::CudaBackend as TensorComplexScalePrims<Complex64>>::has_complex_scale_support(
            ComplexScalePrimsDescriptor::PointwiseMul,
        )
}

fn has_real_norm_support_f32() -> bool {
    thin_svd::has_thin_svd_support::<f32>()
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseUnary {
                op: ScalarUnaryOp::Abs,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Mul,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::Reduction {
                modes_a: vec![0],
                modes_c: vec![],
                op: ScalarReductionOp::Sum,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f32>>>::has_scalar_support(
            ScalarPrimsDescriptor::Reduction {
                modes_a: vec![0],
                modes_c: vec![],
                op: ScalarReductionOp::Max,
            },
        )
        && <tenferro_prims::CudaBackend as TensorAnalyticPrims<Standard<f32>>>::has_analytic_support(
            AnalyticPrimsDescriptor::PointwiseUnary {
                op: AnalyticUnaryOp::Sqrt,
            },
        )
        && <tenferro_prims::CudaBackend as TensorAnalyticPrims<Standard<f32>>>::has_analytic_support(
            AnalyticPrimsDescriptor::PointwiseBinary {
                op: tenferro_prims::AnalyticBinaryOp::Pow,
            },
        )
}

fn has_real_norm_support_f64() -> bool {
    thin_svd::has_thin_svd_support::<f64>()
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseUnary {
                op: ScalarUnaryOp::Abs,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Mul,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::Reduction {
                modes_a: vec![0],
                modes_c: vec![],
                op: ScalarReductionOp::Sum,
            },
        )
        && <tenferro_prims::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::Reduction {
                modes_a: vec![0],
                modes_c: vec![],
                op: ScalarReductionOp::Max,
            },
        )
        && <tenferro_prims::CudaBackend as TensorAnalyticPrims<Standard<f64>>>::has_analytic_support(
            AnalyticPrimsDescriptor::PointwiseUnary {
                op: AnalyticUnaryOp::Sqrt,
            },
        )
        && <tenferro_prims::CudaBackend as TensorAnalyticPrims<Standard<f64>>>::has_analytic_support(
            AnalyticPrimsDescriptor::PointwiseBinary {
                op: tenferro_prims::AnalyticBinaryOp::Pow,
            },
        )
}

fn has_norm_support<T: CudaLinalgScalar>() -> bool {
    match T::cuda_data_type() {
        scalar_type::CudaDataType::F32 => has_real_norm_support_f32(),
        scalar_type::CudaDataType::F64 => has_real_norm_support_f64(),
        _ => false,
    }
}

fn has_pinv_support<T: CudaLinalgScalar>() -> bool {
    match T::cuda_data_type() {
        scalar_type::CudaDataType::F32 => has_real_pinv_support_f32(),
        scalar_type::CudaDataType::F64 => has_real_pinv_support_f64(),
        scalar_type::CudaDataType::Complex32 => has_complex_pinv_support_c32(),
        scalar_type::CudaDataType::Complex64 => has_complex_pinv_support_c64(),
    }
}

fn has_matrix_power_support<T: CudaLinalgScalar>() -> bool {
    solve::has_solve_support::<T>()
        && matches!(
            T::cuda_data_type(),
            scalar_type::CudaDataType::F32
                | scalar_type::CudaDataType::F64
                | scalar_type::CudaDataType::Complex32
                | scalar_type::CudaDataType::Complex64
        )
}

fn has_matrix_exp_support<T: CudaLinalgScalar>() -> bool {
    solve::has_solve_support::<T>()
        && matches!(
            T::cuda_data_type(),
            scalar_type::CudaDataType::F32
                | scalar_type::CudaDataType::F64
                | scalar_type::CudaDataType::Complex32
                | scalar_type::CudaDataType::Complex64
        )
}

impl<T: CudaLinalgScalar> TensorLinalgPrims<T> for CudaTensorLinalgBackend {
    type Context = tenferro_prims::CudaContext;

    fn has_linalg_support(op: LinalgCapabilityOp) -> bool {
        matches!(
            op,
            LinalgCapabilityOp::Solve
                | LinalgCapabilityOp::SolveEx
                | LinalgCapabilityOp::Inv
                | LinalgCapabilityOp::SolveTriangular
                | LinalgCapabilityOp::Qr
                | LinalgCapabilityOp::ThinSvd
                | LinalgCapabilityOp::LuFactor
                | LinalgCapabilityOp::LuFactorEx
                | LinalgCapabilityOp::Cholesky
                | LinalgCapabilityOp::CholeskyEx
                | LinalgCapabilityOp::Det
                | LinalgCapabilityOp::Slogdet
                | LinalgCapabilityOp::Pinv
                | LinalgCapabilityOp::MatrixPower
                | LinalgCapabilityOp::MatrixExp
                | LinalgCapabilityOp::Norm
        ) && match op {
            LinalgCapabilityOp::Solve => solve::has_solve_support::<T>(),
            LinalgCapabilityOp::SolveEx => solve::has_solve_support::<T>(),
            LinalgCapabilityOp::Inv => solve::has_solve_support::<T>(),
            LinalgCapabilityOp::SolveTriangular => {
                solve_triangular::has_solve_triangular_support::<T>()
            }
            LinalgCapabilityOp::Qr => qr::has_qr_support::<T>(),
            LinalgCapabilityOp::LuFactor | LinalgCapabilityOp::LuFactorEx => {
                lu::has_lu_support::<T>()
            }
            LinalgCapabilityOp::Cholesky | LinalgCapabilityOp::CholeskyEx => {
                cholesky::has_cholesky_support::<T>()
            }
            LinalgCapabilityOp::ThinSvd => thin_svd::has_thin_svd_support::<T>(),
            LinalgCapabilityOp::Det => has_det_support::<T>(),
            LinalgCapabilityOp::Slogdet => has_slogdet_support::<T>(),
            LinalgCapabilityOp::Pinv => has_pinv_support::<T>(),
            LinalgCapabilityOp::MatrixPower => has_matrix_power_support::<T>(),
            LinalgCapabilityOp::MatrixExp => has_matrix_exp_support::<T>(),
            LinalgCapabilityOp::Norm => has_norm_support::<T>(),
            _ => false,
        }
    }

    fn solve_ex(
        ctx: &mut Self::Context,
        a: &Tensor<T>,
        b: &Tensor<T>,
    ) -> Result<SolveTensorExResult<T>> {
        solve::solve_ex(ctx, a, b)
    }

    fn solve(ctx: &mut Self::Context, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>> {
        solve::solve(ctx, a, b)
    }

    fn solve_triangular(
        ctx: &mut Self::Context,
        a: &Tensor<T>,
        b: &Tensor<T>,
        upper: bool,
    ) -> Result<Tensor<T>> {
        solve_triangular::solve_triangular(ctx, a, b, upper)
    }

    fn qr(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<QrTensorResult<T>> {
        qr::qr(_ctx, _a)
    }

    fn thin_svd(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<SvdTensorResult<T>> {
        thin_svd::thin_svd(ctx, a)
    }

    fn svdvals(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<Tensor<T::Real>> {
        svdvals::svdvals(_ctx, _a)
    }

    fn lu_factor_ex(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<LuTensorExResult<T>> {
        lu::lu_factor_ex(_ctx, _a)
    }

    fn lu_factor(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<LuTensorResult<T>> {
        lu::lu_factor(_ctx, _a)
    }

    fn cholesky_ex(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<CholeskyTensorExResult<T>> {
        cholesky::cholesky_ex(_ctx, _a)
    }

    fn cholesky(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<Tensor<T>> {
        cholesky::cholesky(ctx, a)
    }

    fn eigen_sym(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<EigenTensorResult<T>> {
        unsupported::<EigenTensorResult<T>, T>("eigen_sym")
    }

    fn eig(_ctx: &mut Self::Context, _a: &Tensor<T>) -> Result<EigTensorResult<T>> {
        unsupported::<EigTensorResult<T>, T>("eig")
    }
}

impl<T: CudaLinalgScalar> TensorLinalgContextFor<T> for tenferro_prims::CudaContext {
    type Backend = CudaTensorLinalgBackend;
}

#[cfg(test)]
mod tests;
