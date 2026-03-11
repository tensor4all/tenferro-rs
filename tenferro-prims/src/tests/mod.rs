mod analytic_phase1;
mod scalar_phase1;

use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{
    AnalyticBinaryOp, AnalyticPrimsDescriptor, AnalyticReductionOp, AnalyticUnaryOp, CpuBackend,
    CpuContext, Extension, PrimDescriptor, ReduceOp, ScalarBinaryOp, ScalarPrimsDescriptor,
    ScalarReductionOp, ScalarUnaryOp, SemiringBinaryOp, SemiringCoreDescriptor,
    SemiringFastPathDescriptor, TensorAnalyticPrims, TensorScalarPrims, TensorSemiringCore,
    TensorSemiringFastPath, UnaryOp,
};

#[test]
fn protocol_smoke_semiring_core_can_plan_make_contiguous() {
    let mut ctx = CpuContext::new(1);
    let desc = SemiringCoreDescriptor::MakeContiguous;
    let result = <CpuBackend as TensorSemiringCore<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[2, 3], &[2, 3]],
    );
    assert!(result.is_ok());
}

#[test]
fn protocol_smoke_semiring_fast_path_can_plan_elementwise_mul() {
    let mut ctx = CpuContext::new(1);
    let desc = SemiringFastPathDescriptor::ElementwiseBinary {
        op: SemiringBinaryOp::Mul,
    };
    let result = <CpuBackend as TensorSemiringFastPath<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[2, 3], &[2, 3], &[2, 3]],
    );
    assert!(result.is_ok());
}

#[test]
fn protocol_smoke_scalar_prims_can_plan_reciprocal() {
    let mut ctx = CpuContext::new(1);
    let desc = ScalarPrimsDescriptor::PointwiseUnary {
        op: ScalarUnaryOp::Reciprocal,
    };
    let result = <CpuBackend as TensorScalarPrims<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[2, 3], &[2, 3]],
    );
    assert!(result.is_ok());
}

#[test]
fn protocol_smoke_scalar_prims_can_plan_sum_reduction() {
    let mut ctx = CpuContext::new(1);
    let desc = ScalarPrimsDescriptor::Reduction {
        modes_a: vec![0, 1],
        modes_c: vec![0],
        op: ScalarReductionOp::Sum,
    };
    let result = <CpuBackend as TensorScalarPrims<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[2, 3], &[2][..]],
    );
    assert!(result.is_ok());
}

#[test]
fn protocol_smoke_analytic_prims_can_plan_sqrt() {
    let mut ctx = CpuContext::new(1);
    let desc = AnalyticPrimsDescriptor::PointwiseUnary {
        op: AnalyticUnaryOp::Sqrt,
    };
    let result = <CpuBackend as TensorAnalyticPrims<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[2, 3], &[2, 3]],
    );
    assert!(result.is_ok());
}

#[test]
fn scalar_prims_legacy_mapping_and_support_cover_supported_and_unsupported_ops() {
    assert_eq!(
        ScalarPrimsDescriptor::PointwiseUnary {
            op: ScalarUnaryOp::Neg,
        }
        .to_legacy()
        .unwrap(),
        PrimDescriptor::ElementwiseUnary {
            op: UnaryOp::Negate,
        }
    );
    assert_eq!(
        ScalarPrimsDescriptor::PointwiseUnary {
            op: ScalarUnaryOp::Conj,
        }
        .to_legacy()
        .unwrap(),
        PrimDescriptor::ElementwiseUnary { op: UnaryOp::Conj }
    );
    assert_eq!(
        ScalarPrimsDescriptor::PointwiseUnary {
            op: ScalarUnaryOp::Abs,
        }
        .to_legacy()
        .unwrap(),
        PrimDescriptor::ElementwiseUnary { op: UnaryOp::Abs }
    );
    assert_eq!(
        ScalarPrimsDescriptor::PointwiseUnary {
            op: ScalarUnaryOp::Reciprocal,
        }
        .to_legacy()
        .unwrap(),
        PrimDescriptor::ElementwiseUnary {
            op: UnaryOp::Reciprocal,
        }
    );
    for op in [
        ScalarUnaryOp::Real,
        ScalarUnaryOp::Imag,
        ScalarUnaryOp::Square,
    ] {
        assert!(ScalarPrimsDescriptor::PointwiseUnary { op }
            .to_legacy()
            .is_err());
        assert!(
            !<CpuBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
                ScalarPrimsDescriptor::PointwiseUnary { op }
            )
        );
    }

    assert_eq!(
        ScalarPrimsDescriptor::PointwiseBinary {
            op: ScalarBinaryOp::Mul,
        }
        .to_legacy()
        .unwrap(),
        PrimDescriptor::ElementwiseMul
    );
    assert!(
        <CpuBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Mul,
            }
        )
    );
    assert!(
        <CpuBackend as crate::TensorPrims<Standard<f64>>>::has_extension_for(
            Extension::ElementwiseMul
        )
    );
    for op in [
        ScalarBinaryOp::Add,
        ScalarBinaryOp::Sub,
        ScalarBinaryOp::Div,
        ScalarBinaryOp::Maximum,
        ScalarBinaryOp::Minimum,
        ScalarBinaryOp::ClampMin,
        ScalarBinaryOp::ClampMax,
    ] {
        assert!(ScalarPrimsDescriptor::PointwiseBinary { op }
            .to_legacy()
            .is_err());
        assert!(
            !<CpuBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
                ScalarPrimsDescriptor::PointwiseBinary { op }
            )
        );
    }

    for op in [
        ScalarReductionOp::Sum,
        ScalarReductionOp::Max,
        ScalarReductionOp::Min,
    ] {
        let desc = ScalarPrimsDescriptor::Reduction {
            modes_a: vec![0, 1],
            modes_c: vec![0],
            op,
        };
        let expected = PrimDescriptor::Reduce {
            modes_a: vec![0, 1],
            modes_c: vec![0],
            op: match op {
                ScalarReductionOp::Sum => ReduceOp::Sum,
                ScalarReductionOp::Max => ReduceOp::Max,
                ScalarReductionOp::Min => ReduceOp::Min,
                _ => unreachable!(),
            },
        };
        assert_eq!(desc.to_legacy().unwrap(), expected);
        assert!(<CpuBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(desc));
    }
    for op in [ScalarReductionOp::Prod, ScalarReductionOp::Mean] {
        let desc = ScalarPrimsDescriptor::Reduction {
            modes_a: vec![0, 1],
            modes_c: vec![0],
            op,
        };
        assert!(desc.to_legacy().is_err());
        assert!(!<CpuBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(desc));
    }
}

#[test]
fn analytic_prims_legacy_mapping_and_support_cover_supported_and_unsupported_ops() {
    assert_eq!(
        AnalyticPrimsDescriptor::PointwiseUnary {
            op: AnalyticUnaryOp::Sqrt,
        }
        .to_legacy()
        .unwrap(),
        PrimDescriptor::ElementwiseUnary { op: UnaryOp::Sqrt }
    );
    assert!(
        <CpuBackend as TensorAnalyticPrims<Standard<f64>>>::has_analytic_support(
            AnalyticPrimsDescriptor::PointwiseUnary {
                op: AnalyticUnaryOp::Sqrt,
            }
        )
    );

    for op in [
        AnalyticUnaryOp::Rsqrt,
        AnalyticUnaryOp::Exp,
        AnalyticUnaryOp::Expm1,
        AnalyticUnaryOp::Log,
        AnalyticUnaryOp::Log1p,
        AnalyticUnaryOp::Sin,
        AnalyticUnaryOp::Cos,
        AnalyticUnaryOp::Tan,
        AnalyticUnaryOp::Tanh,
    ] {
        let desc = AnalyticPrimsDescriptor::PointwiseUnary { op };
        assert!(desc.to_legacy().is_err());
        assert!(!<CpuBackend as TensorAnalyticPrims<Standard<f64>>>::has_analytic_support(desc));
    }

    for op in [
        AnalyticBinaryOp::Pow,
        AnalyticBinaryOp::Atan2,
        AnalyticBinaryOp::Hypot,
        AnalyticBinaryOp::Xlogy,
    ] {
        let desc = AnalyticPrimsDescriptor::PointwiseBinary { op };
        assert!(desc.to_legacy().is_err());
        assert!(!<CpuBackend as TensorAnalyticPrims<Standard<f64>>>::has_analytic_support(desc));
    }

    for op in [AnalyticReductionOp::Var, AnalyticReductionOp::Std] {
        let desc = AnalyticPrimsDescriptor::Reduction {
            modes_a: vec![0, 1],
            modes_c: vec![0],
            op,
        };
        assert!(desc.to_legacy().is_err());
        assert!(!<CpuBackend as TensorAnalyticPrims<Standard<f64>>>::has_analytic_support(desc));
    }
}

#[test]
fn analytic_prims_execute_sqrt_and_reject_unsupported_plan_requests() {
    let mut ctx = CpuContext::new(1);
    let desc = AnalyticPrimsDescriptor::PointwiseUnary {
        op: AnalyticUnaryOp::Sqrt,
    };
    let plan =
        <CpuBackend as TensorAnalyticPrims<Standard<f64>>>::plan(&mut ctx, &desc, &[&[2], &[2]])
            .unwrap();
    let input = Tensor::from_slice(&[4.0_f64, 9.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let mut output = Tensor::<f64>::zeros(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    <CpuBackend as TensorAnalyticPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        1.0,
        &[&input],
        0.0,
        &mut output,
    )
    .unwrap();
    assert_eq!(output.buffer().as_slice().unwrap(), &[2.0, 3.0]);

    let unsupported = AnalyticPrimsDescriptor::PointwiseBinary {
        op: AnalyticBinaryOp::Pow,
    };
    let err =
        <CpuBackend as TensorAnalyticPrims<Standard<f64>>>::plan(&mut ctx, &unsupported, &[&[2]])
            .unwrap_err();
    assert!(matches!(err, tenferro_device::Error::InvalidArgument(_)));
}
