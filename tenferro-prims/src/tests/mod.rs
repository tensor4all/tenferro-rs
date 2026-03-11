mod analytic_phase1;
mod scalar_phase1;

use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_device::{Error, Result};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{
    AnalyticBinaryOp, AnalyticPrimsDescriptor, AnalyticReductionOp, AnalyticUnaryOp, CpuBackend,
    CpuContext, Extension, PrimDescriptor, ReduceOp, ScalarBinaryOp, ScalarPrimsDescriptor,
    ScalarReductionOp, ScalarUnaryOp, SemiringBinaryOp, SemiringCoreDescriptor,
    SemiringFastPathDescriptor, TensorAnalyticPrims, TensorScalarPrims, TensorSemiringCore,
    TensorSemiringFastPath, UnaryOp,
};

fn scalar_to_legacy(desc: &ScalarPrimsDescriptor) -> Result<PrimDescriptor> {
    match desc {
        ScalarPrimsDescriptor::PointwiseUnary {
            op: ScalarUnaryOp::Neg,
        } => Ok(PrimDescriptor::ElementwiseUnary {
            op: UnaryOp::Negate,
        }),
        ScalarPrimsDescriptor::PointwiseUnary {
            op: ScalarUnaryOp::Conj,
        } => Ok(PrimDescriptor::ElementwiseUnary { op: UnaryOp::Conj }),
        ScalarPrimsDescriptor::PointwiseUnary {
            op: ScalarUnaryOp::Abs,
        } => Ok(PrimDescriptor::ElementwiseUnary { op: UnaryOp::Abs }),
        ScalarPrimsDescriptor::PointwiseUnary {
            op: ScalarUnaryOp::Reciprocal,
        } => Ok(PrimDescriptor::ElementwiseUnary {
            op: UnaryOp::Reciprocal,
        }),
        ScalarPrimsDescriptor::PointwiseUnary { op } => Err(Error::InvalidArgument(format!(
            "scalar unary operation {op:?} is not wired to the legacy prim surface yet"
        ))),
        ScalarPrimsDescriptor::PointwiseBinary {
            op: ScalarBinaryOp::Mul,
        } => Ok(PrimDescriptor::ElementwiseMul),
        ScalarPrimsDescriptor::PointwiseBinary { op } => Err(Error::InvalidArgument(format!(
            "scalar binary operation {op:?} is not wired to the legacy prim surface yet"
        ))),
        ScalarPrimsDescriptor::Reduction {
            modes_a,
            modes_c,
            op: ScalarReductionOp::Sum,
        } => Ok(PrimDescriptor::Reduce {
            modes_a: modes_a.clone(),
            modes_c: modes_c.clone(),
            op: ReduceOp::Sum,
        }),
        ScalarPrimsDescriptor::Reduction {
            modes_a,
            modes_c,
            op: ScalarReductionOp::Max,
        } => Ok(PrimDescriptor::Reduce {
            modes_a: modes_a.clone(),
            modes_c: modes_c.clone(),
            op: ReduceOp::Max,
        }),
        ScalarPrimsDescriptor::Reduction {
            modes_a,
            modes_c,
            op: ScalarReductionOp::Min,
        } => Ok(PrimDescriptor::Reduce {
            modes_a: modes_a.clone(),
            modes_c: modes_c.clone(),
            op: ReduceOp::Min,
        }),
        ScalarPrimsDescriptor::Reduction { op, .. } => Err(Error::InvalidArgument(format!(
            "scalar reduction {op:?} is not wired to the legacy prim surface yet"
        ))),
    }
}

fn analytic_to_legacy(desc: &AnalyticPrimsDescriptor) -> Result<PrimDescriptor> {
    match desc {
        AnalyticPrimsDescriptor::PointwiseUnary {
            op: AnalyticUnaryOp::Sqrt,
        } => Ok(PrimDescriptor::ElementwiseUnary { op: UnaryOp::Sqrt }),
        AnalyticPrimsDescriptor::PointwiseUnary { op } => Err(Error::InvalidArgument(format!(
            "analytic unary operation {op:?} is not wired to the legacy prim surface yet"
        ))),
        AnalyticPrimsDescriptor::PointwiseBinary { op } => Err(Error::InvalidArgument(format!(
            "analytic binary operation {op:?} is not wired to the legacy prim surface yet"
        ))),
        AnalyticPrimsDescriptor::Reduction { op, .. } => Err(Error::InvalidArgument(format!(
            "analytic reduction {op:?} is not wired to the legacy prim surface yet"
        ))),
    }
}

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
fn scalar_prims_legacy_bridge_is_partial_but_family_support_matches_phase1_inventory() {
    assert_eq!(
        scalar_to_legacy(&ScalarPrimsDescriptor::PointwiseUnary {
            op: ScalarUnaryOp::Neg,
        })
        .unwrap(),
        PrimDescriptor::ElementwiseUnary {
            op: UnaryOp::Negate,
        }
    );
    assert_eq!(
        scalar_to_legacy(&ScalarPrimsDescriptor::PointwiseUnary {
            op: ScalarUnaryOp::Conj,
        })
        .unwrap(),
        PrimDescriptor::ElementwiseUnary { op: UnaryOp::Conj }
    );
    assert_eq!(
        scalar_to_legacy(&ScalarPrimsDescriptor::PointwiseUnary {
            op: ScalarUnaryOp::Abs,
        })
        .unwrap(),
        PrimDescriptor::ElementwiseUnary { op: UnaryOp::Abs }
    );
    assert_eq!(
        scalar_to_legacy(&ScalarPrimsDescriptor::PointwiseUnary {
            op: ScalarUnaryOp::Reciprocal,
        })
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
        assert!(scalar_to_legacy(&ScalarPrimsDescriptor::PointwiseUnary { op }).is_err());
        assert!(
            <CpuBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
                ScalarPrimsDescriptor::PointwiseUnary { op }
            )
        );
    }

    assert_eq!(
        scalar_to_legacy(&ScalarPrimsDescriptor::PointwiseBinary {
            op: ScalarBinaryOp::Mul,
        })
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
        assert!(scalar_to_legacy(&ScalarPrimsDescriptor::PointwiseBinary { op }).is_err());
        assert!(
            <CpuBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
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
        assert_eq!(scalar_to_legacy(&desc).unwrap(), expected);
        assert!(<CpuBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(desc));
    }
    for op in [ScalarReductionOp::Prod, ScalarReductionOp::Mean] {
        let desc = ScalarPrimsDescriptor::Reduction {
            modes_a: vec![0, 1],
            modes_c: vec![0],
            op,
        };
        assert!(scalar_to_legacy(&desc).is_err());
        assert!(<CpuBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(desc));
    }
}

#[test]
fn analytic_prims_legacy_bridge_is_partial_but_family_support_matches_phase1_inventory() {
    assert_eq!(
        analytic_to_legacy(&AnalyticPrimsDescriptor::PointwiseUnary {
            op: AnalyticUnaryOp::Sqrt,
        })
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
        AnalyticUnaryOp::Asin,
        AnalyticUnaryOp::Acos,
        AnalyticUnaryOp::Atan,
        AnalyticUnaryOp::Sinh,
        AnalyticUnaryOp::Cosh,
        AnalyticUnaryOp::Asinh,
        AnalyticUnaryOp::Acosh,
        AnalyticUnaryOp::Atanh,
    ] {
        let desc = AnalyticPrimsDescriptor::PointwiseUnary { op };
        assert!(analytic_to_legacy(&desc).is_err());
        assert!(<CpuBackend as TensorAnalyticPrims<Standard<f64>>>::has_analytic_support(desc));
    }

    for op in [
        AnalyticBinaryOp::Pow,
        AnalyticBinaryOp::Atan2,
        AnalyticBinaryOp::Hypot,
        AnalyticBinaryOp::Xlogy,
    ] {
        let desc = AnalyticPrimsDescriptor::PointwiseBinary { op };
        assert!(analytic_to_legacy(&desc).is_err());
        assert!(<CpuBackend as TensorAnalyticPrims<Standard<f64>>>::has_analytic_support(desc));
    }

    for op in [AnalyticReductionOp::Var, AnalyticReductionOp::Std] {
        let desc = AnalyticPrimsDescriptor::Reduction {
            modes_a: vec![0, 1],
            modes_c: vec![0],
            op,
        };
        assert!(analytic_to_legacy(&desc).is_err());
        assert!(<CpuBackend as TensorAnalyticPrims<Standard<f64>>>::has_analytic_support(desc));
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
