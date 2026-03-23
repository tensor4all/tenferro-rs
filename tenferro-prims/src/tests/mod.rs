mod analytic_phase1;
mod backend_stubs;
mod complex_real_phase1;
mod complex_scale_phase1;
mod metadata_contract_phase1;
mod metadata_bridge_phase1;
mod metadata_phase1;
mod metadata_phase2;
mod organization;
mod scalar_phase1;
mod scalar_phase2;

use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{
    AnalyticBinaryOp, AnalyticPrimsDescriptor, AnalyticReductionOp, AnalyticUnaryOp, CpuBackend,
    CpuContext, ScalarBinaryOp, ScalarPrimsDescriptor, ScalarReductionOp, ScalarUnaryOp,
    SemiringBinaryOp, SemiringCoreDescriptor, SemiringFastPathDescriptor, TensorAnalyticPrims,
    TensorScalarPrims, TensorSemiringCore, TensorSemiringFastPath,
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
fn scalar_prims_family_support_matches_phase1_inventory() {
    for op in [
        ScalarUnaryOp::Neg,
        ScalarUnaryOp::Conj,
        ScalarUnaryOp::Abs,
        ScalarUnaryOp::Reciprocal,
        ScalarUnaryOp::Real,
        ScalarUnaryOp::Imag,
        ScalarUnaryOp::Square,
    ] {
        assert!(
            <CpuBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
                ScalarPrimsDescriptor::PointwiseUnary { op }
            )
        );
    }

    for op in [
        ScalarBinaryOp::Add,
        ScalarBinaryOp::Sub,
        ScalarBinaryOp::Mul,
        ScalarBinaryOp::Div,
        ScalarBinaryOp::Maximum,
        ScalarBinaryOp::Minimum,
        ScalarBinaryOp::Greater,
        ScalarBinaryOp::GreaterEqual,
        ScalarBinaryOp::ClampMin,
        ScalarBinaryOp::ClampMax,
    ] {
        assert!(
            <CpuBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
                ScalarPrimsDescriptor::PointwiseBinary { op }
            )
        );
    }

    for op in [
        ScalarReductionOp::Sum,
        ScalarReductionOp::Prod,
        ScalarReductionOp::Mean,
        ScalarReductionOp::Max,
        ScalarReductionOp::Min,
    ] {
        let desc = ScalarPrimsDescriptor::Reduction {
            modes_a: vec![0, 1],
            modes_c: vec![0],
            op,
        };
        assert!(<CpuBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(desc));
    }
}

#[test]
fn analytic_prims_family_support_matches_phase1_inventory() {
    for op in [
        AnalyticUnaryOp::Sqrt,
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
        assert!(<CpuBackend as TensorAnalyticPrims<Standard<f64>>>::has_analytic_support(desc));
    }

    for op in [
        AnalyticBinaryOp::Pow,
        AnalyticBinaryOp::Atan2,
        AnalyticBinaryOp::Hypot,
        AnalyticBinaryOp::Xlogy,
    ] {
        let desc = AnalyticPrimsDescriptor::PointwiseBinary { op };
        assert!(<CpuBackend as TensorAnalyticPrims<Standard<f64>>>::has_analytic_support(desc));
    }

    for op in [AnalyticReductionOp::Var, AnalyticReductionOp::Std] {
        let desc = AnalyticPrimsDescriptor::Reduction {
            modes_a: vec![0, 1],
            modes_c: vec![0],
            op,
        };
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
    )
    .unwrap();
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

#[test]
fn plan_cache_roundtrips_across_family_descriptors() {
    let mut cache = crate::infra::plan_cache::PlanCache::new();

    let semiring_desc = SemiringCoreDescriptor::MakeContiguous;
    let fast_path_desc = SemiringFastPathDescriptor::ElementwiseBinary {
        op: SemiringBinaryOp::Mul,
    };
    let scalar_desc = ScalarPrimsDescriptor::PointwiseUnary {
        op: ScalarUnaryOp::Neg,
    };
    let analytic_desc = AnalyticPrimsDescriptor::PointwiseUnary {
        op: AnalyticUnaryOp::Exp,
    };

    cache.insert::<usize, _>(&semiring_desc, &[&[2, 2]], 11);
    cache.insert::<usize, _>(&fast_path_desc, &[&[2, 2], &[2, 2], &[2, 2]], 13);
    cache.insert::<usize, _>(&scalar_desc, &[&[4], &[4]], 17);
    cache.insert::<usize, _>(&analytic_desc, &[&[4], &[4]], 19);

    assert_eq!(cache.get::<usize, _>(&semiring_desc, &[&[2, 2]]), Some(11));
    assert_eq!(
        cache.get::<usize, _>(&fast_path_desc, &[&[2, 2], &[2, 2], &[2, 2]]),
        Some(13)
    );
    assert_eq!(cache.get::<usize, _>(&scalar_desc, &[&[4], &[4]]), Some(17));
    assert_eq!(
        cache.get::<usize, _>(&analytic_desc, &[&[4], &[4]]),
        Some(19)
    );

    assert_eq!(cache.get::<usize, _>(&semiring_desc, &[&[3, 3]]), None);
    assert_eq!(cache.get::<String, _>(&semiring_desc, &[&[2, 2]]), None);

    cache.clear();
    assert!(cache.is_empty());
}
