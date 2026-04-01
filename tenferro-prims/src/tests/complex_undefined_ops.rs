use num_complex::{Complex32, Complex64};
use tenferro_algebra::Standard;
use tenferro_device::Error;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{
    AnalyticBinaryOp, AnalyticPrimsDescriptor, AnalyticReductionOp, AnalyticUnaryOp, CpuBackend,
    CpuContext, ScalarBinaryOp, ScalarPrimsDescriptor, ScalarReductionOp, TensorAnalyticPrims,
    TensorScalarPrims,
};

type ComplexBackend = CpuBackend;

fn tensor_c64(data: &[Complex64], dims: &[usize]) -> Tensor<Complex64> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

fn tensor_c32(data: &[Complex32], dims: &[usize]) -> Tensor<Complex32> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn cpu_scalar_binary_rejects_maximum_for_complex() {
    let desc = ScalarPrimsDescriptor::PointwiseBinary {
        op: ScalarBinaryOp::Maximum,
    };
    assert!(!<ComplexBackend as TensorScalarPrims<
        Standard<Complex64>,
    >>::has_scalar_support(desc));
    let mut ctx = CpuContext::new(1);
    let err = <ComplexBackend as TensorScalarPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &ScalarPrimsDescriptor::PointwiseBinary {
            op: ScalarBinaryOp::Maximum,
        },
        &[&[2, 2], &[2, 2], &[2, 2]],
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("not supported")));
}

#[test]
fn cpu_scalar_binary_rejects_minimum_for_complex() {
    let desc = ScalarPrimsDescriptor::PointwiseBinary {
        op: ScalarBinaryOp::Minimum,
    };
    assert!(!<ComplexBackend as TensorScalarPrims<
        Standard<Complex64>,
    >>::has_scalar_support(desc));
    let mut ctx = CpuContext::new(1);
    let err = <ComplexBackend as TensorScalarPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &ScalarPrimsDescriptor::PointwiseBinary {
            op: ScalarBinaryOp::Minimum,
        },
        &[&[2, 2], &[2, 2], &[2, 2]],
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("not supported")));
}

#[test]
fn cpu_scalar_binary_rejects_greater_for_complex() {
    let desc = ScalarPrimsDescriptor::PointwiseBinary {
        op: ScalarBinaryOp::Greater,
    };
    assert!(!<ComplexBackend as TensorScalarPrims<
        Standard<Complex64>,
    >>::has_scalar_support(desc));
    let mut ctx = CpuContext::new(1);
    let err = <ComplexBackend as TensorScalarPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &ScalarPrimsDescriptor::PointwiseBinary {
            op: ScalarBinaryOp::Greater,
        },
        &[&[2, 2], &[2, 2], &[2, 2]],
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("not supported")));
}

#[test]
fn cpu_scalar_binary_rejects_greater_equal_for_complex() {
    let desc = ScalarPrimsDescriptor::PointwiseBinary {
        op: ScalarBinaryOp::GreaterEqual,
    };
    assert!(!<ComplexBackend as TensorScalarPrims<
        Standard<Complex64>,
    >>::has_scalar_support(desc));
    let mut ctx = CpuContext::new(1);
    let err = <ComplexBackend as TensorScalarPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &ScalarPrimsDescriptor::PointwiseBinary {
            op: ScalarBinaryOp::GreaterEqual,
        },
        &[&[2, 2], &[2, 2], &[2, 2]],
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("not supported")));
}

#[test]
fn cpu_scalar_binary_rejects_clamp_min_for_complex() {
    let desc = ScalarPrimsDescriptor::PointwiseBinary {
        op: ScalarBinaryOp::ClampMin,
    };
    assert!(!<ComplexBackend as TensorScalarPrims<
        Standard<Complex64>,
    >>::has_scalar_support(desc));
    let mut ctx = CpuContext::new(1);
    let err = <ComplexBackend as TensorScalarPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &ScalarPrimsDescriptor::PointwiseBinary {
            op: ScalarBinaryOp::ClampMin,
        },
        &[&[2, 2], &[2, 2], &[2, 2]],
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("not supported")));
}

#[test]
fn cpu_scalar_binary_rejects_clamp_max_for_complex() {
    let desc = ScalarPrimsDescriptor::PointwiseBinary {
        op: ScalarBinaryOp::ClampMax,
    };
    assert!(!<ComplexBackend as TensorScalarPrims<
        Standard<Complex64>,
    >>::has_scalar_support(desc));
    let mut ctx = CpuContext::new(1);
    let err = <ComplexBackend as TensorScalarPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &ScalarPrimsDescriptor::PointwiseBinary {
            op: ScalarBinaryOp::ClampMax,
        },
        &[&[2, 2], &[2, 2], &[2, 2]],
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("not supported")));
}

#[test]
fn cpu_scalar_reduction_rejects_max_for_complex() {
    let desc = ScalarPrimsDescriptor::Reduction {
        modes_a: vec![0, 1],
        modes_c: vec![1],
        op: ScalarReductionOp::Max,
    };
    assert!(!<ComplexBackend as TensorScalarPrims<
        Standard<Complex64>,
    >>::has_scalar_support(desc));
    let mut ctx = CpuContext::new(1);
    let err = <ComplexBackend as TensorScalarPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &ScalarPrimsDescriptor::Reduction {
            modes_a: vec![0, 1],
            modes_c: vec![1],
            op: ScalarReductionOp::Max,
        },
        &[&[2, 2], &[2]],
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("not supported")));
}

#[test]
fn cpu_scalar_reduction_rejects_min_for_complex() {
    let desc = ScalarPrimsDescriptor::Reduction {
        modes_a: vec![0, 1],
        modes_c: vec![1],
        op: ScalarReductionOp::Min,
    };
    assert!(!<ComplexBackend as TensorScalarPrims<
        Standard<Complex64>,
    >>::has_scalar_support(desc));
    let mut ctx = CpuContext::new(1);
    let err = <ComplexBackend as TensorScalarPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &ScalarPrimsDescriptor::Reduction {
            modes_a: vec![0, 1],
            modes_c: vec![1],
            op: ScalarReductionOp::Min,
        },
        &[&[2, 2], &[2]],
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("not supported")));
}

#[test]
fn cpu_scalar_binary_complex_execute_rejects_ordering_ops() {
    let mut ctx = CpuContext::new(1);
    let lhs = tensor_c64(&[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)], &[2]);
    let rhs = tensor_c64(&[Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)], &[2]);
    let mut output = Tensor::<Complex64>::zeros(
        &[2],
        tenferro_device::LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    for op in [
        ScalarBinaryOp::Maximum,
        ScalarBinaryOp::Minimum,
        ScalarBinaryOp::Greater,
        ScalarBinaryOp::GreaterEqual,
        ScalarBinaryOp::ClampMin,
        ScalarBinaryOp::ClampMax,
    ] {
        let desc = ScalarPrimsDescriptor::PointwiseBinary { op };
        let plan_result = <ComplexBackend as TensorScalarPrims<Standard<Complex64>>>::plan(
            &mut ctx,
            &desc,
            &[&[2], &[2], &[2]],
        );
        assert!(
            plan_result.is_err(),
            "{op:?} should be rejected for Complex64 at plan time"
        );
    }

    let add_desc = ScalarPrimsDescriptor::PointwiseBinary {
        op: ScalarBinaryOp::Add,
    };
    let add_plan = <ComplexBackend as TensorScalarPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &add_desc,
        &[&[2], &[2], &[2]],
    )
    .unwrap();
    <ComplexBackend as TensorScalarPrims<Standard<Complex64>>>::execute(
        &mut ctx,
        &add_plan,
        Complex64::new(1.0, 0.0),
        &[&lhs, &rhs],
        Complex64::new(0.0, 0.0),
        &mut output,
    )
    .unwrap();
    assert_eq!(
        output.buffer().as_slice().unwrap(),
        &[Complex64::new(4.0, 0.0), Complex64::new(6.0, 0.0)]
    );
}

#[test]
fn cpu_scalar_reduction_complex_execute_rejects_extrema() {
    let mut ctx = CpuContext::new(1);
    let input = tensor_c64(
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
        &[2, 2],
    );
    let mut output = Tensor::<Complex64>::zeros(
        &[2],
        tenferro_device::LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    for op in [ScalarReductionOp::Max, ScalarReductionOp::Min] {
        let desc = ScalarPrimsDescriptor::Reduction {
            modes_a: vec![0, 1],
            modes_c: vec![1],
            op,
        };
        let plan_result = <ComplexBackend as TensorScalarPrims<Standard<Complex64>>>::plan(
            &mut ctx,
            &desc,
            &[&[2, 2], &[2]],
        );
        assert!(
            plan_result.is_err(),
            "{op:?} should be rejected for Complex64 at plan time"
        );
    }

    let sum_desc = ScalarPrimsDescriptor::Reduction {
        modes_a: vec![0, 1],
        modes_c: vec![1],
        op: ScalarReductionOp::Sum,
    };
    let sum_plan = <ComplexBackend as TensorScalarPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &sum_desc,
        &[&[2, 2], &[2]],
    )
    .unwrap();
    <ComplexBackend as TensorScalarPrims<Standard<Complex64>>>::execute(
        &mut ctx,
        &sum_plan,
        Complex64::new(1.0, 0.0),
        &[&input],
        Complex64::new(0.0, 0.0),
        &mut output,
    )
    .unwrap();
    assert_eq!(
        output.buffer().as_slice().unwrap(),
        &[Complex64::new(3.0, 0.0), Complex64::new(7.0, 0.0)]
    );
}

#[test]
fn cpu_analytic_binary_rejects_atan2_for_complex() {
    let desc = AnalyticPrimsDescriptor::PointwiseBinary {
        op: AnalyticBinaryOp::Atan2,
    };
    assert!(!<ComplexBackend as TensorAnalyticPrims<
        Standard<Complex64>,
    >>::has_analytic_support(desc));
    let mut ctx = CpuContext::new(1);
    let err = <ComplexBackend as TensorAnalyticPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &AnalyticPrimsDescriptor::PointwiseBinary {
            op: AnalyticBinaryOp::Atan2,
        },
        &[&[2, 2], &[2, 2], &[2, 2]],
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("not supported")));
}

#[test]
fn cpu_analytic_binary_rejects_hypot_for_complex() {
    let desc = AnalyticPrimsDescriptor::PointwiseBinary {
        op: AnalyticBinaryOp::Hypot,
    };
    assert!(!<ComplexBackend as TensorAnalyticPrims<
        Standard<Complex64>,
    >>::has_analytic_support(desc));
    let mut ctx = CpuContext::new(1);
    let err = <ComplexBackend as TensorAnalyticPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &AnalyticPrimsDescriptor::PointwiseBinary {
            op: AnalyticBinaryOp::Hypot,
        },
        &[&[2, 2], &[2, 2], &[2, 2]],
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("not supported")));
}

#[test]
fn cpu_analytic_reduction_rejects_var_for_complex() {
    let desc = AnalyticPrimsDescriptor::Reduction {
        modes_a: vec![0, 1],
        modes_c: vec![1],
        op: AnalyticReductionOp::Var,
    };
    assert!(!<ComplexBackend as TensorAnalyticPrims<
        Standard<Complex64>,
    >>::has_analytic_support(desc));
    let mut ctx = CpuContext::new(1);
    let err = <ComplexBackend as TensorAnalyticPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &AnalyticPrimsDescriptor::Reduction {
            modes_a: vec![0, 1],
            modes_c: vec![1],
            op: AnalyticReductionOp::Var,
        },
        &[&[2, 2], &[2]],
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("not supported")));
}

#[test]
fn cpu_analytic_reduction_rejects_std_for_complex() {
    let desc = AnalyticPrimsDescriptor::Reduction {
        modes_a: vec![0, 1],
        modes_c: vec![1],
        op: AnalyticReductionOp::Std,
    };
    assert!(!<ComplexBackend as TensorAnalyticPrims<
        Standard<Complex64>,
    >>::has_analytic_support(desc));
    let mut ctx = CpuContext::new(1);
    let err = <ComplexBackend as TensorAnalyticPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &AnalyticPrimsDescriptor::Reduction {
            modes_a: vec![0, 1],
            modes_c: vec![1],
            op: AnalyticReductionOp::Std,
        },
        &[&[2, 2], &[2]],
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("not supported")));
}

#[test]
fn cpu_analytic_unary_rejects_ceil_for_complex() {
    let desc = AnalyticPrimsDescriptor::PointwiseUnary {
        op: AnalyticUnaryOp::Ceil,
    };
    assert!(!<ComplexBackend as TensorAnalyticPrims<
        Standard<Complex64>,
    >>::has_analytic_support(desc));
    let mut ctx = CpuContext::new(1);
    let err = <ComplexBackend as TensorAnalyticPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &AnalyticPrimsDescriptor::PointwiseUnary {
            op: AnalyticUnaryOp::Ceil,
        },
        &[&[2], &[2]],
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("not supported")));
}

#[test]
fn cpu_analytic_binary_complex_execute_rejects_atan2_and_hypot() {
    let mut ctx = CpuContext::new(1);
    let lhs = tensor_c64(&[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)], &[2]);
    let rhs = tensor_c64(&[Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)], &[2]);
    let mut output = Tensor::<Complex64>::zeros(
        &[2],
        tenferro_device::LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    for op in [AnalyticBinaryOp::Atan2, AnalyticBinaryOp::Hypot] {
        let desc = AnalyticPrimsDescriptor::PointwiseBinary { op };
        let plan_result = <ComplexBackend as TensorAnalyticPrims<Standard<Complex64>>>::plan(
            &mut ctx,
            &desc,
            &[&[2], &[2], &[2]],
        );
        assert!(
            plan_result.is_err(),
            "{op:?} should be rejected for Complex64 at plan time"
        );
    }

    let pow_desc = AnalyticPrimsDescriptor::PointwiseBinary {
        op: AnalyticBinaryOp::Pow,
    };
    let pow_plan = <ComplexBackend as TensorAnalyticPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &pow_desc,
        &[&[2], &[2], &[2]],
    )
    .unwrap();
    <ComplexBackend as TensorAnalyticPrims<Standard<Complex64>>>::execute(
        &mut ctx,
        &pow_plan,
        Complex64::new(1.0, 0.0),
        &[&lhs, &rhs],
        Complex64::new(0.0, 0.0),
        &mut output,
    )
    .unwrap();
    assert_eq!(
        output.buffer().as_slice().unwrap(),
        &[
            Complex64::new(1.0, 0.0).powc(Complex64::new(3.0, 0.0)),
            Complex64::new(2.0, 0.0).powc(Complex64::new(4.0, 0.0))
        ]
    );
}

#[test]
fn cpu_analytic_reduction_complex_execute_rejects_var_and_std() {
    let mut ctx = CpuContext::new(1);
    let _input = tensor_c64(
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
        &[2, 2],
    );

    for op in [AnalyticReductionOp::Var, AnalyticReductionOp::Std] {
        let desc = AnalyticPrimsDescriptor::Reduction {
            modes_a: vec![0, 1],
            modes_c: vec![1],
            op,
        };
        let plan_result = <ComplexBackend as TensorAnalyticPrims<Standard<Complex64>>>::plan(
            &mut ctx,
            &desc,
            &[&[2, 2], &[2]],
        );
        assert!(
            plan_result.is_err(),
            "{op:?} should be rejected for Complex64 at plan time"
        );
    }
}

#[test]
fn cpu_complex_ops_allowed_for_add_sub_mul_div_pow() {
    let mut ctx = CpuContext::new(1);
    let lhs = tensor_c64(&[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)], &[2]);
    let rhs = tensor_c64(&[Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)], &[2]);
    let mut output = Tensor::<Complex64>::zeros(
        &[2],
        tenferro_device::LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    for op in [
        ScalarBinaryOp::Add,
        ScalarBinaryOp::Sub,
        ScalarBinaryOp::Mul,
        ScalarBinaryOp::Div,
    ] {
        assert!(
            <ComplexBackend as TensorScalarPrims<Standard<Complex64>>>::has_scalar_support(
                ScalarPrimsDescriptor::PointwiseBinary { op }
            )
        );
        let plan = <ComplexBackend as TensorScalarPrims<Standard<Complex64>>>::plan(
            &mut ctx,
            &ScalarPrimsDescriptor::PointwiseBinary { op },
            &[&[2], &[2], &[2]],
        )
        .unwrap();
        <ComplexBackend as TensorScalarPrims<Standard<Complex64>>>::execute(
            &mut ctx,
            &plan,
            Complex64::new(1.0, 0.0),
            &[&lhs, &rhs],
            Complex64::new(0.0, 0.0),
            &mut output,
        )
        .unwrap();
    }

    for op in [AnalyticBinaryOp::Pow, AnalyticBinaryOp::Xlogy] {
        let desc = AnalyticPrimsDescriptor::PointwiseBinary { op };
        assert!(<ComplexBackend as TensorAnalyticPrims<
            Standard<Complex64>,
        >>::has_analytic_support(desc));
    }

    for op in [
        ScalarReductionOp::Sum,
        ScalarReductionOp::Prod,
        ScalarReductionOp::Mean,
    ] {
        let desc = ScalarPrimsDescriptor::Reduction {
            modes_a: vec![0],
            modes_c: vec![],
            op,
        };
        assert!(
            <ComplexBackend as TensorScalarPrims<Standard<Complex64>>>::has_scalar_support(desc)
        );
    }
}

#[test]
fn cpu_scalar_binary_rejects_ordering_ops_for_complex32() {
    for op in [
        ScalarBinaryOp::Maximum,
        ScalarBinaryOp::Minimum,
        ScalarBinaryOp::Greater,
        ScalarBinaryOp::GreaterEqual,
        ScalarBinaryOp::ClampMin,
        ScalarBinaryOp::ClampMax,
    ] {
        assert!(
            !<ComplexBackend as TensorScalarPrims<Standard<Complex32>>>::has_scalar_support(
                ScalarPrimsDescriptor::PointwiseBinary { op }
            ),
            "{op:?} should report no scalar support for Complex32"
        );
        let mut ctx = CpuContext::new(1);
        let desc = ScalarPrimsDescriptor::PointwiseBinary { op };
        let err = <ComplexBackend as TensorScalarPrims<Standard<Complex32>>>::plan(
            &mut ctx,
            &desc,
            &[&[2], &[2], &[2]],
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::InvalidArgument(ref msg) if msg.contains("not supported")),
            "{op:?}: unexpected error variant: {err:?}"
        );
    }
}

#[test]
fn cpu_scalar_reduction_rejects_extrema_for_complex32() {
    for op in [ScalarReductionOp::Max, ScalarReductionOp::Min] {
        assert!(
            !<ComplexBackend as TensorScalarPrims<Standard<Complex32>>>::has_scalar_support(
                ScalarPrimsDescriptor::Reduction {
                    modes_a: vec![0, 1],
                    modes_c: vec![1],
                    op,
                }
            ),
            "{op:?} should report no scalar support for Complex32"
        );
        let mut ctx = CpuContext::new(1);
        let desc = ScalarPrimsDescriptor::Reduction {
            modes_a: vec![0, 1],
            modes_c: vec![1],
            op,
        };
        let err = <ComplexBackend as TensorScalarPrims<Standard<Complex32>>>::plan(
            &mut ctx,
            &desc,
            &[&[2, 2], &[2]],
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::InvalidArgument(ref msg) if msg.contains("not supported")),
            "{op:?}: unexpected error variant: {err:?}"
        );
    }
}

#[test]
fn cpu_analytic_binary_rejects_atan2_and_hypot_for_complex32() {
    for op in [AnalyticBinaryOp::Atan2, AnalyticBinaryOp::Hypot] {
        assert!(
            !<ComplexBackend as TensorAnalyticPrims<Standard<Complex32>>>::has_analytic_support(
                AnalyticPrimsDescriptor::PointwiseBinary { op }
            ),
            "{op:?} should report no analytic support for Complex32"
        );
        let mut ctx = CpuContext::new(1);
        let desc = AnalyticPrimsDescriptor::PointwiseBinary { op };
        let err = <ComplexBackend as TensorAnalyticPrims<Standard<Complex32>>>::plan(
            &mut ctx,
            &desc,
            &[&[2], &[2], &[2]],
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::InvalidArgument(ref msg) if msg.contains("not supported")),
            "{op:?}: unexpected error variant: {err:?}"
        );
    }
}

#[test]
fn cpu_analytic_reduction_rejects_var_and_std_for_complex32() {
    for op in [AnalyticReductionOp::Var, AnalyticReductionOp::Std] {
        assert!(
            !<ComplexBackend as TensorAnalyticPrims<Standard<Complex32>>>::has_analytic_support(
                AnalyticPrimsDescriptor::Reduction {
                    modes_a: vec![0, 1],
                    modes_c: vec![1],
                    op,
                }
            ),
            "{op:?} should report no analytic support for Complex32"
        );
        let mut ctx = CpuContext::new(1);
        let desc = AnalyticPrimsDescriptor::Reduction {
            modes_a: vec![0, 1],
            modes_c: vec![1],
            op,
        };
        let err = <ComplexBackend as TensorAnalyticPrims<Standard<Complex32>>>::plan(
            &mut ctx,
            &desc,
            &[&[2, 2], &[2]],
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::InvalidArgument(ref msg) if msg.contains("not supported")),
            "{op:?}: unexpected error variant: {err:?}"
        );
    }
}

#[test]
fn cpu_analytic_unary_rejects_ceil_for_complex32() {
    assert!(!<ComplexBackend as TensorAnalyticPrims<
        Standard<Complex32>,
    >>::has_analytic_support(
        AnalyticPrimsDescriptor::PointwiseUnary {
            op: AnalyticUnaryOp::Ceil,
        }
    ));
    let mut ctx = CpuContext::new(1);
    let desc = AnalyticPrimsDescriptor::PointwiseUnary {
        op: AnalyticUnaryOp::Ceil,
    };
    let err = <ComplexBackend as TensorAnalyticPrims<Standard<Complex32>>>::plan(
        &mut ctx,
        &desc,
        &[&[2], &[2]],
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("not supported")));
}

#[test]
fn cpu_complex32_ops_allowed_for_add_sub_mul_div_sum_prod_mean() {
    let mut ctx = CpuContext::new(1);
    let lhs = tensor_c32(&[Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)], &[2]);
    let rhs = tensor_c32(&[Complex32::new(5.0, 6.0), Complex32::new(7.0, 8.0)], &[2]);
    let mut output = Tensor::<Complex32>::zeros(
        &[2],
        tenferro_device::LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    for op in [
        ScalarBinaryOp::Add,
        ScalarBinaryOp::Sub,
        ScalarBinaryOp::Mul,
        ScalarBinaryOp::Div,
    ] {
        assert!(
            <ComplexBackend as TensorScalarPrims<Standard<Complex32>>>::has_scalar_support(
                ScalarPrimsDescriptor::PointwiseBinary { op }
            )
        );
        let desc = ScalarPrimsDescriptor::PointwiseBinary { op };
        let plan = <ComplexBackend as TensorScalarPrims<Standard<Complex32>>>::plan(
            &mut ctx,
            &desc,
            &[&[2], &[2], &[2]],
        )
        .unwrap();
        <ComplexBackend as TensorScalarPrims<Standard<Complex32>>>::execute(
            &mut ctx,
            &plan,
            Complex32::new(1.0, 0.0),
            &[&lhs, &rhs],
            Complex32::new(0.0, 0.0),
            &mut output,
        )
        .unwrap();
    }

    for op in [AnalyticBinaryOp::Pow, AnalyticBinaryOp::Xlogy] {
        assert!(<ComplexBackend as TensorAnalyticPrims<
            Standard<Complex32>,
        >>::has_analytic_support(
            AnalyticPrimsDescriptor::PointwiseBinary { op }
        ));
    }

    for op in [
        ScalarReductionOp::Sum,
        ScalarReductionOp::Prod,
        ScalarReductionOp::Mean,
    ] {
        assert!(
            <ComplexBackend as TensorScalarPrims<Standard<Complex32>>>::has_scalar_support(
                ScalarPrimsDescriptor::Reduction {
                    modes_a: vec![0],
                    modes_c: vec![],
                    op,
                }
            )
        );
    }
}
