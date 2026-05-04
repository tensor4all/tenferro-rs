use tenferro_tensor::{CompareDir, DType, DotGeneralConfig};

use crate::ad::support::{ad_rule_support, AdRuleSupport};
use crate::std_tensor_op::StdTensorOp;

#[test]
fn linalg_factorizations_are_marked_supported_via_linearize() {
    let ops = [
        StdTensorOp::Cholesky,
        StdTensorOp::Lu,
        StdTensorOp::FullPivLu,
        StdTensorOp::Svd { eps: 1.0e-12 },
        StdTensorOp::Qr,
        StdTensorOp::Eigh { eps: 1.0e-12 },
        StdTensorOp::Eig {
            input_dtype: DType::F64,
        },
    ];

    for op in ops {
        assert_eq!(ad_rule_support(&op), AdRuleSupport::SupportedViaLinearize);
    }
}

#[test]
fn direct_transpose_ops_are_marked_direct() {
    let ops = [
        StdTensorOp::Add,
        StdTensorOp::Maximum,
        StdTensorOp::Slice(tenferro_tensor::SliceConfig {
            starts: vec![0],
            limits: vec![2],
            strides: vec![1],
        }),
        StdTensorOp::DotGeneral {
            config: DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        },
        StdTensorOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: false,
        },
        StdTensorOp::FullPivLuSolve { transpose_a: false },
    ];

    for op in ops {
        assert_eq!(ad_rule_support(&op), AdRuleSupport::DirectTranspose);
    }
}

#[test]
fn no_tangent_ops_are_marked_explicitly() {
    let ops = [
        StdTensorOp::Constant {
            dtype: DType::F64,
            bytes: 1.0_f64.to_ne_bytes().to_vec(),
        },
        StdTensorOp::Compare(CompareDir::Lt),
        StdTensorOp::ShapeOf { axis: 0 },
    ];

    for op in ops {
        assert_eq!(ad_rule_support(&op), AdRuleSupport::NoTangent);
    }
}
