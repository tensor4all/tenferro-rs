use crate::semiring_op::SemiringOp;
use crate::semiring_op_kind::SemiringOpKind;
use crate::semiring_ops::SemiringOps;
use crate::std_tensor_op::StdTensorOp;
use computegraph::GraphOp;
use tenferro_algebra::Standard;
use tenferro_tensor::DotGeneralConfig;

#[test]
fn test_std_tensor_op_input_output_counts() {
    assert_eq!(StdTensorOp::Add.n_inputs(), 2);
    assert_eq!(StdTensorOp::Mul.n_inputs(), 2);
    assert_eq!(StdTensorOp::Neg.n_inputs(), 1);
    assert_eq!(StdTensorOp::Conj.n_inputs(), 1);
    assert_eq!(
        StdTensorOp::DotGeneral(DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        })
        .n_inputs(),
        2
    );
    assert_eq!(StdTensorOp::ReduceSum { axes: vec![0] }.n_inputs(), 1);
    assert_eq!(
        StdTensorOp::ExtractDiag {
            axis_a: 0,
            axis_b: 1
        }
        .n_inputs(),
        1
    );
    assert_eq!(
        StdTensorOp::EmbedDiag {
            axis_a: 0,
            axis_b: 1
        }
        .n_inputs(),
        1
    );

    assert_eq!(StdTensorOp::Add.n_outputs(), 1);
    assert_eq!(StdTensorOp::Neg.n_outputs(), 1);
    assert_eq!(
        StdTensorOp::EmbedDiag {
            axis_a: 0,
            axis_b: 1
        }
        .n_outputs(),
        1
    );
}

#[test]
fn test_semiring_op_kind_counts() {
    assert_eq!(SemiringOpKind::Add.n_inputs(), 2);
    assert_eq!(SemiringOpKind::Mul.n_inputs(), 2);
    assert_eq!(
        SemiringOpKind::DotGeneral(DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        })
        .n_inputs(),
        2
    );
    assert_eq!(SemiringOpKind::ReduceSum { axes: vec![0] }.n_inputs(), 1);
    assert_eq!(SemiringOpKind::Transpose { perm: vec![1, 0] }.n_inputs(), 1);
    assert_eq!(
        SemiringOpKind::ExtractDiag {
            axis_a: 0,
            axis_b: 1
        }
        .n_inputs(),
        1
    );
    assert_eq!(
        SemiringOpKind::EmbedDiag {
            axis_a: 0,
            axis_b: 1
        }
        .n_inputs(),
        1
    );
}

#[test]
fn test_semiring_op_uses_algebra_marker_type() {
    let add = SemiringOp::<Standard<f64>>::add_op();
    let gemm = SemiringOp::<Standard<f64>>::dot_general(DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    });

    assert_eq!(add.n_inputs(), 2);
    assert_eq!(add.n_outputs(), 1);
    assert_eq!(gemm.n_inputs(), 2);
    assert_eq!(gemm.n_outputs(), 1);
}
