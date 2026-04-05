use crate::semiring_op::{SemiringInputKey, SemiringOp};
use crate::semiring_op_kind::SemiringOpKind;
use crate::semiring_ops::SemiringOps;
use crate::std_tensor_op::StdTensorOp;
use chainrules_core::PrimitiveOp;
use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, OpMode, ValRef};
use computegraph::GraphOp;
use tenferro_algebra::Standard;
use tenferro_tensor::{CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig};

use crate::input_key::TensorInputKey;

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
            lhs_rank: 2,
            rhs_rank: 2,
        })
        .n_inputs(),
        2
    );
    assert_eq!(
        StdTensorOp::ReduceSum {
            axes: vec![0],
            input_shape: vec![2, 3],
        }
        .n_inputs(),
        1
    );
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
    assert_eq!(
        StdTensorOp::Gather(GatherConfig {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1],
        })
        .n_inputs(),
        2
    );
    assert_eq!(
        StdTensorOp::Scatter(ScatterConfig {
            update_window_dims: vec![],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        })
        .n_inputs(),
        3
    );
    assert_eq!(
        StdTensorOp::DynamicSlice {
            slice_sizes: vec![1]
        }
        .n_inputs(),
        2
    );
    assert_eq!(
        StdTensorOp::Pad(PadConfig {
            edge_padding_low: vec![1],
            edge_padding_high: vec![1],
            interior_padding: vec![0],
        })
        .n_inputs(),
        1
    );

    assert_eq!(StdTensorOp::Add.n_outputs(), 1);
    assert_eq!(StdTensorOp::Neg.n_outputs(), 1);
    assert_eq!(StdTensorOp::Compare(CompareDir::Eq).n_inputs(), 2);
    assert_eq!(StdTensorOp::Select.n_inputs(), 3);
    assert_eq!(StdTensorOp::Clamp.n_inputs(), 3);
    assert_eq!(StdTensorOp::Div.n_inputs(), 2);
    assert_eq!(StdTensorOp::Pow.n_inputs(), 2);
    assert_eq!(StdTensorOp::Abs.n_inputs(), 1);
    assert_eq!(StdTensorOp::Scale { factor: 0.5 }.n_inputs(), 1);
    assert_eq!(StdTensorOp::Exp.n_inputs(), 1);
    assert_eq!(StdTensorOp::Log1p.n_inputs(), 1);
    assert_eq!(
        StdTensorOp::EmbedDiag {
            axis_a: 0,
            axis_b: 1
        }
        .n_outputs(),
        1
    );
    assert_eq!(
        StdTensorOp::Gather(GatherConfig {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1],
        })
        .n_outputs(),
        1
    );
    assert_eq!(
        StdTensorOp::Scatter(ScatterConfig {
            update_window_dims: vec![],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        })
        .n_outputs(),
        1
    );
    assert_eq!(
        StdTensorOp::DynamicSlice {
            slice_sizes: vec![1]
        }
        .n_outputs(),
        1
    );
    assert_eq!(
        StdTensorOp::Pad(PadConfig {
            edge_padding_low: vec![1],
            edge_padding_high: vec![1],
            interior_padding: vec![0],
        })
        .n_outputs(),
        1
    );
    assert_eq!(StdTensorOp::Tril { k: -1 }.n_inputs(), 1);
    assert_eq!(StdTensorOp::Tril { k: -1 }.n_outputs(), 1);
    assert_eq!(StdTensorOp::Triu { k: 1 }.n_inputs(), 1);
    assert_eq!(StdTensorOp::Triu { k: 1 }.n_outputs(), 1);
}

#[test]
fn test_std_tensor_op_linalg_input_output_counts() {
    assert_eq!(StdTensorOp::Cholesky.n_inputs(), 1);
    assert_eq!(StdTensorOp::Cholesky.n_outputs(), 1);
    assert_eq!(StdTensorOp::Svd.n_inputs(), 1);
    assert_eq!(StdTensorOp::Svd.n_outputs(), 3);
    assert_eq!(StdTensorOp::Qr.n_inputs(), 1);
    assert_eq!(StdTensorOp::Qr.n_outputs(), 2);
    assert_eq!(StdTensorOp::Eigh.n_inputs(), 1);
    assert_eq!(StdTensorOp::Eigh.n_outputs(), 2);
    assert_eq!(StdTensorOp::Solve.n_inputs(), 2);
    assert_eq!(StdTensorOp::Solve.n_outputs(), 1);
    assert_eq!(
        StdTensorOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: false,
        }
        .n_inputs(),
        2
    );
    assert_eq!(
        StdTensorOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: false,
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
            lhs_rank: 2,
            rhs_rank: 2,
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
        lhs_rank: 2,
        rhs_rank: 2,
    });

    assert_eq!(add.n_inputs(), 2);
    assert_eq!(add.n_outputs(), 1);
    assert_eq!(gemm.n_inputs(), 2);
    assert_eq!(gemm.n_outputs(), 1);
}

#[test]
fn test_semiring_op_clone_eq_hash_depend_only_on_kind() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let lhs = SemiringOp::<Standard<f64>>::transpose_op(vec![1, 0]);
    let rhs = lhs.clone();

    let mut lhs_hasher = DefaultHasher::new();
    lhs.hash(&mut lhs_hasher);
    let mut rhs_hasher = DefaultHasher::new();
    rhs.hash(&mut rhs_hasher);

    assert_eq!(lhs, rhs);
    assert_eq!(lhs_hasher.finish(), rhs_hasher.finish());
    assert_eq!(
        format!("{lhs:?}"),
        "SemiringOp { kind: Transpose { perm: [1, 0] } }"
    );
}

#[test]
fn test_semiring_input_key_clone_eq_and_hash_are_stable() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let lhs = SemiringInputKey { id: 11 };
    let rhs = lhs.clone();

    let mut lhs_hasher = DefaultHasher::new();
    lhs.hash(&mut lhs_hasher);
    let mut rhs_hasher = DefaultHasher::new();
    rhs.hash(&mut rhs_hasher);

    assert_eq!(lhs, rhs);
    assert_eq!(lhs_hasher.finish(), rhs_hasher.finish());
    assert!(format!("{lhs:?}").contains("SemiringInputKey"));
}

#[test]
fn test_semiring_op_constructors_cover_all_supported_kinds() {
    assert_eq!(
        SemiringOp::<Standard<f64>>::reduce_sum(vec![0, 2], vec![2, 3, 4]).kind,
        SemiringOpKind::ReduceSum { axes: vec![0, 2] }
    );
    assert_eq!(
        SemiringOp::<Standard<f64>>::reshape(vec![3, 2], vec![2, 3]).kind,
        SemiringOpKind::Reshape { shape: vec![2, 3] }
    );
    assert_eq!(
        SemiringOp::<Standard<f64>>::broadcast_in_dim(vec![2, 3], vec![0]).kind,
        SemiringOpKind::BroadcastInDim {
            shape: vec![2, 3],
            dims: vec![0]
        }
    );
    assert_eq!(
        SemiringOp::<Standard<f64>>::extract_diag(0, 1).kind,
        SemiringOpKind::ExtractDiag {
            axis_a: 0,
            axis_b: 1
        }
    );
    assert_eq!(
        SemiringOp::<Standard<f64>>::embed_diag(0, 1).kind,
        SemiringOpKind::EmbedDiag {
            axis_a: 0,
            axis_b: 1
        }
    );
}

#[test]
fn test_std_tensor_op_linearize_add_delegates_to_ad_module() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let dx = builder.add_input(TensorInputKey::User { id: 1 });
    let dy = builder.add_input(TensorInputKey::User { id: 2 });

    let result = StdTensorOp::add().linearize(&mut builder, &[], &[], &[Some(dx), Some(dy)]);

    assert_eq!(result.len(), 1);
    assert!(result[0].is_some());
    let fragment = builder.build();
    assert_eq!(fragment.ops().len(), 1);
    assert_eq!(fragment.ops()[0].op, StdTensorOp::Add);
    assert_eq!(
        fragment.ops()[0].mode,
        OpMode::Linear {
            active_mask: vec![true, true],
        }
    );
}

#[test]
fn test_std_tensor_op_transpose_rule_add_fans_out_cotangent() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let ct = builder.add_input(TensorInputKey::User { id: 3 });
    let inputs = vec![
        ValRef::External(GlobalValKey::Input(TensorInputKey::User { id: 10 })),
        ValRef::External(GlobalValKey::Input(TensorInputKey::User { id: 11 })),
    ];

    let result =
        StdTensorOp::add().transpose_rule(&mut builder, &[Some(ct)], &inputs, &OpMode::Primal);

    assert_eq!(result, vec![Some(ct), Some(ct)]);
    let fragment = builder.build();
    assert!(fragment.ops().is_empty());
}
