use crate::ad::context::ShapeGuardContext;
use crate::dim_expr::DimExpr;
use crate::semiring_op::{SemiringInputKey, SemiringOp};
use crate::semiring_op_kind::SemiringOpKind;
use crate::semiring_ops::SemiringOps;
use crate::std_tensor_op::StdTensorOp;
use crate::{SymDim, TensorMeta};
use chainrules_core::PrimitiveOp;
use computegraph::fragment::{Fragment, FragmentBuilder};
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::GraphOp;
use num_complex::{Complex32, Complex64};
use tenferro_algebra::Standard;
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig,
};

use crate::input_key::TensorInputKey;

macro_rules! shape {
    ($($dim:expr),* $(,)?) => {
        DimExpr::from_concrete(&[$($dim),*])
    };
}

fn tensor_input_key(id: u64) -> TensorInputKey {
    TensorInputKey::User { id }
}

fn add_input_keys(
    builder: &mut FragmentBuilder<StdTensorOp>,
    start_id: u64,
    count: usize,
) -> Vec<GlobalValKey<StdTensorOp>> {
    (0..count)
        .map(|offset| {
            let local = builder.add_input(tensor_input_key(start_id + offset as u64));
            builder.global_key(local).clone()
        })
        .collect()
}

fn external_inputs(start_id: u64, count: usize) -> Vec<ValRef<StdTensorOp>> {
    (0..count)
        .map(|offset| {
            ValRef::External(GlobalValKey::Input(tensor_input_key(
                start_id + offset as u64,
            )))
        })
        .collect()
}

fn linear_mode(active_mask: &[bool]) -> OpMode {
    OpMode::Linear {
        active_mask: active_mask.to_vec(),
    }
}

fn seed_dot_general_input_metadata(
    ctx: &mut ShapeGuardContext,
    keys: &[GlobalValKey<StdTensorOp>],
) {
    let shapes = [
        vec![SymDim::from(2usize), SymDim::from(3usize)],
        vec![SymDim::from(3usize), SymDim::from(4usize)],
    ];
    for (key, shape) in keys.iter().zip(shapes) {
        ctx.insert_metadata(
            key.clone(),
            TensorMeta {
                dtype: DType::F64,
                shape,
            },
        );
    }
}

fn seed_dot_general_ref_metadata(ctx: &mut ShapeGuardContext, inputs: &[ValRef<StdTensorOp>]) {
    let keys: Vec<_> = inputs
        .iter()
        .map(|input| match input {
            ValRef::External(key) => key.clone(),
            ValRef::Local(local_id) => {
                panic!("expected external input in test helper, got local {local_id}")
            }
        })
        .collect();
    seed_dot_general_input_metadata(ctx, &keys);
}

fn run_linearize_case(
    op: StdTensorOp,
    n_primal_in: usize,
    n_primal_out: usize,
    tangent_mask: &[bool],
) -> (Vec<Option<LocalValId>>, Fragment<StdTensorOp>) {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let primal_in = add_input_keys(&mut builder, 100, n_primal_in);
    let primal_out = add_input_keys(&mut builder, 200, n_primal_out);
    if matches!(&op, StdTensorOp::DotGeneral { .. }) {
        seed_dot_general_input_metadata(&mut ad_ctx, &primal_in);
    }
    let tangent_in: Vec<Option<LocalValId>> = tangent_mask
        .iter()
        .enumerate()
        .map(|(offset, &active)| {
            active.then(|| builder.add_input(tensor_input_key(300 + offset as u64)))
        })
        .collect();
    let result = op.linearize(
        &mut builder,
        &primal_in,
        &primal_out,
        &tangent_in,
        &mut ad_ctx,
    );
    (result, builder.build())
}

fn run_transpose_case(
    op: StdTensorOp,
    n_inputs: usize,
    active_mask: &[bool],
    cotangent_present: bool,
) -> (
    Vec<Option<LocalValId>>,
    Option<LocalValId>,
    Fragment<StdTensorOp>,
) {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let cotangent = cotangent_present.then(|| builder.add_input(tensor_input_key(400)));
    let inputs = external_inputs(500, n_inputs);
    if matches!(&op, StdTensorOp::DotGeneral { .. }) {
        seed_dot_general_ref_metadata(&mut ad_ctx, &inputs);
    }
    let result = op.transpose_rule(
        &mut builder,
        &[cotangent],
        &inputs,
        &linear_mode(active_mask),
        &mut ad_ctx,
    );
    (result, cotangent, builder.build())
}

#[test]
fn test_std_tensor_op_input_output_counts() {
    assert_eq!(StdTensorOp::Add.n_inputs(), 2);
    assert_eq!(StdTensorOp::Mul.n_inputs(), 2);
    assert_eq!(StdTensorOp::Neg.n_inputs(), 1);
    assert_eq!(StdTensorOp::Conj.n_inputs(), 1);
    assert_eq!(
        StdTensorOp::DotGeneral {
            config: DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        }
        .n_inputs(),
        2
    );
    assert_eq!(
        StdTensorOp::ReduceSum {
            axes: vec![0],
            input_shape: shape![2, 3],
        }
        .n_inputs(),
        1
    );
    assert_eq!(StdTensorOp::constant_f64(1.0).n_inputs(), 0);
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
        StdTensorOp::NaryEinsum {
            subscripts: "ij,jk,kl->il".into(),
            n_inputs: 3,
        }
        .n_inputs(),
        3
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
    assert_eq!(StdTensorOp::Exp.n_inputs(), 1);
    assert_eq!(StdTensorOp::Log1p.n_inputs(), 1);
    assert_eq!(StdTensorOp::constant_f64(1.0).n_outputs(), 1);
    assert_eq!(
        StdTensorOp::NaryEinsum {
            subscripts: "ij,jk->ik".into(),
            n_inputs: 2,
        }
        .n_outputs(),
        1
    );
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
    assert_eq!(StdTensorOp::Svd { eps: 1.0e-12 }.n_inputs(), 1);
    assert_eq!(StdTensorOp::Svd { eps: 1.0e-12 }.n_outputs(), 3);
    assert_eq!(StdTensorOp::Qr.n_inputs(), 1);
    assert_eq!(StdTensorOp::Qr.n_outputs(), 2);
    assert_eq!(StdTensorOp::Eigh { eps: 1.0e-12 }.n_inputs(), 1);
    assert_eq!(StdTensorOp::Eigh { eps: 1.0e-12 }.n_outputs(), 2);
    assert_eq!(StdTensorOp::Lu.n_inputs(), 1);
    assert_eq!(StdTensorOp::Lu.n_outputs(), 4);
    assert_eq!(
        StdTensorOp::Eig {
            input_dtype: DType::F64,
        }
        .n_inputs(),
        1
    );
    assert_eq!(
        StdTensorOp::Eig {
            input_dtype: DType::F64,
        }
        .n_outputs(),
        2
    );
    assert_eq!(
        StdTensorOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: false,
            lhs_shape: shape![2, 2],
            rhs_shape: shape![2, 1],
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
            lhs_shape: shape![2, 2],
            rhs_shape: shape![2, 1],
        }
        .n_outputs(),
        1
    );
}

#[test]
fn test_std_tensor_op_remaining_input_output_counts() {
    let slice = tenferro_tensor::SliceConfig {
        starts: vec![0, 1],
        limits: vec![2, 3],
        strides: vec![1, 1],
    };

    assert_eq!(StdTensorOp::Maximum.n_inputs(), 2);
    assert_eq!(StdTensorOp::Maximum.n_outputs(), 1);
    assert_eq!(StdTensorOp::Minimum.n_inputs(), 2);
    assert_eq!(StdTensorOp::Minimum.n_outputs(), 1);
    assert_eq!(
        StdTensorOp::Convert {
            from: DType::F64,
            to: DType::C64,
        }
        .n_inputs(),
        1
    );
    assert_eq!(
        StdTensorOp::Convert {
            from: DType::F64,
            to: DType::C64,
        }
        .n_outputs(),
        1
    );
    assert_eq!(StdTensorOp::Slice(slice.clone()).n_inputs(), 1);
    assert_eq!(StdTensorOp::Slice(slice).n_outputs(), 1);
    assert_eq!(StdTensorOp::Reverse { axes: vec![0, 2] }.n_inputs(), 1);
    assert_eq!(StdTensorOp::Reverse { axes: vec![0, 2] }.n_outputs(), 1);
    assert_eq!(StdTensorOp::ShapeOf { axis: 1 }.n_inputs(), 1);
    assert_eq!(StdTensorOp::ShapeOf { axis: 1 }.n_outputs(), 1);
    assert_eq!(StdTensorOp::DynamicTruncate { axis: 0 }.n_inputs(), 2);
    assert_eq!(StdTensorOp::DynamicTruncate { axis: 0 }.n_outputs(), 1);
    assert_eq!(StdTensorOp::PadToMatch { axis: 0 }.n_inputs(), 2);
    assert_eq!(StdTensorOp::PadToMatch { axis: 0 }.n_outputs(), 1);
}

#[test]
#[should_panic(expected = "n_inputs not yet implemented for variable-arity op")]
fn test_std_tensor_op_n_inputs_panics_for_concatenate() {
    let _ = StdTensorOp::Concatenate { axis: 0 }.n_inputs();
}

#[test]
#[should_panic(expected = "n_outputs not yet implemented for variable-arity op")]
fn test_std_tensor_op_n_outputs_panics_for_concatenate() {
    let _ = StdTensorOp::Concatenate { axis: 0 }.n_outputs();
}

#[test]
fn test_std_tensor_op_reduction_counts_cover_remaining_variants() {
    assert_eq!(
        StdTensorOp::ReduceProd {
            axes: vec![1],
            input_shape: shape![2, 3]
        }
        .n_inputs(),
        1
    );
    assert_eq!(
        StdTensorOp::ReduceProd {
            axes: vec![1],
            input_shape: shape![2, 3]
        }
        .n_outputs(),
        1
    );
    assert_eq!(
        StdTensorOp::ReduceMax {
            axes: vec![0],
            input_shape: shape![2, 3]
        }
        .n_inputs(),
        1
    );
    assert_eq!(
        StdTensorOp::ReduceMax {
            axes: vec![0],
            input_shape: shape![2, 3]
        }
        .n_outputs(),
        1
    );
    assert_eq!(
        StdTensorOp::ReduceMin {
            axes: vec![0, 1],
            input_shape: shape![2, 3]
        }
        .n_inputs(),
        1
    );
    assert_eq!(
        StdTensorOp::ReduceMin {
            axes: vec![0, 1],
            input_shape: shape![2, 3]
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
    let gemm = SemiringOp::<Standard<f64>>::dot_general(
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
        2,
        2,
    );

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
        SemiringOp::<Standard<f64>>::add_op().kind,
        SemiringOpKind::Add
    );
    assert_eq!(
        SemiringOp::<Standard<f64>>::mul_op().kind,
        SemiringOpKind::Mul
    );
    assert_eq!(
        SemiringOp::<Standard<f64>>::transpose_op(vec![1, 0]).kind,
        SemiringOpKind::Transpose { perm: vec![1, 0] }
    );
    assert_eq!(
        SemiringOp::<Standard<f64>>::reduce_sum(vec![0, 2], shape![2, 3, 4]).kind,
        SemiringOpKind::ReduceSum { axes: vec![0, 2] }
    );
    assert_eq!(
        SemiringOp::<Standard<f64>>::reshape(shape![3, 2], shape![2, 3]).kind,
        SemiringOpKind::Reshape { shape: vec![2, 3] }
    );
    assert_eq!(
        SemiringOp::<Standard<f64>>::broadcast_in_dim(shape![2, 3], vec![0]).kind,
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

    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    assert_eq!(
        SemiringOp::<Standard<f64>>::dot_general(config.clone(), 2, 2).kind,
        SemiringOpKind::DotGeneral(config)
    );
}

#[test]
fn test_std_tensor_op_semiring_ops_impl_constructors_cover_remaining_variants() {
    assert_eq!(<StdTensorOp as SemiringOps>::add_op(), StdTensorOp::Add);
    assert_eq!(
        <StdTensorOp as SemiringOps>::reshape(shape![6], shape![2, 3]),
        StdTensorOp::Reshape {
            from_shape: shape![6],
            to_shape: shape![2, 3],
        }
    );
}

#[test]
fn test_std_tensor_op_linearize_add_delegates_to_ad_module() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let dx = builder.add_input(TensorInputKey::User { id: 1 });
    let dy = builder.add_input(TensorInputKey::User { id: 2 });

    let result =
        StdTensorOp::add().linearize(&mut builder, &[], &[], &[Some(dx), Some(dy)], &mut ad_ctx);

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
    let mut ad_ctx = ShapeGuardContext::default();
    let ct = builder.add_input(TensorInputKey::User { id: 3 });
    let inputs = vec![
        ValRef::External(GlobalValKey::Input(TensorInputKey::User { id: 10 })),
        ValRef::External(GlobalValKey::Input(TensorInputKey::User { id: 11 })),
    ];

    let result = StdTensorOp::add().transpose_rule(
        &mut builder,
        &[Some(ct)],
        &inputs,
        &OpMode::Primal,
        &mut ad_ctx,
    );

    assert_eq!(result, vec![Some(ct), Some(ct)]);
    let fragment = builder.build();
    assert!(fragment.ops().is_empty());
}

#[test]
fn test_std_tensor_op_hash_covers_remaining_variants() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let variants = [
        StdTensorOp::Compare(CompareDir::Ge),
        StdTensorOp::Tril { k: -2 },
        StdTensorOp::Triu { k: 3 },
        StdTensorOp::Gather(GatherConfig {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1],
        }),
        StdTensorOp::Scatter(ScatterConfig {
            update_window_dims: vec![],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        }),
        StdTensorOp::Slice(tenferro_tensor::SliceConfig {
            starts: vec![0],
            limits: vec![1],
            strides: vec![1],
        }),
        StdTensorOp::DynamicSlice {
            slice_sizes: vec![1],
        },
        StdTensorOp::Pad(PadConfig {
            edge_padding_low: vec![0],
            edge_padding_high: vec![0],
            interior_padding: vec![0],
        }),
        StdTensorOp::Convert {
            from: DType::F64,
            to: DType::C64,
        },
        StdTensorOp::NaryEinsum {
            subscripts: "ij,jk,kl->il".into(),
            n_inputs: 3,
        },
        StdTensorOp::Concatenate { axis: 1 },
        StdTensorOp::Reverse { axes: vec![0] },
        StdTensorOp::ShapeOf { axis: 0 },
        StdTensorOp::DynamicTruncate { axis: 0 },
        StdTensorOp::PadToMatch { axis: 0 },
        StdTensorOp::ReduceProd {
            axes: vec![0],
            input_shape: shape![2, 2],
        },
        StdTensorOp::ReduceMax {
            axes: vec![0],
            input_shape: shape![2, 2],
        },
        StdTensorOp::ReduceMin {
            axes: vec![0],
            input_shape: shape![2, 2],
        },
    ];

    for op in variants {
        let mut hasher = DefaultHasher::new();
        op.hash(&mut hasher);
        assert_ne!(hasher.finish(), 0, "unexpected zero hash for {op:?}");
    }

    let mut lhs = DefaultHasher::new();
    StdTensorOp::constant_f64(1.25).hash(&mut lhs);
    let mut rhs = DefaultHasher::new();
    StdTensorOp::constant_f64(1.25).hash(&mut rhs);
    assert_eq!(lhs.finish(), rhs.finish());
}

#[test]
fn test_std_tensor_op_nary_einsum_linearize_emits_term_sum() {
    let op = StdTensorOp::NaryEinsum {
        subscripts: "ij,jk,kl->il".into(),
        n_inputs: 3,
    };
    let (result, fragment) = run_linearize_case(op.clone(), 3, 0, &[true, false, true]);

    assert!(result[0].is_some());
    assert_eq!(fragment.ops().len(), 3);
    assert_eq!(fragment.ops()[0].op, op);
    assert_eq!(
        fragment.ops()[0].mode,
        OpMode::Linear {
            active_mask: vec![true, false, false],
        }
    );
    assert_eq!(fragment.ops()[1].op, op);
    assert_eq!(
        fragment.ops()[1].mode,
        OpMode::Linear {
            active_mask: vec![false, false, true],
        }
    );
    assert_eq!(fragment.ops()[2].op, StdTensorOp::Add);
}

#[test]
fn test_std_tensor_op_nary_einsum_transpose_emits_conjugates_and_vjp_term() {
    let op = StdTensorOp::NaryEinsum {
        subscripts: "ij,jk,kl->il".into(),
        n_inputs: 3,
    };
    let (result, _, fragment) = run_transpose_case(op, 3, &[false, true, false], true);

    assert_eq!(result[0], None);
    assert!(result[1].is_some());
    assert_eq!(result[2], None);
    assert_eq!(fragment.ops().len(), 3);
    assert_eq!(fragment.ops()[0].op, StdTensorOp::Conj);
    assert_eq!(fragment.ops()[1].op, StdTensorOp::Conj);
    assert_eq!(
        fragment.ops()[2].op,
        StdTensorOp::NaryEinsum {
            subscripts: "il,ij,kl->jk".into(),
            n_inputs: 3,
        }
    );
    assert_eq!(
        fragment.ops()[2].mode,
        OpMode::Linear {
            active_mask: vec![true, false, false],
        }
    );
}

#[test]
fn test_std_tensor_op_constant_constructors_encode_expected_bytes() {
    assert_eq!(
        StdTensorOp::constant_f64(1.25),
        StdTensorOp::Constant {
            dtype: DType::F64,
            bytes: 1.25_f64.to_le_bytes().to_vec(),
        }
    );
    assert_eq!(
        StdTensorOp::constant_f32(1.25),
        StdTensorOp::Constant {
            dtype: DType::F32,
            bytes: 1.25_f32.to_le_bytes().to_vec(),
        }
    );

    let c64 = Complex64::new(1.0, -2.0);
    let mut c64_bytes = Vec::new();
    c64_bytes.extend_from_slice(&c64.re.to_le_bytes());
    c64_bytes.extend_from_slice(&c64.im.to_le_bytes());
    assert_eq!(
        StdTensorOp::constant_c64(c64),
        StdTensorOp::Constant {
            dtype: DType::C64,
            bytes: c64_bytes,
        }
    );

    let c32 = Complex32::new(1.0, -2.0);
    let mut c32_bytes = Vec::new();
    c32_bytes.extend_from_slice(&c32.re.to_le_bytes());
    c32_bytes.extend_from_slice(&c32.im.to_le_bytes());
    assert_eq!(
        StdTensorOp::constant_c32(c32),
        StdTensorOp::Constant {
            dtype: DType::C32,
            bytes: c32_bytes,
        }
    );
}

#[test]
fn test_std_tensor_op_linearize_none_tangent_paths_return_none() {
    let unary_primal_in = [
        StdTensorOp::Sin,
        StdTensorOp::Cos,
        StdTensorOp::Log1p,
        StdTensorOp::Abs,
    ];
    for op in unary_primal_in {
        let (result, fragment) = run_linearize_case(op.clone(), 1, 0, &[false]);
        assert_eq!(result, vec![None], "unexpected result for {op:?}");
        assert!(fragment.ops().is_empty(), "expected no ops for {op:?}");
    }

    let unary_primal_out = [StdTensorOp::Tanh, StdTensorOp::Sqrt, StdTensorOp::Expm1];
    for op in unary_primal_out {
        let (result, fragment) = run_linearize_case(op.clone(), 0, 1, &[false]);
        assert_eq!(result, vec![None], "unexpected result for {op:?}");
        assert!(fragment.ops().is_empty(), "expected no ops for {op:?}");
    }

    let (result, fragment) = run_linearize_case(StdTensorOp::Rsqrt, 1, 1, &[false]);
    assert_eq!(result, vec![None]);
    assert!(fragment.ops().is_empty());

    let (result, fragment) = run_linearize_case(StdTensorOp::Pow, 2, 1, &[false, false]);
    assert_eq!(result, vec![None]);
    assert!(fragment.ops().is_empty());

    let (constant_result, constant_fragment) =
        run_linearize_case(StdTensorOp::constant_f64(1.0), 0, 0, &[]);
    assert_eq!(constant_result, vec![None]);
    assert!(constant_fragment.ops().is_empty());
}

#[test]
fn test_std_tensor_op_analytic_linearize_emits_ops_for_remaining_variants() {
    let (sin_result, sin_fragment) = run_linearize_case(StdTensorOp::Sin, 1, 0, &[true]);
    assert!(sin_result[0].is_some());
    assert_eq!(sin_fragment.ops().len(), 2);
    assert_eq!(sin_fragment.ops()[0].op, StdTensorOp::Cos);
    assert_eq!(sin_fragment.ops()[1].op, StdTensorOp::Mul);

    let (cos_result, cos_fragment) = run_linearize_case(StdTensorOp::Cos, 1, 0, &[true]);
    assert!(cos_result[0].is_some());
    assert_eq!(cos_fragment.ops().len(), 3);
    assert_eq!(cos_fragment.ops()[0].op, StdTensorOp::Sin);
    assert_eq!(cos_fragment.ops()[1].op, StdTensorOp::Neg);
    assert_eq!(cos_fragment.ops()[2].op, StdTensorOp::Mul);

    let (tanh_result, tanh_fragment) = run_linearize_case(StdTensorOp::Tanh, 0, 1, &[true]);
    assert!(tanh_result[0].is_some());
    assert_eq!(tanh_fragment.ops().last().unwrap().op, StdTensorOp::Mul);

    let (sqrt_result, sqrt_fragment) = run_linearize_case(StdTensorOp::Sqrt, 0, 1, &[true]);
    assert!(sqrt_result[0].is_some());
    assert_eq!(sqrt_fragment.ops().last().unwrap().op, StdTensorOp::Div);

    let (rsqrt_result, rsqrt_fragment) = run_linearize_case(StdTensorOp::Rsqrt, 1, 1, &[true]);
    assert!(rsqrt_result[0].is_some());
    assert_eq!(rsqrt_fragment.ops().last().unwrap().op, StdTensorOp::Mul);

    let (pow_result, pow_fragment) = run_linearize_case(StdTensorOp::Pow, 2, 1, &[true, true]);
    assert!(pow_result[0].is_some());
    assert_eq!(pow_fragment.ops().last().unwrap().op, StdTensorOp::Add);

    let (expm1_result, expm1_fragment) = run_linearize_case(StdTensorOp::Expm1, 0, 1, &[true]);
    assert!(expm1_result[0].is_some());
    assert_eq!(expm1_fragment.ops().last().unwrap().op, StdTensorOp::Mul);

    let (log1p_result, log1p_fragment) = run_linearize_case(StdTensorOp::Log1p, 1, 0, &[true]);
    assert!(log1p_result[0].is_some());
    assert_eq!(log1p_fragment.ops().last().unwrap().op, StdTensorOp::Div);
}

#[test]
fn test_std_tensor_op_elementwise_tier2_special_cases_are_covered() {
    let (div_sum_result, div_sum_fragment) =
        run_linearize_case(StdTensorOp::Div, 2, 1, &[true, true]);
    assert!(div_sum_result[0].is_some());
    assert_eq!(div_sum_fragment.ops().last().unwrap().op, StdTensorOp::Add);

    let (div_result, div_fragment) = run_linearize_case(StdTensorOp::Div, 2, 1, &[false, true]);
    assert!(div_result[0].is_some());
    assert_eq!(div_fragment.ops().last().unwrap().op, StdTensorOp::Mul);

    let (abs_result, abs_fragment) = run_linearize_case(StdTensorOp::Abs, 1, 0, &[true]);
    assert!(abs_result[0].is_some());
    assert_eq!(abs_fragment.ops()[0].op, StdTensorOp::Sign);
    assert_eq!(abs_fragment.ops()[1].op, StdTensorOp::Mul);

    let (sign_result, sign_fragment) = run_linearize_case(StdTensorOp::Sign, 0, 0, &[true]);
    assert!(sign_result[0].is_some());
    assert_eq!(sign_fragment.ops().len(), 2);
    assert_eq!(sign_fragment.ops()[0].op, StdTensorOp::Neg);
    assert_eq!(sign_fragment.ops()[1].op, StdTensorOp::Add);

    let (transpose_div_result, _, transpose_div_fragment) =
        run_transpose_case(StdTensorOp::Div, 2, &[false, true], true);
    assert_eq!(transpose_div_result[0], None);
    assert!(transpose_div_result[1].is_some());
    assert_eq!(
        transpose_div_fragment.ops().last().unwrap().op,
        StdTensorOp::Mul
    );

    let (transpose_sign_result, _, transpose_sign_fragment) =
        run_transpose_case(StdTensorOp::Sign, 1, &[true], true);
    assert!(transpose_sign_result[0].is_some());
    assert_eq!(transpose_sign_fragment.ops().len(), 2);
    assert_eq!(transpose_sign_fragment.ops()[0].op, StdTensorOp::Neg);
    assert_eq!(transpose_sign_fragment.ops()[1].op, StdTensorOp::Add);

    let (transpose_div_none_result, _, transpose_div_none_fragment) =
        run_transpose_case(StdTensorOp::Div, 2, &[true, true], false);
    assert_eq!(transpose_div_none_result, vec![None, None]);
    assert!(transpose_div_none_fragment.ops().is_empty());

    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input_key(920));
    let div_primal_mode = StdTensorOp::Div.transpose_rule(
        &mut builder,
        &[Some(cotangent)],
        &external_inputs(921, 2),
        &OpMode::Primal,
        &mut ad_ctx,
    );
    assert_eq!(div_primal_mode, vec![None, None]);
    assert!(builder.build().ops().is_empty());

    let (transpose_abs_result, _, transpose_abs_fragment) =
        run_transpose_case(StdTensorOp::Abs, 1, &[true], true);
    assert!(transpose_abs_result[0].is_some());
    assert_eq!(transpose_abs_fragment.ops()[0].op, StdTensorOp::Sign);
    assert_eq!(transpose_abs_fragment.ops()[1].op, StdTensorOp::Mul);
}

#[test]
fn test_std_tensor_op_constant_transpose_rule_has_no_inputs_or_ops() {
    let (result, _, fragment) = run_transpose_case(StdTensorOp::constant_f64(1.0), 0, &[], true);
    assert!(result.is_empty());
    assert!(fragment.ops().is_empty());
}

#[test]
fn test_std_tensor_op_analytic_transpose_rule_emits_ops_for_remaining_variants() {
    let (exp_result, _, exp_fragment) = run_transpose_case(StdTensorOp::Exp, 1, &[true], true);
    assert!(exp_result[0].is_some());
    assert_eq!(exp_fragment.ops()[0].op, StdTensorOp::Exp);
    assert_eq!(exp_fragment.ops()[1].op, StdTensorOp::Mul);

    let (log_result, _, log_fragment) = run_transpose_case(StdTensorOp::Log, 1, &[true], true);
    assert!(log_result[0].is_some());
    assert_eq!(log_fragment.ops()[0].op, StdTensorOp::Div);

    let (sin_result, _, sin_fragment) = run_transpose_case(StdTensorOp::Sin, 1, &[true], true);
    assert!(sin_result[0].is_some());
    assert_eq!(sin_fragment.ops().len(), 2);
    assert_eq!(sin_fragment.ops()[0].op, StdTensorOp::Cos);
    assert_eq!(sin_fragment.ops()[1].op, StdTensorOp::Mul);

    let (cos_result, _, cos_fragment) = run_transpose_case(StdTensorOp::Cos, 1, &[true], true);
    assert!(cos_result[0].is_some());
    assert_eq!(cos_fragment.ops().len(), 3);
    assert_eq!(cos_fragment.ops()[0].op, StdTensorOp::Sin);
    assert_eq!(cos_fragment.ops()[1].op, StdTensorOp::Neg);
    assert_eq!(cos_fragment.ops()[2].op, StdTensorOp::Mul);

    let (tanh_result, _, tanh_fragment) = run_transpose_case(StdTensorOp::Tanh, 1, &[true], true);
    assert!(tanh_result[0].is_some());
    assert_eq!(tanh_fragment.ops().last().unwrap().op, StdTensorOp::Mul);

    let (sqrt_result, _, sqrt_fragment) = run_transpose_case(StdTensorOp::Sqrt, 1, &[true], true);
    assert!(sqrt_result[0].is_some());
    assert_eq!(sqrt_fragment.ops().last().unwrap().op, StdTensorOp::Div);

    let (rsqrt_result, _, rsqrt_fragment) =
        run_transpose_case(StdTensorOp::Rsqrt, 1, &[true], true);
    assert!(rsqrt_result[0].is_some());
    assert_eq!(rsqrt_fragment.ops().last().unwrap().op, StdTensorOp::Mul);

    let (pow_result, _, pow_fragment) =
        run_transpose_case(StdTensorOp::Pow, 2, &[true, true], true);
    assert!(pow_result[0].is_some());
    assert!(pow_result[1].is_some());
    assert_eq!(pow_fragment.ops().last().unwrap().op, StdTensorOp::Mul);

    let (expm1_result, _, expm1_fragment) =
        run_transpose_case(StdTensorOp::Expm1, 1, &[true], true);
    assert!(expm1_result[0].is_some());
    assert_eq!(expm1_fragment.ops().last().unwrap().op, StdTensorOp::Mul);

    let (log1p_result, _, log1p_fragment) =
        run_transpose_case(StdTensorOp::Log1p, 1, &[true], true);
    assert!(log1p_result[0].is_some());
    assert_eq!(log1p_fragment.ops().last().unwrap().op, StdTensorOp::Div);
}

#[test]
fn test_std_tensor_op_transpose_none_or_inactive_paths_return_none() {
    let unary_ops = [
        StdTensorOp::Sin,
        StdTensorOp::Cos,
        StdTensorOp::Tanh,
        StdTensorOp::Sqrt,
        StdTensorOp::Rsqrt,
        StdTensorOp::Expm1,
        StdTensorOp::Log1p,
        StdTensorOp::Abs,
        StdTensorOp::Sign,
    ];

    for op in unary_ops {
        let (none_result, _, none_fragment) = run_transpose_case(op.clone(), 1, &[true], false);
        assert_eq!(none_result, vec![None], "unexpected None path for {op:?}");
        assert!(none_fragment.ops().is_empty(), "expected no ops for {op:?}");

        let mut builder = FragmentBuilder::<StdTensorOp>::new();
        let mut ad_ctx = ShapeGuardContext::default();
        let cotangent = builder.add_input(tensor_input_key(900));
        let result = op.transpose_rule(
            &mut builder,
            &[Some(cotangent)],
            &external_inputs(901, 1),
            &OpMode::Primal,
            &mut ad_ctx,
        );
        assert_eq!(result, vec![None], "unexpected inactive path for {op:?}");
        assert!(
            builder.build().ops().is_empty(),
            "expected no ops for {op:?}"
        );
    }
}

#[test]
fn test_std_tensor_op_structural_special_cases_cover_identity_and_empty_axes() {
    let (transpose_none_result, transpose_none_fragment) =
        run_linearize_case(StdTensorOp::Transpose { perm: vec![1, 0] }, 0, 0, &[false]);
    assert_eq!(transpose_none_result, vec![None]);
    assert!(transpose_none_fragment.ops().is_empty());

    let reshape = StdTensorOp::Reshape {
        from_shape: shape![4],
        to_shape: shape![2, 2],
    };
    let (reshape_linear_result, reshape_linear_fragment) =
        run_linearize_case(reshape.clone(), 0, 0, &[true]);
    assert!(reshape_linear_result[0].is_some());
    assert_eq!(reshape_linear_fragment.ops().len(), 1);
    assert_eq!(reshape_linear_fragment.ops()[0].op, reshape);

    let (transpose_result, _, transpose_fragment) = run_transpose_case(
        StdTensorOp::Transpose { perm: vec![0, 1] },
        1,
        &[true],
        true,
    );
    assert!(transpose_result[0].is_some());
    assert_eq!(transpose_fragment.ops().len(), 1);
    assert_eq!(
        transpose_fragment.ops()[0].op,
        StdTensorOp::Transpose { perm: vec![0, 1] }
    );

    let (broadcast_linear_result, broadcast_linear_fragment) = run_linearize_case(
        StdTensorOp::BroadcastInDim {
            shape: shape![2, 2, 3],
            dims: vec![0, 2],
        },
        0,
        0,
        &[true],
    );
    assert!(broadcast_linear_result[0].is_some());
    assert_eq!(
        broadcast_linear_fragment.ops()[0].op,
        StdTensorOp::BroadcastInDim {
            shape: shape![2, 2, 3],
            dims: vec![0, 2],
        }
    );

    let (broadcast_none_result, broadcast_none_fragment) = run_linearize_case(
        StdTensorOp::BroadcastInDim {
            shape: shape![2, 2, 3],
            dims: vec![0, 2],
        },
        0,
        0,
        &[false],
    );
    assert_eq!(broadcast_none_result, vec![None]);
    assert!(broadcast_none_fragment.ops().is_empty());

    let (reshape_transpose_result, _, reshape_transpose_fragment) =
        run_transpose_case(reshape, 1, &[true], true);
    assert!(reshape_transpose_result[0].is_some());
    assert_eq!(reshape_transpose_fragment.ops().len(), 1);
    assert_eq!(
        reshape_transpose_fragment.ops()[0].op,
        StdTensorOp::Reshape {
            from_shape: shape![2, 2],
            to_shape: shape![4],
        }
    );

    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input_key(910));
    let result = StdTensorOp::BroadcastInDim {
        shape: shape![2, 3],
        dims: vec![0, 1],
    }
    .transpose_rule(
        &mut builder,
        &[Some(cotangent)],
        &external_inputs(911, 1),
        &linear_mode(&[true]),
        &mut ad_ctx,
    );
    assert_eq!(result, vec![Some(cotangent)]);
    assert!(builder.build().ops().is_empty());

    let (tril_result, tril_fragment) =
        run_linearize_case(StdTensorOp::Tril { k: -1 }, 0, 0, &[true]);
    assert!(tril_result[0].is_some());
    assert_eq!(tril_fragment.ops()[0].op, StdTensorOp::Tril { k: -1 });

    let (triu_result, triu_fragment) =
        run_linearize_case(StdTensorOp::Triu { k: 2 }, 0, 0, &[true]);
    assert!(triu_result[0].is_some());
    assert_eq!(triu_fragment.ops()[0].op, StdTensorOp::Triu { k: 2 });

    let (transpose_tril_result, _, transpose_tril_fragment) =
        run_transpose_case(StdTensorOp::Tril { k: 0 }, 1, &[true], true);
    assert!(transpose_tril_result[0].is_some());
    assert_eq!(
        transpose_tril_fragment.ops()[0].op,
        StdTensorOp::Tril { k: 0 }
    );

    let (transpose_triu_result, _, transpose_triu_fragment) =
        run_transpose_case(StdTensorOp::Triu { k: 1 }, 1, &[true], true);
    assert!(transpose_triu_result[0].is_some());
    assert_eq!(
        transpose_triu_fragment.ops()[0].op,
        StdTensorOp::Triu { k: 1 }
    );

    let (transpose_transpose_none_result, _, transpose_transpose_none_fragment) =
        run_transpose_case(
            StdTensorOp::Transpose {
                perm: vec![2, 0, 1],
            },
            1,
            &[true],
            false,
        );
    assert_eq!(transpose_transpose_none_result, vec![None]);
    assert!(transpose_transpose_none_fragment.ops().is_empty());

    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let none_broadcast = StdTensorOp::BroadcastInDim {
        shape: shape![2, 2, 3],
        dims: vec![0, 2],
    }
    .transpose_rule(
        &mut builder,
        &[None],
        &external_inputs(930, 1),
        &linear_mode(&[true]),
        &mut ad_ctx,
    );
    assert_eq!(none_broadcast, vec![None]);
    assert!(builder.build().ops().is_empty());
}

#[test]
fn test_std_tensor_op_convert_linearize_and_transpose_swap_dtypes() {
    let convert = StdTensorOp::Convert {
        from: DType::F64,
        to: DType::C64,
    };

    let (linear_result, linear_fragment) = run_linearize_case(convert.clone(), 0, 0, &[true]);
    assert!(linear_result[0].is_some());
    assert_eq!(linear_fragment.ops().len(), 1);
    assert_eq!(linear_fragment.ops()[0].op, convert);

    let (linear_none_result, linear_none_fragment) =
        run_linearize_case(convert.clone(), 0, 0, &[false]);
    assert_eq!(linear_none_result, vec![None]);
    assert!(linear_none_fragment.ops().is_empty());

    let (transpose_result, _, transpose_fragment) =
        run_transpose_case(convert.clone(), 1, &[true], true);
    assert!(transpose_result[0].is_some());
    assert_eq!(transpose_fragment.ops().len(), 1);
    assert_eq!(
        transpose_fragment.ops()[0].op,
        StdTensorOp::Convert {
            from: DType::C64,
            to: DType::F64,
        }
    );

    let (transpose_inactive_result, _, transpose_inactive_fragment) =
        run_transpose_case(convert, 1, &[false], true);
    assert_eq!(transpose_inactive_result, vec![None]);
    assert!(transpose_inactive_fragment.ops().is_empty());
}

#[test]
fn test_std_tensor_op_contraction_special_cases_cover_none_and_scalar_paths() {
    let matmul = StdTensorOp::DotGeneral {
        config: DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    };
    let (linearize_none_result, linearize_none_fragment) =
        run_linearize_case(matmul.clone(), 2, 0, &[false, false]);
    assert_eq!(linearize_none_result, vec![None]);
    assert!(linearize_none_fragment.ops().is_empty());

    let (transpose_none_result, _, transpose_none_fragment) =
        run_transpose_case(matmul.clone(), 2, &[true, true], false);
    assert_eq!(transpose_none_result, vec![None, None]);
    assert!(transpose_none_fragment.ops().is_empty());

    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input_key(980));
    seed_dot_general_ref_metadata(&mut ad_ctx, &external_inputs(981, 2));
    let primal_mode_result = matmul.transpose_rule(
        &mut builder,
        &[Some(cotangent)],
        &external_inputs(981, 2),
        &OpMode::Primal,
        &mut ad_ctx,
    );
    assert_eq!(primal_mode_result, vec![None, None]);
    assert!(builder.build().ops().is_empty());

    let reduce = StdTensorOp::ReduceSum {
        axes: vec![1],
        input_shape: shape![2, 3],
    };
    let (reduce_linearize_none_result, reduce_linearize_none_fragment) =
        run_linearize_case(reduce.clone(), 0, 0, &[false]);
    assert_eq!(reduce_linearize_none_result, vec![None]);
    assert!(reduce_linearize_none_fragment.ops().is_empty());

    let (reduce_transpose_result, _, reduce_transpose_fragment) =
        run_transpose_case(reduce.clone(), 1, &[true], true);
    assert!(reduce_transpose_result[0].is_some());
    assert_eq!(reduce_transpose_fragment.ops().len(), 1);
    assert_eq!(
        reduce_transpose_fragment.ops()[0].op,
        StdTensorOp::BroadcastInDim {
            shape: shape![2, 3],
            dims: vec![0],
        }
    );

    let (reduce_transpose_none_result, _, reduce_transpose_none_fragment) =
        run_transpose_case(reduce, 1, &[true], false);
    assert_eq!(reduce_transpose_none_result, vec![None]);
    assert!(reduce_transpose_none_fragment.ops().is_empty());

    let scalar_contract = StdTensorOp::DotGeneral {
        config: DotGeneralConfig {
            lhs_contracting_dims: vec![1, 0],
            rhs_contracting_dims: vec![0, 1],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    };
    let (scalar_transpose_result, _, scalar_transpose_fragment) =
        run_transpose_case(scalar_contract, 2, &[true, false], true);
    assert!(scalar_transpose_result[0].is_some());
    assert_eq!(scalar_transpose_result[1], None);
    assert_eq!(
        scalar_transpose_fragment.ops()[0].op,
        StdTensorOp::Reshape {
            from_shape: shape![1],
            to_shape: shape![],
        }
    );
    assert_eq!(scalar_transpose_fragment.ops()[1].op, StdTensorOp::Conj);
    assert_eq!(
        scalar_transpose_fragment.ops().last().unwrap().op,
        StdTensorOp::Transpose { perm: vec![1, 0] }
    );
}

#[test]
#[should_panic(expected = "linearize not implemented for Maximum")]
fn test_std_tensor_op_linearize_panics_for_unimplemented_variant() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let _ = StdTensorOp::Maximum.linearize(&mut builder, &[], &[], &[None, None], &mut ad_ctx);
}

#[test]
#[should_panic(expected = "transpose_rule not implemented for Maximum")]
fn test_std_tensor_op_transpose_rule_panics_for_unimplemented_variant() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let _ = StdTensorOp::Maximum.transpose_rule(
        &mut builder,
        &[None],
        &external_inputs(950, 2),
        &OpMode::Primal,
        &mut ad_ctx,
    );
}
