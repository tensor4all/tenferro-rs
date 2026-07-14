use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

use super::*;
use crate::optimize::EinsumPlanSpec;
#[cfg(feature = "autodiff")]
use computegraph::graph::GraphBuilder;
#[cfg(feature = "autodiff")]
use computegraph::types::OperationRole;
use tenferro_cpu::CpuBackend;
#[cfg(feature = "autodiff")]
use tenferro_ops::ext_op::ExtensionLinearTransposeRule;
use tenferro_ops::ext_op::{invoke_extension_shape_inference, ExtensionOp};
#[cfg(feature = "autodiff")]
use tenferro_ops::input_key::TensorInputKey;
#[cfg(feature = "autodiff")]
use tenferro_ops::TensorMeta;
use tenferro_runtime::ExtensionCacheStore;
use tenferro_tensor::{TensorOwnedView, TensorRead};
#[cfg(feature = "autodiff")]
use tidu::PrimitiveTransposeInput;

#[test]
fn infer_output_meta_uses_output_labels_and_promotes_dtype() {
    let op = EinsumExtensionOp::new(EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]));
    let lhs_shape = [SymDim::from(2usize), SymDim::from(3usize)];
    let rhs_shape = [SymDim::from(3usize), SymDim::from(4usize)];

    let meta = invoke_extension_shape_inference(
        &op,
        &[DType::F32, DType::F64],
        &[lhs_shape.as_slice(), rhs_shape.as_slice()],
    )
    .unwrap()
    .output_metas;

    assert_eq!(meta[0].0, DType::F64);
    assert_eq!(meta[0].1, vec![SymDim::from(2usize), SymDim::from(4usize)]);
}

#[test]
fn extension_dtype_promotion_delegates_to_canonical_tensor_rules() {
    let source = include_str!("../extension.rs");
    assert!(
        !source.contains("fn promote_dtype("),
        "einsum extension metadata must not duplicate the canonical dtype promotion lattice"
    );

    let dtypes = [
        DType::Bool,
        DType::I32,
        DType::I64,
        DType::F32,
        DType::F64,
        DType::C32,
        DType::C64,
    ];
    for lhs in dtypes {
        for rhs in dtypes {
            assert_eq!(
                promote_dtypes([lhs, rhs]),
                tenferro_tensor::validate::promote_dtype(lhs, rhs),
                "promotion mismatch for {lhs:?}, {rhs:?}"
            );
        }
    }
}

#[test]
fn infer_output_meta_keeps_structural_errors_and_records_extent_equality() {
    let op = EinsumExtensionOp::new(EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]));
    let lhs_shape = [SymDim::from(2usize), SymDim::from(3usize)];
    let bad_rhs_rank = [SymDim::from(3usize)];
    let bad_rhs_extent = [SymDim::from(5usize), SymDim::from(4usize)];

    assert!(invoke_extension_shape_inference(
        &op,
        &[DType::F64],
        &[lhs_shape.as_slice(), bad_rhs_rank.as_slice()]
    )
    .is_err());
    assert!(invoke_extension_shape_inference(
        &op,
        &[DType::F64, DType::F64],
        &[lhs_shape.as_slice(), bad_rhs_rank.as_slice()]
    )
    .is_err());
    let inferred = invoke_extension_shape_inference(
        &op,
        &[DType::F64, DType::F64],
        &[lhs_shape.as_slice(), bad_rhs_extent.as_slice()],
    )
    .expect("extent mismatch is represented as a shape equality");
    assert_eq!(inferred.constraints.len(), 1);
}

#[test]
fn payload_identity_ignores_static_tree_execution_hint() {
    let subscripts = EinsumSubscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let raw_subscripts = crate::Subscripts::from(&subscripts);
    let shapes = [&[2, 3][..], &[3, 4][..], &[4, 5][..]];
    let left_first =
        Arc::new(ContractionTree::from_pairs(&raw_subscripts, &shapes, &[(0, 1), (3, 2)]).unwrap());
    let right_first =
        Arc::new(ContractionTree::from_pairs(&raw_subscripts, &shapes, &[(1, 2), (0, 3)]).unwrap());
    let plan_spec = EinsumPlanSpec::LeftToRight;

    let default_without_hint = EinsumExtensionOp::new(subscripts.clone());
    let default_hinted =
        EinsumExtensionOp::with_static_tree(subscripts.clone(), Arc::clone(&left_first));
    let without_hint = EinsumExtensionOp::with_plan_spec(subscripts.clone(), plan_spec.clone());
    let hinted_left = EinsumExtensionOp::with_plan_spec(subscripts.clone(), plan_spec.clone())
        .with_static_tree_hint(left_first);
    let hinted_right =
        EinsumExtensionOp::with_plan_spec(subscripts, plan_spec).with_static_tree_hint(right_first);

    assert!(default_without_hint.payload_eq(&default_hinted));
    assert_eq!(
        payload_hash(&default_without_hint),
        payload_hash(&default_hinted)
    );
    assert!(without_hint.payload_eq(&hinted_left));
    assert!(hinted_left.payload_eq(&hinted_right));
    assert_eq!(payload_hash(&without_hint), payload_hash(&hinted_left));
    assert_eq!(payload_hash(&hinted_left), payload_hash(&hinted_right));
}

#[test]
fn payload_identity_includes_plan_spec() {
    let subscripts = EinsumSubscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let left_to_right =
        EinsumExtensionOp::with_plan_spec(subscripts.clone(), EinsumPlanSpec::LeftToRight);
    let explicit_path =
        EinsumExtensionOp::with_plan_spec(subscripts, EinsumPlanSpec::Path(vec![(1, 2), (0, 1)]));

    assert!(!left_to_right.payload_eq(&explicit_path));
    assert_ne!(payload_hash(&left_to_right), payload_hash(&explicit_path));
}

#[test]
fn payload_identity_includes_output_shape_hint() {
    let subscripts = EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let plan_spec = EinsumPlanSpec::LeftToRight;
    let without_hint = EinsumExtensionOp::with_plan_spec(subscripts.clone(), plan_spec.clone());
    let with_hint = EinsumExtensionOp::with_output_shape_hint(
        subscripts,
        vec![SymDim::from(2usize), SymDim::from(4usize)],
        plan_spec,
    );

    assert!(!without_hint.payload_eq(&with_hint));
    assert_ne!(payload_hash(&without_hint), payload_hash(&with_hint));
}

#[test]
fn einsum_extension_caches_verify_exact_key_data_after_hash_lookup() {
    let extension_source = include_str!("../extension.rs");
    assert!(extension_source.contains("struct RuntimeTreeCacheKeyData"));
    assert!(extension_source.contains("struct CachedRuntimeTree"));
    assert!(extension_source.contains("struct RuntimeExecProgramCacheKeyData"));
    assert!(extension_source.contains("key_data.matches_runtime_tree("));
    assert!(extension_source.contains("key_data.matches_runtime_exec_program("));
    assert!(!extension_source.contains("get::<Arc<ContractionTree>>(&key)"));

    let traced_source = include_str!("../traced.rs");
    assert!(traced_source.contains("struct ParsedEinsumCacheEntry"));
    assert!(traced_source.contains("struct StaticTreeCacheKeyData"));
    assert!(traced_source.contains("struct CachedStaticTree"));
    assert!(traced_source.contains("key_data.matches_static_tree("));
    assert!(!traced_source.contains("get::<Arc<ParsedEinsum>>(&key)"));
    assert!(!traced_source.contains("get::<Arc<ContractionTree>>(&key)"));

    let eager_source = include_str!("../eager_ad.rs");
    assert!(eager_source.contains("struct ExpandedEagerProgramCacheKeyData"));
    assert!(eager_source.contains("struct CachedExpandedEagerProgram"));
    assert!(eager_source.contains("key_data.matches_expanded_eager_program("));
    assert!(!eager_source.contains("get::<Arc<ExpandedEagerProgram>>(&key)"));
}

#[test]
fn runtime_input_index_vec_stays_inline_for_common_arity() {
    let mut indices = InputIndexVec::new();
    indices.extend(0..4);

    assert!(!indices.spilled());
}

#[test]
fn execute_einsum_extension_reads_consumes_strided_view_inputs() {
    let base = Arc::new(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    let view = TensorOwnedView::from_parts(Arc::clone(&base), vec![3, 2], vec![2, 1], 0).unwrap();
    let input = TensorRead::from_view(view.tensor_view());
    let op = EinsumExtensionOp::new(EinsumSubscripts::new(&[&[0, 1]], &[0, 1]));
    let mut backend = CpuBackend::new();
    let mut caches = ExtensionCacheStore::new();
    let mut ctx = ExtensionExecutionContext::new(&mut backend, &mut caches);

    let outputs = execute_einsum_extension_reads(&op, &[input], &mut ctx)
        .expect("read-capable einsum extension execution");

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape(), &[3, 2]);
    assert_eq!(
        outputs[0].as_slice::<f64>().unwrap(),
        &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );
}

#[test]
#[cfg(feature = "autodiff")]
fn vjp_einsum_op_inherits_plan_spec_and_precomputes_concrete_tree() {
    let primal_op = EinsumExtensionOp::with_plan_spec(
        EinsumSubscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]),
        EinsumPlanSpec::Path(vec![(1, 2), (0, 1)]),
    );
    let vjp_subscripts = EinsumSubscripts {
        inputs: vec![vec![0, 3], vec![1, 2], vec![2, 3]],
        output: vec![0, 1],
    };
    let vjp_shapes = vec![
        vec![SymDim::from(2usize), SymDim::from(5usize)],
        vec![SymDim::from(3usize), SymDim::from(4usize)],
        vec![SymDim::from(4usize), SymDim::from(5usize)],
    ];

    let op = vjp_einsum_op_with_inherited_plan(
        &primal_op,
        0,
        vjp_subscripts,
        vec![SymDim::from(2usize), SymDim::from(3usize)],
        &vjp_shapes,
    )
    .unwrap();

    assert!(matches!(
        op.plan_spec(),
        EinsumPlanSpec::FixedPairs(pairs) if pairs == &vec![(1, 2), (0, 3)]
    ));
    let tree = op.static_tree().expect("expected concrete VJP tree");
    assert_eq!(tree.step_pair(0), Some((1, 2)));
    assert_eq!(tree.step_pair(1), Some((0, 3)));
}

#[test]
#[cfg(feature = "autodiff")]
fn vjp_einsum_op_derives_plan_for_nonfirst_active_input() {
    let primal_op = EinsumExtensionOp::with_plan_spec(
        EinsumSubscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]),
        EinsumPlanSpec::Path(vec![(1, 2), (0, 1)]),
    );
    let vjp_subscripts = EinsumSubscripts {
        inputs: vec![vec![0, 3], vec![0, 1], vec![2, 3]],
        output: vec![1, 2],
    };
    let vjp_shapes = vec![
        vec![SymDim::from(2usize), SymDim::from(5usize)],
        vec![SymDim::from(2usize), SymDim::from(3usize)],
        vec![SymDim::from(4usize), SymDim::from(5usize)],
    ];

    let op = vjp_einsum_op_with_inherited_plan(
        &primal_op,
        1,
        vjp_subscripts,
        vec![SymDim::from(3usize), SymDim::from(4usize)],
        &vjp_shapes,
    )
    .unwrap();

    assert!(matches!(
        op.plan_spec(),
        EinsumPlanSpec::FixedPairs(pairs) if pairs == &vec![(0, 1), (3, 2)]
    ));
    let tree = op.static_tree().expect("expected concrete VJP tree");
    assert_eq!(tree.step_pair(0), Some((0, 1)));
    assert_eq!(tree.step_pair(1), Some((3, 2)));
}

#[test]
#[cfg(feature = "autodiff")]
fn repeated_label_projection_projects_each_extra_occurrence() {
    let mut builder = RecordingRuleBuilder::default();

    let result = project_repeated_labels_to_diagonal(&mut builder, 0, &[0, 1, 1, 1]);

    assert_eq!(result, 4);
    assert_eq!(
        builder.ops,
        vec![
            StdTensorOp::ExtractDiag {
                axis_a: 1,
                axis_b: 2,
            },
            StdTensorOp::EmbedDiag {
                axis_a: 1,
                axis_b: 2,
            },
            StdTensorOp::ExtractDiag {
                axis_a: 1,
                axis_b: 3,
            },
            StdTensorOp::EmbedDiag {
                axis_a: 1,
                axis_b: 3,
            },
        ]
    );
}

#[test]
#[cfg(feature = "autodiff")]
fn vjp_broadcast_remap_failure_returns_error() {
    let mut builder = RecordingRuleBuilder::default();

    let err = broadcast_einsum_vjp_to_input_shape(
        &mut builder,
        0,
        &[0, 2],
        &[0, 1],
        vec![
            DimExpr::InputDim {
                input_idx: 1,
                axis: 0,
            },
            DimExpr::InputDim {
                input_idx: 1,
                axis: 1,
            },
        ],
        vec![ValueRef::Local(1)],
    )
    .expect_err("unmappable VJP labels should be an AD rule error");

    let message = err.to_string();
    assert!(message.contains("einsum VJP broadcast remap"));
    assert!(message.contains("cotangent"));
    assert!(builder.ops.is_empty());
}

#[cfg(feature = "autodiff")]
fn ad_input_key(id: u64) -> TensorInputKey {
    TensorInputKey::User { id }
}

#[cfg(feature = "autodiff")]
fn ad_value_key(id: u64) -> ValueKey<StdTensorOp> {
    ValueKey::Input(ad_input_key(id))
}

#[test]
#[cfg(feature = "autodiff")]
fn linear_transpose_broadcasts_linear_only_active_input_from_metadata() {
    let rule = EinsumAdRule;
    let op = EinsumExtensionOp::with_output_shape_hint(
        EinsumSubscripts {
            inputs: vec![vec![b'i' as u32]],
            output: vec![],
        },
        vec![],
        EinsumPlanSpec::LeftToRight,
    );
    let active_key = ad_value_key(10);
    let mut ctx = ShapeGuardContext::default();
    ctx.insert_metadata(
        active_key.clone(),
        TensorMeta::exact(DType::F64, vec![SymDim::from(3usize)]),
    );

    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let cotangent = builder.add_input(ad_input_key(0));
    let result = rule
        .linear_transpose(
            &op,
            &mut builder,
            &[Some(cotangent)],
            &[PrimitiveTransposeInput::Linear {
                key: active_key.clone(),
                primal: None,
            }],
            &[true],
            &mut ctx,
        )
        .unwrap();

    assert_eq!(result.len(), 1);
    assert!(result[0].is_some());
    let active_ref = ValueRef::External(active_key);
    let graph = builder.build();
    assert!(graph
        .operations()
        .iter()
        .all(|node| !node.inputs.iter().any(|input| input == &active_ref)));
    assert!(graph.operations().iter().any(|node| {
        matches!(
            (&node.operation, &node.role, node.inputs.len()),
            (
                StdTensorOp::BroadcastInDim { shape, dims },
                OperationRole::Linearized { active_mask },
                1
            ) if shape == &[DimExpr::Const(3)] && dims.is_empty() && active_mask == &[true]
        )
    }));
}

#[test]
#[cfg(feature = "autodiff")]
fn linear_transpose_rejects_linear_only_coefficient_input() {
    let rule = EinsumAdRule;
    let op = EinsumExtensionOp::with_output_shape_hint(
        EinsumSubscripts {
            inputs: vec![vec![b'i' as u32], vec![b'i' as u32]],
            output: vec![],
        },
        vec![],
        EinsumPlanSpec::LeftToRight,
    );
    let active_key = ad_value_key(10);
    let coefficient_key = ad_value_key(11);
    let mut ctx = ShapeGuardContext::default();
    for key in [&active_key, &coefficient_key] {
        ctx.insert_metadata(
            key.clone(),
            TensorMeta::exact(DType::F64, vec![SymDim::from(3usize)]),
        );
    }

    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let cotangent = builder.add_input(ad_input_key(0));
    let err = rule
        .linear_transpose(
            &op,
            &mut builder,
            &[Some(cotangent)],
            &[
                PrimitiveTransposeInput::Linear {
                    key: active_key,
                    primal: None,
                },
                PrimitiveTransposeInput::Linear {
                    key: coefficient_key,
                    primal: None,
                },
            ],
            &[true, false],
            &mut ctx,
        )
        .unwrap_err();

    let message = err.to_string();
    assert!(message.contains("linear-only"), "{message}");
    assert!(message.contains("einsum VJP"), "{message}");
}

#[cfg(feature = "autodiff")]
#[derive(Default)]
struct RecordingRuleBuilder {
    ops: Vec<StdTensorOp>,
    next_id: LocalValueId,
}

#[cfg(feature = "autodiff")]
impl PrimitiveRuleBuilder for RecordingRuleBuilder {
    fn add_operation(
        &mut self,
        operation: StdTensorOp,
        _inputs: Vec<ValueRef<StdTensorOp>>,
        _role: OperationRole,
    ) -> Vec<LocalValueId> {
        self.ops.push(operation);
        self.next_id += 1;
        vec![self.next_id]
    }
}

fn payload_hash(op: &EinsumExtensionOp) -> u64 {
    let mut hasher = DefaultHasher::new();
    op.payload_hash(&mut hasher);
    hasher.finish()
}
