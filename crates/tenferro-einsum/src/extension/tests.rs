use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

use super::*;
use crate::optimize::EinsumPlanSpec;
use tenferro_cpu::CpuBackend;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_runtime::ExtensionCacheStore;
use tenferro_tensor::{TensorOwnedView, TensorRead};

#[test]
fn infer_output_meta_uses_output_labels_and_promotes_dtype() {
    let op = EinsumExtensionOp::new(EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]));
    let lhs_shape = [SymDim::from(2usize), SymDim::from(3usize)];
    let rhs_shape = [SymDim::from(3usize), SymDim::from(4usize)];

    let meta = op.infer_output_meta(
        &[DType::F32, DType::F64],
        &[lhs_shape.as_slice(), rhs_shape.as_slice()],
    );

    assert_eq!(meta[0].0, DType::F64);
    assert_eq!(meta[0].1, vec![SymDim::from(2usize), SymDim::from(4usize)]);
}

#[test]
fn infer_output_meta_returns_empty_for_invalid_extension_metadata() {
    let op = EinsumExtensionOp::new(EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]));
    let lhs_shape = [SymDim::from(2usize), SymDim::from(3usize)];
    let bad_rhs_rank = [SymDim::from(3usize)];
    let bad_rhs_extent = [SymDim::from(5usize), SymDim::from(4usize)];

    assert!(op
        .infer_output_meta(
            &[DType::F64],
            &[lhs_shape.as_slice(), bad_rhs_rank.as_slice()]
        )
        .is_empty());
    assert!(op
        .infer_output_meta(
            &[DType::F64, DType::F64],
            &[lhs_shape.as_slice(), bad_rhs_rank.as_slice()]
        )
        .is_empty());
    assert!(op
        .infer_output_meta(
            &[DType::F64, DType::F64],
            &[lhs_shape.as_slice(), bad_rhs_extent.as_slice()]
        )
        .is_empty());
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
fn runtime_input_index_vec_stays_inline_for_common_arity() {
    let mut indices = InputIndexVec::new();
    indices.extend(0..4);

    assert!(!indices.spilled());
}

#[test]
fn execute_einsum_extension_reads_consumes_strided_view_inputs() {
    let base = Arc::new(Tensor::from_vec_col_major(
        vec![2, 3],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
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
