use std::collections::HashMap;
use std::sync::Arc;

use computegraph::graph::GraphBuilder;
use computegraph::types::ValueKey;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::ShapeRelation;
use tenferro_tensor::{DType, Tensor};

use crate::shape_constraint::{ConstraintSource, LocalShapeConstraint, ScopedShapeConstraint};
use crate::sym_dim::SymDim;

use super::{
    allocate_input_key, ones_tensor, tensor_from_parts, ConstraintScopeTransfer, RetainedValue,
    ShapeConstraintScope, TracedTensorParts,
};

fn nonempty_constraint_scope(id: u64) -> ShapeConstraintScope {
    let key = ValueKey::Input(TensorInputKey::User { id });
    ShapeConstraintScope::new(vec![ScopedShapeConstraint {
        origins: vec![key.clone()],
        inputs: vec![key],
        local: LocalShapeConstraint {
            source: ConstraintSource {
                family_id: "tenferro-tests.ad-support-transfer.v1",
                instruction_index: None,
            },
            relation: ShapeRelation::Equal,
            lhs: DimExpr::InputDim {
                input_idx: 0,
                axis: 0,
            },
            rhs: DimExpr::InputDim {
                input_idx: 0,
                axis: 0,
            },
        },
    }])
}

#[test]
fn traced_tensor_parts_debug_summarizes_without_graph_payload() {
    let input_key = allocate_input_key();
    let mut builder = GraphBuilder::new();
    let val = builder.add_input(input_key.clone());
    builder.set_outputs(vec![val]);
    let graph = Arc::new(builder.build());
    let data = Arc::new(RetainedValue::from_tensor(
        Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap(),
    ));
    let inputs_map = Arc::new(HashMap::from([(input_key, Arc::clone(&data))]));
    let parts = TracedTensorParts {
        rank: 1,
        dtype: DType::F64,
        graph,
        val,
        data: Some(data),
        shape_hint: Some(vec![SymDim::from(1)]),
        inputs_map,
        leaf_metas: Arc::new(HashMap::new()),
        extra_roots: Vec::new(),
        checkpoint_chain: None,
        metadata_scopes: Vec::new(),
        constraint_scope_transfer: ConstraintScopeTransfer::empty(),
    };

    let debug = format!("{parts:?}");

    assert!(debug.contains("TracedTensorParts"));
    assert!(debug.contains("rank: 1"));
    assert!(debug.contains("has_data: true"));
    assert!(debug.contains("inputs_len: 1"));
    assert!(debug.contains("extra_roots_len: 0"));
    assert!(debug.contains("constraint_scope_transfer"));
}

#[test]
fn tensor_from_parts_preserves_summary_fields() {
    let input_key = allocate_input_key();
    let mut builder = GraphBuilder::new();
    let val = builder.add_input(input_key.clone());
    builder.set_outputs(vec![val]);
    let graph = Arc::new(builder.build());
    let data = Arc::new(RetainedValue::from_tensor(
        Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap(),
    ));
    let inputs_map = Arc::new(HashMap::from([(input_key.clone(), Arc::clone(&data))]));
    let transfer = ConstraintScopeTransfer::with_new(nonempty_constraint_scope(11), []);
    let parts = TracedTensorParts {
        rank: 1,
        dtype: DType::F64,
        graph,
        val,
        data: Some(data),
        shape_hint: Some(vec![SymDim::from(1)]),
        inputs_map,
        leaf_metas: Arc::new(HashMap::new()),
        extra_roots: Vec::new(),
        checkpoint_chain: None,
        metadata_scopes: Vec::new(),
        constraint_scope_transfer: transfer,
    };

    let tensor = tensor_from_parts(parts);

    assert_eq!(tensor.rank, 1);
    assert_eq!(tensor.dtype, DType::F64);
    assert!(matches!(
        tensor.graph.values()[tensor.val].key,
        ValueKey::Input(_)
    ));
    assert_eq!(tensor.constraint_scopes.materialize().len(), 1);
}

#[test]
fn nested_empty_constraint_transfers_keep_zero_scopes_and_linear_chain_depth() {
    let leaf = ConstraintScopeTransfer::empty();
    let mut transfer = leaf.clone();
    for _ in 0..512 {
        transfer = ConstraintScopeTransfer::with_new(
            ShapeConstraintScope::default(),
            [&transfer, &leaf, &leaf],
        );
    }

    let (scope_count, visited_nodes) = transfer.test_scope_and_node_counts();
    assert_eq!(scope_count, 0);
    assert_eq!(visited_nodes, 513);
}

#[test]
fn nested_constraint_transfers_grow_scopes_and_nodes_linearly() {
    let depth = 512;
    let mut transfer = ConstraintScopeTransfer::empty();
    for id in 0..depth {
        transfer = ConstraintScopeTransfer::with_new(nonempty_constraint_scope(id), [&transfer]);
    }

    let (scope_count, visited_nodes) = transfer.test_scope_and_node_counts();
    assert_eq!(scope_count, depth as usize);
    assert_eq!(visited_nodes, depth as usize + 1);
}

#[test]
fn nary_constraint_transfer_deduplicates_shared_history_in_one_walk() {
    let shared = ConstraintScopeTransfer::with_new(nonempty_constraint_scope(1), []);
    let left = ConstraintScopeTransfer::with_new(nonempty_constraint_scope(2), [&shared]);
    let right = ConstraintScopeTransfer::with_new(nonempty_constraint_scope(3), [&shared]);
    let merged = ConstraintScopeTransfer::merge([&left, &right]);

    let (scope_count, visited_nodes) = merged.test_scope_and_node_counts();
    assert_eq!(scope_count, 3);
    assert_eq!(visited_nodes, 4);
}

#[test]
fn bool_ones_tensor_rejects_shape_product_overflow_without_panicking() {
    let result = std::panic::catch_unwind(|| ones_tensor(DType::Bool, vec![usize::MAX, 2]));

    assert!(
        result.is_ok(),
        "ones_tensor must return a typed error, not panic"
    );
    let err = result.unwrap().unwrap_err();

    assert!(matches!(
        err,
        crate::Error::TensorRuntime(tenferro_tensor::Error::Validation {
            op: "ones_tensor",
            ..
        })
    ));
}

#[test]
fn a_retained_pooled_value_reads_back_and_renders_its_container() {
    // A pooled value is retained through its allocation group, read back as a borrowed
    // descriptor view, and its handle renders the container it holds rather than a slot.
    let tensor =
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).expect("shape matches data");
    let retained = RetainedValue::from_tensor(tensor);

    assert_eq!(retained.dtype(), DType::F64);
    assert_eq!(retained.shape(), &[2]);
    let rendered = format!("{retained:?}");
    assert!(rendered.contains("RetainedValue"), "{rendered}");
    assert!(rendered.contains("container"), "{rendered}");

    match retained.tensor_read().expect("a pooled read") {
        tenferro_tensor::TensorRead::View(view) => {
            assert_eq!(view.as_slice::<f64>().expect("slice"), &[1.0, 2.0]);
        }
        tenferro_tensor::TensorRead::Tensor(_) => {
            panic!("a pooled value is read as a borrowed descriptor view");
        }
    }
}

#[test]
fn ones_tensor_fills_every_core_dtype_and_rejects_external_scalars() {
    use tenferro_tensor::DType;

    assert_eq!(
        ones_tensor(DType::F32, vec![2])
            .unwrap()
            .as_slice::<f32>()
            .unwrap(),
        &[1.0, 1.0]
    );
    assert_eq!(
        ones_tensor(DType::F64, vec![2])
            .unwrap()
            .as_slice::<f64>()
            .unwrap(),
        &[1.0, 1.0]
    );
    assert_eq!(
        ones_tensor(DType::I32, vec![2])
            .unwrap()
            .as_slice::<i32>()
            .unwrap(),
        &[1, 1]
    );
    assert_eq!(
        ones_tensor(DType::I64, vec![2])
            .unwrap()
            .as_slice::<i64>()
            .unwrap(),
        &[1, 1]
    );
    assert_eq!(
        ones_tensor(DType::Bool, vec![2])
            .unwrap()
            .as_slice::<bool>()
            .unwrap(),
        &[true, true]
    );
    assert_eq!(
        ones_tensor(DType::C32, vec![1])
            .unwrap()
            .as_slice::<tenferro_tensor::Complex32>()
            .unwrap(),
        &[tenferro_tensor::Complex32::new(1.0, 0.0)]
    );
    assert_eq!(
        ones_tensor(DType::C64, vec![1])
            .unwrap()
            .as_slice::<tenferro_tensor::Complex64>()
            .unwrap(),
        &[tenferro_tensor::Complex64::new(1.0, 0.0)]
    );

    // An externally defined scalar is caller-owned; the runtime refuses to
    // fabricate a core identity tensor for it instead of guessing one.
    let external = DType::External(std::any::TypeId::of::<f64>());
    let error = ones_tensor(external, vec![1]).expect_err("external dtype must be rejected");
    assert!(
        matches!(error, crate::Error::Validation { .. }),
        "expected a typed validation error, got {error:?}"
    );
}

#[test]
fn leaf_and_linear_input_keys_reject_computed_values() {
    use super::{leaf_input_key, linear_input_key};
    use crate::TracedTensor;

    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let computed = x.neg().unwrap();
    let error = leaf_input_key(&computed).expect_err("a computed value has no leaf input key");
    assert!(
        matches!(error, crate::Error::Validation { .. }),
        "expected a typed validation error, got {error:?}"
    );
    assert!(leaf_input_key(&x).is_ok());

    let graph = computed.graph();
    let output = graph.outputs()[0];
    let error = linear_input_key(graph, output).expect_err("an op output is not a linear input");
    assert!(
        matches!(error, crate::Error::Validation { .. }),
        "expected a typed validation error, got {error:?}"
    );
}

#[test]
fn constraint_scope_transfer_reports_whether_it_carries_scopes() {
    assert!(ConstraintScopeTransfer::empty().is_empty());
    assert!(!ConstraintScopeTransfer::with_new(nonempty_constraint_scope(21), []).is_empty());
}
