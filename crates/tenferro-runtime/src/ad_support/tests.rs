use std::collections::HashMap;
use std::sync::Arc;

use computegraph::graph::GraphBuilder;
use computegraph::types::ValueKey;
use tenferro_tensor::{DType, Tensor};

use crate::sym_dim::SymDim;

use super::{
    allocate_input_key, constraint_scopes, constraint_scopes_with_new, ones_tensor,
    push_constraint_scope, tensor_from_parts, ShapeConstraintScope, TracedTensorParts,
};

#[test]
fn traced_tensor_parts_debug_summarizes_without_graph_payload() {
    let input_key = allocate_input_key();
    let mut builder = GraphBuilder::new();
    let val = builder.add_input(input_key.clone());
    builder.set_outputs(vec![val]);
    let graph = Arc::new(builder.build());
    let data = Arc::new(Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap());
    let inputs_map = Arc::new(HashMap::from([(input_key, Arc::clone(&data))]));
    let parts = TracedTensorParts {
        rank: 1,
        dtype: DType::F64,
        graph,
        val,
        data: Some(data),
        shape_hint: Some(vec![SymDim::from(1)]),
        inputs_map,
        extra_roots: Vec::new(),
        checkpoint_chain: None,
        metadata_scopes: Vec::new(),
        constraint_scopes: Vec::new(),
    };

    let debug = format!("{parts:?}");

    assert!(debug.contains("TracedTensorParts"));
    assert!(debug.contains("rank: 1"));
    assert!(debug.contains("has_data: true"));
    assert!(debug.contains("inputs_len: 1"));
    assert!(debug.contains("extra_roots_len: 0"));
    assert!(debug.contains("constraint_scopes_len: 0"));
}

#[test]
fn tensor_from_parts_preserves_summary_fields() {
    let input_key = allocate_input_key();
    let mut builder = GraphBuilder::new();
    let val = builder.add_input(input_key.clone());
    builder.set_outputs(vec![val]);
    let graph = Arc::new(builder.build());
    let data = Arc::new(Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap());
    let inputs_map = Arc::new(HashMap::from([(input_key.clone(), Arc::clone(&data))]));
    let constraint_scope = Arc::new(ShapeConstraintScope::default());
    let parts = TracedTensorParts {
        rank: 1,
        dtype: DType::F64,
        graph,
        val,
        data: Some(data),
        shape_hint: Some(vec![SymDim::from(1)]),
        inputs_map,
        extra_roots: Vec::new(),
        checkpoint_chain: None,
        metadata_scopes: Vec::new(),
        constraint_scopes: vec![Arc::clone(&constraint_scope)],
    };

    let tensor = tensor_from_parts(parts);

    assert_eq!(tensor.rank, 1);
    assert_eq!(tensor.dtype, DType::F64);
    assert!(matches!(
        tensor.graph.values()[tensor.val].key,
        ValueKey::Input(_)
    ));
    assert_eq!(constraint_scopes(&tensor).len(), 1);
    assert!(Arc::ptr_eq(
        &constraint_scopes(&tensor)[0],
        &constraint_scope
    ));
}

#[test]
fn constraint_scope_helpers_preserve_order_and_pointer_deduplicate() {
    let new_scope = ShapeConstraintScope::default();
    let inherited_first = Arc::new(ShapeConstraintScope::default());
    let inherited_second = Arc::new(ShapeConstraintScope::default());
    let inherited = [
        Arc::clone(&inherited_first),
        Arc::clone(&inherited_second),
        Arc::clone(&inherited_first),
    ];

    let mut scopes = constraint_scopes_with_new(new_scope, [&inherited[..], &inherited[..]]);

    assert_eq!(scopes.len(), 3);
    assert!(Arc::ptr_eq(&scopes[1], &inherited_first));
    assert!(Arc::ptr_eq(&scopes[2], &inherited_second));
    push_constraint_scope(&mut scopes, Arc::clone(&inherited_second));
    assert_eq!(scopes.len(), 3);
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
        crate::Error::TensorRuntime(tenferro_tensor::Error::InvalidConfig {
            op: "ones_tensor",
            ..
        })
    ));
}
