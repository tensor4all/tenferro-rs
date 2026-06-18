use std::collections::HashMap;
use std::sync::Arc;

use computegraph::graph::GraphBuilder;
use computegraph::types::ValueKey;
use tenferro_tensor::{DType, Tensor};

use crate::sym_dim::SymDim;

use super::{allocate_input_key, tensor_from_parts, TracedTensorParts};

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
    };

    let debug = format!("{parts:?}");

    assert!(debug.contains("TracedTensorParts"));
    assert!(debug.contains("rank: 1"));
    assert!(debug.contains("has_data: true"));
    assert!(debug.contains("inputs_len: 1"));
    assert!(debug.contains("extra_roots_len: 0"));
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
    };

    let tensor = tensor_from_parts(parts);

    assert_eq!(tensor.rank, 1);
    assert_eq!(tensor.dtype, DType::F64);
    assert!(matches!(
        tensor.graph.values()[tensor.val].key,
        ValueKey::Input(_)
    ));
}
