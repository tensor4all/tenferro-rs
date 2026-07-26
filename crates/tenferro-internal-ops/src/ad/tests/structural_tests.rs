//! Unit tests for structural AD helper rules.

use computegraph::graph::GraphBuilder;
use computegraph::types::{OperationRole, ValueKey, ValueRef};
use tenferro_tensor::DType;

use crate::ad::context::ShapeGuardContext;
use crate::ad::{ADRuleError, PrimitiveTransposeInput};
use crate::dim_expr::DimExpr;
use crate::input_key::TensorInputKey;
use crate::std_tensor_op::StdTensorOp;
use crate::{ShapeExtent, SymDim, TensorMeta};

fn tensor_input(id: u64) -> TensorInputKey {
    TensorInputKey::User { id }
}

fn input_key(id: u64) -> ValueKey<StdTensorOp> {
    ValueKey::Input(tensor_input(id))
}

fn meta(shape: &[usize]) -> TensorMeta {
    TensorMeta::exact(
        DType::F64,
        shape.iter().copied().map(SymDim::from).collect(),
    )
}

fn linear_mode(active_mask: &[bool]) -> OperationRole {
    OperationRole::Linearized {
        active_mask: active_mask.to_vec(),
    }
}

#[test]
fn linearize_reshape_reuses_dynamic_shape_sources_as_inactive_inputs() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let data_key = input_key(1);
    let shape_key = input_key(2);
    let data_tangent = builder.add_input(tensor_input(3));
    let op = StdTensorOp::Reshape {
        to_shape: vec![
            DimExpr::InputDim {
                input_idx: 1,
                axis: 0,
            },
            DimExpr::Const(2),
        ],
    };

    let result = op
        .jvp_rule(
            &mut builder,
            &[data_key, shape_key.clone()],
            &[],
            &[Some(data_tangent), None],
            &mut ctx,
        )
        .unwrap();

    assert_eq!(result.len(), 1);
    let tangent_out = result[0].expect("reshape tangent output must be active");
    let graph = builder.build();
    assert_eq!(graph.operations().len(), 1);
    let reshape = &graph.operations()[0];
    assert_eq!(reshape.operation, op);
    assert_eq!(reshape.inputs[0], ValueRef::Local(data_tangent));
    assert_eq!(reshape.inputs[1], ValueRef::External(shape_key));
    assert_eq!(
        reshape.role,
        OperationRole::Linearized {
            active_mask: vec![true, false],
        }
    );
    assert!(reshape.outputs.contains(&tangent_out));
}

#[test]
fn transpose_reshape_returns_none_for_dynamic_shape_sources() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input(10));
    let data_key = input_key(11);
    let shape_key = input_key(12);
    let inputs = vec![
        ValueRef::External(data_key.clone()),
        ValueRef::External(shape_key.clone()),
    ];
    ctx.insert_metadata(data_key, meta(&[2, 3]));
    ctx.insert_metadata(shape_key, meta(&[3]));

    let result = StdTensorOp::Reshape {
        to_shape: vec![
            DimExpr::InputDim {
                input_idx: 1,
                axis: 0,
            },
            DimExpr::Const(2),
        ],
    }
    .transpose_rule(
        &mut builder,
        &[Some(cotangent)],
        &inputs,
        &linear_mode(&[true, false]),
        &mut ctx,
    )
    .unwrap();

    assert_eq!(result.len(), 2);
    assert!(result[0].is_some());
    assert_eq!(result[1], None);

    let graph = builder.build();
    assert_eq!(graph.operations().len(), 1);
    let reshape = &graph.operations()[0];
    assert_eq!(reshape.inputs[0], ValueRef::Local(cotangent));
    assert_eq!(
        reshape.operation,
        StdTensorOp::Reshape {
            to_shape: vec![DimExpr::Const(2), DimExpr::Const(3)],
        }
    );
    assert_eq!(
        reshape.role,
        OperationRole::Linearized {
            active_mask: vec![true],
        }
    );
}

#[test]
fn transpose_reshape_accepts_upper_bound_input_metadata() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input(15));
    let data_key = input_key(16);
    let inputs = vec![ValueRef::External(data_key.clone())];
    ctx.insert_metadata(
        data_key,
        TensorMeta::with_extents(
            DType::F64,
            vec![
                ShapeExtent::upper_bound(SymDim::from(4usize)),
                ShapeExtent::upper_bound(SymDim::from(5usize)),
            ],
        ),
    );

    let result = StdTensorOp::Reshape { to_shape: vec![] }
        .transpose_rule(
            &mut builder,
            &[Some(cotangent)],
            &inputs,
            &linear_mode(&[true]),
            &mut ctx,
        )
        .unwrap();

    assert_eq!(result.len(), 1);
    assert!(result[0].is_some());

    let graph = builder.build();
    assert_eq!(graph.operations().len(), 1);
    let reshape = &graph.operations()[0];
    assert_eq!(
        reshape.operation,
        StdTensorOp::Reshape {
            to_shape: DimExpr::input_shape(1, 2),
        }
    );
    assert_eq!(reshape.inputs[0], ValueRef::Local(cotangent));
    assert_eq!(reshape.inputs[1], inputs[0]);
    assert_eq!(
        reshape.role,
        OperationRole::Linearized {
            active_mask: vec![true, false],
        }
    );
}

#[test]
fn transpose_pad_to_match_rejects_axis_outside_input_metadata() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input(18));
    let input = input_key(19);
    let reference = input_key(20);
    ctx.insert_metadata(input.clone(), meta(&[3]));

    let err = StdTensorOp::PadToMatch { axis: 1 }
        .transpose_rule(
            &mut builder,
            &[Some(cotangent)],
            &[ValueRef::External(input), ValueRef::External(reference)],
            &linear_mode(&[true, false]),
            &mut ctx,
        )
        .unwrap_err();

    assert!(matches!(
        err,
        ADRuleError::InvalidInput {
            ref op,
            ref message,
            ..
        } if op == "PadToMatch" && message.contains("axis 1")
    ));
}

#[test]
fn linearize_broadcast_reuses_dynamic_shape_sources_as_inactive_inputs() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let data_key = input_key(20);
    let shape_key = input_key(21);
    let data_tangent = builder.add_input(tensor_input(22));
    let op = StdTensorOp::BroadcastInDim {
        shape: vec![
            DimExpr::Const(3),
            DimExpr::InputDim {
                input_idx: 1,
                axis: 0,
            },
        ],
        dims: vec![1],
    };

    let result = op
        .jvp_rule(
            &mut builder,
            &[data_key, shape_key.clone()],
            &[],
            &[Some(data_tangent), None],
            &mut ctx,
        )
        .unwrap();

    assert_eq!(result.len(), 1);
    let tangent_out = result[0].expect("broadcast tangent output must be active");
    let graph = builder.build();
    assert_eq!(graph.operations().len(), 1);
    let broadcast = &graph.operations()[0];
    assert_eq!(broadcast.operation, op);
    assert_eq!(broadcast.inputs[0], ValueRef::Local(data_tangent));
    assert_eq!(broadcast.inputs[1], ValueRef::External(shape_key));
    assert_eq!(
        broadcast.role,
        OperationRole::Linearized {
            active_mask: vec![true, false],
        }
    );
    assert!(broadcast.outputs.contains(&tangent_out));
}

#[test]
fn transpose_broadcast_returns_none_for_dynamic_shape_sources() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input(30));
    let data_key = input_key(31);
    let shape_key = input_key(32);
    let inputs = vec![ValueRef::External(data_key), ValueRef::External(shape_key)];

    let result = StdTensorOp::BroadcastInDim {
        shape: vec![
            DimExpr::Const(3),
            DimExpr::InputDim {
                input_idx: 1,
                axis: 0,
            },
        ],
        dims: vec![1],
    }
    .transpose_rule(
        &mut builder,
        &[Some(cotangent)],
        &inputs,
        &linear_mode(&[true, false]),
        &mut ctx,
    )
    .unwrap();

    assert_eq!(result.len(), 2);
    assert!(result[0].is_some());
    assert_eq!(result[1], None);
}

#[test]
fn transpose_broadcast_propagates_unresolved_local_shape_error() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input(35));
    let data_key = input_key(36);
    ctx.insert_metadata(
        data_key.clone(),
        TensorMeta::with_extents(
            DType::F64,
            vec![ShapeExtent::upper_bound(SymDim::from(1usize))],
        ),
    );
    let inputs = vec![PrimitiveTransposeInput::Linear {
        key: data_key,
        primal: None,
    }];

    let op = StdTensorOp::BroadcastInDim {
        shape: vec![DimExpr::Const(3)],
        dims: vec![0],
    };
    let err = crate::ad::transpose_rule(
        &op,
        &mut builder,
        &[Some(cotangent)],
        &inputs,
        &linear_mode(&[true]),
        &mut ctx,
    )
    .unwrap_err();

    assert!(err.to_string().contains("linear-only"));
}

#[test]
fn transpose_broadcast_reduces_singleton_input_axes() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input(40));
    let data_key = input_key(41);
    let inputs = vec![ValueRef::External(data_key.clone())];
    ctx.insert_metadata(data_key, meta(&[1]));

    let result = StdTensorOp::BroadcastInDim {
        shape: vec![DimExpr::Const(3)],
        dims: vec![0],
    }
    .transpose_rule(
        &mut builder,
        &[Some(cotangent)],
        &inputs,
        &linear_mode(&[true]),
        &mut ctx,
    )
    .unwrap();

    assert_eq!(result.len(), 1);
    let cotangent_in = result[0].expect("broadcast cotangent input must be active");
    let graph = builder.build();
    assert_eq!(graph.operations().len(), 2);
    let reduce = &graph.operations()[0];
    assert_eq!(reduce.operation, StdTensorOp::ReduceSum { axes: vec![0] });
    assert_eq!(reduce.inputs, vec![ValueRef::Local(cotangent)]);
    let reshape = &graph.operations()[1];
    assert_eq!(
        reshape.operation,
        StdTensorOp::Reshape {
            to_shape: vec![DimExpr::Const(1)],
        }
    );
    assert_eq!(reshape.inputs[0], ValueRef::Local(reduce.outputs[0]));
    assert!(reshape.outputs.contains(&cotangent_in));
}

#[test]
fn transpose_broadcast_reduces_upper_bound_singleton_input_axes() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input(45));
    let data_key = input_key(46);
    let inputs = vec![ValueRef::External(data_key.clone())];
    ctx.insert_metadata(
        data_key,
        TensorMeta::with_extents(
            DType::F64,
            vec![ShapeExtent::upper_bound(SymDim::from(1usize))],
        ),
    );

    let result = StdTensorOp::BroadcastInDim {
        shape: vec![DimExpr::Const(3)],
        dims: vec![0],
    }
    .transpose_rule(
        &mut builder,
        &[Some(cotangent)],
        &inputs,
        &linear_mode(&[true]),
        &mut ctx,
    )
    .unwrap();

    assert_eq!(result.len(), 1);
    let cotangent_in = result[0].expect("broadcast cotangent input must be active");
    let graph = builder.build();
    assert_eq!(graph.operations().len(), 2);
    let reduce = &graph.operations()[0];
    assert_eq!(reduce.operation, StdTensorOp::ReduceSum { axes: vec![0] });
    assert_eq!(reduce.inputs, vec![ValueRef::Local(cotangent)]);
    let reshape = &graph.operations()[1];
    assert_eq!(
        reshape.operation,
        StdTensorOp::Reshape {
            to_shape: DimExpr::input_shape(1, 1),
        }
    );
    assert_eq!(reshape.inputs[0], ValueRef::Local(reduce.outputs[0]));
    assert_eq!(reshape.inputs[1], inputs[0]);
    assert!(reshape.outputs.contains(&cotangent_in));
}

#[test]
fn transpose_broadcast_restores_non_monotonic_dimension_order() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input(50));
    let data_key = input_key(51);
    let inputs = vec![ValueRef::External(data_key.clone())];
    ctx.insert_metadata(data_key, meta(&[2, 3]));

    let result = StdTensorOp::BroadcastInDim {
        shape: vec![
            DimExpr::InputDim {
                input_idx: 0,
                axis: 1,
            },
            DimExpr::InputDim {
                input_idx: 0,
                axis: 0,
            },
        ],
        dims: vec![1, 0],
    }
    .transpose_rule(
        &mut builder,
        &[Some(cotangent)],
        &inputs,
        &linear_mode(&[true]),
        &mut ctx,
    )
    .unwrap();

    assert_eq!(result.len(), 1);
    let cotangent_in = result[0].expect("broadcast cotangent input must be active");
    let graph = builder.build();
    assert_eq!(graph.operations().len(), 1);
    let transpose = &graph.operations()[0];
    assert_eq!(
        transpose.operation,
        StdTensorOp::Transpose { perm: vec![1, 0] }
    );
    assert_eq!(transpose.inputs, vec![ValueRef::Local(cotangent)]);
    assert!(transpose.outputs.contains(&cotangent_in));
}

#[test]
fn transpose_embed_diag_accepts_upper_bound_input_metadata_for_rank_only_perm() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input(55));
    let data_key = input_key(56);
    let inputs = vec![ValueRef::External(data_key.clone())];
    ctx.insert_metadata(
        data_key,
        TensorMeta::with_extents(
            DType::F64,
            vec![
                ShapeExtent::upper_bound(SymDim::from(4usize)),
                ShapeExtent::upper_bound(SymDim::from(5usize)),
            ],
        ),
    );

    let result = StdTensorOp::EmbedDiag {
        axis_a: 1,
        axis_b: 0,
    }
    .transpose_rule(
        &mut builder,
        &[Some(cotangent)],
        &inputs,
        &linear_mode(&[true]),
        &mut ctx,
    )
    .unwrap();

    assert_eq!(result.len(), 1);
    assert!(result[0].is_some());

    let graph = builder.build();
    assert_eq!(graph.operations().len(), 2);
    assert_eq!(
        graph.operations()[1].operation,
        StdTensorOp::Transpose { perm: vec![1, 0] }
    );
}

#[test]
fn transpose_extract_diag_uses_runtime_shape_source_for_symbolic_padding() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input(61));
    let data_key = input_key(62);
    ctx.insert_metadata(
        data_key.clone(),
        TensorMeta::exact(
            DType::F64,
            vec![SymDim::tensor_axis(900, 0), SymDim::from(5usize)],
        ),
    );
    ctx.insert_shape_source(900, data_key.clone());

    let result = StdTensorOp::ExtractDiag {
        axis_a: 0,
        axis_b: 1,
    }
    .transpose_rule(
        &mut builder,
        &[Some(cotangent)],
        &[ValueRef::External(data_key.clone())],
        &linear_mode(&[true]),
        &mut ctx,
    )
    .unwrap();

    assert_eq!(result.len(), 1);
    let cotangent_in = result[0].expect("extract-diag input cotangent must be active");
    let graph = builder.build();
    assert_eq!(graph.operations().len(), 3);
    assert_eq!(
        graph.operations()[0].operation,
        StdTensorOp::EmbedDiag {
            axis_a: 0,
            axis_b: 1,
        }
    );
    assert_eq!(
        graph.operations()[1].operation,
        StdTensorOp::PadToMatch { axis: 0 }
    );
    assert_eq!(
        graph.operations()[1].inputs,
        vec![
            ValueRef::Local(graph.operations()[0].outputs[0]),
            ValueRef::External(data_key.clone())
        ]
    );
    assert_eq!(
        graph.operations()[2].operation,
        StdTensorOp::PadToMatch { axis: 1 }
    );
    assert_eq!(
        graph.operations()[2].inputs,
        vec![
            ValueRef::Local(graph.operations()[1].outputs[0]),
            ValueRef::External(data_key)
        ]
    );
    assert!(graph.operations()[2].outputs.contains(&cotangent_in));
}

#[test]
fn transpose_extract_diag_reports_static_padding_axis_errors() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input(63));
    let data_key = input_key(64);
    ctx.insert_metadata(data_key.clone(), meta(&[2, 2]));

    let err = StdTensorOp::ExtractDiag {
        axis_a: 2,
        axis_b: 0,
    }
    .transpose_rule(
        &mut builder,
        &[Some(cotangent)],
        &[ValueRef::External(data_key)],
        &linear_mode(&[true]),
        &mut ctx,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        ADRuleError::InvalidInput {
            ref op,
            ref message,
            ..
        } if op == "ExtractDiag" && message.contains("out of bounds")
    ));
}

#[test]
fn transpose_diagonal_rules_respect_inactive_data_input() {
    let mut extract_builder = GraphBuilder::<StdTensorOp>::new();
    let mut extract_ctx = ShapeGuardContext::default();
    let extract_ct = extract_builder.add_input(tensor_input(57));
    let extract_input = input_key(58);
    extract_ctx.insert_metadata(extract_input.clone(), meta(&[2, 2]));

    let extract_result = StdTensorOp::ExtractDiag {
        axis_a: 0,
        axis_b: 1,
    }
    .transpose_rule(
        &mut extract_builder,
        &[Some(extract_ct)],
        &[ValueRef::External(extract_input)],
        &linear_mode(&[false]),
        &mut extract_ctx,
    )
    .unwrap();

    assert_eq!(extract_result, vec![None]);
    assert!(extract_builder.build().operations().is_empty());

    let mut embed_builder = GraphBuilder::<StdTensorOp>::new();
    let mut embed_ctx = ShapeGuardContext::default();
    let embed_ct = embed_builder.add_input(tensor_input(59));
    let embed_input = input_key(60);
    embed_ctx.insert_metadata(embed_input.clone(), meta(&[2]));

    let embed_result = StdTensorOp::EmbedDiag {
        axis_a: 0,
        axis_b: 1,
    }
    .transpose_rule(
        &mut embed_builder,
        &[Some(embed_ct)],
        &[ValueRef::External(embed_input)],
        &linear_mode(&[false]),
        &mut embed_ctx,
    )
    .unwrap();

    assert_eq!(embed_result, vec![None]);
    assert!(embed_builder.build().operations().is_empty());
}

#[test]
fn transpose_concatenate_returns_none_for_symbolic_concat_axis() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input(60));
    let input_key = input_key(61);
    let inputs = vec![ValueRef::External(input_key.clone())];
    ctx.insert_metadata(
        input_key,
        TensorMeta::exact(
            DType::F64,
            vec![SymDim::tensor_axis(100, 0), SymDim::from(2usize)],
        ),
    );

    let result = StdTensorOp::Concatenate {
        axis: 0,
        input_count: 1,
    }
    .transpose_rule(
        &mut builder,
        &[Some(cotangent)],
        &inputs,
        &linear_mode(&[true]),
        &mut ctx,
    )
    .unwrap();

    assert_eq!(result, vec![None]);
    assert!(builder.build().operations().is_empty());
}
