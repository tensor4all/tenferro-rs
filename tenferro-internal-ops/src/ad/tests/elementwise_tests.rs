//! Unit tests for elementwise AD helper rules.

use computegraph::graph::GraphBuilder;
use computegraph::types::{OperationRole, ValueKey, ValueRef};
use tenferro_tensor::{CompareDir, DType};

use crate::ad::context::ShapeGuardContext;
use crate::input_key::TensorInputKey;
use crate::std_tensor_op::StdTensorOp;

fn tensor_input(id: u64) -> TensorInputKey {
    TensorInputKey::User { id }
}

fn input_key(id: u64) -> ValueKey<StdTensorOp> {
    ValueKey::Input(tensor_input(id))
}

#[test]
fn zero_like_covers_scalar_and_broadcasted_dtype_paths() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let anchor = builder.add_input(tensor_input(1));

    let dtype_cases = [
        (DType::F32, 4),
        (DType::F64, 8),
        (DType::I64, 8),
        (DType::C32, 8),
        (DType::C64, 16),
    ];
    let mut scalar_zeros = Vec::new();
    for (dtype, byte_len) in dtype_cases {
        let zero =
            super::super::zeros::build_zero_like(&mut builder, dtype, ValueRef::Local(anchor), 0);
        scalar_zeros.push((zero, dtype, byte_len));
    }

    let vector_zero =
        super::super::zeros::build_zero_like(&mut builder, DType::F64, ValueRef::Local(anchor), 2);
    let graph = builder.build();

    for (zero, dtype, byte_len) in scalar_zeros {
        let (op_id, _) = graph.values()[zero]
            .producer
            .expect("zero scalar must be produced by a constant op");
        let op = &graph.operations()[op_id];
        match &op.operation {
            StdTensorOp::Constant { dtype: got, bytes } => {
                assert_eq!(*got, dtype);
                assert_eq!(bytes.len(), byte_len);
                assert!(bytes.iter().all(|byte| *byte == 0));
            }
            other => panic!("expected constant zero, got {other:?}"),
        }
    }

    let (broadcast, _) = graph.values()[vector_zero]
        .producer
        .expect("ranked zero must be produced by broadcast");
    let op = &graph.operations()[broadcast];
    assert_eq!(
        op.operation,
        StdTensorOp::BroadcastInDim {
            shape: vec![
                crate::dim_expr::DimExpr::InputDim {
                    input_idx: 1,
                    axis: 0
                },
                crate::dim_expr::DimExpr::InputDim {
                    input_idx: 1,
                    axis: 1
                },
            ],
            dims: vec![],
        }
    );
    assert_eq!(op.inputs[1], ValueRef::Local(anchor));
}

#[test]
fn linearize_elementwise_inactive_inputs_return_none_without_ops() {
    let mut ctx = ShapeGuardContext::default();
    let keys = vec![input_key(1), input_key(2), input_key(3)];

    for op in [
        StdTensorOp::Div,
        StdTensorOp::Abs,
        StdTensorOp::Sign,
        StdTensorOp::Maximum,
        StdTensorOp::Minimum,
        StdTensorOp::Select,
        StdTensorOp::Clamp,
    ] {
        let mut builder = GraphBuilder::<StdTensorOp>::new();
        let result = op.jvp_rule(
            &mut builder,
            &keys,
            &[input_key(4)],
            &[None, None, None],
            &mut ctx,
        );
        assert_eq!(result, vec![None], "{op:?}");
        assert!(
            builder.build().operations().is_empty(),
            "inactive {op:?} should not emit linearized ops"
        );
    }
}

#[test]
fn linearize_div_with_two_active_inputs_sums_both_terms() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let dx = builder.add_input(tensor_input(10));
    let dy = builder.add_input(tensor_input(11));
    let mut ctx = ShapeGuardContext::default();
    let op = StdTensorOp::Div;

    let result = op.jvp_rule(
        &mut builder,
        &[input_key(12), input_key(13)],
        &[input_key(14)],
        &[Some(dx), Some(dy)],
        &mut ctx,
    );

    assert!(result[0].is_some());
    let graph = builder.build();
    assert_eq!(
        graph.operations().last().map(|op| &op.operation),
        Some(&StdTensorOp::Add)
    );
}

#[test]
fn linearize_extrema_and_select_emit_zero_fill_for_one_sided_tangents() {
    let mut ctx = ShapeGuardContext::default();

    let mut max_builder = GraphBuilder::<StdTensorOp>::new();
    let dx = max_builder.add_input(tensor_input(20));
    let result = StdTensorOp::Maximum.jvp_rule(
        &mut max_builder,
        &[input_key(21), input_key(22)],
        &[],
        &[Some(dx), None],
        &mut ctx,
    );
    assert!(result[0].is_some());
    let graph = max_builder.build();
    assert_eq!(
        graph.operations()[0].operation,
        StdTensorOp::Compare(CompareDir::Ge)
    );
    assert!(graph
        .operations()
        .iter()
        .any(|op| op.operation == StdTensorOp::Select));

    let mut min_builder = GraphBuilder::<StdTensorOp>::new();
    let dy = min_builder.add_input(tensor_input(23));
    let result = StdTensorOp::Minimum.jvp_rule(
        &mut min_builder,
        &[input_key(24), input_key(25)],
        &[],
        &[None, Some(dy)],
        &mut ctx,
    );
    assert!(result[0].is_some());
    let graph = min_builder.build();
    assert_eq!(
        graph.operations()[0].operation,
        StdTensorOp::Compare(CompareDir::Le)
    );
    assert!(graph
        .operations()
        .iter()
        .any(|op| op.operation == StdTensorOp::Select));
}

#[test]
fn linearize_clamp_builds_nested_selects_for_active_bounds() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let dx = builder.add_input(tensor_input(30));
    let dlower = builder.add_input(tensor_input(31));
    let dupper = builder.add_input(tensor_input(32));
    let mut ctx = ShapeGuardContext::default();

    let result = StdTensorOp::Clamp.jvp_rule(
        &mut builder,
        &[input_key(33), input_key(34), input_key(35)],
        &[],
        &[Some(dx), Some(dlower), Some(dupper)],
        &mut ctx,
    );

    assert!(result[0].is_some());
    let graph = builder.build();
    assert!(
        graph
            .operations()
            .iter()
            .filter(|op| op.operation == StdTensorOp::Select)
            .count()
            >= 2
    );
}

#[test]
fn transpose_elementwise_handles_missing_or_inactive_cotangents() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let inputs = vec![
        ValueRef::External(input_key(40)),
        ValueRef::External(input_key(41)),
    ];

    assert_eq!(
        StdTensorOp::Div.transpose_rule(
            &mut builder,
            &[None],
            &inputs,
            &OperationRole::Primary,
            &mut ctx
        ),
        vec![None, None]
    );
    let abs_ct = builder.add_input(tensor_input(42));
    assert_eq!(
        StdTensorOp::Abs.transpose_rule(
            &mut builder,
            &[Some(abs_ct)],
            &[inputs[0].clone()],
            &OperationRole::Primary,
            &mut ctx,
        ),
        vec![None]
    );
    assert_eq!(
        StdTensorOp::Maximum.transpose_rule(
            &mut builder,
            &[None],
            &inputs,
            &OperationRole::Linearized {
                active_mask: vec![true, true],
            },
            &mut ctx,
        ),
        vec![None, None]
    );
}

#[test]
fn transpose_select_splits_cotangent_only_to_active_value_inputs() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let ct = builder.add_input(tensor_input(50));
    let mut ctx = ShapeGuardContext::default();
    let inputs = vec![
        ValueRef::External(input_key(51)),
        ValueRef::External(input_key(52)),
        ValueRef::External(input_key(53)),
    ];

    let result = StdTensorOp::Select.transpose_rule(
        &mut builder,
        &[Some(ct)],
        &inputs,
        &OperationRole::Linearized {
            active_mask: vec![false, true, false],
        },
        &mut ctx,
    );

    assert_eq!(result[0], None);
    assert!(result[1].is_some());
    assert_eq!(result[2], None);
    let graph = builder.build();
    assert!(graph
        .operations()
        .iter()
        .any(|op| op.operation == StdTensorOp::Select));
}

#[test]
fn transpose_clamp_covers_lower_only_and_inner_paths() {
    let mut ctx = ShapeGuardContext::default();
    let inputs = vec![
        ValueRef::External(input_key(60)),
        ValueRef::External(input_key(61)),
        ValueRef::External(input_key(62)),
    ];

    let mut lower_builder = GraphBuilder::<StdTensorOp>::new();
    let lower_ct = lower_builder.add_input(tensor_input(63));
    let result = StdTensorOp::Clamp.transpose_rule(
        &mut lower_builder,
        &[Some(lower_ct)],
        &inputs,
        &OperationRole::Linearized {
            active_mask: vec![false, true, false],
        },
        &mut ctx,
    );
    assert_eq!(result[0], None);
    assert!(result[1].is_some());
    assert_eq!(result[2], None);

    let mut full_builder = GraphBuilder::<StdTensorOp>::new();
    let full_ct = full_builder.add_input(tensor_input(64));
    let result = StdTensorOp::Clamp.transpose_rule(
        &mut full_builder,
        &[Some(full_ct)],
        &inputs,
        &OperationRole::Linearized {
            active_mask: vec![true, true, true],
        },
        &mut ctx,
    );
    assert!(result.iter().all(Option::is_some));
}
