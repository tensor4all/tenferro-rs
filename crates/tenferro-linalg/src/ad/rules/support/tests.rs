use std::collections::BTreeMap;

use computegraph::graph::{Graph, GraphBuilder};
use computegraph::types::ValueRef;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::DType;

use super::{
    broadcast_scalar_constant_with_dtype, identity_matrix_fixed, leading_column_selector_symbolic,
    one_like_fixed,
};

fn op_kind_name(op: &StdTensorOp) -> &'static str {
    match op {
        StdTensorOp::Constant { .. } => "Constant",
        StdTensorOp::BroadcastInDim { .. } => "BroadcastInDim",
        StdTensorOp::EmbedDiag { .. } => "EmbedDiag",
        StdTensorOp::Exp => "Exp",
        StdTensorOp::Log => "Log",
        StdTensorOp::Sin => "Sin",
        StdTensorOp::Cos => "Cos",
        StdTensorOp::Tanh => "Tanh",
        StdTensorOp::Sqrt => "Sqrt",
        StdTensorOp::Rsqrt => "Rsqrt",
        StdTensorOp::Expm1 => "Expm1",
        StdTensorOp::Log1p => "Log1p",
        _ => "Other",
    }
}

fn op_kind_histogram(graph: &Graph<StdTensorOp>) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for op in graph.operations() {
        *counts.entry(op_kind_name(&op.operation)).or_insert(0) += 1;
    }
    counts
}

fn assert_no_analytic_constant_shortcuts(graph: &computegraph::graph::Graph<StdTensorOp>) {
    for op in graph.operations() {
        assert!(
            !matches!(
                op.operation,
                StdTensorOp::Exp
                    | StdTensorOp::Log
                    | StdTensorOp::Sin
                    | StdTensorOp::Cos
                    | StdTensorOp::Tanh
                    | StdTensorOp::Sqrt
                    | StdTensorOp::Rsqrt
                    | StdTensorOp::Expm1
                    | StdTensorOp::Log1p
            ),
            "constant and identity rule helpers must use semantic constant emission, not analytic shortcuts; saw {:?}",
            op.operation
        );
    }
}

#[test]
fn one_like_fixed_uses_semantic_constant_not_analytic_shortcut() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let anchor = builder.add_input(TensorInputKey::User { id: 2 });

    let one = one_like_fixed(&mut builder, DType::F64, ValueRef::Local(anchor), 2);
    let graph = builder.build();

    assert!(graph.values()[one].producer.is_some());
    assert_no_analytic_constant_shortcuts(&graph);
    assert_eq!(
        op_kind_histogram(&graph),
        BTreeMap::from([("BroadcastInDim", 1), ("Constant", 1)])
    );
}

#[test]
fn identity_matrix_fixed_uses_semantic_constant_not_analytic_shortcut() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let anchor = builder.add_input(TensorInputKey::User { id: 1 });

    let identity = identity_matrix_fixed(&mut builder, DType::F64, 2, &[], ValueRef::Local(anchor));
    let graph = builder.build();

    assert!(graph.values()[identity].producer.is_some());
    assert_no_analytic_constant_shortcuts(&graph);
    assert_eq!(
        op_kind_histogram(&graph),
        BTreeMap::from([("BroadcastInDim", 1), ("Constant", 1), ("EmbedDiag", 1)])
    );
}

#[test]
fn symbolic_leading_selector_broadcasts_scalar_constants_once() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let anchor = builder.add_input(TensorInputKey::User { id: 3 });

    let selector = leading_column_selector_symbolic(
        &mut builder,
        DType::F64,
        DimExpr::InputDim {
            input_idx: 0,
            axis: 1,
        },
        DimExpr::InputDim {
            input_idx: 0,
            axis: 0,
        },
        &[],
        ValueRef::Local(anchor),
    );
    let graph = builder.build();

    assert!(graph.values()[selector].producer.is_some());
    assert_eq!(
        op_kind_histogram(&graph).get("BroadcastInDim"),
        Some(&2),
        "selector constants must remain scalar until their one required broadcast"
    );
}

#[test]
fn scale_constant_helpers_reject_integer_and_bool_without_emitting_any_op() {
    for dtype in [DType::I32, DType::I64, DType::Bool] {
        let mut builder = GraphBuilder::<StdTensorOp>::new();
        let err = broadcast_scalar_constant_with_dtype(&mut builder, 0.5, vec![], dtype, "svd")
            .expect_err("integer/bool scale constants must be rejected");
        assert!(
            matches!(
                err,
                tenferro_ops::ad::ADRuleError::InvalidInput {
                    rule: tenferro_ops::ad::ADRuleKind::Jvp,
                    ..
                }
            ),
            "expected InvalidInput for {dtype:?}, got {err:?}"
        );
        let graph = builder.build();
        assert!(
            graph.operations().is_empty(),
            "rejected dtype must not publish any constant op, got {} ops for {dtype:?}",
            graph.operations().len()
        );
    }
}
