use computegraph::graph::GraphBuilder;
use computegraph::types::{ValueKey, ValueRef};

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;

use crate::builder::{build_einsum_graph, build_einsum_graph_dim_expr};
use crate::planning::tree::ContractionTree;
use crate::syntax::subscripts::Subscripts;
use crate::{Error, Result};

fn input_key(id: u64) -> TensorInputKey {
    TensorInputKey::User { id }
}

fn make_tree(notation: &str, shapes: &[&[usize]]) -> ContractionTree {
    let subscripts = Subscripts::parse(notation).expect("bad notation");
    ContractionTree::optimize(&subscripts, shapes).expect("optimize failed")
}

#[test]
fn builder_axis_vec_stays_inline_for_common_rank() {
    let mut axes = crate::builder::AxisVec::new();
    axes.extend(0..4);

    assert!(!axes.spilled());
}

fn unwrap_local(result: Result<ValueRef<StdTensorOp>>) -> usize {
    match result.expect("builder should succeed") {
        ValueRef::Local(id) => id,
        ValueRef::External(_) => panic!("expected local"),
    }
}

#[test]
fn builder_reports_missing_output_label_instead_of_panicking() {
    let mut tree = make_tree("i->i", &[&[3]]);
    tree.subscripts.output = vec![b'j' as u32];
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));

    let err =
        build_einsum_graph(&mut builder, &tree, &[ValueRef::Local(a)], &[vec![3]]).unwrap_err();

    assert!(matches!(
        err,
        Error::InvalidArgument(message)
            if message.contains("missing") && message.contains("label")
    ));
}

#[test]
fn graph_identity_external_input_is_localized() {
    let tree = make_tree("ij->ij", &[&[2, 3]]);
    let mut builder = GraphBuilder::<StdTensorOp>::new();

    let result = build_einsum_graph_dim_expr(
        &mut builder,
        &tree,
        &[ValueRef::External(ValueKey::Input(input_key(0)))],
        &[vec![
            DimExpr::InputDim {
                input_idx: 0,
                axis: 0,
            },
            DimExpr::InputDim {
                input_idx: 0,
                axis: 1,
            },
        ]],
    );

    let local = unwrap_local(result);
    builder.set_outputs(vec![local]);
    let graph = builder.build();
    assert_eq!(graph.operations().len(), 1);
    assert!(matches!(
        graph.operations()[0].operation,
        StdTensorOp::Reshape { .. }
    ));
}

#[test]
fn graph_matmul_ij_jk_ik() {
    let tree = make_tree("ij,jk->ik", &[&[2, 3], &[3, 4]]);
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));
    let b = builder.add_input(input_key(1));

    let result = build_einsum_graph(
        &mut builder,
        &tree,
        &[ValueRef::Local(a), ValueRef::Local(b)],
        &[vec![2, 3], vec![3, 4]],
    );

    builder.set_outputs(vec![unwrap_local(result)]);
    let graph = builder.build();
    assert!(!graph.operations().is_empty());
}

#[test]
fn graph_row_sum_ij_i() {
    let tree = make_tree("ij->i", &[&[3, 4]]);
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));

    let result = build_einsum_graph(&mut builder, &tree, &[ValueRef::Local(a)], &[vec![3, 4]]);

    builder.set_outputs(vec![unwrap_local(result)]);
    let graph = builder.build();
    // Should have a reduce_sum op
    assert!(!graph.operations().is_empty());
}

#[test]
fn graph_outer_product_i_j_ij() {
    let tree = make_tree("i,j->ij", &[&[3], &[4]]);
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));
    let b = builder.add_input(input_key(1));

    let result = build_einsum_graph(
        &mut builder,
        &tree,
        &[ValueRef::Local(a), ValueRef::Local(b)],
        &[vec![3], vec![4]],
    );

    builder.set_outputs(vec![unwrap_local(result)]);
    let graph = builder.build();
    assert!(!graph.operations().is_empty());
}

#[test]
fn graph_outer_product_accepts_symbolic_dim_expr_shapes() {
    let tree = make_tree("i,j->ij", &[&[3], &[4]]);
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));
    let b = builder.add_input(input_key(1));

    let result = build_einsum_graph_dim_expr(
        &mut builder,
        &tree,
        &[ValueRef::Local(a), ValueRef::Local(b)],
        &[
            vec![DimExpr::InputDim {
                input_idx: 0,
                axis: 0,
            }],
            vec![DimExpr::InputDim {
                input_idx: 1,
                axis: 0,
            }],
        ],
    );

    builder.set_outputs(vec![unwrap_local(result)]);
    let graph = builder.build();
    let broadcasts: Vec<_> = graph
        .operations()
        .iter()
        .filter_map(|node| match &node.operation {
            StdTensorOp::BroadcastInDim { shape, dims } => Some((shape, dims)),
            _ => None,
        })
        .collect();

    assert_eq!(broadcasts.len(), 2);
    assert!(broadcasts.iter().any(|(shape, dims)| {
        dims.as_slice() == [0]
            && matches!(
                shape.as_slice(),
                [
                    DimExpr::InputDim {
                        input_idx: 0,
                        axis: 0
                    },
                    DimExpr::InputDim {
                        input_idx: 1,
                        axis: 0
                    }
                ]
            )
    }));
    assert!(broadcasts.iter().any(|(shape, dims)| {
        dims.as_slice() == [1]
            && matches!(
                shape.as_slice(),
                [
                    DimExpr::InputDim {
                        input_idx: 1,
                        axis: 0
                    },
                    DimExpr::InputDim {
                        input_idx: 0,
                        axis: 0
                    }
                ]
            )
    }));
}

#[test]
fn graph_outer_product_uses_single_broadcast_input_when_secondary_shape_is_unused() {
    let tree = make_tree("ij,j->ij", &[&[3, 4], &[4]]);
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));
    let b = builder.add_input(input_key(1));

    let result = build_einsum_graph_dim_expr(
        &mut builder,
        &tree,
        &[ValueRef::Local(a), ValueRef::Local(b)],
        &[
            vec![
                DimExpr::InputDim {
                    input_idx: 0,
                    axis: 0,
                },
                DimExpr::InputDim {
                    input_idx: 0,
                    axis: 1,
                },
            ],
            vec![DimExpr::InputDim {
                input_idx: 1,
                axis: 0,
            }],
        ],
    );

    builder.set_outputs(vec![unwrap_local(result)]);
    let graph = builder.build();
    let broadcast_inputs: Vec<_> = graph
        .operations()
        .iter()
        .filter_map(|node| match &node.operation {
            StdTensorOp::BroadcastInDim { dims, .. } => Some((dims.as_slice(), node.inputs.len())),
            _ => None,
        })
        .collect();

    assert!(broadcast_inputs
        .iter()
        .any(|(dims, input_count)| *dims == [0, 1] && *input_count == 1));
    assert!(broadcast_inputs
        .iter()
        .any(|(dims, input_count)| *dims == [1] && *input_count == 2));
}

#[test]
fn graph_inner_product_i_i() {
    let tree = make_tree("i,i->", &[&[3], &[3]]);
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));
    let b = builder.add_input(input_key(1));

    let result = build_einsum_graph(
        &mut builder,
        &tree,
        &[ValueRef::Local(a), ValueRef::Local(b)],
        &[vec![3], vec![3]],
    );

    builder.set_outputs(vec![unwrap_local(result)]);
    let graph = builder.build();
    assert!(!graph.operations().is_empty());
}

#[test]
fn graph_hadamard_ij_ij_ij() {
    let tree = make_tree("ij,ij->ij", &[&[3, 4], &[3, 4]]);
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));
    let b = builder.add_input(input_key(1));

    let result = build_einsum_graph(
        &mut builder,
        &tree,
        &[ValueRef::Local(a), ValueRef::Local(b)],
        &[vec![3, 4], vec![3, 4]],
    );

    builder.set_outputs(vec![unwrap_local(result)]);
    let graph = builder.build();
    assert!(!graph.operations().is_empty());
}

#[test]
fn graph_batched_chain_avoids_intermediate_transpose() {
    let subs = Subscripts::parse("bik,bkj,bjl->bil").expect("bad notation");
    let shapes = [&[2, 3, 4][..], &[2, 4, 5][..], &[2, 5, 6][..]];
    let tree = ContractionTree::from_pairs(&subs, &shapes, &[(0, 1), (3, 2)]).unwrap();

    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));
    let b = builder.add_input(input_key(1));
    let c = builder.add_input(input_key(2));

    let result = build_einsum_graph(
        &mut builder,
        &tree,
        &[ValueRef::Local(a), ValueRef::Local(b), ValueRef::Local(c)],
        &[vec![2, 3, 4], vec![2, 4, 5], vec![2, 5, 6]],
    );

    builder.set_outputs(vec![unwrap_local(result)]);
    let graph = builder.build();
    let transpose_count = graph
        .operations()
        .iter()
        .filter(|node| matches!(node.operation, StdTensorOp::Transpose { .. }))
        .count();

    assert_eq!(transpose_count, 1, "expected only the final transpose");
    match &graph
        .operations()
        .last()
        .expect("graph should have ops")
        .operation
    {
        StdTensorOp::Transpose { perm } => assert_eq!(perm.as_slice(), &[2, 0, 1]),
        other => panic!("expected final transpose, got {other:?}"),
    }
}

#[test]
fn tree_binary() {
    let tree = make_tree("ij,jk->ik", &[&[2, 3], &[3, 4]]);
    assert_eq!(tree.step_count(), 1);
    assert_eq!(tree.step_pair(0), Some((0, 1)));
}

#[test]
fn tree_unary() {
    let tree = make_tree("ij->i", &[&[3, 4]]);
    assert_eq!(tree.step_count(), 0);
}

#[test]
fn tree_ternary() {
    let tree = make_tree("ij,jk,kl->il", &[&[2, 3], &[3, 4], &[4, 5]]);
    assert_eq!(tree.step_count(), 2);
}

#[test]
fn test_diagonalize_repeated() {
    // "ii->i" should produce an ExtractDiag in the graph
    let tree = make_tree("ii->i", &[&[3, 3]]);
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));

    let result = build_einsum_graph(&mut builder, &tree, &[ValueRef::Local(a)], &[vec![3, 3]]);

    builder.set_outputs(vec![unwrap_local(result)]);
    let graph = builder.build();
    let ops: Vec<_> = graph.operations().iter().map(|n| &n.operation).collect();
    // Should contain exactly one ExtractDiag
    let extract_count = ops
        .iter()
        .filter(|op| matches!(op, StdTensorOp::ExtractDiag { .. }))
        .count();
    assert_eq!(
        extract_count, 1,
        "expected 1 ExtractDiag, got {extract_count}"
    );
    // No ReduceSum (output has 'i', so nothing to reduce)
    let reduce_count = ops
        .iter()
        .filter(|op| matches!(op, StdTensorOp::ReduceSum { .. }))
        .count();
    assert_eq!(reduce_count, 0, "expected 0 ReduceSum, got {reduce_count}");
}

#[test]
fn test_trace_graph() {
    // "ii->" should produce ExtractDiag + ReduceSum in the graph
    let tree = make_tree("ii->", &[&[3, 3]]);
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));

    let result = build_einsum_graph(&mut builder, &tree, &[ValueRef::Local(a)], &[vec![3, 3]]);

    builder.set_outputs(vec![unwrap_local(result)]);
    let graph = builder.build();
    let ops: Vec<_> = graph.operations().iter().map(|n| &n.operation).collect();
    // Should contain ExtractDiag followed by ReduceSum
    let extract_count = ops
        .iter()
        .filter(|op| matches!(op, StdTensorOp::ExtractDiag { .. }))
        .count();
    assert_eq!(
        extract_count, 1,
        "expected 1 ExtractDiag, got {extract_count}"
    );
    let reduce_count = ops
        .iter()
        .filter(|op| matches!(op, StdTensorOp::ReduceSum { .. }))
        .count();
    assert_eq!(reduce_count, 1, "expected 1 ReduceSum, got {reduce_count}");
}
