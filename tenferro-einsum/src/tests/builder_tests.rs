use computegraph::fragment::FragmentBuilder;
use computegraph::types::ValRef;

use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;

use crate::builder::build_einsum_fragment;
use crate::planning::tree::ContractionTree;
use crate::syntax::subscripts::Subscripts;

fn input_key(id: u64) -> TensorInputKey {
    TensorInputKey::User { id }
}

fn make_tree(notation: &str, shapes: &[&[usize]]) -> ContractionTree {
    let subscripts = Subscripts::parse(notation).expect("bad notation");
    ContractionTree::optimize(&subscripts, shapes).expect("optimize failed")
}

fn unwrap_local(result: tenferro_device::Result<ValRef<StdTensorOp>>) -> usize {
    match result.expect("builder should succeed") {
        ValRef::Local(id) => id,
        ValRef::External(_) => panic!("expected local"),
    }
}

#[test]
fn builder_reports_missing_output_label_instead_of_panicking() {
    let mut tree = make_tree("i->i", &[&[3]]);
    tree.subscripts.output = vec![b'j' as u32];
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));

    let err =
        build_einsum_fragment(&mut builder, &tree, &[ValRef::Local(a)], &[vec![3]]).unwrap_err();

    assert!(matches!(
        err,
        tenferro_device::Error::InvalidArgument(message)
            if message.contains("missing") && message.contains("label")
    ));
}

#[test]
fn fragment_matmul_ij_jk_ik() {
    let tree = make_tree("ij,jk->ik", &[&[2, 3], &[3, 4]]);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));
    let b = builder.add_input(input_key(1));

    let result = build_einsum_fragment(
        &mut builder,
        &tree,
        &[ValRef::Local(a), ValRef::Local(b)],
        &[vec![2, 3], vec![3, 4]],
    );

    builder.set_outputs(vec![unwrap_local(result)]);
    let fragment = builder.build();
    assert!(!fragment.ops().is_empty());
}

#[test]
fn fragment_row_sum_ij_i() {
    let tree = make_tree("ij->i", &[&[3, 4]]);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));

    let result = build_einsum_fragment(&mut builder, &tree, &[ValRef::Local(a)], &[vec![3, 4]]);

    builder.set_outputs(vec![unwrap_local(result)]);
    let fragment = builder.build();
    // Should have a reduce_sum op
    assert!(!fragment.ops().is_empty());
}

#[test]
fn fragment_outer_product_i_j_ij() {
    let tree = make_tree("i,j->ij", &[&[3], &[4]]);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));
    let b = builder.add_input(input_key(1));

    let result = build_einsum_fragment(
        &mut builder,
        &tree,
        &[ValRef::Local(a), ValRef::Local(b)],
        &[vec![3], vec![4]],
    );

    builder.set_outputs(vec![unwrap_local(result)]);
    let fragment = builder.build();
    assert!(!fragment.ops().is_empty());
}

#[test]
fn fragment_inner_product_i_i() {
    let tree = make_tree("i,i->", &[&[3], &[3]]);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));
    let b = builder.add_input(input_key(1));

    let result = build_einsum_fragment(
        &mut builder,
        &tree,
        &[ValRef::Local(a), ValRef::Local(b)],
        &[vec![3], vec![3]],
    );

    builder.set_outputs(vec![unwrap_local(result)]);
    let fragment = builder.build();
    assert!(!fragment.ops().is_empty());
}

#[test]
fn fragment_hadamard_ij_ij_ij() {
    let tree = make_tree("ij,ij->ij", &[&[3, 4], &[3, 4]]);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));
    let b = builder.add_input(input_key(1));

    let result = build_einsum_fragment(
        &mut builder,
        &tree,
        &[ValRef::Local(a), ValRef::Local(b)],
        &[vec![3, 4], vec![3, 4]],
    );

    builder.set_outputs(vec![unwrap_local(result)]);
    let fragment = builder.build();
    assert!(!fragment.ops().is_empty());
}

#[test]
fn fragment_batched_chain_avoids_intermediate_transpose() {
    let subs = Subscripts::parse("bik,bkj,bjl->bil").expect("bad notation");
    let shapes = [&[2, 3, 4][..], &[2, 4, 5][..], &[2, 5, 6][..]];
    let tree = ContractionTree::from_pairs(&subs, &shapes, &[(0, 1), (3, 2)]).unwrap();

    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));
    let b = builder.add_input(input_key(1));
    let c = builder.add_input(input_key(2));

    let result = build_einsum_fragment(
        &mut builder,
        &tree,
        &[ValRef::Local(a), ValRef::Local(b), ValRef::Local(c)],
        &[vec![2, 3, 4], vec![2, 4, 5], vec![2, 5, 6]],
    );

    builder.set_outputs(vec![unwrap_local(result)]);
    let fragment = builder.build();
    let transpose_count = fragment
        .ops()
        .iter()
        .filter(|node| matches!(node.op, StdTensorOp::Transpose { .. }))
        .count();

    assert_eq!(transpose_count, 1, "expected only the final transpose");
    match &fragment.ops().last().expect("fragment should have ops").op {
        StdTensorOp::Transpose { perm } => assert_eq!(perm, &vec![2, 0, 1]),
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
    // "ii->i" should produce an ExtractDiag in the fragment
    let tree = make_tree("ii->i", &[&[3, 3]]);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));

    let result = build_einsum_fragment(&mut builder, &tree, &[ValRef::Local(a)], &[vec![3, 3]]);

    builder.set_outputs(vec![unwrap_local(result)]);
    let fragment = builder.build();
    let ops: Vec<_> = fragment.ops().iter().map(|n| &n.op).collect();
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
fn test_trace_fragment() {
    // "ii->" should produce ExtractDiag + ReduceSum in the fragment
    let tree = make_tree("ii->", &[&[3, 3]]);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));

    let result = build_einsum_fragment(&mut builder, &tree, &[ValRef::Local(a)], &[vec![3, 3]]);

    builder.set_outputs(vec![unwrap_local(result)]);
    let fragment = builder.build();
    let ops: Vec<_> = fragment.ops().iter().map(|n| &n.op).collect();
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
