use computegraph::fragment::FragmentBuilder;
use computegraph::types::ValRef;

use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;

use crate::v2::builder::{build_einsum_fragment, optimize_contraction_path};
use crate::v2::types::Subscripts;

fn input_key(id: u64) -> TensorInputKey {
    TensorInputKey { id }
}

#[test]
fn fragment_matmul_ij_jk_ik() {
    let subscripts = Subscripts::parse("ij,jk->ik");
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));
    let b = builder.add_input(input_key(1));

    let result = build_einsum_fragment(
        &mut builder,
        &subscripts,
        &[ValRef::Local(a), ValRef::Local(b)],
        &[vec![2, 3], vec![3, 4]],
    );

    builder.set_outputs(vec![match &result {
        ValRef::Local(id) => *id,
        _ => panic!("expected local"),
    }]);
    let fragment = builder.build();
    assert!(!fragment.ops().is_empty());
}

#[test]
fn fragment_row_sum_ij_i() {
    let subscripts = Subscripts::parse("ij->i");
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));

    let result = build_einsum_fragment(
        &mut builder,
        &subscripts,
        &[ValRef::Local(a)],
        &[vec![3, 4]],
    );

    builder.set_outputs(vec![match &result {
        ValRef::Local(id) => *id,
        _ => panic!("expected local"),
    }]);
    let fragment = builder.build();
    // Should have a reduce_sum op
    assert!(!fragment.ops().is_empty());
}

#[test]
fn fragment_outer_product_i_j_ij() {
    let subscripts = Subscripts::parse("i,j->ij");
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));
    let b = builder.add_input(input_key(1));

    let result = build_einsum_fragment(
        &mut builder,
        &subscripts,
        &[ValRef::Local(a), ValRef::Local(b)],
        &[vec![3], vec![4]],
    );

    builder.set_outputs(vec![match &result {
        ValRef::Local(id) => *id,
        _ => panic!("expected local"),
    }]);
    let fragment = builder.build();
    assert!(!fragment.ops().is_empty());
}

#[test]
fn fragment_inner_product_i_i() {
    let subscripts = Subscripts::parse("i,i->");
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));
    let b = builder.add_input(input_key(1));

    let result = build_einsum_fragment(
        &mut builder,
        &subscripts,
        &[ValRef::Local(a), ValRef::Local(b)],
        &[vec![3], vec![3]],
    );

    builder.set_outputs(vec![match &result {
        ValRef::Local(id) => *id,
        _ => panic!("expected local"),
    }]);
    let fragment = builder.build();
    assert!(!fragment.ops().is_empty());
}

#[test]
fn fragment_hadamard_ij_ij_ij() {
    let subscripts = Subscripts::parse("ij,ij->ij");
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key(0));
    let b = builder.add_input(input_key(1));

    let result = build_einsum_fragment(
        &mut builder,
        &subscripts,
        &[ValRef::Local(a), ValRef::Local(b)],
        &[vec![3, 4], vec![3, 4]],
    );

    builder.set_outputs(vec![match &result {
        ValRef::Local(id) => *id,
        _ => panic!("expected local"),
    }]);
    let fragment = builder.build();
    assert!(!fragment.ops().is_empty());
}

#[test]
fn contraction_path_binary() {
    let subscripts = Subscripts::parse("ij,jk->ik");
    let path = optimize_contraction_path(&subscripts, &[vec![2, 3], vec![3, 4]]);
    assert_eq!(path, vec![(0, 1)]);
}

#[test]
fn contraction_path_unary() {
    let subscripts = Subscripts::parse("ij->i");
    let path = optimize_contraction_path(&subscripts, &[vec![3, 4]]);
    assert!(path.is_empty());
}

#[test]
fn contraction_path_ternary() {
    let subscripts = Subscripts::parse("ij,jk,kl->il");
    let path = optimize_contraction_path(&subscripts, &[vec![2, 3], vec![3, 4], vec![4, 5]]);
    assert_eq!(path.len(), 2);
}
