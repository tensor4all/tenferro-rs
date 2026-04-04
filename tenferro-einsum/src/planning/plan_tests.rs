use crate::planning::tree::ContractionTree;
use crate::syntax::subscripts::Subscripts;

use super::{
    compile_binary_contraction_plan, compile_pairwise_step_plan,
    compile_strict_binary_lowering_plan,
};

#[test]
fn binary_plan_matches_fixed_pair_tree_step_plan() {
    let subs = Subscripts::new(&[&[0, 0, 1], &[0, 2]], &[2]);
    let shapes = [&[3, 3, 5][..], &[3, 7][..]];

    let direct = compile_binary_contraction_plan(&subs, &shapes, None).unwrap();
    let tree = ContractionTree::from_pairs(&subs, &shapes, &[(0, 1)]).unwrap();

    assert_eq!(direct.size_dict, tree.size_dict);
    assert_eq!(direct.step_plan, tree.step_plans[0]);
    assert_eq!(direct.output_shape, vec![7]);
}

#[test]
fn strict_binary_lowering_plan_builds_dense_matmul() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let shapes = [&[2, 3][..], &[3, 4][..]];

    let plan = compile_strict_binary_lowering_plan(&subs, &shapes, None)
        .unwrap()
        .expect("simple dense matmul should admit strict lowering");

    assert_eq!(plan.lhs_free_labels, vec![0]);
    assert_eq!(plan.rhs_free_labels, vec![2]);
    assert_eq!(plan.contract_labels, vec![1]);
    assert_eq!(plan.lhs_perm, vec![0, 1]);
    assert_eq!(plan.rhs_perm, vec![0, 1]);
    assert_eq!(plan.m, 2);
    assert_eq!(plan.k, 3);
    assert_eq!(plan.n, 4);
    assert_eq!(plan.lhs_matrix_dims, vec![2, 3]);
    assert_eq!(plan.rhs_matrix_dims, vec![3, 4]);
    assert_eq!(plan.canonical_output_dims, vec![2, 4]);
    assert_eq!(plan.output_perm, vec![0, 1]);
}

#[test]
fn strict_binary_lowering_rejects_repeated_labels() {
    let subs = Subscripts::new(&[&[0, 0, 1], &[1, 2]], &[0, 2]);
    let shapes = [&[2, 2, 3][..], &[3, 4][..]];

    let plan = compile_strict_binary_lowering_plan(&subs, &shapes, None).unwrap();

    assert!(
        plan.is_none(),
        "repeated labels must stay on the generic binary lowering path"
    );
}

#[test]
fn strict_binary_lowering_plan_tracks_non_identity_output_permutation() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[2, 0]);
    let shapes = [&[2, 3][..], &[3, 4][..]];

    let plan = compile_strict_binary_lowering_plan(&subs, &shapes, None)
        .unwrap()
        .expect("output permutations should still admit strict lowering");

    assert_eq!(plan.canonical_output_dims, vec![2, 4]);
    assert_eq!(plan.output_perm, vec![1, 0]);
}

#[test]
fn pairwise_step_plan_embeds_strict_binary_lowering_when_available() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[2, 0]);
    let shapes = [&[2, 3][..], &[3, 4][..]];
    let size_dict = crate::util::build_size_dict(&subs, &shapes, None).unwrap();

    let step_plan =
        compile_pairwise_step_plan(&subs.inputs[0], &subs.inputs[1], &subs.output, &size_dict)
            .unwrap();

    let strict = step_plan
        .strict_binary
        .as_ref()
        .expect("dense binary matmul step should cache strict lowering");
    assert_eq!(strict.output_perm, vec![1, 0]);
}
