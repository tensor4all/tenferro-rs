use crate::syntax::subscripts::Subscripts;
use crate::{Error, Result as EinsumResult};

use super::{compile_pairwise_step_plan, compile_strict_binary_lowering_step_plan};

fn compile_strict_step_plan(
    subs: &Subscripts,
    shapes: &[&[usize]],
) -> EinsumResult<Option<super::StrictBinaryLoweringPlan>> {
    let size_dict = crate::util::build_size_dict(subs, shapes, None)?;
    compile_strict_binary_lowering_step_plan(
        &subs.inputs[0],
        &subs.inputs[1],
        &subs.output,
        &size_dict,
    )
}

#[test]
fn strict_binary_lowering_plan_builds_dense_matmul() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let shapes = [&[2, 3][..], &[3, 4][..]];

    let plan = compile_strict_step_plan(&subs, &shapes)
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

    let plan = compile_strict_step_plan(&subs, &shapes).unwrap();

    assert!(
        plan.is_none(),
        "repeated labels must stay on the generic binary lowering path"
    );
}

#[test]
fn strict_binary_lowering_plan_tracks_non_identity_output_permutation() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[2, 0]);
    let shapes = [&[2, 3][..], &[3, 4][..]];

    let plan = compile_strict_step_plan(&subs, &shapes)
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

#[test]
fn pairwise_step_plan_preserves_strict_lowering_error_type() {
    let size_dict = [(0, 2), (1, 3)].into_iter().collect();

    let err = compile_pairwise_step_plan(&[0], &[1], &[2], &size_dict).unwrap_err();

    assert!(matches!(
        err,
        Error::InvalidArgument(message)
            if message.contains("unknown size")
    ));
}

#[test]
fn pairwise_step_plan_rejects_overflowing_fused_dimensions() {
    let size_dict = [(0, usize::MAX), (1, 2), (2, 3), (4, 2)]
        .into_iter()
        .collect();

    let err = compile_pairwise_step_plan(&[0, 4, 1], &[1, 2], &[0, 4, 2], &size_dict).unwrap_err();

    assert!(matches!(
        err,
        Error::InvalidArgument(message)
            if message.contains("dimension product overflow")
    ));
}

#[test]
fn strict_binary_lowering_rejects_overflowing_fused_dimensions() {
    let subs = Subscripts::new(&[&[0, 4, 1], &[1, 2]], &[0, 4, 2]);
    let shapes = [&[usize::MAX, 2, 2][..], &[2, 3][..]];

    let err = compile_strict_step_plan(&subs, &shapes).unwrap_err();

    assert!(matches!(
        err,
        Error::InvalidArgument(message)
            if message.contains("dimension product overflow")
    ));
}
