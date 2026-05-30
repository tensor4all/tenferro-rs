use super::*;
use std::hash::Hasher;

#[test]
fn jax_path_to_v1_pairs_matches_shrinking_list_semantics() {
    let pairs = jax_path_to_v1_pairs(&[(1, 2), (0, 1)], 3).unwrap();

    assert_eq!(pairs, vec![(1, 2), (0, 3)]);
}

#[test]
fn false_strategy_builds_left_to_right_tree() {
    let subs = Subscripts::parse("ij,jk,kl->il").unwrap();
    let shapes = [&[2, 3][..], &[3, 4][..], &[4, 5][..]];
    let tree = resolve_einsum_strategy(EinsumOptimize::False, &subs, &shapes).unwrap();

    assert_eq!(tree.step_pair(0), Some((0, 1)));
    assert_eq!(tree.step_pair(1), Some((2, 3)));
}

#[test]
fn plan_spec_from_false_preserves_left_to_right_policy() {
    let subs = Subscripts::parse("ij,jk,kl->il").unwrap();
    let spec = plan_spec_from_optimize(EinsumOptimize::False, &subs).unwrap();
    assert!(matches!(spec, EinsumPlanSpec::LeftToRight));

    let shapes = [&[2, 3][..], &[3, 4][..], &[4, 5][..]];
    let tree = resolve_plan_spec(&spec, &subs, &shapes).unwrap();
    assert_eq!(tree.step_pair(0), Some((0, 1)));
    assert_eq!(tree.step_pair(1), Some((2, 3)));
}

#[test]
fn plan_spec_from_path_validates_shape_independent_indices() {
    let subs = Subscripts::parse("ij,jk,kl->il").unwrap();
    let err =
        plan_spec_from_optimize(EinsumOptimize::Path(vec![(0, 99), (0, 1)]), &subs).unwrap_err();

    assert!(
        format!("{err}").contains("path step 0 references operand positions"),
        "got {err}"
    );
}

#[test]
fn plan_spec_rejects_symbolic_tree_strategy() {
    let subs = Subscripts::parse("ij,jk->ik").unwrap();
    let shapes = [&[2, 3][..], &[3, 4][..]];
    let tree = ContractionTree::optimize(&subs, &shapes).unwrap();
    let err = plan_spec_from_optimize(EinsumOptimize::Tree(tree), &subs).unwrap_err();

    assert!(
        format!("{err}").contains("precomputed contraction tree requires concrete input shapes"),
        "got {err}"
    );
}

#[test]
fn plan_spec_hash_distinguishes_auto_options() {
    let lhs = EinsumPlanSpec::Auto(ContractionOptimizerOptions::default());
    let rhs = EinsumPlanSpec::Auto(ContractionOptimizerOptions {
        ntrials: 2,
        ..ContractionOptimizerOptions::default()
    });

    assert!(!plan_specs_equal(&lhs, &rhs));
    assert_ne!(hash_plan_spec(&lhs), hash_plan_spec(&rhs));
}

#[test]
fn concrete_tree_strategy_converts_to_fixed_pairs() {
    let subs = Subscripts::parse("ij,jk,kl->il").unwrap();
    let shapes = [&[2, 3][..], &[3, 4][..], &[4, 5][..]];
    let tree = ContractionTree::from_pairs(&subs, &shapes, &[(1, 2), (0, 3)]).unwrap();

    let (spec, resolved_tree) =
        resolve_einsum_strategy_with_spec(EinsumOptimize::Tree(tree), &subs, &shapes).unwrap();

    assert!(
        matches!(spec, EinsumPlanSpec::FixedPairs(ref pairs) if pairs == &vec![(1, 2), (0, 3)])
    );
    assert_eq!(resolved_tree.step_pair(0), Some((1, 2)));
    assert_eq!(resolved_tree.step_pair(1), Some((0, 3)));
}

fn hash_plan_spec(spec: &EinsumPlanSpec) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hash_einsum_plan_spec(spec, &mut hasher);
    hasher.finish()
}
