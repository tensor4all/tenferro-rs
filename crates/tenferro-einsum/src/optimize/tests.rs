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
    let spec = plan_spec_from_optimize(EinsumOptimize::False, &subs).unwrap();
    let tree = resolve_plan_spec(&spec, &subs, &shapes).unwrap();

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

#[test]
fn concrete_tree_strategy_revalidates_current_shapes() {
    let subs = Subscripts::parse("ij,jk->ik").unwrap();
    let original_shapes = [&[2, 3][..], &[3, 4][..]];
    let tree = ContractionTree::from_pairs(&subs, &original_shapes, &[(0, 1)]).unwrap();
    let current_shapes = [&[2, 3][..], &[5, 4][..]];

    let err =
        match resolve_einsum_strategy_with_spec(EinsumOptimize::Tree(tree), &subs, &current_shapes)
        {
            Ok(_) => panic!("expected incompatible shapes to be rejected"),
            Err(err) => err,
        };

    assert!(
        matches!(err, Error::Validation { .. }),
        "expected ShapeMismatch, got {err}"
    );
}

#[test]
fn concrete_tree_strategy_rejects_different_current_arity() {
    let original_subs = Subscripts::parse("ij,jk,kl->il").unwrap();
    let original_shapes = [&[2, 3][..], &[3, 4][..], &[4, 5][..]];
    let tree =
        ContractionTree::from_pairs(&original_subs, &original_shapes, &[(1, 2), (0, 3)]).unwrap();
    let current_subs = Subscripts::parse("ij,jk->ik").unwrap();
    let current_shapes = [&[2, 3][..], &[3, 4][..]];

    let err = match resolve_einsum_strategy_with_spec(
        EinsumOptimize::Tree(tree),
        &current_subs,
        &current_shapes,
    ) {
        Ok(_) => panic!("expected different current arity to be rejected"),
        Err(err) => err,
    };

    assert!(format!("{err}").contains("must have"), "got {err}");
}

#[test]
fn plan_spec_from_nested_rejects_duplicate_leaves() {
    let subs = Subscripts::parse("ij,jk->ik").unwrap();
    let nested = NestedEinsum::Node {
        subscripts: subs.clone(),
        children: vec![NestedEinsum::Leaf(0), NestedEinsum::Leaf(0)],
    };

    let err = plan_spec_from_optimize(EinsumOptimize::Nested(nested), &subs).unwrap_err();

    assert!(format!("{err}").contains("distinct"), "got {err}");
}

#[test]
fn plan_spec_from_nested_rejects_missing_leaves() {
    let subs = Subscripts::parse("ij,jk,kl->il").unwrap();
    let nested = NestedEinsum::Node {
        subscripts: Subscripts::parse("ij,jk->ik").unwrap(),
        children: vec![NestedEinsum::Leaf(0), NestedEinsum::Leaf(1)],
    };

    let err = plan_spec_from_optimize(EinsumOptimize::Nested(nested), &subs).unwrap_err();

    assert!(format!("{err}").contains("must have"), "got {err}");
}

#[test]
fn plan_spec_from_auto_rejects_nan_options() {
    let subs = Subscripts::parse("ij,jk->ik").unwrap();
    let options = ContractionOptimizerOptions {
        betas: vec![f64::NAN],
        ..ContractionOptimizerOptions::default()
    };

    let err = plan_spec_from_optimize(EinsumOptimize::Auto(options), &subs).unwrap_err();

    assert!(
        format!("{err}").contains("must not contain NaN"),
        "got {err}"
    );
}

#[test]
fn identical_plan_specs_have_equal_hashes() {
    let lhs = EinsumPlanSpec::FixedPairs(vec![(1, 2), (0, 3)]);
    let rhs = EinsumPlanSpec::FixedPairs(vec![(1, 2), (0, 3)]);

    assert!(plan_specs_equal(&lhs, &rhs));
    assert_eq!(hash_plan_spec(&lhs), hash_plan_spec(&rhs));
}

fn hash_plan_spec(spec: &EinsumPlanSpec) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hash_einsum_plan_spec(spec, &mut hasher);
    hasher.finish()
}
