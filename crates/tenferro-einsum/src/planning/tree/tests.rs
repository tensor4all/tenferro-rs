use std::cell::Cell;
use std::collections::HashSet;

// Scalar test instrumentation, not cached planning state. Thread isolation keeps
// parallel tests independent; these counters do not exist in non-test builds.
thread_local! {
    pub(super) static OMECO_CALLS: Cell<usize> = const { Cell::new(0) };
    pub(super) static SELF_GREEDY_CALLS: Cell<usize> = const { Cell::new(0) };
}

use std::collections::HashMap;

use omeco::{EinCode as OmecoEinCode, Initializer, NestedEinsum, ScoreFunction};

use crate::syntax::subscripts::Subscripts;
use crate::Error;

use super::{
    build_needed_label_counts, collect_candidate_intermediate_subs, nested_to_pairs,
    optimize_self_greedy_pairs, ContractionOptimizerOptions, ContractionTree,
};

#[test]
fn binary_public_planning_bypasses_both_general_optimizers() {
    OMECO_CALLS.with(|count| count.set(0));
    SELF_GREEDY_CALLS.with(|count| count.set(0));
    let counts = || {
        (
            OMECO_CALLS.with(Cell::get),
            SELF_GREEDY_CALLS.with(Cell::get),
        )
    };
    let binary = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let shapes = [&[2, 3][..], &[3, 4][..]];
    let tree = ContractionTree::optimize(&binary, &shapes).unwrap();
    assert_eq!(counts(), (0, 0));
    assert_eq!(tree.step_pair(0), Some((0, 1)));
    let options = ContractionOptimizerOptions {
        ntrials: 2,
        niters: 3,
        betas: vec![0.1, 1.0],
        ..ContractionOptimizerOptions::default()
    };
    let tree = ContractionTree::optimize_with_options(&binary, &shapes, &options).unwrap();
    assert_eq!(tree.step_count(), 1);
    assert_eq!(counts(), (0, 0));

    let a = tenferro_tensor::Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    let b = tenferro_tensor::Tensor::from_vec_col_major(vec![3, 4], vec![2.0_f64; 12]).unwrap();
    let _plan = crate::ConcreteEinsumPlan::prepare([&a, &b], "ij,jk->ik").unwrap();
    assert_eq!(counts(), (0, 0));

    // A positive control proves that the real general entry is instrumented.
    let nary = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let shapes = [&[2, 3][..], &[3, 4][..], &[4, 2][..]];
    let tree = ContractionTree::optimize(&nary, &shapes).unwrap();
    assert_eq!(tree.step_count(), 2);
    assert_eq!(OMECO_CALLS.with(Cell::get), 1);

    // Also prove the fallback counter is connected, independently of omeco's
    // choice to return a plan for the positive-control problem above.
    let before = SELF_GREEDY_CALLS.with(Cell::get);
    let sizes = super::build_size_dict(&nary, &shapes, None).unwrap();
    assert_eq!(optimize_self_greedy_pairs(&nary, &sizes).unwrap().len(), 2);
    assert_eq!(SELF_GREEDY_CALLS.with(Cell::get), before + 1);
}

#[test]
fn default_options_build_zero_iter_greedy_initialized_treesa() {
    let options = ContractionOptimizerOptions::default();
    let optimizer = options.to_treesa();

    assert!(optimizer.betas.is_empty());
    assert_eq!(optimizer.ntrials, 1);
    assert_eq!(optimizer.niters, 0);
    assert_eq!(optimizer.initializer, Initializer::Greedy);
    assert_eq!(optimizer.score.tc_weight, 1.0);
    assert_eq!(optimizer.score.sc_weight, 1.0);
    assert_eq!(optimizer.score.rw_weight, 0.0);
}

#[test]
fn optimize_with_default_options_builds_valid_tree_for_issue_336_env6() {
    let subs = Subscripts::new(
        &[
            &[1, 8, 0, 2],
            &[3, 0, 9, 4],
            &[6, 10, 5, 1],
            &[7, 5, 11, 3],
            &[2, 4, 12],
            &[6, 7, 13],
        ],
        &[8, 9, 10, 11, 12, 13],
    );
    let shapes = [
        &[8, 2, 2, 8][..],
        &[8, 2, 2, 8][..],
        &[8, 2, 2, 8][..],
        &[8, 2, 2, 8][..],
        &[8, 8, 16][..],
        &[8, 8, 16][..],
    ];

    let tree = ContractionTree::optimize_with_options(
        &subs,
        &shapes,
        &ContractionOptimizerOptions::default(),
    )
    .unwrap();

    assert_eq!(tree.step_count(), subs.inputs.len() - 1);
    let final_labels: HashSet<u32> = tree
        .step_subscripts(tree.step_count() - 1)
        .unwrap()
        .2
        .iter()
        .copied()
        .collect();
    let output_labels: HashSet<u32> = subs.output.iter().copied().collect();
    assert_eq!(final_labels, output_labels);
}

#[test]
fn optimize_with_options_accepts_time_optimized_score() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let shapes = [&[3, 4][..], &[4, 5][..], &[5, 6][..]];
    let options = ContractionOptimizerOptions {
        betas: vec![0.1],
        ntrials: 1,
        niters: 1,
        score: ScoreFunction::time_optimized(),
    };

    let optimizer = options.to_treesa();
    let tree = ContractionTree::optimize_with_options(&subs, &shapes, &options).unwrap();
    assert_eq!(optimizer.betas, options.betas);
    assert_eq!(optimizer.ntrials, options.ntrials);
    assert_eq!(optimizer.niters, options.niters);
    assert_eq!(optimizer.initializer, Initializer::Greedy);
    assert_eq!(optimizer.score.tc_weight, options.score.tc_weight);
    assert_eq!(optimizer.score.sc_weight, options.score.sc_weight);
    assert_eq!(optimizer.score.rw_weight, options.score.rw_weight);
    assert_eq!(optimizer.score.sc_target, options.score.sc_target);
    assert_eq!(tree.step_count(), subs.inputs.len() - 1);
}

#[test]
fn optimize_with_options_rejects_zero_trials() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let shapes = [&[2, 3][..], &[3, 4][..]];
    let options = ContractionOptimizerOptions {
        ntrials: 0,
        ..ContractionOptimizerOptions::default()
    };

    let err = match ContractionTree::optimize_with_options(&subs, &shapes, &options) {
        Ok(_) => panic!("expected invalid ntrials to be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::Planning { source } if source.to_string().contains("ntrials")));
}

#[test]
fn optimizer_options_reject_nan_betas() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let shapes = [&[2, 3][..], &[3, 4][..]];
    let options = ContractionOptimizerOptions {
        betas: vec![f64::NAN],
        ..ContractionOptimizerOptions::default()
    };

    let err = match ContractionTree::optimize_with_options(&subs, &shapes, &options) {
        Ok(_) => panic!("expected NaN betas to be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::Planning { source } if source.to_string().contains("NaN")));
}

#[test]
fn optimizer_options_reject_nan_score_fields() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let shapes = [&[2, 3][..], &[3, 4][..]];
    let cases = [
        (
            "tc_weight",
            ScoreFunction::new(f64::NAN, 1.0, 0.0, f64::INFINITY),
        ),
        (
            "sc_weight",
            ScoreFunction::new(1.0, f64::NAN, 0.0, f64::INFINITY),
        ),
        (
            "rw_weight",
            ScoreFunction::new(1.0, 1.0, f64::NAN, f64::INFINITY),
        ),
        ("sc_target", ScoreFunction::new(1.0, 1.0, 0.0, f64::NAN)),
    ];

    for (field, score) in cases {
        let options = ContractionOptimizerOptions {
            score,
            ..ContractionOptimizerOptions::default()
        };

        let err = match ContractionTree::optimize_with_options(&subs, &shapes, &options) {
            Ok(_) => panic!("expected NaN score field {field} to be rejected"),
            Err(err) => err,
        };
        assert!(
            matches!(err, Error::Planning { source } if source.to_string().contains("NaN")),
            "expected InvalidArgument mentioning NaN for {field}"
        );
    }
}

#[test]
fn single_operand_optimize_with_options_rejects_invalid_options() {
    let subs = Subscripts::new(&[&[0, 1]], &[0, 1]);
    let shapes = [&[2, 3][..]];
    let options = ContractionOptimizerOptions {
        ntrials: 0,
        ..ContractionOptimizerOptions::default()
    };

    let err = match ContractionTree::optimize_with_options(&subs, &shapes, &options) {
        Ok(_) => panic!("expected invalid single-operand options to be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::Planning { source } if source.to_string().contains("ntrials")));
}

#[test]
fn optimizer_options_permit_infinite_betas() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let shapes = [&[2, 3][..], &[3, 4][..]];
    let options = ContractionOptimizerOptions {
        betas: vec![f64::INFINITY],
        ..ContractionOptimizerOptions::default()
    };

    let tree = ContractionTree::optimize_with_options(&subs, &shapes, &options).unwrap();
    assert_eq!(tree.step_count(), 1);
}

#[test]
fn single_operand_tree_reports_no_steps() {
    let subs = Subscripts::new(&[&[0, 1]], &[0, 1]);
    let tree = ContractionTree::optimize(&subs, &[&[2, 3][..]]).unwrap();

    assert_eq!(tree.step_count(), 0);
    assert_eq!(tree.step_pair(0), None);
    assert_eq!(tree.step_subscripts(0), None);
}

#[test]
fn nested_to_pairs_rejects_non_binary_nodes() {
    let nested = NestedEinsum::node(
        vec![
            NestedEinsum::leaf(0),
            NestedEinsum::leaf(1),
            NestedEinsum::leaf(2),
        ],
        OmecoEinCode::new(vec![vec![0], vec![0], vec![0]], vec![]),
    );
    let mut next_operand = 3;
    let mut pairs = Vec::new();

    let err = nested_to_pairs(&nested, &mut next_operand, &mut pairs).unwrap_err();
    assert!(matches!(err, Error::Planning { source } if source.to_string().contains("non-binary")));
}

#[test]
fn self_greedy_pair_optimizer_returns_valid_sequence() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let shapes = [&[2, 3][..], &[3, 5][..], &[5, 7][..]];
    let size_dict = crate::util::build_size_dict(&subs, &shapes, None).unwrap();

    let pairs = optimize_self_greedy_pairs(&subs, &size_dict).unwrap();

    assert_eq!(pairs.len(), 2);
    assert_eq!(pairs[0], (0, 1));
    assert_eq!(pairs[1], (2, 3));
}

#[test]
fn self_greedy_pair_optimizer_rejects_missing_needed_label() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 4], &[4, 2]], &[0, 2]);
    let size_dict: HashMap<u32, usize> = [(0, 2), (1, 3), (2, 5)].into_iter().collect();

    let err = optimize_self_greedy_pairs(&subs, &size_dict).unwrap_err();

    assert!(
        matches!(err, Error::Planning { source } if source.to_string().contains("unknown size for label 4"))
    );
}

#[test]
fn self_greedy_precomputed_needed_counts_match_rebuilt_needed_sets() {
    let operand_subs = [vec![0, 1, 1], vec![1, 2], vec![2, 3], vec![3, 4]];
    let output_subs = vec![0, 4];
    let available = vec![0, 1, 2, 3];
    let operand_label_sets: Vec<HashSet<u32>> = operand_subs
        .iter()
        .map(|labels| labels.iter().copied().collect())
        .collect();
    let needed_label_counts =
        build_needed_label_counts(&output_subs, &available, &operand_label_sets);
    let mut actual = Vec::new();

    for i in 0..available.len() {
        for j in (i + 1)..available.len() {
            let left = available[i];
            let right = available[j];
            let mut rebuilt_needed: HashSet<u32> = output_subs.iter().copied().collect();
            for &idx in &available {
                if idx != left && idx != right {
                    rebuilt_needed.extend(operand_subs[idx].iter().copied());
                }
            }
            let expected = crate::util::intermediate_subs(
                &operand_subs[left],
                &operand_subs[right],
                &rebuilt_needed,
            );

            collect_candidate_intermediate_subs(
                &operand_subs[left],
                &operand_subs[right],
                left,
                right,
                &operand_label_sets,
                &needed_label_counts,
                &mut actual,
            );

            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn from_pairs_rejects_duplicate_pair_indices() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let shapes = [&[2, 3][..], &[3, 4][..]];
    let result = ContractionTree::from_pairs(&subs, &shapes, &[(0, 0)]);
    match result {
        Err(Error::Planning { source }) if source.to_string().contains("distinct") => {}
        other => panic!(
            "expected InvalidArgument with 'distinct', got: {:?}",
            other.as_ref().map(|_| "Ok").map_err(|e| e.to_string())
        ),
    }
}

#[test]
fn from_pairs_rejects_pair_referencing_nonexistent_operand() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let shapes = [&[2, 3][..], &[3, 4][..]];
    let result = ContractionTree::from_pairs(&subs, &shapes, &[(0, 5)]);
    match result {
        Err(Error::Planning { source }) if source.to_string().contains("non-existent") => {}
        other => panic!(
            "expected InvalidArgument with 'non-existent', got: {:?}",
            other.as_ref().map(|_| "Ok").map_err(|e| e.to_string())
        ),
    }
}

#[test]
fn from_pairs_rejects_pair_referencing_dead_operand() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let shapes = [&[2, 2][..], &[2, 2][..], &[2, 2][..]];
    let result = ContractionTree::from_pairs(&subs, &shapes, &[(0, 1), (0, 3)]);
    match result {
        Err(Error::Planning { source }) if source.to_string().contains("no longer live") => {}
        other => panic!(
            "expected InvalidArgument with 'no longer live', got: {:?}",
            other.as_ref().map(|_| "Ok").map_err(|e| e.to_string())
        ),
    }
}

#[test]
fn from_pairs_rejects_wrong_step_count() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let shapes = [&[2, 2][..], &[2, 2][..], &[2, 2][..]];
    let result = ContractionTree::from_pairs(&subs, &shapes, &[(0, 1)]);
    match result {
        Err(Error::Planning { source }) if source.to_string().contains("must have") => {}
        other => panic!(
            "expected InvalidArgument with 'must have', got: {:?}",
            other.as_ref().map(|_| "Ok").map_err(|e| e.to_string())
        ),
    }
}

#[test]
fn optimize_binary_matches_explicit_pair_for_general_labels_and_shapes() {
    let cases: &[(&str, &[&[usize]])] = &[
        ("ij,jk->ki", &[&[2, 3], &[3, 4]]),
        ("bij,bjk->bik", &[&[2, 3, 4], &[2, 4, 5]]),
        ("ii,jj->", &[&[2, 2], &[3, 3]]),
        ("i,j->ji", &[&[2], &[3]]),
        ("αβ,βγ->γα", &[&[2, 3], &[3, 4]]),
    ];
    for &(notation, shapes) in cases {
        let subs = Subscripts::parse(notation).unwrap();
        let automatic = ContractionTree::optimize(&subs, shapes).unwrap();
        let explicit = ContractionTree::from_pairs(&subs, shapes, &[(0, 1)]).unwrap();
        assert_eq!(automatic.step_pair(0), explicit.step_pair(0), "{notation}");
        assert_eq!(automatic.size_dict, explicit.size_dict, "{notation}");
        assert_eq!(automatic.operand_subs, explicit.operand_subs, "{notation}");
        assert_eq!(
            automatic.step_subscripts(0),
            explicit.step_subscripts(0),
            "{notation}"
        );
    }
}

#[test]
fn optimize_binary_preserves_shape_validation() {
    let subs = Subscripts::parse("ij,jk->ik").unwrap();
    let invalid_shapes: &[&[&[usize]]] = &[&[], &[&[2, 3]], &[&[2], &[3, 4]], &[&[2, 3], &[5, 4]]];
    for shapes in invalid_shapes {
        let automatic = ContractionTree::optimize(&subs, shapes).unwrap_err();
        let explicit = ContractionTree::from_pairs(&subs, shapes, &[(0, 1)]).unwrap_err();
        assert_eq!(automatic.to_string(), explicit.to_string());
    }
}

#[test]
fn optimize_single_operand_returns_tree_with_no_steps() {
    let subs = Subscripts::new(&[&[0, 1]], &[0, 1]);
    let tree = ContractionTree::optimize(&subs, &[&[3, 4][..]]).unwrap();
    assert_eq!(tree.step_count(), 0);
    assert_eq!(tree.step_pair(0), None);
}

#[test]
fn optimize_two_operands_returns_single_pair() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let shapes = [&[2, 3][..], &[3, 4][..]];
    let tree = ContractionTree::optimize_with_options(
        &subs,
        &shapes,
        &ContractionOptimizerOptions::default(),
    )
    .unwrap();
    assert_eq!(tree.step_count(), 1);
    assert_eq!(tree.step_pair(0), Some((0, 1)));
}

#[test]
fn step_subscripts_returns_correct_labels_for_each_step() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let shapes = [&[2, 3][..], &[3, 4][..], &[4, 5][..]];
    let tree = ContractionTree::from_pairs(&subs, &shapes, &[(1, 2), (0, 3)]).unwrap();

    let (lhs0, rhs0, out0) = tree.step_subscripts(0).unwrap();
    assert_eq!(lhs0, &[1, 2]);
    assert_eq!(rhs0, &[2, 3]);
    assert_eq!(out0, &[1, 3]);

    let (lhs1, rhs1, out1) = tree.step_subscripts(1).unwrap();
    assert_eq!(lhs1, &[0, 1]);
    assert_eq!(rhs1, &[1, 3]);
    assert_eq!(out1, &[0, 3]);
}

#[test]
fn step_subscripts_returns_requested_final_output_order() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[2, 0]);
    let shapes = [&[2, 3][..], &[3, 4][..]];
    let tree = ContractionTree::from_pairs(&subs, &shapes, &[(0, 1)]).unwrap();

    let (lhs, rhs, out) = tree.step_subscripts(0).unwrap();

    assert_eq!(lhs, &[0, 1]);
    assert_eq!(rhs, &[1, 2]);
    assert_eq!(out, &[2, 0]);
}

#[test]
fn self_greedy_with_four_operands_chooses_cheapest_pairs() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3], &[3, 4]], &[0, 4]);
    let size_dict: HashMap<u32, usize> = [(0, 10), (1, 2), (2, 10), (3, 2), (4, 10)]
        .iter()
        .cloned()
        .collect();
    let pairs = optimize_self_greedy_pairs(&subs, &size_dict).unwrap();
    assert_eq!(pairs.len(), 3);
    let mut live: Vec<usize> = (0..4).collect();
    for (next, &(l, r)) in (4..).zip(pairs.iter()) {
        assert!(l < next);
        assert!(r < next);
        assert_ne!(l, r);
        live.retain(|&x| x != l && x != r);
        live.push(next);
    }
    assert_eq!(live.len(), 1);
}

#[test]
fn nested_to_pairs_handles_leaf_nodes() {
    let nested = NestedEinsum::leaf(42);
    let mut next_operand = 43;
    let mut pairs = Vec::new();
    let result = nested_to_pairs(&nested, &mut next_operand, &mut pairs).unwrap();
    assert_eq!(result, 42);
    assert!(pairs.is_empty());
}

#[test]
fn from_pairs_with_two_operands_builds_single_step_tree() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let shapes = [&[3, 4][..], &[4, 5][..]];
    let tree = ContractionTree::from_pairs(&subs, &shapes, &[(0, 1)]).unwrap();

    assert_eq!(tree.step_count(), 1);
    assert_eq!(tree.step_pair(0), Some((0, 1)));
    assert_eq!(tree.step_pair(1), None);

    let (lhs, rhs, out) = tree.step_subscripts(0).unwrap();
    assert_eq!(lhs, &[0, 1]);
    assert_eq!(rhs, &[1, 2]);
    assert_eq!(out, &[0, 2]);
}

#[test]
fn public_lowering_step_plan_exposes_gemm_layout() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();

    let step = tree.step_plan(0).expect("one pairwise step");
    let gemm = step.gemm();

    assert_eq!(gemm.left_only_modes(), &[0]);
    assert_eq!(gemm.right_only_modes(), &[2]);
    assert_eq!(gemm.contracted_modes(), &[1]);
    assert_eq!(gemm.batch_modes(), &[] as &[u32]);
    assert_eq!(gemm.left_only_shape(), &[2]);
    assert_eq!(gemm.right_only_shape(), &[4]);
    assert_eq!(gemm.contracted_shape(), &[3]);
    assert_eq!(gemm.batch_shape(), &[] as &[usize]);
    assert_eq!(gemm.m(), 2);
    assert_eq!(gemm.k(), 3);
    assert_eq!(gemm.n(), 4);
    assert_eq!(gemm.lhs_gemm_shape(), &[2, 3]);
    assert_eq!(gemm.rhs_gemm_shape(), &[3, 4]);
    assert_eq!(gemm.output_gemm_shape(), &[2, 4]);
    assert!(!gemm.needs_final_permute());
}

#[test]
fn public_lowering_step_plan_reports_final_permutation() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[2, 0]);
    let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();

    let gemm = tree.step_plan(0).unwrap().gemm();

    assert_eq!(gemm.canonical_output_modes(), &[0, 2]);
    assert!(gemm.needs_final_permute());
}

#[test]
fn public_lowering_step_plan_exposes_multi_contracted_shape() {
    let subs = Subscripts::new(&[&[0, 1, 2], &[1, 2, 3]], &[0, 3]);
    let tree = ContractionTree::from_pairs(&subs, &[&[2, 3, 4], &[3, 4, 5]], &[(0, 1)]).unwrap();

    let gemm = tree.step_plan(0).unwrap().gemm();

    assert_eq!(gemm.left_only_modes(), &[0]);
    assert_eq!(gemm.right_only_modes(), &[3]);
    assert_eq!(gemm.contracted_modes(), &[1, 2]);
    assert_eq!(gemm.left_only_shape(), &[2]);
    assert_eq!(gemm.right_only_shape(), &[5]);
    assert_eq!(gemm.contracted_shape(), &[3, 4]);
    assert_eq!(gemm.batch_shape(), &[] as &[usize]);
}

#[test]
fn public_lowering_step_plan_preserves_zero_sized_gemm_metadata() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let tree = ContractionTree::from_pairs(&subs, &[&[0, 3], &[3, 4]], &[(0, 1)]).unwrap();

    let gemm = tree.step_plan(0).unwrap().gemm();

    assert_eq!(gemm.left_only_modes(), &[0]);
    assert_eq!(gemm.m(), 0);
    assert_eq!(gemm.k(), 3);
    assert_eq!(gemm.n(), 4);
    assert_eq!(gemm.lhs_gemm_shape(), &[0, 3]);
    assert_eq!(gemm.output_gemm_shape(), &[0, 4]);
    assert_eq!(gemm.expanded_output_shape(), &[0, 4]);
}

#[test]
fn public_lowering_step_plan_uses_identity_for_empty_mode_groups() {
    let subs = Subscripts::new(&[&[0], &[0]], &[]);
    let tree = ContractionTree::from_pairs(&subs, &[&[3], &[3]], &[(0, 1)]).unwrap();

    let gemm = tree.step_plan(0).unwrap().gemm();

    assert_eq!(gemm.left_only_modes(), &[] as &[u32]);
    assert_eq!(gemm.right_only_modes(), &[] as &[u32]);
    assert_eq!(gemm.m(), 1);
    assert_eq!(gemm.n(), 1);
    assert_eq!(gemm.k(), 3);
    assert_eq!(gemm.lhs_gemm_shape(), &[1, 3]);
    assert_eq!(gemm.rhs_gemm_shape(), &[3, 1]);
    assert_eq!(gemm.output_gemm_shape(), &[1, 1]);
    assert_eq!(gemm.expanded_output_shape(), &[] as &[usize]);
}

#[test]
fn public_lowering_step_plan_exposes_diagonal_stages_for_both_operands() {
    let subs = Subscripts::new(&[&[0, 0, 1], &[1, 2, 2]], &[0, 2]);
    let tree = ContractionTree::from_pairs(&subs, &[&[3, 3, 4], &[4, 5, 5]], &[(0, 1)]).unwrap();

    let step = tree.step_plan(0).unwrap();
    let lhs_diag = step.lhs_diag().unwrap();
    let rhs_diag = step.rhs_diag().unwrap();

    let mut lhs_stages = lhs_diag.stages();
    assert_eq!(lhs_stages.len(), 1);
    let lhs_stage = lhs_stages.next().unwrap();
    assert_eq!(lhs_stage.axis_pairs(), &[(0, 1)]);
    assert_eq!(lhs_stage.result_subs(), &[1, 0]);
    assert_eq!(lhs_diag.result_subs(), &[1, 0]);

    let mut rhs_stages = rhs_diag.stages();
    assert_eq!(rhs_stages.len(), 1);
    let rhs_stage = rhs_stages.next().unwrap();
    assert_eq!(rhs_stage.axis_pairs(), &[(1, 2)]);
    assert_eq!(rhs_stage.result_subs(), &[1, 2]);
    assert_eq!(rhs_diag.result_subs(), &[1, 2]);
}

#[test]
fn public_lowering_step_plan_exposes_pre_reduction_plans_for_both_operands() {
    let subs = Subscripts::new(&[&[0, 1, 3], &[1, 2, 4]], &[2]);
    let tree = ContractionTree::from_pairs(&subs, &[&[2, 3, 5], &[3, 4, 7]], &[(0, 1)]).unwrap();

    let step = tree.step_plan(0).unwrap();
    let lhs_reduce = step.lhs_reduce().unwrap();
    let rhs_reduce = step.rhs_reduce().unwrap();

    assert_eq!(lhs_reduce.original_subs(), &[0, 1, 3]);
    assert_eq!(lhs_reduce.kept_subs(), &[1]);
    assert_eq!(lhs_reduce.out_shape(), &[3]);
    assert_eq!(rhs_reduce.original_subs(), &[1, 2, 4]);
    assert_eq!(rhs_reduce.kept_subs(), &[1, 2]);
    assert_eq!(rhs_reduce.out_shape(), &[3, 4]);
}

#[test]
fn public_lowering_step_plan_exposes_multi_batch_layout() {
    let subs = Subscripts::new(&[&[3, 4, 0, 1], &[1, 2, 3, 4]], &[3, 4, 0, 2]);
    let tree =
        ContractionTree::from_pairs(&subs, &[&[5, 6, 2, 3], &[3, 4, 5, 6]], &[(0, 1)]).unwrap();

    let gemm = tree.step_plan(0).unwrap().gemm();

    assert_eq!(gemm.left_only_modes(), &[0]);
    assert_eq!(gemm.right_only_modes(), &[2]);
    assert_eq!(gemm.contracted_modes(), &[1]);
    assert_eq!(gemm.batch_modes(), &[3, 4]);
    assert_eq!(gemm.left_only_shape(), &[2]);
    assert_eq!(gemm.right_only_shape(), &[4]);
    assert_eq!(gemm.contracted_shape(), &[3]);
    assert_eq!(gemm.batch_shape(), &[5, 6]);
    assert_eq!(gemm.lhs_target_modes(), &[0, 1, 3, 4]);
    assert_eq!(gemm.rhs_target_modes(), &[1, 2, 3, 4]);
    assert_eq!(gemm.canonical_output_modes(), &[0, 2, 3, 4]);
    assert_eq!(gemm.lhs_gemm_shape(), &[2, 3, 5, 6]);
    assert_eq!(gemm.rhs_gemm_shape(), &[3, 4, 5, 6]);
    assert_eq!(gemm.output_gemm_shape(), &[2, 4, 5, 6]);
    assert_eq!(gemm.expanded_output_shape(), &[2, 4, 5, 6]);
    assert!(gemm.needs_final_permute());
}

#[test]
fn public_lowering_wrapper_types_implement_debug() {
    let diag_subs = Subscripts::new(&[&[0, 0, 1], &[1, 2, 2]], &[0, 2]);
    let diag_tree =
        ContractionTree::from_pairs(&diag_subs, &[&[3, 3, 4], &[4, 5, 5]], &[(0, 1)]).unwrap();
    let step = diag_tree.step_plan(0).unwrap();
    let diag = step.lhs_diag().unwrap();
    let stage = diag.stages().next().unwrap();

    let reduce_subs = Subscripts::new(&[&[0, 1, 3], &[1, 2, 4]], &[2]);
    let reduce_tree =
        ContractionTree::from_pairs(&reduce_subs, &[&[2, 3, 5], &[3, 4, 7]], &[(0, 1)]).unwrap();
    let reduce = reduce_tree.step_plan(0).unwrap().lhs_reduce().unwrap();

    let _ = format!("{step:?}");
    let _ = format!("{diag:?}");
    let _ = format!("{stage:?}");
    let _ = format!("{reduce:?}");
    let _ = format!("{:?}", step.gemm());
}

#[test]
fn optimize_with_annealing_schedule_produces_valid_tree() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3], &[3, 4], &[4, 5]], &[0, 5]);
    let shapes = [
        &[2, 3][..],
        &[3, 4][..],
        &[4, 3][..],
        &[3, 2][..],
        &[2, 5][..],
    ];
    let options = ContractionOptimizerOptions {
        betas: vec![0.5, 1.0, 2.0],
        ntrials: 2,
        niters: 3,
        score: ScoreFunction::space_optimized(8.0),
    };

    let tree = ContractionTree::optimize_with_options(&subs, &shapes, &options).unwrap();
    assert_eq!(tree.step_count(), subs.inputs.len() - 1);
}

#[test]
fn step_pair_returns_none_for_out_of_bounds() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let shapes = [&[2, 3][..], &[3, 4][..]];
    let tree = ContractionTree::from_pairs(&subs, &shapes, &[(0, 1)]).unwrap();

    assert_eq!(tree.step_pair(0), Some((0, 1)));
    assert_eq!(tree.step_pair(5), None);
    assert_eq!(tree.step_subscripts(5), None);
}

#[test]
fn optimize_with_space_optimized_score_builds_tree() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let shapes = [&[3, 4][..], &[4, 5][..], &[5, 6][..]];
    let options = ContractionOptimizerOptions {
        score: ScoreFunction::space_optimized(8.0),
        ..ContractionOptimizerOptions::default()
    };

    let tree = ContractionTree::optimize_with_options(&subs, &shapes, &options).unwrap();
    assert_eq!(tree.step_count(), 2);
}
