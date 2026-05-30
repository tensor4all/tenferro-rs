use std::collections::HashSet;

use std::collections::HashMap;

use omeco::{EinCode as OmecoEinCode, Initializer, NestedEinsum, ScoreFunction};
use tenferro_device::Error;

use crate::syntax::subscripts::Subscripts;

use super::{
    nested_to_pairs, optimize_self_greedy_pairs, ContractionOptimizerOptions, ContractionTree,
};

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
    assert!(matches!(err, Error::InvalidArgument(message) if message.contains("ntrials")));
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
    assert!(matches!(err, Error::InvalidArgument(message) if message.contains("non-binary")));
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
        matches!(err, Error::InvalidArgument(message) if message.contains("unknown size for label 4"))
    );
}

#[test]
fn from_pairs_rejects_duplicate_pair_indices() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let shapes = [&[2, 3][..], &[3, 4][..]];
    let result = ContractionTree::from_pairs(&subs, &shapes, &[(0, 0)]);
    match result {
        Err(Error::InvalidArgument(msg)) if msg.contains("distinct") => {}
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
        Err(Error::InvalidArgument(msg)) if msg.contains("non-existent") => {}
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
        Err(Error::InvalidArgument(msg)) if msg.contains("no longer live") => {}
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
        Err(Error::InvalidArgument(msg)) if msg.contains("must have") => {}
        other => panic!(
            "expected InvalidArgument with 'must have', got: {:?}",
            other.as_ref().map(|_| "Ok").map_err(|e| e.to_string())
        ),
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
fn optimize_with_options_falls_back_to_self_greedy_when_omeco_returns_none() {
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
