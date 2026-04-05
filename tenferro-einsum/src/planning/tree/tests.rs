use std::collections::HashSet;

use std::collections::HashMap;

use omeco::{EinCode as OmecoEinCode, Initializer, NestedEinsum, ScoreFunction};
use tenferro_device::Error;

use crate::syntax::subscripts::Subscripts;

use super::{
    nested_to_pairs, optimize_self_greedy_pairs, ChainAttachment, ContractionOptimizerOptions,
    ContractionStep, ContractionTree, LinearChainPlan,
};

#[test]
fn linear_chain_plan_detects_progressive_chain() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3], &[3, 4]], &[0, 4]);
    let shapes = [&[2, 3][..], &[3, 4][..], &[4, 5][..], &[5, 6][..]];
    let tree = ContractionTree::from_pairs(&subs, &shapes, &[(0, 1), (2, 4), (3, 5)]).unwrap();

    assert_eq!(
        tree.linear_chain_plan(),
        Some(LinearChainPlan {
            first_pair: (0, 1),
            attachments: vec![
                ChainAttachment {
                    prev_on_left: false,
                    operand: 2,
                },
                ChainAttachment {
                    prev_on_left: false,
                    operand: 3,
                },
            ],
        })
    );
}

#[test]
fn linear_chain_plan_rejects_branching_tree() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3], &[3, 4]], &[0, 4]);
    let shapes = [&[2, 3][..], &[3, 4][..], &[4, 5][..], &[5, 6][..]];
    let tree = ContractionTree::from_pairs(&subs, &shapes, &[(0, 1), (2, 3), (4, 5)]).unwrap();

    assert_eq!(tree.linear_chain_plan(), None);
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
    assert!(matches!(err, Error::InvalidArgument(message) if message.contains("ntrials")));
}

#[test]
fn single_operand_tree_reports_no_steps_and_empty_chain() {
    let subs = Subscripts::new(&[&[0, 1]], &[0, 1]);
    let tree = ContractionTree::optimize(&subs, &[&[2, 3][..]]).unwrap();

    assert_eq!(tree.step_count(), 0);
    assert_eq!(tree.step_pair(0), None);
    assert_eq!(tree.step_subscripts(0), None);
    assert_eq!(
        tree.linear_chain_plan(),
        Some(LinearChainPlan {
            first_pair: (0, 0),
            attachments: Vec::new(),
        })
    );
}

#[test]
fn linear_chain_plan_rejects_first_step_that_starts_from_intermediate() {
    let subs = Subscripts::new(&[&[0], &[1], &[2]], &[0, 1, 2]);
    let tree = ContractionTree {
        subscripts: subs,
        steps: vec![ContractionStep { left: 3, right: 1 }],
        size_dict: HashMap::new(),
        operand_subs: vec![vec![0], vec![1], vec![2], vec![0, 1]],
        step_output_shapes: vec![vec![2, 2]],
        step_plans: Vec::new(),
    };

    assert_eq!(tree.linear_chain_plan(), None);
}

#[test]
fn linear_chain_plan_rejects_reusing_an_input_operand() {
    let subs = Subscripts::new(&[&[0], &[1], &[2], &[3]], &[0, 1, 2, 3]);
    let tree = ContractionTree {
        subscripts: subs,
        steps: vec![
            ContractionStep { left: 0, right: 1 },
            ContractionStep { left: 4, right: 2 },
            ContractionStep { left: 5, right: 2 },
        ],
        size_dict: HashMap::new(),
        operand_subs: vec![
            vec![0],
            vec![1],
            vec![2],
            vec![3],
            vec![0, 1],
            vec![0, 1, 2],
            vec![0, 1, 2, 3],
        ],
        step_output_shapes: vec![vec![2], vec![2], vec![2]],
        step_plans: Vec::new(),
    };

    assert_eq!(tree.linear_chain_plan(), None);
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

    let pairs = optimize_self_greedy_pairs(&subs, &size_dict);

    assert_eq!(pairs.len(), 2);
    assert_eq!(pairs[0], (0, 1));
    assert_eq!(pairs[1], (2, 3));
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
fn linear_chain_plan_accepts_prev_on_right_for_first_attachment() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let shapes = [&[2, 2][..], &[2, 2][..], &[2, 2][..]];
    let tree = ContractionTree::from_pairs(&subs, &shapes, &[(0, 1), (2, 3)]).unwrap();
    let plan = tree.linear_chain_plan().unwrap();
    assert_eq!(plan.first_pair, (0, 1));
    assert_eq!(plan.attachments.len(), 1);
    assert!(!plan.attachments[0].prev_on_left);
    assert_eq!(plan.attachments[0].operand, 2);
}

#[test]
fn self_greedy_with_four_operands_chooses_cheapest_pairs() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3], &[3, 4]], &[0, 4]);
    let size_dict: HashMap<u32, usize> = [(0, 10), (1, 2), (2, 10), (3, 2), (4, 10)]
        .iter()
        .cloned()
        .collect();
    let pairs = optimize_self_greedy_pairs(&subs, &size_dict);
    assert_eq!(pairs.len(), 3);
    let mut live: Vec<usize> = (0..4).collect();
    let mut next = 4;
    for &(l, r) in &pairs {
        assert!(l < next);
        assert!(r < next);
        assert_ne!(l, r);
        live.retain(|&x| x != l && x != r);
        live.push(next);
        next += 1;
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
