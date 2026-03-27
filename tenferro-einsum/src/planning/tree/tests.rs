use std::collections::HashSet;

use omeco::{Initializer, ScoreFunction};

use crate::syntax::subscripts::Subscripts;

use super::{ChainAttachment, ContractionOptimizerOptions, ContractionTree, LinearChainPlan};

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
