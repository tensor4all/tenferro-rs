use std::collections::HashMap;

use super::{decode, omeco_pairs, to_omeco, tree_cost};
use crate::{ContractionTree, Subscripts};

#[test]
fn decode_counts_the_first_label_fastest() {
    let mut assignment = [0; 2];
    decode(5, &[2, 3], &mut assignment);
    assert_eq!(assignment, [1, 2]);
}

#[test]
fn omeco_round_trip_keeps_the_pairs() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3], &[3, 0]], &[]);
    let shapes = [&[2, 3][..], &[3, 4], &[4, 5], &[5, 2]];
    let tree = ContractionTree::from_pairs(&subs, &shapes, &[(1, 2), (0, 4), (3, 5)]).unwrap();
    let pairs = omeco_pairs(&to_omeco(&tree, |_| true), 4).unwrap();
    assert_eq!(pairs, vec![(1, 2), (0, 4), (3, 5)]);
}

#[test]
fn tree_cost_counts_flops_largest_and_peak() {
    // (ij,jk)->ik then (ik,kl)->il with i=2, j=3, k=4, l=5.
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let shapes = [&[2, 3][..], &[3, 4], &[4, 5]];
    let tree = ContractionTree::from_pairs(&subs, &shapes, &[(0, 1), (3, 2)]).unwrap();
    let sizes: HashMap<u32, f64> = [(0, 2.0), (1, 3.0), (2, 4.0), (3, 5.0)].into();
    let cost = tree_cost(&tree, |l| sizes[&l]);
    assert_eq!(cost.flops, 2.0 * 3.0 * 4.0 + 2.0 * 4.0 * 5.0);
    assert_eq!(cost.largest, 10.0);
    // ik (8) is alive while il (10) is written.
    assert_eq!(cost.peak, 18.0);
}
