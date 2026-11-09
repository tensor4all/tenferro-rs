use tenferro_ops::sym_dim::SymDim;

use super::*;

fn symbolic(axis: usize) -> SymDim {
    SymDim::tensor_axis(0, axis)
}

#[test]
fn classification_covers_every_known_and_symbolic_pairing() {
    assert_eq!(classify_sizes(3, 3), ExtentPair::Equal);
    assert_eq!(classify_sizes(1, 3), ExtentPair::LhsSingleton);
    assert_eq!(classify_sizes(3, 1), ExtentPair::RhsSingleton);
    assert_eq!(classify_sizes(2, 3), ExtentPair::Conflict);
    assert_eq!(
        classify_sym_dims(&symbolic(0), &symbolic(0)),
        ExtentPair::Equal
    );
    assert_eq!(
        classify_sym_dims(&SymDim::from(1usize), &symbolic(0)),
        ExtentPair::LhsSingleton
    );
    assert_eq!(
        classify_sym_dims(&symbolic(0), &SymDim::from(1usize)),
        ExtentPair::RhsSingleton
    );
    assert_eq!(
        classify_sym_dims(&SymDim::from(3usize), &symbolic(0)),
        ExtentPair::LhsConstant
    );
    assert_eq!(
        classify_sym_dims(&symbolic(0), &SymDim::from(3usize)),
        ExtentPair::RhsConstant
    );
    assert_eq!(
        classify_sym_dims(&symbolic(0), &symbolic(1)),
        ExtentPair::BothSymbolic
    );
}

#[test]
fn merge_keeps_equality_without_broadcast() {
    assert_eq!(
        merge_broadcast_extent(&symbolic(0), &SymDim::from(3usize), false),
        ExtentMerge::RequireEqual
    );
    assert_eq!(
        merge_broadcast_extent(&SymDim::from(1usize), &SymDim::from(3usize), false),
        ExtentMerge::RequireEqual
    );
}

#[test]
fn merge_of_symbolic_and_constant_extents_takes_the_constant() {
    // A symbolic extent bound to 1 or 3 broadcasts against a constant 3 to 3,
    // in either fold order.
    assert_eq!(
        merge_broadcast_extent(&symbolic(0), &SymDim::from(3usize), true),
        ExtentMerge::Merged(SymDim::from(3usize))
    );
    assert_eq!(
        merge_broadcast_extent(&SymDim::from(3usize), &symbolic(0), true),
        ExtentMerge::Merged(SymDim::from(3usize))
    );
    // A known singleton yields to whatever the other side is.
    assert_eq!(
        merge_broadcast_extent(&SymDim::from(1usize), &symbolic(0), true),
        ExtentMerge::Merged(symbolic(0))
    );
    assert_eq!(
        merge_broadcast_extent(&symbolic(0), &SymDim::from(1usize), true),
        ExtentMerge::Merged(symbolic(0))
    );
}

#[test]
fn merge_of_two_symbolic_extents_is_their_maximum() {
    assert_eq!(
        merge_broadcast_extent(&symbolic(0), &symbolic(1), true),
        ExtentMerge::Merged(symbolic(0).max(symbolic(1)))
    );
    assert_eq!(
        merge_broadcast_extent(&symbolic(0), &symbolic(0), true),
        ExtentMerge::Merged(symbolic(0))
    );
}

#[test]
fn merge_of_known_conflict_requires_equality() {
    assert_eq!(
        merge_broadcast_extent(&SymDim::from(2usize), &SymDim::from(3usize), true),
        ExtentMerge::RequireEqual
    );
}

#[test]
fn reverse_mode_sums_a_known_singleton_that_was_broadcast() {
    // Active extent 1 against a peer 3: summed over the broadcast extent.
    assert_eq!(
        broadcast_axis_reduce([classify_sizes(1, 3)]),
        Ok(AxisCotangent::SumBroadcast)
    );
    // ... also when another peer is 1, or the peer is symbolic.
    assert_eq!(
        broadcast_axis_reduce([classify_sizes(1, 1), classify_sizes(1, 3)]),
        Ok(AxisCotangent::SumBroadcast)
    );
    assert_eq!(
        broadcast_axis_reduce([classify_sym_dims(&SymDim::from(1usize), &symbolic(0))]),
        Ok(AxisCotangent::SumBroadcast)
    );
    // Active extent 1 against a peer 1: nothing was broadcast.
    assert_eq!(
        broadcast_axis_reduce([classify_sizes(1, 1)]),
        Ok(AxisCotangent::Keep)
    );
}

#[test]
fn reverse_mode_expands_an_axis_only_singleton_peers_broadcast_to() {
    // Active extent 3 against peers that are all 1: constant along the axis.
    assert_eq!(
        broadcast_axis_reduce([classify_sizes(3, 1)]),
        Ok(AxisCotangent::ExpandSingletonPeers)
    );
    // One peer of the same extent already carries the axis at 3.
    assert_eq!(
        broadcast_axis_reduce([classify_sizes(3, 1), classify_sizes(3, 3)]),
        Ok(AxisCotangent::Keep)
    );
    assert_eq!(
        broadcast_axis_reduce([
            classify_sym_dims(&SymDim::from(3usize), &symbolic(0)),
            classify_sizes(3, 3),
        ]),
        Ok(AxisCotangent::Keep)
    );
    // The label occurs nowhere else.
    assert_eq!(broadcast_axis_reduce([]), Ok(AxisCotangent::Keep));
    assert!(AxisCotangent::SumBroadcast.sums_label_out());
    assert!(AxisCotangent::ExpandSingletonPeers.sums_label_out());
    assert!(!AxisCotangent::Keep.sums_label_out());
}

#[test]
fn reverse_mode_refuses_to_guess_an_undecided_axis() {
    // Symbolic active extent bound to 1 or 3 against a constant 3.
    assert_eq!(
        broadcast_axis_reduce([classify_sym_dims(&symbolic(0), &SymDim::from(3usize))]),
        Err(UndecidedBroadcastAxis)
    );
    assert_eq!(
        broadcast_axis_reduce([classify_sym_dims(&symbolic(0), &symbolic(1))]),
        Err(UndecidedBroadcastAxis)
    );
    // Constant active extent 3 whose only peer is symbolic: the peer may be 3
    // (keep) or a broadcast 1 (expand).
    assert_eq!(
        broadcast_axis_reduce([classify_sym_dims(&SymDim::from(3usize), &symbolic(0))]),
        Err(UndecidedBroadcastAxis)
    );
    // The same symbolic extent on both sides is provably not broadcast.
    assert_eq!(
        broadcast_axis_reduce([classify_sym_dims(&symbolic(0), &symbolic(0))]),
        Ok(AxisCotangent::Keep)
    );
}

#[test]
fn broadcast_decision_only_counts_labels_shared_between_operands() {
    assert!(requires_broadcast(
        &[&[2, 1], &[4, 2, 3]],
        &[vec![0, 1], vec![2, 3, 1]]
    ));
    assert!(!requires_broadcast(
        &[&[2, 3], &[4, 2, 3]],
        &[vec![0, 1], vec![2, 3, 1]]
    ));
    // A label repeated inside one operand is a diagonal, whose extents must
    // match exactly; it is never a singleton broadcast.
    assert!(!requires_broadcast(&[&[1, 3]], &[vec![0, 0]]));
    // A symbolic extent is skipped, not guessed.
    assert!(requires_broadcast(
        &[&[None, Some(1)], &[Some(4), Some(2), Some(3)]],
        &[vec![0, 1], vec![2, 3, 1]]
    ));
    assert!(!requires_broadcast(
        &[&[None, Some(2)], &[Some(4), Some(2), Some(3)]],
        &[vec![0, 1], vec![2, 3, 1]]
    ));
}
