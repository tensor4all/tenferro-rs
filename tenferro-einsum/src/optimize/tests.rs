use super::*;

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
