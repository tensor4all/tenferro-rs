use super::*;

#[test]
fn broadcast_batch_indexer_maps_column_major_indices() {
    let indexer = BroadcastBatchIndexer::new(&[1, 3], &[2, 3], "solve", "b").unwrap();
    assert_eq!(indexer.output_batch_dims(), &[2, 3]);
    assert!(!indexer.is_identity());
    let mapped: Vec<_> = (0..6)
        .map(|i| indexer.source_linear_batch_index(i))
        .collect();
    assert_eq!(mapped, vec![0, 0, 1, 1, 2, 2]);
}

#[test]
fn broadcast_batch_indexer_identity_is_reported() {
    let indexer = BroadcastBatchIndexer::new(&[2, 3], &[2, 3], "solve", "a").unwrap();
    assert!(indexer.is_identity());
    let mapped: Vec<_> = (0..6)
        .map(|i| indexer.source_linear_batch_index(i))
        .collect();
    assert_eq!(mapped, vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn broadcast_batch_indexer_rejects_rank_and_broadcast_errors() {
    let rank_err = BroadcastBatchIndexer::new(&[2, 3, 4], &[2, 3], "solve", "a").unwrap_err();
    assert!(rank_err
        .to_string()
        .contains("batch rank 3 exceeds target batch rank 2"));

    let broadcast_err = BroadcastBatchIndexer::new(&[2, 2], &[2, 3], "solve", "b").unwrap_err();
    assert!(broadcast_err.to_string().contains("not broadcastable"));
}

#[test]
fn broadcast_batch_indexer_handles_scalar_batches() {
    let indexer = BroadcastBatchIndexer::new(&[], &[], "solve", "a").unwrap();
    assert_eq!(indexer.output_batch_dims(), &[]);
    assert!(indexer.is_identity());
    assert_eq!(indexer.source_linear_batch_index(0), 0);
}

#[test]
fn broadcast_batch_dims_merges_unit_axes() {
    let dims = broadcast_batch_dims(&[2, 1], &[1, 3], "solve", "a", "b").unwrap();
    assert_eq!(dims, vec![2, 3]);
}

#[test]
fn broadcast_batch_dims_rejects_non_broadcastable_shapes() {
    let err = broadcast_batch_dims(&[2, 2], &[2, 3], "solve", "a", "b").unwrap_err();
    assert!(err.to_string().contains("not broadcastable"));
}

#[test]
fn checked_batch_count_detects_overflow() {
    let err = checked_batch_count(&[usize::MAX, 2]).unwrap_err();
    assert!(err.to_string().contains("batch iteration count overflow"));
}

#[test]
fn checked_batch_count_of_empty_shape_is_one() {
    assert_eq!(checked_batch_count(&[]).unwrap(), 1);
}

#[test]
fn flatten_col_major_index_matches_expected_linear_index() {
    let flat = flatten_col_major_index(&[2, 3, 4], &[1, 0, 2]).unwrap();
    assert_eq!(flat, 13);
}

#[test]
fn flatten_col_major_index_rejects_rank_mismatch() {
    let err = flatten_col_major_index(&[2, 3], &[1, 0, 2]).unwrap_err();
    assert!(err.to_string().contains("rank mismatch"));
}

#[test]
fn flatten_col_major_index_detects_stride_overflow() {
    let err = flatten_col_major_index(&[usize::MAX, 2], &[0, 0]).unwrap_err();
    assert!(err.to_string().contains("flatten stride overflow"));
}

#[test]
fn unflatten_col_major_index_into_matches_expected_coordinates() {
    let mut out = [0usize; 3];
    unflatten_col_major_index_into(13, &[2, 3, 4], &mut out).unwrap();
    assert_eq!(out, [1, 0, 2]);
}

#[test]
fn unflatten_col_major_index_into_rejects_rank_mismatch() {
    let mut out = [0usize; 2];
    let err = unflatten_col_major_index_into(0, &[2, 3, 4], &mut out).unwrap_err();
    assert!(err.to_string().contains("output rank mismatch"));
}

#[test]
fn unflatten_col_major_index_into_rejects_out_of_range_flat_index() {
    let mut out = [0usize; 2];
    let err = unflatten_col_major_index_into(6, &[2, 3], &mut out).unwrap_err();
    assert!(err.to_string().contains("out of range"));
}

#[test]
fn unflatten_col_major_index_into_handles_scalar_shape() {
    let mut out = [];
    unflatten_col_major_index_into(0, &[], &mut out).unwrap();
}
