use super::*;
use tenferro_tensor::{MemoryOrder, Tensor};

fn make(data: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn tensor_construction_and_extraction_helpers_keep_column_major_layout() {
    let tensor = tensor_from_data(vec![1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert_eq!(tensor.dims(), &[2, 2]);
    assert_eq!(tensor.strides(), &[1, 2]);

    let scalar_tensor = tensor_from_data_scalar(vec![1_i32, 2, 3], &[3]).unwrap();
    assert_eq!(scalar_tensor.dims(), &[3]);
    assert_eq!(scalar_tensor.strides(), &[1]);

    let offset_tensor = Tensor::from_vec(vec![9.0_f64, 1.0, 2.0], &[2], &[1], 1).unwrap();
    let (data, offset) = extract_data(&offset_tensor).unwrap();
    assert_eq!(offset, 0);
    assert_eq!(data, vec![1.0, 2.0]);

    let scalar_offset = Tensor::from_vec(vec![7_i32, 4, 5], &[2], &[1], 1).unwrap();
    assert_eq!(extract_data_scalar(&scalar_offset).unwrap(), vec![4, 5]);
}

#[test]
fn ensure_col_major_repackages_row_major_views() {
    let row_major =
        Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::RowMajor).unwrap();
    let contiguous = ensure_col_major(&row_major);

    assert!(contiguous.is_contiguous());
    assert_eq!(contiguous.strides(), &[1, 2]);
    assert_eq!(
        extract_data(&contiguous).unwrap().0,
        vec![1.0, 3.0, 2.0, 4.0]
    );
}

#[test]
fn forward_perm_handles_empty_identity_and_batched_permutations() {
    let empty = tensor_from_data::<f64>(Vec::new(), &[0]).unwrap();
    assert_eq!(
        forward_perm_from_permutation_matrix(&empty, 3, 2).unwrap(),
        vec![0, 1, 2, 0, 1, 2]
    );

    let batched = make(&[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], &[2, 2, 2]);
    assert_eq!(
        forward_perm_from_permutation_matrix(&batched, 2, 2).unwrap(),
        vec![0, 1, 1, 0]
    );
}

#[test]
fn forward_perm_validates_rank_shape_batch_and_rows() {
    let rank_one = make(&[1.0, 0.0], &[2]);
    let err = forward_perm_from_permutation_matrix(&rank_one, 2, 1).unwrap_err();
    assert!(
        matches!(err, chainrules_core::AutodiffError::InvalidArgument(msg) if msg.contains("rank >= 2"))
    );

    let wrong_shape = make(&[1.0, 0.0, 0.0, 1.0, 2.0, 3.0], &[2, 3]);
    let err = forward_perm_from_permutation_matrix(&wrong_shape, 2, 1).unwrap_err();
    assert!(
        matches!(err, chainrules_core::AutodiffError::InvalidArgument(msg) if msg.contains("shape mismatch"))
    );

    let wrong_batch = make(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let err = forward_perm_from_permutation_matrix(&wrong_batch, 2, 2).unwrap_err();
    assert!(
        matches!(err, chainrules_core::AutodiffError::InvalidArgument(msg) if msg.contains("batch count"))
    );

    let missing_row = make(&[0.0, 0.0, 0.0, 1.0], &[2, 2]);
    let err = forward_perm_from_permutation_matrix(&missing_row, 2, 1).unwrap_err();
    assert!(
        matches!(err, chainrules_core::AutodiffError::InvalidArgument(msg) if msg.contains("has no nonzero entry"))
    );
}
