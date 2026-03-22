use super::*;
use tenferro_tensor::MemoryOrder;

fn make(data: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn validate_matrix_shape_2d() {
    let a = make(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let (m, n, batch) = validate_matrix_shape(&a).unwrap();
    assert_eq!((m, n), (2, 3));
    assert!(batch.is_empty());
}

#[test]
fn validate_matrix_shape_1d_fails() {
    let a = make(&[1.0, 2.0], &[2]);
    assert!(validate_matrix_shape(&a).is_err());
}

#[test]
fn validate_square_ok() {
    let a = make(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let (n, batch) = validate_square(&a).unwrap();
    assert_eq!(n, 2);
    assert!(batch.is_empty());
}

#[test]
fn validate_square_nonsquare_fails() {
    let a = make(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert!(validate_square(&a).is_err());
}

#[test]
fn batch_count_empty() {
    assert_eq!(batch_count(&[]), 1);
}

#[test]
fn batch_count_nonempty() {
    assert_eq!(batch_count(&[2, 3]), 6);
}

#[test]
fn batch_count_zero_dim_batch_is_zero() {
    assert_eq!(batch_count(&[0]), 0);
    assert_eq!(batch_count(&[2, 0, 3]), 0);
}

#[test]
fn ensure_col_major_contiguous() {
    let a = make(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = ensure_col_major(&a);
    assert!(b.is_contiguous());
}

#[test]
fn ensure_col_major_row_major_input_is_repacked() {
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::RowMajor).unwrap();
    let b = ensure_col_major(&a);
    assert!(b.is_contiguous());
    assert_eq!(b.strides(), &[1, 2]);
    assert_eq!(extract_contiguous_slice(&b).unwrap(), &[1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn extract_contiguous_slice_ok() {
    let a = make(&[1.0, 2.0], &[2]);
    let s = extract_contiguous_slice(&a).unwrap();
    assert_eq!(s.len(), 2);
}

#[test]
fn validate_solve_rhs_shape_vector() {
    let b = make(&[1.0, 2.0], &[2]);
    let layout = validate_solve_rhs_shape(&b, 2, &[], "solve").unwrap();
    assert_eq!(layout.nrhs, 1);
    assert_eq!(layout.output_dims, vec![2]);
}

#[test]
fn validate_solve_rhs_shape_scalar_fails() {
    let b = make(&[1.0], &[]);
    assert!(validate_solve_rhs_shape(&b, 2, &[], "solve").is_err());
}

#[test]
fn zero_trailing_by_counts_linalg_wrapper_matches_tensor_helper() {
    let payload = make(&[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], &[2, 2, 2]);
    let keep_counts = make(&[1.0, 2.0], &[2]);

    let got = zero_trailing_by_counts(&payload, &keep_counts, 1, 2).unwrap();
    let expected = payload.zero_trailing_by_counts(&keep_counts, 1, 2).unwrap();

    assert_eq!(got.dims(), expected.dims());
    assert_eq!(got.strides(), expected.strides());
    assert_eq!(
        extract_contiguous_slice(&got).unwrap(),
        extract_contiguous_slice(&expected).unwrap()
    );
}
