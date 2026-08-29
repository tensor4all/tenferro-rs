use super::super::blas1::blas1_len;

#[test]
fn blas1_length_stays_within_the_portable_cublas_interface() {
    assert_eq!(blas1_len(i32::MAX as usize, "test").unwrap(), i32::MAX);
    let error = blas1_len(i32::MAX as usize + 1, "test").unwrap_err();
    assert!(error.to_string().contains("exceeds i32::MAX"));
}
