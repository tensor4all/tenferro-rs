use super::checked_product;

#[test]
fn checked_matrix_element_count_reports_typed_overflow() {
    let err = checked_product("test_faer_allocation", "matrix", &[usize::MAX, 2])
        .expect_err("overflowing matrix dimensions must fail");

    assert!(matches!(
        err,
        tenferro_tensor::Error::Validation {
            op: "test_faer_allocation",
            ..
        }
    ));
}
