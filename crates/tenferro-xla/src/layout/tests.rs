use super::{col_major_byte_strides, col_major_to_row_major, row_major_to_col_major};

#[test]
fn converts_2d_col_major_host_buffer_to_row_major_for_xla() {
    let row_major = col_major_to_row_major(&[2, 3], &[1, 2, 3, 4, 5, 6]).unwrap();

    assert_eq!(row_major, vec![1, 3, 5, 2, 4, 6]);
}

#[test]
fn converts_2d_row_major_xla_buffer_back_to_col_major() {
    let col_major = row_major_to_col_major(&[2, 3], &[1, 3, 5, 2, 4, 6]).unwrap();

    assert_eq!(col_major, vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn scalar_and_vector_layouts_are_identity() {
    assert_eq!(col_major_to_row_major::<i32>(&[], &[7]).unwrap(), vec![7]);
    assert_eq!(
        row_major_to_col_major(&[3], &[1.0_f64, 2.0, 3.0]).unwrap(),
        vec![1.0, 2.0, 3.0]
    );
}

#[test]
fn layout_conversion_rejects_wrong_element_count() {
    let err = col_major_to_row_major(&[2, 2], &[1, 2, 3]).unwrap_err();

    assert!(err.to_string().contains("expected 4 elements"));
}

#[test]
fn computes_column_major_byte_strides_for_pjrt_upload() {
    assert_eq!(col_major_byte_strides::<f32>(&[2, 3]).unwrap(), vec![4, 8]);
    assert_eq!(
        col_major_byte_strides::<f64>(&[2, 3, 4]).unwrap(),
        vec![8, 16, 48]
    );
}
