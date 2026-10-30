use super::col_major_byte_strides;

#[test]
fn computes_column_major_byte_strides_for_pjrt_upload() {
    assert_eq!(col_major_byte_strides::<f32>(&[2, 3]).unwrap(), vec![4, 8]);
    assert_eq!(
        col_major_byte_strides::<f64>(&[2, 3, 4]).unwrap(),
        vec![8, 16, 48]
    );
}
