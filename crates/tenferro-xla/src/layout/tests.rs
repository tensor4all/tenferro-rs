use super::col_major_byte_strides;
use crate::Error;

#[test]
fn computes_column_major_byte_strides_for_pjrt_upload() {
    assert_eq!(col_major_byte_strides::<f32>(&[2, 3]).unwrap(), vec![4, 8]);
    assert_eq!(
        col_major_byte_strides::<f64>(&[2, 3, 4]).unwrap(),
        vec![8, 16, 48]
    );
}

#[test]
fn column_major_byte_strides_handle_scalars_and_vectors() {
    assert_eq!(
        col_major_byte_strides::<f64>(&[]).unwrap(),
        Vec::<i64>::new()
    );
    assert_eq!(col_major_byte_strides::<f32>(&[5]).unwrap(), vec![4]);
    assert_eq!(col_major_byte_strides::<f64>(&[0]).unwrap(), vec![8]);
}

#[test]
fn column_major_byte_strides_handle_singleton_and_empty_axes() {
    assert_eq!(
        col_major_byte_strides::<f32>(&[2, 1, 3]).unwrap(),
        vec![4, 8, 8]
    );
    assert_eq!(
        col_major_byte_strides::<f64>(&[2, 0, 3]).unwrap(),
        vec![8, 16, 0]
    );
}

#[test]
fn column_major_byte_strides_differ_from_row_major_for_rectangular_shapes() {
    let strides = col_major_byte_strides::<f64>(&[2, 5, 3]).unwrap();
    assert_eq!(strides, vec![8, 16, 80]);
    assert_ne!(strides, vec![120, 24, 8]);
}

#[test]
fn column_major_byte_strides_reject_byte_stride_overflow() {
    let shape = [usize::MAX / std::mem::size_of::<f64>() + 1, 1];
    assert!(matches!(
        col_major_byte_strides::<f64>(&shape),
        Err(Error::InvalidProgram { message }) if message.contains("byte stride overflows usize")
    ));
}

#[test]
#[cfg(target_pointer_width = "64")]
fn column_major_byte_strides_reject_pjrt_stride_overflow() {
    let shape = [i64::MAX as usize / std::mem::size_of::<f32>() + 1, 1];
    assert!(matches!(
        col_major_byte_strides::<f32>(&shape),
        Err(Error::InvalidProgram { message }) if message.contains("exceeds i64 for PJRT")
    ));
}

#[test]
fn column_major_byte_strides_reject_element_stride_overflow() {
    let shape = [usize::MAX / 16, 32];
    assert!(matches!(
        col_major_byte_strides::<f32>(&shape),
        Err(Error::InvalidProgram { message }) if message.contains("element stride overflows usize")
    ));
}
