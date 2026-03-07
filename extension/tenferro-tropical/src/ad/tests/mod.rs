use super::*;
use crate::MaxPlus;

const COL: MemoryOrder = MemoryOrder::ColumnMajor;

#[test]
fn test_promote_extract_roundtrip() {
    let t = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let tropical = promote_to_tropical::<MaxPlus<f64>>(&t).unwrap();
    let back = extract_inner::<MaxPlus<f64>>(&tropical).unwrap();
    let orig_data = t.buffer().as_slice().unwrap();
    let back_data = back.buffer().as_slice().unwrap();
    for i in 0..4 {
        assert_eq!(orig_data[i], back_data[i]);
    }
}

#[test]
fn test_col_major_flat_index() {
    // 2x3 matrix
    assert_eq!(col_major_flat_index(&[2, 3], &[0, 0]), 0);
    assert_eq!(col_major_flat_index(&[2, 3], &[1, 0]), 1);
    assert_eq!(col_major_flat_index(&[2, 3], &[0, 1]), 2);
    assert_eq!(col_major_flat_index(&[2, 3], &[1, 1]), 3);
    assert_eq!(col_major_flat_index(&[2, 3], &[0, 2]), 4);
    assert_eq!(col_major_flat_index(&[2, 3], &[1, 2]), 5);
}
