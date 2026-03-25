use crate::dlpack::row_major_strides;

#[test]
fn row_major_strides_empty_dims() {
    let dims = &[];
    let strides = row_major_strides(dims).unwrap();
    assert!(strides.is_empty());
}

#[test]
fn row_major_strides_single_dim() {
    let dims = &[5_usize];
    let strides = row_major_strides(dims).unwrap();
    assert_eq!(strides, &[1]);
}

#[test]
fn row_major_strides_2d() {
    let dims = &[3_usize, 4];
    let strides = row_major_strides(dims).unwrap();
    assert_eq!(strides, &[4, 1]);
}

#[test]
fn row_major_strides_3d() {
    let dims = &[2_usize, 3, 4];
    let strides = row_major_strides(dims).unwrap();
    assert_eq!(strides, &[12, 4, 1]);
}

#[test]
fn row_major_strides_with_zero_dim() {
    let dims = &[2_usize, 0, 3];
    let strides = row_major_strides(dims).unwrap();
    assert_eq!(strides, &[0, 3, 1]);
}

#[test]
fn row_major_strides_dimension_too_large_for_isize() {
    if cfg!(target_pointer_width = "64") {
        let too_large = (isize::MAX as usize) + 1;
        let dims = &[2_usize, too_large, 3];
        let result = row_major_strides(dims);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("too large for stride calculation"));
    }
}

#[test]
fn row_major_strides_overflow_on_multiplication() {
    let large = (isize::MAX as usize / 2) + 1;
    let dims = &[2_usize, 2, large];
    let result = row_major_strides(dims);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("stride overflow"));
}

#[test]
fn row_major_strides_overflow_cumulative() {
    let large = (isize::MAX as usize) / 2 + 1;
    let dims = &[2_usize, 2, large];
    let result = row_major_strides(dims);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("stride overflow"));
}

#[test]
fn row_major_strides_overflow_with_multiple_dims() {
    let large = (isize::MAX as usize / 10) + 1;
    let dims = &[10_usize, 10, large];
    let result = row_major_strides(dims);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("stride overflow"));
}

#[test]
fn row_major_strides_near_max_isize() {
    let half_max = (isize::MAX as usize) / 2;
    let dims = &[2_usize, half_max];
    let result = row_major_strides(dims);
    assert!(result.is_ok());
    let strides = result.unwrap();
    assert_eq!(strides[1], 1);
    assert_eq!(strides[0], half_max as isize);
}

#[test]
fn row_major_strides_just_below_overflow() {
    let base = 1000_usize;
    let dims = &[base, base];
    let result = row_major_strides(dims);
    assert!(result.is_ok());
    let strides = result.unwrap();
    assert_eq!(strides[1], 1);
    assert_eq!(strides[0], 1000);
}

#[test]
fn row_major_strides_4d_tensor() {
    let dims = &[2_usize, 3, 4, 5];
    let strides = row_major_strides(dims).unwrap();
    assert_eq!(strides, &[60, 20, 5, 1]);
}

#[test]
fn row_major_strides_5d_tensor() {
    let dims = &[2_usize, 2, 2, 2, 2];
    let strides = row_major_strides(dims).unwrap();
    assert_eq!(strides, &[16, 8, 4, 2, 1]);
}
