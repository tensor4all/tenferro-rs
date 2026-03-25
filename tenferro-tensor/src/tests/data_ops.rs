use crate::{MemoryOrder, Tensor};
use tenferro_device::LogicalMemorySpace;

// ---------------------------------------------------------------------------
// get
// ---------------------------------------------------------------------------

#[test]
fn get_returns_correct_element_col_major() {
    // Column-major [2,2]: data [1,2,3,4] → col0=[1,2], col1=[3,4]
    let t = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    assert_eq!(t.get(&[0, 0]), Some(&1.0));
    assert_eq!(t.get(&[1, 0]), Some(&2.0));
    assert_eq!(t.get(&[0, 1]), Some(&3.0));
    assert_eq!(t.get(&[1, 1]), Some(&4.0));
}

#[test]
fn get_returns_correct_element_row_major() {
    // Row-major [2,2]: data [1,2,3,4] → row0=[1,2], row1=[3,4]
    let t = Tensor::<f64>::from_row_major_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert_eq!(t.get(&[0, 0]), Some(&1.0));
    assert_eq!(t.get(&[0, 1]), Some(&2.0));
    assert_eq!(t.get(&[1, 0]), Some(&3.0));
    assert_eq!(t.get(&[1, 1]), Some(&4.0));
}

#[test]
fn get_returns_none_for_out_of_bounds() {
    let t = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    assert_eq!(t.get(&[3]), None);
    assert_eq!(t.get(&[100]), None);
}

#[test]
fn get_returns_none_for_wrong_rank() {
    let t = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    assert_eq!(t.get(&[0]), None); // too few indices
    assert_eq!(t.get(&[0, 0, 0]), None); // too many indices
}

#[test]
fn get_works_after_permute() {
    // Col-major [2,3] then permute → [3,2]
    let t = Tensor::<f64>::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let tp = t.permute(&[1, 0]).unwrap();
    // Original col-major: col0=[1,2], col1=[3,4], col2=[5,6]
    // After permute, t[i,j] becomes tp[j,i]
    assert_eq!(tp.get(&[0, 0]), Some(&1.0));
    assert_eq!(tp.get(&[0, 1]), Some(&2.0));
    assert_eq!(tp.get(&[1, 0]), Some(&3.0));
    assert_eq!(tp.get(&[2, 1]), Some(&6.0));
}

#[test]
fn get_scalar_tensor() {
    let t = Tensor::<f64>::from_slice(&[42.0], &[], MemoryOrder::ColumnMajor).unwrap();
    assert_eq!(t.get(&[]), Some(&42.0));
}

// ---------------------------------------------------------------------------
// to_vec
// ---------------------------------------------------------------------------

#[test]
fn to_vec_col_major_identity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::<f64>::from_slice(&data, &[2, 3], MemoryOrder::ColumnMajor).unwrap();
    assert_eq!(t.to_vec(), data);
}

#[test]
fn to_vec_row_major_reorders() {
    let t = Tensor::<f64>::from_row_major_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    // Row-major input: [[1,2,3],[4,5,6]]
    // Column-major output: col0=[1,4], col1=[2,5], col2=[3,6]
    assert_eq!(t.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn to_vec_after_permute() {
    let t = Tensor::<f64>::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let tp = t.permute(&[1, 0]).unwrap();
    // tp is [3,2], col-major output: col0=[1,3,5], col1=[2,4,6]
    assert_eq!(tp.to_vec(), vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
}

#[test]
fn to_vec_empty_tensor() {
    let t = Tensor::<f64>::zeros(
        &[0, 3],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    assert!(t.to_vec().is_empty());
}

// ---------------------------------------------------------------------------
// from_row_major_slice
// ---------------------------------------------------------------------------

#[test]
fn from_row_major_slice_matches_from_slice_row_major() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Tensor::<f64>::from_row_major_slice(&data, &[2, 3]).unwrap();
    let b = Tensor::<f64>::from_slice(&data, &[2, 3], MemoryOrder::RowMajor).unwrap();
    assert_eq!(a.to_vec(), b.to_vec());
}

#[test]
fn from_row_major_slice_rejects_length_mismatch() {
    let result = Tensor::<f64>::from_row_major_slice(&[1.0, 2.0, 3.0], &[2, 2]);
    assert!(result.is_err());
}
