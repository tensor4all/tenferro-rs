use crate::{MemoryOrder, StructuredTensor, Tensor};

#[test]
fn from_dense_roundtrip() {
    let t = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let s = StructuredTensor::from_dense(t.clone());
    assert!(s.is_dense());
    assert!(!s.is_diag());
    assert_eq!(s.logical_dims(), &[2, 2]);
    assert_eq!(s.axis_classes(), &[0, 1]);
    assert_eq!(s.payload().dims(), t.dims());
}

#[test]
fn from_diagonal_vector() {
    let payload =
        Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    let s = StructuredTensor::from_diagonal_vector(payload, 2).unwrap();
    assert!(s.is_diag());
    assert!(!s.is_dense());
    assert_eq!(s.logical_dims(), &[3, 3]);
    assert_eq!(s.axis_classes(), &[0, 0]);
    assert_eq!(s.payload().dims(), &[3]);
}

#[test]
fn axis_classes_canonicalized() {
    let payload = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let s = StructuredTensor::new(vec![2, 2], vec![5, 5], payload).unwrap();
    assert_eq!(s.axis_classes(), &[0, 0]);
}

#[test]
fn inconsistent_dims_rejected() {
    let payload = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    assert!(StructuredTensor::new(vec![2, 3], vec![0, 0], payload).is_err());
}

#[test]
fn class_count() {
    let payload = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let s = StructuredTensor::from_diagonal_vector(payload, 3).unwrap();
    assert_eq!(s.class_count(), 1);
}

#[test]
fn scalar_tensor_from_dense() {
    let t = Tensor::<f64>::from_slice(&[42.0], &[], MemoryOrder::ColumnMajor).unwrap();
    let s = StructuredTensor::from_dense(t);
    assert!(s.is_dense());
    assert_eq!(s.logical_dims(), &[] as &[usize]);
}

#[test]
fn empty_tensor_from_dense() {
    let t = Tensor::<f64>::from_slice(&[], &[0, 0], MemoryOrder::ColumnMajor).unwrap();
    let s = StructuredTensor::from_dense(t);
    assert!(s.is_dense());
    assert_eq!(s.logical_dims(), &[0, 0]);
}

#[test]
fn permute_logical_diagonal_is_noop() {
    let payload = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let s = StructuredTensor::from_diagonal_vector(payload, 2).unwrap();
    let p = s.permute_logical(&[1, 0]).unwrap();
    assert_eq!(p.logical_dims(), &[2, 2]);
    assert!(p.is_diag());
}

#[test]
fn permute_logical_dense() {
    let t = Tensor::<f64>::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let s = StructuredTensor::from_dense(t);
    let p = s.permute_logical(&[1, 0]).unwrap();
    assert_eq!(p.logical_dims(), &[3, 2]);
    assert_eq!(p.axis_classes(), &[0, 1]);
}

#[test]
fn permute_logical_wrong_rank_rejected() {
    let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let s = StructuredTensor::from_dense(t);
    assert!(s.permute_logical(&[0, 1]).is_err());
}

#[test]
fn permute_logical_duplicate_axis_rejected() {
    let t = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let s = StructuredTensor::from_dense(t);
    assert!(s.permute_logical(&[0, 0]).is_err());
}

#[test]
fn conj_preserves_structure() {
    use num_complex::Complex64;

    let payload = Tensor::<Complex64>::from_slice(
        &[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let s = StructuredTensor::from_diagonal_vector(payload, 2).unwrap();
    let c = s.conj();
    assert!(c.is_diag());
    assert_eq!(c.logical_dims(), s.logical_dims());
    assert_eq!(c.axis_classes(), s.axis_classes());
}
