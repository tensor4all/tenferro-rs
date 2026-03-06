use num_complex::Complex64;
use tenferro_dyadtensor::{AdTensor, DynAdScalar, DynAdTensor, Error, StructuredTensor};
use tenferro_tensor::{MemoryOrder, Tensor};

fn vector(values: &[f64]) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn matrix2(values: &[f64; 4]) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[2, 2], MemoryOrder::ColumnMajor).unwrap()
}

fn c64_vector(values: &[Complex64]) -> Tensor<Complex64> {
    Tensor::<Complex64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn axpby_rejects_diag_and_dense_vector_layout_mismatch() {
    let diag: DynAdTensor = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap(),
    )
    .into();
    let dense_vec: DynAdTensor = AdTensor::new_primal(vector(&[3.0, 4.0])).into();

    let err = match diag.axpby(
        &DynAdScalar::from(1.0_f64),
        &dense_vec,
        &DynAdScalar::from(1.0_f64),
    ) {
        Ok(_) => panic!("axpby should reject incompatible structured layouts"),
        Err(err) => err,
    };

    assert!(matches!(err, Error::InvalidAdTensor { .. }));
}

#[test]
fn compose_complex_rejects_diag_and_dense_vector_layout_mismatch() {
    let re: DynAdTensor = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap(),
    )
    .into();
    let im: DynAdTensor = AdTensor::new_primal(vector(&[5.0, 6.0])).into();

    let err = match DynAdTensor::compose_complex(re, im) {
        Ok(_) => panic!("compose_complex should reject incompatible structured layouts"),
        Err(err) => err,
    };

    assert!(matches!(err, Error::InvalidAdTensor { .. }));
}

#[test]
fn axpby_rejects_same_dims_but_different_axis_classes() {
    let lhs: DynAdTensor = AdTensor::new_primal(
        StructuredTensor::new(vec![2, 2, 2], vec![0, 1, 1], matrix2(&[1.0, 2.0, 3.0, 4.0]))
            .unwrap(),
    )
    .into();
    let rhs: DynAdTensor = AdTensor::new_primal(
        StructuredTensor::new(vec![2, 2, 2], vec![0, 0, 1], matrix2(&[5.0, 6.0, 7.0, 8.0]))
            .unwrap(),
    )
    .into();

    let err = match lhs.axpby(
        &DynAdScalar::from(1.0_f64),
        &rhs,
        &DynAdScalar::from(1.0_f64),
    ) {
        Ok(_) => panic!("axpby should reject axis_class mismatches"),
        Err(err) => err,
    };

    assert!(matches!(err, Error::InvalidAdTensor { .. }));
}

#[test]
fn axpby_accepts_matching_structured_layouts() {
    let lhs: DynAdTensor = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap(),
    )
    .into();
    let rhs: DynAdTensor = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[3.0, 4.0]), 2).unwrap(),
    )
    .into();

    let out = lhs
        .axpby(&DynAdScalar::from(2.0_f64), &rhs, &DynAdScalar::from(-1.0_f64))
        .unwrap();

    assert!(out.is_diag());
    let values = out
        .as_f64()
        .unwrap()
        .primal()
        .buffer()
        .as_slice()
        .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"));
    assert_eq!(values, &[-1.0, 0.0]);
}

#[test]
fn compose_complex_accepts_matching_structured_layouts() {
    let re: DynAdTensor = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap(),
    )
    .into();
    let im: DynAdTensor = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[3.0, 4.0]), 2).unwrap(),
    )
    .into();

    let out = DynAdTensor::compose_complex(re, im).unwrap();

    assert!(out.is_diag());
    let values = out
        .as_c64()
        .unwrap()
        .primal()
        .buffer()
        .as_slice()
        .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"));
    assert_eq!(values, c64_vector(&[Complex64::new(1.0, 3.0), Complex64::new(2.0, 4.0)])
        .buffer()
        .as_slice()
        .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor")));
}
