use num_complex::Complex64;
use tenferro::{Error, Tensor};
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

use crate::support::primal_rank0_f64;

fn vector(values: &[f64]) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn matrix2(values: &[f64; 4]) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(values, &[2, 2], MemoryOrder::ColumnMajor).unwrap()
}

fn c64_vector(values: &[Complex64]) -> DenseTensor<Complex64> {
    DenseTensor::<Complex64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn diag(values: &[f64]) -> Tensor {
    Tensor::diag(&Tensor::from_tensor(vector(values))).unwrap()
}

fn structured(values: &[f64; 4], axis_classes: &[usize]) -> Tensor {
    Tensor::with_axis_classes(Tensor::from_tensor(matrix2(values)), axis_classes).unwrap()
}

#[test]
fn axpby_rejects_diag_and_dense_vector_layout_mismatch() {
    let diag = diag(&[1.0, 2.0]);
    let dense_vec = Tensor::from_tensor(vector(&[3.0, 4.0]));

    let err = match diag.axpby(
        &primal_rank0_f64(1.0_f64),
        &dense_vec,
        &primal_rank0_f64(1.0_f64),
    ) {
        Ok(_) => panic!("axpby should reject incompatible structured layouts"),
        Err(err) => err,
    };

    assert!(matches!(err, Error::InvalidAdTensor { .. }));
}

#[test]
fn compose_complex_rejects_diag_and_dense_vector_layout_mismatch() {
    let re = diag(&[1.0, 2.0]);
    let im = Tensor::from_tensor(vector(&[5.0, 6.0]));

    let err = match Tensor::compose_complex(re, im) {
        Ok(_) => panic!("compose_complex should reject incompatible structured layouts"),
        Err(err) => err,
    };

    assert!(matches!(err, Error::InvalidAdTensor { .. }));
}

#[test]
fn axpby_rejects_same_dims_but_different_axis_classes() {
    let lhs = structured(&[1.0, 2.0, 3.0, 4.0], &[0, 1, 1]);
    let rhs = structured(&[5.0, 6.0, 7.0, 8.0], &[0, 0, 1]);

    let err = match lhs.axpby(&primal_rank0_f64(1.0_f64), &rhs, &primal_rank0_f64(1.0_f64)) {
        Ok(_) => panic!("axpby should reject axis_class mismatches"),
        Err(err) => err,
    };

    assert!(matches!(err, Error::InvalidAdTensor { .. }));
}

#[test]
fn axpby_accepts_matching_structured_layouts() {
    let lhs = diag(&[1.0, 2.0]);
    let rhs = diag(&[3.0, 4.0]);

    let out = lhs
        .axpby(
            &primal_rank0_f64(2.0_f64),
            &rhs,
            &primal_rank0_f64(-1.0_f64),
        )
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
    let re = diag(&[1.0, 2.0]);
    let im = diag(&[3.0, 4.0]);

    let out = Tensor::compose_complex(re, im).unwrap();

    assert!(out.is_diag());
    let values = out
        .as_c64()
        .unwrap()
        .primal()
        .buffer()
        .as_slice()
        .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"));
    assert_eq!(
        values,
        c64_vector(&[Complex64::new(1.0, 3.0), Complex64::new(2.0, 4.0)])
            .buffer()
            .as_slice()
            .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"))
    );
}
