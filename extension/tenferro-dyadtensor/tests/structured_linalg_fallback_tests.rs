use tenferro_dyadtensor::{
    set_default_runtime, DynAdTensor, Error, RuntimeContext, StructuredTensor,
};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

fn vector(values: &[f64]) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn structured_diag_qr_primal_rejects_non_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let x = DynAdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap(),
    );

    let err = match x.qr() {
        Ok(_) => panic!("structured qr should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedStructuredLinalg { op } if op == "qr"));
}

#[test]
fn structured_diag_inv_primal_rejects_non_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let x = DynAdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[2.0, 4.0]), 2).unwrap(),
    );

    let err = match x.inv() {
        Ok(_) => panic!("structured inv should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedStructuredLinalg { op } if op == "inv"));
}

#[test]
fn structured_diag_solve_primal_rejects_non_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let a = DynAdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[2.0, 4.0]), 2).unwrap(),
    );
    let b = DynAdTensor::new_primal(vector(&[6.0, 8.0]));

    let err = match a.solve(&b) {
        Ok(_) => panic!("structured solve should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedStructuredLinalg { op } if op == "solve"));
}

#[test]
fn structured_diag_qr_reverse_rejects_non_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let structured_x = DynAdTensor::new_reverse_leaf(
        StructuredTensor::from_diagonal_vector(vector(&[2.0, 3.0]), 2).unwrap(),
    )
    .unwrap();

    let err = match structured_x.qr() {
        Ok(_) => panic!("structured qr reverse should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedStructuredLinalg { op } if op == "qr"));
}

#[test]
fn structured_diag_inv_reverse_rejects_non_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let structured_x = DynAdTensor::new_reverse_leaf(
        StructuredTensor::from_diagonal_vector(vector(&[2.0, 4.0]), 2).unwrap(),
    )
    .unwrap();

    let err = match structured_x.inv() {
        Ok(_) => panic!("structured inv reverse should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedStructuredLinalg { op } if op == "inv"));
}

#[test]
fn structured_diag_solve_reverse_rejects_non_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let structured_a = DynAdTensor::new_reverse_leaf(
        StructuredTensor::from_diagonal_vector(vector(&[2.0, 4.0]), 2).unwrap(),
    )
    .unwrap();
    let structured_b = structured_a
        .new_reverse_sibling(vector(&[6.0, 8.0]))
        .unwrap();

    let err = match structured_a.solve(&structured_b) {
        Ok(_) => panic!("structured solve reverse should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedStructuredLinalg { op } if op == "solve"));
}
