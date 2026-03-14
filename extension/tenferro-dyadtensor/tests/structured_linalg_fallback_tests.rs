use tenferro_dyadtensor::{
    ad, set_default_runtime, DynAdTensor, DynTape, Error, RuntimeContext, StructuredTensor,
};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

fn vector(values: &[f64]) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn structured_diag_input_can_flow_through_qr_via_internal_dense_fallback() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let x = DynAdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap(),
    );

    let out = ad::qr(x.as_f64().unwrap()).unwrap();

    assert!(out.q.is_dense());
    assert!(out.r.is_dense());
    assert_eq!(out.q.dims(), &[2, 2]);
    assert_eq!(out.r.dims(), &[2, 2]);
}

#[test]
fn structured_diag_input_can_flow_through_inv_via_internal_dense_fallback() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let x = DynAdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[2.0, 4.0]), 2).unwrap(),
    );

    let out = ad::inv(x.as_f64().unwrap()).unwrap();

    assert!(out.is_dense());
    assert_eq!(out.dims(), &[2, 2]);
}

#[test]
fn structured_diag_input_can_flow_through_solve_via_internal_dense_fallback() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let a = DynAdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[2.0, 4.0]), 2).unwrap(),
    );
    let b = DynAdTensor::new_primal(vector(&[6.0, 8.0]));

    let out = ad::solve(a.as_f64().unwrap(), b.as_f64().unwrap()).unwrap();

    assert!(out.is_dense());
    assert_eq!(out.dims(), &[2]);
}

#[test]
fn structured_diag_qr_reverse_rejects_non_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let tape = DynTape::new();
    let structured_x = DynAdTensor::new_reverse_leaf(
        StructuredTensor::from_diagonal_vector(vector(&[2.0, 3.0]), 2).unwrap(),
        &tape,
    )
    .unwrap();

    let err = match ad::qr(structured_x.as_f64().unwrap()) {
        Ok(_) => panic!("structured qr reverse should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedAdOp { op } if op == "qr_ad_structured"));
}

#[test]
fn structured_diag_inv_reverse_rejects_non_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let tape = DynTape::new();
    let structured_x = DynAdTensor::new_reverse_leaf(
        StructuredTensor::from_diagonal_vector(vector(&[2.0, 4.0]), 2).unwrap(),
        &tape,
    )
    .unwrap();

    let err = match ad::inv(structured_x.as_f64().unwrap()) {
        Ok(_) => panic!("structured inv reverse should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedAdOp { op } if op == "inv_ad_structured"));
}

#[test]
fn structured_diag_solve_reverse_rejects_non_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let tape = DynTape::new();
    let structured_a = DynAdTensor::new_reverse_leaf(
        StructuredTensor::from_diagonal_vector(vector(&[2.0, 4.0]), 2).unwrap(),
        &tape,
    )
    .unwrap();
    let structured_b = DynAdTensor::new_reverse_leaf(vector(&[6.0, 8.0]), &tape).unwrap();

    let err = match ad::solve(
        structured_a.as_f64().unwrap(),
        structured_b.as_f64().unwrap(),
    ) {
        Ok(_) => panic!("structured solve reverse should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedAdOp { op } if op == "solve_ad_structured"));
}
