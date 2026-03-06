use tenferro_dyadtensor::{ad, set_default_runtime, AdTensor, RuntimeContext, StructuredTensor};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

fn vector(values: &[f64]) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn matrix2(values: &[f64; 4]) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[2, 2], MemoryOrder::ColumnMajor).unwrap()
}

fn as_slice(tensor: &Tensor<f64>) -> &[f64] {
    tensor
        .buffer()
        .as_slice()
        .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"))
}

#[test]
fn structured_diag_input_can_flow_through_qr_via_internal_dense_fallback() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let x = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap(),
    );

    let out = ad::qr(&x).unwrap();

    assert!(out.q.is_dense());
    assert!(out.r.is_dense());
    assert_eq!(out.q.dims(), &[2, 2]);
    assert_eq!(out.r.dims(), &[2, 2]);
}

#[test]
fn structured_diag_input_can_flow_through_inv_via_internal_dense_fallback() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let x = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[2.0, 4.0]), 2).unwrap(),
    );

    let out = ad::inv(&x).unwrap();

    assert!(out.is_dense());
    assert_eq!(out.dims(), &[2, 2]);
}

#[test]
fn structured_diag_input_can_flow_through_solve_via_internal_dense_fallback() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let a = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[2.0, 4.0]), 2).unwrap(),
    );
    let b = AdTensor::new_primal(vector(&[6.0, 8.0]));

    let out = ad::solve(&a, &b).unwrap();

    assert!(out.is_dense());
    assert_eq!(out.dims(), &[2]);
}

#[test]
fn structured_diag_qr_pullback_matches_dense_projection() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let structured_x = AdTensor::new_reverse(
        StructuredTensor::from_diagonal_vector(vector(&[2.0, 3.0]), 2).unwrap(),
        tenferro_dyadtensor::NodeId(11),
        tenferro_dyadtensor::TapeId(91),
        None,
    );
    let dense_x = AdTensor::new_reverse(
        matrix2(&[2.0, 0.0, 0.0, 3.0]),
        tenferro_dyadtensor::NodeId(12),
        tenferro_dyadtensor::TapeId(92),
        None,
    );

    let structured_out = ad::qr(&structured_x).unwrap();
    let dense_out = ad::qr(&dense_x).unwrap();
    let cotangent = AdTensor::new_primal(matrix2(&[1.0, -0.5, 0.25, 2.0]));

    let structured_grad = ad::pullback_wrt(&structured_out.r, &cotangent, &[&structured_x])
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
        .unwrap();
    let dense_grad = ad::pullback_wrt(&dense_out.r, &cotangent, &[&dense_x])
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
        .unwrap();

    assert!(structured_grad.is_diag());
    let structured_dense = structured_grad.to_dense().unwrap();
    let dense_values = as_slice(dense_grad.payload());
    let structured_values = as_slice(&structured_dense);
    assert!((structured_values[0] - dense_values[0]).abs() < 1e-12);
    assert!((structured_values[3] - dense_values[3]).abs() < 1e-12);
    assert!(structured_values[1].abs() < 1e-12);
    assert!(structured_values[2].abs() < 1e-12);
}

#[test]
fn structured_diag_inv_pullback_matches_dense_projection() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let structured_x = AdTensor::new_reverse(
        StructuredTensor::from_diagonal_vector(vector(&[2.0, 4.0]), 2).unwrap(),
        tenferro_dyadtensor::NodeId(21),
        tenferro_dyadtensor::TapeId(93),
        None,
    );
    let dense_x = AdTensor::new_reverse(
        matrix2(&[2.0, 0.0, 0.0, 4.0]),
        tenferro_dyadtensor::NodeId(22),
        tenferro_dyadtensor::TapeId(94),
        None,
    );

    let structured_out = ad::inv(&structured_x).unwrap();
    let dense_out = ad::inv(&dense_x).unwrap();
    let cotangent = AdTensor::new_primal(matrix2(&[1.5, -0.25, 0.5, 2.0]));

    let structured_grad = ad::pullback_wrt(&structured_out, &cotangent, &[&structured_x])
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
        .unwrap();
    let dense_grad = ad::pullback_wrt(&dense_out, &cotangent, &[&dense_x])
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
        .unwrap();

    assert!(structured_grad.is_diag());
    let structured_dense = structured_grad.to_dense().unwrap();
    let dense_values = as_slice(dense_grad.payload());
    let structured_values = as_slice(&structured_dense);
    assert!((structured_values[0] - dense_values[0]).abs() < 1e-12);
    assert!((structured_values[3] - dense_values[3]).abs() < 1e-12);
    assert!(structured_values[1].abs() < 1e-12);
    assert!(structured_values[2].abs() < 1e-12);
}

#[test]
fn structured_diag_solve_pullback_matches_dense_projection() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let structured_a = AdTensor::new_reverse(
        StructuredTensor::from_diagonal_vector(vector(&[2.0, 4.0]), 2).unwrap(),
        tenferro_dyadtensor::NodeId(31),
        tenferro_dyadtensor::TapeId(95),
        None,
    );
    let dense_a = AdTensor::new_reverse(
        matrix2(&[2.0, 0.0, 0.0, 4.0]),
        tenferro_dyadtensor::NodeId(32),
        tenferro_dyadtensor::TapeId(96),
        None,
    );
    let structured_b = AdTensor::new_reverse(
        vector(&[6.0, 8.0]),
        tenferro_dyadtensor::NodeId(33),
        tenferro_dyadtensor::TapeId(95),
        None,
    );
    let dense_b = AdTensor::new_reverse(
        vector(&[6.0, 8.0]),
        tenferro_dyadtensor::NodeId(34),
        tenferro_dyadtensor::TapeId(96),
        None,
    );

    let structured_out = ad::solve(&structured_a, &structured_b).unwrap();
    let dense_out = ad::solve(&dense_a, &dense_b).unwrap();
    let cotangent = AdTensor::new_primal(vector(&[0.5, -1.0]));

    let structured_grads =
        ad::pullback_wrt(&structured_out, &cotangent, &[&structured_a, &structured_b]).unwrap();
    let dense_grads = ad::pullback_wrt(&dense_out, &cotangent, &[&dense_a, &dense_b]).unwrap();

    let structured_grad_a = structured_grads[0].clone().unwrap();
    let dense_grad_a = dense_grads[0].clone().unwrap();
    assert!(structured_grad_a.is_diag());
    let structured_dense_a = structured_grad_a.to_dense().unwrap();
    let dense_values_a = as_slice(dense_grad_a.payload());
    let structured_values_a = as_slice(&structured_dense_a);
    assert!((structured_values_a[0] - dense_values_a[0]).abs() < 1e-12);
    assert!((structured_values_a[3] - dense_values_a[3]).abs() < 1e-12);
    assert!(structured_values_a[1].abs() < 1e-12);
    assert!(structured_values_a[2].abs() < 1e-12);

    let structured_grad_b = structured_grads[1].clone().unwrap();
    let dense_grad_b = dense_grads[1].clone().unwrap();
    assert!(structured_grad_b.is_dense());
    assert_eq!(
        as_slice(structured_grad_b.payload()),
        as_slice(dense_grad_b.payload())
    );
}
