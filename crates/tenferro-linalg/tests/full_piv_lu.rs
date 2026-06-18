use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_linalg::LinalgBackend;
use tenferro_tensor::{DType, DotGeneralConfig, Tensor, TensorDot, TensorStructural, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().expect("expected f64 tensor")
}

fn matmul(backend: &mut CpuBackend, lhs: &Tensor, rhs: &Tensor) -> Tensor {
    backend
        .dot_general(
            lhs,
            rhs,
            &DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
        .unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1.0e-10,
            "index {index}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn full_piv_lu_reconstructs_permuted_matrix() {
    let a = f64_tensor(vec![2, 2], vec![0.0, 2.0, 1.0, 3.0]);
    let mut backend = CpuBackend::new();

    let outputs = backend.full_piv_lu(&a).unwrap();
    let [p, l, u, q, parity]: [Tensor; 5] = outputs.try_into().unwrap();
    let pa = matmul(&mut backend, &p, &a);
    let qt = backend.transpose(&q, &[1, 0]).unwrap();
    let paqt = matmul(&mut backend, &pa, &qt);
    let lu = matmul(&mut backend, &l, &u);

    assert_eq!(p.shape(), &[2, 2]);
    assert_eq!(l.shape(), &[2, 2]);
    assert_eq!(u.shape(), &[2, 2]);
    assert_eq!(q.shape(), &[2, 2]);
    assert_eq!(parity.shape(), &[] as &[usize]);
    assert_close(f64_data(&paqt), f64_data(&lu));
}

#[test]
fn full_piv_lu_complex_parity_uses_real_counterpart_dtype() {
    let a = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(1.0, -1.0),
                Complex64::new(3.0, 0.0),
            ],
        )
        .unwrap(),
    );
    let mut backend = CpuBackend::new();

    let outputs = backend.full_piv_lu(&a).unwrap();

    assert_eq!(outputs[0].dtype(), DType::C64);
    assert_eq!(outputs[1].dtype(), DType::C64);
    assert_eq!(outputs[2].dtype(), DType::C64);
    assert_eq!(outputs[3].dtype(), DType::C64);
    assert_eq!(outputs[4].dtype(), DType::F64);
    assert_eq!(outputs[4].shape(), &[] as &[usize]);
    let parity = outputs[4].as_slice::<f64>().unwrap()[0];
    assert!(parity == 1.0 || parity == -1.0);
}

#[test]
fn full_piv_lu_uses_column_pivot_when_max_pivot_is_off_column() {
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 100.0, 3.0]);
    let mut backend = CpuBackend::new();

    let outputs = backend.full_piv_lu(&a).unwrap();
    let [_p, _l, u, q, _parity]: [Tensor; 5] = outputs.try_into().unwrap();

    assert_close(f64_data(&q), &[0.0, 1.0, 1.0, 0.0]);
    assert_close(&f64_data(&u)[..1], &[100.0]);
}

#[cfg(feature = "cpu-blas")]
#[test]
fn full_piv_lu_blas_rejects_singular_matrix() {
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 2.0, 4.0]);
    let mut backend = CpuBackend::with_kind(tenferro_cpu::CpuBackendKind::Blas).unwrap();

    let err = backend.full_piv_lu(&a).unwrap_err();

    assert!(matches!(
        err,
        tenferro_tensor::Error::BackendFailure {
            op: "full_piv_lu",
            ref message,
        } if message.contains("singular")
    ));
}

#[test]
fn full_piv_lu_solve_returns_expected_solution() {
    let a = f64_tensor(vec![2, 2], vec![0.0, 2.0, 1.0, 3.0]);
    let b = f64_tensor(vec![2, 1], vec![-1.0, 5.0]);
    let mut backend = CpuBackend::new();

    let x = backend.full_piv_lu_solve(&a, &b, false).unwrap();

    assert_eq!(x.shape(), &[2, 1]);
    assert_close(f64_data(&x), &[4.0, -1.0]);
}

#[test]
fn full_piv_lu_solve_accepts_vector_rhs() {
    let a = f64_tensor(vec![2, 2], vec![0.0, 2.0, 1.0, 3.0]);
    let b = f64_tensor(vec![2], vec![-1.0, 5.0]);
    let mut backend = CpuBackend::new();

    let x = backend.full_piv_lu_solve(&a, &b, false).unwrap();

    assert_eq!(x.shape(), &[2]);
    assert_close(f64_data(&x), &[4.0, -1.0]);
}
