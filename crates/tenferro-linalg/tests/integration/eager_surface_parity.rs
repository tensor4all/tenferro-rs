#![cfg(feature = "autodiff")]

use num_complex::Complex64;
use tenferro_ad::{AdContext, EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;
use tenferro_linalg::EagerTensorLinalgExt;

fn eager(data: Vec<f64>, shape: Vec<usize>) -> EagerTensor {
    EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(shape, data).unwrap(),
        EagerRuntime::with_cpu_backend(CpuBackend::new()),
    )
    .unwrap()
}

fn f64_values(tensor: &EagerTensor) -> Vec<f64> {
    tensor
        .materialized()
        .unwrap()
        .as_slice::<f64>()
        .unwrap()
        .to_vec()
}

#[test]
fn eager_composites_match_diagonal_matrix_values() {
    let a = eager(vec![2.0, 0.0, 0.0, 4.0], vec![2, 2]);

    let (sign, logabsdet) = a.slogdet().unwrap();
    assert_eq!(f64_values(&sign), vec![1.0]);
    assert!((f64_values(&logabsdet)[0] - 8.0_f64.ln()).abs() < 1.0e-12);
    assert!((f64_values(&a.det().unwrap())[0] - 8.0).abs() < 1.0e-12);
    assert_eq!(f64_values(&a.inv().unwrap()), vec![0.5, 0.0, 0.0, 0.25]);
    assert_eq!(f64_values(&a.eigvalsh().unwrap()), vec![2.0, 4.0]);

    let mut eigvals = a
        .eigvals()
        .unwrap()
        .materialized()
        .unwrap()
        .as_slice::<Complex64>()
        .unwrap()
        .to_vec();
    eigvals.sort_by(|lhs, rhs| lhs.re.total_cmp(&rhs.re));
    assert_eq!(
        eigvals,
        vec![Complex64::new(2.0, 0.0), Complex64::new(4.0, 0.0)]
    );

    assert_eq!(f64_values(&a.pinv().unwrap()), vec![0.5, 0.0, 0.0, 0.25]);
    assert_eq!(
        f64_values(&a.pinv_with_rtol(1.0e-12).unwrap()),
        vec![0.5, 0.0, 0.0, 0.25]
    );
    assert!((f64_values(&a.norm(None, None, false).unwrap())[0] - 20.0_f64.sqrt()).abs() < 1.0e-12);
}

#[test]
fn eager_vector_norm_and_keepdim_follow_traced_contract() {
    let x = eager(vec![3.0, 4.0], vec![2]);

    let norm = x.norm(Some(2.0), Some(&[0]), true).unwrap();

    assert_eq!(norm.shape(), &[1]);
    assert!((f64_values(&norm)[0] - 5.0).abs() < 1.0e-12);
}

#[test]
fn eager_norm_supports_zero_and_matrix_induced_orders() {
    let vector = eager(vec![1.0, 0.0, 2.0, -3.0], vec![4]);
    let matrix = eager(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]);

    let cases = [
        (vector.norm(Some(0.0), Some(&[0]), false).unwrap(), 3.0),
        (matrix.norm(Some(1.0), Some(&[0, 1]), false).unwrap(), 6.0),
        (matrix.norm(Some(-1.0), Some(&[0, 1]), false).unwrap(), 4.0),
        (
            matrix
                .norm(Some(f64::INFINITY), Some(&[0, 1]), false)
                .unwrap(),
            7.0,
        ),
        (
            matrix
                .norm(Some(f64::NEG_INFINITY), Some(&[0, 1]), false)
                .unwrap(),
            3.0,
        ),
    ];

    for (actual, expected) in cases {
        assert!((f64_values(&actual)[0] - expected).abs() < 1.0e-12);
    }
}

#[test]
fn eager_composite_records_existing_primitives_for_backward() {
    let ad = AdContext::builder()
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().unwrap())
        .unwrap()
        .build()
        .unwrap();
    let runtime = EagerRuntime::with_cpu_backend_and_ad_context(CpuBackend::new(), &ad);
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap(),
        runtime,
    )
    .unwrap();

    a.det().unwrap().backward().unwrap();

    let grad = a.grad().unwrap().unwrap();
    let actual = grad.as_slice::<f64>().unwrap();
    for (&actual, expected) in actual.iter().zip([4.0, 0.0, 0.0, 2.0]) {
        assert!((actual - expected).abs() < 1.0e-12);
    }
}

#[test]
fn eager_norm_covers_remaining_orders_permutations_and_errors() {
    let matrix = eager(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]);
    let tensor = eager(vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0], vec![2, 2, 2]);

    for order in [Some(2.0), Some(-2.0), Some(0.0), Some(3.0)] {
        matrix.norm(order, Some(&[0, 1]), false).unwrap();
    }
    tensor.norm(None, Some(&[2, 0]), true).unwrap();
    tensor.norm(Some(f64::INFINITY), None, false).unwrap();
    tensor.norm(Some(f64::NEG_INFINITY), None, false).unwrap();
    tensor.norm(Some(0.0), None, false).unwrap();
    tensor.norm(Some(3.0), None, false).unwrap();

    assert!(tensor.norm(None, Some(&[0, 0]), false).is_err());
    assert!(tensor.norm(None, Some(&[3]), false).is_err());
    assert!(tensor.norm(Some(f64::NAN), None, false).is_err());
}
