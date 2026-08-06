#![cfg(feature = "autodiff")]

use num_complex::Complex64;
use tenferro_ad::{AdContext, EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;
use tenferro_linalg::{EagerTensorLinalgExt, TracedTensorLinalgExt};
use tenferro_runtime::{DType, Error, ErrorPhase, TracedTensor, TypedTensor};

fn eager(data: Vec<f64>, shape: Vec<usize>) -> EagerTensor {
    EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(shape, data).unwrap(),
        EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap(),
    )
    .unwrap()
}

fn eager_complex(data: Vec<Complex64>, shape: Vec<usize>) -> EagerTensor {
    EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(shape, data).unwrap(),
        EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap(),
    )
    .unwrap()
}

fn eager_i32(data: Vec<i32>, shape: Vec<usize>) -> EagerTensor {
    EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(shape, data).unwrap(),
        EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap(),
    )
    .unwrap()
}

fn traced_i32(data: Vec<i32>, shape: Vec<usize>) -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape(Tensor::I32(
        TypedTensor::from_vec_col_major(shape, data).unwrap(),
    ))
    .unwrap()
}

fn assert_lstsq_validation_reason(
    label: &str,
    eager_result: tenferro_ad::Result<EagerTensor>,
    traced_result: tenferro_runtime::Result<TracedTensor>,
    expected_reason: &str,
) {
    let eager_error = eager_result.expect_err(label);
    let traced_error = traced_result.expect_err(label);
    assert!(
        eager_error.to_string().contains(expected_reason),
        "{label}: eager expected {expected_reason:?}, got {eager_error}"
    );
    assert!(
        traced_error.to_string().contains(expected_reason),
        "{label}: traced expected {expected_reason:?}, got {traced_error}"
    );
}

fn assert_lstsq_wide_error_contract(
    eager_error: &Error,
    traced_error: &Error,
    expected_reason: &str,
) {
    assert!(matches!(
        eager_error,
        Error::Validation {
            phase: ErrorPhase::GraphBuild,
            ..
        }
    ));
    assert_eq!(eager_error.phase(), Some(ErrorPhase::GraphBuild));
    assert_eq!(
        eager_error.to_string(),
        Error::invalid_argument("lstsq", ErrorPhase::GraphBuild, "shape", expected_reason)
            .to_string()
    );

    assert!(matches!(
        traced_error,
        Error::TensorRuntime(tenferro_tensor::Error::Validation { .. })
    ));
    assert_eq!(traced_error.phase(), Some(ErrorPhase::Execution));
    assert_eq!(
        traced_error.to_string(),
        tenferro_tensor::Error::invalid_argument("lstsq", "shape", expected_reason).to_string()
    );
}

fn f64_values(tensor: &EagerTensor) -> Vec<f64> {
    tensor
        .to_tensor()
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
        .to_tensor()
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
    let no_op = x.norm(None, Some(&[]), false).unwrap();

    assert_eq!(norm.shape(), &[1]);
    assert!((f64_values(&norm)[0] - 5.0).abs() < 1.0e-12);
    assert_eq!(no_op.shape(), x.shape());
    assert_eq!(f64_values(&no_op), f64_values(&x));
}

#[test]
fn eager_norm_supports_zero_and_matrix_induced_orders() {
    let vector = eager(vec![1.0, 0.0, 2.0, -3.0], vec![4]);
    let complex_vector = eager_complex(
        vec![
            Complex64::new(3.0, 4.0),
            Complex64::new(0.0, -12.0),
            Complex64::new(5.0, 0.0),
        ],
        vec![3],
    );
    let matrix = eager(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]);

    let cases = [
        (vector.norm(Some(0.0), Some(&[0]), false).unwrap(), 3.0),
        (
            vector.norm(Some(f64::INFINITY), Some(&[0]), false).unwrap(),
            3.0,
        ),
        (
            vector
                .norm(Some(f64::NEG_INFINITY), Some(&[0]), false)
                .unwrap(),
            0.0,
        ),
        (
            vector.norm(Some(3.0), Some(&[0]), false).unwrap(),
            36.0_f64.cbrt(),
        ),
        (
            complex_vector.norm(Some(2.0), Some(&[0]), false).unwrap(),
            194.0_f64.sqrt(),
        ),
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
    let runtime = EagerRuntime::with_cpu_backend_and_ad_context(CpuBackend::new(), &ad).unwrap();
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
fn lstsq_validation_is_paired_across_eager_and_traced_surfaces() {
    let eager_a = eager_i32(vec![1, 0, 0, 1], vec![2, 2]);
    let eager_b = eager_i32(vec![1, 2], vec![2, 1]);
    let traced_a = traced_i32(vec![1, 0, 0, 1], vec![2, 2]);
    let traced_b = traced_i32(vec![1, 2], vec![2, 1]);
    assert_lstsq_validation_reason(
        "integer dtype",
        eager_a.lstsq(&eager_b),
        traced_a.lstsq(&traced_b),
        "does not support dtype I32",
    );

    let eager_a = eager(vec![1.0, 2.0], vec![2]);
    let eager_b = eager(vec![1.0, 2.0], vec![2, 1]);
    let traced_a = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let traced_b = TracedTensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0]).unwrap();
    assert_lstsq_validation_reason(
        "rank-one A",
        eager_a.lstsq(&eager_b),
        traced_a.lstsq(&traced_b),
        "rank mismatch",
    );

    let eager_a = eager(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
    let eager_b = eager(vec![1.0, 2.0], vec![2]);
    let traced_a =
        TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap();
    let traced_b = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    assert_lstsq_validation_reason(
        "rank-one B",
        eager_a.lstsq(&eager_b),
        traced_a.lstsq(&traced_b),
        "rank mismatch",
    );

    let eager_a = eager(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![2, 3]);
    let eager_b = eager(vec![1.0, 2.0], vec![2, 1]);
    let traced_a =
        TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0])
            .unwrap();
    let traced_b = TracedTensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0]).unwrap();
    let eager_error = eager_a.lstsq(&eager_b).unwrap_err();
    let traced_error = traced_a.lstsq(&traced_b).unwrap_err();
    const WIDE_REASON: &str = "lstsq requires a tall or square matrix (rows 2 >= cols 3); \
        underdetermined (wide) systems are not supported";
    assert!(eager_error.to_string().contains("tall or square"));
    assert!(traced_error.to_string().contains("tall or square"));
    assert_lstsq_wide_error_contract(&eager_error, &traced_error, WIDE_REASON);
}

#[test]
fn traced_lstsq_symbolic_shape_keeps_dtype_and_rank_precedence() {
    let b = TracedTensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0]).unwrap();

    let symbolic_i32 = TracedTensor::input_symbolic_shape(DType::I32, 2).unwrap();
    let error = symbolic_i32.lstsq(&b).unwrap_err();
    assert!(error.to_string().contains("does not support dtype I32"));
    assert!(!error.to_string().contains("symbolic shape"));

    let symbolic_rank_one = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let error = symbolic_rank_one.lstsq(&b).unwrap_err();
    assert!(error.to_string().contains("rank mismatch"));
    assert!(!error.to_string().contains("symbolic shape"));

    let a = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let rank_one_b = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let error = a.lstsq(&rank_one_b).unwrap_err();
    assert!(error.to_string().contains("rank mismatch"));
    assert!(!error.to_string().contains("symbolic shape"));

    let error = a.lstsq(&b).unwrap_err();
    let expected = Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
        "lstsq",
        "shape",
        "symbolic shape is not supported by this traced linalg helper",
    ));
    assert!(matches!(error, Error::TensorRuntime(_)));
    assert_eq!(error.phase(), Some(ErrorPhase::Execution));
    assert_eq!(error.to_string(), expected.to_string());
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
