//! The CPU and runtime boundaries an externally defined scalar reaches.
//!
//! Each of these operations has an implementation for the scalars tenferro declares and
//! none for a caller-owned one, so every one of them must refuse the request with a typed
//! error instead of guessing a representation.

use tenferro_cpu::CpuBackend;
use tenferro_df64_proof::Df64;
use tenferro_runtime::ad_support::ones_tensor;
use tenferro_tensor::backend::TensorReduction;
use tenferro_tensor::BackendSessionHost;
use tenferro_tensor::{DType, Tensor, TensorRead};
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

fn external(values: Vec<Df64>, shape: Vec<usize>) -> Tensor {
    Tensor::external(ErasedHostTensor::new(
        HostTensor::from_vec_col_major(shape, values).expect("shape matches data"),
    ))
}

fn external_dtype() -> DType {
    DType::External(std::any::TypeId::of::<Df64>())
}

#[test]
fn the_runtime_builds_no_core_identity_tensor_for_a_caller_owned_scalar() {
    let error = ones_tensor(external_dtype(), vec![2])
        .expect_err("a caller-owned scalar has no core runtime tensor");
    assert!(
        error.to_string().contains("externally defined"),
        "unexpected error: {error}"
    );
}

#[test]
fn a_reduction_refuses_a_caller_owned_payload() {
    let mut backend = CpuBackend::new();
    let values = external(vec![Df64::from_f64(1.0), Df64::from_f64(2.0)], vec![2]);

    // A reduction over no axes is the identity for every scalar, so it returns the
    // caller's value unchanged rather than rejecting it.
    let identity = backend
        .with_backend_session(|__s| __s.reduce_sum_read(TensorRead::from_tensor(&values), &[]))
        .expect("sum over no axes is the identity");
    assert_eq!(identity.dtype(), external_dtype());

    // A reduction that actually computes something has no CPU implementation for a
    // caller-owned payload and says so.
    let error = backend
        .with_backend_session(|__s| __s.reduce_sum_read(TensorRead::from_tensor(&values), &[0]))
        .expect_err("no CPU reduction exists for a caller-owned payload");
    assert!(
        error.to_string().contains("external") || error.to_string().contains("unsupported"),
        "unexpected error: {error}"
    );
    assert!(backend
        .with_backend_session(|__s| __s.reduce_prod_read(TensorRead::from_tensor(&values), &[0]))
        .is_err());
    assert!(backend
        .with_backend_session(|__s| __s.reduce_max_read(TensorRead::from_tensor(&values), &[0]))
        .is_err());
    assert!(backend
        .with_backend_session(|__s| __s.reduce_min_read(TensorRead::from_tensor(&values), &[0]))
        .is_err());

    // The same refusals are reached through the borrowed-read entry points, which is
    // where a session hands a value to a kernel.
    assert!(backend
        .reduce_sum_read(TensorRead::from_tensor(&values), &[0])
        .is_err());
    assert!(backend
        .reduce_sum_squares_read(TensorRead::from_tensor(&values), &[0])
        .is_err());
    assert!(backend
        .reduce_prod_read(TensorRead::from_tensor(&values), &[0])
        .is_err());
    assert!(backend
        .reduce_max_read(TensorRead::from_tensor(&values), &[0])
        .is_err());
    assert!(backend
        .reduce_min_read(TensorRead::from_tensor(&values), &[0])
        .is_err());
}

#[test]
fn an_elementwise_operation_refuses_a_caller_owned_payload() {
    let mut backend = CpuBackend::new();
    let values = external(vec![Df64::from_f64(1.0)], vec![1]);
    assert!(backend
        .with_backend_session(|__s| __s.neg_read(TensorRead::from_tensor(&values)))
        .is_err());
}

#[test]
fn every_preset_real_scalar_reduces_to_the_expected_value() {
    // The reductions have a typed implementation per preset scalar. Each case asserts the
    // arithmetic rather than only that the call returns, so this covers the per-dtype arms
    // without being a call for coverage's sake.
    let mut backend = CpuBackend::new();

    macro_rules! check {
        ($ty:ty, $values:expr, $sum:expr, $prod:expr, $max:expr, $min:expr) => {{
            let tensor = Tensor::from_vec_col_major(vec![2], $values).expect("shape matches data");
            assert_eq!(
                backend
                    .with_backend_session(
                        |__s| __s.reduce_sum_read(TensorRead::from_tensor(&tensor), &[0])
                    )
                    .expect("sum")
                    .as_slice::<$ty>()
                    .expect("slice"),
                &[$sum]
            );
            assert_eq!(
                backend
                    .with_backend_session(
                        |__s| __s.reduce_prod_read(TensorRead::from_tensor(&tensor), &[0])
                    )
                    .expect("product")
                    .as_slice::<$ty>()
                    .expect("slice"),
                &[$prod]
            );
            assert_eq!(
                backend
                    .with_backend_session(
                        |__s| __s.reduce_max_read(TensorRead::from_tensor(&tensor), &[0])
                    )
                    .expect("maximum")
                    .as_slice::<$ty>()
                    .expect("slice"),
                &[$max]
            );
            assert_eq!(
                backend
                    .with_backend_session(
                        |__s| __s.reduce_min_read(TensorRead::from_tensor(&tensor), &[0])
                    )
                    .expect("minimum")
                    .as_slice::<$ty>()
                    .expect("slice"),
                &[$min]
            );
        }};
    }

    check!(f32, vec![2.0_f32, 3.0], 5.0, 6.0, 3.0, 2.0);
    check!(f64, vec![2.0_f64, 3.0], 5.0, 6.0, 3.0, 2.0);
    check!(i32, vec![2_i32, 3], 5, 6, 3, 2);
    check!(i64, vec![2_i64, 3], 5, 6, 3, 2);

    // The sum of squares is defined for the real floating-point scalars only, so an
    // integer input is refused rather than widened silently.
    let integers = Tensor::from_vec_col_major(vec![2], vec![2_i32, 3]).expect("shape");
    assert!(backend
        .reduce_sum_squares_read(TensorRead::from_tensor(&integers), &[0])
        .is_err());
    let floats = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).expect("shape");
    assert_eq!(
        backend
            .reduce_sum_squares_read(TensorRead::from_tensor(&floats), &[0])
            .expect("squares")
            .as_slice::<f64>()
            .expect("slice"),
        &[13.0]
    );

    // The boolean scalar has no ordered reduction.
    let boolean = Tensor::from_vec_col_major(vec![2], vec![true, false]).expect("shape");
    assert!(backend
        .with_backend_session(|__s| __s.reduce_max_read(TensorRead::from_tensor(&boolean), &[0]))
        .is_err());
    assert!(backend
        .with_backend_session(|__s| __s.reduce_min_read(TensorRead::from_tensor(&boolean), &[0]))
        .is_err());
}
