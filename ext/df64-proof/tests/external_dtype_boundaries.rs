//! The CPU and runtime boundaries an externally defined scalar reaches.
//!
//! Each of these operations has an implementation for the scalars tenferro declares and
//! none for a caller-owned one, so every one of them must refuse the request with a typed
//! error instead of guessing a representation.

use tenferro_cpu::CpuBackend;
use tenferro_df64_proof::Df64;
use tenferro_runtime::ad_support::ones_tensor;
use tenferro_tensor::backend::{TensorElementwise, TensorReduction};
use tenferro_tensor::{DType, Tensor};
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
        .reduce_sum(&values, &[])
        .expect("sum over no axes is the identity");
    assert_eq!(identity.dtype(), external_dtype());

    // A reduction that actually computes something has no CPU implementation for a
    // caller-owned payload and says so.
    let error = backend
        .reduce_sum(&values, &[0])
        .expect_err("no CPU reduction exists for a caller-owned payload");
    assert!(
        error.to_string().contains("external") || error.to_string().contains("unsupported"),
        "unexpected error: {error}"
    );
    assert!(backend.reduce_prod(&values, &[0]).is_err());
}

#[test]
fn an_elementwise_operation_refuses_a_caller_owned_payload() {
    let mut backend = CpuBackend::new();
    let values = external(vec![Df64::from_f64(1.0)], vec![1]);
    assert!(backend.neg(&values).is_err());
}
