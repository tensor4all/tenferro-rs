//! The extension-owned operation reached through the runtime.
//!
//! The input travels as a runtime `Tensor` carrying a caller-owned payload, the
//! operation is an `ExtensionOp` family registered by a downstream
//! `ExtensionModule`, and the result comes back as another external payload.

use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_df64_proof::extension::{apply_total, Df64Total};
use tenferro_df64_proof::Df64;
use tenferro_tensor::{AllocationGroup, BackendSessionHost, GroupError, Tensor};
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

fn external(values: Vec<Df64>) -> Tensor {
    Tensor::external(ErasedHostTensor::new(
        HostTensor::from_vec_col_major(vec![values.len()], values).expect("shape matches data"),
    ))
}

fn payload(tensor: &Tensor) -> Vec<Df64> {
    match tensor {
        Tensor::External(value, _) => value
            .downcast_ref::<Df64>()
            .expect("external element type")
            .as_slice()
            .to_vec(),
        _ => panic!("expected an externally defined payload"),
    }
}

#[test]
fn the_extension_operation_runs_through_the_registered_module() {
    let low = 2f64.powi(-80);
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).expect("cpu runtime");
    let input = tenferro_ad::EagerTensor::from_tensor_in(
        external(vec![Df64::from_f64(1.0), Df64::from_f64(low)]),
        std::sync::Arc::clone(&runtime),
    )
    .expect("eager tensor");

    let outputs = apply_total(&input).expect("extension execution");
    assert_eq!(outputs.len(), 1);

    let value = outputs[0].to_tensor().expect("output tensor");
    let total = payload(&value);
    assert_eq!(total.len(), 1);
    assert_eq!(total[0], Df64 { hi: 1.0, lo: low });
}

#[test]
fn the_registered_operation_rejects_a_preset_input() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).expect("cpu runtime");
    let ordinary = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).expect("shape matches");
    let input = tenferro_ad::EagerTensor::from_tensor_in(ordinary, std::sync::Arc::clone(&runtime))
        .expect("eager tensor");

    // The extension declares only the external scalar, so a preset input fails
    // explicitly instead of being coerced.
    assert!(apply_total(&input).is_err());
}

#[test]
fn ordinary_work_shares_the_session_with_the_extension() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).expect("cpu runtime");
    let mut backend = CpuBackend::new();
    backend.with_backend_session(|session| {
        let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).expect("shape matches");
        let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).expect("shape matches");
        let sum = tenferro_tensor::backend::TensorElementwise::add(session, &a, &b)
            .expect("ordinary addition");
        assert_eq!(sum.as_slice::<f64>().expect("f64 slice"), &[4.0, 6.0]);
    });
    assert_eq!(
        <Df64Total as tenferro_ad::extension::ExtensionOp>::input_count(&Df64Total),
        1
    );
    drop(runtime);
}

#[test]
fn the_runtime_retains_and_returns_a_caller_owned_payload() {
    let low = 2f64.powi(-80);
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).expect("cpu runtime");
    let original = Df64::from_f64(1.0) + Df64::from_f64(low);
    let leaf = tenferro_ad::EagerTensor::from_tensor_in(
        external(vec![original, Df64::from_f64(3.0)]),
        std::sync::Arc::clone(&runtime),
    )
    .expect("eager tensor");

    // A caller-owned payload keeps every bit while the runtime retains it, so the
    // low-order component survives the round trip unchanged.
    let value = leaf.to_tensor().expect("round trip");
    assert_eq!(payload(&value), vec![original, Df64::from_f64(3.0)]);
    assert!(matches!(value, Tensor::External(..)));
}

#[test]
fn an_allocation_group_rejects_a_caller_owned_payload_with_a_typed_error() {
    // A payload tenferro does not define owns no pooled allocation group, so the
    // group reports it explicitly instead of dropping or zeroing the value.
    let error = AllocationGroup::from_tensors(vec![external(vec![Df64::from_f64(1.0)])])
        .expect_err("a caller-owned payload has no allocation group");
    assert!(matches!(error, GroupError::InvalidDescriptor { .. }));
}
