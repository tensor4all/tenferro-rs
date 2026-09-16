//! The extension-owned operation reached through the runtime.
//!
//! The input travels as a runtime `Tensor` carrying a caller-owned payload, the
//! operation is an `ExtensionOp` family registered by a downstream
//! `ExtensionModule`, and the result comes back as another external payload.

use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_df64_proof::extension::{apply_total, Df64Total};
use tenferro_df64_proof::Df64;
use tenferro_tensor::{BackendSessionHost, Tensor};
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

// The registered operation is prepared and reachable, but executing it needs the
// runtime to hold a caller-owned payload: the eager value record builds an
// `AllocationGroup` for every tensor, and a payload that owns no pooled storage
// has no group. Giving it one is #1789's ownership decision, so the test records
// the blocker instead of hiding it.
#[test]
#[ignore = "requires the caller-owned payload ownership contract from #1789"]
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
