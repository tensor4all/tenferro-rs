//! An external scalar composes with ordinary operations on one engine.
//!
//! The runtime carries the external payload, the extension's kernel runs on the
//! caller-held engine, and ordinary tensor operations run on the same engine.

// Calls without an engine pass `Exec::serial()`: no token is reachable, so they run on the
// calling thread. No engine is reachable in these helpers.
use tenferro_cpu::{scalar_binary_into, CpuEngine};
use tenferro_df64_proof::{Df64, Df64Add};
use tenferro_tensor::Tensor;
use tenferro_tensor::{DynRank, ErasedHostTensor, Host, TypedTensor};
use tenferro_tensor::{TensorElementwise, TensorRead};

fn external<T>(values: Vec<T>) -> Tensor
where
    T: tenferro_tensor_core::Scalar,
{
    Tensor::external(ErasedHostTensor::new(
        TypedTensor::<_, DynRank, Host>::from_host_vec_col_major(vec![values.len()], values)
            .expect("shape matches data"),
    ))
}

fn payload<T>(tensor: &Tensor) -> &TypedTensor<T, DynRank, Host>
where
    T: tenferro_tensor_core::Scalar,
{
    match tensor.external_payload() {
        Some(value) => value.downcast_ref::<T>().expect("external element type"),
        _ => panic!("expected an externally defined payload"),
    }
}

#[test]
fn an_external_scalar_composes_with_ordinary_operations_on_one_engine() {
    let low = 2f64.powi(-80);
    let lhs = external(vec![Df64::from_f64(1.0), Df64::zero()]);
    let rhs = external(vec![Df64::from_f64(low), Df64::zero()]);
    let mut destination = TypedTensor::<_, DynRank, Host>::from_host_vec_col_major(
        vec![2],
        vec![Df64::zero(), Df64::zero()],
    )
    .expect("shape matches data");

    let mut backend = CpuEngine::serial();
    // Ordinary tensor work on the same engine.
    let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).expect("shape matches");
    let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).expect("shape matches");
    let ordinary = backend
        .add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
        .expect("ordinary addition");
    assert_eq!(ordinary.as_slice::<f64>().expect("f64 slice"), &[4.0, 6.0]);

    // Extension-owned work on the carried payload.
    scalar_binary_into::<Df64, Df64Add>(
        "df64_add",
        &mut destination,
        payload::<Df64>(&lhs),
        payload::<Df64>(&rhs),
        tenferro_cpu::strided_exec::Exec::serial(),
    )
    .expect("extension addition");

    // The low-order component survived the engine composition.
    assert_eq!(destination.as_slice()[0], Df64 { hi: 1.0, lo: low });
}
