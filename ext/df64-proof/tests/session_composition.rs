//! An external scalar composes with ordinary operations in one admitted session.
//!
//! The runtime carries the external payload, the extension's kernel runs inside
//! the session's admission scope with the session's thread budget, and ordinary
//! tensor operations run in the same session.

use tenferro_cpu::{scalar_binary_into, CpuBackend};
use tenferro_df64_proof::{Df64, Df64Add};
use tenferro_tensor::TensorRead;
use tenferro_tensor::{BackendSessionHost, Tensor};
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

fn external<T>(values: Vec<T>) -> Tensor
where
    T: tenferro_tensor_core::Scalar,
{
    Tensor::external(ErasedHostTensor::new(
        HostTensor::from_vec_col_major(vec![values.len()], values).expect("shape matches data"),
    ))
}

fn payload<T>(tensor: &Tensor) -> &HostTensor<T>
where
    T: tenferro_tensor_core::Scalar,
{
    match tensor.external_payload() {
        Some(value) => value.downcast_ref::<T>().expect("external element type"),
        _ => panic!("expected an externally defined payload"),
    }
}

#[test]
fn an_external_scalar_composes_with_ordinary_operations_in_one_session() {
    let low = 2f64.powi(-80);
    let lhs = external(vec![Df64::from_f64(1.0), Df64::zero()]);
    let rhs = external(vec![Df64::from_f64(low), Df64::zero()]);
    let mut destination = HostTensor::from_vec_col_major(vec![2], vec![Df64::zero(), Df64::zero()])
        .expect("shape matches data");

    let mut backend = CpuBackend::new();
    backend.with_backend_session(|session| {
        // Ordinary tensor work inside the admitted session.
        let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).expect("shape matches");
        let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).expect("shape matches");
        let ordinary = session
            .add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&b))
            .expect("ordinary addition");
        assert_eq!(ordinary.as_slice::<f64>().expect("f64 slice"), &[4.0, 6.0]);

        // Extension-owned work in the same session, on the carried payload.
        scalar_binary_into::<Df64, Df64Add>(
            "df64_add",
            &mut destination,
            payload::<Df64>(&lhs),
            payload::<Df64>(&rhs),
        )
        .expect("extension addition");
    });

    // The low-order component survived the session.
    assert_eq!(destination.as_slice()[0], Df64 { hi: 1.0, lo: low });
}
