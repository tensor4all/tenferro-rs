//! Focused Apple shared-storage and Accelerate coverage without the general integration suite.
#![cfg(target_os = "macos")]

#[path = "integration/apple_shared.rs"]
mod apple_shared;
#[path = "integration/support.rs"]
pub mod support;

use tenferro_cpu::CpuBackend;
use tenferro_linalg::LinalgBackend;
use tenferro_tensor::{BackendSessionHost, DotGeneralConfig, Tensor, TensorRead};

#[test]
fn accelerate_gemm_and_cholesky_without_metal() {
    // This target only builds with the Accelerate-backed `blas` adapter.
    let mut cpu = CpuBackend::with_threads(1).unwrap();
    let input = Tensor::from_vec_col_major([2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]).unwrap();
    let product = cpu
        .with_backend_session(|session| {
            session.dot_general_read(
                TensorRead::from_tensor(&input),
                TensorRead::from_tensor(&input),
                &DotGeneralConfig {
                    lhs_contracting_dims: [1].as_slice().into(),
                    rhs_contracting_dims: [0].as_slice().into(),
                    lhs_batch_dims: [].as_slice().into(),
                    rhs_batch_dims: [].as_slice().into(),
                },
            )
        })
        .unwrap()
        .unwrap();
    assert_eq!(
        product.as_slice::<f64>().unwrap(),
        &[20.0, 14.0, 14.0, 13.0]
    );
    let lower = support::with_cpu_linalg(&mut cpu, |session| session.cholesky(&input)).unwrap();
    for (&actual, expected) in
        lower
            .as_slice::<f64>()
            .unwrap()
            .iter()
            .zip([2.0, 1.0, 0.0, 2.0_f64.sqrt()])
    {
        assert!((actual - expected).abs() < 1.0e-12);
    }
}
