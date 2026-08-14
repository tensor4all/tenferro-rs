use tenferro_cpu::CpuBackend;
use tenferro_linalg::prelude::*;
use tenferro_runtime::prelude::*;

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error < 1.0e-12,
            "value {index}: actual={actual}, expected={expected}, error={error}"
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = CpuBackend::new();

    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![3.0, 0.0, 0.0, 1.0])?;
    let identity = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])?;

    let product = backend.with_backend_session(|session| a.matmul(&identity, session))?;
    assert_eq!(product.shape(), &[2, 2]);
    assert_close(product.host_data()?, &[3.0, 0.0, 0.0, 1.0]);

    let (u, s, vt) = backend.with_backend_session(|session| product.svd(session))?;
    assert_eq!(u.shape(), &[2, 2]);
    assert_eq!(s.shape(), &[2]);
    assert_eq!(vt.shape(), &[2, 2]);
    assert_close(s.as_slice()?, &[3.0, 1.0]);

    Ok(())
}
