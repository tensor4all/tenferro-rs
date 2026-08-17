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
    let rhs = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![6.0, 2.0])?;

    let (product, solution, singular_values) = backend.with_backend_session(|session| {
        let product = a.matmul(&identity, session)?;
        let solution = product.solve(&rhs, session)?;
        let singular_values = product.svdvals(session)?;
        Ok::<_, tenferro_runtime::Error>((product, solution, singular_values))
    })?;
    assert_eq!(product.shape(), &[2, 2]);
    assert_close(product.host_data()?, &[3.0, 0.0, 0.0, 1.0]);
    assert_close(solution.as_slice()?, &[2.0, 2.0]);
    assert_close(singular_values.as_slice()?, &[3.0, 1.0]);

    Ok(())
}
