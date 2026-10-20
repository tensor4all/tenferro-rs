use tenferro_cpu::CpuEngine;
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
    let mut engine = CpuEngine::serial();

    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![3.0, 0.0, 0.0, 1.0])?;
    let identity = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0])?;
    let rhs = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![6.0, 2.0])?;

    // One borrowed engine carries every operation, the plan cache and the
    // buffer pool.
    let product = a.matmul(&identity, &mut engine)?;
    let solution = product.solve(&rhs, &mut engine)?;
    let singular_values = product.svdvals(&mut engine)?;
    assert_eq!(product.shape(), &[2, 2]);
    assert_close(product.host_data()?, &[3.0, 0.0, 0.0, 1.0]);
    assert_close(solution.as_slice()?, &[2.0, 2.0]);
    assert_close(singular_values.as_slice()?, &[3.0, 1.0]);

    Ok(())
}
