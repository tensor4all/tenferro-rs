use tenferro_cpu::CpuBackend;
use tenferro_runtime::{typed_tensor, TypedTensor};

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

    let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])?;
    assert_eq!(x.shape(), &[2, 3]);
    assert_eq!(x.rank(), 2);
    assert_eq!(x.host_data()?, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    let column_offsets =
        TypedTensor::<f64>::from_vec_col_major(vec![1, 3], vec![10.0, 20.0, 30.0])?;
    let shifted = typed_tensor::add(&x, &column_offsets, &mut backend)?;
    assert_close(shifted.host_data()?, &[11.0, 14.0, 22.0, 25.0, 33.0, 36.0]);

    let squared = typed_tensor::mul(&shifted, &shifted, &mut backend)?;
    let squared_total = typed_tensor::reduce_sum(&squared, &[0, 1], &mut backend)?;
    assert_eq!(squared_total.shape(), &[]);
    assert_close(squared_total.host_data()?, &[3811.0]);

    let transposed = typed_tensor::transpose(&x, &[1, 0], &mut backend)?;
    assert_eq!(transposed.shape(), &[3, 2]);
    assert_close(transposed.host_data()?, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let weights =
        TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![0.5, -1.0, 1.5, 1.0, 2.0, -0.5])?;
    let projected = typed_tensor::matmul(&x, &weights, &mut backend)?;
    assert_eq!(projected.shape(), &[2, 2]);
    assert_close(projected.host_data()?, &[3.0, 6.0, 3.5, 11.0]);

    Ok(())
}
