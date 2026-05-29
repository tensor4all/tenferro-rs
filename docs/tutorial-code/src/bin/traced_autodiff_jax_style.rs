use tenferro_ad::TracedTensorAdExt;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};

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

fn run(tensor: &TracedTensor) -> Result<tenferro_runtime::Tensor, tenferro_runtime::Error> {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(tensor)?;
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor.run(&program)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = TracedTensor::from_vec_row_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
    let y = (&x * &x).reduce_sum(&[0]);

    let y_value = run(&y)?;
    assert_eq!(y_value.shape(), &[]);
    assert_close(y_value.as_slice::<f64>().unwrap(), &[14.0]);

    let grad = y.grad(&x)?;
    let grad_value = run(&grad)?;
    assert_eq!(grad_value.shape(), &[3]);
    assert_close(grad_value.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);

    let tangent = TracedTensor::from_vec_row_major(vec![3], vec![0.1_f64, 1.0, -2.0]);
    let directional = y.jvp(&x, &tangent);
    let directional_value = run(&directional)?;
    assert_eq!(directional_value.shape(), &[]);
    assert_close(directional_value.as_slice::<f64>().unwrap(), &[-7.8]);

    Ok(())
}
