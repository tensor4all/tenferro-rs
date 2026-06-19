use tenferro_ad::{EagerRuntime, TracedTensorAdExt};
use tenferro_cpu::CpuBackend;
use tenferro_einsum::EagerEinsumExt;
use tenferro_einsum::EinsumOptimize;
use tenferro_einsum::GraphCompilerEinsumExt;
use tenferro_runtime::{GraphCompiler, GraphExecutor, Tensor, TracedTensor};

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

fn matrix_a() -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(Tensor::from_vec_col_major(
        vec![2, 3],
        vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0],
    )?)
}

fn matrix_b() -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(Tensor::from_vec_col_major(
        vec![3, 2],
        vec![7.0_f64, 9.0, 11.0, 8.0, 10.0, 12.0],
    )?)
}

fn matrix_c() -> Result<TracedTensor, Box<dyn std::error::Error>> {
    Ok(TracedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0_f64, 3.0, 2.0, 4.0],
    )?)
}

fn run(tensor: &TracedTensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(tensor)?;
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor.register_extension(tenferro_einsum::register_runtime)?;
    Ok(executor.run(&program)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = EagerRuntime::new();
    let a = runtime.variable_from(matrix_a()?)?;
    let b = runtime.variable_from(matrix_b()?)?;
    let product = [&a, &b].einsum("ij,jk->ik")?;

    assert_eq!(product.shape(), &[2, 2]);
    assert_close(
        product.materialized()?.as_slice::<f64>().unwrap(),
        &[58.0, 139.0, 64.0, 154.0],
    );

    let a = TracedTensor::from_tensor_concrete_shape(matrix_a()?)?;
    let b = TracedTensor::from_tensor_concrete_shape(matrix_b()?)?;
    let c = matrix_c()?;

    let mut compiler = GraphCompiler::new();
    let auto = compiler.einsum_with(&[&a, &b, &c], "ij,jk,kl->il", EinsumOptimize::default())?;
    let left_to_right =
        compiler.einsum_with(&[&a, &b, &c], "ij,jk,kl->il", EinsumOptimize::False)?;

    let auto_value = run(&auto)?;
    let left_to_right_value = run(&left_to_right)?;
    assert_close(
        auto_value.as_slice::<f64>().unwrap(),
        left_to_right_value.as_slice::<f64>().unwrap(),
    );
    assert_close(
        auto_value.as_slice::<f64>().unwrap(),
        &[250.0, 601.0, 372.0, 894.0],
    );

    let y = compiler.einsum_with(&[&a, &b], "ij,jk->ik", EinsumOptimize::Path(vec![(0, 1)]))?;
    let grad_a = y.reduce_sum(&[0, 1])?.grad(&a)?;
    let grad_value = run(&grad_a)?;

    assert_eq!(grad_value.shape(), &[2, 3]);
    assert_close(
        grad_value.as_slice::<f64>().unwrap(),
        &[15.0, 15.0, 19.0, 19.0, 23.0, 23.0],
    );

    Ok(())
}
