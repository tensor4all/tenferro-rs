use num_complex::{Complex64, ComplexFloat};
use tenferro_ad::TracedTensorAdExt;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, GraphExecutor, Tensor, TracedTensor};

const TOL: f64 = 1.0e-12;

fn assert_complex_close(actual: Complex64, expected: Complex64) {
    let error = (actual - expected).norm();
    assert!(
        error < TOL,
        "actual={actual}, expected={expected}, error={error}"
    );
}

fn assert_complex_slice_close(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert_complex_close(actual, expected);
    }
}

fn run(tensor: &TracedTensor) -> Result<Tensor, tenferro_runtime::Error> {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(tensor)?;
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor.run(&program)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let z_data = vec![
        Complex64::new(0.0, std::f64::consts::FRAC_PI_2),
        Complex64::new(std::f64::consts::LN_2, 0.0),
    ];
    let z = TracedTensor::from_vec_col_major(vec![2], z_data.clone())?;

    let holomorphic_loss = z.exp()?.reduce_sum(Some(&[0]))?;
    let tenferro_grad = holomorphic_loss.grad(&z)?;
    let tenferro_grad_value = run(&tenferro_grad)?;
    let expected_grad: Vec<_> = z_data.iter().map(|z| z.exp().conj()).collect();
    assert_complex_slice_close(
        tenferro_grad_value.as_slice::<Complex64>().unwrap(),
        &expected_grad,
    );

    let magnitude = z.abs()?;
    let real_loss = (&magnitude * &magnitude)?.reduce_sum(Some(&[0]));
    let real_loss_grad = real_loss?.grad(&z)?;
    let real_loss_grad_value = run(&real_loss_grad)?;
    let expected_real_loss_grad: Vec<_> = z_data.iter().map(|z| *z * 2.0).collect();
    assert_complex_slice_close(
        real_loss_grad_value.as_slice::<Complex64>().unwrap(),
        &expected_real_loss_grad,
    );

    let scalar_z = TracedTensor::from_vec_col_major(
        vec![1],
        vec![Complex64::new(0.0, std::f64::consts::FRAC_PI_2)],
    )?;
    let y = scalar_z.exp()?;
    let seed = Complex64::new(2.0, -3.0);
    let cotangent = TracedTensor::from_vec_col_major(vec![1], vec![seed])?;
    let vjp = y.vjp(&scalar_z, &cotangent)?;
    let vjp_value = run(&vjp)?;
    let expected_vjp = seed
        * Complex64::new(0.0, std::f64::consts::FRAC_PI_2)
            .exp()
            .conj();
    assert_complex_close(vjp_value.as_slice::<Complex64>().unwrap()[0], expected_vjp);

    Ok(())
}
