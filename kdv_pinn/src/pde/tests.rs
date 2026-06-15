use super::*;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{DType, GraphCompiler, GraphExecutor, TracedTensor};
use tenferro_tensor::Tensor;

fn eval(tensor: &TracedTensor, bindings: &[(&TracedTensor, &Tensor)]) -> Tensor {
    let mut compiler = GraphCompiler::new();
    let specs: Vec<(&TracedTensor, DType, &[usize])> = bindings
        .iter()
        .map(|(p, t)| (*p, t.dtype(), t.shape()))
        .collect();
    let program = compiler.compile_with_input_specs(tensor, &specs).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor.run_with_inputs(&program, bindings).unwrap()
}

#[test]
fn kdv_residual_of_zero_solution_is_zero() {
    let x = TracedTensor::input_concrete_shape(DType::F64, &[2, 1]);
    let t = TracedTensor::input_concrete_shape(DType::F64, &[2, 1]);
    // Build a connected function that is identically zero but has non-trivial
    // graph dependencies on both x and t so that repeated JVPs stay active.
    let zero = TracedTensor::from_vec_col_major(vec![2, 1], vec![0.0_f64; 2]);
    let u = x.mul(&x).mul(&x).mul(&t).mul(&zero);
    let r = kdv_residual(&u, &x, &t).unwrap();

    let x_tensor = Tensor::from_vec_col_major(vec![2, 1], vec![0.0_f64; 2]);
    let t_tensor = Tensor::from_vec_col_major(vec![2, 1], vec![0.0_f64; 2]);
    let out = eval(&r, &[(&x, &x_tensor), (&t, &t_tensor)]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[0.0, 0.0]);
}

#[test]
fn third_derivative_of_cube() {
    let x = TracedTensor::input_concrete_shape(DType::F64, &[3, 1]);
    let y = x.mul(&x).mul(&x).sum(&[0, 1]); // x^3 reduced to scalar
    let y_x = grad(&y, &x).unwrap();
    let y_xx = grad(&y_x.sum(&[0, 1]), &x).unwrap();
    let y_xxx = grad(&y_xx.sum(&[0, 1]), &x).unwrap();

    let x_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0]);
    let result = eval(&y_xxx, &[(&x, &x_tensor)]);
    let data = result.as_slice::<f64>().unwrap();
    // d^3/dx^3 x^3 = 6
    assert!((data[0] - 6.0).abs() < 1e-6);
    assert!((data[1] - 6.0).abs() < 1e-6);
    assert!((data[2] - 6.0).abs() < 1e-6);
}
