use super::*;
use tenferro_ad::TracedTensorAdExt;
use tenferro_runtime::{DType, GraphCompiler, Result, TracedTensor};
use tenferro_tensor::Tensor;

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

fn grad(output: &TracedTensor, input: &TracedTensor) -> Result<TracedTensor> {
    output.grad(input)
}

fn eval(tensor: &TracedTensor, bindings: &[(&TracedTensor, &Tensor)]) -> Tensor {
    let mut compiler = GraphCompiler::new();
    let specs: Vec<(&TracedTensor, DType, &[usize])> = bindings
        .iter()
        .map(|(p, t)| (*p, t.dtype(), t.shape()))
        .collect();
    let program = compiler.compile_with_input_specs(tensor, &specs).unwrap();
    let runtime = crate::cpu_runtime().unwrap();
    let inputs: Vec<&Tensor> = bindings.iter().map(|(_, tensor)| *tensor).collect();
    crate::run_single(&runtime, &program, &inputs).unwrap()
}

#[test]
fn kdv_residual_of_zero_solution_is_zero() -> TestResult {
    let x = TracedTensor::input_concrete_shape(DType::F64, &[2, 1])?;
    let t = TracedTensor::input_concrete_shape(DType::F64, &[2, 1])?;
    let zero = TracedTensor::from_vec_col_major(vec![2, 1], vec![0.0_f64; 2])?;
    let u = x.mul(&x)?.mul(&x)?.mul(&t)?.mul(&zero)?;
    let r = kdv_residual(&u, &x, &t)?;

    let x_tensor = Tensor::from_vec_col_major(vec![2, 1], vec![0.0_f64; 2])?;
    let t_tensor = Tensor::from_vec_col_major(vec![2, 1], vec![0.0_f64; 2])?;
    let out = eval(&r, &[(&x, &x_tensor), (&t, &t_tensor)]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[0.0, 0.0]);
    Ok(())
}

#[test]
fn jvp_higher_order_derivatives_of_cube() -> TestResult {
    let x = TracedTensor::input_concrete_shape(DType::F64, &[3, 1])?;
    let y = x.mul(&x)?.mul(&x)?;
    let ones = TracedTensor::from_vec_col_major(vec![3, 1], vec![1.0_f64; 3])?;
    let y_x = y.jvp(&x, &ones)?;
    let y_xx = y_x.jvp(&x, &ones)?;
    let y_xxx = y_xx.jvp(&x, &ones)?;

    let x_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0])?;
    let out_xx = eval(&y_xx, &[(&x, &x_tensor)]);
    let out_xxx = eval(&y_xxx, &[(&x, &x_tensor)]);
    let xx = out_xx.as_slice::<f64>().unwrap();
    let xxx = out_xxx.as_slice::<f64>().unwrap();
    assert!((xx[0] - 6.0).abs() < 1e-6);
    assert!((xx[1] - 12.0).abs() < 1e-6);
    assert!((xx[2] - 18.0).abs() < 1e-6);
    assert!((xxx[0] - 6.0).abs() < 1e-6);
    assert!((xxx[1] - 6.0).abs() < 1e-6);
    assert!((xxx[2] - 6.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn jvp_gives_elementwise_derivative() -> TestResult {
    let x = TracedTensor::input_concrete_shape(DType::F64, &[3, 1])?;
    let y = x.mul(&x)?;
    let ones = TracedTensor::from_vec_col_major(vec![3, 1], vec![1.0_f64; 3])?;
    let dy_dx = y.jvp(&x, &ones)?;

    let x_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0])?;
    let out = eval(&dy_dx, &[(&x, &x_tensor)]);
    let data = out.as_slice::<f64>().unwrap();
    assert!((data[0] - 2.0).abs() < 1e-6);
    assert!((data[1] - 4.0).abs() < 1e-6);
    assert!((data[2] - 6.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn jvp_mixed_variable_second_derivative() -> TestResult {
    let x = TracedTensor::input_concrete_shape(DType::F64, &[3, 1])?;
    let t = TracedTensor::input_concrete_shape(DType::F64, &[3, 1])?;
    let z = x.add(&t.scale_real(-4.0)?)?;
    let u = z.exp()?;
    let ones = TracedTensor::from_vec_col_major(vec![3, 1], vec![1.0_f64; 3])?;
    let u_x = u.jvp(&x, &ones)?;
    let u_xx = u_x.jvp(&x, &ones)?;

    let x_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![-1.0_f64, 0.0_f64, 1.0_f64])?;
    let t_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![0.0_f64, 0.5_f64, 1.0_f64])?;
    let out = eval(&u_xx, &[(&x, &x_tensor), (&t, &t_tensor)]);
    let data = out.as_slice::<f64>().unwrap();
    let u_val = eval(&u, &[(&x, &x_tensor), (&t, &t_tensor)])
        .as_slice::<f64>()
        .unwrap()
        .to_vec();
    for i in 0..3 {
        assert!((data[i] - u_val[i]).abs() < 1e-5);
    }
    Ok(())
}

#[test]
fn jvp_pow_second_derivative() -> TestResult {
    let x = TracedTensor::input_concrete_shape(DType::F64, &[3, 1])?;
    let t = TracedTensor::input_concrete_shape(DType::F64, &[3, 1])?;
    let z = x.add(&t.scale_real(-4.0)?)?;
    let two = TracedTensor::from_vec_col_major(vec![3, 1], vec![2.0_f64; 3])?;
    let u = z.pow(&two)?;
    let ones = TracedTensor::from_vec_col_major(vec![3, 1], vec![1.0_f64; 3])?;
    let u_x = u.jvp(&x, &ones)?;
    let u_xx = u_x.jvp(&x, &ones)?;

    let x_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![-1.0_f64, 0.0_f64, 1.0_f64])?;
    let t_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![0.0_f64, 0.5_f64, 1.0_f64])?;
    let out = eval(&u_xx, &[(&x, &x_tensor), (&t, &t_tensor)]);
    for &v in out.as_slice::<f64>().unwrap() {
        assert!((v - 2.0).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn jvp_division_third_derivative() -> TestResult {
    let x = TracedTensor::input_concrete_shape(DType::F64, &[3, 1])?;
    let t = TracedTensor::input_concrete_shape(DType::F64, &[3, 1])?;
    let z = x.add(&t.scale_real(-4.0)?)?;
    let one = TracedTensor::from_vec_col_major(vec![3, 1], vec![1.0_f64; 3])?;
    let u = one.div(&z.add(&one)?)?;
    let ones = TracedTensor::from_vec_col_major(vec![3, 1], vec![1.0_f64; 3])?;
    let u_x = u.jvp(&x, &ones)?;
    let u_xx = u_x.jvp(&x, &ones)?;
    let u_xxx = u_xx.jvp(&x, &ones)?;

    let x_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![-0.5_f64, 0.0_f64, 0.5_f64])?;
    let t_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![0.0_f64, 0.0_f64, 0.0_f64])?;
    let out = eval(&u_xxx, &[(&x, &x_tensor), (&t, &t_tensor)]);
    let data = out.as_slice::<f64>().unwrap();
    for (i, &actual) in data.iter().enumerate().take(3) {
        let xi = -0.5 + i as f64 * 0.5;
        let expected = -6.0 / (xi + 1.0).powi(4);
        assert!((actual - expected).abs() < 1e-5);
    }
    Ok(())
}

#[test]
fn jvp_pow_division_third_derivative() -> TestResult {
    let x = TracedTensor::input_concrete_shape(DType::F64, &[3, 1])?;
    let t = TracedTensor::input_concrete_shape(DType::F64, &[3, 1])?;
    let z = x.add(&t.scale_real(-4.0)?)?;
    let one = TracedTensor::from_vec_col_major(vec![3, 1], vec![1.0_f64; 3])?;
    let two = TracedTensor::from_vec_col_major(vec![3, 1], vec![2.0_f64; 3])?;
    let u = one.div(&z.add(&one)?.pow(&two)?)?;
    let ones = TracedTensor::from_vec_col_major(vec![3, 1], vec![1.0_f64; 3])?;
    let u_x = u.jvp(&x, &ones)?;
    let u_xx = u_x.jvp(&x, &ones)?;
    let u_xxx = u_xx.jvp(&x, &ones)?;

    let x_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![-0.5_f64, 0.0_f64, 0.5_f64])?;
    let t_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![0.0_f64, 0.0_f64, 0.0_f64])?;
    let out = eval(&u_xxx, &[(&x, &x_tensor), (&t, &t_tensor)]);
    let data = out.as_slice::<f64>().unwrap();
    for (i, &actual) in data.iter().enumerate().take(3) {
        let xi = -0.5 + i as f64 * 0.5;
        let expected = -24.0 / (xi + 1.0).powi(5);
        assert!((actual - expected).abs() < 1e-4);
    }
    Ok(())
}

#[test]
fn jvp_scale_real_third_derivative() -> TestResult {
    let x = TracedTensor::input_concrete_shape(DType::F64, &[3, 1])?;
    let t = TracedTensor::input_concrete_shape(DType::F64, &[3, 1])?;
    let z = x.add(&t.scale_real(-4.0)?)?;
    let three = TracedTensor::from_vec_col_major(vec![3, 1], vec![3.0_f64; 3])?;
    let u = z.scale_real(2.0)?.pow(&three)?;
    let ones = TracedTensor::from_vec_col_major(vec![3, 1], vec![1.0_f64; 3])?;
    let u_x = u.jvp(&x, &ones)?;
    let u_xx = u_x.jvp(&x, &ones)?;
    let u_xxx = u_xx.jvp(&x, &ones)?;

    let x_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![-1.0_f64, 0.0_f64, 1.0_f64])?;
    let t_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![0.0_f64, 0.5_f64, 1.0_f64])?;
    let out = eval(&u_xxx, &[(&x, &x_tensor), (&t, &t_tensor)]);
    for &v in out.as_slice::<f64>().unwrap() {
        assert!((v - 48.0).abs() < 1e-5);
    }
    Ok(())
}

#[test]
fn kdv_residual_of_exact_solution_is_small() -> TestResult {
    let x = TracedTensor::input_concrete_shape(DType::F64, &[3, 1])?;
    let t = TracedTensor::input_concrete_shape(DType::F64, &[3, 1])?;
    let z = x.add(&t.scale_real(-4.0)?)?;
    let one = TracedTensor::from_vec_col_major(vec![3, 1], vec![1.0_f64; 3])?;
    let two = TracedTensor::from_vec_col_major(vec![3, 1], vec![2.0_f64; 3])?;
    let exp_z = z.exp()?;
    let neg_z = z.neg()?;
    let exp_neg_z = neg_z.exp()?;
    let cosh = exp_z.add(&exp_neg_z)?.div(&two)?;
    let sech2 = one.div(&cosh.pow(&two)?)?;
    let u = sech2.scale_real(2.0)?;
    let r = kdv_residual(&u, &x, &t)?;

    let x_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![-1.0_f64, 0.0_f64, 1.0_f64])?;
    let t_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![0.0_f64, 0.5_f64, 1.0_f64])?;
    let out = eval(&r, &[(&x, &x_tensor), (&t, &t_tensor)]);
    for &v in out.as_slice::<f64>().unwrap() {
        assert!(v.abs() < 1e-3, "residual of exact solution too large: {v}");
    }
    Ok(())
}

#[test]
fn third_derivative_of_cube() -> TestResult {
    let x = TracedTensor::input_concrete_shape(DType::F64, &[3, 1])?;
    let y = x.mul(&x)?.mul(&x)?.reduce_sum(Some(&[0, 1]))?;
    let y_x = grad(&y, &x)?;
    let y_xx = grad(&y_x.reduce_sum(Some(&[0, 1]))?, &x)?;
    let y_xxx = grad(&y_xx.reduce_sum(Some(&[0, 1]))?, &x)?;

    let x_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0])?;
    let result = eval(&y_xxx, &[(&x, &x_tensor)]);
    let data = result.as_slice::<f64>().unwrap();
    assert!((data[0] - 6.0).abs() < 1e-6);
    assert!((data[1] - 6.0).abs() < 1e-6);
    assert!((data[2] - 6.0).abs() < 1e-6);
    Ok(())
}
