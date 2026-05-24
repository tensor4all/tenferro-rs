#![cfg(feature = "autodiff")]

use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, Tensor, TracedTensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data))
}

fn get_f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().unwrap()
}

fn eval(output: &TracedTensor) -> Tensor {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(output).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_linalg::register_runtime)
        .unwrap();
    executor.run(&program).unwrap()
}

#[test]
fn svd_singular_value_sum_jvp_uses_extension_ad_rule() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 2],
        vec![9.0, 0.0, 0.0, 0.0, 4.0, 0.0],
    ));
    let da = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 2],
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
    ));

    let (_u, s, _vt) = tenferro_linalg::svd(&a);
    let y = s.reduce_sum(&[0]);
    let dy = y.jvp(&a, &da);
    let out = eval(&dy);

    assert_eq!(out.shape(), &[] as &[usize]);
    assert_eq!(get_f64_data(&out), &[3.0]);
}

#[test]
fn full_piv_lu_solve_grad_uses_extension_ad_rule() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![0.0, 2.0, 1.0, 3.0]));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![-1.0, 5.0]));

    let x = tenferro_linalg::full_piv_lu_solve(&a, &b);
    let loss = x.reduce_sum(&[0, 1]);
    let grad_b = loss.grad(&b).unwrap();
    let out = eval(&grad_b);

    assert_eq!(out.shape(), &[2, 1]);
}

#[test]
fn svd_values_grad_matches_finite_diff() {
    let data = vec![3.0, 0.1, 0.2, 0.3, 2.0, 0.4];
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], data.clone()));

    let (_u, s, _vt) = tenferro_linalg::svd(&a);
    let loss = s.reduce_sum(&[0]);
    let grad = loss.grad(&a).unwrap();
    let out = eval(&grad);

    let actual = get_f64_data(&out);
    assert_eq!(actual.len(), data.len());
    for idx in 0..data.len() {
        let expected = finite_diff_scalar(
            |xs| {
                let input =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], xs.to_vec()));
                let (_u, s, _vt) = tenferro_linalg::svd(&input);
                get_f64_data(&eval(&s.reduce_sum(&[0])))[0]
            },
            &data,
            idx,
            1.0e-6,
        );
        let diff = (actual[idx] - expected).abs();
        assert!(
            diff <= 1.0e-4,
            "idx {idx}: expected {expected}, got {}, diff {diff}",
            actual[idx]
        );
    }
}

fn finite_diff_scalar(f: impl Fn(&[f64]) -> f64, base: &[f64], index: usize, step: f64) -> f64 {
    let mut plus = base.to_vec();
    let mut minus = base.to_vec();
    plus[index] += step;
    minus[index] -= step;
    (f(&plus) - f(&minus)) / (2.0 * step)
}
