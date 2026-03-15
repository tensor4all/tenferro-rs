use super::*;

#[test]
fn linalg_solve_triangular_forward_matches_finite_difference_f64() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = f64_2x2([2.0, 0.0, 1.0, 3.0]);
    let b = DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let da = f64_2x2([0.2, -0.05, 0.1, 0.15]);
    let eps = 1e-6;

    let out = {
        let ad_a = AdTensor::new_forward(a.clone(), da.clone()).unwrap();
        let ad_b = AdTensor::new_primal(b.clone());
        solve_triangular(&ad_a, &ad_b).unwrap()
    };
    let tangent = out.tangent().expect("forward tangent missing");
    assert_eq!(tangent.dims(), out.primal().dims());

    let out_plus = {
        let ad_a = AdTensor::new_primal(add_scaled_f64(&a, &da, eps));
        let ad_b = AdTensor::new_primal(b.clone());
        solve_triangular(&ad_a, &ad_b).unwrap().primal().clone()
    };
    let out_minus = {
        let ad_a = AdTensor::new_primal(add_scaled_f64(&a, &da, -eps));
        let ad_b = AdTensor::new_primal(b.clone());
        solve_triangular(&ad_a, &ad_b).unwrap().primal().clone()
    };

    let fd = central_diff_f64(&out_plus, &out_minus, eps);
    let err = max_abs_diff(tangent, &fd);
    assert!(err < 2e-6, "solve_triangular forward fd mismatch: {err}");
}

#[test]
fn linalg_solve_triangular_backward_matches_finite_difference_f64() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = f64_2x2([2.0, 0.0, 1.0, 3.0]);
    let b = DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let cotangent =
        DenseTensor::<f64>::from_slice(&[0.5, -0.25], &[2], MemoryOrder::ColumnMajor).unwrap();
    let eps = 1e-6;

    let grad_a = solve_triangular_rrule(
        &AdTensor::new_primal(a.clone()),
        &AdTensor::new_primal(b.clone()),
        &AdTensor::new_primal(cotangent.clone()),
        true,
    )
    .unwrap()
    .a;

    let objective = |a_now: &DenseTensor<f64>| -> f64 {
        let ad_a = AdTensor::new_primal(a_now.clone());
        let ad_b = AdTensor::new_primal(b.clone());
        let out = solve_triangular(&ad_a, &ad_b).unwrap();
        sum_mul_f64(out.primal(), &cotangent)
    };

    let base = tensor_to_vec_f64(&a);
    let dims = a.dims().to_vec();
    let mut fd_grad = vec![0.0_f64; base.len()];
    for i in 0..base.len() {
        let mut plus = base.clone();
        plus[i] += eps;
        let mut minus = base.clone();
        minus[i] -= eps;
        let a_plus = tensor_from_vec_f64(&plus, &dims);
        let a_minus = tensor_from_vec_f64(&minus, &dims);
        fd_grad[i] = (objective(&a_plus) - objective(&a_minus)) / (2.0 * eps);
    }

    let fd = tensor_from_vec_f64(&fd_grad, &dims);
    let err = max_abs_diff(&grad_a, &fd);
    assert!(err < 1e-8, "solve_triangular backward fd mismatch: {err}");
}

#[test]
fn linalg_solve_triangular_forward_matches_finite_difference_c64() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = c64_2x2([
        Complex64::new(2.0, 0.3),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, -0.5),
        Complex64::new(3.0, 0.2),
    ]);
    let b = DenseTensor::<Complex64>::from_slice(
        &[Complex64::new(1.0, 0.4), Complex64::new(2.0, -0.3)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let da = c64_2x2([
        Complex64::new(0.2, 0.1),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.1, -0.08),
        Complex64::new(-0.15, 0.12),
    ]);
    let eps = 1e-6;

    let out = {
        let ad_a = AdTensor::new_forward(a.clone(), da.clone()).unwrap();
        let ad_b = AdTensor::new_primal(b.clone());
        solve_triangular(&ad_a, &ad_b).unwrap()
    };
    let tangent = out.tangent().expect("forward tangent missing");
    assert_eq!(tangent.dims(), out.primal().dims());

    let out_plus = {
        let ad_a = AdTensor::new_primal(add_scaled_c64(&a, &da, eps));
        let ad_b = AdTensor::new_primal(b.clone());
        solve_triangular(&ad_a, &ad_b).unwrap().primal().clone()
    };
    let out_minus = {
        let ad_a = AdTensor::new_primal(add_scaled_c64(&a, &da, -eps));
        let ad_b = AdTensor::new_primal(b.clone());
        solve_triangular(&ad_a, &ad_b).unwrap().primal().clone()
    };

    let fd = central_diff_c64(&out_plus, &out_minus, eps);
    let err = complex_max_abs_diff(tangent, &fd);
    assert!(
        err < 3e-6,
        "solve_triangular complex forward fd mismatch: {err}"
    );
}

#[test]
fn linalg_solve_triangular_backward_matches_finite_difference_c64_directional() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = c64_2x2([
        Complex64::new(2.0, 0.3),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, -0.5),
        Complex64::new(3.0, 0.2),
    ]);
    let b = DenseTensor::<Complex64>::from_slice(
        &[Complex64::new(1.0, 0.4), Complex64::new(2.0, -0.3)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let cotangent = DenseTensor::<Complex64>::from_slice(
        &[Complex64::new(0.5, 0.2), Complex64::new(-0.25, 0.1)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let da = c64_2x2([
        Complex64::new(0.2, 0.1),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.1, -0.08),
        Complex64::new(-0.15, 0.12),
    ]);
    let eps = 1e-6;

    let grad_a = solve_triangular_rrule(
        &AdTensor::new_primal(a.clone()),
        &AdTensor::new_primal(b.clone()),
        &AdTensor::new_primal(cotangent.clone()),
        true,
    )
    .unwrap()
    .a;

    let objective = |a_now: &DenseTensor<Complex64>| -> f64 {
        let ad_a = AdTensor::new_primal(a_now.clone());
        let ad_b = AdTensor::new_primal(b.clone());
        let out = solve_triangular(&ad_a, &ad_b).unwrap();
        sum_conj_mul_real_c64(&cotangent, out.primal())
    };

    let fd = (objective(&add_scaled_c64(&a, &da, eps)) - objective(&add_scaled_c64(&a, &da, -eps)))
        / (2.0 * eps);
    let predicted = sum_conj_mul_real_c64(&grad_a, &da);
    let err = (predicted - fd).abs();
    assert!(
        err < 1e-8,
        "solve_triangular complex backward directional fd mismatch: {err}"
    );
}

#[test]
fn eager_local_solve_triangular_rrule_runs() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = f64_2x2([2.0, 0.0, 1.0, 3.0]);
    let b = DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let cotangent =
        DenseTensor::<f64>::from_slice(&[0.5, -0.25], &[2], MemoryOrder::ColumnMajor).unwrap();

    let ad_a = AdTensor::new_primal(a);
    let ad_b = AdTensor::new_primal(b);
    let ad_cotangent = AdTensor::new_primal(cotangent);

    let grad = solve_triangular_rrule(&ad_a, &ad_b, &ad_cotangent, true).unwrap();
    assert_eq!(grad.a.dims(), &[2, 2]);
    assert_eq!(grad.b.dims(), &[2]);
}
