use super::*;

#[test]
fn einsum_frule_matches_finite_difference_c64_one_stage() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = c64_2x2([
        Complex64::new(1.0, 0.5),
        Complex64::new(3.0, -0.3),
        Complex64::new(2.0, 0.2),
        Complex64::new(4.0, -0.1),
    ]);
    let b = c64_2x2([
        Complex64::new(2.0, 0.1),
        Complex64::new(-1.0, 0.7),
        Complex64::new(0.5, -0.2),
        Complex64::new(1.5, 0.3),
    ]);
    let da = c64_2x2([
        Complex64::new(0.2, -0.1),
        Complex64::new(-0.1, 0.05),
        Complex64::new(0.3, 0.2),
        Complex64::new(0.05, -0.15),
    ]);
    let eps = 1e-6;

    let ad_a = AdTensor::new_primal(a.clone());
    let ad_b = AdTensor::new_primal(b.clone());
    let ad_da = AdTensor::new_primal(da.clone());
    let jvp = einsum_frule("ij,jk->ik", &[&ad_a, &ad_b], &[Some(&ad_da), None]).unwrap();

    let a_plus = add_scaled_c64(&a, &da, eps);
    let a_minus = add_scaled_c64(&a, &da, -eps);

    let out_plus = {
        let ad_a_plus = AdTensor::new_primal(a_plus);
        let ad_b = AdTensor::new_primal(b.clone());
        einsum("ij,jk->ik", &[&ad_a_plus, &ad_b])
            .unwrap()
            .primal()
            .clone()
    };
    let out_minus = {
        let ad_a_minus = AdTensor::new_primal(a_minus);
        let ad_b = AdTensor::new_primal(b.clone());
        einsum("ij,jk->ik", &[&ad_a_minus, &ad_b])
            .unwrap()
            .primal()
            .clone()
    };

    let fd = central_diff_c64(&out_plus, &out_minus, eps);
    let err = complex_max_abs_diff(&jvp, &fd);
    assert!(err < 1e-8, "einsum complex frule fd mismatch: {err}");
}

#[test]
fn einsum_rrule_matches_finite_difference_c64_one_stage_directional() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = c64_2x2([
        Complex64::new(1.0, 0.5),
        Complex64::new(3.0, -0.3),
        Complex64::new(2.0, 0.2),
        Complex64::new(4.0, -0.1),
    ]);
    let b = c64_2x2([
        Complex64::new(2.0, 0.1),
        Complex64::new(-1.0, 0.7),
        Complex64::new(0.5, -0.2),
        Complex64::new(1.5, 0.3),
    ]);
    let cotangent = c64_2x2([
        Complex64::new(0.4, -0.2),
        Complex64::new(-0.7, 0.6),
        Complex64::new(0.2, 0.3),
        Complex64::new(0.9, -0.4),
    ]);
    let da = c64_2x2([
        Complex64::new(0.2, -0.1),
        Complex64::new(-0.1, 0.05),
        Complex64::new(0.3, 0.2),
        Complex64::new(0.05, -0.15),
    ]);
    let eps = 1e-6;

    let (grad_a, grad_b) = {
        let ad_a = AdTensor::new_primal(a.clone());
        let ad_b = AdTensor::new_primal(b.clone());
        let ad_cot = AdTensor::new_primal(cotangent.clone());
        let grads = einsum_rrule("ij,jk->ik", &[&ad_a, &ad_b], &ad_cot).unwrap();
        (grads[0].clone(), grads[1].clone())
    };

    let objective = |a_now: &Tensor<Complex64>| -> Complex64 {
        let ad_a = AdTensor::new_primal(a_now.clone());
        let ad_b = AdTensor::new_primal(b.clone());
        let out = einsum("ij,jk->ik", &[&ad_a, &ad_b]).unwrap();
        sum_mul_c64(out.primal(), &cotangent)
    };

    let fd = (objective(&add_scaled_c64(&a, &da, eps)) - objective(&add_scaled_c64(&a, &da, -eps)))
        / (2.0 * eps);
    let predicted = sum_mul_c64(&grad_a, &da);
    let err = (predicted - fd).norm();
    assert!(
        err < 1e-8,
        "einsum complex rrule directional fd mismatch: {err}"
    );

    let db = c64_2x2([
        Complex64::new(0.07, -0.04),
        Complex64::new(-0.11, 0.03),
        Complex64::new(0.13, 0.09),
        Complex64::new(-0.05, -0.08),
    ]);
    let objective_b = |b_now: &Tensor<Complex64>| -> Complex64 {
        let ad_a = AdTensor::new_primal(a.clone());
        let ad_b = AdTensor::new_primal(b_now.clone());
        let out = einsum("ij,jk->ik", &[&ad_a, &ad_b]).unwrap();
        sum_mul_c64(out.primal(), &cotangent)
    };
    let fd_b = (objective_b(&add_scaled_c64(&b, &db, eps))
        - objective_b(&add_scaled_c64(&b, &db, -eps)))
        / (2.0 * eps);
    let predicted_b = sum_mul_c64(&grad_b, &db);
    let err_b = (predicted_b - fd_b).norm();
    assert!(
        err_b < 1e-8,
        "einsum complex rrule dB directional fd mismatch: {err_b}"
    );
}

#[test]
fn einsum_hvp_matches_finite_difference_c64_two_stage() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = c64_2x2([
        Complex64::new(1.0, 0.5),
        Complex64::new(3.0, -0.3),
        Complex64::new(2.0, 0.2),
        Complex64::new(4.0, -0.1),
    ]);
    let b = c64_2x2([
        Complex64::new(2.0, 0.1),
        Complex64::new(-1.0, 0.7),
        Complex64::new(0.5, -0.2),
        Complex64::new(1.5, 0.3),
    ]);
    let da = c64_2x2([
        Complex64::new(0.2, -0.1),
        Complex64::new(-0.1, 0.05),
        Complex64::new(0.3, 0.2),
        Complex64::new(0.05, -0.15),
    ]);
    let eps = 1e-6;

    let c = {
        let ad_a = AdTensor::new_primal(a.clone());
        let ad_b = AdTensor::new_primal(b.clone());
        einsum("ij,jk->ik", &[&ad_a, &ad_b])
            .unwrap()
            .primal()
            .clone()
    };
    let two = Complex64::new(2.0, 0.0);
    let grad_c = scale_c64(&c, two);
    let da_b = {
        let ad_da = AdTensor::new_primal(da.clone());
        let ad_b = AdTensor::new_primal(b.clone());
        einsum("ij,jk->ik", &[&ad_da, &ad_b])
            .unwrap()
            .primal()
            .clone()
    };
    let dgrad_c = scale_c64(&da_b, two);

    let hvp_a = {
        let ad_a = AdTensor::new_primal(a.clone());
        let ad_b = AdTensor::new_primal(b.clone());
        let ad_da = AdTensor::new_primal(da.clone());
        let ad_grad_c = AdTensor::new_primal(grad_c.clone());
        let ad_dgrad_c = AdTensor::new_primal(dgrad_c.clone());
        einsum_hvp(
            "ij,jk->ik",
            &[&ad_a, &ad_b],
            &[Some(&ad_da), None],
            &ad_grad_c,
            &ad_dgrad_c,
        )
        .unwrap()
        .remove(0)
        .1
    };

    let grad_from_two_stage = |a_now: &Tensor<Complex64>| -> Tensor<Complex64> {
        let c_now = {
            let ad_a = AdTensor::new_primal(a_now.clone());
            let ad_b = AdTensor::new_primal(b.clone());
            einsum("ij,jk->ik", &[&ad_a, &ad_b])
                .unwrap()
                .primal()
                .clone()
        };
        let grad_c_now = scale_c64(&c_now, two);
        let ad_a = AdTensor::new_primal(a_now.clone());
        let ad_b = AdTensor::new_primal(b.clone());
        let ad_grad_c = AdTensor::new_primal(grad_c_now);
        einsum_rrule("ij,jk->ik", &[&ad_a, &ad_b], &ad_grad_c)
            .unwrap()
            .remove(0)
    };

    let grad_plus = grad_from_two_stage(&add_scaled_c64(&a, &da, eps));
    let grad_minus = grad_from_two_stage(&add_scaled_c64(&a, &da, -eps));
    let fd_hvp = central_diff_c64(&grad_plus, &grad_minus, eps);

    let err = complex_max_abs_diff(&hvp_a, &fd_hvp);
    assert!(err < 2e-6, "einsum complex hvp fd mismatch: {err}");
}
