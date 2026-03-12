use super::*;

#[test]
fn einsum_frule_matches_finite_difference_f64_one_stage() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = f64_2x2([1.0, 3.0, 2.0, 4.0]);
    let b = f64_2x2([2.0, -1.0, 0.5, 1.5]);
    let da = f64_2x2([0.2, -0.1, 0.3, 0.05]);
    let eps = 1e-6;

    let ad_a = AdTensor::new_primal(a.clone());
    let ad_b = AdTensor::new_primal(b.clone());
    let ad_da = AdTensor::new_primal(da.clone());
    let jvp = einsum_frule("ij,jk->ik", &[&ad_a, &ad_b], &[Some(&ad_da), None]).unwrap();

    let a_plus = add_scaled_f64(&a, &da, eps);
    let a_minus = add_scaled_f64(&a, &da, -eps);

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

    let fd = central_diff_f64(&out_plus, &out_minus, eps);
    let err = max_abs_diff(&jvp, &fd);
    assert!(err < 1e-8, "einsum frule fd mismatch: {err}");
}

#[test]
fn einsum_rrule_matches_finite_difference_f64_one_stage() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = f64_2x2([1.0, 3.0, 2.0, 4.0]);
    let b = f64_2x2([2.0, -1.0, 0.5, 1.5]);
    let cotangent = f64_2x2([0.4, -0.7, 0.2, 0.9]);
    let eps = 1e-6;

    let (grad_a, grad_b) = {
        let ad_a = AdTensor::new_primal(a.clone());
        let ad_b = AdTensor::new_primal(b.clone());
        let ad_cot = AdTensor::new_primal(cotangent.clone());
        let grads = einsum_rrule("ij,jk->ik", &[&ad_a, &ad_b], &ad_cot).unwrap();
        (grads[0].clone(), grads[1].clone())
    };

    let objective_a = |a_now: &Tensor<f64>| -> f64 {
        let ad_a = AdTensor::new_primal(a_now.clone());
        let ad_b = AdTensor::new_primal(b.clone());
        let out = einsum("ij,jk->ik", &[&ad_a, &ad_b]).unwrap();
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
        fd_grad[i] = (objective_a(&a_plus) - objective_a(&a_minus)) / (2.0 * eps);
    }

    let fd = tensor_from_vec_f64(&fd_grad, &dims);
    let err = max_abs_diff(&grad_a, &fd);
    assert!(err < 1e-8, "einsum rrule dA fd mismatch: {err}");

    let objective_b = |b_now: &Tensor<f64>| -> f64 {
        let ad_a = AdTensor::new_primal(a.clone());
        let ad_b = AdTensor::new_primal(b_now.clone());
        let out = einsum("ij,jk->ik", &[&ad_a, &ad_b]).unwrap();
        sum_mul_f64(out.primal(), &cotangent)
    };
    let base_b = tensor_to_vec_f64(&b);
    let dims_b = b.dims().to_vec();
    let mut fd_grad_b = vec![0.0_f64; base_b.len()];
    for i in 0..base_b.len() {
        let mut plus = base_b.clone();
        plus[i] += eps;
        let mut minus = base_b.clone();
        minus[i] -= eps;
        let b_plus = tensor_from_vec_f64(&plus, &dims_b);
        let b_minus = tensor_from_vec_f64(&minus, &dims_b);
        fd_grad_b[i] = (objective_b(&b_plus) - objective_b(&b_minus)) / (2.0 * eps);
    }
    let fd_b = tensor_from_vec_f64(&fd_grad_b, &dims_b);
    let err_b = max_abs_diff(&grad_b, &fd_b);
    assert!(err_b < 1e-8, "einsum rrule dB fd mismatch: {err_b}");
}

#[test]
fn einsum_hvp_matches_finite_difference_f64_two_stage() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = f64_2x2([1.0, 3.0, 2.0, 4.0]);
    let b = f64_2x2([2.0, -1.0, 0.5, 1.5]);
    let da = f64_2x2([0.2, -0.1, 0.3, 0.05]);
    let eps = 1e-6;

    let c = {
        let ad_a = AdTensor::new_primal(a.clone());
        let ad_b = AdTensor::new_primal(b.clone());
        einsum("ij,jk->ik", &[&ad_a, &ad_b])
            .unwrap()
            .primal()
            .clone()
    };
    let grad_c = scale_f64(&c, 2.0);
    let da_b = {
        let ad_da = AdTensor::new_primal(da.clone());
        let ad_b = AdTensor::new_primal(b.clone());
        einsum("ij,jk->ik", &[&ad_da, &ad_b])
            .unwrap()
            .primal()
            .clone()
    };
    let dgrad_c = scale_f64(&da_b, 2.0);

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

    let grad_from_two_stage = |a_now: &Tensor<f64>| -> Tensor<f64> {
        let c_now = {
            let ad_a = AdTensor::new_primal(a_now.clone());
            let ad_b = AdTensor::new_primal(b.clone());
            einsum("ij,jk->ik", &[&ad_a, &ad_b])
                .unwrap()
                .primal()
                .clone()
        };
        let grad_c_now = scale_f64(&c_now, 2.0);
        let ad_a = AdTensor::new_primal(a_now.clone());
        let ad_b = AdTensor::new_primal(b.clone());
        let ad_grad_c = AdTensor::new_primal(grad_c_now);
        einsum_rrule("ij,jk->ik", &[&ad_a, &ad_b], &ad_grad_c)
            .unwrap()
            .remove(0)
    };

    let grad_plus = grad_from_two_stage(&add_scaled_f64(&a, &da, eps));
    let grad_minus = grad_from_two_stage(&add_scaled_f64(&a, &da, -eps));
    let fd_hvp = central_diff_f64(&grad_plus, &grad_minus, eps);

    let err = max_abs_diff(&hvp_a, &fd_hvp);
    assert!(err < 1e-6, "einsum hvp fd mismatch: {err}");
}
