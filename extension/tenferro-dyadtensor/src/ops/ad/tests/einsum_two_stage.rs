use super::*;

#[test]
fn einsum_forward_two_stage_matches_finite_difference_f64() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = f64_2x2([1.0, 3.0, 2.0, 4.0]);
    let b = f64_2x2([2.0, -1.0, 0.5, 1.5]);
    let c = f64_2x2([0.7, -0.4, 1.2, 0.3]);
    let da = f64_2x2([0.2, -0.1, 0.3, 0.05]);
    let eps = 1e-6;

    let out = {
        let ad_a = AdTensor::new_forward(a.clone(), da.clone()).unwrap();
        let ad_b = AdTensor::new_primal(b.clone());
        let ad_c = AdTensor::new_primal(c.clone());
        let y1 = einsum("ij,jk->ik", &[&ad_a, &ad_b]).unwrap();
        einsum("ij,jk->ik", &[&y1, &ad_c]).unwrap()
    };
    let tangent = out.tangent().expect("forward tangent missing").clone();

    let out_plus = {
        let ad_a = AdTensor::new_primal(add_scaled_f64(&a, &da, eps));
        let ad_b = AdTensor::new_primal(b.clone());
        let ad_c = AdTensor::new_primal(c.clone());
        let y1 = einsum("ij,jk->ik", &[&ad_a, &ad_b]).unwrap();
        einsum("ij,jk->ik", &[&y1, &ad_c]).unwrap().primal().clone()
    };
    let out_minus = {
        let ad_a = AdTensor::new_primal(add_scaled_f64(&a, &da, -eps));
        let ad_b = AdTensor::new_primal(b.clone());
        let ad_c = AdTensor::new_primal(c.clone());
        let y1 = einsum("ij,jk->ik", &[&ad_a, &ad_b]).unwrap();
        einsum("ij,jk->ik", &[&y1, &ad_c]).unwrap().primal().clone()
    };

    let fd = central_diff_f64(&out_plus, &out_minus, eps);
    let err = max_abs_diff(&tangent, &fd);
    assert!(err < 1e-8, "einsum 2-stage forward fd mismatch: {err}");
}

#[test]
fn einsum_backward_two_stage_matches_finite_difference_f64() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = f64_2x2([1.0, 3.0, 2.0, 4.0]);
    let b = f64_2x2([2.0, -1.0, 0.5, 1.5]);
    let c = f64_2x2([0.7, -0.4, 1.2, 0.3]);
    let cotangent = f64_2x2([0.4, -0.7, 0.2, 0.9]);
    let eps = 1e-6;

    let tape = Tape::<StructuredTensor<f64>>::new();
    let (grad_a, grad_b) = {
        let ad_a = reverse_leaf_f64(a.clone(), &tape);
        let ad_b = reverse_leaf_f64(b.clone(), &tape);
        let ad_c = reverse_leaf_f64(c.clone(), &tape);
        let y1 = einsum("ij,jk->ik", &[&ad_a, &ad_b]).unwrap();
        let y2 = einsum("ij,jk->ik", &[&y1, &ad_c]).unwrap();
        let grads = pullback_wrt(
            &y2,
            &AdTensor::new_primal(cotangent.clone()),
            &[&ad_a, &ad_b],
        )
        .unwrap();
        (
            grads[0].as_ref().expect("missing dA").clone(),
            grads[1].as_ref().expect("missing dB").clone(),
        )
    };

    let objective = |a_now: &Tensor<f64>| -> f64 {
        let ad_a = AdTensor::new_primal(a_now.clone());
        let ad_b = AdTensor::new_primal(b.clone());
        let ad_c = AdTensor::new_primal(c.clone());
        let y1 = einsum("ij,jk->ik", &[&ad_a, &ad_b]).unwrap();
        let y2 = einsum("ij,jk->ik", &[&y1, &ad_c]).unwrap();
        sum_mul_f64(y2.primal(), &cotangent)
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
    assert!(err < 1e-8, "einsum 2-stage backward dA fd mismatch: {err}");

    let objective_b = |b_now: &Tensor<f64>| -> f64 {
        let ad_a = AdTensor::new_primal(a.clone());
        let ad_b = AdTensor::new_primal(b_now.clone());
        let ad_c = AdTensor::new_primal(c.clone());
        let y1 = einsum("ij,jk->ik", &[&ad_a, &ad_b]).unwrap();
        let y2 = einsum("ij,jk->ik", &[&y1, &ad_c]).unwrap();
        sum_mul_f64(y2.primal(), &cotangent)
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
    assert!(
        err_b < 1e-8,
        "einsum 2-stage backward dB fd mismatch: {err_b}"
    );
}

#[test]
fn einsum_forward_two_stage_matches_finite_difference_c64() {
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
    let c = c64_2x2([
        Complex64::new(0.7, -0.2),
        Complex64::new(-0.4, 0.1),
        Complex64::new(1.2, 0.4),
        Complex64::new(0.3, -0.3),
    ]);
    let da = c64_2x2([
        Complex64::new(0.2, -0.1),
        Complex64::new(-0.1, 0.05),
        Complex64::new(0.3, 0.2),
        Complex64::new(0.05, -0.15),
    ]);
    let eps = 1e-6;

    let out = {
        let ad_a = AdTensor::new_forward(a.clone(), da.clone()).unwrap();
        let ad_b = AdTensor::new_primal(b.clone());
        let ad_c = AdTensor::new_primal(c.clone());
        let y1 = einsum("ij,jk->ik", &[&ad_a, &ad_b]).unwrap();
        einsum("ij,jk->ik", &[&y1, &ad_c]).unwrap()
    };
    let tangent = out.tangent().expect("forward tangent missing").clone();

    let out_plus = {
        let ad_a = AdTensor::new_primal(add_scaled_c64(&a, &da, eps));
        let ad_b = AdTensor::new_primal(b.clone());
        let ad_c = AdTensor::new_primal(c.clone());
        let y1 = einsum("ij,jk->ik", &[&ad_a, &ad_b]).unwrap();
        einsum("ij,jk->ik", &[&y1, &ad_c]).unwrap().primal().clone()
    };
    let out_minus = {
        let ad_a = AdTensor::new_primal(add_scaled_c64(&a, &da, -eps));
        let ad_b = AdTensor::new_primal(b.clone());
        let ad_c = AdTensor::new_primal(c.clone());
        let y1 = einsum("ij,jk->ik", &[&ad_a, &ad_b]).unwrap();
        einsum("ij,jk->ik", &[&y1, &ad_c]).unwrap().primal().clone()
    };

    let fd = central_diff_c64(&out_plus, &out_minus, eps);
    let err = complex_max_abs_diff(&tangent, &fd);
    assert!(
        err < 1e-8,
        "einsum complex 2-stage forward fd mismatch: {err}"
    );
}

#[test]
fn einsum_backward_two_stage_matches_finite_difference_c64_directional() {
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
    let c = c64_2x2([
        Complex64::new(0.7, -0.2),
        Complex64::new(-0.4, 0.1),
        Complex64::new(1.2, 0.4),
        Complex64::new(0.3, -0.3),
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

    let tape = Tape::<StructuredTensor<Complex64>>::new();
    let grad_a = {
        let ad_a = reverse_leaf_c64(a.clone(), &tape);
        let ad_b = reverse_leaf_c64(b.clone(), &tape);
        let ad_c = reverse_leaf_c64(c.clone(), &tape);
        let y1 = einsum("ij,jk->ik", &[&ad_a, &ad_b]).unwrap();
        let y2 = einsum("ij,jk->ik", &[&y1, &ad_c]).unwrap();
        let grads = pullback_wrt(&y2, &AdTensor::new_primal(cotangent.clone()), &[&ad_a]).unwrap();
        grads[0].as_ref().expect("missing dA").clone()
    };

    let objective = |a_now: &Tensor<Complex64>| -> Complex64 {
        let ad_a = AdTensor::new_primal(a_now.clone());
        let ad_b = AdTensor::new_primal(b.clone());
        let ad_c = AdTensor::new_primal(c.clone());
        let y1 = einsum("ij,jk->ik", &[&ad_a, &ad_b]).unwrap();
        let y2 = einsum("ij,jk->ik", &[&y1, &ad_c]).unwrap();
        sum_mul_c64(y2.primal(), &cotangent)
    };

    let fd = (objective(&add_scaled_c64(&a, &da, eps)) - objective(&add_scaled_c64(&a, &da, -eps)))
        / (2.0 * eps);
    let predicted = sum_mul_c64(&grad_a, &da);
    let err = (predicted - fd).norm();
    assert!(
        err < 1e-8,
        "einsum complex 2-stage backward directional fd mismatch: {err}"
    );
}
