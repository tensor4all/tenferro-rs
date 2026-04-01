use num_complex::Complex64;
use tenferro::{
    backward, grad, set_default_runtime, BackwardOptions, GradOptions, RuntimeContext, Tensor,
};
use tenferro_prims::CpuContext;

fn complex_approx_eq(lhs: &[Complex64], rhs: &[Complex64]) {
    assert_eq!(lhs.len(), rhs.len());
    for (left, right) in lhs.iter().zip(rhs.iter()) {
        assert!(
            (left.re - right.re).abs() < 1.0e-12,
            "real parts differ: {left:?} vs {right:?}"
        );
        assert!(
            (left.im - right.im).abs() < 1.0e-12,
            "imaginary parts differ: {left:?} vs {right:?}"
        );
    }
}

#[test]
fn tensor_backward_accumulates_leaf_gradient() {
    let x = Tensor::from_slice(&[0.0_f64, 1.0], &[2])
        .unwrap()
        .with_requires_grad(true);

    let out = x.exp().unwrap().sum().unwrap();
    out.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    let values = grad.try_to_vec::<f64>().unwrap();
    assert_eq!(values.len(), 2);
    assert!((values[0] - 1.0).abs() < 1.0e-12);
    assert!((values[1] - std::f64::consts::E).abs() < 1.0e-12);
}

#[test]
fn functional_grad_matches_additive_vjp() {
    let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let y = Tensor::from_slice(&[3.0_f64, 4.0], &[2])
        .unwrap()
        .with_requires_grad(true);

    let out = x.add(&y).unwrap().sum().unwrap();
    let grads = grad(&[&out], &[&x, &y], None, GradOptions::default()).unwrap();

    assert_eq!(grads.len(), 2);
    assert_eq!(
        grads[0].as_ref().unwrap().try_to_vec::<f64>().unwrap(),
        vec![1.0, 1.0]
    );
    assert_eq!(
        grads[1].as_ref().unwrap().try_to_vec::<f64>().unwrap(),
        vec![1.0, 1.0]
    );
}

#[test]
fn free_backward_uses_default_seed() {
    let x = Tensor::from_slice(&[2.0_f64, 3.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let out = x.exp().unwrap().sum().unwrap();

    backward(&[&out], None, BackwardOptions::default()).unwrap();

    let grad = x.grad().unwrap().unwrap();
    let values = grad.try_to_vec::<f64>().unwrap();
    assert!((values[0] - 2.0_f64.exp()).abs() < 1.0e-12);
    assert!((values[1] - 3.0_f64.exp()).abs() < 1.0e-12);
}

#[test]
fn complex_inv_grad_uses_complex_vjp() {
    let _runtime = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let x = Tensor::from_slice(
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, -1.0),
        ],
        &[2, 2],
    )
    .unwrap()
    .with_requires_grad(true);
    let out = x.inv().unwrap();
    let seed = Tensor::from_slice(
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        &[2, 2],
    )
    .unwrap();

    let grads = grad(&[&out], &[&x], Some(&[seed]), GradOptions::default()).unwrap();
    let grad = grads[0]
        .as_ref()
        .unwrap()
        .try_to_vec::<Complex64>()
        .unwrap();
    assert_eq!(
        grad,
        vec![
            Complex64::new(0.0, -0.5),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]
    );
}

#[test]
fn complex_det_and_slogdet_grad_use_complex_vjp() {
    let _runtime = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let x = Tensor::from_slice(
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, -1.0),
        ],
        &[2, 2],
    )
    .unwrap()
    .with_requires_grad(true);

    let det = x.det().unwrap();
    let det_seed = Tensor::from_slice(&[Complex64::new(1.0, 0.0)], &[]).unwrap();
    let det_grads = grad(&[&det], &[&x], Some(&[det_seed]), GradOptions::default()).unwrap();
    complex_approx_eq(
        &det_grads[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<Complex64>()
            .unwrap(),
        &[
            Complex64::new(2.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, -1.0),
        ],
    );

    let x = Tensor::from_slice(
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, -1.0),
        ],
        &[2, 2],
    )
    .unwrap()
    .with_requires_grad(true);
    let slogdet = x.slogdet().unwrap();
    let seed = Tensor::from_slice(&[1.0_f64], &[]).unwrap();
    let grads = grad(
        &[&slogdet.logabsdet],
        &[&x],
        Some(&[seed]),
        GradOptions::default(),
    )
    .unwrap();
    complex_approx_eq(
        &grads[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<Complex64>()
            .unwrap(),
        &[
            Complex64::new(0.5, 0.5),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.4, -0.2),
        ],
    );
}

#[test]
fn complex_eigh_values_grad_returns_identity_for_trace_loss() {
    let _runtime = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = Tensor::from_slice(
        &[
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(3.0, 0.0),
        ],
        &[2, 2],
    )
    .unwrap()
    .with_requires_grad(true);

    let out = a.eigh().unwrap();
    let loss = out.values.sum().unwrap();
    let grads = grad(&[&loss], &[&a], None, GradOptions::default()).unwrap();
    let grad_a = grads[0]
        .as_ref()
        .unwrap()
        .try_to_vec::<Complex64>()
        .unwrap();

    complex_approx_eq(
        &grad_a,
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    );
}

#[test]
fn complex_vector_and_matrix_norm_support_backward() {
    let _runtime = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let vector = Tensor::from_slice(
        &[
            Complex64::new(3.0, 4.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        &[3],
    )
    .unwrap()
    .with_requires_grad(true);
    let vector_norm = vector
        .vector_norm(tenferro::VectorNormOrd::P(2.0), None, false)
        .unwrap();
    assert_eq!(vector_norm.dims(), &[] as &[usize]);
    assert_eq!(vector_norm.try_to_vec::<f64>().unwrap(), vec![5.0]);
    let vector_grads = grad(&[&vector_norm], &[&vector], None, GradOptions::default()).unwrap();
    complex_approx_eq(
        &vector_grads[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<Complex64>()
            .unwrap(),
        &[
            Complex64::new(3.0 / 5.0, 4.0 / 5.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );

    let matrix = Tensor::from_slice(
        &[
            Complex64::new(3.0, 4.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        &[2, 2],
    )
    .unwrap()
    .with_requires_grad(true);
    let matrix_norm = matrix
        .matrix_norm(tenferro::MatrixNormOrd::Fro, Some((0, 1)), false)
        .unwrap();
    assert_eq!(matrix_norm.dims(), &[] as &[usize]);
    assert_eq!(matrix_norm.try_to_vec::<f64>().unwrap(), vec![5.0]);
    let matrix_grads = grad(&[&matrix_norm], &[&matrix], None, GradOptions::default()).unwrap();
    complex_approx_eq(
        &matrix_grads[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<Complex64>()
            .unwrap(),
        &[
            Complex64::new(3.0 / 5.0, 4.0 / 5.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );
}

#[test]
fn complex_whole_tensor_norm_supports_backward() {
    let _runtime = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let vector = Tensor::from_slice(
        &[
            Complex64::new(3.0, 4.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        &[3],
    )
    .unwrap()
    .with_requires_grad(true);
    let vector_norm = vector.norm(tenferro::NormKind::Lp(2.0)).unwrap();
    assert_eq!(vector_norm.dims(), &[] as &[usize]);
    assert_eq!(vector_norm.try_to_vec::<f64>().unwrap(), vec![5.0]);
    let vector_grads = grad(&[&vector_norm], &[&vector], None, GradOptions::default()).unwrap();
    complex_approx_eq(
        &vector_grads[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<Complex64>()
            .unwrap(),
        &[
            Complex64::new(3.0 / 5.0, 4.0 / 5.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );

    let matrix = Tensor::from_slice(
        &[
            Complex64::new(3.0, 4.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        &[2, 2],
    )
    .unwrap()
    .with_requires_grad(true);
    let matrix_norm = matrix.norm(tenferro::NormKind::Fro).unwrap();
    assert_eq!(matrix_norm.dims(), &[] as &[usize]);
    assert_eq!(matrix_norm.try_to_vec::<f64>().unwrap(), vec![5.0]);
    let matrix_grads = grad(&[&matrix_norm], &[&matrix], None, GradOptions::default()).unwrap();
    complex_approx_eq(
        &matrix_grads[0]
            .as_ref()
            .unwrap()
            .try_to_vec::<Complex64>()
            .unwrap(),
        &[
            Complex64::new(3.0 / 5.0, 4.0 / 5.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );
}
