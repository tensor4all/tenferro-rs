use num_complex::Complex64;
use tenferro_internal_ad_surface::{backward, grad, BackwardOptions, GradOptions, Tensor};
use tenferro_linalg::{MatrixNormOrd, VectorNormOrd};

#[test]
fn reverse_only_surface_uses_tidu_value_carrier() {
    let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let out = x.exp().unwrap().sum().unwrap();

    backward(&[&out], None, BackwardOptions::default()).unwrap();

    let grad = x.grad().unwrap().unwrap();
    let values = grad.try_to_vec::<f64>().unwrap();
    assert!((values[0] - 1.0_f64.exp()).abs() < 1.0e-12);
    assert!((values[1] - 2.0_f64.exp()).abs() < 1.0e-12);
}

#[test]
fn functional_grad_uses_value_vjp() {
    let x = Tensor::from_slice(&[1.0_f64, 0.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let y = Tensor::from_slice(&[2.0_f64, 3.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let out = x.add(&y).unwrap().sum().unwrap();

    let grads = grad(&[&out], &[&x, &y], None, GradOptions::default()).unwrap();

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
fn grad_and_backward_reject_grad_output_length_mismatch() {
    let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let out = x.sum().unwrap();
    let empty: Vec<Tensor> = Vec::new();

    let grad_err = grad(&[&out], &[&x], Some(&empty), GradOptions::default()).unwrap_err();
    assert!(grad_err
        .to_string()
        .contains("grad_outputs length mismatch: expected 1, found 0"));

    let backward_err = backward(&[&out], Some(&empty), BackwardOptions::default()).unwrap_err();
    assert!(backward_err
        .to_string()
        .contains("grad_outputs length mismatch: expected 1, found 0"));
}

#[test]
fn grad_accumulates_multiple_outputs_and_backward_uses_explicit_seed() {
    let x = Tensor::from_slice(&[0.0_f64, 1.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let out_exp = x.exp().unwrap().sum().unwrap();
    let out_sum = x.sum().unwrap();

    let grads = grad(
        &[&out_exp, &out_sum],
        &[&x],
        None,
        GradOptions { retain_graph: true },
    )
    .unwrap();
    let grad_values = grads[0].as_ref().unwrap().try_to_vec::<f64>().unwrap();
    assert!((grad_values[0] - 2.0).abs() < 1.0e-12);
    assert!((grad_values[1] - (1.0 + std::f64::consts::E)).abs() < 1.0e-12);

    x.zero_grad().unwrap();
    let seed = Tensor::from_slice(&[3.0_f64], &[]).unwrap();
    out_sum.backward_with_seed(&seed).unwrap();
    assert_eq!(
        x.grad().unwrap().unwrap().try_to_vec::<f64>().unwrap(),
        vec![3.0, 3.0]
    );
}

#[test]
fn tensor_helpers_cover_detach_materialization_and_norm_validation_paths() {
    let base = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
    assert!(!base.requires_grad());
    assert_eq!(base.len(), 2);
    assert!(!base.is_empty());
    assert_eq!(base.try_get::<f64>(&[1]).unwrap(), 2.0);
    assert_eq!(base.try_to_vec::<f64>().unwrap(), vec![1.0, 2.0]);
    assert!(base.try_to_vec::<Complex64>().is_err());

    let tracked = base.detach().with_requires_grad(true);
    let detached = tracked.detach();
    assert!(tracked.requires_grad());
    assert!(!tracked.shares_reverse_graph(&detached));

    assert!(tracked
        .vector_norm(VectorNormOrd::P(2.0), Some(&[0]), false)
        .is_err());
    assert!(tracked
        .vector_norm(VectorNormOrd::P(2.0), None, true)
        .is_err());

    let matrix = Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 2.0], &[2, 2]).unwrap();
    assert!(matrix
        .matrix_norm(MatrixNormOrd::Two, Some((0, 0)), false)
        .is_err());
    assert!(matrix
        .matrix_norm(MatrixNormOrd::Two, Some((0, 1)), true)
        .is_err());
}
