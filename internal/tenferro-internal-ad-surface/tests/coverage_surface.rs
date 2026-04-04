use chainrules_core::AutodiffError;
use tenferro_internal_ad_surface::{
    backward, grad, BackwardOptions, Error, GradOptions, MatrixNormOrd, Tensor, VectorNormOrd,
};

fn approx_eq(lhs: &[f64], rhs: &[f64]) {
    assert_eq!(lhs.len(), rhs.len());
    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
        assert!((lhs - rhs).abs() < 1.0e-12, "lhs={lhs}, rhs={rhs}");
    }
}

#[test]
fn grad_and_backward_reject_length_mismatches() {
    let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let out = x.sum().unwrap();

    let grad_err = grad(
        &[&out],
        &[&x],
        Some(&[
            Tensor::from_slice(&[1.0_f64], &[]).unwrap(),
            Tensor::from_slice(&[1.0_f64], &[]).unwrap(),
        ]),
        GradOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(
        grad_err,
        Error::Autodiff(AutodiffError::InvalidArgument(message)) if message.contains("grad_outputs length mismatch")
    ));

    let backward_err = backward(
        &[&out],
        Some(&[
            Tensor::from_slice(&[1.0_f64], &[]).unwrap(),
            Tensor::from_slice(&[1.0_f64], &[]).unwrap(),
        ]),
        BackwardOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(
        backward_err,
        Error::Autodiff(AutodiffError::InvalidArgument(message)) if message.contains("grad_outputs length mismatch")
    ));
}

#[test]
fn grad_accumulates_multiple_outputs_and_tensor_helpers_work() {
    let x = Tensor::from_slice(&[0.0_f64, 1.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let out1 = x.exp().unwrap().sum().unwrap();
    let out2 = x.sum().unwrap();

    let grads = grad(&[&out1, &out2], &[&x], None, GradOptions::default()).unwrap();
    let grad = grads[0].as_ref().unwrap();
    approx_eq(
        &grad.try_to_vec::<f64>().unwrap(),
        &[2.0, std::f64::consts::E + 1.0],
    );

    let detached = x.detach();
    assert!(!detached.requires_grad());
    assert!(!x.shares_reverse_graph(&detached));
    x.zero_grad().unwrap();
}

#[test]
fn tensor_norm_validation_and_materialization_errors_are_reported() {
    let vector = Tensor::from_slice(&[3.0_f64, 4.0], &[2]).unwrap();
    let matrix = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    let vector_keepdim_err = vector
        .vector_norm(VectorNormOrd::P(2.0), None, true)
        .unwrap_err();
    assert!(matches!(
        vector_keepdim_err,
        Error::Autodiff(AutodiffError::InvalidArgument(message))
            if message.contains("keepdim=false only")
    ));

    let vector_ord_err = vector
        .vector_norm(VectorNormOrd::Zero, None, false)
        .unwrap_err();
    assert!(vector_ord_err.to_string().contains("vector_norm order"));

    let matrix_dim_err = matrix
        .matrix_norm(MatrixNormOrd::Fro, Some((1, 2)), false)
        .unwrap_err();
    assert!(matches!(
        matrix_dim_err,
        Error::Autodiff(AutodiffError::InvalidArgument(message))
            if message.contains("dim=(0, 1) only")
    ));

    let matrix_ord_err = matrix
        .matrix_norm(MatrixNormOrd::NegTwo, Some((0, 1)), false)
        .unwrap_err();
    assert!(matrix_ord_err.to_string().contains("matrix_norm order"));

    let materialize_err = matrix.try_to_vec::<f32>().unwrap_err();
    assert!(materialize_err
        .to_string()
        .contains("dtype mismatch in try_to_vec"));
}
