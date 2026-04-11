use tenferro::{
    backward, grad, set_default_runtime, BackwardOptions, Error, GradOptions, RuntimeContext,
    Tensor,
};
use tenferro_prims::CpuContext;

fn with_cpu() -> DefaultRuntimeGuard {
    set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)))
}

fn scalar_tensor_f64(val: f64) -> Tensor {
    Tensor::from_slice(&[val], &[]).unwrap()
}

#[test]
fn grad_rejects_mismatched_grad_outputs_length() {
    let _rt = with_cpu();
    let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let out = x.exp().unwrap().sum().unwrap();

    let too_many = vec![scalar_tensor_f64(1.0), scalar_tensor_f64(1.0)];
    let result = grad(&[&out], &[&x], Some(&too_many), GradOptions::default());
    match result {
        Err(Error::Autodiff(err)) => {
            let msg = format!("{err}");
            assert!(
                msg.contains("grad_outputs length mismatch"),
                "unexpected error message: {msg}"
            );
            assert!(
                msg.contains("expected 1, found 2"),
                "unexpected error message: {msg}"
            );
        }
        Err(other) => panic!("expected Autodiff error, got: {other}"),
        Ok(_) => panic!("expected error, got success"),
    }
}

#[test]
fn grad_rejects_zero_grad_outputs_when_outputs_exist() {
    let _rt = with_cpu();
    let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let out = x.exp().unwrap().sum().unwrap();

    let empty: Vec<Tensor> = vec![];
    let result = grad(&[&out], &[&x], Some(&empty), GradOptions::default());
    match result {
        Err(Error::Autodiff(err)) => {
            let msg = format!("{err}");
            assert!(
                msg.contains("grad_outputs length mismatch"),
                "unexpected error message: {msg}"
            );
            assert!(
                msg.contains("expected 1, found 0"),
                "unexpected error message: {msg}"
            );
        }
        Err(other) => panic!("expected Autodiff error, got: {other}"),
        Ok(_) => panic!("expected error, got success"),
    }
}

#[test]
fn backward_rejects_mismatched_grad_outputs_length() {
    let _rt = with_cpu();
    let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let out = x.exp().unwrap().sum().unwrap();

    let too_many = vec![scalar_tensor_f64(1.0), scalar_tensor_f64(1.0)];
    let result = backward(&[&out], Some(&too_many), BackwardOptions::default());
    match result {
        Err(Error::Autodiff(err)) => {
            let msg = format!("{err}");
            assert!(
                msg.contains("grad_outputs length mismatch"),
                "unexpected error message: {msg}"
            );
        }
        Err(other) => panic!("expected Autodiff error, got: {other}"),
        Ok(_) => panic!("expected error, got success"),
    }
}

#[test]
fn grad_returns_none_for_non_ad_input() {
    let _rt = with_cpu();
    let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let y = Tensor::from_slice(&[3.0_f64, 4.0], &[2]).unwrap();
    let out = x.exp().unwrap().sum().unwrap();

    let grads = grad(&[&out], &[&y], None, GradOptions::default()).unwrap();
    assert_eq!(grads.len(), 1, "expected one gradient entry");
    assert!(grads[0].is_none(), "non-AD input should have None gradient");
}

#[test]
fn grad_returns_none_for_disconnected_input() {
    let _rt = with_cpu();
    let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let y = Tensor::from_slice(&[3.0_f64, 4.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let out = x.exp().unwrap().sum().unwrap();

    let grads = grad(&[&out], &[&y], None, GradOptions::default()).unwrap();
    assert_eq!(grads.len(), 1, "expected one gradient entry");
    assert!(
        grads[0].is_none(),
        "disconnected input should have None gradient"
    );
}

#[test]
fn grad_works_with_matching_grad_outputs() {
    let _rt = with_cpu();
    let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let out = x.exp().unwrap().sum().unwrap();

    let seed = scalar_tensor_f64(1.0);
    let grads = grad(&[&out], &[&x], Some(&[seed]), GradOptions::default()).unwrap();
    assert_eq!(grads.len(), 1, "expected one gradient entry");
    assert!(grads[0].is_some(), "AD input should have a gradient");
}

#[test]
fn backward_works_with_matching_grad_outputs() {
    let _rt = with_cpu();
    let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2])
        .unwrap()
        .with_requires_grad(true);
    let out = x.exp().unwrap().sum().unwrap();

    let seed = scalar_tensor_f64(1.0);
    backward(&[&out], Some(&[seed]), BackwardOptions::default()).unwrap();
    let grad = x.grad().unwrap().unwrap();
    let values = grad.try_to_vec::<f64>().unwrap();
    assert!((values[0] - 1.0_f64.exp()).abs() < 1e-12);
    assert!((values[1] - 2.0_f64.exp()).abs() < 1e-12);
}
