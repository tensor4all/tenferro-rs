use tenferro::{CpuBackend, Engine, Tensor, TracedTensor, TypedTensor};

const TOL: f64 = 1.0e-6;
const FD_H: f64 = 1.0e-6;

fn f64_scalar(value: f64) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(vec![], vec![value]))
}

fn get_f64_scalar(tensor: &Tensor) -> f64 {
    match tensor {
        Tensor::F64(inner) => inner.host_data()[0],
        other => panic!("expected F64 tensor, got {:?}", other.dtype()),
    }
}

fn eval_tensor(traced: TracedTensor) -> Tensor {
    let mut engine = Engine::new(CpuBackend::new());
    let mut traced = traced;
    traced.eval(&mut engine).unwrap().clone()
}

#[test]
fn checkpoint_preserves_eval_value() {
    let mut engine = Engine::new(CpuBackend::new());
    let x = TracedTensor::from_tensor(f64_scalar(3.0));
    let mut y = &x * &x;

    y.checkpoint(&mut engine).unwrap();

    let value = get_f64_scalar(y.data.as_ref().expect("checkpoint should retain data"));
    assert!((value - 9.0).abs() < 1.0e-12);
}

#[test]
fn checkpoint_downstream_eval_uses_leaf() {
    let mut engine = Engine::new(CpuBackend::new());
    let x = TracedTensor::from_tensor(f64_scalar(3.0));
    let mut y = &x * &x;

    y.checkpoint(&mut engine).unwrap();

    let one = TracedTensor::from_tensor(f64_scalar(1.0));
    let mut z = &y + &one;
    let value = get_f64_scalar(z.eval(&mut engine).unwrap());
    assert!((value - 10.0).abs() < 1.0e-12);
}

#[test]
fn checkpoint_grad_correct() {
    let x_value = 2.0_f64;
    let mut engine = Engine::new(CpuBackend::new());

    let x = TracedTensor::from_tensor(f64_scalar(x_value));
    let mut y = &x * &x;
    y.checkpoint(&mut engine).unwrap();

    let z = &y * &y;
    let grad = z.grad(&x).unwrap();
    let mut grad = grad;
    let grad_value = get_f64_scalar(grad.eval(&mut engine).unwrap());

    let fd = (((x_value + FD_H).powi(4)) - ((x_value - FD_H).powi(4))) / (2.0 * FD_H);
    assert!((grad_value - fd).abs() < TOL, "grad={grad_value}, fd={fd}");
}

#[test]
fn checkpoint_hvp_correct() {
    let x_value = 2.0_f64;
    let mut engine = Engine::new(CpuBackend::new());

    let x = TracedTensor::from_tensor(f64_scalar(x_value));
    let mut y = &x * &x;
    y.checkpoint(&mut engine).unwrap();
    let z = &y * &y;

    let grad = z.grad(&x).unwrap();
    let v = TracedTensor::from_tensor(f64_scalar(1.0));
    let hv = grad.jvp(&x, &v);
    let hv_value = get_f64_scalar(&eval_tensor(hv));
    let expected = 12.0 * x_value * x_value;
    assert!(
        (hv_value - expected).abs() < TOL,
        "HVP: actual={hv_value}, expected={expected}"
    );

    let fd_grad = |value: f64| {
        let x = TracedTensor::from_tensor(f64_scalar(value));
        let mut y = &x * &x;
        y.checkpoint(&mut Engine::new(CpuBackend::new())).unwrap();
        let z = &y * &y;
        let grad = z.grad(&x).unwrap();
        get_f64_scalar(&eval_tensor(grad))
    };
    let fd_hv = (fd_grad(x_value + FD_H) - fd_grad(x_value - FD_H)) / (2.0 * FD_H);
    assert!(
        (hv_value - fd_hv).abs() < TOL,
        "HVP: ad={hv_value}, fd={fd_hv}"
    );
}

#[test]
fn checkpoint_loop_grad_correct() {
    let a_value = 0.8_f64;
    let x0_value = 0.5_f64;
    let steps = 3;
    let mut engine = Engine::new(CpuBackend::new());

    let a = TracedTensor::from_tensor(f64_scalar(a_value));
    let mut x = TracedTensor::from_tensor(f64_scalar(x0_value));

    for _ in 0..steps {
        x = &a * &x.cos();
        x.checkpoint(&mut engine).unwrap();
    }

    let grad = x.grad(&a).unwrap();
    let mut grad = grad;
    let grad_value = get_f64_scalar(grad.eval(&mut engine).unwrap());

    let f_concrete = |a_value: f64| {
        let mut x = x0_value;
        for _ in 0..steps {
            x = a_value * x.cos();
        }
        x
    };
    let fd = (f_concrete(a_value + FD_H) - f_concrete(a_value - FD_H)) / (2.0 * FD_H);
    assert!(
        (grad_value - fd).abs() < TOL,
        "loop grad: actual={grad_value}, fd={fd}"
    );
}

#[test]
fn grad_both_independently_checkpointed_add_wrt_rhs() {
    let mut engine = Engine::new(CpuBackend::new());

    let x = TracedTensor::from_tensor(f64_scalar(2.0));
    let y = TracedTensor::from_tensor(f64_scalar(3.0));

    let mut x2 = &x * &x;
    let mut y2 = &y * &y;

    x2.checkpoint(&mut engine).unwrap();
    y2.checkpoint(&mut engine).unwrap();

    let z = &x2 + &y2;

    let grad_y = z.grad(&y).unwrap();
    let mut grad_y = grad_y;
    let grad_y_value = get_f64_scalar(grad_y.eval(&mut engine).unwrap());
    assert!(
        (grad_y_value - 6.0).abs() < TOL,
        "d/dy (x^2 + y^2) at x=2, y=3: expected 6.0, got {grad_y_value}"
    );
}

#[test]
fn grad_both_independently_checkpointed_add_wrt_lhs() {
    let mut engine = Engine::new(CpuBackend::new());

    let x = TracedTensor::from_tensor(f64_scalar(2.0));
    let y = TracedTensor::from_tensor(f64_scalar(3.0));

    let mut x2 = &x * &x;
    let mut y2 = &y * &y;

    x2.checkpoint(&mut engine).unwrap();
    y2.checkpoint(&mut engine).unwrap();

    let z = &x2 + &y2;

    let grad_x = z.grad(&x).unwrap();
    let mut grad_x = grad_x;
    let grad_x_value = get_f64_scalar(grad_x.eval(&mut engine).unwrap());
    assert!(
        (grad_x_value - 4.0).abs() < TOL,
        "d/dx (x^2 + y^2) at x=2, y=3: expected 4.0, got {grad_x_value}"
    );
}

#[test]
fn grad_both_independently_checkpointed_mul_wrt_rhs() {
    let mut engine = Engine::new(CpuBackend::new());

    let x = TracedTensor::from_tensor(f64_scalar(2.0));
    let y = TracedTensor::from_tensor(f64_scalar(3.0));

    let mut x2 = &x * &x;
    let mut y2 = &y * &y;

    x2.checkpoint(&mut engine).unwrap();
    y2.checkpoint(&mut engine).unwrap();

    let z = &x2 * &y2;

    let grad_y = z.grad(&y).unwrap();
    let mut grad_y = grad_y;
    let grad_y_value = get_f64_scalar(grad_y.eval(&mut engine).unwrap());
    let expected = 2.0 * 3.0 * 4.0;
    assert!(
        (grad_y_value - expected).abs() < TOL,
        "d/dy (x^2 * y^2) at x=2, y=3: expected {expected}, got {grad_y_value}"
    );
}

#[test]
fn grad_both_independently_checkpointed_matches_fd() {
    let x_val = 2.0_f64;
    let y_val = 3.0_f64;

    let f_concrete = |y: f64| x_val * x_val + y * y;
    let fd = (f_concrete(y_val + FD_H) - f_concrete(y_val - FD_H)) / (2.0 * FD_H);

    let mut engine = Engine::new(CpuBackend::new());
    let x = TracedTensor::from_tensor(f64_scalar(x_val));
    let y = TracedTensor::from_tensor(f64_scalar(y_val));
    let mut x2 = &x * &x;
    let mut y2 = &y * &y;
    x2.checkpoint(&mut engine).unwrap();
    y2.checkpoint(&mut engine).unwrap();
    let z = &x2 + &y2;
    let mut grad_y = z.grad(&y).unwrap();
    let ad_value = get_f64_scalar(grad_y.eval(&mut engine).unwrap());
    assert!(
        (ad_value - fd).abs() < TOL,
        "AD grad={ad_value}, FD grad={fd}"
    );
}
