use tenferro::{CpuBackend, Engine, Tensor, TracedTensor, TypedTensor};

fn f64_scalar(value: f64) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(vec![], vec![value]))
}

fn get_f64_scalar(tensor: &Tensor) -> f64 {
    match tensor {
        Tensor::F64(inner) => inner.host_data()[0],
        other => panic!("expected F64 tensor, got {:?}", other.dtype()),
    }
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
