use tenferro::engine::Engine;
use tenferro::traced::{eval_all, TracedTensor};
use tenferro_tensor::cpu::CpuBackend;
use tenferro_tensor::{DotGeneralConfig, Tensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn get_f64_data(t: &Tensor) -> &[f64] {
    match t {
        Tensor::F64(inner) => inner.host_data(),
        _ => panic!("expected F64"),
    }
}

#[test]
fn test_add() {
    let a = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);
    let b = f64_tensor(vec![3], vec![4.0, 5.0, 6.0]);
    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let mut tc = &ta + &tb;
    let mut engine = Engine::new(CpuBackend::new());
    let result = tc.eval(&mut engine).unwrap();
    assert_eq!(get_f64_data(result), &[5.0, 7.0, 9.0]);
}

#[test]
fn test_mul() {
    let a = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);
    let b = f64_tensor(vec![3], vec![4.0, 5.0, 6.0]);
    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let mut tc = &ta * &tb;
    let mut engine = Engine::new(CpuBackend::new());
    let result = tc.eval(&mut engine).unwrap();
    assert_eq!(get_f64_data(result), &[4.0, 10.0, 18.0]);
}

#[test]
fn test_neg() {
    let a = f64_tensor(vec![3], vec![1.0, -2.0, 3.0]);
    let ta = TracedTensor::from_tensor(a);
    let mut tb = -&ta;
    let mut engine = Engine::new(CpuBackend::new());
    let result = tb.eval(&mut engine).unwrap();
    assert_eq!(get_f64_data(result), &[-1.0, 2.0, -3.0]);
}

#[test]
fn test_dot_general() {
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
        lhs_rank: 2,
        rhs_rank: 2,
    };
    let mut tc = ta.dot_general(&tb, config);
    let mut engine = Engine::new(CpuBackend::new());
    let result = tc.eval(&mut engine).unwrap();
    // C = [[22, 49], [28, 64]] col-major: [22, 28, 49, 64]
    assert_eq!(get_f64_data(result), &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn test_broadcast_reduce() {
    let v = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);
    let tv = TracedTensor::from_tensor(v);
    let tb = tv.broadcast_in_dim(&[3, 2], &[0]);
    let mut tr = tb.reduce_sum(&[1]);
    let mut engine = Engine::new(CpuBackend::new());
    let result = tr.eval(&mut engine).unwrap();
    assert_eq!(get_f64_data(result), &[2.0, 4.0, 6.0]);
}

#[test]
fn test_transpose() {
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let ta = TracedTensor::from_tensor(a);
    let mut tb = ta.transpose(&[1, 0]);
    let mut engine = Engine::new(CpuBackend::new());
    let result = tb.eval(&mut engine).unwrap();
    assert_eq!(get_f64_data(result), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
}

#[test]
fn test_reshape() {
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let ta = TracedTensor::from_tensor(a);
    let mut tb = ta.reshape(&[6]);
    let mut engine = Engine::new(CpuBackend::new());
    let result = tb.eval(&mut engine).unwrap();
    assert_eq!(get_f64_data(result), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_eval_all() {
    let a = f64_tensor(vec![2], vec![1.0, 2.0]);
    let b = f64_tensor(vec![2], vec![3.0, 4.0]);
    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let mut sum = &ta + &tb;
    let mut prod = &ta * &tb;
    let mut engine = Engine::new(CpuBackend::new());
    let results = eval_all(&mut engine, &mut [&mut sum, &mut prod]).unwrap();
    assert_eq!(get_f64_data(&results[0]), &[4.0, 6.0]);
    assert_eq!(get_f64_data(&results[1]), &[3.0, 8.0]);
}

#[test]
fn test_chained_ops() {
    let a = f64_tensor(vec![2], vec![1.0, 2.0]);
    let b = f64_tensor(vec![2], vec![3.0, 4.0]);
    let c = f64_tensor(vec![2], vec![10.0, 20.0]);
    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let tc = TracedTensor::from_tensor(c);
    let sum = &ta + &tb;
    let mut result = &sum * &tc;
    let mut engine = Engine::new(CpuBackend::new());
    let out = result.eval(&mut engine).unwrap();
    assert_eq!(get_f64_data(out), &[40.0, 120.0]);
}
