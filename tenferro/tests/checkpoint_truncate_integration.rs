mod support;
use support::{
    einsum, einsum_subscripts, einsum_subscripts_with, einsum_with, run_many_traced_with, RunTraced,
};
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, Tensor, TracedTensor, TypedTensor};

const TOL: f64 = 1.0e-4;
const FD_H: f64 = 1.0e-5;

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data))
}

fn f64_scalar(value: f64) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![value]))
}

fn get_f64_scalar(tensor: &Tensor) -> f64 {
    match tensor {
        Tensor::F64(inner) => inner.host_data()[0],
        other => panic!("expected F64 tensor, got {:?}", other.dtype()),
    }
}

#[test]
fn checkpoint_truncate_loop_grad() {
    let steps = 3;
    let a_value = 0.5_f64;
    let x0_data = vec![1.0, 2.0, 3.0];
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let mut compiler = GraphCompiler::new();
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let a = TracedTensor::from_tensor_concrete_shape(f64_scalar(a_value));
    let size = TracedTensor::from_tensor_concrete_shape(f64_scalar(2.0));
    let mut x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x0_data.clone()));

    for _ in 0..steps {
        x = &a * &x;
        x = x.dynamic_truncate(&size, 0);
        x.checkpoint(&mut compiler, &mut executor).unwrap();
    }

    let loss = x.reduce_sum(&[0]);
    let mut grad = loss.grad(&a).unwrap();
    let grad_value = get_f64_scalar(&grad.run_with(&mut engine).unwrap());

    let f_concrete = |a_value: f64| {
        let mut x = x0_data.clone();
        for _ in 0..steps {
            x = x.into_iter().map(|value| a_value * value).collect();
            x.truncate(2);
        }
        x.into_iter().sum::<f64>()
    };
    let fd = (f_concrete(a_value + FD_H) - f_concrete(a_value - FD_H)) / (2.0 * FD_H);

    assert!(
        (grad_value - fd).abs() < TOL,
        "integration: grad={grad_value}, fd={fd}"
    );
}
