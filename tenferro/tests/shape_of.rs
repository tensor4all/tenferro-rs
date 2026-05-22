mod support;
use support::{
    einsum, einsum_subscripts, einsum_subscripts_with, einsum_with, run_many_traced_with, RunTraced,
};
use tenferro::{CpuBackend, GraphExecutor, Tensor, TracedTensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data))
}

fn get_f64_scalar(tensor: &Tensor) -> f64 {
    match tensor {
        Tensor::F64(inner) => {
            assert_eq!(inner.shape.as_slice(), &[] as &[usize]);
            inner.host_data()[0]
        }
        other => panic!("expected F64 tensor, got {:?}", other.dtype()),
    }
}

#[test]
fn shape_of_returns_axis_size() {
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 5, 7], vec![0.0; 105]));
    let mut s0 = x.shape_of(0);
    let mut s1 = x.shape_of(1);
    let mut s2 = x.shape_of(2);

    assert_eq!(get_f64_scalar(&s0.run_with(&mut engine).unwrap()), 3.0);
    assert_eq!(get_f64_scalar(&s1.run_with(&mut engine).unwrap()), 5.0);
    assert_eq!(get_f64_scalar(&s2.run_with(&mut engine).unwrap()), 7.0);
}

#[test]
fn shape_of_grad_is_zero() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4, 3], vec![0.0; 12]));
    let s = x.shape_of(0);
    assert!(s.try_grad(&x).unwrap().is_none());
}
