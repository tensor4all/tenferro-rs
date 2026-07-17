use crate::support;
use support::RunTraced;
use tenferro_ad::TracedTensorAdExt;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphExecutor, Tensor, TracedTensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn get_f64_scalar(tensor: &Tensor) -> f64 {
    match tensor {
        Tensor::F64(inner) => {
            assert_eq!(inner.shape(), &[] as &[usize]);
            inner.host_data().unwrap()[0]
        }
        other => panic!("expected F64 tensor, got {:?}", other.dtype()),
    }
}

#[test]
fn shape_of_returns_axis_size() {
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 5, 7], vec![0.0; 105]))
        .unwrap();
    let s0 = x.shape_of(0).unwrap();
    let s1 = x.shape_of(1).unwrap();
    let s2 = x.shape_of(2).unwrap();

    assert_eq!(get_f64_scalar(&s0.run_with(&mut engine).unwrap()), 3.0);
    assert_eq!(get_f64_scalar(&s1.run_with(&mut engine).unwrap()), 5.0);
    assert_eq!(get_f64_scalar(&s2.run_with(&mut engine).unwrap()), 7.0);
}

#[test]
fn shape_of_grad_is_zero() {
    let x =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4, 3], vec![0.0; 12])).unwrap();
    let s = x.shape_of(0).unwrap();
    assert!(s.grad_optional(&x).unwrap().is_none());
}
