use tenferro_cpu::CpuEngine;
use tenferro_einsum::prelude::*;

#[test]
fn prelude_calls_concrete_einsum() {
    let lhs = Tensor::from_vec_col_major([2, 3], vec![1.0_f64; 6]).unwrap();
    let rhs = Tensor::from_vec_col_major([3, 2], vec![1.0_f64; 6]).unwrap();
    let mut backend = CpuEngine::serial();
    let result = [&lhs, &rhs].einsum("ij,jk->ik", &mut backend).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
}

#[cfg(feature = "autodiff")]
#[test]
fn prelude_calls_eager_einsum() {
    let runtime = EagerRuntime::new().unwrap();
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major([2, 3], vec![1.0_f64; 6]).unwrap(),
        runtime.clone(),
    )
    .unwrap();
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major([3, 2], vec![1.0_f64; 6]).unwrap(),
        runtime,
    )
    .unwrap();
    let result = lhs
        .runtime()
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            s.einsum(&[&lhs, &rhs], "ij,jk->ik")
        })
        .unwrap();
    assert_eq!(result.shape(), &[2, 2]);
}
