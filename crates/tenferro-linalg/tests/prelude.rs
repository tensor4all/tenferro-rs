use tenferro_linalg::prelude::*;

#[test]
fn prelude_calls_concrete_linalg_operation() {
    let input = Tensor::from_vec_col_major([2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap();
    let mut backend = tenferro_cpu::CpuEngine::serial();
    let Svd {
        s: singular_values, ..
    } = input.svd(&mut backend).unwrap();
    assert_eq!(singular_values.shape(), &[2]);
}

#[cfg(feature = "autodiff")]
#[test]
fn prelude_calls_eager_linalg_operation() {
    let runtime = EagerRuntime::new().unwrap();
    let input = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major([2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap(),
        runtime.clone(),
    )
    .unwrap();
    let Svd {
        s: singular_values, ..
    } = runtime
        .with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |session| {
            session.svd(&input)
        })
        .unwrap();
    assert_eq!(singular_values.shape(), &[2]);
}
