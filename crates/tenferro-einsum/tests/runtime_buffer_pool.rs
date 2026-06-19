use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, GraphExecutor, Tensor, TracedTensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn get_f64_data(tensor: &Tensor) -> &[f64] {
    match tensor {
        Tensor::F64(inner) => inner.host_data().unwrap(),
        _ => panic!("expected F64"),
    }
}

#[test]
fn cpu_backend_pool_reuses_nary_einsum_intermediates() {
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let c = f64_tensor(vec![2, 2], vec![9.0, 10.0, 11.0, 12.0]);

    let mut compiler = GraphCompiler::new();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    engine
        .register_extension(tenferro_einsum::register_runtime)
        .unwrap();

    let ta1 = TracedTensor::from_tensor_concrete_shape(a.clone()).unwrap();
    let tb1 = TracedTensor::from_tensor_concrete_shape(b.clone()).unwrap();
    let tc1 = TracedTensor::from_tensor_concrete_shape(c.clone()).unwrap();
    let out1 =
        tenferro_einsum::traced_tensor::einsum(&mut compiler, &[&ta1, &tb1, &tc1], "ij,jk,kl->il")
            .unwrap();
    let program1 = compiler.compile(&out1).unwrap();

    let result1 = engine.run(&program1).unwrap();
    assert_eq!(get_f64_data(&result1), &[517.0, 766.0, 625.0, 926.0]);

    let pooled_after_first = engine.backend().buffer_pool_len();
    assert!(pooled_after_first > 0);

    let ta2 = TracedTensor::from_tensor_concrete_shape(a).unwrap();
    let tb2 = TracedTensor::from_tensor_concrete_shape(b).unwrap();
    let tc2 = TracedTensor::from_tensor_concrete_shape(c).unwrap();
    let out2 =
        tenferro_einsum::traced_tensor::einsum(&mut compiler, &[&ta2, &tb2, &tc2], "ij,jk,kl->il")
            .unwrap();
    let program2 = compiler.compile(&out2).unwrap();

    let result2 = engine.run(&program2).unwrap();
    assert_eq!(get_f64_data(&result2), &[517.0, 766.0, 625.0, 926.0]);
    let pooled_after_second = engine.backend().buffer_pool_len();
    assert!(pooled_after_second < pooled_after_first * 2);
}
