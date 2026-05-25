use tenferro::{traced_tensor, CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

#[test]
fn traced_add_uses_numpy_broadcasting_for_rank_padding_and_singletons() {
    let lhs = TracedTensor::from_vec_row_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0]);
    let rhs = TracedTensor::from_vec_row_major(vec![1, 4], vec![10.0_f64, 20.0, 30.0, 40.0]);
    let y = traced_tensor::add(&lhs, &rhs);

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let out = executor.run(&program).unwrap();

    assert_eq!(out.shape(), &[3, 4]);
    assert_eq!(
        out.try_into_vec_row_major::<f64>().unwrap().1,
        vec![11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0,]
    );
}
