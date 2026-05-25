use tenferro::{traced_tensor, CompareDir, CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

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

#[test]
fn traced_tensor_module_exposes_initial_elementwise_free_functions() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]);
    let y = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]);
    let cond = traced_tensor::compare(&x, &y, CompareDir::Gt);

    let _ = traced_tensor::sub(&x, &y);
    let _ = traced_tensor::mul(&x, &y);
    let _ = traced_tensor::div(&x, &y);
    let _ = traced_tensor::pow(&x, &y);
    let _ = traced_tensor::maximum(&x, &y);
    let _ = traced_tensor::minimum(&x, &y);
    let _ = traced_tensor::where_select(&cond, &x, &y);
    let _ = traced_tensor::clamp(&x, &y, &x);
    let _ = traced_tensor::neg(&x);
    let _ = traced_tensor::abs(&x);
    let _ = traced_tensor::sign(&x);
    let _ = traced_tensor::conj(&x);
    let _ = traced_tensor::exp(&x);
    let _ = traced_tensor::log(&x);
    let _ = traced_tensor::sin(&x);
    let _ = traced_tensor::cos(&x);
    let _ = traced_tensor::tanh(&x);
    let _ = traced_tensor::sqrt(&x);
    let _ = traced_tensor::rsqrt(&x);
    let _ = traced_tensor::expm1(&x);
    let _ = traced_tensor::log1p(&x);
}
