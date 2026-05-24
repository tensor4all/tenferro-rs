use tenferro::{CpuBackend, DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};

#[test]
fn svd_executes_after_runtime_registration() {
    let a = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0_f64, 0.0, 0.0, 2.0],
    ));
    let (u, s, vt) = tenferro_linalg::svd(&a);

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&[&u, &s, &vt]).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_linalg::register_runtime)
        .unwrap();
    let outputs = executor.run_many(&program).unwrap();

    assert_eq!(outputs.len(), 3);
    assert_eq!(outputs[0].shape(), &[2, 2]);
    assert_eq!(outputs[1].shape(), &[2]);
    assert_eq!(outputs[2].shape(), &[2, 2]);
}

#[test]
fn missing_runtime_reports_linalg_family() {
    let a = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0_f64, 0.0, 0.0, 1.0],
    ));
    let y = tenferro_linalg::cholesky(&a);

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let err = GraphExecutor::new(CpuBackend::new())
        .run(&program)
        .unwrap_err();

    assert!(err
        .to_string()
        .contains(tenferro_linalg::LINALG_EXTENSION_FAMILY_ID));
}

#[test]
fn full_piv_lu_multi_output_slots_are_preserved() {
    let a = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec_col_major(
        vec![2, 2],
        vec![0.0_f64, 2.0, 1.0, 3.0],
    ));
    let (p, l, u, q, parity) = tenferro_linalg::full_piv_lu(&a);

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&[&p, &l, &u, &q, &parity]).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_linalg::register_runtime)
        .unwrap();
    let outputs = executor.run_many(&program).unwrap();

    assert_eq!(outputs[0].shape(), &[2, 2]);
    assert_eq!(outputs[1].shape(), &[2, 2]);
    assert_eq!(outputs[2].shape(), &[2, 2]);
    assert_eq!(outputs[3].shape(), &[2, 2]);
    assert_eq!(outputs[4].shape(), &[] as &[usize]);
}

#[test]
fn traced_metadata_matches_linalg_extension_shapes_and_dtypes() {
    let rectangular = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec_col_major(
        vec![3, 4, 2],
        vec![1.0_f64; 24],
    ));
    let square = TracedTensor::from_tensor_concrete_shape(Tensor::from_vec_col_major(
        vec![3, 3, 2],
        vec![1.0_f64; 18],
    ));
    let ints = TracedTensor::from_tensor_concrete_shape(Tensor::I64(
        tenferro::TypedTensor::from_vec_col_major(vec![2, 2], vec![1, 0, 0, 2]),
    ));

    let (u, s, vt) = tenferro_linalg::svd(&rectangular);
    assert_eq!(u.concrete_shape(), vec![3, 3, 2]);
    assert_eq!(s.concrete_shape(), vec![3, 2]);
    assert_eq!(vt.concrete_shape(), vec![3, 4, 2]);

    let (q, r) = tenferro_linalg::qr(&rectangular);
    assert_eq!(q.concrete_shape(), vec![3, 3, 2]);
    assert_eq!(r.concrete_shape(), vec![3, 4, 2]);

    let (values, vectors) = tenferro_linalg::eig(&ints);
    assert_eq!(values.dtype, DType::C64);
    assert_eq!(vectors.dtype, DType::C64);

    let (eigh_values, eigh_vectors) = tenferro_linalg::eigh(&square);
    assert_eq!(eigh_values.concrete_shape(), vec![3, 2]);
    assert_eq!(eigh_vectors.concrete_shape(), vec![3, 3, 2]);
}
