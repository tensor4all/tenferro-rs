use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, GraphExecutor, Tensor, TracedTensor};

#[test]
fn traced_tensor_col_major_constructor_preserves_shape_and_storage() {
    let traced =
        TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0])
            .unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&traced).unwrap();
    assert_eq!(program.input_count(), 1);
    assert_eq!(program.input_specs()[0].shape(), &[2, 3]);
    let out = GraphExecutor::new(CpuBackend::new()).run(&program).unwrap();
    assert_eq!(
        out.as_slice::<f64>().unwrap(),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
    );
}

#[test]
fn traced_tensor_col_major_constructor_keeps_physical_order() {
    let traced = TracedTensor::from_vec_col_major(vec![2, 2], vec![1_i64, 3, 2, 4]).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&traced).unwrap();
    let out = GraphExecutor::new(CpuBackend::new()).run(&program).unwrap();
    assert_eq!(
        out.into_vec_col_major::<i64>().unwrap(),
        (vec![2, 2], vec![1, 3, 2, 4]),
    );
}

#[test]
fn tensor_col_major_constructor_is_the_facade_import_path() {
    let tensor = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]).unwrap();
    assert_eq!(tensor.as_slice::<f64>().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
}
