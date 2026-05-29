use tenferro_cpu::CpuBackend;
use tenferro_runtime::{tensor, DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};

#[test]
fn runtime_crate_exposes_traced_graph_execution_api() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let y = &x + &x;

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let out = GraphExecutor::new(CpuBackend::default())
        .run(&program)
        .unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
}

#[test]
fn tensor_module_free_functions_cover_eager_runtime_paths() {
    let mut backend = CpuBackend::new();
    let input = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);

    let converted = tensor::convert(&input, DType::F32, &mut backend).unwrap();
    assert_eq!(converted.dtype(), DType::F32);
    assert_eq!(converted.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

    let reshaped = tensor::reshape(&input, &[4], &mut backend).unwrap();
    assert_eq!(reshaped.shape(), &[4]);
    assert_eq!(reshaped.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

    let transposed = tensor::transpose(&input, &[1, 0], &mut backend).unwrap();
    assert_eq!(transposed.shape(), &[2, 2]);
    assert_eq!(transposed.as_slice::<f64>().unwrap(), &[1.0, 3.0, 2.0, 4.0]);

    let summed = tensor::reduce_sum(&input, &[0], &mut backend).unwrap();
    assert_eq!(summed.shape(), &[2]);
    assert_eq!(summed.as_slice::<f64>().unwrap(), &[3.0, 7.0]);
}
