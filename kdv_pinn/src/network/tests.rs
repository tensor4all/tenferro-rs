use super::*;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
use tenferro_tensor::Tensor;

#[test]
fn mlp_forward_shape() {
    let net = Mlp::new(&[2, 16, 16, 1]);
    let x = TracedTensor::input_concrete_shape(DType::F64, &[4, 2]);
    let y = net.forward(&x);
    assert_eq!(y.rank, 2);
    assert_eq!(y.try_concrete_shape(), Some(vec![4, 1]));

    let specs = net.input_specs();
    let bindings: Vec<(&TracedTensor, DType, &[usize])> = specs
        .iter()
        .map(|(p, dtype, shape)| (*p, *dtype, shape.as_slice()))
        .chain(std::iter::once((&x, DType::F64, &[4, 2][..])))
        .collect();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &bindings).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let x_tensor = Tensor::from_vec_col_major(vec![4, 2], vec![0.0_f64; 8]);
    let param_tensors: Vec<Tensor> = specs
        .iter()
        .map(|(_, _, shape)| {
            let len = shape.iter().product::<usize>();
            Tensor::from_vec_col_major(shape.clone(), vec![0.1_f64; len])
        })
        .collect();
    let mut bind_tensors: Vec<(&TracedTensor, &Tensor)> = net
        .parameters()
        .iter()
        .zip(param_tensors.iter())
        .map(|(p, t)| (*p, t))
        .collect();
    bind_tensors.push((&x, &x_tensor));

    let out = executor.run_with_inputs(&program, &bind_tensors).unwrap();
    assert_eq!(out.shape(), &[4, 1]);
}
