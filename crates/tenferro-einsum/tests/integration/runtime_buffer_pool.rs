use tenferro_cpu::CpuBackend;
use tenferro_einsum::TraceContextEinsumExt;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{GraphCompiler, Tensor, TraceContext, TypedTensor};

use super::support;

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn get_f64_data(tensor: &Tensor) -> &[f64] {
    tensor
        .as_typed::<f64>()
        .expect("expected F64")
        .host_data()
        .unwrap()
}

fn traced_input(trace: &mut TraceContext, tensor: &Tensor) -> tenferro_runtime::TraceValue {
    trace
        .input_with_default(
            ProgramInputSpec::new(tensor.dtype(), DimExpr::from_concrete(tensor.shape())),
            tensor.duplicate().unwrap(),
        )
        .unwrap()
}

fn compile_nary(
    compiler: &mut GraphCompiler,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
) -> tenferro_runtime::CompiledGraph {
    let mut trace = TraceContext::new();
    let a = traced_input(&mut trace, a);
    let b = traced_input(&mut trace, b);
    let c = traced_input(&mut trace, c);
    let output = trace.einsum(&[a, b, c], "ij,jk,kl->il").unwrap();
    let graph = trace.finish(&[output]).unwrap();
    compiler.compile_traced_graph(&graph).unwrap()
}

#[test]
fn cpu_backend_nary_einsum_leaves_no_host_pool_growth() {
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let c = f64_tensor(vec![2, 2], vec![9.0, 10.0, 11.0, 12.0]);

    let mut compiler = GraphCompiler::new();
    let backend = CpuBackend::new();
    let runtime = support::cpu_runtime_with_einsum(&backend).unwrap();

    let program1 = compile_nary(&mut compiler, &a, &b, &c);

    let mut outputs1 = runtime.run_compiled(&program1, &[]).unwrap();
    assert_eq!(outputs1.len(), 1);
    let result1 = outputs1.remove(0);
    assert_eq!(get_f64_data(&result1), &[517.0, 766.0, 625.0, 926.0]);

    // The N-ary intermediates are owned by the lower library's execution
    // workspace, so the host buffer pool retains no per-run growth.
    let retained_after_first = backend.buffer_pool_stats().unwrap().capacity_bytes;

    let program2 = compile_nary(&mut compiler, &a, &b, &c);

    let mut outputs2 = runtime.run_compiled(&program2, &[]).unwrap();
    assert_eq!(outputs2.len(), 1);
    let result2 = outputs2.remove(0);
    assert_eq!(get_f64_data(&result2), &[517.0, 766.0, 625.0, 926.0]);
    assert_eq!(
        backend.buffer_pool_stats().unwrap().capacity_bytes,
        retained_after_first,
        "repeated N-ary runs must not grow the host buffer pool"
    );
}
