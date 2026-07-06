use tenferro_ad::TracedTensorAdExt;
use tenferro_cpu::CpuBackend;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
use tenferro_tensor::Tensor;

fn f64_vec(data: Vec<f64>) -> TracedTensor {
    TracedTensor::from_vec_col_major(vec![data.len()], data).unwrap()
}

fn eval_f64(tensor: &TracedTensor) -> Vec<f64> {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(tensor).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    match executor.run(&program).unwrap() {
        Tensor::F64(tensor) => tensor.host_data().unwrap().to_vec(),
        other => panic!("expected f64 result, got {other:?}"),
    }
}

fn op_count(tensor: &TracedTensor, op: StdTensorOp) -> usize {
    tensor
        .graph()
        .operations()
        .iter()
        .filter(|node| node.operation == op)
        .count()
}

#[test]
fn traced_jvp_zero_propagation_returns_none_without_dead_graph() {
    let x = f64_vec(vec![1.0, 2.0]);
    let y = f64_vec(vec![3.0, 4.0]);
    let tangent = f64_vec(vec![1.0, 1.0]);
    let output = y.neg();

    assert!(output.jvp_optional(&x, &tangent).unwrap().is_none());
}

#[test]
fn traced_jvp_optimizer_removes_identity_chain_before_compile() {
    let x = f64_vec(vec![2.0, -3.0]);
    let tangent = f64_vec(vec![0.25, -0.5]);
    let y = x.neg().neg();

    let dy = y.jvp(&x, &tangent).unwrap();
    assert_eq!(
        dy.graph().operations().len(),
        0,
        "double-neg JVP should alias the tangent input before compile"
    );

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&dy).unwrap();
    assert_eq!(program.lowering_view().instructions().count(), 0);
    assert_eq!(eval_f64(&dy), vec![0.25, -0.5]);
}

#[test]
fn traced_vjp_cotangent_accumulation_stays_bounded_before_compile() {
    let x = f64_vec(vec![2.0, -3.0]);
    let cotangent = f64_vec(vec![1.0, 1.0]);
    let pair = (&x + &x).unwrap();
    let y = (&pair + &pair).unwrap();

    let dx = y.vjp(&x, &cotangent).unwrap();
    let add_count = op_count(&dx, StdTensorOp::Add);
    assert!(
        (1..=3).contains(&add_count),
        "cotangent accumulation should stay linear for four x uses, got {add_count} Add ops"
    );
    assert_eq!(eval_f64(&dx), vec![4.0, 4.0]);

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&dx).unwrap();
    let instruction_count = program.lowering_view().instructions().count();
    assert!(
        instruction_count <= 3,
        "compiled accumulation graph should remain bounded, got {instruction_count} instructions"
    );
}
