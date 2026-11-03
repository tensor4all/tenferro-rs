use tenferro_ad::TracedTensorAdExt;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::{GraphCompiler, TraceContext, TracedTensor};

use crate::support::{cpu_runtime, run_compiled_one};

fn f64_vec(trace: &TraceContext, data: Vec<f64>) -> TracedTensor {
    trace.from_vec_col_major(vec![data.len()], data).unwrap()
}

fn eval_f64(tensor: &TracedTensor) -> Vec<f64> {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(tensor).unwrap();
    let executor = cpu_runtime();
    run_compiled_one(&executor, &program, &[])
        .unwrap()
        .as_typed::<f64>()
        .expect("expected f64 result")
        .host_data()
        .unwrap()
        .to_vec()
}

fn op_count(tensor: &TracedTensor, op: StdTensorOp) -> usize {
    frozen_operations(tensor)
        .iter()
        .filter(|operation| operation == &&op)
        .count()
}

fn frozen_operations(tensor: &TracedTensor) -> Vec<StdTensorOp> {
    tensor
        .context()
        .frozen_snapshot(&[tensor.trace_value()])
        .unwrap()
        .program
        .operations()
        .map(|operation| match operation.op() {
            tenferro_runtime::program::SemanticOpRef::Core(op) => StdTensorOp::from(op),
            tenferro_runtime::program::SemanticOpRef::Extension(op) => {
                StdTensorOp::Extension(op.clone_arc())
            }
            _ => StdTensorOp::Neg,
        })
        .collect()
}

#[test]
fn traced_jvp_zero_propagation_returns_none_without_dead_graph() {
    let trace = TraceContext::new();
    let x = f64_vec(&trace, vec![1.0, 2.0]);
    let y = f64_vec(&trace, vec![3.0, 4.0]);
    let tangent = f64_vec(&trace, vec![1.0, 1.0]);
    let output = y.neg().unwrap();

    assert!(output.jvp_optional(&x, &tangent).unwrap().is_none());
}

#[test]
fn traced_jvp_optimizer_removes_identity_chain_before_compile() {
    let trace = TraceContext::new();
    let x = f64_vec(&trace, vec![2.0, -3.0]);
    let tangent = f64_vec(&trace, vec![0.25, -0.5]);
    let y = x.neg().unwrap().neg().unwrap();

    let dy = y.jvp(&x, &tangent).unwrap();
    assert_eq!(
        frozen_operations(&dy).len(),
        0,
        "double-neg JVP should alias the tangent input before compile"
    );

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&dy).unwrap();
    assert_eq!(program.program().operations().count(), 0);
    assert_eq!(eval_f64(&dy), vec![0.25, -0.5]);
}

#[test]
fn traced_vjp_cotangent_accumulation_stays_bounded_before_compile() {
    let trace = TraceContext::new();
    let x = f64_vec(&trace, vec![2.0, -3.0]);
    let cotangent = f64_vec(&trace, vec![1.0, 1.0]);
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
    let instruction_count = program.program().operations().count();
    assert!(
        instruction_count <= 3,
        "compiled accumulation graph should remain bounded, got {instruction_count} instructions"
    );
}
