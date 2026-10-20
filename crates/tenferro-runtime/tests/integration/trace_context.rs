use tenferro_cpu::CpuEngine;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::program::{CoreSemanticOp, ProgramBuildError, ProgramInputSpec};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext};
use tenferro_tensor::{DType, Tensor, TensorRead};

fn cpu_runtime(engine: &CpuEngine) -> Runtime {
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(engine).unwrap())
        .unwrap();
    builder.build().unwrap()
}

#[test]
fn trace_values_are_context_owned_and_opaque() {
    let mut left = TraceContext::new();
    let value = left
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let mut right = TraceContext::new();

    assert!(matches!(
        right.add_op(CoreSemanticOp::Neg, &[value]),
        Err(ProgramBuildError::ForeignValue)
    ));
    assert_eq!(format!("{value:?}"), "TraceValue(<opaque>)");
}

#[test]
fn trace_finish_preserves_ordered_inputs_defaults_and_duplicate_outputs() {
    let mut trace = TraceContext::new();
    let lhs_tensor =
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).expect("test tensor must be valid");
    let lhs = trace
        .input_with_default(
            ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]),
            lhs_tensor.duplicate().unwrap(),
        )
        .unwrap();
    let rhs = trace
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let sum = trace.add_op(CoreSemanticOp::Add, &[lhs, rhs]).unwrap()[0];

    let graph = trace.finish(&[sum, sum]).unwrap();

    assert_eq!(graph.program().inputs().len(), 2);
    assert_eq!(graph.program().outputs().len(), 2);
    assert_eq!(graph.bindings().len(), 1);
    let frozen_default = graph.bindings().iter().next().unwrap().1;
    assert_eq!(frozen_default.shape(), lhs_tensor.shape());
    assert_eq!(
        frozen_default
            .tensor_read()
            .unwrap()
            .tensor_view()
            .as_slice::<f64>()
            .unwrap(),
        lhs_tensor.as_slice::<f64>().unwrap()
    );
}

#[test]
fn traced_graph_compiles_and_executes_with_ordered_inputs() {
    let mut trace = TraceContext::new();
    let lhs = trace
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let rhs = trace
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let sum = trace.add_op(CoreSemanticOp::Add, &[lhs, rhs]).unwrap()[0];
    let graph = trace.finish(&[sum]).unwrap();

    let compiled = GraphCompiler::new().compile_traced_graph(&graph).unwrap();
    let lhs = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
    let reads = [TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)];
    let prepared = runtime
        .prepare(&compiled, &reads, &[engine.target()])
        .unwrap();
    let mut result = runtime
        .run(&mut [&mut engine], &prepared, &reads)
        .map(|outputs| {
            outputs
                .into_iter()
                .map(|output| output.into_tensor(&mut engine).unwrap())
                .collect::<Vec<_>>()
        })
        .unwrap();

    assert_eq!(compiled.program().inputs().len(), 2);
    assert_eq!(compiled.program().outputs().len(), 1);
    assert_eq!(
        result.pop().unwrap().as_slice::<f64>().unwrap(),
        &[4.0, 6.0]
    );
}

#[test]
fn duplicate_finish_outputs_execute_in_order_as_independent_results() {
    let mut trace = TraceContext::new();
    let lhs = trace
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let rhs = trace
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let sum = trace.add_op(CoreSemanticOp::Add, &[lhs, rhs]).unwrap()[0];
    let neg = trace.add_op(CoreSemanticOp::Neg, &[sum]).unwrap()[0];
    let graph = trace.finish(&[sum, neg, sum]).unwrap();

    let compiled = GraphCompiler::new().compile_traced_graph(&graph).unwrap();
    let lhs = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
    let reads = [TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)];
    let prepared = runtime
        .prepare(&compiled, &reads, &[engine.target()])
        .unwrap();
    let outputs = runtime
        .run(&mut [&mut engine], &prepared, &reads)
        .expect("a duplicated computed output is not a missing value");
    assert_eq!(outputs.len(), 3);

    let outputs: Vec<_> = outputs
        .into_iter()
        .map(|output| output.into_tensor(&mut engine).unwrap())
        .collect();
    assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[4.0, 6.0]);
    assert_eq!(outputs[1].as_slice::<f64>().unwrap(), &[-4.0, -6.0]);
    assert_eq!(outputs[2].as_slice::<f64>().unwrap(), &[4.0, 6.0]);
    assert_ne!(
        outputs[0].as_slice::<f64>().unwrap().as_ptr(),
        outputs[2].as_slice::<f64>().unwrap().as_ptr(),
        "repeated owned outputs are independent copies"
    );
}
