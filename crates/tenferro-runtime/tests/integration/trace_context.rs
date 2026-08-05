use tenferro_cpu::CpuBackend;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::program::{CoreSemanticOp, ProgramBuildError, ProgramInputSpec};
use tenferro_runtime::{GraphCompiler, Runtime, TraceContext};
use tenferro_tensor::{DType, Tensor};

fn cpu_runtime() -> Runtime {
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).unwrap())
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
    let mut result = cpu_runtime()
        .run_compiled(&compiled, &[&lhs, &rhs])
        .unwrap();

    assert_eq!(compiled.program().inputs().len(), 2);
    assert_eq!(compiled.program().outputs().len(), 1);
    assert_eq!(
        result.pop().unwrap().as_slice::<f64>().unwrap(),
        &[4.0, 6.0]
    );
}
