use tenferro_cpu::CpuEngine;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::program::{CoreSemanticOp, ProgramBuildError, ProgramInputSpec};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext, TracedTensor};
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
    let left = TraceContext::new();
    let value = left
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let right = TraceContext::new();

    assert!(matches!(
        right.add_op(CoreSemanticOp::Neg, &[value]),
        Err(ProgramBuildError::ForeignValue)
    ));
    assert_eq!(format!("{value:?}"), "TraceValue(<opaque>)");
}

#[test]
fn trace_finish_preserves_ordered_inputs_defaults_and_duplicate_outputs() {
    let trace = TraceContext::new();
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
            .expect("a preset dtype always has the requested representation")
            .as_slice::<f64>()
            .unwrap(),
        lhs_tensor.as_slice::<f64>().unwrap()
    );
}

#[test]
fn traced_graph_compiles_and_executes_with_ordered_inputs() {
    let trace = TraceContext::new();
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
    let trace = TraceContext::new();
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

#[test]
fn single_input_stack_and_concatenate_compile() {
    // #2599: the dtype of a one-input `Concatenate` was read from input 1,
    // so these graphs failed at compile even though one input is valid.
    let trace = TraceContext::new();
    let vector = trace
        .from_vec_col_major(vec![3], vec![2.0_f64, 3.0, 4.0])
        .unwrap();

    let stacked = TracedTensor::stack(&[&vector], 0).unwrap();
    let concatenated = TracedTensor::concatenate(&[&vector], 0).unwrap();

    let mut compiler = GraphCompiler::new();
    let stacked = compiler
        .compile(&stacked)
        .expect("a single-input stack must compile");
    let concatenated = compiler
        .compile(&concatenated)
        .expect("a single-input concatenate must compile");

    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
    for program in [&stacked, &concatenated] {
        let prepared = runtime
            .prepare(program, &[], &[engine.target()])
            .expect("preparation must accept the compiled program");
        let mut outputs = runtime
            .run(&mut [&mut engine], &prepared, &[])
            .expect("the single-input graph must run");
        assert_eq!(outputs.len(), 1);
        let out = outputs.remove(0).into_tensor(&mut engine).unwrap();
        assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 3.0, 4.0]);
    }
}

#[test]
fn filled_constants_are_graph_constants_that_run_to_the_filled_value() {
    let trace = TraceContext::new();
    let zeros = trace.zeros(DType::I32, [2, 3]).unwrap();
    let ones = trace.ones(DType::Bool, [2]).unwrap();
    let full = trace.full(DType::F64, [3], -0.5).unwrap();
    let scalar = trace.full(DType::F32, [0_usize; 0], 2.0).unwrap();
    assert_eq!(zeros.concrete_shape().unwrap(), vec![2, 3]);
    assert_eq!(scalar.rank(), 0);

    // A constant, not a bound input: the program needs nothing from the host.
    let values = [&zeros, &ones, &full, &scalar].map(TracedTensor::trace_value);
    assert!(trace.finish(&values).unwrap().program().inputs().is_empty());

    let mut compiler = GraphCompiler::new();
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine);
    let mut run = |value: &TracedTensor| {
        let compiled = compiler.compile(value).unwrap();
        let prepared = runtime.prepare(&compiled, &[], &[engine.target()]).unwrap();
        let mut outputs = runtime.run(&mut [&mut engine], &prepared, &[]).unwrap();
        outputs.remove(0).into_tensor(&mut engine).unwrap()
    };
    assert_eq!(run(&zeros).as_slice::<i32>().unwrap(), &[0; 6]);
    assert_eq!(run(&ones).as_slice::<bool>().unwrap(), &[true, true]);
    assert_eq!(run(&full).as_slice::<f64>().unwrap(), &[-0.5; 3]);
    assert_eq!(run(&scalar).as_slice::<f32>().unwrap(), &[2.0]);

    assert!(trace.full(DType::I64, [2], 0.5).is_err());
}

#[test]
fn traced_full_shares_the_concrete_scalar_conversion_and_checks_the_shape() {
    let trace = TraceContext::new();

    // The same exact-integer conversion as `Tensor::full`: 2^53 is inside i64.
    let exact = trace.full(DType::I64, [1], 2.0_f64.powi(53)).unwrap();
    assert_eq!(exact.dtype(), DType::I64);
    assert_eq!(exact.concrete_shape().unwrap(), vec![1]);
    assert!(trace.full(DType::I64, [1], 2.0_f64.powi(63)).is_err());

    // The overflowing element count fails at construction, so nothing records a
    // graph that only fails once compiled.
    let error = trace
        .full(DType::F64, [usize::MAX, 2], 1.0)
        .expect_err("an overflowing fill shape must be rejected while tracing");
    assert!(
        matches!(
            error,
            tenferro_runtime::Error::TensorRuntime(tenferro_tensor::Error::Validation {
                source: tenferro_tensor::ValidationError::InvalidArgument {
                    argument: "shape",
                    ..
                },
                ..
            })
        ),
        "{error}"
    );

    // Scalar and empty shapes stay valid.
    assert_eq!(
        trace
            .full(DType::I32, [0_usize; 0], 7.0)
            .unwrap()
            .concrete_shape()
            .unwrap(),
        Vec::<usize>::new()
    );
    assert_eq!(
        trace
            .full(DType::I32, [2, 0], 7.0)
            .unwrap()
            .concrete_shape()
            .unwrap(),
        vec![2, 0]
    );
}
