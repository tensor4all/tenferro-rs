use tenferro_ops::{dim_expr::DimExpr, shape_extent::ShapeExtent};
use tenferro_runtime::{CompilerOptions, OptimizerConfig};
use tenferro_runtime::{DType, GraphCompiler, TracedTensor};

#[test]
fn graph_compiler_compiles_without_backend() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y = (&x + &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();

    assert_eq!(program.input_count(), 1);
    assert_eq!(program.output_count(), 1);
}

#[test]
fn graph_compiler_default_inputs_use_concrete_extent_identity() {
    let x2 = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y2 = (&x2 + &x2).unwrap();
    let x3 = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let y3 = (&x3 + &x3).unwrap();

    let mut compiler = GraphCompiler::new();
    let program2 = compiler.compile(&y2).unwrap();
    let program3 = compiler.compile(&y3).unwrap();

    assert_ne!(
        program2.program().semantic_fingerprint(),
        program3.program().semantic_fingerprint()
    );
    assert!(!program2.program().semantic_eq(program3.program()));

    let input = program2.program().inputs()[0];
    let metadata = program2.program().value_metadata(input).unwrap();
    assert_eq!(metadata.shape(), &[ShapeExtent::Exact(DimExpr::Const(2))]);
    let input = program3.program().inputs()[0];
    let metadata = program3.program().value_metadata(input).unwrap();
    assert_eq!(metadata.shape(), &[ShapeExtent::Exact(DimExpr::Const(3))]);
    assert_eq!(program2.bindings().iter().next().unwrap().1.shape(), &[2]);
    assert_eq!(program3.bindings().iter().next().unwrap().1.shape(), &[3]);
}

#[test]
fn graph_compiler_validates_placeholder_specs() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let y = (&x + &x).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[3])])
        .unwrap();

    assert_eq!(program.input_count(), 1);
    let input = program.program().inputs()[0];
    let metadata = program.program().value_metadata(input).unwrap();
    assert_eq!(metadata.shape(), &[ShapeExtent::Exact(DimExpr::Const(3))]);

    let err = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F32, &[3])])
        .unwrap_err();
    assert!(format!("{err}").contains("dtype"));

    let z = TracedTensor::input_concrete_shape(DType::F64, &[3]).unwrap();
    let err = compiler
        .compile_with_input_specs(&z.neg().unwrap(), &[(&z, DType::F64, &[2])])
        .unwrap_err();
    assert!(format!("{err}").contains("shape"));
}

#[test]
fn graph_program_input_accessors_report_compiled_contract() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let y = x.neg().unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[4])])
        .unwrap();
    let program = std::hint::black_box(program);
    let input = std::hint::black_box(program.program().inputs()[0]);
    let metadata = program.program().value_metadata(input).unwrap();

    assert_eq!(std::hint::black_box(&program).input_count(), 1);
    assert_eq!(std::hint::black_box(&program).output_count(), 1);
    assert_eq!(metadata.dtype(), DType::F64);
    assert_eq!(metadata.shape(), &[ShapeExtent::Exact(DimExpr::Const(4))]);
}

#[test]
fn graph_compiler_compiler_options_setter_updates_options() {
    let mut compiler = GraphCompiler::new();
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let _ = compiler.compile(&x.neg().unwrap()).unwrap();

    let options = CompilerOptions {
        optimizer: OptimizerConfig {
            dot_decomposer: true,
            ..OptimizerConfig::default()
        },
    };
    compiler.set_compiler_options(options);

    assert_eq!(compiler.compiler_options(), options);
}

#[test]
fn graph_compiler_compile_many_returns_multi_output_program() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y = (&x + &x).unwrap();
    let z = x.neg().unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&[&y, &z]).unwrap();

    assert_eq!(program.input_count(), 1);
    assert_eq!(program.output_count(), 2);
}
