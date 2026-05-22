use std::num::NonZeroUsize;

use tenferro::{DType, GraphCompiler, TracedTensor};

#[test]
fn graph_compiler_compiles_without_backend() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let y = &x + &x;

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();

    assert_eq!(program.input_count(), 1);
    assert_eq!(program.output_count(), 1);
    assert_eq!(compiler.compile_cache_len(), 1);
}

#[test]
fn graph_compiler_validates_placeholder_specs() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let y = &x + &x;

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[3])])
        .unwrap();

    assert_eq!(program.input_count(), 1);
    assert_eq!(program.input_specs()[0].shape(), &[3]);

    let err = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F32, &[3])])
        .unwrap_err();
    assert!(format!("{err}").contains("dtype"));
}

#[test]
fn graph_compiler_cache_is_bounded_and_reports_stats() {
    let mut compiler = GraphCompiler::new();
    compiler.set_compile_cache_capacity(NonZeroUsize::new(1).unwrap());

    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let _ = compiler.compile(&(&x + &x)).unwrap();
    let _ = compiler.compile(&x.neg()).unwrap();

    let stats = compiler.cache_stats();
    assert_eq!(compiler.compile_cache_capacity().get(), 1);
    assert_eq!(stats.compile.entries, 1);
    assert!(stats.compile.retained_bytes > 0);
}
