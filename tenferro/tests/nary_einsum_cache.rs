//! Tests for traced einsum caches owned by graph compiler and executor.

use std::num::NonZeroUsize;

use tenferro::traced_tensor::einsum;
use tenferro::{CpuBackend, DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};

fn build_static_matmul(compiler: &mut GraphCompiler, rows: usize, cols: usize, mid: usize) {
    let a = TracedTensor::from_vec_col_major(
        vec![rows, mid],
        (0..rows * mid).map(|i| i as f64).collect::<Vec<_>>(),
    );
    let b = TracedTensor::from_vec_col_major(
        vec![mid, cols],
        (0..mid * cols).map(|i| i as f64).collect::<Vec<_>>(),
    );
    let _ = einsum(compiler, &[&a, &b], "ij,jk->ik").expect("einsum");
}

fn run_symbolic_matmul(
    executor: &mut GraphExecutor<CpuBackend>,
    rows: usize,
    cols: usize,
    mid: usize,
) {
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let mut compiler = GraphCompiler::new();
    let c = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").expect("einsum");
    let program = compiler
        .compile_with_input_specs(
            &c,
            &[
                (&a, DType::F64, &[rows, mid]),
                (&b, DType::F64, &[mid, cols]),
            ],
        )
        .expect("compile");

    let lhs = Tensor::from_vec_col_major(
        vec![rows, mid],
        (0..rows * mid).map(|i| i as f64).collect::<Vec<_>>(),
    );
    let rhs = Tensor::from_vec_col_major(
        vec![mid, cols],
        (0..mid * cols).map(|i| i as f64).collect::<Vec<_>>(),
    );
    executor
        .run_with_inputs(&program, &[(&a, &lhs), (&b, &rhs)])
        .expect("eval");
}

#[test]
fn compiler_and_executor_default_einsum_capacities_are_nonzero() {
    let compiler = GraphCompiler::new();
    assert!(compiler.einsum_cache_capacity().get() > 0);

    let executor = GraphExecutor::new(CpuBackend::new());
    assert!(executor.einsum_cache_capacity().get() > 0);
}

#[test]
fn compiler_with_einsum_cache_capacity_sets_static_cache_capacity() {
    let cap = NonZeroUsize::new(4).unwrap();
    let compiler = GraphCompiler::with_einsum_cache_capacity(cap);
    assert_eq!(compiler.einsum_cache_capacity(), cap);
}

#[test]
fn executor_with_einsum_cache_capacity_sets_runtime_cache_capacity() {
    let cap = NonZeroUsize::new(4).unwrap();
    let executor = GraphExecutor::with_einsum_cache_capacity(CpuBackend::new(), cap);
    assert_eq!(executor.einsum_cache_capacity(), cap);
}

#[test]
fn compiler_set_einsum_cache_capacity_shrinks_static_cache_len() {
    let mut compiler = GraphCompiler::with_einsum_cache_capacity(NonZeroUsize::new(10).unwrap());

    for k in 1..=5 {
        build_static_matmul(&mut compiler, 2, 2, k);
    }

    assert_eq!(compiler.einsum_cache_len(), 5);
    compiler.set_einsum_cache_capacity(NonZeroUsize::new(3).unwrap());
    assert_eq!(compiler.einsum_cache_len(), 3);
    assert_eq!(
        compiler.einsum_cache_capacity(),
        NonZeroUsize::new(3).unwrap()
    );
}

#[test]
fn executor_set_einsum_cache_capacity_shrinks_runtime_cache_len() {
    let mut executor = GraphExecutor::with_einsum_cache_capacity(
        CpuBackend::new(),
        NonZeroUsize::new(10).unwrap(),
    );

    for k in 1..=5 {
        run_symbolic_matmul(&mut executor, 2, 2, k);
    }

    assert_eq!(executor.einsum_cache_len(), 5);
    executor.set_einsum_cache_capacity(NonZeroUsize::new(3).unwrap());
    assert_eq!(executor.einsum_cache_len(), 3);
    assert_eq!(
        executor.einsum_cache_capacity(),
        NonZeroUsize::new(3).unwrap()
    );
}

#[test]
fn concrete_traced_einsum_reuses_compiler_static_and_parse_caches() {
    let mut compiler = GraphCompiler::with_einsum_cache_capacity(NonZeroUsize::new(2).unwrap());

    build_static_matmul(&mut compiler, 2, 2, 3);
    let after_first = compiler.cache_stats();
    assert_eq!(after_first.static_einsum_plans.entries, 1);
    assert_eq!(after_first.einsum_parse.entries, 1);

    build_static_matmul(&mut compiler, 2, 2, 3);
    let after_second = compiler.cache_stats();
    assert_eq!(after_second.static_einsum_plans.entries, 1);
    assert_eq!(after_second.einsum_parse.entries, 1);

    build_static_matmul(&mut compiler, 2, 2, 4);
    build_static_matmul(&mut compiler, 2, 2, 5);
    assert_eq!(compiler.einsum_cache_len(), 2);
}

#[test]
fn symbolic_nary_einsum_reuses_executor_runtime_cache() {
    let mut executor = GraphExecutor::new(CpuBackend::new());

    run_symbolic_matmul(&mut executor, 2, 4, 3);
    let len_after_first = executor.einsum_cache_len();
    assert_eq!(
        len_after_first, 1,
        "expected one cache entry after first eval"
    );

    run_symbolic_matmul(&mut executor, 2, 4, 3);
    let len_after_second = executor.einsum_cache_len();
    assert_eq!(
        len_after_second, 1,
        "cache len must stay at 1 on repeated identical (subscripts, shapes)"
    );
}
