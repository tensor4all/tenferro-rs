#![cfg(feature = "autodiff")]

use tenferro_cpu::CpuBackend;
use tenferro_einsum::{EinsumOptimize, TraceContextEinsumExt};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::program::{ProgramInputSpec, SemanticFingerprint};
use tenferro_runtime::{ExtensionCacheSelector, GraphCompiler, GraphExecutor, TraceContext};
use tenferro_tensor::{DType, Tensor};

fn matrix() -> ProgramInputSpec {
    ProgramInputSpec::new(DType::F64, [DimExpr::Const(2), DimExpr::Const(2)])
}

#[test]
fn trace_context_parse_cache_reuses_exact_notation() {
    let mut trace = TraceContext::new();
    let lhs = trace.input(matrix()).unwrap();
    let rhs = trace.input(matrix()).unwrap();

    trace.einsum(&[lhs, rhs], "ij,jk->ik").unwrap();
    let first = trace
        .extension_caches_mut()
        .stats(ExtensionCacheSelector::All);
    trace.einsum(&[lhs, rhs], "ij,jk->ik").unwrap();
    let second = trace
        .extension_caches_mut()
        .stats(ExtensionCacheSelector::All);

    assert_eq!(first.entries, second.entries);
    assert!(first.entries >= 1);
}

#[test]
fn plan_policy_participates_in_semantic_fingerprint() {
    fn fingerprint(optimize: EinsumOptimize) -> SemanticFingerprint {
        let mut trace = TraceContext::new();
        let lhs = trace.input(matrix()).unwrap();
        let rhs = trace.input(matrix()).unwrap();
        let output = trace
            .einsum_with(&[lhs, rhs], "ij,jk->ik", optimize)
            .unwrap();
        trace
            .finish(&[output])
            .unwrap()
            .program()
            .semantic_fingerprint()
    }

    assert_ne!(
        fingerprint(EinsumOptimize::False),
        fingerprint(EinsumOptimize::Path(vec![(0, 1)]))
    );
}

#[test]
fn compiler_reuses_exact_semantic_staging() {
    let mut trace = TraceContext::new();
    let lhs = trace.input(matrix()).unwrap();
    let rhs = trace.input(matrix()).unwrap();
    let output = trace.einsum(&[lhs, rhs], "ij,jk->ik").unwrap();
    let graph = trace.finish(&[output]).unwrap();
    let mut compiler = GraphCompiler::new();

    compiler.compile_traced_graph(&graph).unwrap();
    let after_first = compiler.compile_cache_len();
    compiler.compile_traced_graph(&graph).unwrap();

    assert_eq!(compiler.compile_cache_len(), after_first);
    assert!(after_first >= 1);
}

#[test]
fn runtime_plan_cache_reuses_identical_shapes() {
    let mut trace = TraceContext::new();
    let lhs = trace.input(matrix()).unwrap();
    let rhs = trace.input(matrix()).unwrap();
    let output = trace.einsum(&[lhs, rhs], "ij,jk->ik").unwrap();
    let graph = trace.finish(&[output]).unwrap();
    let compiled = GraphCompiler::new().compile_traced_graph(&graph).unwrap();
    let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0]).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_einsum::register_runtime)
        .unwrap();

    executor.run_with_inputs(&compiled, &[&lhs, &rhs]).unwrap();
    let after_first = executor.cache_stats().extensions.entries;
    executor.run_with_inputs(&compiled, &[&lhs, &rhs]).unwrap();

    assert_eq!(executor.cache_stats().extensions.entries, after_first);
    assert!(after_first >= 1);
}
