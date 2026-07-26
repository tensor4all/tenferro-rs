use tenferro_einsum::{
    parse_einsum_subscripts, EinsumOptimize, TraceContextEinsumExt, EINSUM_EXTENSION_FAMILY_ID,
};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::program::{ProgramInputSpec, SemanticOpRef};
use tenferro_runtime::{GraphCompiler, TraceContext};
use tenferro_tensor::{DType, Tensor};

use super::support;

#[test]
fn trace_context_einsum_ext_emits_one_ordered_semantic_extension() {
    let matrix = || ProgramInputSpec::new(DType::F64, [DimExpr::Const(2), DimExpr::Const(2)]);
    let mut trace = TraceContext::new();
    let lhs = trace.input(matrix()).unwrap();
    let rhs = trace.input(matrix()).unwrap();

    let output = trace.einsum(&[lhs, rhs], "ij,jk->ik").unwrap();
    let graph = trace.finish(&[output]).unwrap();
    let operations = graph.program().operations().collect::<Vec<_>>();

    assert_eq!(operations.len(), 1);
    assert_eq!(operations[0].inputs(), graph.program().inputs());
    assert_eq!(operations[0].outputs(), graph.program().outputs());
    assert_eq!(
        graph
            .program()
            .value_metadata(graph.program().outputs()[0])
            .unwrap()
            .shape()
            .iter()
            .map(|extent| extent.as_exact())
            .collect::<Vec<_>>(),
        [Some(&DimExpr::Const(2)), Some(&DimExpr::Const(2))]
    );
}

#[test]
fn trace_context_einsum_ext_exposes_all_four_entry_points() {
    let parsed = parse_einsum_subscripts("ij,jk->ik").unwrap();
    let mut fingerprints = Vec::new();
    for entry in 0..4 {
        let matrix = || ProgramInputSpec::new(DType::F64, [DimExpr::Const(2), DimExpr::Const(2)]);
        let mut trace = TraceContext::new();
        let lhs = trace.input(matrix()).unwrap();
        let rhs = trace.input(matrix()).unwrap();
        let inputs = [lhs, rhs];
        let output = match entry {
            0 => trace.einsum(&inputs, "ij,jk->ik").unwrap(),
            1 => trace.einsum_subscripts(&inputs, &parsed).unwrap(),
            2 => trace
                .einsum_with(&inputs, "ij,jk->ik", EinsumOptimize::False)
                .unwrap(),
            3 => trace
                .einsum_subscripts_with(&inputs, &parsed, EinsumOptimize::Path(vec![(0, 1)]))
                .unwrap(),
            _ => unreachable!(),
        };
        let graph = trace.finish(&[output]).unwrap();
        fingerprints.push(graph.program().semantic_fingerprint());
    }

    assert_eq!(fingerprints[0], fingerprints[1]);
    assert_ne!(fingerprints[1], fingerprints[2]);
    assert_ne!(fingerprints[2], fingerprints[3]);
}

#[test]
fn trace_context_einsum_rejects_foreign_values_without_mutating_the_trace() {
    let vector = || ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]);
    let mut owner = TraceContext::new();
    let foreign = owner.input(vector()).unwrap();
    let mut trace = TraceContext::new();
    let local = trace.input(vector()).unwrap();

    let error = trace.einsum(&[local, foreign], "i,i->").unwrap_err();
    assert!(matches!(error, tenferro_einsum::Error::Runtime(_)));

    let graph = trace.finish(&[local]).unwrap();
    assert!(graph.program().operations().next().is_none());
}

#[test]
fn traced_semantic_einsum_compiles_and_executes_with_ordered_inputs() {
    let matrix = || ProgramInputSpec::new(DType::F64, [DimExpr::Const(2), DimExpr::Const(2)]);
    let mut trace = TraceContext::new();
    let lhs = trace.input(matrix()).unwrap();
    let rhs = trace.input(matrix()).unwrap();
    let output = trace.einsum(&[lhs, rhs], "ij,jk->ik").unwrap();
    let traced = trace.finish(&[output]).unwrap();
    let compiled = GraphCompiler::new().compile_traced_graph(&traced).unwrap();

    let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0]).unwrap();
    let result = support::run_one(&compiled, &[&lhs, &rhs]).unwrap();

    let operation = compiled.program().operations().next().unwrap();
    let SemanticOpRef::Extension(extension) = operation.op() else {
        panic!("expected semantic einsum extension");
    };
    assert_eq!(extension.family_id(), EINSUM_EXTENSION_FAMILY_ID);
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice::<f64>().unwrap(), &[19.0, 43.0, 22.0, 50.0]);
}
