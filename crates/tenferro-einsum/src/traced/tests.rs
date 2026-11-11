use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::program::{ProgramInputSpec, SemanticOpRef};
use tenferro_runtime::TraceContext;
use tenferro_tensor::DType;

use crate::{EinsumOptimize, EinsumOptions, TracedEinsumOps, EINSUM_EXTENSION_FAMILY_ID};

fn matrix(rows: usize, cols: usize) -> ProgramInputSpec {
    ProgramInputSpec::new(DType::F64, [DimExpr::Const(rows), DimExpr::Const(cols)])
}

#[test]
fn trace_context_nary_einsum_records_one_ordered_semantic_extension() {
    let trace = TraceContext::new();
    let a = trace
        .derive_tensor(trace.input(matrix(2, 3)).unwrap())
        .unwrap();
    let b = trace
        .derive_tensor(trace.input(matrix(3, 4)).unwrap())
        .unwrap();
    let c = trace
        .derive_tensor(trace.input(matrix(4, 5)).unwrap())
        .unwrap();

    let out = trace.einsum(&[&a, &b, &c], "ij,jk,kl->il").unwrap();
    let graph = trace.finish(&[out.trace_value()]).unwrap();
    let operations: Vec<_> = graph.program().operations().collect();

    assert_eq!(operations.len(), 1);
    assert_eq!(operations[0].inputs(), graph.program().inputs());
    let SemanticOpRef::Extension(op) = operations[0].op() else {
        panic!("einsum trace must remain a semantic extension");
    };
    assert_eq!(op.family_id(), EINSUM_EXTENSION_FAMILY_ID);
}

#[test]
fn trace_context_explicit_path_remains_semantic_until_compilation() {
    let trace = TraceContext::new();
    let a = trace
        .derive_tensor(trace.input(matrix(2, 3)).unwrap())
        .unwrap();
    let b = trace
        .derive_tensor(trace.input(matrix(3, 4)).unwrap())
        .unwrap();
    let c = trace
        .derive_tensor(trace.input(matrix(4, 5)).unwrap())
        .unwrap();

    let out = trace
        .einsum_with(
            &[&a, &b, &c],
            "ij,jk,kl->il",
            &EinsumOptions::default().optimize(EinsumOptimize::Path(vec![(0, 1), (0, 1)])),
        )
        .unwrap();
    let graph = trace.finish(&[out.trace_value()]).unwrap();

    assert!(matches!(
        graph.program().operations().next().unwrap().op(),
        SemanticOpRef::Extension(_)
    ));
}
