use tenferro_einsum::{
    parse_einsum_subscripts, ContractionTree, EinsumOptimize, Subscripts, TensorDotAxes,
    TracedEinsumOps, TracedTensordotOps,
};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{GraphCompiler, TraceContext};
use tenferro_tensor::{DType, Tensor};

use super::support;

fn spec(tensor: &Tensor) -> ProgramInputSpec {
    ProgramInputSpec::new(tensor.dtype(), DimExpr::from_concrete(tensor.shape()))
}

fn trace_default(trace: &TraceContext, tensor: &Tensor) -> tenferro_runtime::TracedTensor {
    let value = trace
        .input_with_default(spec(tensor), tensor.duplicate().unwrap())
        .unwrap();
    trace.derive_tensor(value).unwrap()
}

fn compile_and_run(
    trace: TraceContext,
    output: tenferro_runtime::TracedTensor,
) -> tenferro_runtime::Result<Tensor> {
    let graph = trace.finish(&[output.trace_value()]).unwrap();
    let compiled = GraphCompiler::new().compile_traced_graph(&graph)?;
    support::run_one(&compiled, &[])
}

#[test]
fn trace_context_einsum_ext_exposes_einsum() {
    let trace = TraceContext::new();
    let lhs = trace
        .derive_tensor(
            trace
                .input(ProgramInputSpec::new(
                    DType::F64,
                    [DimExpr::Const(2), DimExpr::Const(3)],
                ))
                .unwrap(),
        )
        .unwrap();
    let rhs = trace
        .derive_tensor(
            trace
                .input(ProgramInputSpec::new(
                    DType::F64,
                    [DimExpr::Const(3), DimExpr::Const(4)],
                ))
                .unwrap(),
        )
        .unwrap();

    let output = trace.einsum(&[&lhs, &rhs], "ij,jk->ik").unwrap();
    let graph = trace.finish(&[output.trace_value()]).unwrap();
    let metadata = graph
        .program()
        .value_metadata(graph.program().outputs()[0])
        .unwrap();

    assert_eq!(metadata.shape().len(), 2);
    assert_eq!(graph.program().operations().count(), 1);
}

#[test]
fn trace_context_reports_malformed_notation_without_mutation() {
    let trace = TraceContext::new();
    let input = trace
        .derive_tensor(
            trace
                .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
                .unwrap(),
        )
        .unwrap();

    assert!(trace.einsum(&[&input], "not einsum").is_err());
    let graph = trace.finish(&[input.trace_value()]).unwrap();
    assert_eq!(graph.program().operations().count(), 0);
}

#[test]
fn explicit_path_matmul_executes_numerically() {
    let trace = TraceContext::new();
    let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0]).unwrap();
    let lhs_value = trace_default(&trace, &lhs);
    let rhs_value = trace_default(&trace, &rhs);
    let output = trace
        .einsum_with(
            &[&lhs_value, &rhs_value],
            "ij,jk->ik",
            &tenferro_einsum::EinsumOptions::default().optimize(EinsumOptimize::Path(vec![(0, 1)])),
        )
        .unwrap();

    let result = compile_and_run(trace, output).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice::<f64>().unwrap(), &[19.0, 43.0, 22.0, 50.0]);
}

#[test]
fn concrete_precomputed_tree_executes_nary_chain() {
    let trace = TraceContext::new();
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]).unwrap();
    let c = Tensor::from_vec_col_major(vec![2, 2], vec![9.0_f64, 10.0, 11.0, 12.0]).unwrap();
    let parsed = parse_einsum_subscripts("ij,jk,kl->il").unwrap();
    let raw = Subscripts::from(&parsed);
    let shapes = [a.shape(), b.shape(), c.shape()];
    let tree = ContractionTree::from_pairs(&raw, &shapes, &[(0, 1), (3, 2)]).unwrap();
    let a_value = trace_default(&trace, &a);
    let b_value = trace_default(&trace, &b);
    let c_value = trace_default(&trace, &c);
    let output = trace
        .einsum_with(
            &[&a_value, &b_value, &c_value],
            "ij,jk,kl->il",
            &tenferro_einsum::EinsumOptions::default().optimize(EinsumOptimize::Tree(tree)),
        )
        .unwrap();

    let result = compile_and_run(trace, output).unwrap();
    assert_eq!(
        result.as_slice::<f64>().unwrap(),
        &[517.0, 766.0, 625.0, 926.0]
    );
}

#[test]
fn constant_shape_constraint_mismatch_fails_compilation() {
    let trace = TraceContext::new();
    let lhs = trace
        .derive_tensor(
            trace
                .input(ProgramInputSpec::new(
                    DType::F64,
                    [DimExpr::Const(2), DimExpr::Const(3)],
                ))
                .unwrap(),
        )
        .unwrap();
    let rhs = trace
        .derive_tensor(
            trace
                .input(ProgramInputSpec::new(
                    DType::F64,
                    [DimExpr::Const(4), DimExpr::Const(2)],
                ))
                .unwrap(),
        )
        .unwrap();
    let output = trace.einsum(&[&lhs, &rhs], "ij,jk->ik").unwrap();
    let graph = trace.finish(&[output.trace_value()]).unwrap();

    assert!(GraphCompiler::new().compile_traced_graph(&graph).is_err());
}

#[test]
fn traced_tensor_tensordot_compatibility_still_executes() {
    let trace = tenferro_runtime::TraceContext::new();
    let lhs = trace
        .from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])
        .unwrap();
    let rhs = trace
        .from_vec_col_major(vec![3, 2], vec![1.0_f64; 6])
        .unwrap();
    let output = lhs.tensordot(&rhs, TensorDotAxes::Count(1)).unwrap();
    let compiled = GraphCompiler::new().compile(&output).unwrap();
    let result = support::run_one(&compiled, &[]).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.as_slice::<f64>().unwrap(), &[3.0; 4]);
}
