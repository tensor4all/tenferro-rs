use tenferro_cpu::CpuBackend;
use tenferro_einsum::{EinsumOptimize, TraceContextEinsumExt};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{GraphCompiler, GraphExecutor, TraceContext};
use tenferro_tensor::{DType, Tensor};

#[test]
fn ordered_inputs_enforce_symbolic_contracted_dimension_equality() {
    let mut trace = TraceContext::new();
    let lhs = trace
        .input(ProgramInputSpec::new(
            DType::F64,
            [
                DimExpr::InputDim {
                    input_idx: 0,
                    axis: 0,
                },
                DimExpr::InputDim {
                    input_idx: 0,
                    axis: 1,
                },
            ],
        ))
        .unwrap();
    let rhs = trace
        .input(ProgramInputSpec::new(
            DType::F64,
            [
                DimExpr::InputDim {
                    input_idx: 1,
                    axis: 0,
                },
                DimExpr::InputDim {
                    input_idx: 1,
                    axis: 1,
                },
            ],
        ))
        .unwrap();
    let output = trace.einsum(&[lhs, rhs], "ij,jk->ik").unwrap();
    let graph = trace.finish(&[output]).unwrap();
    let compiled = GraphCompiler::new().compile_traced_graph(&graph).unwrap();
    let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    let bad_rhs = Tensor::from_vec_col_major(vec![4, 2], vec![1.0_f64; 8]).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_einsum::register_runtime)
        .unwrap();

    assert!(executor
        .run_with_inputs(&compiled, &[&lhs, &bad_rhs])
        .is_err());
}

#[test]
fn explicit_nary_path_preserves_shape_constraints() {
    let spec = |rows, cols| {
        ProgramInputSpec::new(DType::F64, [DimExpr::Const(rows), DimExpr::Const(cols)])
    };
    let mut trace = TraceContext::new();
    let a = trace.input(spec(2, 3)).unwrap();
    let b = trace.input(spec(3, 4)).unwrap();
    let c = trace.input(spec(4, 5)).unwrap();
    let output = trace
        .einsum_with(
            &[a, b, c],
            "ab,bc,cd->ad",
            EinsumOptimize::Path(vec![(0, 1), (0, 1)]),
        )
        .unwrap();
    let graph = trace.finish(&[output]).unwrap();

    assert!(GraphCompiler::new().compile_traced_graph(&graph).is_ok());
}
