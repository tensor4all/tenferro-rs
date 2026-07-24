use tenferro_cpu::CpuBackend;
use tenferro_einsum::TraceContextEinsumExt;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{GraphCompiler, GraphExecutor, Tensor, TraceContext};
use tenferro_xla::XlaExecutor;

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error < 1.0e-12,
            "value {index}: actual={actual}, expected={expected}, error={error}"
        );
    }
}

fn lhs_value() -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(Tensor::from_vec_col_major(
        vec![2, 3],
        vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0],
    )?)
}

fn middle_value() -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(Tensor::from_vec_col_major(
        vec![3, 4],
        vec![
            10.0_f64, 20.0, 30.0, 11.0, 21.0, 31.0, 12.0, 22.0, 32.0, 13.0, 23.0, 33.0,
        ],
    )?)
}

fn tail_value() -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(Tensor::from_vec_col_major(
        vec![4, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a_value = lhs_value()?;
    let b_value = middle_value()?;
    let c_value = tail_value()?;
    let mut trace = TraceContext::new();
    let a = trace.input(ProgramInputSpec::new(
        a_value.dtype(),
        DimExpr::from_concrete(a_value.shape()),
    ))?;
    let b = trace.input(ProgramInputSpec::new(
        b_value.dtype(),
        DimExpr::from_concrete(b_value.shape()),
    ))?;
    let c = trace.input(ProgramInputSpec::new(
        c_value.dtype(),
        DimExpr::from_concrete(c_value.shape()),
    ))?;
    let product = trace.einsum(&[a, b, c], "ij,jk,kl->il")?;
    let graph = trace.finish(&[product])?;
    let program = GraphCompiler::new().compile_traced_graph(&graph)?;

    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor.register_extension(tenferro_einsum::register_runtime)?;
    let cpu_value = executor.run_with_inputs(&program, &[&a_value, &b_value, &c_value])?;
    assert_eq!(cpu_value.shape(), &[2, 2]);
    assert_close(
        cpu_value.as_slice::<f64>().unwrap(),
        &[1520.0, 3500.0, 3904.0, 8980.0],
    );

    let module = XlaExecutor::default().lower_compiled_to_stablehlo(&program)?;
    let stablehlo = module.as_str();

    assert!(stablehlo.contains("stablehlo.dot_general"));
    assert_eq!(stablehlo.matches("stablehlo.dot_general").count(), 2);
    assert!(stablehlo.contains("contracting_dims = [1] x [0]"));
    assert!(stablehlo.contains("tensor<2x2xf64>"));

    Ok(())
}
