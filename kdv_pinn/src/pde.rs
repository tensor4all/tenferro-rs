//! KdV residual and differentiation helpers.
//!
//! This module builds the PDE residual terms for the Korteweg–de Vries equation
//! and provides small wrappers around the traced-graph automatic-differentiation
//! APIs so the PINN code can request spatial and temporal derivatives concisely.

use tenferro_ad::TracedTensorAdExt;
use tenferro_runtime::{Result, TracedTensor};

/// Compute the gradient of `output` with respect to `input` on a traced graph.
///
/// This is a thin helper over [`TracedTensorAdExt::grad`] so that PDE residual
/// code can request derivatives without importing the AD extension trait
/// directly.
#[allow(dead_code)]
// Temporary scaffolding: `grad` is used by the third-derivative PoC test today
// and will be consumed by `kdv_residual` in Task 5.
pub(crate) fn grad(output: &TracedTensor, input: &TracedTensor) -> Result<TracedTensor> {
    output.grad(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_cpu::CpuBackend;
    use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
    use tenferro_tensor::Tensor;

    fn eval(tensor: &TracedTensor, bindings: &[(&TracedTensor, &Tensor)]) -> Tensor {
        let mut compiler = GraphCompiler::new();
        let specs: Vec<(&TracedTensor, tenferro_runtime::DType, &[usize])> = bindings
            .iter()
            .map(|(p, t)| (*p, t.dtype(), t.shape()))
            .collect();
        let program = compiler.compile_with_input_specs(tensor, &specs).unwrap();
        let mut executor = GraphExecutor::new(CpuBackend::new());
        executor.run_with_inputs(&program, bindings).unwrap()
    }

    #[test]
    fn third_derivative_of_cube() {
        let x = TracedTensor::input_concrete_shape(tenferro_runtime::DType::F64, &[3, 1]);
        let y = x.mul(&x).mul(&x).sum(&[0, 1]); // x^3 reduced to scalar
        let y_x = grad(&y, &x).unwrap();
        let y_xx = grad(&y_x.sum(&[0, 1]), &x).unwrap();
        let y_xxx = grad(&y_xx.sum(&[0, 1]), &x).unwrap();

        let x_tensor = Tensor::from_vec_col_major(vec![3, 1], vec![1.0_f64, 2.0, 3.0]);
        let result = eval(&y_xxx, &[(&x, &x_tensor)]);
        let data = result.as_slice::<f64>().unwrap();
        // d^3/dx^3 x^3 = 6
        assert!((data[0] - 6.0).abs() < 1e-6);
        assert!((data[1] - 6.0).abs() < 1e-6);
        assert!((data[2] - 6.0).abs() < 1e-6);
    }
}
