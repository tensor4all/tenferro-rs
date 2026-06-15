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
// `grad` is used by the third-derivative PoC test. It may be used later for
// scalar loss gradients; `kdv_residual` uses `jvp` instead because `u` is not a
// scalar.
pub(crate) fn grad(output: &TracedTensor, input: &TracedTensor) -> Result<TracedTensor> {
    output.grad(input)
}

/// Compute the KdV residual `u_t + u * u_x + u_xxx`.
///
/// `u`, `x`, and `t` must have concrete shapes. `x` and `t` are the
/// independent-variable placeholders with respect to which derivatives are
/// taken. The returned tensor has the same shape as `u`.
#[allow(dead_code)]
pub(crate) fn kdv_residual(
    u: &TracedTensor,
    x: &TracedTensor,
    t: &TracedTensor,
) -> Result<TracedTensor> {
    let ones_x = ones_like(x);
    let ones_t = ones_like(t);
    let u_t = u.jvp(t, &ones_t)?;
    let u_x = u.jvp(x, &ones_x)?;
    let u_xx = u_x.jvp(x, &ones_x)?;
    let u_xxx = u_xx.jvp(x, &ones_x)?;
    let nonlinear = u.mul(&u_x);
    Ok(u_t.add(&nonlinear).add(&u_xxx))
}

#[allow(dead_code)]
fn ones_like(tensor: &TracedTensor) -> TracedTensor {
    let shape = tensor
        .try_concrete_shape()
        .expect("placeholder shape is concrete");
    let len = shape.iter().product::<usize>();
    TracedTensor::from_vec_col_major(shape.clone(), vec![1.0_f64; len])
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
    fn kdv_residual_of_zero_solution_is_zero() {
        use tenferro_cpu::CpuBackend;
        use tenferro_runtime::{DType, GraphCompiler, GraphExecutor};

        let x = TracedTensor::input_concrete_shape(DType::F64, &[2, 1]);
        let t = TracedTensor::input_concrete_shape(DType::F64, &[2, 1]);
        // Build a connected function that is identically zero but has non-trivial
        // graph dependencies on both x and t so that repeated JVPs stay active.
        let zero = TracedTensor::from_vec_col_major(vec![2, 1], vec![0.0_f64; 2]);
        let u = x.mul(&x).mul(&x).mul(&t).mul(&zero);
        let r = kdv_residual(&u, &x, &t).unwrap();

        let specs: Vec<(&TracedTensor, DType, &[usize])> =
            vec![(&x, DType::F64, &[2, 1]), (&t, DType::F64, &[2, 1])];
        let mut compiler = GraphCompiler::new();
        let program = compiler.compile_with_input_specs(&r, &specs).unwrap();
        let mut executor = GraphExecutor::new(CpuBackend::new());

        let x_tensor = Tensor::from_vec_col_major(vec![2, 1], vec![0.0_f64; 2]);
        let t_tensor = Tensor::from_vec_col_major(vec![2, 1], vec![0.0_f64; 2]);
        let out = executor
            .run_with_inputs(&program, &[(&x, &x_tensor), (&t, &t_tensor)])
            .unwrap();
        assert_eq!(out.as_slice::<f64>().unwrap(), &[0.0, 0.0]);
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
