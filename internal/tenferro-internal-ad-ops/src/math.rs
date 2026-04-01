use tenferro_algebra::{Scalar, Standard};
use tenferro_einsum as tf_einsum;
use tenferro_linalg::SolveGrad;
use tenferro_tensor::Tensor;

use chainrules_core::AutodiffError;

use crate::runtime::contracts::{EinsumRuntimeValue, LinalgRuntimeValue};
use crate::runtime::dispatch::{dispatch_einsum_runtime, with_linalg_runtime};
use crate::{Error, Result};

/// Stateless reverse-mode rule (VJP) for einsum over dense primals.
pub fn einsum_primal<'a, T>(subscripts: &'a str, operands: &'a [&'a Tensor<T>]) -> Result<Tensor<T>>
where
    T: EinsumRuntimeValue,
{
    dispatch_einsum_runtime!(T, "einsum", |ctx, Backend| {
        tf_einsum::einsum::<Standard<T>, Backend>(ctx, subscripts, operands, None)
            .map_err(Error::from)
    })
}

/// Stateless reverse-mode rule (VJP) for einsum over dense primals.
pub fn einsum_rrule<'a, T>(
    subscripts: &'a str,
    operands: &'a [&'a Tensor<T>],
    cotangent: &Tensor<T>,
) -> Result<Vec<Tensor<T>>>
where
    T: EinsumRuntimeValue,
{
    dispatch_einsum_runtime!(T, "einsum_rrule", |ctx, Backend| {
        tf_einsum::einsum_rrule::<Standard<T>, Backend>(ctx, subscripts, operands, cotangent)
            .map_err(Error::from)
    })
}

/// Stateless forward-mode rule (JVP) for einsum over dense primals.
pub fn einsum_frule<'a, T>(
    subscripts: &'a str,
    primals: &'a [&'a Tensor<T>],
    tangents: &'a [Option<&'a Tensor<T>>],
) -> Result<Tensor<T>>
where
    T: EinsumRuntimeValue,
{
    if primals.len() != tangents.len() {
        return Err(Error::Autodiff(AutodiffError::InvalidArgument(format!(
            "einsum_frule requires tangents.len() == primals.len(), got {} vs {}",
            tangents.len(),
            primals.len()
        ))));
    }

    dispatch_einsum_runtime!(T, "einsum_frule", |ctx, Backend| {
        tf_einsum::einsum_frule::<Standard<T>, Backend>(ctx, subscripts, primals, tangents)
            .map_err(Error::from)
    })
}

/// Stateless reverse-mode rule (VJP) for triangular solve.
pub fn solve_triangular_rrule<T>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    cotangent: &Tensor<T>,
    upper: bool,
) -> Result<SolveGrad<T>>
where
    T: Scalar + LinalgRuntimeValue,
{
    with_linalg_runtime::<T, _>(
        "solve_triangular_rrule",
        tenferro_linalg::backend::LinalgCapabilityOp::SolveTriangular,
        |ctx| {
            tenferro_linalg::solve_triangular_rrule::<T, _>(ctx, a, b, cotangent, upper)
                .map_err(Error::from)
        },
        |ctx| {
            tenferro_linalg::solve_triangular_rrule::<T, _>(ctx, a, b, cotangent, upper)
                .map_err(Error::from)
        },
        |ctx| {
            tenferro_linalg::solve_triangular_rrule::<T, _>(ctx, a, b, cotangent, upper)
                .map_err(Error::from)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenferro_internal_runtime::{set_default_runtime, RuntimeContext};
    use tenferro_prims::CpuContext;
    use tenferro_tensor::MemoryOrder;

    fn dense_f64(values: &[f64], dims: &[usize]) -> Tensor<f64> {
        Tensor::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap()
    }

    #[test]
    fn einsum_frule_rejects_tangent_arity_mismatch() {
        let x = dense_f64(&[1.0, 2.0], &[2]);
        let y = dense_f64(&[3.0, 4.0], &[2]);

        let err = einsum_frule("i,i->", &[&x, &y], &[Some(&x)]).unwrap_err();
        assert!(err
            .to_string()
            .contains("einsum_frule requires tangents.len() == primals.len()"));
    }

    #[test]
    fn einsum_helpers_run_with_cpu_runtime() {
        let _runtime = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
        let x = dense_f64(&[1.0, 2.0], &[2]);
        let y = dense_f64(&[3.0, 5.0], &[2]);
        let dx = dense_f64(&[0.5, -1.0], &[2]);
        let cotangent = dense_f64(&[2.0], &[]);

        let primal = einsum_primal("i,i->", &[&x, &y]).unwrap();
        let tangent = einsum_frule("i,i->", &[&x, &y], &[Some(&dx), None]).unwrap();
        let grads = einsum_rrule("i,i->", &[&x, &y], &cotangent).unwrap();

        assert_eq!(primal.buffer().as_slice().unwrap(), &[13.0]);
        assert_eq!(tangent.buffer().as_slice().unwrap(), &[-3.5]);
        assert_eq!(grads[0].buffer().as_slice().unwrap(), &[6.0, 10.0]);
        assert_eq!(grads[1].buffer().as_slice().unwrap(), &[2.0, 4.0]);
    }

    #[test]
    fn solve_triangular_rrule_requires_runtime_and_preserves_shapes() {
        let a = dense_f64(&[2.0, 0.0, 1.0, 3.0], &[2, 2]);
        let b = dense_f64(&[4.0, 9.0], &[2]);
        let cotangent = dense_f64(&[1.0, 2.0], &[2]);

        let err = solve_triangular_rrule(&a, &b, &cotangent, false).unwrap_err();
        assert!(matches!(err, Error::RuntimeNotConfigured));

        let _runtime = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
        let grad = solve_triangular_rrule(&a, &b, &cotangent, false).unwrap();
        assert_eq!(grad.a.dims(), &[2, 2]);
        assert_eq!(grad.b.dims(), &[2]);
    }
}
