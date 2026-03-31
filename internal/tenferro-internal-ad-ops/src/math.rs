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
