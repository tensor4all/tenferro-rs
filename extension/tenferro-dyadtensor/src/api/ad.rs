//! Eager AD entry points without `_ad` suffix.
//!
//! These functions are thin wrappers around the existing builder APIs
//! (`*_ad(...).run()`) and are intended for integration code paths that prefer
//! explicit eager execution.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_dyadtensor::{ad, set_default_runtime, AdTensor, RuntimeContext};
//! use tenferro_prims::CpuContext;
//! use tenferro_tensor::{MemoryOrder, Tensor};
//!
//! let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
//! let a = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
//!     .unwrap();
//! let ad_a = AdTensor::new_primal(a);
//! let out = ad::qr(&ad_a).unwrap();
//! assert_eq!(out.q.dims(), &[2, 2]);
//! ```

use std::collections::HashMap;

use chainrules_scalarops::ScalarAd;
use num_complex::Complex;
use num_traits::Float;
use tenferro_algebra::{HasAlgebra, Scalar, Standard};
use tenferro_einsum as tf_einsum;
use tenferro_linalg::backend::CpuLinalgScalar;
use tenferro_linalg::{LinalgScalar, NormKind, SolveGrad};
use tenferro_prims::{CpuBackend, CpuContext, TensorPrims};
use tenferro_tensor::Tensor;

use crate::{reverse_tape, AdScalar, AdTensor, AdValue, Error, NodeId, Result, StructuredTensor};

use super::{
    AdEigResult, AdEigenResult, AdLstsqResult, AdLuResult, AdQrResult, AdSlogdetResult, AdSvdResult,
};

/// Eager AD einsum.
///
/// Equivalent to `crate::einsum_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::ad::einsum("ij,jk->ik", &[&a, &b])?;
/// ```
pub fn einsum<'a, T>(subscripts: &'a str, operands: &'a [&'a AdTensor<T>]) -> Result<AdTensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    super::einsum_ad(subscripts, operands).run()
}

/// Eager AD full reduction / sum.
///
/// Equivalent to `crate::sum_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::ad::sum(&x)?;
/// ```
pub fn sum<T>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>> + Copy,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    super::sum_ad(tensor).run()
}

/// Eager AD SVD.
///
/// Equivalent to `crate::svd_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::ad::svd(&a)?;
/// ```
pub fn svd<T: Scalar>(tensor: &AdTensor<T>) -> Result<AdSvdResult<T>>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    super::svd_ad(tensor).run()
}

/// Eager AD QR.
///
/// Equivalent to `crate::qr_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::ad::qr(&a)?;
/// ```
pub fn qr<T: Scalar>(tensor: &AdTensor<T>) -> Result<AdQrResult<T>>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    super::qr_ad(tensor).run()
}

/// Eager AD LU (partial pivot by default).
///
/// Equivalent to `crate::lu_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::ad::lu(&a)?;
/// ```
pub fn lu<T: Scalar>(tensor: &AdTensor<T>) -> Result<AdLuResult<T>>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    super::lu_ad(tensor).run()
}

/// Eager AD symmetric/Hermitian eigen decomposition.
///
/// Equivalent to `crate::eigen_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::ad::eigen(&a)?;
/// ```
pub fn eigen<T: Scalar>(tensor: &AdTensor<T>) -> Result<AdEigenResult<T>>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    super::eigen_ad(tensor).run()
}

/// Eager AD least-squares solve.
///
/// Equivalent to `crate::lstsq_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::ad::lstsq(&a, &b)?;
/// ```
pub fn lstsq<T: Scalar>(a: &AdTensor<T>, b: &AdTensor<T>) -> Result<AdLstsqResult<T>>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    super::lstsq_ad(a, b).run()
}

/// Eager AD Cholesky.
///
/// Equivalent to `crate::cholesky_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::ad::cholesky(&a)?;
/// ```
pub fn cholesky<T: Scalar>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    super::cholesky_ad(tensor).run()
}

/// Eager AD linear solve.
///
/// Equivalent to `crate::solve_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::ad::solve(&a, &b)?;
/// ```
pub fn solve<T: Scalar>(a: &AdTensor<T>, b: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    super::solve_ad(a, b).run()
}

/// Eager AD inverse.
///
/// Equivalent to `crate::inv_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::ad::inv(&a)?;
/// ```
pub fn inv<T: Scalar>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    super::inv_ad(tensor).run()
}

/// Eager AD determinant.
///
/// Equivalent to `crate::det_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::ad::det(&a)?;
/// ```
pub fn det<T: Scalar>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    super::det_ad(tensor).run()
}

/// Eager AD slogdet.
///
/// Equivalent to `crate::slogdet_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::ad::slogdet(&a)?;
/// ```
pub fn slogdet<T: Scalar>(tensor: &AdTensor<T>) -> Result<AdSlogdetResult<T>>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    super::slogdet_ad(tensor).run()
}

/// Eager AD general eigendecomposition.
///
/// Equivalent to `crate::eig_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::ad::eig(&a)?;
/// ```
pub fn eig<T: Scalar>(tensor: &AdTensor<T>) -> Result<AdEigResult<T>>
where
    T: LinalgScalar<Real = T, Complex = Complex<T>>
        + Float
        + CpuLinalgScalar
        + HasAlgebra<Algebra = Standard<T>>,
    Complex<T>: Scalar,
{
    super::eig_ad(tensor).run()
}

/// Eager AD pseudoinverse.
///
/// Equivalent to `crate::pinv_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::ad::pinv(&a)?;
/// ```
pub fn pinv<T: Scalar>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    super::pinv_ad(tensor).run()
}

/// Eager AD matrix exponential.
///
/// Equivalent to `crate::matrix_exp_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::ad::matrix_exp(&a)?;
/// ```
pub fn matrix_exp<T: Scalar>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    super::matrix_exp_ad(tensor).run()
}

/// Eager AD triangular solve (upper=true by default).
///
/// Equivalent to `crate::solve_triangular_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::ad::solve_triangular(&a, &b)?;
/// ```
pub fn solve_triangular<T: Scalar>(a: &AdTensor<T>, b: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: LinalgScalar + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    super::solve_triangular_ad(a, b).run()
}

/// Reverse pullback from a reverse-mode output tensor.
///
/// The returned map is keyed by reverse `NodeId` and includes the seed
/// cotangent for the output node itself.
pub fn pullback<T: Scalar + 'static>(
    output: &AdTensor<T>,
    cotangent: &AdTensor<T>,
) -> Result<HashMap<NodeId, Tensor<T>>> {
    let (output_node, tape) = match output.as_value() {
        AdValue::Reverse { node, tape, .. } => (*node, *tape),
        _ => {
            return Err(Error::InvalidAdTensor {
                message: "ad::pullback requires reverse-mode output tensor".to_string(),
            })
        }
    };

    if cotangent.dims() != output.dims() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "ad::pullback cotangent shape mismatch: expected {:?}, got {:?}",
                output.dims(),
                cotangent.dims()
            ),
        });
    }

    reverse_tape::pullback(tape, output_node, cotangent.primal())
}

/// Reverse pullback projected to requested `wrt` tensors.
///
/// Returns `None` for non-reverse tensors or disconnected reverse tensors.
pub fn pullback_wrt<T: Scalar + 'static>(
    output: &AdTensor<T>,
    cotangent: &AdTensor<T>,
    wrt: &[&AdTensor<T>],
) -> Result<Vec<Option<StructuredTensor<T>>>> {
    let tape = match output.as_value() {
        AdValue::Reverse { tape, .. } => *tape,
        _ => {
            return Err(Error::InvalidAdTensor {
                message: "ad::pullback_wrt requires reverse-mode output tensor".to_string(),
            })
        }
    };

    let all_grads = pullback(output, cotangent)?;
    let mut out = Vec::with_capacity(wrt.len());

    for wrt_tensor in wrt {
        match wrt_tensor.as_value() {
            AdValue::Reverse { node, tape: t, .. } => {
                if *t != tape {
                    return Err(Error::MixedReverseTape {
                        expected: tape.0,
                        found: t.0,
                    });
                }
                let grad = all_grads
                    .get(node)
                    .map(|payload| {
                        StructuredTensor::new(
                            wrt_tensor.dims().to_vec(),
                            wrt_tensor.axis_classes().to_vec(),
                            payload.clone(),
                        )
                    })
                    .transpose()?;
                out.push(grad);
            }
            _ => out.push(None),
        }
    }

    Ok(out)
}

/// Reverse pullback projected to requested `wrt` tensors with a different scalar type.
///
/// This is used for mixed-domain rules such as `eig_ad` where outputs are complex
/// while inputs are real.
pub fn pullback_wrt_mixed<TOut: Scalar + 'static, TWrt: Scalar + 'static>(
    output: &AdTensor<TOut>,
    cotangent: &AdTensor<TOut>,
    wrt: &[&AdTensor<TWrt>],
) -> Result<Vec<Option<StructuredTensor<TWrt>>>> {
    let (output_node, tape) = match output.as_value() {
        AdValue::Reverse { node, tape, .. } => (*node, *tape),
        _ => {
            return Err(Error::InvalidAdTensor {
                message: "ad::pullback_wrt_mixed requires reverse-mode output tensor".to_string(),
            })
        }
    };

    if cotangent.dims() != output.dims() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "ad::pullback_wrt_mixed cotangent shape mismatch: expected {:?}, got {:?}",
                output.dims(),
                cotangent.dims()
            ),
        });
    }

    let mut wrt_nodes = Vec::with_capacity(wrt.len());
    for wrt_tensor in wrt {
        match wrt_tensor.as_value() {
            AdValue::Reverse { node, tape: t, .. } => {
                if *t != tape {
                    return Err(Error::MixedReverseTape {
                        expected: tape.0,
                        found: t.0,
                    });
                }
                wrt_nodes.push(Some(*node));
            }
            _ => wrt_nodes.push(None),
        }
    }

    let grads = reverse_tape::pullback_wrt_mixed::<TOut, TWrt>(
        tape,
        output_node,
        cotangent.primal(),
        &wrt_nodes,
    )?;

    grads
        .into_iter()
        .zip(wrt.iter())
        .map(|(grad, wrt_tensor)| {
            grad.map(|payload| {
                StructuredTensor::new(
                    wrt_tensor.dims().to_vec(),
                    wrt_tensor.axis_classes().to_vec(),
                    payload,
                )
            })
            .transpose()
        })
        .collect()
}

/// Reverse pullback projected to requested scalar inputs.
///
/// This is used by tensor outputs whose reverse rule depends on scalar
/// coefficients, such as `DynAdTensor::scale` and `DynAdTensor::axpby`.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{ad, AdScalar, AdTensor, AdValue, DynAdScalar, DynAdTensor, NodeId, TapeId};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let x: DynAdTensor = AdTensor::new_reverse(
///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
///     NodeId(1),
///     TapeId(7),
///     None,
/// )
/// .unwrap()
/// .into();
/// let a = DynAdScalar::from(AdValue::reverse(3.0_f64, NodeId(2), TapeId(7), None));
/// let y = x.scale(&a).unwrap();
/// let cotangent = AdTensor::new_primal(
///     Tensor::<f64>::from_slice(&[0.5, 1.25], &[2], MemoryOrder::ColumnMajor).unwrap(),
/// );
/// let a_typed = AdScalar::from(a.as_f64().unwrap().clone());
/// let grads = ad::pullback_wrt_scalars(y.as_f64().unwrap(), &cotangent, &[&a_typed]).unwrap();
/// assert_eq!(grads, vec![Some(3.0)]);
/// ```
pub fn pullback_wrt_scalars<TOut: Scalar + 'static, TWrt: ScalarAd + 'static>(
    output: &AdTensor<TOut>,
    cotangent: &AdTensor<TOut>,
    wrt: &[&AdScalar<TWrt>],
) -> Result<Vec<Option<TWrt>>> {
    let (output_node, tape) = match output.as_value() {
        AdValue::Reverse { node, tape, .. } => (*node, *tape),
        _ => {
            return Err(Error::InvalidAdTensor {
                message: "ad::pullback_wrt_scalars requires reverse-mode output tensor".to_string(),
            })
        }
    };

    if cotangent.dims() != output.dims() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "ad::pullback_wrt_scalars cotangent shape mismatch: expected {:?}, got {:?}",
                output.dims(),
                cotangent.dims()
            ),
        });
    }

    let mut wrt_nodes = Vec::with_capacity(wrt.len());
    for wrt_scalar in wrt {
        match wrt_scalar.as_value() {
            AdValue::Reverse { node, tape: t, .. } => {
                if *t != tape {
                    return Err(Error::MixedReverseTape {
                        expected: tape.0,
                        found: t.0,
                    });
                }
                wrt_nodes.push(Some(*node));
            }
            _ => wrt_nodes.push(None),
        }
    }

    reverse_tape::pullback_wrt_scalars::<TOut, TWrt>(
        tape,
        output_node,
        cotangent.primal(),
        &wrt_nodes,
    )
}

/// Eager AD norm (Frobenius by default).
///
/// Equivalent to `crate::norm_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::ad::norm(&a)?;
/// ```
pub fn norm<T: Scalar>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar + HasAlgebra<Algebra = Standard<T>>,
{
    super::norm_ad(tensor).kind(NormKind::Fro).run()
}

/// Local reverse-mode rule (VJP) for einsum.
///
/// Stateless helper for interop/manual AD paths. Inputs are AD tensors, but
/// derivatives are computed from their primal payloads.
pub fn einsum_rrule<'a, T>(
    subscripts: &'a str,
    operands: &'a [&'a AdTensor<T>],
    cotangent: &AdTensor<T>,
) -> Result<Vec<Tensor<T>>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    let primals: Vec<&Tensor<T>> = operands.iter().map(|op| op.primal()).collect();
    super::with_cpu_runtime("einsum_rrule", |ctx| {
        tf_einsum::einsum_rrule::<Standard<T>, CpuBackend>(
            ctx,
            subscripts,
            &primals,
            cotangent.primal(),
        )
        .map_err(Error::from)
    })
}

/// Local forward-mode rule (JVP) for einsum.
///
/// `tangents` must have the same length as `primals`.
pub fn einsum_frule<'a, T>(
    subscripts: &'a str,
    primals: &'a [&'a AdTensor<T>],
    tangents: &'a [Option<&'a AdTensor<T>>],
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    if primals.len() != tangents.len() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "einsum_frule requires tangents.len() == primals.len(), got {} vs {}",
                tangents.len(),
                primals.len()
            ),
        });
    }

    let primal_refs: Vec<&Tensor<T>> = primals.iter().map(|op| op.primal()).collect();
    let tangent_refs: Vec<Option<&Tensor<T>>> = tangents
        .iter()
        .map(|opt| opt.as_ref().map(|t| t.primal()))
        .collect();

    super::with_cpu_runtime("einsum_frule", |ctx| {
        tf_einsum::einsum_frule::<Standard<T>, CpuBackend>(
            ctx,
            subscripts,
            &primal_refs,
            &tangent_refs,
        )
        .map_err(Error::from)
    })
}

/// Local Hessian-vector product helper for einsum.
///
/// Returns one `(grad_k, hvp_k)` pair per input operand.
///
/// `tangents` must have the same length as `primals`.
pub fn einsum_hvp<'a, T>(
    subscripts: &'a str,
    primals: &'a [&'a AdTensor<T>],
    tangents: &'a [Option<&'a AdTensor<T>>],
    cotangent: &AdTensor<T>,
    cotangent_tangent: &AdTensor<T>,
) -> Result<Vec<(Tensor<T>, Tensor<T>)>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
{
    if primals.len() != tangents.len() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "einsum_hvp requires tangents.len() == primals.len(), got {} vs {}",
                tangents.len(),
                primals.len()
            ),
        });
    }

    let primal_refs: Vec<&Tensor<T>> = primals.iter().map(|op| op.primal()).collect();
    let tangent_refs: Vec<Option<&Tensor<T>>> = tangents
        .iter()
        .map(|opt| opt.as_ref().map(|t| t.primal()))
        .collect();

    super::with_cpu_runtime("einsum_hvp", |ctx| {
        tf_einsum::einsum_hvp::<Standard<T>, CpuBackend>(
            ctx,
            subscripts,
            &primal_refs,
            &tangent_refs,
            cotangent.primal(),
            cotangent_tangent.primal(),
        )
        .map_err(Error::from)
    })
}

/// Local reverse-mode rule (VJP) for triangular solve.
///
/// This is the stateless wrapper for `tenferro_linalg::solve_triangular_rrule`.
pub fn solve_triangular_rrule<T: Scalar>(
    a: &AdTensor<T>,
    b: &AdTensor<T>,
    cotangent: &AdTensor<T>,
    upper: bool,
) -> Result<SolveGrad<T>>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    super::with_cpu_runtime("solve_triangular_rrule", |ctx| {
        tenferro_linalg::solve_triangular_rrule::<T>(
            ctx,
            a.primal(),
            b.primal(),
            cotangent.primal(),
            upper,
        )
        .map_err(Error::from)
    })
}

#[cfg(test)]
mod tests;
