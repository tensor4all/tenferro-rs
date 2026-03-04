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

use num_complex::Complex;
use num_traits::Float;
use tenferro_algebra::{HasAlgebra, Scalar, Standard};
use tenferro_einsum as tf_einsum;
use tenferro_linalg::backend::CpuLinalgScalar;
use tenferro_linalg::{LinalgScalar, NormKind, SolveGrad};
use tenferro_prims::{CpuBackend, CpuContext, TensorPrims};
use tenferro_tensor::Tensor;

use crate::{reverse_tape, AdTensor, AdValue, Error, NodeId, Result};

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
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
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
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
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
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
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
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
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
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
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
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
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
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
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
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
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
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
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
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
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
    T: LinalgScalar<Real = T, Complex = Complex<T>> + Float + CpuLinalgScalar,
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
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
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
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
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
    T: LinalgScalar + CpuLinalgScalar,
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
) -> Result<Vec<Option<Tensor<T>>>> {
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
                out.push(all_grads.get(node).cloned());
            }
            _ => out.push(None),
        }
    }

    Ok(out)
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
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
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
mod tests {
    use num_complex::Complex64;

    use super::*;
    use crate::{AdValue, NodeId, RuntimeContext, TapeId};
    use tenferro_prims::CpuContext;
    use tenferro_tensor::{MemoryOrder, Tensor};

    fn f64_2x2(values: [f64; 4]) -> Tensor<f64> {
        Tensor::<f64>::from_slice(&values, &[2, 2], MemoryOrder::ColumnMajor).unwrap()
    }

    fn as_slice(t: &Tensor<f64>) -> &[f64] {
        t.buffer()
            .as_slice()
            .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"))
    }

    fn max_abs_diff(a: &Tensor<f64>, b: &Tensor<f64>) -> f64 {
        assert_eq!(a.dims(), b.dims());
        as_slice(a)
            .iter()
            .zip(as_slice(b).iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    fn complex_max_abs_diff(a: &Tensor<Complex64>, b: &Tensor<Complex64>) -> f64 {
        assert_eq!(a.dims(), b.dims());
        let a = a.contiguous(MemoryOrder::ColumnMajor);
        let b = b.contiguous(MemoryOrder::ColumnMajor);
        let a_off = a.offset() as usize;
        let b_off = b.offset() as usize;
        let len: usize = a.dims().iter().product();
        let a_data = &a
            .buffer()
            .as_slice()
            .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"))[a_off..a_off + len];
        let b_data = &b
            .buffer()
            .as_slice()
            .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"))[b_off..b_off + len];
        a_data
            .iter()
            .zip(b_data.iter())
            .map(|(x, y)| (*x - *y).norm())
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn eager_ad_linalg_and_einsum_cover_all_ops() {
        let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

        let a = f64_2x2([4.0, 1.0, 1.0, 3.0]);
        let b = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        let tri = f64_2x2([2.0, 0.0, 1.0, 3.0]);
        let general = f64_2x2([0.0, 1.0, -1.0, 0.0]);
        let rect = Tensor::<f64>::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let a_ls = Tensor::<f64>::from_slice(
            &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            &[3, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let b_ls =
            Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();

        let ad_a = AdTensor::new_primal(a.clone());
        let ad_b = AdTensor::new_primal(b);
        let _ad_tri = AdTensor::new_primal(tri);
        let ad_general = AdTensor::new_primal(general);
        let ad_rect = AdTensor::new_primal(rect);
        let ad_ls_a = AdTensor::new_primal(a_ls);
        let ad_ls_b = AdTensor::new_primal(b_ls);

        let out_einsum = einsum("ij,jk->ik", &[&ad_a, &ad_a]).unwrap();
        assert_eq!(out_einsum.dims(), &[2, 2]);
        let out_svd = svd(&ad_a).unwrap();
        assert_eq!(out_svd.s.dims(), &[2]);
        let out_qr = qr(&ad_a).unwrap();
        assert_eq!(out_qr.q.dims(), &[2, 2]);
        let out_lu = lu(&ad_a).unwrap();
        assert_eq!(out_lu.l.dims(), &[2, 2]);
        let out_eigen = eigen(&ad_a).unwrap();
        assert_eq!(out_eigen.values.dims(), &[2]);
        let out_lstsq = lstsq(&ad_ls_a, &ad_ls_b).unwrap();
        assert_eq!(out_lstsq.x.dims(), &[2]);
        assert_eq!(cholesky(&ad_a).unwrap().dims(), &[2, 2]);
        assert_eq!(solve(&ad_a, &ad_b).unwrap().dims(), &[2]);
        assert_eq!(inv(&ad_a).unwrap().dims(), &[2, 2]);
        assert_eq!(det(&ad_a).unwrap().dims(), &[]);
        let out_slogdet = slogdet(&ad_a).unwrap();
        assert_eq!(out_slogdet.sign.dims(), &[]);
        let out_eig = eig(&ad_general).unwrap();
        assert_eq!(out_eig.values.dims(), &[2]);
        assert_eq!(pinv(&ad_rect).unwrap().dims(), &[3, 2]);
        assert_eq!(matrix_exp(&ad_a).unwrap().dims(), &[2, 2]);
        assert_eq!(solve_triangular(&ad_a, &ad_b).unwrap().dims(), &[2]);
        assert_eq!(norm(&ad_a).unwrap().dims(), &[]);
    }

    #[test]
    fn eager_ad_preserves_mode_propagation() {
        let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

        let a = f64_2x2([4.0, 1.0, 1.0, 3.0]);
        let da = f64_2x2([0.1, 0.0, 0.0, 0.1]);
        let b = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();

        let ad_a_fwd = AdTensor::new_forward(a.clone(), da);
        let ad_b = AdTensor::new_primal(b);
        let out_fwd = solve(&ad_a_fwd, &ad_b).unwrap();
        assert!(matches!(out_fwd.as_value(), AdValue::Forward { .. }));

        let ad_a_rev = AdTensor::new_reverse(a.clone(), NodeId(1), TapeId(11), None);
        let ad_b_rev = AdTensor::new_reverse(a, NodeId(2), TapeId(11), None);
        let out_rev = einsum("ij,jk->ik", &[&ad_a_rev, &ad_b_rev]).unwrap();
        assert!(matches!(out_rev.as_value(), AdValue::Reverse { tape, .. } if *tape == TapeId(11)));
    }

    #[test]
    fn eager_local_einsum_rules_cover_rrule_frule_hvp() {
        let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

        let a = f64_2x2([1.0, 3.0, 2.0, 4.0]);
        let b = f64_2x2([2.0, -1.0, 0.5, 1.5]);
        let da = f64_2x2([0.1, 0.0, -0.2, 0.3]);
        let grad_c = f64_2x2([1.0, 0.0, 0.0, 1.0]);
        let dgrad_c = f64_2x2([0.5, 0.0, 0.0, 0.5]);

        let ad_a = AdTensor::new_primal(a);
        let ad_b = AdTensor::new_primal(b);
        let ad_da = AdTensor::new_primal(da);
        let ad_grad_c = AdTensor::new_primal(grad_c);
        let ad_dgrad_c = AdTensor::new_primal(dgrad_c);

        let grads = einsum_rrule("ij,jk->ik", &[&ad_a, &ad_b], &ad_grad_c).unwrap();
        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].dims(), &[2, 2]);
        assert_eq!(grads[1].dims(), &[2, 2]);

        let jvp = einsum_frule("ij,jk->ik", &[&ad_a, &ad_b], &[Some(&ad_da), None]).unwrap();
        assert_eq!(jvp.dims(), &[2, 2]);

        let hvp = einsum_hvp(
            "ij,jk->ik",
            &[&ad_a, &ad_b],
            &[Some(&ad_da), None],
            &ad_grad_c,
            &ad_dgrad_c,
        )
        .unwrap();
        assert_eq!(hvp.len(), 2);
        assert_eq!(hvp[0].0.dims(), &[2, 2]);
        assert_eq!(hvp[0].1.dims(), &[2, 2]);
    }

    #[test]
    fn eager_local_solve_triangular_rrule_runs() {
        let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

        let a = f64_2x2([2.0, 0.0, 1.0, 3.0]);
        let b = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        let cotangent =
            Tensor::<f64>::from_slice(&[0.5, -0.25], &[2], MemoryOrder::ColumnMajor).unwrap();

        let ad_a = AdTensor::new_primal(a);
        let ad_b = AdTensor::new_primal(b);
        let ad_cotangent = AdTensor::new_primal(cotangent);

        let grad = solve_triangular_rrule(&ad_a, &ad_b, &ad_cotangent, true).unwrap();
        assert_eq!(grad.a.dims(), &[2, 2]);
        assert_eq!(grad.b.dims(), &[2, 1]);
    }

    #[test]
    fn solve_triangular_builder_reverse_pullback_matches_rrule() {
        let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

        let tape = TapeId(101);
        let node_a = NodeId(11);
        let node_b = NodeId(12);

        let a = f64_2x2([2.0, 0.0, 1.0, 3.0]);
        let b = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        let cotangent =
            Tensor::<f64>::from_slice(&[0.5, -0.25], &[2], MemoryOrder::ColumnMajor).unwrap();

        let ad_a_rev = AdTensor::new_reverse(a.clone(), node_a, tape, None);
        let ad_b_rev = AdTensor::new_reverse(b.clone(), node_b, tape, None);
        let out = solve_triangular(&ad_a_rev, &ad_b_rev).unwrap();
        assert!(matches!(out.as_value(), AdValue::Reverse { tape: t, .. } if *t == tape));

        let ad_cotangent = AdTensor::new_primal(cotangent);
        let grad_map = pullback(&out, &ad_cotangent).unwrap();
        let grad_a = grad_map.get(&node_a).expect("missing dA");
        let grad_b = grad_map.get(&node_b).expect("missing dB");

        let expected = solve_triangular_rrule(
            &AdTensor::new_primal(a),
            &AdTensor::new_primal(b.clone()),
            &ad_cotangent,
            true,
        )
        .unwrap();

        assert_eq!(grad_a.dims(), &[2, 2]);
        assert!(max_abs_diff(grad_a, &expected.a) < 1e-12);

        let expected_b = expected.b.reshape(b.dims()).unwrap();
        assert_eq!(grad_b.dims(), b.dims());
        assert!(max_abs_diff(grad_b, &expected_b) < 1e-12);
    }

    #[test]
    fn einsum_builder_reverse_pullback_wrt_matches_rrule() {
        let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

        let tape = TapeId(202);
        let node_a = NodeId(31);
        let node_b = NodeId(32);

        let a = f64_2x2([1.0, 3.0, 2.0, 4.0]);
        let b = f64_2x2([2.0, -1.0, 0.5, 1.5]);
        let cotangent = f64_2x2([1.0, 0.0, 0.0, 1.0]);

        let ad_a_rev = AdTensor::new_reverse(a.clone(), node_a, tape, None);
        let ad_b_rev = AdTensor::new_reverse(b.clone(), node_b, tape, None);
        let out = einsum("ij,jk->ik", &[&ad_a_rev, &ad_b_rev]).unwrap();
        assert!(matches!(out.as_value(), AdValue::Reverse { tape: t, .. } if *t == tape));

        let ad_cotangent = AdTensor::new_primal(cotangent.clone());
        let grads = pullback_wrt(&out, &ad_cotangent, &[&ad_a_rev, &ad_b_rev]).unwrap();
        let grad_a = grads[0].as_ref().expect("missing einsum dA");
        let grad_b = grads[1].as_ref().expect("missing einsum dB");

        let expected = einsum_rrule(
            "ij,jk->ik",
            &[&AdTensor::new_primal(a), &AdTensor::new_primal(b)],
            &AdTensor::new_primal(cotangent),
        )
        .unwrap();

        assert!(max_abs_diff(grad_a, &expected[0]) < 1e-12);
        assert!(max_abs_diff(grad_b, &expected[1]) < 1e-12);
    }

    #[test]
    fn solve_triangular_reverse_pullback_complex_matches_rrule() {
        let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

        let tape = TapeId(303);
        let node_a = NodeId(41);
        let node_b = NodeId(42);

        let a = Tensor::<Complex64>::from_slice(
            &[
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, -0.5),
                Complex64::new(3.0, 0.0),
            ],
            &[2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let b = Tensor::<Complex64>::from_slice(
            &[Complex64::new(1.0, 0.5), Complex64::new(2.0, -0.25)],
            &[2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let cotangent = Tensor::<Complex64>::from_slice(
            &[Complex64::new(0.5, 0.0), Complex64::new(-0.25, 0.1)],
            &[2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();

        let ad_a_rev = AdTensor::new_reverse(a.clone(), node_a, tape, None);
        let ad_b_rev = AdTensor::new_reverse(b.clone(), node_b, tape, None);
        let out = solve_triangular(&ad_a_rev, &ad_b_rev).unwrap();
        let grads = pullback(&out, &AdTensor::new_primal(cotangent.clone())).unwrap();
        let grad_a = grads.get(&node_a).expect("missing complex dA");
        let grad_b = grads.get(&node_b).expect("missing complex dB");

        let expected = solve_triangular_rrule(
            &AdTensor::new_primal(a),
            &AdTensor::new_primal(b.clone()),
            &AdTensor::new_primal(cotangent),
            true,
        )
        .unwrap();

        let expected_b = expected.b.reshape(b.dims()).unwrap();
        assert!(complex_max_abs_diff(grad_a, &expected.a) < 1e-12);
        assert!(complex_max_abs_diff(grad_b, &expected_b) < 1e-12);
    }
}
