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

use num_complex::Complex;
use num_traits::Float;
use tenferro_algebra::{HasAlgebra, Scalar, Standard};
use tenferro_linalg::backend::CpuLinalgScalar;
use tenferro_linalg::{LinalgScalar, NormKind};
use tenferro_prims::{CpuBackend, CpuContext, TensorPrims};

use crate::{AdTensor, Result};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AdValue, NodeId, RuntimeContext, TapeId};
    use tenferro_prims::CpuContext;
    use tenferro_tensor::{MemoryOrder, Tensor};

    fn f64_2x2(values: [f64; 4]) -> Tensor<f64> {
        Tensor::<f64>::from_slice(&values, &[2, 2], MemoryOrder::ColumnMajor).unwrap()
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
}
