use std::sync::Arc;

use tenferro_ad::error::{Error, Result};
use tenferro_ad::extension::apply_eager;
use tenferro_ad::EagerTensor;

use crate::extension::{LinalgExtensionOp, LinalgOp};
use crate::register_runtime;

fn apply_linalg_eager(op: LinalgOp, inputs: &[&EagerTensor]) -> Result<Vec<EagerTensor>> {
    if let Some(first) = inputs.first() {
        first
            .runtime()
            .register_extension(register_runtime)
            .map_err(|err| Error::Internal(err.to_string()))?;
    }
    apply_eager(Arc::new(LinalgExtensionOp::new(op)), inputs)
}

/// Singular value decomposition for eager tensors.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
///
/// let ctx = EagerRuntime::new();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]),
///     ctx,
/// );
/// let (_u, s, _vt) = tenferro_linalg::eager_tensor::svd(&a)?;
/// assert_eq!(s.data().shape(), &[2]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
pub fn svd(a: &EagerTensor) -> Result<(EagerTensor, EagerTensor, EagerTensor)> {
    let mut outputs = apply_linalg_eager(LinalgOp::Svd { eps: 0.0 }, &[a])?.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(u), Some(s), Some(vt), None) => Ok((u, s, vt)),
        _ => Err(Error::Internal(
            "svd eager op returned an unexpected number of outputs".to_string(),
        )),
    }
}

/// QR decomposition for eager tensors.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
///
/// let ctx = EagerRuntime::new();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]),
///     ctx,
/// );
/// let (q, r) = tenferro_linalg::eager_tensor::qr(&a)?;
/// assert_eq!(q.data().shape(), &[2, 2]);
/// assert_eq!(r.data().shape(), &[2, 2]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
pub fn qr(a: &EagerTensor) -> Result<(EagerTensor, EagerTensor)> {
    two_outputs(apply_linalg_eager(LinalgOp::Qr, &[a])?, "qr")
}

/// LU factorization for eager tensors.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
///
/// let ctx = EagerRuntime::new();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 1.0, 1.0, 0.0]),
///     ctx,
/// );
/// let (_p, l, u, parity) = tenferro_linalg::eager_tensor::lu(&a)?;
/// assert_eq!(l.data().shape(), &[2, 2]);
/// assert_eq!(u.data().shape(), &[2, 2]);
/// assert_eq!(parity.data().shape(), &[] as &[usize]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
pub fn lu(a: &EagerTensor) -> Result<(EagerTensor, EagerTensor, EagerTensor, EagerTensor)> {
    let mut outputs = apply_linalg_eager(LinalgOp::Lu, &[a])?.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(p), Some(l), Some(u), Some(parity), None) => Ok((p, l, u, parity)),
        _ => Err(Error::Internal(
            "lu eager op returned an unexpected number of outputs".to_string(),
        )),
    }
}

/// Complete-pivot LU factorization for eager tensors.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
///
/// let ctx = EagerRuntime::new();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0]),
///     ctx,
/// );
/// let (p, _l, _u, q, parity) = tenferro_linalg::eager_tensor::full_piv_lu(&a)?;
/// assert_eq!(p.data().shape(), &[2, 2]);
/// assert_eq!(q.data().shape(), &[2, 2]);
/// assert_eq!(parity.data().shape(), &[] as &[usize]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
pub fn full_piv_lu(
    a: &EagerTensor,
) -> Result<(
    EagerTensor,
    EagerTensor,
    EagerTensor,
    EagerTensor,
    EagerTensor,
)> {
    let mut outputs = apply_linalg_eager(LinalgOp::FullPivLu, &[a])?.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(p), Some(l), Some(u), Some(q), Some(parity), None) => Ok((p, l, u, q, parity)),
        _ => Err(Error::Internal(
            "full_piv_lu eager op returned an unexpected number of outputs".to_string(),
        )),
    }
}

/// Solve a linear system using complete-pivot LU behavior.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
///
/// let ctx = EagerRuntime::new();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0]),
///     ctx.clone(),
/// );
/// let b = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 1], vec![-1.0_f64, 5.0]),
///     ctx,
/// );
/// let x = tenferro_linalg::eager_tensor::full_piv_lu_solve(&a, &b)?;
/// assert_eq!(x.data().shape(), &[2, 1]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
pub fn full_piv_lu_solve(a: &EagerTensor, b: &EagerTensor) -> Result<EagerTensor> {
    one_output(
        apply_linalg_eager(LinalgOp::FullPivLuSolve { transpose_a: false }, &[a, b])?,
        "full_piv_lu_solve",
    )
}

/// Solve a linear system for eager tensors.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
///
/// let ctx = EagerRuntime::new();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]),
///     ctx.clone(),
/// );
/// let b = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 8.0]),
///     ctx,
/// );
/// let x = tenferro_linalg::eager_tensor::solve(&a, &b)?;
/// assert_eq!(x.data().shape(), &[2, 1]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
pub fn solve(a: &EagerTensor, b: &EagerTensor) -> Result<EagerTensor> {
    let mut factor_outputs = apply_linalg_eager(LinalgOp::LuFactor, &[a])?.into_iter();
    let (packed_lu, pivots) = match (
        factor_outputs.next(),
        factor_outputs.next(),
        factor_outputs.next(),
        factor_outputs.next(),
    ) {
        (Some(packed_lu), Some(pivots), Some(_parity), None) => (packed_lu, pivots),
        _ => {
            return Err(Error::Internal(
                "lu_factor eager op returned an unexpected number of outputs".to_string(),
            ));
        }
    };
    one_output(
        apply_linalg_eager(
            LinalgOp::LuSolvePrepared {
                transpose_a: false,
                conjugate_a: false,
            },
            &[a, &packed_lu, &pivots, b],
        )?,
        "solve",
    )
}

/// Cholesky factorization for eager tensors.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
///
/// let ctx = EagerRuntime::new();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]),
///     ctx,
/// );
/// let l = tenferro_linalg::eager_tensor::cholesky(&a)?;
/// assert_eq!(l.data().shape(), &[2, 2]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
pub fn cholesky(a: &EagerTensor) -> Result<EagerTensor> {
    one_output(apply_linalg_eager(LinalgOp::Cholesky, &[a])?, "cholesky")
}

/// Hermitian eigenvalue decomposition for eager tensors.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
///
/// let ctx = EagerRuntime::new();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]),
///     ctx,
/// );
/// let (values, vectors) = tenferro_linalg::eager_tensor::eigh(&a)?;
/// assert_eq!(values.data().shape(), &[2]);
/// assert_eq!(vectors.data().shape(), &[2, 2]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
pub fn eigh(a: &EagerTensor) -> Result<(EagerTensor, EagerTensor)> {
    two_outputs(
        apply_linalg_eager(LinalgOp::Eigh { eps: 0.0 }, &[a])?,
        "eigh",
    )
}

/// General eigenvalue decomposition for eager tensors.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
///
/// let ctx = EagerRuntime::new();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]),
///     ctx,
/// );
/// let (values, vectors) = tenferro_linalg::eager_tensor::eig(&a)?;
/// assert_eq!(values.data().shape(), &[2]);
/// assert_eq!(vectors.data().shape(), &[2, 2]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
pub fn eig(a: &EagerTensor) -> Result<(EagerTensor, EagerTensor)> {
    two_outputs(
        apply_linalg_eager(
            LinalgOp::Eig {
                input_dtype: a.data().dtype(),
            },
            &[a],
        )?,
        "eig",
    )
}

/// Triangular solve for eager tensors.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
///
/// let ctx = EagerRuntime::new();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 1.0, 3.0]),
///     ctx.clone(),
/// );
/// let b = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 7.0]),
///     ctx,
/// );
/// let x = tenferro_linalg::eager_tensor::triangular_solve(
///     &a, &b, true, true, false, false,
/// )?;
/// assert_eq!(x.data().shape(), &[2, 1]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
pub fn triangular_solve(
    a: &EagerTensor,
    b: &EagerTensor,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> Result<EagerTensor> {
    one_output(
        apply_linalg_eager(
            LinalgOp::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            },
            &[a, b],
        )?,
        "triangular_solve",
    )
}

fn one_output(outputs: Vec<EagerTensor>, name: &str) -> Result<EagerTensor> {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next()) {
        (Some(output), None) => Ok(output),
        _ => Err(Error::Internal(format!(
            "{name} eager op returned an unexpected number of outputs"
        ))),
    }
}

fn two_outputs(outputs: Vec<EagerTensor>, name: &str) -> Result<(EagerTensor, EagerTensor)> {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next(), outputs.next()) {
        (Some(lhs), Some(rhs), None) => Ok((lhs, rhs)),
        _ => Err(Error::Internal(format!(
            "{name} eager op returned an unexpected number of outputs"
        ))),
    }
}
