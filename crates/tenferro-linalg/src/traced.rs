use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenferro_runtime::extension::apply;
use tenferro_runtime::{CompareDir, DType, DotGeneralConfig, Error, Result, TracedTensor};

use crate::extension::{LinalgExtensionOp, LinalgOp};

/// Linear algebra extension methods for [`TracedTensor`].
pub trait TracedTensorLinalgExt {
    fn svd(&self) -> Result<(TracedTensor, TracedTensor, TracedTensor)>;
    fn svd_with_eps(&self, eps: f64) -> Result<(TracedTensor, TracedTensor, TracedTensor)>;
    fn qr(&self) -> Result<(TracedTensor, TracedTensor)>;
    fn eigh(&self) -> Result<(TracedTensor, TracedTensor)>;
    fn eigh_with_eps(&self, eps: f64) -> Result<(TracedTensor, TracedTensor)>;
    fn cholesky(&self) -> Result<TracedTensor>;
    fn lu(&self) -> Result<(TracedTensor, TracedTensor, TracedTensor, TracedTensor)>;
    fn full_piv_lu(
        &self,
    ) -> Result<(
        TracedTensor,
        TracedTensor,
        TracedTensor,
        TracedTensor,
        TracedTensor,
    )>;
    fn eig(&self) -> Result<(TracedTensor, TracedTensor)>;
    fn solve(&self, b: &TracedTensor) -> Result<TracedTensor>;
    fn full_piv_lu_solve(&self, b: &TracedTensor) -> Result<TracedTensor>;
    fn triangular_solve(
        &self,
        b: &TracedTensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<TracedTensor>;
    fn slogdet(&self) -> Result<(TracedTensor, TracedTensor)>;
    fn det(&self) -> Result<TracedTensor>;
    fn inv(&self) -> Result<TracedTensor>;
    fn eigvalsh(&self) -> Result<TracedTensor>;
    fn eigvals(&self) -> Result<TracedTensor>;
    fn pinv(&self) -> Result<TracedTensor>;
    fn pinv_with_rtol(&self, rtol: f64) -> Result<TracedTensor>;
    fn norm(&self, ord: Option<f64>, dim: Option<&[usize]>, keepdim: bool) -> Result<TracedTensor>;
}

impl TracedTensorLinalgExt for TracedTensor {
    fn svd(&self) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
        svd(self)
    }

    fn svd_with_eps(&self, eps: f64) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
        svd_with_eps(self, eps)
    }

    fn qr(&self) -> Result<(TracedTensor, TracedTensor)> {
        qr(self)
    }

    fn eigh(&self) -> Result<(TracedTensor, TracedTensor)> {
        eigh(self)
    }

    fn eigh_with_eps(&self, eps: f64) -> Result<(TracedTensor, TracedTensor)> {
        eigh_with_eps(self, eps)
    }

    fn cholesky(&self) -> Result<TracedTensor> {
        cholesky(self)
    }

    fn lu(&self) -> Result<(TracedTensor, TracedTensor, TracedTensor, TracedTensor)> {
        lu(self)
    }

    fn full_piv_lu(
        &self,
    ) -> Result<(
        TracedTensor,
        TracedTensor,
        TracedTensor,
        TracedTensor,
        TracedTensor,
    )> {
        full_piv_lu(self)
    }

    fn eig(&self) -> Result<(TracedTensor, TracedTensor)> {
        eig(self)
    }

    fn solve(&self, b: &TracedTensor) -> Result<TracedTensor> {
        solve(self, b)
    }

    fn full_piv_lu_solve(&self, b: &TracedTensor) -> Result<TracedTensor> {
        full_piv_lu_solve(self, b)
    }

    fn triangular_solve(
        &self,
        b: &TracedTensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<TracedTensor> {
        triangular_solve(self, b, left_side, lower, transpose_a, unit_diagonal)
    }

    fn slogdet(&self) -> Result<(TracedTensor, TracedTensor)> {
        slogdet(self)
    }

    fn det(&self) -> Result<TracedTensor> {
        det(self)
    }

    fn inv(&self) -> Result<TracedTensor> {
        inv(self)
    }

    fn eigvalsh(&self) -> Result<TracedTensor> {
        eigvalsh(self)
    }

    fn eigvals(&self) -> Result<TracedTensor> {
        eigvals(self)
    }

    fn pinv(&self) -> Result<TracedTensor> {
        pinv(self)
    }

    fn pinv_with_rtol(&self, rtol: f64) -> Result<TracedTensor> {
        pinv_with_rtol(self, rtol)
    }

    fn norm(&self, ord: Option<f64>, dim: Option<&[usize]>, keepdim: bool) -> Result<TracedTensor> {
        norm(self, ord, dim, keepdim)
    }
}

/// Build a traced singular value decomposition op using the default epsilon.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap();
/// let (u, s, vt) = a.svd().unwrap();
/// assert_eq!(u.rank, 2);
/// assert_eq!(s.rank, 1);
/// assert_eq!(vt.rank, 2);
/// ```
pub fn svd(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
    svd_with_eps(a, 1e-12)
}

/// Build a traced singular value decomposition op with an explicit epsilon.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap();
/// let (_u, s, _vt) = a.svd_with_eps(1e-10).unwrap();
/// assert_eq!(s.rank, 1);
/// ```
pub fn svd_with_eps(
    a: &TracedTensor,
    eps: f64,
) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
    three_outputs(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Svd { eps })),
            &[a],
        )?,
        "svd",
    )
}

/// Build a traced QR decomposition op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap();
/// let (q, r) = a.qr().unwrap();
/// assert_eq!(q.rank, 2);
/// assert_eq!(r.rank, 2);
/// ```
pub fn qr(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor)> {
    two_outputs(
        apply(Arc::new(LinalgExtensionOp::new(LinalgOp::Qr)), &[a])?,
        "qr",
    )
}

/// Build a traced Hermitian eigenvalue decomposition op using the default epsilon.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]).unwrap();
/// let (values, vectors) = a.eigh().unwrap();
/// assert_eq!(values.rank, 1);
/// assert_eq!(vectors.rank, 2);
/// ```
pub fn eigh(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor)> {
    eigh_with_eps(a, 1e-12)
}

/// Build a traced Hermitian eigenvalue decomposition op with an explicit epsilon.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]).unwrap();
/// let (values, _vectors) = a.eigh_with_eps(1e-10).unwrap();
/// assert_eq!(values.rank, 1);
/// ```
pub fn eigh_with_eps(a: &TracedTensor, eps: f64) -> Result<(TracedTensor, TracedTensor)> {
    two_outputs(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Eigh { eps })),
            &[a],
        )?,
        "eigh",
    )
}

/// Build a traced Cholesky decomposition op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]).unwrap();
/// let factor = a.cholesky().unwrap();
/// assert_eq!(factor.rank, 2);
/// ```
pub fn cholesky(a: &TracedTensor) -> Result<TracedTensor> {
    one_output(
        apply(Arc::new(LinalgExtensionOp::new(LinalgOp::Cholesky)), &[a])?,
        "cholesky",
    )
}

/// Build a traced LU decomposition op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]).unwrap();
/// let (p, l, u, parity) = a.lu().unwrap();
/// assert_eq!(p.rank, 2);
/// assert_eq!(l.rank, 2);
/// assert_eq!(u.rank, 2);
/// assert_eq!(parity.rank, 0);
/// ```
pub fn lu(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor, TracedTensor, TracedTensor)> {
    four_outputs(
        apply(Arc::new(LinalgExtensionOp::new(LinalgOp::Lu)), &[a])?,
        "lu",
    )
}

/// Build a traced full-pivot LU decomposition op.
///
/// Returns `(P, L, U, Q, parity)` with reconstruction convention
/// `A = P^T * L * U * Q`, equivalently `P * A * Q^T = L * U`. `parity` is a
/// scalar real tensor containing `+1` or `-1`: `F32` for `F32`/`C32` inputs and
/// `F64` for `F64`/`C64` inputs.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]).unwrap();
/// let (p, l, u, q, parity) = a.full_piv_lu().unwrap();
/// assert_eq!(p.rank, 2);
/// assert_eq!(l.rank, 2);
/// assert_eq!(u.rank, 2);
/// assert_eq!(q.rank, 2);
/// assert_eq!(parity.rank, 0);
/// ```
pub fn full_piv_lu(
    a: &TracedTensor,
) -> Result<(
    TracedTensor,
    TracedTensor,
    TracedTensor,
    TracedTensor,
    TracedTensor,
)> {
    five_outputs(
        apply(Arc::new(LinalgExtensionOp::new(LinalgOp::FullPivLu)), &[a])?,
        "full_piv_lu",
    )
}

/// Build a traced general eigendecomposition op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]).unwrap();
/// let (values, vectors) = a.eig().unwrap();
/// assert_eq!(values.rank, 1);
/// assert_eq!(vectors.rank, 2);
/// ```
pub fn eig(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor)> {
    two_outputs(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Eig {
                input_dtype: a.dtype,
            })),
            &[a],
        )?,
        "eig",
    )
}

/// Build a traced linear solve op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]).unwrap();
/// let b = TracedTensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 9.0]).unwrap();
/// let x = a.solve(&b).unwrap();
/// assert_eq!(x.rank, 2);
/// ```
pub fn solve(a: &TracedTensor, b: &TracedTensor) -> Result<TracedTensor> {
    let mut factor_outputs =
        apply(Arc::new(LinalgExtensionOp::new(LinalgOp::LuFactor)), &[a])?.into_iter();
    let (packed_lu, pivots) = match (
        factor_outputs.next(),
        factor_outputs.next(),
        factor_outputs.next(),
        factor_outputs.next(),
    ) {
        (Some(packed_lu), Some(pivots), Some(_parity), None) => (packed_lu, pivots),
        _ => return Err(unexpected_output_count("lu_factor", 3)),
    };
    one_output(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::LuSolvePrepared {
                transpose_a: false,
                conjugate_a: false,
            })),
            &[a, &packed_lu, &pivots, b],
        )?,
        "solve",
    )
}

/// Build a traced full-pivot LU solve op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]).unwrap();
/// let b = TracedTensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 9.0]).unwrap();
/// let x = a.full_piv_lu_solve(&b).unwrap();
/// assert_eq!(x.rank, 2);
/// ```
pub fn full_piv_lu_solve(a: &TracedTensor, b: &TracedTensor) -> Result<TracedTensor> {
    one_output(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::FullPivLuSolve {
                transpose_a: false,
            })),
            &[a, b],
        )?,
        "full_piv_lu_solve",
    )
}

/// Build a traced triangular solve op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 1.0, 3.0]).unwrap();
/// let b = TracedTensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 9.0]).unwrap();
/// let x = a.triangular_solve(&b, true, true, false, false).unwrap();
/// assert_eq!(x.rank, 2);
/// ```
pub fn triangular_solve(
    a: &TracedTensor,
    b: &TracedTensor,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> Result<TracedTensor> {
    one_output(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            })),
            &[a, b],
        )?,
        "triangular_solve",
    )
}

/// Build traced sign and log-absolute-determinant ops.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]).unwrap();
/// let (sign, logabsdet) = a.slogdet().unwrap();
/// assert_eq!(sign.rank, 0);
/// assert_eq!(logabsdet.rank, 0);
/// ```
pub fn slogdet(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor)> {
    let mut factor_outputs =
        apply(Arc::new(LinalgExtensionOp::new(LinalgOp::LuFactor)), &[a])?.into_iter();
    let (packed_lu, parity) = match (
        factor_outputs.next(),
        factor_outputs.next(),
        factor_outputs.next(),
        factor_outputs.next(),
    ) {
        (Some(packed_lu), Some(_pivots), Some(parity), None) => (packed_lu, parity),
        _ => return Err(unexpected_output_count("lu_factor", 3)),
    };
    let diag_u = packed_lu.extract_diag(0, 1)?;
    let sign_u = diag_u.sign().reduce_prod(&[0])?;
    let sign = (&parity * &sign_u)?;
    let logabsdet = diag_u.abs().log().reduce_sum(&[0])?;
    Ok((sign, logabsdet))
}

/// Build a traced determinant op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]).unwrap();
/// let determinant = a.det().unwrap();
/// assert_eq!(determinant.rank, 0);
/// ```
pub fn det(a: &TracedTensor) -> Result<TracedTensor> {
    let (sign, logabsdet) = slogdet(a)?;
    &sign * &logabsdet.exp()
}

/// Build a traced matrix inverse op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]).unwrap();
/// let inverse = a.inv().unwrap();
/// assert_eq!(inverse.rank, 2);
/// ```
pub fn inv(a: &TracedTensor) -> Result<TracedTensor> {
    ensure_min_rank("inv", a.rank, 2)?;
    let shape = require_concrete_shape("inv", a)?;
    let eye = eye_like(a, shape[0])?;
    solve(a, &eye)
}

/// Build a traced Hermitian eigenvalue-only op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]).unwrap();
/// let values = a.eigvalsh().unwrap();
/// assert_eq!(values.rank, 1);
/// ```
pub fn eigvalsh(a: &TracedTensor) -> Result<TracedTensor> {
    eigh_values(a)
}

/// Build a traced general eigenvalue-only op.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]).unwrap();
/// let values = a.eigvals().unwrap();
/// assert_eq!(values.rank, 1);
/// ```
pub fn eigvals(a: &TracedTensor) -> Result<TracedTensor> {
    eig_values(a)
}

/// Build a traced Moore-Penrose pseudoinverse op.
///
/// Floating-point and complex inputs are supported. Integer and boolean inputs
/// return an unsupported-dtype error.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]).unwrap();
/// let inverse = a.pinv().unwrap();
/// assert_eq!(inverse.rank, 2);
/// ```
pub fn pinv(a: &TracedTensor) -> Result<TracedTensor> {
    ensure_float_or_complex("pinv", a.dtype)?;
    let shape = require_concrete_shape("pinv", a)?;
    let max_dim = match (shape.first(), shape.get(1)) {
        (Some(&m), Some(&n)) => m.max(n),
        (Some(&m), None) => m,
        _ => 0,
    };
    pinv_with_rtol(a, default_pinv_rtol(a.dtype, max_dim))
}

/// Build a traced Moore-Penrose pseudoinverse op with an explicit relative tolerance.
///
/// Floating-point and complex inputs are supported. Integer and boolean inputs
/// return an unsupported-dtype error.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]).unwrap();
/// let inverse = a.pinv_with_rtol(1e-12).unwrap();
/// assert_eq!(inverse.rank, 2);
/// ```
pub fn pinv_with_rtol(a: &TracedTensor, rtol: f64) -> Result<TracedTensor> {
    ensure_float_or_complex("pinv_with_rtol", a.dtype)?;
    require_concrete_shape("pinv_with_rtol", a)?;
    let (u, s, vt) = svd(a)?;
    let abs_s = s.abs();
    let s_max = abs_s.reduce_max(&[0])?;
    let s_max_shape = s_max.concrete_shape()?;
    let threshold_scalar = broadcast_scalar(scalar_real(s.dtype, rtol.max(0.0))?, &s_max_shape)?;
    let threshold = (&s_max * &threshold_scalar)?;
    let s_shape = s.concrete_shape()?;
    let threshold = broadcast_batch_scalar_to_leading_axis(&threshold, &s_shape)?;
    let mask = abs_s.compare(&threshold, CompareDir::Gt)?;
    let mask = mask.convert(s.dtype)?;
    let ones = ones_like(&s)?;
    let denom = (&s + &(&ones + &(-&mask))?)?;
    let s_inv = (&mask / &denom)?;

    let v = vt.conj().transpose(&matrix_transpose_perm(vt.rank))?;
    let uh = u.conj().transpose(&matrix_transpose_perm(u.rank))?;
    let vs = scale_matrix_columns(&v, &s_inv)?;
    matmul_preserve_trailing_batch(&vs, &uh)
}

/// Build a traced vector, matrix, or tensor norm op.
///
/// Floating-point and complex inputs are supported. Integer and boolean inputs
/// return an unsupported-dtype error.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
/// let length = x.norm(Some(2.0), Some(&[0]), false).unwrap();
/// assert_eq!(length.rank, 0);
/// ```
pub fn norm(
    a: &TracedTensor,
    ord: Option<f64>,
    dim: Option<&[usize]>,
    keepdim: bool,
) -> Result<TracedTensor> {
    ensure_float_or_complex("norm", a.dtype)?;
    let shape = require_concrete_shape("norm", a)?;
    let axes = dim.map_or_else(|| (0..a.rank).collect::<Vec<_>>(), |dims| dims.to_vec());
    if axes.is_empty() {
        return Ok(a.clone());
    }
    validate_axes("norm", a.rank, &axes)?;

    let out = match axes.len() {
        1 => vector_norm(a, axes[0], ord)?,
        2 => matrix_norm(a, &axes, ord)?,
        _ => {
            let abs = a.abs();
            match ord {
                None => frobenius_norm(&abs, &axes)?,
                Some(p) if p == f64::INFINITY => abs.reduce_max(&axes)?,
                Some(p) if p == f64::NEG_INFINITY => abs.reduce_min(&axes)?,
                Some(0.0) => count_nonzero(&abs, &axes)?,
                Some(p) => p_norm(&abs, &axes, p)?,
            }
        }
    };
    Ok(restore_keepdim(out, &shape, &axes, keepdim))
}

fn unexpected_output_count(name: &str, expected: usize) -> Error {
    Error::Internal(format!("{name} must produce exactly {expected} outputs"))
}

fn one_output(outputs: Vec<TracedTensor>, name: &str) -> Result<TracedTensor> {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next()) {
        (Some(output), None) => Ok(output),
        _ => Err(unexpected_output_count(name, 1)),
    }
}

fn two_outputs(outputs: Vec<TracedTensor>, name: &str) -> Result<(TracedTensor, TracedTensor)> {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next(), outputs.next()) {
        (Some(lhs), Some(rhs), None) => Ok((lhs, rhs)),
        _ => Err(unexpected_output_count(name, 2)),
    }
}

fn three_outputs(
    outputs: Vec<TracedTensor>,
    name: &str,
) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
    let mut outputs = outputs.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(first), Some(second), Some(third), None) => Ok((first, second, third)),
        _ => Err(unexpected_output_count(name, 3)),
    }
}

fn four_outputs(
    outputs: Vec<TracedTensor>,
    name: &str,
) -> Result<(TracedTensor, TracedTensor, TracedTensor, TracedTensor)> {
    let mut outputs = outputs.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(first), Some(second), Some(third), Some(fourth), None) => {
            Ok((first, second, third, fourth))
        }
        _ => Err(unexpected_output_count(name, 4)),
    }
}

fn five_outputs(
    outputs: Vec<TracedTensor>,
    name: &str,
) -> Result<(
    TracedTensor,
    TracedTensor,
    TracedTensor,
    TracedTensor,
    TracedTensor,
)> {
    let mut outputs = outputs.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(first), Some(second), Some(third), Some(fourth), Some(fifth), None) => {
            Ok((first, second, third, fourth, fifth))
        }
        _ => Err(unexpected_output_count(name, 5)),
    }
}

fn scalar_real(dtype: DType, value: f64) -> Result<TracedTensor> {
    match dtype {
        DType::F64 => TracedTensor::from_vec_col_major(vec![], vec![value]),
        DType::F32 => TracedTensor::from_vec_col_major(vec![], vec![value as f32]),
        DType::I32 => TracedTensor::from_vec_col_major(vec![], vec![value.round() as i32]),
        DType::I64 => TracedTensor::from_vec_col_major(vec![], vec![value.round() as i64]),
        DType::Bool => TracedTensor::from_vec_col_major(vec![], vec![value != 0.0]),
        DType::C64 => TracedTensor::from_vec_col_major(vec![], vec![Complex64::new(value, 0.0)]),
        DType::C32 => {
            TracedTensor::from_vec_col_major(vec![], vec![Complex32::new(value as f32, 0.0)])
        }
    }
}

fn ensure_float_or_complex(op: &'static str, dtype: DType) -> Result<()> {
    match dtype {
        DType::F32 | DType::F64 | DType::C32 | DType::C64 => Ok(()),
        DType::I32 | DType::I64 | DType::Bool => Err(Error::TensorRuntime(
            tenferro_tensor::Error::backend_failure(op, format!("unsupported dtype {dtype:?}")),
        )),
    }
}

fn ensure_min_rank(op: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual < expected {
        return Err(Error::TensorRuntime(tenferro_tensor::Error::RankMismatch {
            op,
            expected,
            actual,
        }));
    }
    Ok(())
}

fn validate_axes(op: &'static str, rank: usize, axes: &[usize]) -> Result<()> {
    for &axis in axes {
        if axis >= rank {
            return Err(Error::TensorRuntime(
                tenferro_tensor::Error::AxisOutOfBounds { op, axis, rank },
            ));
        }
    }
    Ok(())
}

fn require_concrete_shape(op: &'static str, input: &TracedTensor) -> Result<Vec<usize>> {
    input.try_concrete_shape().ok_or_else(|| {
        Error::TensorRuntime(tenferro_tensor::Error::backend_failure(
            op,
            "symbolic shape is not supported by this traced linalg helper",
        ))
    })
}

fn zero_scalar(dtype: DType) -> Result<TracedTensor> {
    scalar_real(dtype, 0.0)
}

fn one_scalar(dtype: DType) -> Result<TracedTensor> {
    scalar_real(dtype, 1.0)
}

fn ones_like(input: &TracedTensor) -> Result<TracedTensor> {
    let shape = input.concrete_shape()?;
    broadcast_scalar(one_scalar(input.dtype)?, &shape)
}

fn eye_like(anchor: &TracedTensor, size: usize) -> Result<TracedTensor> {
    let mut vector_shape = vec![size];
    let anchor_shape = anchor.concrete_shape()?;
    vector_shape.extend_from_slice(&anchor_shape[2..]);
    let diagonal = broadcast_scalar(one_scalar(anchor.dtype)?, &vector_shape)?;
    diagonal.embed_diag(0, 1)
}

fn broadcast_scalar(input: TracedTensor, shape: &[usize]) -> Result<TracedTensor> {
    let input_shape = input.concrete_shape()?;
    if input_shape == shape {
        return Ok(input);
    }
    input.broadcast_in_dim(shape, &[])
}

fn broadcast_batch_scalar_to_leading_axis(
    input: &TracedTensor,
    shape: &[usize],
) -> Result<TracedTensor> {
    let input_shape = input.concrete_shape()?;
    if input_shape == shape {
        return Ok(input.clone());
    }
    let dims: Vec<usize> = (1..shape.len()).collect();
    input.broadcast_in_dim(shape, &dims)
}

fn matmul_preserve_trailing_batch(lhs: &TracedTensor, rhs: &TracedTensor) -> Result<TracedTensor> {
    let rank = lhs.rank;
    let batch_dims: Vec<usize> = (2..rank).collect();
    lhs.dot_general(
        rhs,
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: batch_dims.clone(),
            rhs_batch_dims: batch_dims,
        },
    )
}

fn matrix_transpose_perm(rank: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.swap(0, 1);
    perm
}

fn frobenius_norm(abs: &TracedTensor, axes: &[usize]) -> Result<TracedTensor> {
    let squared = abs.pow(&scalar_real(abs.dtype, 2.0)?)?;
    Ok(squared.reduce_sum(axes)?.sqrt())
}

fn p_norm(abs: &TracedTensor, axes: &[usize], p: f64) -> Result<TracedTensor> {
    let power = abs.pow(&scalar_real(abs.dtype, p)?)?;
    let inv_p = scalar_real(abs.dtype, 1.0 / p)?;
    power.reduce_sum(axes)?.pow(&inv_p)
}

fn default_pinv_rtol(dtype: DType, max_dim: usize) -> f64 {
    let eps = match dtype {
        DType::F32 | DType::C32 => f32::EPSILON as f64,
        DType::F64 | DType::C64 => f64::EPSILON,
        DType::I32 | DType::I64 | DType::Bool => 0.0,
    };
    eps * max_dim as f64
}

fn vector_norm(a: &TracedTensor, axis: usize, ord: Option<f64>) -> Result<TracedTensor> {
    let abs = a.abs();
    match ord {
        None => frobenius_norm(&abs, &[axis]),
        Some(0.0) => count_nonzero(&abs, &[axis]),
        Some(p) if p == f64::INFINITY => abs.reduce_max(&[axis]),
        Some(p) if p == f64::NEG_INFINITY => abs.reduce_min(&[axis]),
        Some(p) => p_norm(&abs, &[axis], p),
    }
}

fn matrix_norm(a: &TracedTensor, axes: &[usize], ord: Option<f64>) -> Result<TracedTensor> {
    let matrix = move_axes_to_front(a, axes)?;
    let abs = matrix.abs();
    match ord {
        None => frobenius_norm(&abs, &[0, 1]),
        Some(p) if p == f64::INFINITY => matrix_row_sum_norm(&abs, true),
        Some(p) if p == f64::NEG_INFINITY => matrix_row_sum_norm(&abs, false),
        Some(1.0) => matrix_col_sum_norm(&abs, true),
        Some(-1.0) => matrix_col_sum_norm(&abs, false),
        Some(2.0) => {
            let singular_values = svd_values(&matrix)?.abs();
            singular_values.reduce_max(&[0])
        }
        Some(-2.0) => {
            let singular_values = svd_values(&matrix)?.abs();
            singular_values.reduce_min(&[0])
        }
        Some(0.0) => count_nonzero(&abs, &[0, 1]),
        Some(p) => p_norm(&abs, &[0, 1], p),
    }
}

fn svd_values(a: &TracedTensor) -> Result<TracedTensor> {
    one_output(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::SvdVals { eps: 1e-12 })),
            &[a],
        )?,
        "svd_values",
    )
}

fn eigh_values(a: &TracedTensor) -> Result<TracedTensor> {
    one_output(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::EighVals { eps: 1e-12 })),
            &[a],
        )?,
        "eigh_values",
    )
}

fn eig_values(a: &TracedTensor) -> Result<TracedTensor> {
    one_output(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::EigVals {
                input_dtype: a.dtype,
            })),
            &[a],
        )?,
        "eig_values",
    )
}

fn scale_matrix_columns(matrix: &TracedTensor, scale: &TracedTensor) -> Result<TracedTensor> {
    let matrix_shape = matrix.concrete_shape()?;
    let scale_shape_input = scale.concrete_shape()?;
    let mut scale_shape = vec![1, scale_shape_input[0]];
    scale_shape.extend_from_slice(&matrix_shape[2..]);
    let dims: Vec<usize> = (0..matrix_shape.len()).collect();
    let scale = scale
        .reshape(&scale_shape)
        .broadcast_in_dim(&matrix_shape, &dims)?;
    matrix * &scale
}

fn count_nonzero(abs: &TracedTensor, axes: &[usize]) -> Result<TracedTensor> {
    let mask = abs.compare(&zero_scalar(abs.dtype)?, CompareDir::Gt)?;
    mask.convert(abs.dtype)?.reduce_sum(axes)
}

fn matrix_row_sum_norm(abs: &TracedTensor, take_max: bool) -> Result<TracedTensor> {
    let row_sums = abs.reduce_sum(&[1])?;
    if take_max {
        row_sums.reduce_max(&[0])
    } else {
        row_sums.reduce_min(&[0])
    }
}

fn matrix_col_sum_norm(abs: &TracedTensor, take_max: bool) -> Result<TracedTensor> {
    let col_sums = abs.reduce_sum(&[0])?;
    if take_max {
        col_sums.reduce_max(&[0])
    } else {
        col_sums.reduce_min(&[0])
    }
}

fn move_axes_to_front(tensor: &TracedTensor, axes: &[usize]) -> Result<TracedTensor> {
    if axes.iter().enumerate().all(|(index, &axis)| index == axis) {
        return Ok(tensor.clone());
    }

    let mut selected = vec![false; tensor.rank];
    for &axis in axes {
        selected[axis] = true;
    }

    let mut perm = Vec::with_capacity(tensor.rank);
    perm.extend_from_slice(axes);
    for (axis, is_selected) in selected.iter().enumerate().take(tensor.rank) {
        if !*is_selected {
            perm.push(axis);
        }
    }
    tensor.transpose(&perm)
}

fn restore_keepdim(
    reduced: TracedTensor,
    original_shape: &[usize],
    axes: &[usize],
    keepdim: bool,
) -> TracedTensor {
    if !keepdim {
        return reduced;
    }
    let mut kept_shape = original_shape.to_vec();
    for &axis in axes {
        kept_shape[axis] = 1;
    }
    reduced.reshape(&kept_shape)
}
