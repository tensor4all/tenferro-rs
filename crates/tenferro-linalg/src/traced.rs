use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenferro_runtime::extension::apply;
use tenferro_runtime::{
    CompareDir, DType, DotGeneralConfig, Error, ErrorPhase, Result, TracedTensor,
};

use crate::extension::{
    validate_derivative_eps, EighOptions, LinalgExtensionOp, LinalgOp, QrOptions, SvdOptions,
};

/// Linear algebra extension methods for [`TracedTensor`].
pub trait TracedTensorLinalgExt {
    /// Build a traced SVD operation with default options.
    ///
    /// # Errors
    ///
    /// Returns `Error::Extension` with `ErrorKind::Unsupported` for an
    /// unsupported dtype, or `Error::Validation` for invalid graph metadata.
    ///
    /// # Deferred errors
    ///
    /// Backend numerical failures and concrete shape mismatches can be
    /// reported as `Error::Extension` or `Error::Validation` during compile or
    /// execution when symbolic inputs are bound.
    fn svd(&self) -> Result<(TracedTensor, TracedTensor, TracedTensor)>;

    /// Build a traced SVD operation with explicit derivative and gauge options.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation::InvalidArgument` for a non-finite or
    /// non-positive derivative epsilon, or `Error::Extension` for unsupported
    /// dtype and graph registration failures.
    ///
    /// # Deferred errors
    ///
    /// Solver convergence and symbolic shape checks may be reported during
    /// compile or execution.
    fn svd_with_options(
        &self,
        options: SvdOptions,
    ) -> Result<(TracedTensor, TracedTensor, TracedTensor)>;

    /// Build a traced full-matrices SVD operation returning square `U (m x m)`
    /// and `Vh (n x n)`, whose trailing `n - rank` rows span the input's right
    /// nullspace.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` when the input is not a batched matrix
    /// (rank `>= 2`), or `Error::Extension` for graph registration failures.
    ///
    /// # Deferred errors
    ///
    /// The active backend returns `Error::Extension` with
    /// `ErrorKind::Unsupported` at execution if it does not implement
    /// full-matrices SVD (only the CPU faer provider does in this slice; the
    /// LAPACK provider and GPU backends are unsupported). Automatic
    /// differentiation is intentionally unsupported for the full variant (see
    /// the linalg AD support manifest) and surfaces a typed AD error rather
    /// than a silent thin-SVD fallback.
    fn svd_full(&self) -> Result<(TracedTensor, TracedTensor, TracedTensor)>;

    /// Build a traced QR operation.
    ///
    /// # Errors
    ///
    /// Returns `Error::Extension` with `ErrorKind::Unsupported` for an
    /// unsupported dtype or `Error::Validation` for invalid graph metadata.
    ///
    /// # Deferred errors
    ///
    /// Concrete shape validation and backend QR failures may be reported at
    /// compile or execution time for symbolic inputs.
    fn qr(&self) -> Result<(TracedTensor, TracedTensor)>;

    /// Build a traced QR operation with explicit gauge options.
    ///
    /// # Errors
    ///
    /// Returns `Error::Extension` with `ErrorKind::Unsupported` for an
    /// unsupported dtype, or `Error::Validation` for invalid graph metadata.
    ///
    /// # Deferred errors
    ///
    /// Symbolic shape checks and backend QR failures can be deferred to compile
    /// or execution.
    fn qr_with_options(&self, options: QrOptions) -> Result<(TracedTensor, TracedTensor)>;

    /// Build a traced Hermitian eigendecomposition operation.
    ///
    /// # Errors
    ///
    /// Returns `Error::Extension` with `ErrorKind::Unsupported` for an
    /// unsupported dtype or `Error::Validation` for invalid graph metadata.
    ///
    /// # Deferred errors
    ///
    /// Concrete square-shape validation and solver failures may be reported at
    /// compile or execution time.
    fn eigh(&self) -> Result<(TracedTensor, TracedTensor)>;

    /// Build a traced Hermitian eigendecomposition with explicit options.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation::InvalidArgument` for an invalid derivative
    /// epsilon, or `Error::Extension` for unsupported dtype and registration
    /// failures.
    ///
    /// # Deferred errors
    ///
    /// Symbolic square-shape checks and numerical eigensolver failures may be
    /// reported during compile or execution.
    fn eigh_with_options(&self, options: EighOptions) -> Result<(TracedTensor, TracedTensor)>;

    /// Build a traced Cholesky factorization operation.
    ///
    /// # Errors
    ///
    /// Returns `Error::Extension` with `ErrorKind::Unsupported` for an
    /// unsupported dtype or `Error::Validation` for invalid graph metadata.
    ///
    /// # Deferred errors
    ///
    /// Non-square or non-positive-definite concrete inputs can produce
    /// validation or numerical extension errors during compile or execution.
    fn cholesky(&self) -> Result<TracedTensor>;

    /// Build a traced LU factorization operation.
    ///
    /// # Errors
    ///
    /// Returns `Error::Extension` with `ErrorKind::Unsupported` for an
    /// unsupported dtype or `Error::Validation` for invalid graph metadata.
    ///
    /// # Deferred errors
    ///
    /// Concrete shape checks and backend factorization failures may be
    /// reported during compile or execution.
    fn lu(&self) -> Result<(TracedTensor, TracedTensor, TracedTensor, TracedTensor)>;

    /// Build a traced complete-pivot LU factorization operation.
    ///
    /// # Errors
    ///
    /// Returns `Error::Extension` with `ErrorKind::Unsupported` for an
    /// unsupported dtype or `Error::Validation` for invalid graph metadata.
    ///
    /// # Deferred errors
    ///
    /// Concrete square-shape checks and backend factorization failures may be
    /// reported during compile or execution.
    fn full_piv_lu(
        &self,
    ) -> Result<(
        TracedTensor,
        TracedTensor,
        TracedTensor,
        TracedTensor,
        TracedTensor,
    )>;
    /// Build a traced general eigendecomposition operation.
    ///
    /// # Errors
    ///
    /// Returns `Error::Extension` with `ErrorKind::Unsupported` for an
    /// unsupported dtype or `Error::Validation` for invalid graph metadata.
    ///
    /// # Deferred errors
    ///
    /// Concrete shape validation and numerical eigensolver failures may be
    /// reported during compile or execution.
    fn eig(&self) -> Result<(TracedTensor, TracedTensor)>;

    /// Build a traced linear solve operation.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for incompatible coefficient/rhs metadata
    /// and `Error::Extension` for unsupported dtype or registration failures.
    ///
    /// # Deferred errors
    ///
    /// Singular systems and concrete shape mismatches are reported as
    /// numerical or validation errors during compile or execution.
    fn solve(&self, b: &TracedTensor) -> Result<TracedTensor>;

    /// Build a traced least-squares solve `argmin_x ||A x - b||_2` for a tall
    /// or square, full-column-rank `A`, via the thin QR factorization.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for an invalid rank (`A` or `b` not a
    /// batched matrix, rank `< 2`), a symbolic shape, a wide/underdetermined
    /// `A` (`rows < cols`), or an unsupported dtype (not floating-point or
    /// complex).
    ///
    /// # Deferred errors
    ///
    /// Backend QR and triangular-solve failures and concrete shape mismatches
    /// are reported during compile or execution. Rank-deficient `A` is not
    /// detected: `R` is singular and the result is ill-defined, so callers must
    /// ensure full column rank.
    fn lstsq(&self, b: &TracedTensor) -> Result<TracedTensor>;

    /// Build a traced complete-pivot LU solve operation.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for incompatible coefficient/rhs metadata
    /// and `Error::Extension` for unsupported dtype or registration failures.
    ///
    /// # Deferred errors
    ///
    /// Singular systems and concrete shape mismatches may be reported during
    /// compile or execution.
    fn full_piv_lu_solve(&self, b: &TracedTensor) -> Result<TracedTensor>;

    /// Build a traced triangular solve operation.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for incompatible coefficient/rhs shapes or
    /// invalid solve flags, and `Error::Extension` for unsupported dtype.
    ///
    /// # Deferred errors
    ///
    /// Singular or zero-diagonal systems can fail numerically during compile or
    /// execution after symbolic inputs are bound.
    fn triangular_solve(
        &self,
        b: &TracedTensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<TracedTensor>;
    /// Build a traced sign/log-determinant operation.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for invalid matrix metadata or
    /// `Error::Extension` for unsupported dtype and registration failures.
    ///
    /// # Deferred errors
    ///
    /// Concrete singularity and shape failures can be reported during compile
    /// or execution.
    fn slogdet(&self) -> Result<(TracedTensor, TracedTensor)>;

    /// Build a traced determinant operation.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for invalid matrix metadata or
    /// `Error::Extension` for unsupported dtype.
    ///
    /// # Deferred errors
    ///
    /// Concrete singularity and shape failures may be reported during compile
    /// or execution.
    fn det(&self) -> Result<TracedTensor>;

    /// Build a traced matrix-inverse operation.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for incompatible rank/shape metadata or
    /// `Error::Extension` for unsupported dtype.
    ///
    /// # Deferred errors
    ///
    /// Singular matrices produce a numerical error during compile or execution.
    fn inv(&self) -> Result<TracedTensor>;

    /// Build a traced Hermitian eigenvalue-only operation.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for non-square metadata or
    /// `Error::Extension` for unsupported dtype.
    ///
    /// # Deferred errors
    ///
    /// Concrete square-shape and solver failures may be reported during compile
    /// or execution.
    fn eigvalsh(&self) -> Result<TracedTensor>;

    /// Build a traced general eigenvalue-only operation.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for invalid matrix metadata or
    /// `Error::Extension` for unsupported dtype.
    ///
    /// # Deferred errors
    ///
    /// Concrete shape and eigensolver failures may be reported during compile
    /// or execution.
    fn eigvals(&self) -> Result<TracedTensor>;

    /// Build a traced pseudoinverse operation with the default tolerance.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for invalid rank/shape metadata or
    /// `Error::Extension` for unsupported dtype.
    ///
    /// # Deferred errors
    ///
    /// SVD convergence and concrete shape failures may be reported during
    /// compile or execution.
    fn pinv(&self) -> Result<TracedTensor>;

    /// Build a traced pseudoinverse with an explicit relative tolerance.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation::InvalidArgument` when `rtol` is non-finite
    /// or negative, or `Error::Extension` for unsupported dtype.
    ///
    /// # Deferred errors
    ///
    /// SVD convergence and concrete shape failures may be reported during
    /// compile or execution.
    fn pinv_with_rtol(&self, rtol: f64) -> Result<TracedTensor>;

    /// Build a traced vector/matrix norm operation.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for an invalid norm order or axis and
    /// `Error::Extension` for unsupported dtype.
    ///
    /// # Deferred errors
    ///
    /// Symbolic axis and shape checks may be reported during compile or
    /// execution.
    fn norm(&self, ord: Option<f64>, dim: Option<&[usize]>, keepdim: bool) -> Result<TracedTensor>;
}

impl TracedTensorLinalgExt for TracedTensor {
    fn svd(&self) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
        svd(self)
    }

    fn svd_with_options(
        &self,
        options: SvdOptions,
    ) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
        svd_with_options(self, options)
    }

    fn svd_full(&self) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
        svd_full(self)
    }

    fn qr(&self) -> Result<(TracedTensor, TracedTensor)> {
        qr(self)
    }

    fn qr_with_options(&self, options: QrOptions) -> Result<(TracedTensor, TracedTensor)> {
        qr_with_options(self, options)
    }

    fn eigh(&self) -> Result<(TracedTensor, TracedTensor)> {
        eigh(self)
    }

    fn eigh_with_options(&self, options: EighOptions) -> Result<(TracedTensor, TracedTensor)> {
        eigh_with_options(self, options)
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

    fn lstsq(&self, b: &TracedTensor) -> Result<TracedTensor> {
        lstsq(self, b)
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

/// Build a traced singular value decomposition op using default options.
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
///
/// # Errors
///
/// Returns `Error::Validation` for a known invalid rank, matrix shape, or
/// dtype, `Error::Extension` with an unsupported-dtype or non-convergence
/// source when the registered linalg backend cannot construct the operation,
/// and `Error::RuntimeState` when extension registration is unavailable.
///
/// # Deferred errors
///
/// A symbolic matrix or batch-shape mismatch is reported later as
/// `ShapeConstraintViolation` or `ShapeConstraintEvaluation` during compile or
/// execution.
pub fn svd(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
    svd_with_options(a, SvdOptions::default())
}

/// Build a traced singular value decomposition op with explicit options.
///
/// `derivative_eps` regularizes decomposition derivative formulas. It is not a
/// backend SVD solver tolerance.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{SvdGauge, SvdOptions, TracedTensorLinalgExt};
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap();
/// let options = SvdOptions::default()
///     .gauge(SvdGauge::CanonicalPivot)
///     .derivative_eps(1e-10);
/// let (_u, s, _vt) = a.svd_with_options(options).unwrap();
/// assert_eq!(s.rank, 1);
/// ```
///
/// # Errors
///
/// Returns `Error::Validation` when `derivative_eps` is non-finite or
/// non-positive, `Error::Extension` for an unsupported dtype or numerical
/// non-convergence, and `Error::Internal` if the extension output contract is
/// violated.
///
/// # Deferred errors
///
/// Symbolic rank or shape constraints are checked later and can produce
/// `ShapeConstraintViolation` or `ShapeConstraintEvaluation`.
pub fn svd_with_options(
    a: &TracedTensor,
    options: SvdOptions,
) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
    validate_derivative_eps("svd_with_options", options.derivative_eps)?;
    three_outputs(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Svd {
                derivative_eps: options.derivative_eps,
                gauge: options.gauge,
            })),
            &[a],
        )?,
        "svd",
    )
}

/// Build a traced full-matrices singular value decomposition op.
///
/// Unlike [`svd`], the returned factors are square: `U` is `m x m` and `Vh` is
/// `n x n`, while `S` still holds `min(m, n)` singular values. The trailing
/// `n - rank` rows of `Vh` span the right nullspace of the input, so this is
/// the decomposition to use for kernel-basis extraction.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// // A wide 1x2 system: the trailing row of the 2x2 Vh spans the nullspace.
/// let a = TracedTensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0]).unwrap();
/// let (u, s, vh) = a.svd_full().unwrap();
/// assert_eq!(u.rank, 2);
/// assert_eq!(s.rank, 1);
/// assert_eq!(vh.rank, 2);
/// ```
///
/// # Errors
///
/// Returns `Error::Validation` when the input is not a batched matrix
/// (rank `>= 2`), or `Error::RuntimeState` when extension registration is
/// unavailable.
///
/// # Deferred errors
///
/// The active backend returns `Error::Extension` with `ErrorKind::Unsupported`
/// during execution if it does not implement full-matrices SVD (only the CPU
/// faer provider does in this slice). Automatic differentiation is
/// intentionally unsupported for the full variant (see the linalg AD support
/// manifest) and surfaces a typed AD error, not a silent thin-SVD fallback.
pub fn svd_full(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor, TracedTensor)> {
    three_outputs(
        apply(Arc::new(LinalgExtensionOp::new(LinalgOp::SvdFull)), &[a])?,
        "svd_full",
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
///
/// # Errors
///
/// Returns `Error::Validation` for a known invalid rank or matrix shape,
/// `Error::Extension` for an unsupported dtype or numerical failure, and
/// `Error::RuntimeState` when the linalg extension is not registered.
///
/// # Deferred errors
///
/// Unknown matrix or batch dimensions can fail later as
/// `ShapeConstraintViolation` or `ShapeConstraintEvaluation`.
pub fn qr(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor)> {
    qr_with_options(a, QrOptions::default())
}

/// Build a traced QR decomposition op with explicit options.
///
/// `gauge` controls optional sign or phase post-processing.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{QrGauge, QrOptions, TracedTensorLinalgExt};
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap();
/// let (q, r) = a.qr_with_options(QrOptions::default().gauge(QrGauge::PositiveDiagonal)).unwrap();
/// assert_eq!(q.rank, 2);
/// assert_eq!(r.rank, 2);
/// ```
///
/// # Errors
///
/// Returns `Error::Validation` for a known invalid rank or matrix shape,
/// `Error::Extension` for an unsupported dtype or numerical failure, and
/// `Error::Internal` if the extension output contract is violated.
///
/// # Deferred errors
///
/// Symbolic matrix or batch constraints are checked later and can produce
/// `ShapeConstraintViolation` or `ShapeConstraintEvaluation`.
pub fn qr_with_options(
    a: &TracedTensor,
    options: QrOptions,
) -> Result<(TracedTensor, TracedTensor)> {
    two_outputs(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Qr {
                gauge: options.gauge,
            })),
            &[a],
        )?,
        "qr",
    )
}

/// Build a traced Hermitian eigenvalue decomposition op using default options.
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
///
/// # Errors
///
/// Returns `Error::Validation` for a known non-square or invalid-rank input,
/// `Error::Extension` for an unsupported dtype or eigensolver
/// non-convergence, and `Error::RuntimeState` when the extension is not
/// registered.
///
/// # Deferred errors
///
/// Symbolic square-shape constraints can fail later as
/// `ShapeConstraintViolation` or `ShapeConstraintEvaluation`.
pub fn eigh(a: &TracedTensor) -> Result<(TracedTensor, TracedTensor)> {
    eigh_with_options(a, EighOptions::default())
}

/// Build a traced Hermitian eigenvalue decomposition op with explicit options.
///
/// `derivative_eps` regularizes derivative formulas for repeated or nearly
/// repeated eigenvalues. It is not a backend eigensolver tolerance.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::{EighGauge, EighOptions, TracedTensorLinalgExt};
/// use tenferro_runtime::TracedTensor;
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]).unwrap();
/// let (values, _vectors) = a
///     .eigh_with_options(
///         EighOptions::default()
///             .gauge(EighGauge::CanonicalPivot)
///             .derivative_eps(1e-10),
///     )
///     .unwrap();
/// assert_eq!(values.rank, 1);
/// ```
///
/// # Errors
///
/// Returns `Error::Validation` for a known non-square or invalid-rank input,
/// or for non-finite/non-positive `derivative_eps`; `Error::Extension` for an
/// unsupported dtype or eigensolver non-convergence; and `Error::Internal` for
/// an output-count contract violation.
///
/// # Deferred errors
///
/// Symbolic square-shape constraints can fail later as
/// `ShapeConstraintViolation` or `ShapeConstraintEvaluation`.
pub fn eigh_with_options(
    a: &TracedTensor,
    options: EighOptions,
) -> Result<(TracedTensor, TracedTensor)> {
    validate_derivative_eps("eigh_with_options", options.derivative_eps)?;
    two_outputs(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Eigh {
                derivative_eps: options.derivative_eps,
                gauge: options.gauge,
            })),
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
///
/// # Errors
///
/// Returns `Error::Validation` for a known non-square or invalid-rank input,
/// `Error::Extension` for an unsupported dtype or a non-positive-definite
/// matrix, and `Error::RuntimeState` when the extension is not registered.
///
/// # Deferred errors
///
/// Symbolic square-shape constraints can fail later as
/// `ShapeConstraintViolation` or `ShapeConstraintEvaluation`.
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
///
/// # Errors
///
/// Returns `Error::Validation` for a known invalid rank or matrix shape,
/// `Error::Extension` for an unsupported dtype or singular numerical result,
/// and `Error::RuntimeState` when the extension is not registered.
///
/// # Deferred errors
///
/// Symbolic square-shape constraints can fail later as
/// `ShapeConstraintViolation` or `ShapeConstraintEvaluation`.
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
///
/// # Errors
///
/// Returns `Error::Validation` for a known invalid rank or matrix shape,
/// `Error::Extension` for an unsupported dtype or singular numerical result,
/// and `Error::Internal` for an output-count contract violation.
///
/// # Deferred errors
///
/// Symbolic square-shape constraints can fail later as
/// `ShapeConstraintViolation` or `ShapeConstraintEvaluation`.
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
///
/// # Errors
///
/// Returns `Error::Validation` for a known non-square or invalid-rank input,
/// `Error::Extension` for an unsupported dtype or eigensolver
/// non-convergence, and `Error::RuntimeState` when the extension is not
/// registered.
///
/// # Deferred errors
///
/// Symbolic square-shape constraints can fail later as
/// `ShapeConstraintViolation` or `ShapeConstraintEvaluation`.
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
///
/// # Errors
///
/// Returns `Error::Validation` for known incompatible matrix, batch, or dtype
/// metadata, `Error::Extension` for an unsupported dtype or singular system,
/// and `Error::RuntimeState` when the extension is not registered.
///
/// # Deferred errors
///
/// Symbolic matrix and batch constraints can fail later as
/// `ShapeConstraintViolation`, `ShapeConstraintEvaluation`, or
/// `ShapeExpressionEvaluation`.
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

/// Build a traced least-squares solve `argmin_x ||A x - b||_2` for a tall or
/// square, full-column-rank `A`.
///
/// The solution is computed through the thin QR factorization `A = Q R`: since
/// `R` is nonsingular for full column rank, `x = R^{-1} (Qᴴ b)`. This composes
/// existing traced decomposition ops (`qr`, `dot_general`, `triangular_solve`),
/// so, unlike the value-only [`svd_full`], it participates in autodiff through
/// its component rules.
///
/// # Examples
///
/// ```
/// use tenferro_linalg::TracedTensorLinalgExt;
/// use tenferro_runtime::TracedTensor;
///
/// // Overdetermined 3x2 system.
/// let a = TracedTensor::from_vec_col_major(
///     vec![3, 2],
///     vec![1.0_f64, 1.0, 1.0, 0.0, 1.0, 2.0],
/// )
/// .unwrap();
/// let b = TracedTensor::from_vec_col_major(vec![3, 1], vec![1.0_f64, 2.0, 2.0]).unwrap();
/// let x = a.lstsq(&b).unwrap();
/// assert_eq!(x.rank, 2);
/// ```
///
/// # Errors
///
/// Returns `Error::Validation` when `A` or `b` is not a batched matrix
/// (rank `>= 2`), when `A` has a symbolic shape, when `A` is wide
/// (`rows < cols`, underdetermined), or when the dtype is not floating-point or
/// complex. Rank-deficient `A` is not detected here: `R` is singular and the
/// triangular solve yields a non-finite or ill-defined result, so callers must
/// ensure full column rank.
///
/// # Deferred errors
///
/// Backend QR and triangular-solve failures and concrete shape mismatches are
/// reported during compile or execution.
pub fn lstsq(a: &TracedTensor, b: &TracedTensor) -> Result<TracedTensor> {
    ensure_float_or_complex("lstsq", a.dtype)?;
    ensure_min_rank("lstsq", a.rank, 2)?;
    ensure_min_rank("lstsq", b.rank, 2)?;
    let a_shape = require_concrete_shape("lstsq", a)?;
    let (m, n) = (a_shape[0], a_shape[1]);
    if m < n {
        return Err(Error::TensorRuntime(
            tenferro_tensor::Error::invalid_argument(
                "lstsq",
                "shape",
                format!(
                    "lstsq requires a tall or square matrix (rows {m} >= cols {n}); \
                     underdetermined (wide) systems are not supported"
                ),
            ),
        ));
    }
    let (q, r) = qr(a)?;
    let qh = q.conj()?.transpose(&matrix_transpose_perm(q.rank))?;
    let qh_b = matmul_preserve_trailing_batch(&qh, b)?;
    triangular_solve(&r, &qh_b, true, false, false, false)
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
///
/// # Errors
///
/// Returns `Error::Validation` for known incompatible matrix, batch, or dtype
/// metadata, `Error::Extension` for an unsupported dtype or singular system,
/// and `Error::RuntimeState` when the extension is not registered.
///
/// # Deferred errors
///
/// Symbolic matrix and batch constraints can fail later as
/// `ShapeConstraintViolation`, `ShapeConstraintEvaluation`, or
/// `ShapeExpressionEvaluation`.
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
///
/// # Errors
///
/// Returns `Error::Validation` for incompatible matrix, batch, or dtype
/// metadata, `Error::Extension` for an unsupported dtype or singular system,
/// and `Error::RuntimeState` when the extension is not registered.
///
/// # Deferred errors
///
/// Symbolic matrix and batch constraints can fail later as
/// `ShapeConstraintViolation`, `ShapeConstraintEvaluation`, or
/// `ShapeExpressionEvaluation`.
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
///
/// # Errors
///
/// Returns `Error::Validation` for a known non-square or invalid-rank input,
/// `Error::Extension` for an unsupported dtype or singular factorization, and
/// `Error::Internal` if the factorization output contract is violated.
///
/// # Deferred errors
///
/// Symbolic square-shape constraints can fail later as
/// `ShapeConstraintViolation` or `ShapeConstraintEvaluation`.
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
    let sign_u = diag_u.sign()?.reduce_prod(Some(&[0]))?;
    let sign = (&parity * &sign_u)?;
    let logabsdet = diag_u.abs()?.log()?.reduce_sum(Some(&[0]))?;
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
///
/// # Errors
///
/// Returns the same `Error::Validation`, `Error::Extension`, and
/// `Error::RuntimeState` failures as [`slogdet`], including a singular
/// factorization and an invalid matrix shape.
///
/// # Deferred errors
///
/// Symbolic shape checks can later produce `ShapeConstraintViolation`,
/// `ShapeConstraintEvaluation`, or `ShapeExpressionEvaluation`.
pub fn det(a: &TracedTensor) -> Result<TracedTensor> {
    let (sign, logabsdet) = slogdet(a)?;
    &sign * &logabsdet.exp()?
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
///
/// # Errors
///
/// Returns `Error::Validation` when the input is not at least rank two or is
/// not square, `Error::Extension` for an unsupported dtype or singular system,
/// and `Error::RuntimeState` when the extension is not registered.
///
/// # Deferred errors
///
/// A symbolic shape that cannot provide the identity size fails later as
/// `ShapeConstraintEvaluation` or `ShapeExpressionEvaluation`.
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
///
/// # Errors
///
/// Returns `Error::Validation` for a known non-square or invalid-rank input,
/// `Error::Extension` for an unsupported dtype or eigensolver
/// non-convergence, and `Error::RuntimeState` when the extension is not
/// registered.
///
/// # Deferred errors
///
/// Symbolic square-shape constraints can fail later as
/// `ShapeConstraintViolation` or `ShapeConstraintEvaluation`.
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
///
/// # Errors
///
/// Returns `Error::Validation` for a known non-square or invalid-rank input,
/// `Error::Extension` for an unsupported dtype or eigensolver
/// non-convergence, and `Error::RuntimeState` when the extension is not
/// registered.
///
/// # Deferred errors
///
/// Symbolic square-shape constraints can fail later as
/// `ShapeConstraintViolation` or `ShapeConstraintEvaluation`.
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
///
/// # Errors
///
/// Returns `Error::Validation` for an invalid rank, shape, or negative/non-
/// finite `rtol`, `Error::Extension` for unsupported integer or boolean dtypes,
/// numerical non-convergence, or a backend failure, and `Error::RuntimeState`
/// when the extension is not registered.
///
/// # Deferred errors
///
/// Symbolic shapes are materialized by this helper; failures are reported as
/// `ShapeConstraintEvaluation` or `ShapeExpressionEvaluation`.
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
///
/// # Errors
///
/// Returns `Error::Validation` for an invalid rank, shape, or non-finite
/// `rtol`, `Error::Extension` for unsupported integer or boolean dtypes,
/// numerical non-convergence, or a backend failure, and `Error::RuntimeState`
/// when the extension is not registered.
///
/// # Deferred errors
///
/// Symbolic shapes are materialized by this helper; failures are reported as
/// `ShapeConstraintEvaluation` or `ShapeExpressionEvaluation`.
pub fn pinv_with_rtol(a: &TracedTensor, rtol: f64) -> Result<TracedTensor> {
    ensure_float_or_complex("pinv_with_rtol", a.dtype)?;
    require_concrete_shape("pinv_with_rtol", a)?;
    let (u, s, vt) = svd(a)?;
    let abs_s = s.abs()?;
    let s_max = abs_s.reduce_max(Some(&[0]))?;
    let s_max_shape = s_max.concrete_shape()?;
    let threshold_scalar = broadcast_scalar(scalar_real(s.dtype, rtol.max(0.0))?, &s_max_shape)?;
    let threshold = (&s_max * &threshold_scalar)?;
    let s_shape = s.concrete_shape()?;
    let threshold = broadcast_batch_scalar_to_leading_axis(&threshold, &s_shape)?;
    let mask = abs_s.compare(&threshold, CompareDir::Gt)?;
    let mask = mask.convert(s.dtype)?;
    let ones = ones_like(&s)?;
    let neg_mask = (-&mask)?;
    let denom = (&s + &(&ones + &neg_mask)?)?;
    let s_inv = (&mask / &denom)?;

    let v = vt.conj()?.transpose(&matrix_transpose_perm(vt.rank))?;
    let uh = u.conj()?.transpose(&matrix_transpose_perm(u.rank))?;
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
///
/// # Errors
///
/// Returns `Error::Validation` for an invalid axis, rank, or norm order,
/// `Error::Extension` for unsupported integer or boolean dtypes or a backend
/// numerical failure, and `Error::RuntimeState` when the extension is not
/// registered.
///
/// # Deferred errors
///
/// Symbolic shapes needed to restore `keepdim` are evaluated later and can
/// produce `ShapeConstraintEvaluation` or `ShapeExpressionEvaluation`.
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
            let abs = a.abs()?;
            match ord {
                None => frobenius_norm(&abs, &axes)?,
                Some(p) if p == f64::INFINITY => abs.reduce_max(Some(&axes))?,
                Some(p) if p == f64::NEG_INFINITY => abs.reduce_min(Some(&axes))?,
                Some(0.0) => count_nonzero(&abs, &axes)?,
                Some(p) => p_norm(&abs, &axes, p)?,
            }
        }
    };
    restore_keepdim(out, &shape, &axes, keepdim)
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
            crate::error::unsupported_dtype(op, dtype),
        )),
    }
}

fn ensure_min_rank(op: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual < expected {
        return Err(Error::TensorRuntime(tenferro_tensor::Error::rank_mismatch(
            op, expected, actual,
        )));
    }
    Ok(())
}

fn validate_axes(op: &'static str, rank: usize, axes: &[usize]) -> Result<()> {
    for &axis in axes {
        if axis >= rank {
            return Err(Error::TensorRuntime(
                tenferro_tensor::Error::axis_out_of_bounds(op, axis, rank),
            ));
        }
    }
    Ok(())
}

fn require_concrete_shape(op: &'static str, input: &TracedTensor) -> Result<Vec<usize>> {
    input.try_concrete_shape().ok_or_else(|| {
        Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
            op,
            "shape",
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
    squared.reduce_sum(Some(axes))?.sqrt()
}

fn p_norm(abs: &TracedTensor, axes: &[usize], p: f64) -> Result<TracedTensor> {
    if !p.is_finite() || p == 0.0 {
        return Err(Error::invalid_argument(
            "norm",
            ErrorPhase::GraphBuild,
            "p",
            format!("p-norm order must be finite and nonzero, got {p}"),
        ));
    }
    let power = abs.pow(&scalar_real(abs.dtype, p)?)?;
    let inv_p = scalar_real(abs.dtype, 1.0 / p)?;
    power.reduce_sum(Some(axes))?.pow(&inv_p)
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
    let abs = a.abs()?;
    match ord {
        None => frobenius_norm(&abs, &[axis]),
        Some(0.0) => count_nonzero(&abs, &[axis]),
        Some(p) if p == f64::INFINITY => abs.reduce_max(Some(&[axis])),
        Some(p) if p == f64::NEG_INFINITY => abs.reduce_min(Some(&[axis])),
        Some(p) => p_norm(&abs, &[axis], p),
    }
}

fn matrix_norm(a: &TracedTensor, axes: &[usize], ord: Option<f64>) -> Result<TracedTensor> {
    let matrix = move_axes_to_front(a, axes)?;
    let abs = matrix.abs()?;
    match ord {
        None => frobenius_norm(&abs, &[0, 1]),
        Some(p) if p == f64::INFINITY => matrix_row_sum_norm(&abs, true),
        Some(p) if p == f64::NEG_INFINITY => matrix_row_sum_norm(&abs, false),
        Some(1.0) => matrix_col_sum_norm(&abs, true),
        Some(-1.0) => matrix_col_sum_norm(&abs, false),
        Some(2.0) => {
            let singular_values = svd_values(&matrix)?.abs()?;
            singular_values.reduce_max(Some(&[0]))
        }
        Some(-2.0) => {
            let singular_values = svd_values(&matrix)?.abs()?;
            singular_values.reduce_min(Some(&[0]))
        }
        Some(0.0) => count_nonzero(&abs, &[0, 1]),
        Some(p) => p_norm(&abs, &[0, 1], p),
    }
}

fn svd_values(a: &TracedTensor) -> Result<TracedTensor> {
    let (_u, s, _vt) = three_outputs(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Svd {
                derivative_eps: SvdOptions::default().derivative_eps,
                gauge: SvdOptions::default().gauge,
            })),
            &[a],
        )?,
        "svd_values",
    )?;
    Ok(s)
}

fn eigh_values(a: &TracedTensor) -> Result<TracedTensor> {
    let (values, _vectors) = two_outputs(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Eigh {
                derivative_eps: EighOptions::default().derivative_eps,
                gauge: EighOptions::default().gauge,
            })),
            &[a],
        )?,
        "eigh_values",
    )?;
    Ok(values)
}

fn eig_values(a: &TracedTensor) -> Result<TracedTensor> {
    let (values, _vectors) = two_outputs(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::Eig {
                input_dtype: a.dtype,
            })),
            &[a],
        )?,
        "eig_values",
    )?;
    Ok(values)
}

fn scale_matrix_columns(matrix: &TracedTensor, scale: &TracedTensor) -> Result<TracedTensor> {
    let matrix_shape = matrix.concrete_shape()?;
    let scale_shape_input = scale.concrete_shape()?;
    let mut scale_shape = vec![1, scale_shape_input[0]];
    scale_shape.extend_from_slice(&matrix_shape[2..]);
    let dims: Vec<usize> = (0..matrix_shape.len()).collect();
    let scale = scale
        .reshape(&scale_shape)?
        .broadcast_in_dim(&matrix_shape, &dims)?;
    matrix * &scale
}

fn count_nonzero(abs: &TracedTensor, axes: &[usize]) -> Result<TracedTensor> {
    let mask = abs.compare(&zero_scalar(abs.dtype)?, CompareDir::Gt)?;
    mask.convert(abs.dtype)?.reduce_sum(Some(axes))
}

fn matrix_row_sum_norm(abs: &TracedTensor, take_max: bool) -> Result<TracedTensor> {
    let row_sums = abs.reduce_sum(Some(&[1]))?;
    if take_max {
        row_sums.reduce_max(Some(&[0]))
    } else {
        row_sums.reduce_min(Some(&[0]))
    }
}

fn matrix_col_sum_norm(abs: &TracedTensor, take_max: bool) -> Result<TracedTensor> {
    let col_sums = abs.reduce_sum(Some(&[0]))?;
    if take_max {
        col_sums.reduce_max(Some(&[0]))
    } else {
        col_sums.reduce_min(Some(&[0]))
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
) -> Result<TracedTensor> {
    if !keepdim {
        return Ok(reduced);
    }
    let mut kept_shape = original_shape.to_vec();
    for &axis in axes {
        kept_shape[axis] = 1;
    }
    reduced.reshape(&kept_shape)
}

#[cfg(test)]
mod tests {
    use super::p_norm;
    use tenferro_runtime::TracedTensor;

    #[test]
    fn p_norm_rejects_zero_and_non_finite_orders() {
        let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
        let abs = x.abs().unwrap();

        for p in [0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = p_norm(&abs, &[0], p).unwrap_err();
            assert!(
                err.to_string().contains("finite") || err.to_string().contains("nonzero"),
                "expected finite nonzero order error, got {err:?}"
            );
        }
    }
}
