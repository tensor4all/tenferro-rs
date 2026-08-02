use std::sync::{Arc, OnceLock};

use tenferro_ad::error::{Error, Result};
use tenferro_ad::extension::apply_eager_with_extension_session;
use tenferro_ad::EagerTensor;
use tenferro_runtime::{ErrorPhase, ExtensionModule};

use crate::eager_composites;
use crate::extension::{
    execute_linalg_extension_reads, extension_module, validate_derivative_eps, EighOptions,
    LinalgExtensionOp, LinalgOp, QrOptions, SvdOptions,
};

/// Linear algebra extension methods for [`EagerTensor`].
pub trait EagerTensorLinalgExt {
    /// # Errors
    ///
    /// Returns `Error::Validation` for an invalid rank, shape, or dtype,
    /// `Error::Extension` with an unsupported-operation or unsupported-dtype
    /// source when the selected backend cannot execute the decomposition, and
    /// `Error::RuntimeState` when the eager runtime or backend is unavailable.
    fn svd(&self) -> Result<(EagerTensor, EagerTensor, EagerTensor)>;
    /// # Errors
    ///
    /// Returns `Error::Validation` when `derivative_eps` is non-finite or
    /// non-positive, `Error::Extension` for unsupported dtypes or numerical
    /// non-convergence, and `Error::Internal` if the extension violates its
    /// output-count contract.
    fn svd_with_options(
        &self,
        options: SvdOptions,
    ) -> Result<(EagerTensor, EagerTensor, EagerTensor)>;
    /// Full-matrices SVD returning square `U (m x m)` and `Vh (n x n)`, whose
    /// trailing `n - rank` rows span the input's right nullspace.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for an invalid rank, and `Error::Extension`
    /// with an unsupported-operation source when the active backend does not
    /// implement full-matrices SVD (the CPU faer provider supports it; the
    /// LAPACK provider and GPU backends return an unsupported error in this
    /// slice). Automatic differentiation through the full variant is
    /// unsupported and surfaces a typed error rather than a silent thin
    /// fallback.
    fn svd_full(&self) -> Result<(EagerTensor, EagerTensor, EagerTensor)>;
    /// # Errors
    ///
    /// Returns `Error::Validation` for invalid matrix rank or shape,
    /// `Error::Extension` for unsupported dtypes or numerical failure, and
    /// `Error::RuntimeState` when the eager runtime or backend is unavailable.
    fn qr(&self) -> Result<(EagerTensor, EagerTensor)>;
    /// # Errors
    ///
    /// Returns `Error::Validation` for invalid matrix rank or shape,
    /// `Error::Extension` for unsupported dtypes or numerical failure, and
    /// `Error::Internal` if the extension violates its output-count contract.
    fn qr_with_options(&self, options: QrOptions) -> Result<(EagerTensor, EagerTensor)>;
    /// # Errors
    ///
    /// Returns `Error::Validation` for an invalid matrix rank or shape,
    /// `Error::Extension` for an unsupported dtype or singular numerical
    /// result, and `Error::RuntimeState` when execution cannot access its
    /// backend.
    fn lu(&self) -> Result<(EagerTensor, EagerTensor, EagerTensor, EagerTensor)>;
    /// # Errors
    ///
    /// Returns `Error::Validation` for an invalid matrix rank or shape,
    /// `Error::Extension` for unsupported dtypes or singular numerical
    /// results, and `Error::Internal` if the extension violates its output
    /// contract.
    fn full_piv_lu(
        &self,
    ) -> Result<(
        EagerTensor,
        EagerTensor,
        EagerTensor,
        EagerTensor,
        EagerTensor,
    )>;
    /// # Errors
    ///
    /// Returns `Error::Validation` when `a` and `b` have incompatible matrix
    /// or batch shapes, `Error::Extension` for an unsupported dtype or
    /// singular system, and `Error::RuntimeState` when the backend is
    /// unavailable.
    fn full_piv_lu_solve(&self, b: &EagerTensor) -> Result<EagerTensor>;
    /// # Errors
    ///
    /// Returns `Error::Validation` for incompatible matrix, batch, or dtype
    /// metadata, `Error::Extension` for an unsupported dtype or singular
    /// system, and `Error::RuntimeState` when the backend is unavailable.
    fn solve(&self, b: &EagerTensor) -> Result<EagerTensor>;
    /// Least-squares solve `argmin_x ||A x - b||_2` for a tall or square,
    /// full-column-rank `A`, via the thin QR factorization.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` when `A` or `b` is not a matrix (rank
    /// `>= 2`), when `A` is wide (`rows < cols`; underdetermined), or when the
    /// dtype is not floating-point or complex; `Error::Extension` for backend
    /// QR or triangular-solve failures; and `Error::RuntimeState` when the
    /// backend is unavailable. Rank-deficient `A` is not detected and yields an
    /// ill-defined result.
    fn lstsq(&self, b: &EagerTensor) -> Result<EagerTensor>;
    /// # Errors
    ///
    /// Returns `Error::Validation` for a non-square or invalid-rank input,
    /// `Error::Extension` for unsupported dtypes or a non-positive-definite
    /// matrix, and `Error::RuntimeState` when the backend is unavailable.
    fn cholesky(&self) -> Result<EagerTensor>;
    /// # Errors
    ///
    /// Returns `Error::Validation` for a non-square or invalid-rank input,
    /// `Error::Extension` for unsupported dtypes or numerical non-convergence,
    /// and `Error::RuntimeState` when the backend is unavailable.
    fn eigh(&self) -> Result<(EagerTensor, EagerTensor)>;
    /// # Errors
    ///
    /// Returns `Error::Validation` for an invalid rank, shape, or
    /// `derivative_eps`, `Error::Extension` for unsupported dtypes or
    /// non-convergence, and `Error::Internal` for an output-count violation.
    fn eigh_with_options(&self, options: EighOptions) -> Result<(EagerTensor, EagerTensor)>;
    /// # Errors
    ///
    /// Returns `Error::Validation` for a non-square or invalid-rank input,
    /// `Error::Extension` for unsupported dtypes or numerical non-convergence,
    /// and `Error::RuntimeState` when the backend is unavailable.
    fn eig(&self) -> Result<(EagerTensor, EagerTensor)>;
    /// # Errors
    ///
    /// Returns `Error::Validation` for incompatible matrix, batch, or dtype
    /// metadata, `Error::Extension` for unsupported dtypes or a singular
    /// system, and `Error::RuntimeState` when the backend is unavailable.
    fn triangular_solve(
        &self,
        b: &EagerTensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<EagerTensor>;
    /// Return the determinant sign and logarithm of its absolute value.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for a non-square input, `Error::Extension`
    /// for an unsupported dtype or numerical failure, and `Error::Internal`
    /// if a primitive violates its output contract.
    fn slogdet(&self) -> Result<(EagerTensor, EagerTensor)>;
    /// Return the determinant.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation`, `Error::Extension`, `Error::RuntimeState`,
    /// or `Error::Internal` under the conditions documented by [`Self::slogdet`].
    fn det(&self) -> Result<EagerTensor>;
    /// Return the matrix inverse.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for invalid matrix metadata,
    /// `Error::Extension` for an unsupported dtype or singular solve, and
    /// `Error::RuntimeState` when eager execution cannot access its backend.
    fn inv(&self) -> Result<EagerTensor>;
    /// Return Hermitian eigenvalues without eigenvectors.
    ///
    /// # Errors
    ///
    /// Returns the validation, unsupported-dtype, numerical-convergence,
    /// runtime-state, or output-contract errors reported by [`Self::eigh`].
    fn eigvalsh(&self) -> Result<EagerTensor>;
    /// Return general eigenvalues without eigenvectors.
    ///
    /// # Errors
    ///
    /// Returns the validation, unsupported-dtype, numerical-convergence,
    /// runtime-state, or output-contract errors reported by [`Self::eig`].
    fn eigvals(&self) -> Result<EagerTensor>;
    /// Return the Moore-Penrose pseudoinverse with the default tolerance.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for invalid matrix metadata,
    /// `Error::Extension` for unsupported SVD execution, and
    /// `Error::Internal` for an unexpected decomposition output contract.
    fn pinv(&self) -> Result<EagerTensor>;
    /// Return the Moore-Penrose pseudoinverse with an explicit relative tolerance.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` when `rtol` is negative or non-finite, plus
    /// the `Error::Extension` and output-contract errors reported by [`Self::pinv`].
    fn pinv_with_rtol(&self, rtol: f64) -> Result<EagerTensor>;
    /// Return a vector, matrix, or tensor norm.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for an unsupported order/dimension
    /// combination or invalid axes, `Error::Extension` when a required matrix
    /// decomposition is unsupported, and eager runtime/backend failures.
    fn norm(&self, ord: Option<f64>, dim: Option<&[usize]>, keepdim: bool) -> Result<EagerTensor>;
}

impl EagerTensorLinalgExt for EagerTensor {
    fn svd(&self) -> Result<(EagerTensor, EagerTensor, EagerTensor)> {
        svd(self)
    }

    fn svd_with_options(
        &self,
        options: SvdOptions,
    ) -> Result<(EagerTensor, EagerTensor, EagerTensor)> {
        svd_with_options(self, options)
    }

    fn svd_full(&self) -> Result<(EagerTensor, EagerTensor, EagerTensor)> {
        svd_full(self)
    }

    fn qr(&self) -> Result<(EagerTensor, EagerTensor)> {
        qr(self)
    }

    fn qr_with_options(&self, options: QrOptions) -> Result<(EagerTensor, EagerTensor)> {
        qr_with_options(self, options)
    }

    fn lu(&self) -> Result<(EagerTensor, EagerTensor, EagerTensor, EagerTensor)> {
        lu(self)
    }

    fn full_piv_lu(
        &self,
    ) -> Result<(
        EagerTensor,
        EagerTensor,
        EagerTensor,
        EagerTensor,
        EagerTensor,
    )> {
        full_piv_lu(self)
    }

    fn full_piv_lu_solve(&self, b: &EagerTensor) -> Result<EagerTensor> {
        full_piv_lu_solve(self, b)
    }

    fn solve(&self, b: &EagerTensor) -> Result<EagerTensor> {
        solve(self, b)
    }

    fn lstsq(&self, b: &EagerTensor) -> Result<EagerTensor> {
        eager_composites::lstsq(self, b)
    }

    fn cholesky(&self) -> Result<EagerTensor> {
        cholesky(self)
    }

    fn eigh(&self) -> Result<(EagerTensor, EagerTensor)> {
        eigh(self)
    }

    fn eigh_with_options(&self, options: EighOptions) -> Result<(EagerTensor, EagerTensor)> {
        eigh_with_options(self, options)
    }

    fn eig(&self) -> Result<(EagerTensor, EagerTensor)> {
        eig(self)
    }

    fn triangular_solve(
        &self,
        b: &EagerTensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<EagerTensor> {
        triangular_solve(self, b, left_side, lower, transpose_a, unit_diagonal)
    }

    fn slogdet(&self) -> Result<(EagerTensor, EagerTensor)> {
        eager_composites::slogdet(self)
    }

    fn det(&self) -> Result<EagerTensor> {
        eager_composites::det(self)
    }

    fn inv(&self) -> Result<EagerTensor> {
        eager_composites::inv(self)
    }

    fn eigvalsh(&self) -> Result<EagerTensor> {
        eager_composites::eigvalsh(self)
    }

    fn eigvals(&self) -> Result<EagerTensor> {
        eager_composites::eigvals(self)
    }

    fn pinv(&self) -> Result<EagerTensor> {
        eager_composites::pinv(self)
    }

    fn pinv_with_rtol(&self, rtol: f64) -> Result<EagerTensor> {
        eager_composites::pinv_with_rtol(self, rtol)
    }

    fn norm(&self, ord: Option<f64>, dim: Option<&[usize]>, keepdim: bool) -> Result<EagerTensor> {
        eager_composites::norm(self, ord, dim, keepdim)
    }
}

fn apply_linalg_eager(op: LinalgOp, inputs: &[&EagerTensor]) -> Result<Vec<EagerTensor>> {
    let op = Arc::new(LinalgExtensionOp::new(op));
    let execute_op = Arc::clone(&op);
    let module = eager_cpu_extension_module()?;
    apply_eager_with_extension_session(op, inputs, module, move |_op, input_reads, ctx| {
        execute_linalg_extension_reads(&execute_op, input_reads, ctx)
    })
}

fn eager_cpu_extension_module() -> Result<Arc<dyn ExtensionModule>> {
    static MODULE: OnceLock<Arc<dyn ExtensionModule>> = OnceLock::new();
    if let Some(module) = MODULE.get() {
        return Ok(Arc::clone(module));
    }

    let engine_id = tenferro_cpu::runtime_engine_id().map_err(eager_runtime_config_error)?;
    let module = extension_module::<tenferro_cpu::CpuBackend>(engine_id)
        .map_err(eager_runtime_config_error)?;
    let _ = MODULE.set(Arc::clone(&module));
    Ok(MODULE.get().cloned().unwrap_or(module))
}

fn eager_runtime_config_error(source: tenferro_runtime::RuntimeConfigError) -> Error {
    Error::runtime_state_source(
        "tenferro_linalg::eager_extension_module",
        ErrorPhase::Execution,
        source,
    )
}

/// Singular value decomposition for eager tensors.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// use tenferro_linalg::EagerTensorLinalgExt;
///
/// let ctx = EagerRuntime::new().unwrap();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]).unwrap(),
///     ctx,
/// ).unwrap();
/// let (_u, s, _vt) = a.svd()?;
/// assert_eq!(s.shape(), &[2]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
///
/// # Errors
///
/// Returns `Error::Validation` for an invalid rank, matrix shape, or dtype,
/// `Error::Extension` with an unsupported-dtype or non-convergence source when
/// the backend cannot compute the decomposition, and `Error::RuntimeState`
/// when the eager runtime or backend is unavailable.
pub fn svd(a: &EagerTensor) -> Result<(EagerTensor, EagerTensor, EagerTensor)> {
    svd_with_options(a, SvdOptions::default())
}

/// Singular value decomposition for eager tensors with explicit options.
///
/// `derivative_eps` regularizes decomposition derivative formulas. It is not a
/// backend SVD solver tolerance.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// use tenferro_linalg::{EagerTensorLinalgExt, SvdGauge, SvdOptions};
///
/// let ctx = EagerRuntime::new().unwrap();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]).unwrap(),
///     ctx,
/// ).unwrap();
/// let options = SvdOptions::default()
///     .gauge(SvdGauge::CanonicalPivot)
///     .derivative_eps(1.0e-10);
/// let (_u, s, _vt) = a.svd_with_options(options)?;
/// assert_eq!(s.shape(), &[2]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
///
/// # Errors
///
/// Returns `Error::Validation` when `derivative_eps` is non-finite or
/// non-positive, `Error::Extension` for unsupported dtypes or numerical
/// non-convergence, and `Error::Internal` if the extension returns an
/// unexpected number of outputs.
pub fn svd_with_options(
    a: &EagerTensor,
    options: SvdOptions,
) -> Result<(EagerTensor, EagerTensor, EagerTensor)> {
    validate_derivative_eps("svd_with_options", options.derivative_eps)?;
    let mut outputs = apply_linalg_eager(
        LinalgOp::Svd {
            derivative_eps: options.derivative_eps,
            gauge: options.gauge,
        },
        &[a],
    )?
    .into_iter();
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

/// Full-matrices singular value decomposition for eager tensors.
///
/// Returns `(U, S, Vh)` with square `U` (`m x m`) and `Vh` (`n x n`); `S` holds
/// `min(m, n)` singular values. The trailing `n - rank` rows of `Vh` span the
/// input's right nullspace, which the thin [`svd`] cannot represent.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// use tenferro_linalg::EagerTensorLinalgExt;
///
/// let ctx = EagerRuntime::new().unwrap();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0]).unwrap(),
///     ctx,
/// ).unwrap();
/// // Full SVD is provided by the CPU faer backend; other providers (e.g. the
/// // LAPACK provider) report it as unsupported, so guard on the result.
/// if let Ok((u, s, vh)) = a.svd_full() {
///     assert_eq!(u.shape(), &[1, 1]);
///     assert_eq!(s.shape(), &[1]);
///     assert_eq!(vh.shape(), &[2, 2]);
/// }
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
///
/// # Errors
///
/// Returns `Error::Validation` for an invalid rank and `Error::Extension` with
/// an unsupported-operation source when the active backend does not implement
/// full-matrices SVD (the CPU faer provider supports it; the LAPACK provider
/// and GPU backends are unsupported in this slice). AD through the full variant
/// is unsupported and surfaces a typed error, not a silent thin fallback.
pub fn svd_full(a: &EagerTensor) -> Result<(EagerTensor, EagerTensor, EagerTensor)> {
    let mut outputs = apply_linalg_eager(LinalgOp::SvdFull, &[a])?.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(u), Some(s), Some(vh), None) => Ok((u, s, vh)),
        _ => Err(Error::Internal(
            "svd_full eager op returned an unexpected number of outputs".to_string(),
        )),
    }
}

/// QR decomposition for eager tensors.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// use tenferro_linalg::EagerTensorLinalgExt;
///
/// let ctx = EagerRuntime::new().unwrap();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap(),
///     ctx,
/// ).unwrap();
/// let (q, r) = a.qr()?;
/// assert_eq!(q.shape(), &[2, 2]);
/// assert_eq!(r.shape(), &[2, 2]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
///
/// # Errors
///
/// Returns `Error::Validation` for an invalid rank or matrix shape,
/// `Error::Extension` for an unsupported dtype or numerical failure, and
/// `Error::RuntimeState` when the eager runtime or backend is unavailable.
pub fn qr(a: &EagerTensor) -> Result<(EagerTensor, EagerTensor)> {
    qr_with_options(a, QrOptions::default())
}

/// QR decomposition for eager tensors with explicit options.
///
/// `gauge` controls optional sign or phase post-processing.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// use tenferro_linalg::{EagerTensorLinalgExt, QrGauge, QrOptions};
///
/// let ctx = EagerRuntime::new().unwrap();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap(),
///     ctx,
/// ).unwrap();
/// let (q, r) = a.qr_with_options(QrOptions::default().gauge(QrGauge::PositiveDiagonal))?;
/// assert_eq!(q.shape(), &[2, 2]);
/// assert_eq!(r.shape(), &[2, 2]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
///
/// # Errors
///
/// Returns `Error::Validation` for an invalid rank or matrix shape,
/// `Error::Extension` for an unsupported dtype or numerical failure, and
/// `Error::Internal` if the extension returns an unexpected number of outputs.
pub fn qr_with_options(a: &EagerTensor, options: QrOptions) -> Result<(EagerTensor, EagerTensor)> {
    two_outputs(
        apply_linalg_eager(
            LinalgOp::Qr {
                gauge: options.gauge,
            },
            &[a],
        )?,
        "qr",
    )
}

/// LU factorization for eager tensors.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// use tenferro_linalg::EagerTensorLinalgExt;
///
/// let ctx = EagerRuntime::new().unwrap();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 1.0, 1.0, 0.0]).unwrap(),
///     ctx,
/// ).unwrap();
/// let (_p, l, u, parity) = a.lu()?;
/// assert_eq!(l.shape(), &[2, 2]);
/// assert_eq!(u.shape(), &[2, 2]);
/// assert_eq!(parity.shape(), &[] as &[usize]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
///
/// # Errors
///
/// Returns `Error::Validation` for an invalid rank or matrix shape,
/// `Error::Extension` for an unsupported dtype or singular numerical result,
/// and `Error::RuntimeState` when the eager runtime or backend is unavailable.
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
/// Returns `(P, L, U, Q, parity)` with reconstruction convention
/// `A = P^T * L * U * Q`, equivalently `P * A * Q^T = L * U`. `parity` is a
/// scalar real tensor containing `+1` or `-1`: `F32` for `F32`/`C32` inputs and
/// `F64` for `F64`/`C64` inputs.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// use tenferro_linalg::EagerTensorLinalgExt;
///
/// let ctx = EagerRuntime::new().unwrap();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0]).unwrap(),
///     ctx,
/// ).unwrap();
/// let (p, _l, _u, q, parity) = a.full_piv_lu()?;
/// assert_eq!(p.shape(), &[2, 2]);
/// assert_eq!(q.shape(), &[2, 2]);
/// assert_eq!(parity.shape(), &[] as &[usize]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
///
/// # Errors
///
/// Returns `Error::Validation` for an invalid rank or matrix shape,
/// `Error::Extension` for an unsupported dtype or singular numerical result,
/// and `Error::Internal` if the extension returns an unexpected number of
/// outputs.
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
/// use tenferro_linalg::EagerTensorLinalgExt;
///
/// let ctx = EagerRuntime::new().unwrap();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0]).unwrap(),
///     ctx.clone(),
/// ).unwrap();
/// let b = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 1], vec![-1.0_f64, 5.0]).unwrap(),
///     ctx,
/// ).unwrap();
/// let x = a.full_piv_lu_solve(&b)?;
/// assert_eq!(x.shape(), &[2, 1]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
///
/// # Errors
///
/// Returns `Error::Validation` when `a` and `b` have incompatible matrix or
/// batch shapes, `Error::Extension` for an unsupported dtype or singular
/// system, and `Error::RuntimeState` when the backend is unavailable.
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
/// use tenferro_linalg::EagerTensorLinalgExt;
///
/// let ctx = EagerRuntime::new().unwrap();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap(),
///     ctx.clone(),
/// ).unwrap();
/// let b = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 8.0]).unwrap(),
///     ctx,
/// ).unwrap();
/// let x = a.solve(&b)?;
/// assert_eq!(x.shape(), &[2, 1]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
///
/// # Errors
///
/// Returns `Error::Validation` for incompatible matrix, batch, or dtype
/// metadata, `Error::Extension` for an unsupported dtype or singular system,
/// and `Error::RuntimeState` when the backend is unavailable.
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
/// use tenferro_linalg::EagerTensorLinalgExt;
///
/// let ctx = EagerRuntime::new().unwrap();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap(),
///     ctx,
/// ).unwrap();
/// let l = a.cholesky()?;
/// assert_eq!(l.shape(), &[2, 2]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
///
/// # Errors
///
/// Returns `Error::Validation` for a non-square or invalid-rank input,
/// `Error::Extension` for an unsupported dtype or a non-positive-definite
/// matrix, and `Error::RuntimeState` when the backend is unavailable.
pub fn cholesky(a: &EagerTensor) -> Result<EagerTensor> {
    one_output(apply_linalg_eager(LinalgOp::Cholesky, &[a])?, "cholesky")
}

/// Hermitian eigenvalue decomposition for eager tensors.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// use tenferro_linalg::EagerTensorLinalgExt;
///
/// let ctx = EagerRuntime::new().unwrap();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]).unwrap(),
///     ctx,
/// ).unwrap();
/// let (values, vectors) = a.eigh()?;
/// assert_eq!(values.shape(), &[2]);
/// assert_eq!(vectors.shape(), &[2, 2]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
///
/// # Errors
///
/// Returns `Error::Validation` for a non-square or invalid-rank input,
/// `Error::Extension` for an unsupported dtype or numerical non-convergence,
/// and `Error::RuntimeState` when the backend is unavailable.
pub fn eigh(a: &EagerTensor) -> Result<(EagerTensor, EagerTensor)> {
    eigh_with_options(a, EighOptions::default())
}

/// Hermitian eigenvalue decomposition for eager tensors with explicit options.
///
/// `derivative_eps` regularizes derivative formulas for repeated or nearly
/// repeated eigenvalues. It is not a backend eigensolver tolerance.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// use tenferro_linalg::{EagerTensorLinalgExt, EighGauge, EighOptions};
///
/// let ctx = EagerRuntime::new().unwrap();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]).unwrap(),
///     ctx,
/// ).unwrap();
/// let (values, vectors) = a
///     .eigh_with_options(
///         EighOptions::default()
///             .gauge(EighGauge::CanonicalPivot)
///             .derivative_eps(1.0e-10),
///     )?;
/// assert_eq!(values.shape(), &[2]);
/// assert_eq!(vectors.shape(), &[2, 2]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
///
/// # Errors
///
/// Returns `Error::Validation` for an invalid rank, shape, or
/// `derivative_eps`, `Error::Extension` for unsupported dtypes or numerical
/// non-convergence, and `Error::Internal` for an output-count violation.
pub fn eigh_with_options(
    a: &EagerTensor,
    options: EighOptions,
) -> Result<(EagerTensor, EagerTensor)> {
    validate_derivative_eps("eigh_with_options", options.derivative_eps)?;
    two_outputs(
        apply_linalg_eager(
            LinalgOp::Eigh {
                derivative_eps: options.derivative_eps,
                gauge: options.gauge,
            },
            &[a],
        )?,
        "eigh",
    )
}

/// General eigenvalue decomposition for eager tensors.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// use tenferro_linalg::EagerTensorLinalgExt;
///
/// let ctx = EagerRuntime::new().unwrap();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]).unwrap(),
///     ctx,
/// ).unwrap();
/// let (values, vectors) = a.eig()?;
/// assert_eq!(values.shape(), &[2]);
/// assert_eq!(vectors.shape(), &[2, 2]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
///
/// # Errors
///
/// Returns `Error::Validation` for a non-square or invalid-rank input,
/// `Error::Extension` for an unsupported dtype or numerical non-convergence,
/// and `Error::RuntimeState` when the backend is unavailable.
pub fn eig(a: &EagerTensor) -> Result<(EagerTensor, EagerTensor)> {
    two_outputs(
        apply_linalg_eager(
            LinalgOp::Eig {
                input_dtype: a.dtype(),
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
/// use tenferro_linalg::EagerTensorLinalgExt;
///
/// let ctx = EagerRuntime::new().unwrap();
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 1.0, 3.0]).unwrap(),
///     ctx.clone(),
/// ).unwrap();
/// let b = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 7.0]).unwrap(),
///     ctx,
/// ).unwrap();
/// let x = a.triangular_solve(&b, true, true, false, false)?;
/// assert_eq!(x.shape(), &[2, 1]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
///
/// # Errors
///
/// Returns `Error::Validation` for incompatible matrix, batch, or dtype
/// metadata, `Error::Extension` for an unsupported dtype or singular system,
/// and `Error::RuntimeState` when the backend is unavailable.
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
