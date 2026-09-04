use std::ops::Range;

use tenferro_tensor::{BackendSession, Tensor, TensorRead, TensorWrite};

/// Compact provider-neutral Householder QR payload used by backend hooks.
#[doc(hidden)]
#[derive(Debug)]
pub struct CompactQrResult {
    pub(crate) packed: Tensor,
    pub(crate) coeff: Tensor,
}

pub(crate) use crate::error::unsupported_dtype;
use crate::extension::{
    apply_eigh_gauge, apply_qr_gauge, apply_svd_gauge, validate_derivative_eps, EighOptions,
    QrOptions, SvdOptions,
};
use crate::RankRevealingQrOptions;

/// Backend surface required by the linalg extension runtime.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{with_cpu_exec_session, CpuBackend, CpuExecSession};
/// use tenferro_linalg::backend::LinalgBackend;
/// use tenferro_tensor::BackendSessionHost;
///
/// fn assert_linalg_backend<B: LinalgBackend>() {}
///
/// assert_linalg_backend::<CpuExecSession<'static>>();
/// let mut host = CpuBackend::new();
/// host.with_backend_session(|session| {
///     with_cpu_exec_session(session, |_backend| ())
///         .expect("CpuBackend must expose a CpuExecSession");
/// });
/// ```
pub trait LinalgBackend: BackendSession {
    /// Compute a Cholesky factorization.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for non-matrix, non-square, or unsupported
    /// input dtypes; `Error::Extension` with `ErrorKind::NumericalFailure`
    /// when the matrix is not positive definite; or a typed backend source
    /// when the provider cannot execute the factorization.
    fn cholesky(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor>;

    /// Solve a triangular linear system with explicit side, triangle,
    /// transpose, and unit-diagonal flags.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for incompatible matrix/rhs shapes, rank,
    /// or dtype; `Error::Extension` with `ErrorKind::NumericalFailure` for a
    /// singular or zero-diagonal system; or a typed backend source for a
    /// provider failure.
    fn triangular_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> tenferro_tensor::Result<Tensor>;

    // INVARIANT: the six flags are the public triangular-solve contract and mirror the owned hook.
    #[allow(clippy::too_many_arguments)]
    /// Solve a triangular linear system from tensor read targets.
    ///
    /// Backends may canonicalize the inputs inside the same placement family,
    /// but must not silently transfer between CPU and GPU memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
    /// use tenferro_linalg::LinalgBackend;
    /// use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    ///
    /// let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 1.0, 3.0])?;
    /// let b = Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 9.0])?;
    /// let mut host = CpuBackend::new();
    /// let x = host.with_backend_session(|session| {
    ///     with_cpu_exec_session(session, |backend| {
    ///         backend.triangular_solve_read(
    ///             TensorRead::from_tensor(&a),
    ///             TensorRead::from_tensor(&b),
    ///             true,
    ///             false,
    ///             false,
    ///             false,
    ///         )
    ///     })
    ///     .expect("CpuBackend must expose a CpuExecSession")
    /// })?;
    /// let Tensor::F64(x) = x else { unreachable!("F64 inputs return F64 output") };
    /// assert_eq!(x.host_data()?, &[0.5, 3.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The default implementation returns `Error::Unsupported` because the
    /// backend does not accept tensor read targets. Implementations may return
    /// `Error::Validation` for incompatible shapes or dtypes,
    /// `Error::RuntimeState` for invalid placement, `Error::Extension` for a
    /// singular system, or a typed backend-source error.
    fn triangular_solve_read(
        &mut self,
        _a: TensorRead<'_>,
        _b: TensorRead<'_>,
        _left_side: bool,
        _lower: bool,
        _transpose_a: bool,
        _unit_diagonal: bool,
    ) -> tenferro_tensor::Result<Tensor> {
        Err(tenferro_tensor::Error::unsupported(
            "triangular_solve",
            "backend does not accept tensor reads at this execution boundary",
        ))
    }

    /// Compute public LU outputs `(P, L, U, parity)`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` when the input is not a supported matrix or
    /// dtype, and `Error::Extension` or a typed backend source when LU
    /// execution or pivot storage fails.
    fn lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;

    #[doc(hidden)]
    fn lu_factor(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        Err(tenferro_tensor::Error::unsupported(
            "lu_factor",
            format!(
                "backend {} does not implement internal packed LU factorization",
                std::any::type_name::<Self>()
            ),
        ))
    }

    /// Compute complete-pivot LU outputs `(P, L, U, Q, parity)`.
    ///
    /// The reconstruction convention is `A = P^T * L * U * Q`, equivalently
    /// `P * A * Q^T = L * U`. `parity` is a scalar real tensor containing
    /// `+1` or `-1`: `F32` for `F32`/`C32` inputs and `F64` for `F64`/`C64`
    /// inputs.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for an invalid rank, square-shape
    /// requirement, or dtype, and `Error::Extension` or a typed backend source
    /// when complete-pivot factorization cannot be executed.
    fn full_piv_lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;

    /// Solve a linear system through the complete-pivot LU path.
    ///
    /// With `transpose_a = false`, this solves `A * x = b`. With
    /// `transpose_a = true`, this solves `A^T * x = b`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for incompatible coefficient/rhs shapes or
    /// dtypes, `Error::Extension` with `ErrorKind::NumericalFailure` for a
    /// singular system, or a typed backend source for provider failure.
    fn full_piv_lu_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        transpose_a: bool,
    ) -> tenferro_tensor::Result<Tensor>;

    /// Compute public SVD outputs `(U, S, Vt)`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for an unsupported rank or dtype and a
    /// typed `Error::Extension` or backend source when the solver fails.
    fn svd(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;

    /// Compute public SVD outputs `(U, S, Vt)` with explicit options.
    ///
    /// `derivative_eps` is validated for API consistency, but concrete backend
    /// execution does not perform AD. `gauge` controls optional singular-vector
    /// post-processing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
    /// use tenferro_linalg::{LinalgBackend, SvdGauge, SvdOptions};
    /// use tenferro_tensor::{BackendSessionHost, Tensor};
    ///
    /// let input = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0])?;
    /// let mut host = CpuBackend::new();
    /// let outputs = host.with_backend_session(|session| {
    ///     with_cpu_exec_session(session, |backend| {
    ///         backend.svd_with_options(
    ///             &input,
    ///             SvdOptions::default().gauge(SvdGauge::CanonicalPivot),
    ///         )
    ///     })
    ///     .expect("CpuBackend must expose a CpuExecSession")
    /// })?;
    /// assert_eq!(outputs[1].shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] containing
    /// [`tenferro_tensor::ValidationError::InvalidArgument`] when
    /// `derivative_eps` is non-finite or non-positive, or when canonical gauge
    /// output metadata is malformed. It can return
    /// [`tenferro_tensor::Error::Validation`] with
    /// [`tenferro_tensor::ValidationError::RankMismatch`],
    /// [`tenferro_tensor::ValidationError::ShapeMismatch`], or
    /// [`tenferro_tensor::ValidationError::DTypeMismatch`] for the input
    /// or generated outputs, [`tenferro_tensor::Error::Extension`] with the
    /// typed `tenferro_linalg::Error::UnsupportedDType` or
    /// `NonConvergence` source, [`tenferro_tensor::Error::BackendSource`] for
    /// provider calls, and [`tenferro_tensor::Error::RuntimeState`] for
    /// placement failures. A CPU provider that was not compiled is reported
    /// as [`tenferro_tensor::ValidationError::InvalidArgument`] on the
    /// provider configuration.
    fn svd_with_options(
        &mut self,
        input: &Tensor,
        options: SvdOptions,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        validate_derivative_eps("svd_with_options", options.derivative_eps)?;
        let mut outputs = self.svd(input)?;
        apply_svd_gauge(options.gauge, &mut outputs)?;
        Ok(outputs)
    }

    /// Compute public full-matrices SVD outputs `(U, S, Vt)` with `U` shaped
    /// `m x m` and `Vt` shaped `n x n`, so the trailing `Vt` rows span the
    /// input's right nullspace.
    ///
    /// # Errors
    ///
    /// The default implementation returns `Error::Unsupported`: a backend that
    /// does not implement the full variant reports it explicitly rather than
    /// silently falling back to the thin decomposition. Implementing backends
    /// may additionally return `Error::Validation` for an unsupported rank or
    /// dtype and a typed backend source when the solver fails.
    fn svd_full(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        Err(tenferro_tensor::Error::unsupported(
            "svd_full",
            format!(
                "backend {} does not implement full-matrices SVD",
                std::any::type_name::<Self>()
            ),
        ))
    }

    #[doc(hidden)]
    fn svd_values(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        Err(tenferro_tensor::Error::unsupported(
            "svd_values",
            format!(
                "backend {} does not implement internal singular-values-only decomposition",
                std::any::type_name::<Self>()
            ),
        ))
    }

    /// Compute a singular value decomposition from a tensor read target.
    ///
    /// Backends may canonicalize the input inside the same placement family, but
    /// must not silently transfer between CPU and GPU memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::LinalgBackend;
    /// use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
    /// use tenferro_tensor::{BackendSessionHost, TensorRead, TensorView, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(
    ///     vec![2, 2],
    ///     vec![1.0, 0.0, 0.0, 2.0],
    /// )?;
    /// let mut host = CpuBackend::new();
    /// let outputs = host.with_backend_session(|session| {
    ///     with_cpu_exec_session(session, |backend| {
    ///         backend.svd_read(TensorRead::from_view(TensorView::F64(input.as_view())))
    ///     })
    ///     .expect("CpuBackend must expose a CpuExecSession")
    /// })?;
    /// assert_eq!(outputs[1].shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The default implementation returns `Error::Unsupported` because the
    /// backend does not accept tensor read targets; an implementation may instead
    /// return validation or typed backend-source errors after canonicalizing
    /// the view.
    fn svd_read(&mut self, _input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        Err(tenferro_tensor::Error::unsupported(
            "svd",
            "backend does not accept tensor reads at this execution boundary",
        ))
    }

    #[doc(hidden)]
    fn svd_values_read(&mut self, _input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        Err(tenferro_tensor::Error::unsupported(
            "svd_values",
            "backend does not implement borrowed singular-values-only decomposition",
        ))
    }

    /// Compute public QR outputs `(Q, R)`.
    ///
    /// QR is thin: for an `m x n` input, `Q` has shape `m x min(m, n)` and
    /// `R` has shape `min(m, n) x n`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for an unsupported rank, shape, or dtype,
    /// and a typed `Error::Extension` or backend source when QR execution
    /// fails.
    fn qr(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;

    /// Compute public QR outputs `(Q, R)` with explicit options.
    ///
    /// `gauge` controls optional sign or phase post-processing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
    /// use tenferro_linalg::{LinalgBackend, QrGauge, QrOptions};
    /// use tenferro_tensor::{BackendSessionHost, Tensor};
    ///
    /// let input = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0])?;
    /// let mut host = CpuBackend::new();
    /// let outputs = host.with_backend_session(|session| {
    ///     with_cpu_exec_session(session, |backend| {
    ///         backend.qr_with_options(
    ///             &input,
    ///             QrOptions::default().gauge(QrGauge::PositiveDiagonal),
    ///         )
    ///     })
    ///     .expect("CpuBackend must expose a CpuExecSession")
    /// })?;
    /// assert_eq!(outputs[0].shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] containing
    /// [`tenferro_tensor::ValidationError::RankMismatch`] or
    /// [`tenferro_tensor::ValidationError::ShapeMismatch`] for an invalid
    /// matrix input, or [`tenferro_tensor::ValidationError::InvalidArgument`]
    /// for malformed gauge output metadata, checked size arithmetic, or an
    /// unavailable compiled provider. A mismatched generated `Q`/`R` dtype is reported as
    /// [`tenferro_tensor::ValidationError::DTypeMismatch`]. Provider
    /// unsupported dtype or numerical rejection is
    /// [`tenferro_tensor::Error::Extension`] with a typed linalg source, while
    /// provider failures use [`tenferro_tensor::Error::BackendSource`] and a
    /// backend-resident input uses [`tenferro_tensor::Error::RuntimeState`].
    fn qr_with_options(
        &mut self,
        input: &Tensor,
        options: QrOptions,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        let mut outputs = self.qr(input)?;
        apply_qr_gauge(options.gauge, &mut outputs)?;
        Ok(outputs)
    }

    /// Compute public QR outputs `(Q, R)` from a tensor read target.
    ///
    /// Backends may canonicalize the input inside the same placement family, but
    /// must not silently transfer between CPU and GPU memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::LinalgBackend;
    /// use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
    /// use tenferro_tensor::{BackendSessionHost, TensorRead, TensorView, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(
    ///     vec![2, 2],
    ///     vec![1.0, 0.0, 0.0, 2.0],
    /// )?;
    /// let mut host = CpuBackend::new();
    /// let outputs = host.with_backend_session(|session| {
    ///     with_cpu_exec_session(session, |backend| {
    ///         backend.qr_read(TensorRead::from_view(TensorView::F64(input.as_view())))
    ///     })
    ///     .expect("CpuBackend must expose a CpuExecSession")
    /// })?;
    /// assert_eq!(outputs[0].shape(), &[2, 2]);
    /// assert_eq!(outputs[1].shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The default implementation returns `Error::Unsupported` because the
    /// backend does not accept tensor read targets; implementations may return
    /// validation or typed backend-source errors.
    fn qr_read(&mut self, _input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        Err(tenferro_tensor::Error::unsupported(
            "qr",
            "backend does not accept tensor reads at this execution boundary",
        ))
    }

    /// Compute public QR outputs `(Q, R)` from a tensor read target with options.
    ///
    /// Device backends override this hook to keep gauge processing in the input
    /// placement. The default is appropriate for host backends.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
    /// use tenferro_linalg::{LinalgBackend, QrGauge, QrOptions};
    /// use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    ///
    /// let input = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0])?;
    /// let mut host = CpuBackend::new();
    /// let outputs = host.with_backend_session(|session| {
    ///     with_cpu_exec_session(session, |backend| {
    ///         backend.qr_with_options_read(
    ///             TensorRead::from_tensor(&input),
    ///             QrOptions::default().gauge(QrGauge::PositiveDiagonal),
    ///         )
    ///     })
    ///     .expect("CpuBackend must expose a CpuExecSession")
    /// })?;
    /// assert_eq!(outputs[0].shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns the validation, unsupported-dtype, numerical, placement, or
    /// typed backend/provider errors from [`LinalgBackend::qr_read`], plus
    /// gauge metadata and host-access errors from the default host gauge path.
    fn qr_with_options_read(
        &mut self,
        input: TensorRead<'_>,
        options: QrOptions,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        let mut outputs = self.qr_read(input)?;
        apply_qr_gauge(options.gauge, &mut outputs)?;
        Ok(outputs)
    }

    #[doc(hidden)]
    fn rank_revealing_qr(
        &mut self,
        _input: &Tensor,
        _options: RankRevealingQrOptions,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        Err(tenferro_tensor::Error::unsupported(
            "rank_revealing_qr",
            "backend does not implement rank-revealing QR",
        ))
    }

    #[doc(hidden)]
    fn rank_revealing_qr_read(
        &mut self,
        _input: TensorRead<'_>,
        _options: RankRevealingQrOptions,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        Err(tenferro_tensor::Error::unsupported(
            "rank_revealing_qr",
            "backend does not implement borrowed rank-revealing QR",
        ))
    }

    #[doc(hidden)]
    fn householder_qr(&mut self, _input: &Tensor) -> tenferro_tensor::Result<CompactQrResult> {
        Err(tenferro_tensor::Error::unsupported(
            "householder_qr",
            "backend does not implement compact Householder QR",
        ))
    }

    #[doc(hidden)]
    fn householder_qr_from_factors(
        &mut self,
        _q: &Tensor,
        _r: &Tensor,
    ) -> tenferro_tensor::Result<CompactQrResult> {
        Err(tenferro_tensor::Error::unsupported(
            "householder_qr_from_factors",
            "backend does not implement compact Householder QR factor import",
        ))
    }

    #[doc(hidden)]
    fn householder_qr_append(
        &mut self,
        _packed: &Tensor,
        _coeff: &Tensor,
        _block: &Tensor,
    ) -> tenferro_tensor::Result<CompactQrResult> {
        Err(tenferro_tensor::Error::unsupported(
            "householder_qr_append",
            "backend does not implement compact Householder QR append",
        ))
    }

    #[doc(hidden)]
    fn householder_qr_r(
        &mut self,
        _packed: &Tensor,
        _coeff: &Tensor,
        _options: QrOptions,
    ) -> tenferro_tensor::Result<Tensor> {
        Err(tenferro_tensor::Error::unsupported(
            "householder_qr_r",
            "backend does not implement compact Householder QR extraction",
        ))
    }

    #[doc(hidden)]
    fn householder_qr_q_columns(
        &mut self,
        _packed: &Tensor,
        _coeff: &Tensor,
        _columns: Range<usize>,
        _options: QrOptions,
    ) -> tenferro_tensor::Result<Tensor> {
        Err(tenferro_tensor::Error::unsupported(
            "householder_qr_q_columns",
            "backend does not implement compact Householder Q-column materialization",
        ))
    }

    /// Compute public Hermitian eigendecomposition outputs `(values, vectors)`.
    ///
    /// The returned vector order is `[values, vectors]`, where `values` has
    /// shape `[n]` and `vectors` has shape `[n, n]`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for a non-square or unsupported-dtype input
    /// and a typed `Error::Extension` or backend source when eigendecomposition
    /// fails.
    fn eigh(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;

    /// Compute public Hermitian eigendecomposition outputs with explicit options.
    ///
    /// `derivative_eps` is validated for API consistency, but concrete backend
    /// execution does not perform AD. `gauge` controls optional eigenvector
    /// post-processing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
    /// use tenferro_linalg::{EighGauge, EighOptions, LinalgBackend};
    /// use tenferro_tensor::{BackendSessionHost, Tensor};
    ///
    /// let input = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0])?;
    /// let mut host = CpuBackend::new();
    /// let outputs = host.with_backend_session(|session| {
    ///     with_cpu_exec_session(session, |backend| {
    ///         backend.eigh_with_options(
    ///             &input,
    ///             EighOptions::default()
    ///                 .gauge(EighGauge::CanonicalPivot)
    ///                 .derivative_eps(1.0e-10),
    ///         )
    ///     })
    ///     .expect("CpuBackend must expose a CpuExecSession")
    /// })?;
    /// assert_eq!(outputs[0].shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] containing
    /// [`tenferro_tensor::ValidationError::InvalidArgument`] when
    /// `derivative_eps` is non-finite or non-positive, when canonical gauge
    /// output metadata is malformed, or when checked output-size arithmetic
    /// overflows. It can return [`tenferro_tensor::Error::Validation`] with
    /// [`tenferro_tensor::ValidationError::RankMismatch`] or
    /// [`tenferro_tensor::ValidationError::ShapeMismatch`] for the
    /// matrix input, or [`tenferro_tensor::ValidationError::DTypeMismatch`]
    /// for generated outputs. It can also return
    /// [`tenferro_tensor::Error::Extension`] with typed
    /// `tenferro_linalg::Error::UnsupportedDType` or `NonConvergence`, and
    /// [`tenferro_tensor::Error::BackendSource`] or
    /// [`tenferro_tensor::Error::RuntimeState`] for provider and placement
    /// failures.
    fn eigh_with_options(
        &mut self,
        input: &Tensor,
        options: EighOptions,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        validate_derivative_eps("eigh_with_options", options.derivative_eps)?;
        let mut outputs = self.eigh(input)?;
        apply_eigh_gauge(options.gauge, &mut outputs)?;
        Ok(outputs)
    }

    /// Compute public Hermitian eigendecomposition outputs from a tensor read target.
    ///
    /// Backends may canonicalize the input inside the same placement family, but
    /// must not silently transfer between CPU and GPU memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::LinalgBackend;
    /// use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
    /// use tenferro_tensor::{BackendSessionHost, TensorRead, TensorView, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(
    ///     vec![2, 2],
    ///     vec![1.0, 0.0, 0.0, 2.0],
    /// )?;
    /// let mut host = CpuBackend::new();
    /// let outputs = host.with_backend_session(|session| {
    ///     with_cpu_exec_session(session, |backend| {
    ///         backend.eigh_read(TensorRead::from_view(TensorView::F64(input.as_view())))
    ///     })
    ///     .expect("CpuBackend must expose a CpuExecSession")
    /// })?;
    /// assert_eq!(outputs[0].shape(), &[2]);
    /// assert_eq!(outputs[1].shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The default implementation returns `Error::Unsupported` because the
    /// backend does not accept tensor read targets; implementations may return
    /// validation or typed backend-source errors.
    fn eigh_read(&mut self, _input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        Err(tenferro_tensor::Error::unsupported(
            "eigh",
            "backend does not accept tensor reads at this execution boundary",
        ))
    }

    /// Compute Cholesky factorization from a tensor read target.
    ///
    /// Backends may canonicalize the input inside the same placement family, but
    /// must not silently transfer between CPU and GPU memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::LinalgBackend;
    /// use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
    /// use tenferro_tensor::{BackendSessionHost, TensorRead, TensorView, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(
    ///     vec![2, 2],
    ///     vec![4.0, 2.0, 2.0, 3.0],
    /// )?;
    /// let mut host = CpuBackend::new();
    /// let output = host.with_backend_session(|session| {
    ///     with_cpu_exec_session(session, |backend| {
    ///         backend.cholesky_read(TensorRead::from_view(TensorView::F64(input.as_view())))
    ///     })
    ///     .expect("CpuBackend must expose a CpuExecSession")
    /// })?;
    /// assert_eq!(output.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The default implementation returns `Error::Unsupported` because the
    /// backend does not accept tensor read targets; implementations may return
    /// validation or typed backend-source errors.
    fn cholesky_read(&mut self, _input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        Err(tenferro_tensor::Error::unsupported(
            "cholesky",
            "backend does not accept tensor reads at this execution boundary",
        ))
    }

    /// Compute public LU outputs from a tensor read target.
    ///
    /// Backends may canonicalize the input inside the same placement family, but
    /// must not silently transfer between CPU and GPU memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::LinalgBackend;
    /// use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
    /// use tenferro_tensor::{BackendSessionHost, TensorRead, TensorView, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(
    ///     vec![2, 2],
    ///     vec![1.0, 3.0, 2.0, 4.0],
    /// )?;
    /// let mut host = CpuBackend::new();
    /// let outputs = host.with_backend_session(|session| {
    ///     with_cpu_exec_session(session, |backend| {
    ///         backend.lu_read(TensorRead::from_view(TensorView::F64(input.as_view())))
    ///     })
    ///     .expect("CpuBackend must expose a CpuExecSession")
    /// })?;
    /// assert_eq!(outputs.len(), 4);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The default implementation returns `Error::Unsupported` because the
    /// backend does not accept tensor read targets; implementations may return
    /// validation or typed backend-source errors.
    fn lu_read(&mut self, _input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        Err(tenferro_tensor::Error::unsupported(
            "lu",
            "backend does not accept tensor reads at this execution boundary",
        ))
    }

    /// Compute public full-pivoting LU outputs from a tensor read target.
    ///
    /// Backends may canonicalize the input inside the same placement family, but
    /// must not silently transfer between CPU and GPU memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::LinalgBackend;
    /// use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
    /// use tenferro_tensor::{BackendSessionHost, TensorRead, TensorView, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(
    ///     vec![2, 2],
    ///     vec![1.0, 3.0, 2.0, 4.0],
    /// )?;
    /// let mut host = CpuBackend::new();
    /// let outputs = host.with_backend_session(|session| {
    ///     with_cpu_exec_session(session, |backend| {
    ///         backend.full_piv_lu_read(TensorRead::from_view(TensorView::F64(input.as_view())))
    ///     })
    ///     .expect("CpuBackend must expose a CpuExecSession")
    /// })?;
    /// assert_eq!(outputs.len(), 5);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The default implementation returns `Error::Unsupported` because the
    /// backend does not accept tensor read targets; implementations may return
    /// validation or typed backend-source errors.
    fn full_piv_lu_read(&mut self, _input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        Err(tenferro_tensor::Error::unsupported(
            "full_piv_lu",
            "backend does not accept tensor reads at this execution boundary",
        ))
    }

    /// Compute general eigendecomposition outputs from a tensor read target.
    ///
    /// Backends may canonicalize the input inside the same placement family, but
    /// must not silently transfer between CPU and GPU memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::LinalgBackend;
    /// use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
    /// use tenferro_tensor::{BackendSessionHost, TensorRead, TensorView, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(
    ///     vec![2, 2],
    ///     vec![2.0, 0.0, 0.0, 3.0],
    /// )?;
    /// let mut host = CpuBackend::new();
    /// let outputs = host.with_backend_session(|session| {
    ///     with_cpu_exec_session(session, |backend| {
    ///         backend.eig_read(TensorRead::from_view(TensorView::F64(input.as_view())))
    ///     })
    ///     .expect("CpuBackend must expose a CpuExecSession")
    /// })?;
    /// assert_eq!(outputs.len(), 2);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The default implementation returns `Error::Unsupported` because the
    /// backend does not accept tensor read targets; implementations may return
    /// validation or typed backend-source errors.
    fn eig_read(&mut self, _input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        Err(tenferro_tensor::Error::unsupported(
            "eig",
            "backend does not accept tensor reads at this execution boundary",
        ))
    }

    #[doc(hidden)]
    fn eigh_values(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        Err(tenferro_tensor::Error::unsupported(
            "eigh_values",
            format!(
                "backend {} does not implement internal Hermitian eigenvalues-only decomposition",
                std::any::type_name::<Self>()
            ),
        ))
    }

    #[doc(hidden)]
    fn eigh_values_read(&mut self, _input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        Err(tenferro_tensor::Error::unsupported(
            "eigh_values",
            "backend does not implement borrowed Hermitian eigenvalues-only decomposition",
        ))
    }

    /// Compute public general eigendecomposition outputs `(values, vectors)`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for a non-square, rank, or dtype mismatch,
    /// and a typed `Error::Extension` or backend source when the eigensolver
    /// fails.
    fn eig(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;

    #[doc(hidden)]
    fn eig_values(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        Err(tenferro_tensor::Error::unsupported(
            "eig_values",
            format!(
                "backend {} does not implement internal general eigenvalues-only decomposition",
                std::any::type_name::<Self>()
            ),
        ))
    }

    /// Solve a dense linear system.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for incompatible matrix/rhs shapes, rank,
    /// or dtype; `Error::Extension` with `ErrorKind::NumericalFailure` for a
    /// singular system; or a typed backend source for provider failure.
    fn solve(&mut self, a: &Tensor, b: &Tensor) -> tenferro_tensor::Result<Tensor>;

    /// Solve a linear system from tensor read targets.
    ///
    /// Backends may canonicalize the inputs inside the same placement family,
    /// but must not silently transfer between CPU and GPU memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
    /// use tenferro_linalg::LinalgBackend;
    /// use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead};
    ///
    /// let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0])?;
    /// let b = Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 9.0])?;
    /// let mut host = CpuBackend::new();
    /// let x = host.with_backend_session(|session| {
    ///     with_cpu_exec_session(session, |backend| {
    ///         backend.solve_read(
    ///             TensorRead::from_tensor(&a),
    ///             TensorRead::from_tensor(&b),
    ///         )
    ///     })
    ///     .expect("CpuBackend must expose a CpuExecSession")
    /// })?;
    /// let Tensor::F64(x) = x else { unreachable!("F64 inputs return F64 output") };
    /// assert_eq!(x.host_data()?, &[2.0, 3.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The default implementation returns `Error::Unsupported` because the
    /// backend does not accept tensor read targets. Implementations may return
    /// `Error::Validation` for incompatible shapes or dtypes,
    /// `Error::RuntimeState` for invalid placement, `Error::Extension` for a
    /// singular system, or a typed backend-source error.
    fn solve_read(
        &mut self,
        _a: TensorRead<'_>,
        _b: TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        Err(tenferro_tensor::Error::unsupported(
            "solve",
            "backend does not accept tensor reads at this execution boundary",
        ))
    }

    /// Solve into a caller-owned destination.
    ///
    /// The default preserves the ordinary read path and copies its result into
    /// `out`. Backends with a native destination path may override this method,
    /// but must validate the destination before the first write and preserve
    /// the same shape, dtype, placement, aliasing, and error contracts.
    ///
    /// # Errors
    ///
    /// Returns `tenferro_tensor_core::ShapeMismatch` or
    /// `tenferro_tensor_core::ValidationError::DTypeMismatch` for incompatible
    /// destination metadata, `tenferro_tensor_core::ValidationError::InvalidArgument`
    /// for aliasing or placement violations, `Error::Unsupported` when the
    /// provider is unavailable, and `Error::Singular` for a singular system.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
    /// use tenferro_linalg::LinalgBackend;
    /// use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead, TensorWrite};
    ///
    /// let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
    /// let b = Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 8.0])?;
    /// let mut out = Tensor::from_vec_col_major(vec![2, 1], vec![0.0_f64; 2])?;
    /// let mut host = CpuBackend::new();
    /// host.with_backend_session(|session| {
    ///     with_cpu_exec_session(session, |backend| {
    ///         backend.solve_read_into(
    ///             TensorRead::from_tensor(&a),
    ///             TensorRead::from_tensor(&b),
    ///             TensorWrite::from_tensor(&mut out),
    ///         )
    ///     })
    ///     .expect("CpuBackend must expose a CpuExecSession")
    /// })?;
    /// assert_eq!(out.as_slice::<f64>()?, &[2.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn solve_read_into(
        &mut self,
        a: TensorRead<'_>,
        b: TensorRead<'_>,
        out: TensorWrite<'_>,
    ) -> tenferro_tensor::Result<()> {
        solve_read_into_default(self, a, b, out)
    }

    #[doc(hidden)]
    fn lu_solve_prepared(
        &mut self,
        _a: &Tensor,
        _packed_lu: &Tensor,
        _pivots: &Tensor,
        _b: &Tensor,
        _transpose_a: bool,
        _conjugate_a: bool,
    ) -> tenferro_tensor::Result<Tensor> {
        Err(tenferro_tensor::Error::unsupported(
            "lu_solve_prepared",
            format!(
                "backend {} does not implement internal prepared LU solve",
                std::any::type_name::<Self>()
            ),
        ))
    }
}

pub(crate) fn solve_read_into_default<B: LinalgBackend + ?Sized>(
    backend: &mut B,
    a: TensorRead<'_>,
    b: TensorRead<'_>,
    out: TensorWrite<'_>,
) -> tenferro_tensor::Result<()> {
    validate_solve_read_into(&a, &b, &out)?;
    let result = backend.solve_read(a, b)?;
    backend.copy_read_into(TensorRead::from_tensor(&result), out)
}

pub(crate) fn validate_solve_read_into(
    _a: &TensorRead<'_>,
    b: &TensorRead<'_>,
    out: &TensorWrite<'_>,
) -> tenferro_tensor::Result<()> {
    if b.shape() != out.shape() {
        return Err(tenferro_tensor::Error::shape_mismatch(
            "solve_read_into",
            b.shape().to_vec(),
            out.shape().to_vec(),
        ));
    }
    if b.dtype() != out.dtype() {
        return Err(tenferro_tensor::Error::dtype_mismatch(
            "solve_read_into",
            b.dtype(),
            out.dtype(),
        ));
    }
    if b.placement() != out.as_read().placement() {
        return Err(tenferro_tensor::Error::invalid_argument(
            "solve_read_into",
            "out",
            format!(
                "destination placement {:?} does not match rhs placement {:?}",
                out.as_read().placement(),
                b.placement()
            ),
        ));
    }
    let inputs = [_a.clone(), b.clone()];
    tenferro_tensor::backend::validate_read_into_destination("solve_read_into", &inputs, out)
}
