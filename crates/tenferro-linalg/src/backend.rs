use tenferro_tensor::{Tensor, TensorBackend, TensorRead};

pub(crate) use crate::error::unsupported_dtype;
use crate::extension::{
    apply_eigh_gauge, apply_qr_gauge, apply_svd_gauge, validate_derivative_eps, EighOptions,
    QrOptions, SvdOptions,
};

/// Backend surface required by the linalg extension runtime.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::backend::LinalgBackend;
/// use tenferro_cpu::CpuBackend;
///
/// fn accepts_linalg_backend<B: LinalgBackend>(_backend: &mut B) {}
///
/// let mut backend = CpuBackend::new();
/// accepts_linalg_backend(&mut backend);
/// ```
pub trait LinalgBackend: TensorBackend {
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
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_linalg::{LinalgBackend, SvdGauge, SvdOptions};
    /// use tenferro_tensor::Tensor;
    ///
    /// let input = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0])?;
    /// let mut backend = CpuBackend::new();
    /// let outputs = backend.svd_with_options(
    ///     &input,
    ///     SvdOptions::default().gauge(SvdGauge::CanonicalPivot),
    /// )?;
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
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{TensorRead, TensorView, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(
    ///     vec![2, 2],
    ///     vec![1.0, 0.0, 0.0, 2.0],
    /// )?;
    /// let mut backend = CpuBackend::new();
    /// let outputs = backend.svd_read(TensorRead::from_view(TensorView::F64(input.as_view())))?;
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
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_linalg::{LinalgBackend, QrGauge, QrOptions};
    /// use tenferro_tensor::Tensor;
    ///
    /// let input = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0])?;
    /// let mut backend = CpuBackend::new();
    /// let outputs = backend.qr_with_options(
    ///     &input,
    ///     QrOptions::default().gauge(QrGauge::PositiveDiagonal),
    /// )?;
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
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{TensorRead, TensorView, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(
    ///     vec![2, 2],
    ///     vec![1.0, 0.0, 0.0, 2.0],
    /// )?;
    /// let mut backend = CpuBackend::new();
    /// let outputs = backend.qr_read(TensorRead::from_view(TensorView::F64(input.as_view())))?;
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
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_linalg::{EighGauge, EighOptions, LinalgBackend};
    /// use tenferro_tensor::Tensor;
    ///
    /// let input = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0])?;
    /// let mut backend = CpuBackend::new();
    /// let outputs = backend.eigh_with_options(
    ///     &input,
    ///     EighOptions::default()
    ///         .gauge(EighGauge::CanonicalPivot)
    ///         .derivative_eps(1.0e-10),
    /// )?;
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
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{TensorRead, TensorView, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(
    ///     vec![2, 2],
    ///     vec![1.0, 0.0, 0.0, 2.0],
    /// )?;
    /// let mut backend = CpuBackend::new();
    /// let outputs = backend.eigh_read(TensorRead::from_view(TensorView::F64(input.as_view())))?;
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
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{TensorRead, TensorView, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(
    ///     vec![2, 2],
    ///     vec![4.0, 2.0, 2.0, 3.0],
    /// )?;
    /// let mut backend = CpuBackend::new();
    /// let output = backend.cholesky_read(TensorRead::from_view(TensorView::F64(input.as_view())))?;
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
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{TensorRead, TensorView, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(
    ///     vec![2, 2],
    ///     vec![1.0, 3.0, 2.0, 4.0],
    /// )?;
    /// let mut backend = CpuBackend::new();
    /// let outputs = backend.lu_read(TensorRead::from_view(TensorView::F64(input.as_view())))?;
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
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{TensorRead, TensorView, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(
    ///     vec![2, 2],
    ///     vec![1.0, 3.0, 2.0, 4.0],
    /// )?;
    /// let mut backend = CpuBackend::new();
    /// let outputs = backend.full_piv_lu_read(TensorRead::from_view(TensorView::F64(input.as_view())))?;
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
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{TensorRead, TensorView, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(
    ///     vec![2, 2],
    ///     vec![2.0, 0.0, 0.0, 3.0],
    /// )?;
    /// let mut backend = CpuBackend::new();
    /// let outputs = backend.eig_read(TensorRead::from_view(TensorView::F64(input.as_view())))?;
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
