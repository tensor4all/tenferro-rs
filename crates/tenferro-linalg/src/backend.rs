use tenferro_tensor::{Tensor, TensorBackend, TensorView};

/// Build the shared "unsupported dtype" backend-failure error used by the
/// linalg backends.
///
/// Both the CPU and GPU linalg backends reject integer and boolean dtypes for
/// floating-point decompositions with an identical message; this helper keeps
/// that error construction in one place.
pub(crate) fn unsupported_dtype(
    op: &'static str,
    dtype: tenferro_tensor::DType,
) -> tenferro_tensor::Error {
    tenferro_tensor::Error::backend_failure(op, format!("unsupported dtype {dtype:?}"))
}

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
    fn cholesky(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor>;

    /// Solve a triangular linear system with explicit side, triangle,
    /// transpose, and unit-diagonal flags.
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
    fn lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;

    #[doc(hidden)]
    fn lu_factor(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        Err(tenferro_tensor::Error::backend_failure(
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
    fn full_piv_lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;

    /// Solve a linear system through the complete-pivot LU path.
    ///
    /// With `transpose_a = false`, this solves `A * x = b`. With
    /// `transpose_a = true`, this solves `A^T * x = b`.
    fn full_piv_lu_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        transpose_a: bool,
    ) -> tenferro_tensor::Result<Tensor>;

    /// Compute public SVD outputs `(U, S, Vt)`.
    fn svd(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;

    #[doc(hidden)]
    fn svd_values(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        Err(tenferro_tensor::Error::backend_failure(
            "svd_values",
            format!(
                "backend {} does not implement internal singular-values-only decomposition",
                std::any::type_name::<Self>()
            ),
        ))
    }

    /// Compute a singular value decomposition from a borrowed tensor view.
    ///
    /// Backends may canonicalize the view inside the same placement family, but
    /// must not silently transfer between CPU and GPU memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::LinalgBackend;
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{TensorView, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(
    ///     vec![2, 2],
    ///     vec![1.0, 0.0, 0.0, 2.0],
    /// )?;
    /// let mut backend = CpuBackend::new();
    /// let outputs = backend.svd_read(TensorView::F64(input.as_view()))?;
    /// assert_eq!(outputs[1].shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn svd_read(&mut self, _input: TensorView<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        Err(tenferro_tensor::Error::backend_failure(
            "svd",
            "backend does not accept borrowed tensor views at this execution boundary",
        ))
    }

    /// Compute public QR outputs `(Q, R)`.
    ///
    /// QR is thin: for an `m x n` input, `Q` has shape `m x min(m, n)` and
    /// `R` has shape `min(m, n) x n`.
    fn qr(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;

    /// Compute public QR outputs `(Q, R)` from a borrowed tensor view.
    ///
    /// Backends may canonicalize the view inside the same placement family, but
    /// must not silently transfer between CPU and GPU memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::LinalgBackend;
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{TensorView, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(
    ///     vec![2, 2],
    ///     vec![1.0, 0.0, 0.0, 2.0],
    /// )?;
    /// let mut backend = CpuBackend::new();
    /// let outputs = backend.qr_read(TensorView::F64(input.as_view()))?;
    /// assert_eq!(outputs[0].shape(), &[2, 2]);
    /// assert_eq!(outputs[1].shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn qr_read(&mut self, _input: TensorView<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        Err(tenferro_tensor::Error::backend_failure(
            "qr",
            "backend does not accept borrowed tensor views at this execution boundary",
        ))
    }

    /// Compute public Hermitian eigendecomposition outputs `(values, vectors)`.
    ///
    /// The returned vector order is `[values, vectors]`, where `values` has
    /// shape `[n]` and `vectors` has shape `[n, n]`.
    fn eigh(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;

    /// Compute public Hermitian eigendecomposition outputs from a borrowed tensor view.
    ///
    /// Backends may canonicalize the view inside the same placement family, but
    /// must not silently transfer between CPU and GPU memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::LinalgBackend;
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{TensorView, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(
    ///     vec![2, 2],
    ///     vec![1.0, 0.0, 0.0, 2.0],
    /// )?;
    /// let mut backend = CpuBackend::new();
    /// let outputs = backend.eigh_read(TensorView::F64(input.as_view()))?;
    /// assert_eq!(outputs[0].shape(), &[2]);
    /// assert_eq!(outputs[1].shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eigh_read(&mut self, _input: TensorView<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        Err(tenferro_tensor::Error::backend_failure(
            "eigh",
            "backend does not accept borrowed tensor views at this execution boundary",
        ))
    }

    /// Compute Cholesky factorization from a borrowed tensor view.
    ///
    /// Backends may canonicalize the view inside the same placement family, but
    /// must not silently transfer between CPU and GPU memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::LinalgBackend;
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{TensorView, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(
    ///     vec![2, 2],
    ///     vec![4.0, 2.0, 2.0, 3.0],
    /// )?;
    /// let mut backend = CpuBackend::new();
    /// let output = backend.cholesky_read(TensorView::F64(input.as_view()))?;
    /// assert_eq!(output.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn cholesky_read(&mut self, _input: TensorView<'_>) -> tenferro_tensor::Result<Tensor> {
        Err(tenferro_tensor::Error::backend_failure(
            "cholesky",
            "backend does not accept borrowed tensor views at this execution boundary",
        ))
    }

    /// Compute public LU outputs from a borrowed tensor view.
    ///
    /// Backends may canonicalize the view inside the same placement family, but
    /// must not silently transfer between CPU and GPU memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::LinalgBackend;
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{TensorView, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(
    ///     vec![2, 2],
    ///     vec![1.0, 3.0, 2.0, 4.0],
    /// )?;
    /// let mut backend = CpuBackend::new();
    /// let outputs = backend.lu_read(TensorView::F64(input.as_view()))?;
    /// assert_eq!(outputs.len(), 4);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn lu_read(&mut self, _input: TensorView<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        Err(tenferro_tensor::Error::backend_failure(
            "lu",
            "backend does not accept borrowed tensor views at this execution boundary",
        ))
    }

    /// Compute public full-pivoting LU outputs from a borrowed tensor view.
    ///
    /// Backends may canonicalize the view inside the same placement family, but
    /// must not silently transfer between CPU and GPU memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::LinalgBackend;
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{TensorView, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(
    ///     vec![2, 2],
    ///     vec![1.0, 3.0, 2.0, 4.0],
    /// )?;
    /// let mut backend = CpuBackend::new();
    /// let outputs = backend.full_piv_lu_read(TensorView::F64(input.as_view()))?;
    /// assert_eq!(outputs.len(), 5);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn full_piv_lu_read(&mut self, _input: TensorView<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        Err(tenferro_tensor::Error::backend_failure(
            "full_piv_lu",
            "backend does not accept borrowed tensor views at this execution boundary",
        ))
    }

    /// Compute general eigendecomposition outputs from a borrowed tensor view.
    ///
    /// Backends may canonicalize the view inside the same placement family, but
    /// must not silently transfer between CPU and GPU memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::LinalgBackend;
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_tensor::{TensorView, TypedTensor};
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(
    ///     vec![2, 2],
    ///     vec![2.0, 0.0, 0.0, 3.0],
    /// )?;
    /// let mut backend = CpuBackend::new();
    /// let outputs = backend.eig_read(TensorView::F64(input.as_view()))?;
    /// assert_eq!(outputs.len(), 2);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn eig_read(&mut self, _input: TensorView<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        Err(tenferro_tensor::Error::backend_failure(
            "eig",
            "backend does not accept borrowed tensor views at this execution boundary",
        ))
    }

    #[doc(hidden)]
    fn eigh_values(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        Err(tenferro_tensor::Error::backend_failure(
            "eigh_values",
            format!(
                "backend {} does not implement internal Hermitian eigenvalues-only decomposition",
                std::any::type_name::<Self>()
            ),
        ))
    }

    /// Compute public general eigendecomposition outputs `(values, vectors)`.
    fn eig(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;

    #[doc(hidden)]
    fn eig_values(&mut self, _input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        Err(tenferro_tensor::Error::backend_failure(
            "eig_values",
            format!(
                "backend {} does not implement internal general eigenvalues-only decomposition",
                std::any::type_name::<Self>()
            ),
        ))
    }

    /// Solve a dense linear system.
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
        Err(tenferro_tensor::Error::backend_failure(
            "lu_solve_prepared",
            format!(
                "backend {} does not implement internal prepared LU solve",
                std::any::type_name::<Self>()
            ),
        ))
    }
}
