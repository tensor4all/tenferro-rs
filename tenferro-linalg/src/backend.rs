use tenferro_tensor::{Tensor, TensorBackend, TensorView};

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
    fn cholesky(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor>;
    fn triangular_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> tenferro_tensor::Result<Tensor>;
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
    fn full_piv_lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
    fn full_piv_lu_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        transpose_a: bool,
    ) -> tenferro_tensor::Result<Tensor>;
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
    /// );
    /// let outputs = CpuBackend::new().svd_view(TensorView::F64(input.as_view()))?;
    /// assert_eq!(outputs[1].shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn svd_view(&mut self, _input: TensorView<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        Err(tenferro_tensor::Error::backend_failure(
            "svd",
            "backend does not accept borrowed tensor views at this execution boundary",
        ))
    }

    fn qr(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
    fn eigh(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
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
    fn eig(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
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
