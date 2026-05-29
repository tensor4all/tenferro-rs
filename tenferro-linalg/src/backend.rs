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
    fn full_piv_lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
    fn full_piv_lu_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        transpose_a: bool,
    ) -> tenferro_tensor::Result<Tensor>;
    fn svd(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;

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
    fn eig(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
    fn solve(&mut self, a: &Tensor, b: &Tensor) -> tenferro_tensor::Result<Tensor>;
}
