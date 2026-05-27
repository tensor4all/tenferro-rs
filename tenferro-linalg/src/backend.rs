use tenferro_tensor::{Tensor, TensorBackend};

/// Backend surface required by the linalg extension runtime.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::backend::LinalgBackend;
/// use tenferro_tensor::cpu::CpuBackend;
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
    fn qr(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
    fn eigh(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
    fn eig(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
    fn solve(&mut self, a: &Tensor, b: &Tensor) -> tenferro_tensor::Result<Tensor>;
}
