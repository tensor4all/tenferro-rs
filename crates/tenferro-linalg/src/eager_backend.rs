use crate::backend::LinalgBackend;
use tenferro_ad::EagerBackend;
use tenferro_tensor::{Tensor, TensorRead};

macro_rules! dispatch_linalg {
    ($backend:expr, $method:ident($($arg:expr),* $(,)?)) => {
        match $backend {
            EagerBackend::Cpu(backend) => backend.$method($($arg),*),
            #[cfg(feature = "cuda")]
            EagerBackend::Cuda(backend) => backend.$method($($arg),*),
        }
    };
}

impl LinalgBackend for EagerBackend {
    fn cholesky(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        dispatch_linalg!(self, cholesky(input))
    }

    fn triangular_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> tenferro_tensor::Result<Tensor> {
        dispatch_linalg!(
            self,
            triangular_solve(a, b, left_side, lower, transpose_a, unit_diagonal)
        )
    }

    fn lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        dispatch_linalg!(self, lu(input))
    }

    fn lu_factor(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        dispatch_linalg!(self, lu_factor(input))
    }

    fn full_piv_lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        dispatch_linalg!(self, full_piv_lu(input))
    }

    fn full_piv_lu_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        transpose_a: bool,
    ) -> tenferro_tensor::Result<Tensor> {
        dispatch_linalg!(self, full_piv_lu_solve(a, b, transpose_a))
    }

    fn svd(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        dispatch_linalg!(self, svd(input))
    }

    fn svd_full(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        dispatch_linalg!(self, svd_full(input))
    }

    fn svd_values(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        dispatch_linalg!(self, svd_values(input))
    }

    fn svd_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        dispatch_linalg!(self, svd_read(input))
    }

    fn qr(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        dispatch_linalg!(self, qr(input))
    }

    fn qr_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        dispatch_linalg!(self, qr_read(input))
    }

    fn eigh(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        dispatch_linalg!(self, eigh(input))
    }

    fn eigh_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        dispatch_linalg!(self, eigh_read(input))
    }

    fn cholesky_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        dispatch_linalg!(self, cholesky_read(input))
    }

    fn lu_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        dispatch_linalg!(self, lu_read(input))
    }

    fn full_piv_lu_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        dispatch_linalg!(self, full_piv_lu_read(input))
    }

    fn eig_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        dispatch_linalg!(self, eig_read(input))
    }

    fn eigh_values(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        dispatch_linalg!(self, eigh_values(input))
    }

    fn eig(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        dispatch_linalg!(self, eig(input))
    }

    fn solve(&mut self, a: &Tensor, b: &Tensor) -> tenferro_tensor::Result<Tensor> {
        dispatch_linalg!(self, solve(a, b))
    }

    fn lu_solve_prepared(
        &mut self,
        a: &Tensor,
        packed_lu: &Tensor,
        pivots: &Tensor,
        b: &Tensor,
        transpose_a: bool,
        conjugate_a: bool,
    ) -> tenferro_tensor::Result<Tensor> {
        dispatch_linalg!(
            self,
            lu_solve_prepared(a, packed_lu, pivots, b, transpose_a, conjugate_a)
        )
    }
}
