mod ffi;
mod kernels;
mod linalg;

use tenferro_gpu::cubecl::CubeclBackend;
use tenferro_tensor::{DType, Error, Tensor, TensorView, TensorViewCanonicalization};

use crate::backend::LinalgBackend;

impl LinalgBackend for CubeclBackend {
    fn cholesky(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        linalg::cholesky(self, input)
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
        linalg::triangular_solve(self, a, b, left_side, lower, transpose_a, unit_diagonal)
    }

    fn lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        linalg::lu(self, input)
    }

    fn full_piv_lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        linalg::full_piv_lu(self, input)
    }

    fn full_piv_lu_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        transpose_a: bool,
    ) -> tenferro_tensor::Result<Tensor> {
        linalg::full_piv_lu_solve(self, a, b, transpose_a)
    }

    fn svd(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        linalg::svd(self, input)
    }

    fn svd_view(&mut self, input: TensorView<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        match input {
            TensorView::F32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::F32(compact);
                self.svd(&input)
            }
            TensorView::F64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::F64(compact);
                self.svd(&input)
            }
            TensorView::C32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::C32(compact);
                self.svd(&input)
            }
            TensorView::C64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::C64(compact);
                self.svd(&input)
            }
            TensorView::I32(_) => Err(unsupported_view_dtype("svd", DType::I32)),
            TensorView::I64(_) => Err(unsupported_view_dtype("svd", DType::I64)),
            TensorView::Bool(_) => Err(unsupported_view_dtype("svd", DType::Bool)),
        }
    }

    fn qr(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        linalg::qr(self, input)
    }

    fn eigh(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        linalg::eigh(self, input)
    }

    fn eig(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        linalg::eig(self, input)
    }

    fn solve(&mut self, a: &Tensor, b: &Tensor) -> tenferro_tensor::Result<Tensor> {
        linalg::solve(self, a, b)
    }
}

fn unsupported_view_dtype(op: &'static str, dtype: DType) -> Error {
    Error::backend_failure(op, format!("unsupported dtype {dtype:?}"))
}
