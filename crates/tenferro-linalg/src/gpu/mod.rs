mod ffi;
mod kernels;
mod linalg;

use tenferro_gpu::CudaExecSession;
use tenferro_tensor::{Tensor, TensorRead, TensorView};

use crate::backend::{unsupported_dtype, LinalgBackend};

impl LinalgBackend for CudaExecSession<'_> {
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

    fn lu_factor(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        linalg::lu_factor(self, input)
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

    fn svd_values(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        linalg::svd_values(self, input)
    }

    fn svd_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.tensor_view();
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
            TensorView::I32(_) | TensorView::I64(_) | TensorView::Bool(_) => {
                Err(unsupported_dtype("svd", input.dtype()))
            }
        }
    }

    fn qr(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        linalg::qr(self, input)
    }

    fn qr_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.tensor_view();
        match input {
            TensorView::F32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::F32(compact);
                self.qr(&input)
            }
            TensorView::F64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::F64(compact);
                self.qr(&input)
            }
            TensorView::C32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::C32(compact);
                self.qr(&input)
            }
            TensorView::C64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::C64(compact);
                self.qr(&input)
            }
            TensorView::I32(_) | TensorView::I64(_) | TensorView::Bool(_) => {
                Err(unsupported_dtype("qr", input.dtype()))
            }
        }
    }

    fn eigh(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        linalg::eigh(self, input)
    }

    fn eigh_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.tensor_view();
        match input {
            TensorView::F32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::F32(compact);
                self.eigh(&input)
            }
            TensorView::F64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::F64(compact);
                self.eigh(&input)
            }
            TensorView::C32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::C32(compact);
                self.eigh(&input)
            }
            TensorView::C64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::C64(compact);
                self.eigh(&input)
            }
            TensorView::I32(_) | TensorView::I64(_) | TensorView::Bool(_) => {
                Err(unsupported_dtype("eigh", input.dtype()))
            }
        }
    }

    fn cholesky_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        let input = input.tensor_view();
        match input {
            TensorView::F32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::F32(compact);
                self.cholesky(&input)
            }
            TensorView::F64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::F64(compact);
                self.cholesky(&input)
            }
            TensorView::C32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::C32(compact);
                self.cholesky(&input)
            }
            TensorView::C64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::C64(compact);
                self.cholesky(&input)
            }
            TensorView::I32(_) | TensorView::I64(_) | TensorView::Bool(_) => {
                Err(unsupported_dtype("cholesky", input.dtype()))
            }
        }
    }

    fn lu_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.tensor_view();
        match input {
            TensorView::F32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::F32(compact);
                self.lu(&input)
            }
            TensorView::F64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::F64(compact);
                self.lu(&input)
            }
            TensorView::C32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::C32(compact);
                self.lu(&input)
            }
            TensorView::C64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::C64(compact);
                self.lu(&input)
            }
            TensorView::I32(_) | TensorView::I64(_) | TensorView::Bool(_) => {
                Err(unsupported_dtype("lu", input.dtype()))
            }
        }
    }

    fn full_piv_lu_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.tensor_view();
        match input {
            TensorView::F32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::F32(compact);
                self.full_piv_lu(&input)
            }
            TensorView::F64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::F64(compact);
                self.full_piv_lu(&input)
            }
            TensorView::C32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::C32(compact);
                self.full_piv_lu(&input)
            }
            TensorView::C64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::C64(compact);
                self.full_piv_lu(&input)
            }
            TensorView::I32(_) | TensorView::I64(_) | TensorView::Bool(_) => {
                Err(unsupported_dtype("full_piv_lu", input.dtype()))
            }
        }
    }

    fn eig_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.tensor_view();
        match input {
            TensorView::F32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::F32(compact);
                self.eig(&input)
            }
            TensorView::F64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::F64(compact);
                self.eig(&input)
            }
            TensorView::C32(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::C32(compact);
                self.eig(&input)
            }
            TensorView::C64(view) => {
                let compact = self.to_contiguous(&view)?;
                let input = Tensor::C64(compact);
                self.eig(&input)
            }
            TensorView::I32(_) | TensorView::I64(_) | TensorView::Bool(_) => {
                Err(unsupported_dtype("eig", input.dtype()))
            }
        }
    }

    fn eigh_values(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        linalg::eigh_values(self, input)
    }

    fn eig(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        linalg::eig(self, input)
    }

    fn solve(&mut self, a: &Tensor, b: &Tensor) -> tenferro_tensor::Result<Tensor> {
        linalg::solve(self, a, b)
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
        linalg::lu_solve_prepared(self, a, packed_lu, pivots, b, transpose_a, conjugate_a)
    }
}
