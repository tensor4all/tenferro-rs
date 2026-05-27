use crate::backend::LinalgBackend;

use super::linalg;

use tenferro_tensor::cpu::CpuBackend;
use tenferro_tensor::{DType, Error, Tensor, TensorBackend, TypedTensor};

impl LinalgBackend for CpuBackend {
    fn cholesky(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        #[cfg(feature = "cpu-faer")]
        let ctx = self.linalg_context();
        self.with_linalg_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F32(t) => linalg::cholesky(ctx.as_ref(), buffers, t).map(Tensor::F32),
            #[cfg(feature = "cpu-blas")]
            Tensor::F32(t) => linalg::cholesky(buffers, t).map(Tensor::F32),
            #[cfg(feature = "cpu-blas")]
            Tensor::C32(t) => linalg::cholesky(buffers, t).map(Tensor::C32),
            #[cfg(feature = "cpu-faer")]
            Tensor::C32(t) => linalg::cholesky(ctx.as_ref(), buffers, t).map(Tensor::C32),
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => linalg::cholesky(ctx.as_ref(), buffers, t).map(Tensor::F64),
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => linalg::cholesky(buffers, t).map(Tensor::F64),
            #[cfg(feature = "cpu-blas")]
            Tensor::C64(t) => linalg::cholesky(buffers, t).map(Tensor::C64),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => linalg::cholesky(ctx.as_ref(), buffers, t).map(Tensor::C64),
            _ => Err(unsupported_dtype("cholesky", input.dtype())),
        })
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
        #[cfg(feature = "cpu-faer")]
        let ctx = self.linalg_context();
        self.with_linalg_pool(|buffers| match (a, b) {
            #[cfg(feature = "cpu-faer")]
            (Tensor::F32(a), Tensor::F32(b)) => linalg::triangular_solve(
                ctx.as_ref(),
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::F32),
            #[cfg(feature = "cpu-faer")]
            (Tensor::F64(a), Tensor::F64(b)) => linalg::triangular_solve(
                ctx.as_ref(),
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::F64),
            #[cfg(feature = "cpu-blas")]
            (Tensor::F32(a), Tensor::F32(b)) => linalg::triangular_solve(
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::F32),
            #[cfg(feature = "cpu-blas")]
            (Tensor::F64(a), Tensor::F64(b)) => linalg::triangular_solve(
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::F64),
            #[cfg(feature = "cpu-blas")]
            (Tensor::C32(a), Tensor::C32(b)) => linalg::triangular_solve(
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::C32),
            #[cfg(feature = "cpu-blas")]
            (Tensor::C64(a), Tensor::C64(b)) => linalg::triangular_solve(
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::C64),
            #[cfg(feature = "cpu-faer")]
            (Tensor::C32(a), Tensor::C32(b)) => linalg::triangular_solve(
                ctx.as_ref(),
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::C32),
            #[cfg(feature = "cpu-faer")]
            (Tensor::C64(a), Tensor::C64(b)) => linalg::triangular_solve(
                ctx.as_ref(),
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::C64),
            _ => {
                if a.dtype() != b.dtype() {
                    Err(Error::DTypeMismatch {
                        op: "triangular_solve",
                        lhs: a.dtype(),
                        rhs: b.dtype(),
                    })
                } else {
                    Err(unsupported_dtype("triangular_solve", a.dtype()))
                }
            }
        })
    }

    fn lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        #[cfg(feature = "cpu-faer")]
        let ctx = self.linalg_context();
        self.with_linalg_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F32(t) => linalg::lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => linalg::lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F32(t) => {
                linalg::lu(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::F32).collect())
            }
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => {
                linalg::lu(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::F64).collect())
            }
            #[cfg(feature = "cpu-blas")]
            Tensor::C32(t) => {
                linalg::lu(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::C32).collect())
            }
            #[cfg(feature = "cpu-blas")]
            Tensor::C64(t) => {
                linalg::lu(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::C64).collect())
            }
            #[cfg(feature = "cpu-faer")]
            Tensor::C32(t) => linalg::lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => linalg::lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            _ => Err(unsupported_dtype("lu", input.dtype())),
        })
    }

    fn full_piv_lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        #[cfg(feature = "cpu-faer")]
        let ctx = self.linalg_context();
        self.with_linalg_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F32(t) => linalg::full_piv_lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => linalg::full_piv_lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F32(t) => linalg::full_piv_lu(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => linalg::full_piv_lu(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::C32(t) => linalg::full_piv_lu(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::C64(t) => linalg::full_piv_lu(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C32(t) => linalg::full_piv_lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => linalg::full_piv_lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            _ => Err(unsupported_dtype("full_piv_lu", input.dtype())),
        })
    }

    fn full_piv_lu_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        transpose_a: bool,
    ) -> tenferro_tensor::Result<Tensor> {
        if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
            return Ok(zeros_like_tensor(b));
        }

        let (rhs, restore_shape) = if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
            (
                self.reshape(b, &matrix_rhs_shape)?,
                Some(b.shape().to_vec()),
            )
        } else {
            (b.clone(), None)
        };

        #[cfg(feature = "cpu-faer")]
        let ctx = self.linalg_context();
        let result = self.with_linalg_pool(|buffers| match (a, &rhs) {
            #[cfg(feature = "cpu-faer")]
            (Tensor::F32(a), Tensor::F32(b)) => {
                linalg::full_piv_lu_solve(ctx.as_ref(), buffers, a, b, transpose_a).map(Tensor::F32)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::F64(a), Tensor::F64(b)) => {
                linalg::full_piv_lu_solve(ctx.as_ref(), buffers, a, b, transpose_a).map(Tensor::F64)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::F32(a), Tensor::F32(b)) => {
                linalg::full_piv_lu_solve(buffers, a, b, transpose_a).map(Tensor::F32)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::F64(a), Tensor::F64(b)) => {
                linalg::full_piv_lu_solve(buffers, a, b, transpose_a).map(Tensor::F64)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::C32(a), Tensor::C32(b)) => {
                linalg::full_piv_lu_solve(buffers, a, b, transpose_a).map(Tensor::C32)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::C64(a), Tensor::C64(b)) => {
                linalg::full_piv_lu_solve(buffers, a, b, transpose_a).map(Tensor::C64)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::C32(a), Tensor::C32(b)) => {
                linalg::full_piv_lu_solve(ctx.as_ref(), buffers, a, b, transpose_a).map(Tensor::C32)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::C64(a), Tensor::C64(b)) => {
                linalg::full_piv_lu_solve(ctx.as_ref(), buffers, a, b, transpose_a).map(Tensor::C64)
            }
            _ => {
                if a.dtype() != rhs.dtype() {
                    Err(Error::DTypeMismatch {
                        op: "full_piv_lu_solve",
                        lhs: a.dtype(),
                        rhs: rhs.dtype(),
                    })
                } else {
                    Err(unsupported_dtype("full_piv_lu_solve", a.dtype()))
                }
            }
        })?;

        if let Some(shape) = restore_shape {
            self.reshape(&result, &shape)
        } else {
            Ok(result)
        }
    }

    fn svd(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        #[cfg(feature = "cpu-faer")]
        let ctx = self.linalg_context();
        self.with_linalg_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F32(t) => linalg::svd(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => linalg::svd(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F32(t) => linalg::svd(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => linalg::svd(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::C32(t) => linalg::svd(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::C64(t) => linalg::svd(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C32(t) => linalg::svd(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => linalg::svd(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            _ => Err(unsupported_dtype("svd", input.dtype())),
        })
    }

    fn qr(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        #[cfg(feature = "cpu-faer")]
        let ctx = self.linalg_context();
        self.with_linalg_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F32(t) => linalg::qr(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => linalg::qr(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F32(t) => {
                linalg::qr(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::F32).collect())
            }
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => {
                linalg::qr(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::F64).collect())
            }
            #[cfg(feature = "cpu-blas")]
            Tensor::C32(t) => {
                linalg::qr(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::C32).collect())
            }
            #[cfg(feature = "cpu-blas")]
            Tensor::C64(t) => {
                linalg::qr(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::C64).collect())
            }
            #[cfg(feature = "cpu-faer")]
            Tensor::C32(t) => linalg::qr(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => linalg::qr(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            _ => Err(unsupported_dtype("qr", input.dtype())),
        })
    }

    fn eigh(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        #[cfg(feature = "cpu-faer")]
        let ctx = self.linalg_context();
        self.with_linalg_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F32(t) => linalg::eigh(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => linalg::eigh(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F32(t) => linalg::eigh(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => linalg::eigh(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::C32(t) => linalg::eigh(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::C64(t) => linalg::eigh(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C32(t) => linalg::eigh(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => linalg::eigh(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            _ => Err(unsupported_dtype("eigh", input.dtype())),
        })
    }

    fn eig(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        if !matches!(
            input,
            Tensor::F32(_) | Tensor::F64(_) | Tensor::C32(_) | Tensor::C64(_)
        ) {
            return Err(unsupported_dtype("eig", input.dtype()));
        }
        #[cfg(feature = "cpu-faer")]
        let ctx = self.linalg_context();
        self.with_linalg_pool(|buffers| {
            #[cfg(feature = "cpu-faer")]
            {
                linalg::eig(ctx.as_ref(), buffers, input)
            }
            #[cfg(feature = "cpu-blas")]
            {
                linalg::eig(buffers, input)
            }
        })
    }

    fn solve(&mut self, a: &Tensor, b: &Tensor) -> tenferro_tensor::Result<Tensor> {
        if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
            return Ok(zeros_like_tensor(b));
        }

        let (rhs, restore_shape) = if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
            (
                self.reshape(b, &matrix_rhs_shape)?,
                Some(b.shape().to_vec()),
            )
        } else {
            (b.clone(), None)
        };

        #[cfg(feature = "cpu-faer")]
        let ctx = self.linalg_context();
        let result = self.with_linalg_pool(|buffers| match (a, &rhs) {
            #[cfg(feature = "cpu-faer")]
            (Tensor::F32(a), Tensor::F32(b)) => {
                linalg::solve(ctx.as_ref(), buffers, a, b, false).map(Tensor::F32)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::F64(a), Tensor::F64(b)) => {
                linalg::solve(ctx.as_ref(), buffers, a, b, false).map(Tensor::F64)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::F32(a), Tensor::F32(b)) => {
                linalg::solve(buffers, a, b, false).map(Tensor::F32)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::F64(a), Tensor::F64(b)) => {
                linalg::solve(buffers, a, b, false).map(Tensor::F64)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::C32(a), Tensor::C32(b)) => {
                linalg::solve(buffers, a, b, false).map(Tensor::C32)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::C64(a), Tensor::C64(b)) => {
                linalg::solve(buffers, a, b, false).map(Tensor::C64)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::C32(a), Tensor::C32(b)) => {
                linalg::solve(ctx.as_ref(), buffers, a, b, false).map(Tensor::C32)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::C64(a), Tensor::C64(b)) => {
                linalg::solve(ctx.as_ref(), buffers, a, b, false).map(Tensor::C64)
            }
            _ => {
                if a.dtype() != rhs.dtype() {
                    Err(Error::DTypeMismatch {
                        op: "solve",
                        lhs: a.dtype(),
                        rhs: rhs.dtype(),
                    })
                } else {
                    Err(unsupported_dtype("solve", a.dtype()))
                }
            }
        })?;

        if let Some(shape) = restore_shape {
            self.reshape(&result, &shape)
        } else {
            Ok(result)
        }
    }
}

fn has_zero_dim(shape: &[usize]) -> bool {
    shape.contains(&0)
}

fn batched_vector_rhs_shape(a: &Tensor, b: &Tensor) -> Option<Vec<usize>> {
    if b.shape().len() == 1 {
        return Some(vec![b.shape()[0], 1]);
    }

    let is_batched_vector_rhs = a.shape().len() == b.shape().len() + 1
        && !b.shape().is_empty()
        && b.shape()[0] == a.shape()[0]
        && b.shape()[1..] == a.shape()[2..];
    if !is_batched_vector_rhs {
        return None;
    }

    let mut rhs_shape = vec![b.shape()[0], 1];
    rhs_shape.extend_from_slice(&b.shape()[1..]);
    Some(rhs_shape)
}

fn zeros_like_tensor(input: &Tensor) -> Tensor {
    match input {
        Tensor::F32(t) => Tensor::F32(TypedTensor::zeros(t.shape.clone())),
        Tensor::F64(t) => Tensor::F64(TypedTensor::zeros(t.shape.clone())),
        Tensor::I32(t) => Tensor::I32(TypedTensor::zeros(t.shape.clone())),
        Tensor::I64(t) => Tensor::I64(TypedTensor::zeros(t.shape.clone())),
        Tensor::Bool(t) => Tensor::Bool(TypedTensor::from_vec_col_major(
            t.shape.clone(),
            vec![false; t.n_elements()],
        )),
        Tensor::C32(t) => Tensor::C32(TypedTensor::zeros(t.shape.clone())),
        Tensor::C64(t) => Tensor::C64(TypedTensor::zeros(t.shape.clone())),
    }
}

fn unsupported_dtype(op: &'static str, dtype: DType) -> Error {
    Error::backend_failure(op, format!("unsupported dtype {dtype:?}"))
}
