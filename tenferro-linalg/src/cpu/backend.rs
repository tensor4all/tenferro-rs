use crate::backend::LinalgBackend;

use super::linalg;

use tenferro_tensor::cpu::{CpuBackend, CpuBackendKind};
use tenferro_tensor::{DType, Error, Tensor, TensorStructural, TypedTensor};

impl LinalgBackend for CpuBackend {
    fn cholesky(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        match self.kind() {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.linalg_context();
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => {
                            linalg::faer::cholesky(ctx.as_ref(), buffers, t).map(Tensor::F32)
                        }
                        Tensor::F64(t) => {
                            linalg::faer::cholesky(ctx.as_ref(), buffers, t).map(Tensor::F64)
                        }
                        Tensor::C32(t) => {
                            linalg::faer::cholesky(ctx.as_ref(), buffers, t).map(Tensor::C32)
                        }
                        Tensor::C64(t) => {
                            linalg::faer::cholesky(ctx.as_ref(), buffers, t).map(Tensor::C64)
                        }
                        _ => Err(unsupported_dtype("cholesky", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unsupported_provider("cholesky", self.kind()))
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => linalg::blas::cholesky(buffers, t).map(Tensor::F32),
                        Tensor::F64(t) => linalg::blas::cholesky(buffers, t).map(Tensor::F64),
                        Tensor::C32(t) => linalg::blas::cholesky(buffers, t).map(Tensor::C32),
                        Tensor::C64(t) => linalg::blas::cholesky(buffers, t).map(Tensor::C64),
                        _ => Err(unsupported_dtype("cholesky", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    Err(unsupported_provider("cholesky", self.kind()))
                }
            }
        }
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
        match self.kind() {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.linalg_context();
                    self.with_linalg_pool(|buffers| match (a, b) {
                        (Tensor::F32(a), Tensor::F32(b)) => linalg::faer::triangular_solve(
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
                        (Tensor::F64(a), Tensor::F64(b)) => linalg::faer::triangular_solve(
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
                        (Tensor::C32(a), Tensor::C32(b)) => linalg::faer::triangular_solve(
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
                        (Tensor::C64(a), Tensor::C64(b)) => linalg::faer::triangular_solve(
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
                        _ => unsupported_pair("triangular_solve", a, b),
                    })
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unsupported_provider("triangular_solve", self.kind()))
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.with_linalg_pool(|buffers| match (a, b) {
                        (Tensor::F32(a), Tensor::F32(b)) => linalg::blas::triangular_solve(
                            buffers,
                            a,
                            b,
                            left_side,
                            lower,
                            transpose_a,
                            unit_diagonal,
                        )
                        .map(Tensor::F32),
                        (Tensor::F64(a), Tensor::F64(b)) => linalg::blas::triangular_solve(
                            buffers,
                            a,
                            b,
                            left_side,
                            lower,
                            transpose_a,
                            unit_diagonal,
                        )
                        .map(Tensor::F64),
                        (Tensor::C32(a), Tensor::C32(b)) => linalg::blas::triangular_solve(
                            buffers,
                            a,
                            b,
                            left_side,
                            lower,
                            transpose_a,
                            unit_diagonal,
                        )
                        .map(Tensor::C32),
                        (Tensor::C64(a), Tensor::C64(b)) => linalg::blas::triangular_solve(
                            buffers,
                            a,
                            b,
                            left_side,
                            lower,
                            transpose_a,
                            unit_diagonal,
                        )
                        .map(Tensor::C64),
                        _ => unsupported_pair("triangular_solve", a, b),
                    })
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    Err(unsupported_provider("triangular_solve", self.kind()))
                }
            }
        }
    }

    fn lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        match self.kind() {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.linalg_context();
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => linalg::faer::lu(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                        Tensor::F64(t) => linalg::faer::lu(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                        Tensor::C32(t) => linalg::faer::lu(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
                        Tensor::C64(t) => linalg::faer::lu(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
                        _ => Err(unsupported_dtype("lu", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unsupported_provider("lu", self.kind()))
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => linalg::blas::lu(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                        Tensor::F64(t) => linalg::blas::lu(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                        Tensor::C32(t) => linalg::blas::lu(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
                        Tensor::C64(t) => linalg::blas::lu(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
                        _ => Err(unsupported_dtype("lu", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    Err(unsupported_provider("lu", self.kind()))
                }
            }
        }
    }

    fn full_piv_lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        match self.kind() {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.linalg_context();
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => linalg::faer::full_piv_lu(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                        Tensor::F64(t) => linalg::faer::full_piv_lu(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                        Tensor::C32(t) => linalg::faer::full_piv_lu(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
                        Tensor::C64(t) => linalg::faer::full_piv_lu(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
                        _ => Err(unsupported_dtype("full_piv_lu", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unsupported_provider("full_piv_lu", self.kind()))
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => linalg::blas::full_piv_lu(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                        Tensor::F64(t) => linalg::blas::full_piv_lu(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                        Tensor::C32(t) => linalg::blas::full_piv_lu(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
                        Tensor::C64(t) => linalg::blas::full_piv_lu(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
                        _ => Err(unsupported_dtype("full_piv_lu", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    Err(unsupported_provider("full_piv_lu", self.kind()))
                }
            }
        }
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

        let result = match self.kind() {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.linalg_context();
                    self.with_linalg_pool(|buffers| match (a, &rhs) {
                        (Tensor::F32(a), Tensor::F32(b)) => linalg::faer::full_piv_lu_solve(
                            ctx.as_ref(),
                            buffers,
                            a,
                            b,
                            transpose_a,
                        )
                        .map(Tensor::F32),
                        (Tensor::F64(a), Tensor::F64(b)) => linalg::faer::full_piv_lu_solve(
                            ctx.as_ref(),
                            buffers,
                            a,
                            b,
                            transpose_a,
                        )
                        .map(Tensor::F64),
                        (Tensor::C32(a), Tensor::C32(b)) => linalg::faer::full_piv_lu_solve(
                            ctx.as_ref(),
                            buffers,
                            a,
                            b,
                            transpose_a,
                        )
                        .map(Tensor::C32),
                        (Tensor::C64(a), Tensor::C64(b)) => linalg::faer::full_piv_lu_solve(
                            ctx.as_ref(),
                            buffers,
                            a,
                            b,
                            transpose_a,
                        )
                        .map(Tensor::C64),
                        _ => unsupported_pair("full_piv_lu_solve", a, &rhs),
                    })
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unsupported_provider("full_piv_lu_solve", self.kind()))
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.with_linalg_pool(|buffers| match (a, &rhs) {
                        (Tensor::F32(a), Tensor::F32(b)) => {
                            linalg::blas::full_piv_lu_solve(buffers, a, b, transpose_a)
                                .map(Tensor::F32)
                        }
                        (Tensor::F64(a), Tensor::F64(b)) => {
                            linalg::blas::full_piv_lu_solve(buffers, a, b, transpose_a)
                                .map(Tensor::F64)
                        }
                        (Tensor::C32(a), Tensor::C32(b)) => {
                            linalg::blas::full_piv_lu_solve(buffers, a, b, transpose_a)
                                .map(Tensor::C32)
                        }
                        (Tensor::C64(a), Tensor::C64(b)) => {
                            linalg::blas::full_piv_lu_solve(buffers, a, b, transpose_a)
                                .map(Tensor::C64)
                        }
                        _ => unsupported_pair("full_piv_lu_solve", a, &rhs),
                    })
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    Err(unsupported_provider("full_piv_lu_solve", self.kind()))
                }
            }
        }?;

        if let Some(shape) = restore_shape {
            self.reshape(&result, &shape)
        } else {
            Ok(result)
        }
    }

    fn svd(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        match self.kind() {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.linalg_context();
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => linalg::faer::svd(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                        Tensor::F64(t) => linalg::faer::svd(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                        Tensor::C32(t) => linalg::faer::svd(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
                        Tensor::C64(t) => linalg::faer::svd(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
                        _ => Err(unsupported_dtype("svd", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unsupported_provider("svd", self.kind()))
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => linalg::blas::svd(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                        Tensor::F64(t) => linalg::blas::svd(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                        Tensor::C32(t) => linalg::blas::svd(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
                        Tensor::C64(t) => linalg::blas::svd(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
                        _ => Err(unsupported_dtype("svd", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    Err(unsupported_provider("svd", self.kind()))
                }
            }
        }
    }

    fn qr(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        match self.kind() {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.linalg_context();
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => linalg::faer::qr(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                        Tensor::F64(t) => linalg::faer::qr(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                        Tensor::C32(t) => linalg::faer::qr(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
                        Tensor::C64(t) => linalg::faer::qr(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
                        _ => Err(unsupported_dtype("qr", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unsupported_provider("qr", self.kind()))
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => linalg::blas::qr(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                        Tensor::F64(t) => linalg::blas::qr(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                        Tensor::C32(t) => linalg::blas::qr(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
                        Tensor::C64(t) => linalg::blas::qr(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
                        _ => Err(unsupported_dtype("qr", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    Err(unsupported_provider("qr", self.kind()))
                }
            }
        }
    }

    fn eigh(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        match self.kind() {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.linalg_context();
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => linalg::faer::eigh(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                        Tensor::F64(t) => linalg::faer::eigh(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                        Tensor::C32(t) => linalg::faer::eigh(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
                        Tensor::C64(t) => linalg::faer::eigh(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
                        _ => Err(unsupported_dtype("eigh", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unsupported_provider("eigh", self.kind()))
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => linalg::blas::eigh(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                        Tensor::F64(t) => linalg::blas::eigh(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                        Tensor::C32(t) => linalg::blas::eigh(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
                        Tensor::C64(t) => linalg::blas::eigh(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
                        _ => Err(unsupported_dtype("eigh", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    Err(unsupported_provider("eigh", self.kind()))
                }
            }
        }
    }

    fn eig(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        if !matches!(
            input,
            Tensor::F32(_) | Tensor::F64(_) | Tensor::C32(_) | Tensor::C64(_)
        ) {
            return Err(unsupported_dtype("eig", input.dtype()));
        }
        match self.kind() {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.linalg_context();
                    self.with_linalg_pool(|buffers| linalg::faer::eig(ctx.as_ref(), buffers, input))
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unsupported_provider("eig", self.kind()))
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.with_linalg_pool(|buffers| linalg::blas::eig(buffers, input))
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    Err(unsupported_provider("eig", self.kind()))
                }
            }
        }
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

        let result = match self.kind() {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.linalg_context();
                    self.with_linalg_pool(|buffers| match (a, &rhs) {
                        (Tensor::F32(a), Tensor::F32(b)) => {
                            linalg::faer::solve(ctx.as_ref(), buffers, a, b, false).map(Tensor::F32)
                        }
                        (Tensor::F64(a), Tensor::F64(b)) => {
                            linalg::faer::solve(ctx.as_ref(), buffers, a, b, false).map(Tensor::F64)
                        }
                        (Tensor::C32(a), Tensor::C32(b)) => {
                            linalg::faer::solve(ctx.as_ref(), buffers, a, b, false).map(Tensor::C32)
                        }
                        (Tensor::C64(a), Tensor::C64(b)) => {
                            linalg::faer::solve(ctx.as_ref(), buffers, a, b, false).map(Tensor::C64)
                        }
                        _ => unsupported_pair("solve", a, &rhs),
                    })
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unsupported_provider("solve", self.kind()))
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.with_linalg_pool(|buffers| match (a, &rhs) {
                        (Tensor::F32(a), Tensor::F32(b)) => {
                            linalg::blas::solve(buffers, a, b, false).map(Tensor::F32)
                        }
                        (Tensor::F64(a), Tensor::F64(b)) => {
                            linalg::blas::solve(buffers, a, b, false).map(Tensor::F64)
                        }
                        (Tensor::C32(a), Tensor::C32(b)) => {
                            linalg::blas::solve(buffers, a, b, false).map(Tensor::C32)
                        }
                        (Tensor::C64(a), Tensor::C64(b)) => {
                            linalg::blas::solve(buffers, a, b, false).map(Tensor::C64)
                        }
                        _ => unsupported_pair("solve", a, &rhs),
                    })
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    Err(unsupported_provider("solve", self.kind()))
                }
            }
        }?;

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

#[allow(dead_code)]
fn unsupported_provider(op: &'static str, kind: CpuBackendKind) -> Error {
    Error::InvalidConfig {
        op,
        message: format!("CPU linalg provider {kind:?} is not compiled in"),
    }
}

fn unsupported_pair(
    op: &'static str,
    lhs: &Tensor,
    rhs: &Tensor,
) -> tenferro_tensor::Result<Tensor> {
    if lhs.dtype() != rhs.dtype() {
        Err(Error::DTypeMismatch {
            op,
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        })
    } else {
        Err(unsupported_dtype(op, lhs.dtype()))
    }
}

fn unsupported_dtype(op: &'static str, dtype: DType) -> Error {
    Error::backend_failure(op, format!("unsupported dtype {dtype:?}"))
}
