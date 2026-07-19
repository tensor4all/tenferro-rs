use crate::backend::{unsupported_dtype, LinalgBackend};

use super::linalg;

use num_complex::{Complex32, Complex64};
use tenferro_cpu::{CpuBackend, CpuBackendKind};
use tenferro_tensor::{
    validate::validate_nonsingular_u, AllocationDomainId, Buffer, DType, Error, HostAccessError,
    MemoryKind, SharedTensorAllocationDomain, Tensor, TensorElementwise, TensorRead,
    TensorStructural, TensorView, TensorViewCanonicalization, TypedTensor,
};

impl LinalgBackend for CpuBackend {
    fn cholesky(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        if tensor_uses_backend_storage(input) {
            if let Some(domain) = self.shared_allocation_domain().cloned() {
                return managed_cholesky(self, input, domain.as_ref());
            }
        }
        ensure_host_tensor("cholesky", input)?;
        match linalg_provider_kind(self.kind(), "cholesky")? {
            CpuLinalgProvider::Faer => {
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
            CpuLinalgProvider::Blas => {
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
        ensure_host_tensor("triangular_solve", a)?;
        ensure_host_tensor("triangular_solve", b)?;
        match linalg_provider_kind(self.kind(), "triangular_solve")? {
            CpuLinalgProvider::Faer => {
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
            CpuLinalgProvider::Blas => {
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

    fn triangular_solve_read(
        &mut self,
        a: TensorRead<'_>,
        b: TensorRead<'_>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> tenferro_tensor::Result<Tensor> {
        ensure_host_tensor_read("triangular_solve", &a)?;
        ensure_host_tensor_read("triangular_solve", &b)?;
        ensure_supported_linalg_dtypes("triangular_solve", a.dtype(), b.dtype())?;
        let a = canonicalize_tensor_read(self, a)?;
        let b = canonicalize_tensor_read(self, b)?;
        self.triangular_solve(&a, &b, left_side, lower, transpose_a, unit_diagonal)
    }

    fn lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor("lu", input)?;
        match linalg_provider_kind(self.kind(), "lu")? {
            CpuLinalgProvider::Faer => {
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
            CpuLinalgProvider::Blas => {
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

    fn lu_factor(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor("lu_factor", input)?;
        match linalg_provider_kind(self.kind(), "lu_factor")? {
            CpuLinalgProvider::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.linalg_context();
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => linalg::faer::lu_factor(ctx.as_ref(), buffers, t).map(
                            |(lu, pivots, parity)| {
                                vec![Tensor::F32(lu), Tensor::I32(pivots), Tensor::F32(parity)]
                            },
                        ),
                        Tensor::F64(t) => linalg::faer::lu_factor(ctx.as_ref(), buffers, t).map(
                            |(lu, pivots, parity)| {
                                vec![Tensor::F64(lu), Tensor::I32(pivots), Tensor::F64(parity)]
                            },
                        ),
                        Tensor::C32(t) => linalg::faer::lu_factor(ctx.as_ref(), buffers, t).map(
                            |(lu, pivots, parity)| {
                                vec![Tensor::C32(lu), Tensor::I32(pivots), Tensor::C32(parity)]
                            },
                        ),
                        Tensor::C64(t) => linalg::faer::lu_factor(ctx.as_ref(), buffers, t).map(
                            |(lu, pivots, parity)| {
                                vec![Tensor::C64(lu), Tensor::I32(pivots), Tensor::C64(parity)]
                            },
                        ),
                        _ => Err(unsupported_dtype("lu_factor", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unsupported_provider("lu_factor", self.kind()))
                }
            }
            CpuLinalgProvider::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => {
                            linalg::blas::lu_factor(buffers, t).map(|(lu, pivots, parity)| {
                                vec![Tensor::F32(lu), Tensor::I32(pivots), Tensor::F32(parity)]
                            })
                        }
                        Tensor::F64(t) => {
                            linalg::blas::lu_factor(buffers, t).map(|(lu, pivots, parity)| {
                                vec![Tensor::F64(lu), Tensor::I32(pivots), Tensor::F64(parity)]
                            })
                        }
                        Tensor::C32(t) => {
                            linalg::blas::lu_factor(buffers, t).map(|(lu, pivots, parity)| {
                                vec![Tensor::C32(lu), Tensor::I32(pivots), Tensor::C32(parity)]
                            })
                        }
                        Tensor::C64(t) => {
                            linalg::blas::lu_factor(buffers, t).map(|(lu, pivots, parity)| {
                                vec![Tensor::C64(lu), Tensor::I32(pivots), Tensor::C64(parity)]
                            })
                        }
                        _ => Err(unsupported_dtype("lu_factor", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    Err(unsupported_provider("lu_factor", self.kind()))
                }
            }
        }
    }

    fn full_piv_lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor("full_piv_lu", input)?;
        match linalg_provider_kind(self.kind(), "full_piv_lu")? {
            CpuLinalgProvider::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.linalg_context();
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => linalg::faer::full_piv_lu(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                        Tensor::F64(t) => linalg::faer::full_piv_lu(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                        Tensor::C32(t) => linalg::faer::full_piv_lu(ctx.as_ref(), buffers, t)
                            .and_then(full_piv_lu_c32_outputs_to_public_tensors),
                        Tensor::C64(t) => linalg::faer::full_piv_lu(ctx.as_ref(), buffers, t)
                            .and_then(full_piv_lu_c64_outputs_to_public_tensors),
                        _ => Err(unsupported_dtype("full_piv_lu", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unsupported_provider("full_piv_lu", self.kind()))
                }
            }
            CpuLinalgProvider::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => linalg::blas::full_piv_lu(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                        Tensor::F64(t) => linalg::blas::full_piv_lu(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                        Tensor::C32(t) => linalg::blas::full_piv_lu(buffers, t)
                            .and_then(full_piv_lu_c32_outputs_to_public_tensors),
                        Tensor::C64(t) => linalg::blas::full_piv_lu(buffers, t)
                            .and_then(full_piv_lu_c64_outputs_to_public_tensors),
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
        ensure_host_tensor("full_piv_lu_solve", a)?;
        ensure_host_tensor("full_piv_lu_solve", b)?;
        ensure_supported_linalg_pair("full_piv_lu_solve", a, b)?;
        if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
            return zeros_like_tensor(b);
        }

        let (rhs, restore_shape) = if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
            (
                self.reshape(b, &matrix_rhs_shape)?,
                Some(b.shape().to_vec()),
            )
        } else {
            (b.clone(), None)
        };

        let result = match linalg_provider_kind(self.kind(), "full_piv_lu_solve")? {
            CpuLinalgProvider::Faer => {
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
            CpuLinalgProvider::Blas => {
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
        ensure_host_tensor("svd", input)?;
        match linalg_provider_kind(self.kind(), "svd")? {
            CpuLinalgProvider::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.linalg_context();
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => linalg::faer::svd(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                        Tensor::F64(t) => linalg::faer::svd(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                        Tensor::C32(t) => linalg::faer::svd(ctx.as_ref(), buffers, t)
                            .and_then(svd_c32_outputs_to_public_tensors),
                        Tensor::C64(t) => linalg::faer::svd(ctx.as_ref(), buffers, t)
                            .and_then(svd_c64_outputs_to_public_tensors),
                        _ => Err(unsupported_dtype("svd", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unsupported_provider("svd", self.kind()))
                }
            }
            CpuLinalgProvider::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => linalg::blas::svd(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                        Tensor::F64(t) => linalg::blas::svd(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                        Tensor::C32(t) => linalg::blas::svd(buffers, t)
                            .and_then(svd_c32_outputs_to_public_tensors),
                        Tensor::C64(t) => linalg::blas::svd(buffers, t)
                            .and_then(svd_c64_outputs_to_public_tensors),
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

    fn svd_values(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        ensure_host_tensor("svd_values", input)?;
        match linalg_provider_kind(self.kind(), "svd_values")? {
            CpuLinalgProvider::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.linalg_context();
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => {
                            linalg::faer::svd_values(ctx.as_ref(), buffers, t).map(Tensor::F32)
                        }
                        Tensor::F64(t) => {
                            linalg::faer::svd_values(ctx.as_ref(), buffers, t).map(Tensor::F64)
                        }
                        Tensor::C32(t) => {
                            linalg::faer::svd_values(ctx.as_ref(), buffers, t).map(Tensor::F32)
                        }
                        Tensor::C64(t) => {
                            linalg::faer::svd_values(ctx.as_ref(), buffers, t).map(Tensor::F64)
                        }
                        _ => Err(unsupported_dtype("svd_values", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unsupported_provider("svd_values", self.kind()))
                }
            }
            CpuLinalgProvider::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => linalg::blas::svd_values(buffers, t).map(Tensor::F32),
                        Tensor::F64(t) => linalg::blas::svd_values(buffers, t).map(Tensor::F64),
                        Tensor::C32(t) => linalg::blas::svd_values(buffers, t).map(Tensor::F32),
                        Tensor::C64(t) => linalg::blas::svd_values(buffers, t).map(Tensor::F64),
                        _ => Err(unsupported_dtype("svd_values", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    Err(unsupported_provider("svd_values", self.kind()))
                }
            }
        }
    }

    fn svd_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.tensor_view();
        #[cfg(feature = "cpu-faer")]
        if matches!(
            linalg_provider_kind(self.kind(), "svd")?,
            CpuLinalgProvider::Faer
        ) {
            // Fast-path: if the view is already host-resident and 2D with non-negative strides,
            // feed it directly to faer without materializing a contiguous copy.
            let can_skip_materialize = match &input {
                TensorView::F32(view) => linalg::faer::faer_strided_ok(view),
                TensorView::F64(view) => linalg::faer::faer_strided_ok(view),
                TensorView::C32(view) => linalg::faer::faer_strided_ok(view),
                TensorView::C64(view) => linalg::faer::faer_strided_ok(view),
                _ => false,
            };
            if can_skip_materialize {
                let ctx = self.linalg_context();
                return self.with_linalg_pool(|buffers| match input {
                    TensorView::F32(view) => linalg::faer::svd_view(ctx.as_ref(), buffers, view)
                        .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                    TensorView::F64(view) => linalg::faer::svd_view(ctx.as_ref(), buffers, view)
                        .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                    TensorView::C32(view) => linalg::faer::svd_view(ctx.as_ref(), buffers, view)
                        .and_then(svd_c32_outputs_to_public_tensors),
                    TensorView::C64(view) => linalg::faer::svd_view(ctx.as_ref(), buffers, view)
                        .and_then(svd_c64_outputs_to_public_tensors),
                    _ => unreachable!("can_skip_materialize only true for supported dtypes"),
                });
            }
        }
        // Fall through: materialize the view first (handles non-faer backends, GPU tensors,
        // negative strides, rank != 2, etc.).
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
        ensure_host_tensor("qr", input)?;
        match linalg_provider_kind(self.kind(), "qr")? {
            CpuLinalgProvider::Faer => {
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
            CpuLinalgProvider::Blas => {
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

    fn qr_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.tensor_view();
        #[cfg(feature = "cpu-faer")]
        if matches!(
            linalg_provider_kind(self.kind(), "qr")?,
            CpuLinalgProvider::Faer
        ) {
            // Fast-path: feed an already host-resident 2D non-negative-strided view directly to faer.
            let can_skip_materialize = match &input {
                TensorView::F32(view) => linalg::faer::faer_strided_ok(view),
                TensorView::F64(view) => linalg::faer::faer_strided_ok(view),
                TensorView::C32(view) => linalg::faer::faer_strided_ok(view),
                TensorView::C64(view) => linalg::faer::faer_strided_ok(view),
                _ => false,
            };
            if can_skip_materialize {
                let ctx = self.linalg_context();
                return self.with_linalg_pool(|buffers| match input {
                    TensorView::F32(view) => linalg::faer::qr_view(ctx.as_ref(), buffers, view)
                        .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                    TensorView::F64(view) => linalg::faer::qr_view(ctx.as_ref(), buffers, view)
                        .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                    TensorView::C32(view) => linalg::faer::qr_view(ctx.as_ref(), buffers, view)
                        .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
                    TensorView::C64(view) => linalg::faer::qr_view(ctx.as_ref(), buffers, view)
                        .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
                    _ => unreachable!("can_skip_materialize only true for supported dtypes"),
                });
            }
        }
        // Fall through: materialize the view first (non-faer backends, GPU tensors,
        // negative strides, rank != 2, etc.).
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
        ensure_host_tensor("eigh", input)?;
        match linalg_provider_kind(self.kind(), "eigh")? {
            CpuLinalgProvider::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.linalg_context();
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => linalg::faer::eigh(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                        Tensor::F64(t) => linalg::faer::eigh(ctx.as_ref(), buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                        Tensor::C32(t) => linalg::faer::eigh(ctx.as_ref(), buffers, t)
                            .and_then(eigh_c32_outputs_to_public_tensors),
                        Tensor::C64(t) => linalg::faer::eigh(ctx.as_ref(), buffers, t)
                            .and_then(eigh_c64_outputs_to_public_tensors),
                        _ => Err(unsupported_dtype("eigh", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unsupported_provider("eigh", self.kind()))
                }
            }
            CpuLinalgProvider::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => linalg::blas::eigh(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                        Tensor::F64(t) => linalg::blas::eigh(buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                        Tensor::C32(t) => linalg::blas::eigh(buffers, t)
                            .and_then(eigh_c32_outputs_to_public_tensors),
                        Tensor::C64(t) => linalg::blas::eigh(buffers, t)
                            .and_then(eigh_c64_outputs_to_public_tensors),
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

    fn eigh_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.tensor_view();
        #[cfg(feature = "cpu-faer")]
        if matches!(
            linalg_provider_kind(self.kind(), "eigh")?,
            CpuLinalgProvider::Faer
        ) {
            // Fast-path: feed an already host-resident 2D non-negative-strided view directly to faer.
            // Complex eigenvalues are real; mirror the materialized `eigh` path by converting the
            // complex outputs (real eigenvalues, complex eigenvectors) to public tensors.
            let can_skip_materialize = match &input {
                TensorView::F32(view) => linalg::faer::faer_strided_ok(view),
                TensorView::F64(view) => linalg::faer::faer_strided_ok(view),
                TensorView::C32(view) => linalg::faer::faer_strided_ok(view),
                TensorView::C64(view) => linalg::faer::faer_strided_ok(view),
                _ => false,
            };
            if can_skip_materialize {
                let ctx = self.linalg_context();
                return self.with_linalg_pool(|buffers| match input {
                    TensorView::F32(view) => linalg::faer::eigh_view(ctx.as_ref(), buffers, view)
                        .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                    TensorView::F64(view) => linalg::faer::eigh_view(ctx.as_ref(), buffers, view)
                        .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                    TensorView::C32(view) => linalg::faer::eigh_view(ctx.as_ref(), buffers, view)
                        .and_then(eigh_c32_outputs_to_public_tensors),
                    TensorView::C64(view) => linalg::faer::eigh_view(ctx.as_ref(), buffers, view)
                        .and_then(eigh_c64_outputs_to_public_tensors),
                    _ => unreachable!("can_skip_materialize only true for supported dtypes"),
                });
            }
        }
        // Fall through: materialize the view first (non-faer backends, GPU tensors,
        // negative strides, rank != 2, etc.).
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
        if self.shared_allocation_domain().is_some() {
            if let Some(input) = input.as_tensor() {
                return self.cholesky(input);
            }
        }
        let input = input.tensor_view();
        #[cfg(feature = "cpu-faer")]
        if matches!(
            linalg_provider_kind(self.kind(), "cholesky")?,
            CpuLinalgProvider::Faer
        ) {
            let can_skip_materialize = match &input {
                TensorView::F32(view) => linalg::faer::faer_strided_ok(view),
                TensorView::F64(view) => linalg::faer::faer_strided_ok(view),
                TensorView::C32(view) => linalg::faer::faer_strided_ok(view),
                TensorView::C64(view) => linalg::faer::faer_strided_ok(view),
                _ => false,
            };
            if can_skip_materialize {
                let ctx = self.linalg_context();
                return self.with_linalg_pool(|buffers| match input {
                    TensorView::F32(view) => {
                        linalg::faer::cholesky_view(ctx.as_ref(), buffers, view).map(Tensor::F32)
                    }
                    TensorView::F64(view) => {
                        linalg::faer::cholesky_view(ctx.as_ref(), buffers, view).map(Tensor::F64)
                    }
                    TensorView::C32(view) => {
                        linalg::faer::cholesky_view(ctx.as_ref(), buffers, view).map(Tensor::C32)
                    }
                    TensorView::C64(view) => {
                        linalg::faer::cholesky_view(ctx.as_ref(), buffers, view).map(Tensor::C64)
                    }
                    _ => unreachable!("can_skip_materialize only true for supported dtypes"),
                });
            }
        }
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
        #[cfg(feature = "cpu-faer")]
        if matches!(
            linalg_provider_kind(self.kind(), "lu")?,
            CpuLinalgProvider::Faer
        ) {
            let can_skip_materialize = match &input {
                TensorView::F32(view) => linalg::faer::faer_strided_ok(view),
                TensorView::F64(view) => linalg::faer::faer_strided_ok(view),
                TensorView::C32(view) => linalg::faer::faer_strided_ok(view),
                TensorView::C64(view) => linalg::faer::faer_strided_ok(view),
                _ => false,
            };
            if can_skip_materialize {
                let ctx = self.linalg_context();
                return self.with_linalg_pool(|buffers| match input {
                    TensorView::F32(view) => linalg::faer::lu_view(ctx.as_ref(), buffers, view)
                        .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                    TensorView::F64(view) => linalg::faer::lu_view(ctx.as_ref(), buffers, view)
                        .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                    TensorView::C32(view) => linalg::faer::lu_view(ctx.as_ref(), buffers, view)
                        .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
                    TensorView::C64(view) => linalg::faer::lu_view(ctx.as_ref(), buffers, view)
                        .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
                    _ => unreachable!("can_skip_materialize only true for supported dtypes"),
                });
            }
        }
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
        #[cfg(feature = "cpu-faer")]
        if matches!(
            linalg_provider_kind(self.kind(), "full_piv_lu")?,
            CpuLinalgProvider::Faer
        ) {
            let can_skip_materialize = match &input {
                TensorView::F32(view) => linalg::faer::faer_strided_ok(view),
                TensorView::F64(view) => linalg::faer::faer_strided_ok(view),
                TensorView::C32(view) => linalg::faer::faer_strided_ok(view),
                TensorView::C64(view) => linalg::faer::faer_strided_ok(view),
                _ => false,
            };
            if can_skip_materialize {
                let ctx = self.linalg_context();
                return self.with_linalg_pool(|buffers| match input {
                    TensorView::F32(view) => {
                        linalg::faer::full_piv_lu_view(ctx.as_ref(), buffers, view)
                            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect())
                    }
                    TensorView::F64(view) => {
                        linalg::faer::full_piv_lu_view(ctx.as_ref(), buffers, view)
                            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect())
                    }
                    TensorView::C32(view) => {
                        linalg::faer::full_piv_lu_view(ctx.as_ref(), buffers, view)
                            .map(|outputs| outputs.into_iter().map(Tensor::C32).collect())
                    }
                    TensorView::C64(view) => {
                        linalg::faer::full_piv_lu_view(ctx.as_ref(), buffers, view)
                            .map(|outputs| outputs.into_iter().map(Tensor::C64).collect())
                    }
                    _ => unreachable!("can_skip_materialize only true for supported dtypes"),
                });
            }
        }
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
        // eig has no faer fast-path; always materialize first.
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
        ensure_host_tensor("eigh_values", input)?;
        match linalg_provider_kind(self.kind(), "eigh_values")? {
            CpuLinalgProvider::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.linalg_context();
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => {
                            linalg::faer::eigh_values(ctx.as_ref(), buffers, t).map(Tensor::F32)
                        }
                        Tensor::F64(t) => {
                            linalg::faer::eigh_values(ctx.as_ref(), buffers, t).map(Tensor::F64)
                        }
                        Tensor::C32(t) => {
                            linalg::faer::eigh_values(ctx.as_ref(), buffers, t).map(Tensor::F32)
                        }
                        Tensor::C64(t) => {
                            linalg::faer::eigh_values(ctx.as_ref(), buffers, t).map(Tensor::F64)
                        }
                        _ => Err(unsupported_dtype("eigh_values", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unsupported_provider("eigh_values", self.kind()))
                }
            }
            CpuLinalgProvider::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.with_linalg_pool(|buffers| match input {
                        Tensor::F32(t) => linalg::blas::eigh_values(buffers, t).map(Tensor::F32),
                        Tensor::F64(t) => linalg::blas::eigh_values(buffers, t).map(Tensor::F64),
                        Tensor::C32(t) => linalg::blas::eigh_values(buffers, t).map(Tensor::F32),
                        Tensor::C64(t) => linalg::blas::eigh_values(buffers, t).map(Tensor::F64),
                        _ => Err(unsupported_dtype("eigh_values", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    Err(unsupported_provider("eigh_values", self.kind()))
                }
            }
        }
    }

    fn eig(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor("eig", input)?;
        if !matches!(
            input,
            Tensor::F32(_) | Tensor::F64(_) | Tensor::C32(_) | Tensor::C64(_)
        ) {
            return Err(unsupported_dtype("eig", input.dtype()));
        }
        match linalg_provider_kind(self.kind(), "eig")? {
            CpuLinalgProvider::Faer => {
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
            CpuLinalgProvider::Blas => {
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

    fn eig_values(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        ensure_host_tensor("eig_values", input)?;
        if !matches!(
            input,
            Tensor::F32(_) | Tensor::F64(_) | Tensor::C32(_) | Tensor::C64(_)
        ) {
            return Err(unsupported_dtype("eig_values", input.dtype()));
        }
        match linalg_provider_kind(self.kind(), "eig_values")? {
            CpuLinalgProvider::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    let ctx = self.linalg_context();
                    self.with_linalg_pool(|buffers| {
                        linalg::faer::eig_values(ctx.as_ref(), buffers, input)
                    })
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unsupported_provider("eig_values", self.kind()))
                }
            }
            CpuLinalgProvider::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    self.with_linalg_pool(|buffers| linalg::blas::eig_values(buffers, input))
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    Err(unsupported_provider("eig_values", self.kind()))
                }
            }
        }
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
        const OP: &str = "lu_solve_prepared";

        ensure_host_tensor(OP, a)?;
        ensure_host_tensor(OP, packed_lu)?;
        ensure_host_tensor(OP, pivots)?;
        ensure_host_tensor(OP, b)?;
        ensure_supported_linalg_pair(OP, a, b)?;
        ensure_supported_linalg_pair(OP, a, packed_lu)?;
        if !matches!(pivots, Tensor::I32(_)) {
            return Err(Error::dtype_mismatch(OP, DType::I32, pivots.dtype()));
        }
        if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
            return zeros_like_tensor(b);
        }

        let (rhs, restore_shape) = if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
            (
                self.reshape(b, &matrix_rhs_shape)?,
                Some(b.shape().to_vec()),
            )
        } else {
            (b.clone(), None)
        };

        validate_lu_solve_prepared_shapes(packed_lu.shape(), pivots.shape(), rhs.shape())?;
        validate_nonsingular_u(packed_lu)?;
        let lu_op = if conjugate_a {
            self.conj(packed_lu)?
        } else {
            packed_lu.clone()
        };
        let result = if transpose_a {
            let z = self.triangular_solve(&lu_op, &rhs, true, false, true, false)?;
            let y = self.triangular_solve(&lu_op, &z, true, true, true, true)?;
            apply_lu_pivots_cpu(&y, pivots, true)?
        } else {
            let pb = apply_lu_pivots_cpu(&rhs, pivots, false)?;
            let y = self.triangular_solve(&lu_op, &pb, true, true, false, true)?;
            self.triangular_solve(&lu_op, &y, true, false, false, false)?
        };

        if let Some(shape) = restore_shape {
            self.reshape(&result, &shape)
        } else {
            Ok(result)
        }
    }

    fn solve(&mut self, a: &Tensor, b: &Tensor) -> tenferro_tensor::Result<Tensor> {
        ensure_host_tensor("solve", a)?;
        ensure_host_tensor("solve", b)?;
        ensure_supported_linalg_pair("solve", a, b)?;
        if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
            return zeros_like_tensor(b);
        }

        let (rhs, restore_shape) = if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
            (
                self.reshape(b, &matrix_rhs_shape)?,
                Some(b.shape().to_vec()),
            )
        } else {
            (b.clone(), None)
        };

        let result = match linalg_provider_kind(self.kind(), "solve")? {
            CpuLinalgProvider::Faer => {
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
            CpuLinalgProvider::Blas => {
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

    fn solve_read(
        &mut self,
        a: TensorRead<'_>,
        b: TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        ensure_host_tensor_read("solve", &a)?;
        ensure_host_tensor_read("solve", &b)?;
        ensure_supported_linalg_dtypes("solve", a.dtype(), b.dtype())?;
        let a = canonicalize_tensor_read(self, a)?;
        let b = canonicalize_tensor_read(self, b)?;
        self.solve(&a, &b)
    }
}

fn tensor_uses_backend_storage(input: &Tensor) -> bool {
    match input {
        Tensor::F32(input) => matches!(input.buffer(), Buffer::Backend(_)),
        Tensor::F64(input) => matches!(input.buffer(), Buffer::Backend(_)),
        Tensor::I32(input) => matches!(input.buffer(), Buffer::Backend(_)),
        Tensor::I64(input) => matches!(input.buffer(), Buffer::Backend(_)),
        Tensor::Bool(input) => matches!(input.buffer(), Buffer::Backend(_)),
        Tensor::C32(input) => matches!(input.buffer(), Buffer::Backend(_)),
        Tensor::C64(input) => matches!(input.buffer(), Buffer::Backend(_)),
    }
}

fn managed_cholesky(
    backend: &mut CpuBackend,
    input: &Tensor,
    domain: &dyn SharedTensorAllocationDomain,
) -> tenferro_tensor::Result<Tensor> {
    let provider = linalg_provider_kind(backend.kind(), "cholesky")?;
    match input {
        Tensor::F32(input) => managed_cholesky_typed(backend, input, domain, provider),
        Tensor::F64(input) => managed_cholesky_typed(backend, input, domain, provider),
        Tensor::C32(input) => managed_cholesky_typed(backend, input, domain, provider),
        Tensor::C64(input) => managed_cholesky_typed(backend, input, domain, provider),
        _ => Err(unsupported_dtype("cholesky", input.dtype())),
    }
}

trait ManagedCholeskyScalar: Copy + Send + Sync + 'static {
    const DTYPE: DType;

    fn factor(
        backend: &mut CpuBackend,
        data: &[Self],
        n: usize,
        provider: CpuLinalgProvider,
    ) -> tenferro_tensor::Result<Vec<Self>>;

    fn take_output(output: Tensor) -> tenferro_tensor::Result<TypedTensor<Self>>;
    fn wrap(output: TypedTensor<Self>) -> Tensor;
}

macro_rules! impl_managed_cholesky_scalar {
    ($scalar:ty, $dtype:ident, $variant:ident) => {
        impl ManagedCholeskyScalar for $scalar {
            const DTYPE: DType = DType::$dtype;

            fn factor(
                backend: &mut CpuBackend,
                data: &[Self],
                n: usize,
                provider: CpuLinalgProvider,
            ) -> tenferro_tensor::Result<Vec<Self>> {
                match provider {
                    CpuLinalgProvider::Faer => {
                        #[cfg(feature = "cpu-faer")]
                        {
                            let ctx = backend.linalg_context();
                            backend.with_linalg_pool(|buffers| {
                                linalg::faer::cholesky_compact_data(ctx.as_ref(), buffers, data, n)
                            })
                        }
                        #[cfg(not(feature = "cpu-faer"))]
                        {
                            Err(unsupported_provider("cholesky", backend.kind()))
                        }
                    }
                    CpuLinalgProvider::Blas => {
                        #[cfg(feature = "cpu-blas")]
                        {
                            backend.with_linalg_pool(|_buffers| {
                                linalg::blas::cholesky_compact_data(data, n)
                            })
                        }
                        #[cfg(not(feature = "cpu-blas"))]
                        {
                            Err(unsupported_provider("cholesky", backend.kind()))
                        }
                    }
                }
            }

            fn take_output(output: Tensor) -> tenferro_tensor::Result<TypedTensor<Self>> {
                let Tensor::$variant(output) = output else {
                    return Err(tenferro_tensor::Error::runtime_state(
                        "cholesky",
                        concat!(
                            "shared allocation owner returned a non-",
                            stringify!($variant),
                            " output"
                        ),
                    ));
                };
                Ok(output)
            }

            fn wrap(output: TypedTensor<Self>) -> Tensor {
                Tensor::$variant(output)
            }
        }
    };
}

impl_managed_cholesky_scalar!(f32, F32, F32);
impl_managed_cholesky_scalar!(f64, F64, F64);
impl_managed_cholesky_scalar!(Complex32, C32, C32);
impl_managed_cholesky_scalar!(Complex64, C64, C64);

fn managed_cholesky_typed<T>(
    backend: &mut CpuBackend,
    input: &TypedTensor<T>,
    domain: &dyn SharedTensorAllocationDomain,
    provider: CpuLinalgProvider,
) -> tenferro_tensor::Result<Tensor>
where
    T: ManagedCholeskyScalar,
{
    let n = validate_managed_cholesky_input(input, domain.id())?;
    let Buffer::Backend(buffer) = input.buffer() else {
        return Err(tenferro_tensor::Error::host_access(
            "cholesky",
            HostAccessError::Unsupported { backend: "host" },
        ));
    };
    let values = if n == 0 {
        Vec::new()
    } else {
        let read = buffer
            .map_read()
            .map_err(|source| tenferro_tensor::Error::host_access("cholesky", source))?;
        T::factor(backend, &read, n, provider)?
    };
    let typed = T::take_output(domain.allocate(T::DTYPE, &[n, n])?)?;
    write_managed_cholesky_output(&typed, domain.id(), &values)?;
    Ok(T::wrap(typed))
}

fn validate_managed_cholesky_input<T: Copy + Send + Sync + 'static>(
    input: &TypedTensor<T>,
    expected_domain: AllocationDomainId,
) -> tenferro_tensor::Result<usize> {
    if input.rank() != 2 {
        return Err(tenferro_tensor::Error::rank_mismatch(
            "cholesky",
            2,
            input.rank(),
        ));
    }
    let rows = input.shape()[0];
    let cols = input.shape()[1];
    if rows != cols {
        return Err(tenferro_tensor::Error::shape_mismatch(
            "cholesky",
            vec![rows],
            vec![cols],
        ));
    }
    if input.layout().offset() != 0 || !input.is_col_major_contiguous()? {
        return Err(tenferro_tensor::Error::invalid_argument(
            "cholesky",
            "input layout",
            "managed rank-2 Cholesky requires compact column-major full-allocation storage",
        ));
    }
    let expected_len = rows.checked_mul(cols).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(
            "cholesky",
            "input shape",
            "matrix element count overflows usize",
        )
    })?;
    let Buffer::Backend(buffer) = input.buffer() else {
        return Err(tenferro_tensor::Error::host_access(
            "cholesky",
            HostAccessError::Unsupported { backend: "host" },
        ));
    };
    match buffer.allocation_domain() {
        Some(actual) if actual == expected_domain => {}
        Some(actual) => {
            return Err(tenferro_tensor::Error::host_access(
                "cholesky",
                HostAccessError::ForeignDomain {
                    expected: expected_domain,
                    actual,
                },
            ));
        }
        None => {
            return Err(tenferro_tensor::Error::host_access(
                "cholesky",
                HostAccessError::Unsupported {
                    backend: buffer.backend_family(),
                },
            ));
        }
    }
    if input.placement().memory_kind != MemoryKind::Managed || buffer.len() != expected_len {
        return Err(tenferro_tensor::Error::host_access(
            "cholesky",
            HostAccessError::Unsupported {
                backend: buffer.backend_family(),
            },
        ));
    }
    Ok(rows)
}

fn write_managed_cholesky_output<T: Copy + Send + Sync + 'static>(
    output: &TypedTensor<T>,
    expected_domain: AllocationDomainId,
    values: &[T],
) -> tenferro_tensor::Result<()> {
    if output.allocation_domain() != Some(expected_domain)
        || output.placement().memory_kind != MemoryKind::Managed
    {
        return Err(tenferro_tensor::Error::runtime_state(
            "cholesky",
            "shared allocation owner returned an output outside its managed domain",
        ));
    }
    let Buffer::Backend(buffer) = output.buffer() else {
        return Err(tenferro_tensor::Error::runtime_state(
            "cholesky",
            "shared allocation owner returned a host output",
        ));
    };
    let mut write = buffer
        .map_write()
        .map_err(|source| tenferro_tensor::Error::host_access("cholesky", source))?;
    write
        .copy_from_slice(values)
        .map_err(|source| tenferro_tensor::Error::host_access("cholesky", source))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CpuLinalgProvider {
    Faer,
    Blas,
}

fn linalg_provider_kind(
    kind: CpuBackendKind,
    _op: &'static str,
) -> tenferro_tensor::Result<CpuLinalgProvider> {
    match kind {
        CpuBackendKind::Faer => Ok(CpuLinalgProvider::Faer),
        CpuBackendKind::Blas => Ok(CpuLinalgProvider::Blas),
    }
}

fn ensure_host_tensor(op: &'static str, input: &Tensor) -> tenferro_tensor::Result<()> {
    match input {
        Tensor::F32(t) => ensure_host_typed_tensor(op, t),
        Tensor::F64(t) => ensure_host_typed_tensor(op, t),
        Tensor::I32(t) => ensure_host_typed_tensor(op, t),
        Tensor::I64(t) => ensure_host_typed_tensor(op, t),
        Tensor::Bool(t) => ensure_host_typed_tensor(op, t),
        Tensor::C32(t) => ensure_host_typed_tensor(op, t),
        Tensor::C64(t) => ensure_host_typed_tensor(op, t),
    }
}

fn ensure_host_tensor_read(
    op: &'static str,
    input: &TensorRead<'_>,
) -> tenferro_tensor::Result<()> {
    match input {
        TensorRead::Tensor(tensor) => ensure_host_tensor(op, tensor),
        TensorRead::View(view) => ensure_host_tensor_view(op, view),
    }
}

fn ensure_host_tensor_view(
    op: &'static str,
    input: &TensorView<'_>,
) -> tenferro_tensor::Result<()> {
    let is_backend_buffer = match input {
        TensorView::F32(view) => view.backend_buffer().is_some(),
        TensorView::F64(view) => view.backend_buffer().is_some(),
        TensorView::I32(view) => view.backend_buffer().is_some(),
        TensorView::I64(view) => view.backend_buffer().is_some(),
        TensorView::Bool(view) => view.backend_buffer().is_some(),
        TensorView::C32(view) => view.backend_buffer().is_some(),
        TensorView::C64(view) => view.backend_buffer().is_some(),
    };
    if is_backend_buffer {
        return Err(Error::runtime_state(
            op,
            "CPU linalg backend received a backend buffer; download the tensor to host before CPU execution",
        ));
    }
    Ok(())
}

fn canonicalize_tensor_read<'a>(
    backend: &mut CpuBackend,
    input: TensorRead<'a>,
) -> tenferro_tensor::Result<std::borrow::Cow<'a, Tensor>> {
    let input = match input {
        TensorRead::Tensor(tensor) => return Ok(std::borrow::Cow::Borrowed(tensor)),
        TensorRead::View(view) => view,
    };
    let tensor = match input {
        TensorView::F32(view) => backend.to_contiguous(&view).map(Tensor::F32),
        TensorView::F64(view) => backend.to_contiguous(&view).map(Tensor::F64),
        TensorView::I32(view) => backend.to_contiguous(&view).map(Tensor::I32),
        TensorView::I64(view) => backend.to_contiguous(&view).map(Tensor::I64),
        TensorView::Bool(view) => backend.to_contiguous(&view).map(Tensor::Bool),
        TensorView::C32(view) => backend.to_contiguous(&view).map(Tensor::C32),
        TensorView::C64(view) => backend.to_contiguous(&view).map(Tensor::C64),
    }?;
    Ok(std::borrow::Cow::Owned(tensor))
}

fn ensure_host_typed_tensor<T: 'static>(
    op: &'static str,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<()> {
    if input.as_view().backend_buffer().is_some() {
        return Err(Error::runtime_state(
            op,
            "CPU linalg backend received a backend buffer; download the tensor to host before CPU execution",
        ));
    }
    Ok(())
}

fn ensure_supported_linalg_pair(
    op: &'static str,
    lhs: &Tensor,
    rhs: &Tensor,
) -> tenferro_tensor::Result<()> {
    ensure_supported_linalg_dtypes(op, lhs.dtype(), rhs.dtype())
}

fn ensure_supported_linalg_dtypes(
    op: &'static str,
    lhs: DType,
    rhs: DType,
) -> tenferro_tensor::Result<()> {
    if lhs != rhs {
        return Err(Error::dtype_mismatch(op, lhs, rhs));
    }
    match lhs {
        DType::F32 | DType::F64 | DType::C32 | DType::C64 => Ok(()),
        DType::I32 | DType::I64 | DType::Bool => Err(unsupported_dtype(op, lhs)),
    }
}

fn has_zero_dim(shape: &[usize]) -> bool {
    shape.contains(&0)
}

fn checked_product(
    op: &'static str,
    role: &'static str,
    shape: &[usize],
) -> tenferro_tensor::Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            Error::invalid_argument(op, "shape", format!("{role} element count overflow"))
        })
    })
}

fn batch_count(op: &'static str, batch_shape: &[usize]) -> tenferro_tensor::Result<usize> {
    Ok(checked_product(op, "batch shape", batch_shape)?.max(1))
}

fn checked_batch_offset(
    op: &'static str,
    role: &'static str,
    batch: usize,
    stride: usize,
) -> tenferro_tensor::Result<usize> {
    batch
        .checked_mul(stride)
        .ok_or_else(|| Error::invalid_argument(op, "shape", format!("{role} overflows usize")))
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

fn zeros_like_tensor(input: &Tensor) -> tenferro_tensor::Result<Tensor> {
    Ok(match input {
        Tensor::F32(t) => Tensor::F32(TypedTensor::zeros(t.shape().to_vec())?),
        Tensor::F64(t) => Tensor::F64(TypedTensor::zeros(t.shape().to_vec())?),
        Tensor::I32(t) => Tensor::I32(TypedTensor::zeros(t.shape().to_vec())?),
        Tensor::I64(t) => Tensor::I64(TypedTensor::zeros(t.shape().to_vec())?),
        Tensor::Bool(t) => Tensor::Bool(TypedTensor::from_vec_col_major(
            t.shape().to_vec(),
            vec![false; t.n_elements()],
        )?),
        Tensor::C32(t) => Tensor::C32(TypedTensor::zeros(t.shape().to_vec())?),
        Tensor::C64(t) => Tensor::C64(TypedTensor::zeros(t.shape().to_vec())?),
    })
}

fn complex32_real_part_tensor(
    values: TypedTensor<Complex32>,
) -> tenferro_tensor::Result<TypedTensor<f32>> {
    let mut out = TypedTensor::from_vec_col_major(
        values.shape().to_vec(),
        values.host_data()?.iter().map(|value| value.re).collect(),
    )?;
    out.set_placement(values.placement().clone());
    Ok(out)
}

fn complex64_real_part_tensor(
    values: TypedTensor<Complex64>,
) -> tenferro_tensor::Result<TypedTensor<f64>> {
    let mut out = TypedTensor::from_vec_col_major(
        values.shape().to_vec(),
        values.host_data()?.iter().map(|value| value.re).collect(),
    )?;
    out.set_placement(values.placement().clone());
    Ok(out)
}

fn svd_output_count_error(count: usize) -> Error {
    Error::Internal(format!(
        "svd produced an invalid output count: expected 3, got {count}"
    ))
}

fn full_piv_lu_output_count_error(count: usize) -> Error {
    Error::Internal(format!(
        "full_piv_lu produced an invalid output count: expected 5, got {count}"
    ))
}

fn eigh_output_count_error(count: usize) -> Error {
    Error::Internal(format!(
        "eigh produced an invalid output count: expected 2, got {count}"
    ))
}

fn full_piv_lu_c32_outputs_to_public_tensors(
    outputs: Vec<TypedTensor<Complex32>>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let count = outputs.len();
    let mut outputs = outputs.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(p), Some(l), Some(u), Some(q), Some(parity), None) => Ok(vec![
            Tensor::C32(p),
            Tensor::C32(l),
            Tensor::C32(u),
            Tensor::C32(q),
            Tensor::F32(complex32_real_part_tensor(parity)?),
        ]),
        _ => Err(full_piv_lu_output_count_error(count)),
    }
}

fn full_piv_lu_c64_outputs_to_public_tensors(
    outputs: Vec<TypedTensor<Complex64>>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let count = outputs.len();
    let mut outputs = outputs.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(p), Some(l), Some(u), Some(q), Some(parity), None) => Ok(vec![
            Tensor::C64(p),
            Tensor::C64(l),
            Tensor::C64(u),
            Tensor::C64(q),
            Tensor::F64(complex64_real_part_tensor(parity)?),
        ]),
        _ => Err(full_piv_lu_output_count_error(count)),
    }
}

fn svd_c32_outputs_to_public_tensors(
    outputs: Vec<TypedTensor<Complex32>>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let count = outputs.len();
    let mut outputs = outputs.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(u), Some(values), Some(vt), None) => Ok(vec![
            Tensor::C32(u),
            Tensor::F32(complex32_real_part_tensor(values)?),
            Tensor::C32(vt),
        ]),
        _ => Err(svd_output_count_error(count)),
    }
}

fn svd_c64_outputs_to_public_tensors(
    outputs: Vec<TypedTensor<Complex64>>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let count = outputs.len();
    let mut outputs = outputs.into_iter();
    match (
        outputs.next(),
        outputs.next(),
        outputs.next(),
        outputs.next(),
    ) {
        (Some(u), Some(values), Some(vt), None) => Ok(vec![
            Tensor::C64(u),
            Tensor::F64(complex64_real_part_tensor(values)?),
            Tensor::C64(vt),
        ]),
        _ => Err(svd_output_count_error(count)),
    }
}

fn eigh_c32_outputs_to_public_tensors(
    outputs: Vec<TypedTensor<Complex32>>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let count = outputs.len();
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next(), outputs.next()) {
        (Some(values), Some(vectors), None) => Ok(vec![
            Tensor::F32(complex32_real_part_tensor(values)?),
            Tensor::C32(vectors),
        ]),
        _ => Err(eigh_output_count_error(count)),
    }
}

fn eigh_c64_outputs_to_public_tensors(
    outputs: Vec<TypedTensor<Complex64>>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let count = outputs.len();
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next(), outputs.next()) {
        (Some(values), Some(vectors), None) => Ok(vec![
            Tensor::F64(complex64_real_part_tensor(values)?),
            Tensor::C64(vectors),
        ]),
        _ => Err(eigh_output_count_error(count)),
    }
}

fn apply_lu_pivots_cpu(
    input: &Tensor,
    pivots: &Tensor,
    inverse: bool,
) -> tenferro_tensor::Result<Tensor> {
    let Tensor::I32(pivots) = pivots else {
        return Err(Error::dtype_mismatch(
            "lu_solve_prepared",
            DType::I32,
            pivots.dtype(),
        ));
    };
    match input {
        Tensor::F32(t) => apply_lu_pivots_typed(t, pivots, inverse).map(Tensor::F32),
        Tensor::F64(t) => apply_lu_pivots_typed(t, pivots, inverse).map(Tensor::F64),
        Tensor::C32(t) => apply_lu_pivots_typed(t, pivots, inverse).map(Tensor::C32),
        Tensor::C64(t) => apply_lu_pivots_typed(t, pivots, inverse).map(Tensor::C64),
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => {
            Err(unsupported_dtype("lu_solve_prepared", input.dtype()))
        }
    }
}

fn apply_lu_pivots_typed<T: Clone>(
    input: &TypedTensor<T>,
    pivots: &TypedTensor<i32>,
    inverse: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let shape = input.shape();
    if shape.len() < 2 {
        return Err(Error::rank_mismatch("lu_solve_prepared", 2, shape.len()));
    }
    let rows = shape[0];
    let cols = shape[1];
    let k = pivots.shape()[0];
    if k > rows || pivots.shape()[1..] != shape[2..] {
        return Err(Error::shape_mismatch(
            "lu_solve_prepared",
            pivots.shape().to_vec(),
            shape.to_vec(),
        ));
    }
    let batch_total = batch_count("lu_solve_prepared", &shape[2..])?;
    let matrix_stride = checked_product("lu_solve_prepared", "matrix shape", &[rows, cols])?;
    let pivot_stride = k;
    let input_data = input.host_data()?;
    let pivot_data = pivots.host_data()?;
    let mut data = Vec::with_capacity(input_data.len());

    for batch in 0..batch_total {
        let mut perm: Vec<usize> = (0..rows).collect();
        let pivot_offset = checked_batch_offset(
            "lu_solve_prepared",
            "pivot batch offset",
            batch,
            pivot_stride,
        )?;
        for step in 0..k {
            let pivot_one_based = pivot_data[pivot_offset + step];
            if pivot_one_based <= 0 {
                return Err(Error::invalid_argument(
                    "lu_solve_prepared",
                    "pivot",
                    "LU pivot index must be 1-based and positive",
                ));
            }
            let pivot = usize::try_from(pivot_one_based - 1).map_err(|_| {
                Error::invalid_argument("lu_solve_prepared", "pivot", "LU pivot index is invalid")
            })?;
            if pivot >= rows {
                return Err(Error::invalid_argument(
                    "lu_solve_prepared",
                    "pivot",
                    "LU pivot index is out of bounds",
                ));
            }
            perm.swap(step, pivot);
        }
        let row_map = if inverse {
            let mut inv = vec![0usize; rows];
            for (row, &source) in perm.iter().enumerate() {
                inv[source] = row;
            }
            inv
        } else {
            perm
        };
        let batch_offset = checked_batch_offset(
            "lu_solve_prepared",
            "matrix batch offset",
            batch,
            matrix_stride,
        )?;
        for col in 0..cols {
            for &source_row in &row_map {
                data.push(input_data[batch_offset + source_row + col * rows].clone());
            }
        }
    }

    TypedTensor::from_vec_col_major(shape.to_vec(), data)
}

fn validate_lu_solve_prepared_shapes(
    lu_shape: &[usize],
    pivots_shape: &[usize],
    b_shape: &[usize],
) -> tenferro_tensor::Result<()> {
    let n = square_matrix_dim("lu_solve_prepared", lu_shape)?;
    let (b_rows, _) = matrix_dims("lu_solve_prepared", b_shape)?;
    if b_rows != n {
        return Err(Error::invalid_argument(
            "lu_solve_prepared",
            "rhs rows",
            format!("expected {n}, got {b_rows}"),
        ));
    }
    if lu_shape[2..] != b_shape[2..] {
        return Err(Error::shape_mismatch(
            "lu_solve_prepared",
            lu_shape.to_vec(),
            b_shape.to_vec(),
        ));
    }
    let mut expected_pivots = vec![n];
    expected_pivots.extend_from_slice(&lu_shape[2..]);
    if pivots_shape != expected_pivots {
        return Err(Error::shape_mismatch(
            "lu_solve_prepared",
            expected_pivots,
            pivots_shape.to_vec(),
        ));
    }
    Ok(())
}

fn matrix_dims(op: &'static str, shape: &[usize]) -> tenferro_tensor::Result<(usize, usize)> {
    if shape.len() < 2 {
        return Err(Error::rank_mismatch(op, 2, shape.len()));
    }
    Ok((shape[0], shape[1]))
}

fn square_matrix_dim(op: &'static str, shape: &[usize]) -> tenferro_tensor::Result<usize> {
    let (rows, cols) = matrix_dims(op, shape)?;
    if rows != cols {
        return Err(Error::shape_mismatch(op, vec![rows], vec![cols]));
    }
    Ok(rows)
}

// Used only by feature-disabled provider branches, so default feature builds
// may not compile a direct call site.
#[allow(dead_code)]
fn unsupported_provider(op: &'static str, kind: CpuBackendKind) -> Error {
    Error::invalid_argument(
        op,
        "provider",
        format!("CPU linalg provider {kind:?} is not compiled in"),
    )
}

fn unsupported_pair(
    op: &'static str,
    lhs: &Tensor,
    rhs: &Tensor,
) -> tenferro_tensor::Result<Tensor> {
    if lhs.dtype() != rhs.dtype() {
        Err(Error::dtype_mismatch(op, lhs.dtype(), rhs.dtype()))
    } else {
        Err(unsupported_dtype(op, lhs.dtype()))
    }
}
