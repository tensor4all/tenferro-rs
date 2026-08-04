use crate::backend::{unsupported_dtype, LinalgBackend};

use super::linalg;

use num_complex::{Complex32, Complex64};
use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_cpu::{CpuBackendKind, CpuExecSession, CpuExecutionContext};
use tenferro_tensor::{
    validate::validate_nonsingular_u, AllocationDomainId, DType, Error, HostAccessError,
    MemoryKind, SharedTensorAllocationDomain, StorageBuffer, Tensor, TensorElementwise, TensorRead,
    TensorScalar, TensorStructural, TensorView, TensorViewMut, TensorWrite, TypedTensor,
};

trait FreshLinalgOutput {
    fn tag_fresh(&mut self, domain: tenferro_tensor::CpuDomainId);
}

impl FreshLinalgOutput for Tensor {
    fn tag_fresh(&mut self, domain: tenferro_tensor::CpuDomainId) {
        macro_rules! tag {
            ($tensor:expr) => {{
                $tensor.set_cpu_affinity(Some(domain));
            }};
        }
        match self {
            Tensor::F32(tensor) => tag!(tensor),
            Tensor::F64(tensor) => tag!(tensor),
            Tensor::I32(tensor) => tag!(tensor),
            Tensor::I64(tensor) => tag!(tensor),
            Tensor::Bool(tensor) => tag!(tensor),
            Tensor::C32(tensor) => tag!(tensor),
            Tensor::C64(tensor) => tag!(tensor),
        }
    }
}

impl FreshLinalgOutput for Vec<Tensor> {
    fn tag_fresh(&mut self, domain: tenferro_tensor::CpuDomainId) {
        for output in self {
            output.tag_fresh(domain);
        }
    }
}

trait CpuBackendLinalgAffinityExt {
    fn with_linalg_pool_fresh<R: FreshLinalgOutput + Send>(
        &mut self,
        op: impl FnOnce(&CpuExecutionContext<'_>, &mut BufferPool) -> tenferro_tensor::Result<R> + Send,
    ) -> tenferro_tensor::Result<R>;
}

impl CpuBackendLinalgAffinityExt for CpuExecSession<'_> {
    fn with_linalg_pool_fresh<R: FreshLinalgOutput + Send>(
        &mut self,
        op: impl FnOnce(&CpuExecutionContext<'_>, &mut BufferPool) -> tenferro_tensor::Result<R> + Send,
    ) -> tenferro_tensor::Result<R> {
        self.with_linalg_pool(move |context, buffers| {
            let mut output = op(context, buffers)?;
            output.tag_fresh(context.domain_id());
            Ok(output)
        })
    }
}

impl LinalgBackend for CpuExecSession<'_> {
    fn cholesky(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        let domain = self.shared_allocation_domain();
        let kind = self.kind();
        self.with_linalg_pool_fresh(move |context, buffers| {
            let provider = linalg_provider_kind(kind, "cholesky")?;
            if tensor_uses_backend_storage(input) {
                if let Some(domain) = domain.as_deref() {
                    return managed_cholesky(context, buffers, input, domain, provider);
                }
            }
            ensure_host_tensor("cholesky", input)?;
            cholesky_entered(provider, context, buffers, input)
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
        ensure_host_tensor("triangular_solve", a)?;
        ensure_host_tensor("triangular_solve", b)?;
        let provider = linalg_provider_kind(self.kind(), "triangular_solve")?;
        let options = TriangularSolveOptions {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        };
        self.with_linalg_pool_fresh(|context, buffers| {
            triangular_solve_entered(provider, context, buffers, a, b, options)
        })
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
        let provider = linalg_provider_kind(self.kind(), "triangular_solve")?;
        let options = TriangularSolveOptions {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        };
        self.with_linalg_pool_fresh(move |context, buffers| {
            context.with_materialized_tensor_read(buffers, "triangular_solve", a, |a, buffers| {
                context.with_materialized_tensor_read(
                    buffers,
                    "triangular_solve",
                    b,
                    |b, buffers| {
                        triangular_solve_entered(provider, context, buffers, a, b, options)
                    },
                )
            })
        })
    }

    fn lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor("lu", input)?;
        let provider = linalg_provider_kind(self.kind(), "lu")?;
        self.with_linalg_pool_fresh(|context, buffers| {
            lu_entered(provider, context, buffers, input)
        })
    }

    fn lu_factor(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor("lu_factor", input)?;
        match linalg_provider_kind(self.kind(), "lu_factor")? {
            CpuLinalgProvider::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    self.with_linalg_pool_fresh(|ctx, buffers| match input {
                        Tensor::F32(t) => {
                            linalg::faer::lu_factor(ctx, buffers, t).map(|(lu, pivots, parity)| {
                                vec![Tensor::F32(lu), Tensor::I32(pivots), Tensor::F32(parity)]
                            })
                        }
                        Tensor::F64(t) => {
                            linalg::faer::lu_factor(ctx, buffers, t).map(|(lu, pivots, parity)| {
                                vec![Tensor::F64(lu), Tensor::I32(pivots), Tensor::F64(parity)]
                            })
                        }
                        Tensor::C32(t) => {
                            linalg::faer::lu_factor(ctx, buffers, t).map(|(lu, pivots, parity)| {
                                vec![Tensor::C32(lu), Tensor::I32(pivots), Tensor::C32(parity)]
                            })
                        }
                        Tensor::C64(t) => {
                            linalg::faer::lu_factor(ctx, buffers, t).map(|(lu, pivots, parity)| {
                                vec![Tensor::C64(lu), Tensor::I32(pivots), Tensor::C64(parity)]
                            })
                        }
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
                    self.with_linalg_pool_fresh(|_, buffers| match input {
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
        let provider = linalg_provider_kind(self.kind(), "full_piv_lu")?;
        self.with_linalg_pool_fresh(|context, buffers| {
            full_piv_lu_entered(provider, context, buffers, input)
        })
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
        let provider = linalg_provider_kind(self.kind(), "full_piv_lu_solve")?;
        if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
            return self.with_linalg_pool_fresh(|_, _| zeros_like_tensor(b));
        }

        let (rhs, restore_shape) = if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
            (
                self.reshape(b, &matrix_rhs_shape)?,
                Some(b.shape().to_vec()),
            )
        } else {
            (b.duplicate()?, None)
        };

        let result = match provider {
            CpuLinalgProvider::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    self.with_linalg_pool_fresh(|ctx, buffers| match (a, &rhs) {
                        (Tensor::F32(a), Tensor::F32(b)) => {
                            linalg::faer::full_piv_lu_solve(ctx, buffers, a, b, transpose_a)
                                .map(Tensor::F32)
                        }
                        (Tensor::F64(a), Tensor::F64(b)) => {
                            linalg::faer::full_piv_lu_solve(ctx, buffers, a, b, transpose_a)
                                .map(Tensor::F64)
                        }
                        (Tensor::C32(a), Tensor::C32(b)) => {
                            linalg::faer::full_piv_lu_solve(ctx, buffers, a, b, transpose_a)
                                .map(Tensor::C32)
                        }
                        (Tensor::C64(a), Tensor::C64(b)) => {
                            linalg::faer::full_piv_lu_solve(ctx, buffers, a, b, transpose_a)
                                .map(Tensor::C64)
                        }
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
                    self.with_linalg_pool_fresh(|_, buffers| match (a, &rhs) {
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
        let provider = linalg_provider_kind(self.kind(), "svd")?;
        self.with_linalg_pool_fresh(|context, buffers| {
            svd_entered(provider, context, buffers, input)
        })
    }

    fn svd_full(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor("svd_full", input)?;
        match linalg_provider_kind(self.kind(), "svd_full")? {
            CpuLinalgProvider::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    self.with_linalg_pool_fresh(|context, buffers| match input {
                        Tensor::F32(t) => linalg::faer::svd_full(context, buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                        Tensor::F64(t) => linalg::faer::svd_full(context, buffers, t)
                            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                        Tensor::C32(t) => linalg::faer::svd_full(context, buffers, t)
                            .and_then(svd_c32_outputs_to_public_tensors),
                        Tensor::C64(t) => linalg::faer::svd_full(context, buffers, t)
                            .and_then(svd_c64_outputs_to_public_tensors),
                        _ => Err(unsupported_dtype("svd_full", input.dtype())),
                    })
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(unsupported_provider("svd_full", self.kind()))
                }
            }
            // The LAPACK provider is intentionally not wired for full-matrices
            // SVD in this slice; it returns a typed error instead of silently
            // computing a thin decomposition. Full-SVD callers select the faer
            // provider (the default) or download to host and use it explicitly.
            CpuLinalgProvider::Blas => Err(tenferro_tensor::Error::unsupported(
                "svd_full",
                "CPU LAPACK provider does not implement full-matrices SVD; \
                 use the faer provider for full SVD",
            )),
        }
    }

    fn svd_values(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        ensure_host_tensor("svd_values", input)?;
        match linalg_provider_kind(self.kind(), "svd_values")? {
            CpuLinalgProvider::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    self.with_linalg_pool_fresh(|ctx, buffers| match input {
                        Tensor::F32(t) => {
                            linalg::faer::svd_values(ctx, buffers, t).map(Tensor::F32)
                        }
                        Tensor::F64(t) => {
                            linalg::faer::svd_values(ctx, buffers, t).map(Tensor::F64)
                        }
                        Tensor::C32(t) => {
                            linalg::faer::svd_values(ctx, buffers, t).map(Tensor::F32)
                        }
                        Tensor::C64(t) => {
                            linalg::faer::svd_values(ctx, buffers, t).map(Tensor::F64)
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
                    self.with_linalg_pool_fresh(|_, buffers| match input {
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
        ensure_host_tensor_read("svd", &input)?;
        ensure_supported_linalg_dtype("svd", input.dtype())?;
        let provider = linalg_provider_kind(self.kind(), "svd")?;
        self.with_linalg_pool_fresh(move |context, buffers| {
            #[cfg(feature = "cpu-faer")]
            if provider == CpuLinalgProvider::Faer && faer_strided_read_ok(&input) {
                return svd_faer_view_entered(context, buffers, input.tensor_view());
            }
            context.with_materialized_tensor_read(buffers, "svd", input, |input, buffers| {
                svd_entered(provider, context, buffers, input)
            })
        })
    }

    fn qr(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor("qr", input)?;
        let provider = linalg_provider_kind(self.kind(), "qr")?;
        self.with_linalg_pool_fresh(|context, buffers| {
            qr_entered(provider, context, buffers, input)
        })
    }

    fn qr_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor_read("qr", &input)?;
        ensure_supported_linalg_dtype("qr", input.dtype())?;
        let provider = linalg_provider_kind(self.kind(), "qr")?;
        self.with_linalg_pool_fresh(move |context, buffers| {
            #[cfg(feature = "cpu-faer")]
            if provider == CpuLinalgProvider::Faer && faer_strided_read_ok(&input) {
                return qr_faer_view_entered(context, buffers, input.tensor_view());
            }
            context.with_materialized_tensor_read(buffers, "qr", input, |input, buffers| {
                qr_entered(provider, context, buffers, input)
            })
        })
    }

    fn eigh(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor("eigh", input)?;
        let provider = linalg_provider_kind(self.kind(), "eigh")?;
        self.with_linalg_pool_fresh(|context, buffers| {
            eigh_entered(provider, context, buffers, input)
        })
    }

    fn eigh_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor_read("eigh", &input)?;
        ensure_supported_linalg_dtype("eigh", input.dtype())?;
        let provider = linalg_provider_kind(self.kind(), "eigh")?;
        self.with_linalg_pool_fresh(move |context, buffers| {
            #[cfg(feature = "cpu-faer")]
            if provider == CpuLinalgProvider::Faer && faer_strided_read_ok(&input) {
                return eigh_faer_view_entered(context, buffers, input.tensor_view());
            }
            context.with_materialized_tensor_read(buffers, "eigh", input, |input, buffers| {
                eigh_entered(provider, context, buffers, input)
            })
        })
    }

    fn cholesky_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        let domain = self.shared_allocation_domain();
        let kind = self.kind();
        self.with_linalg_pool_fresh(move |context, buffers| {
            let provider = linalg_provider_kind(kind, "cholesky")?;
            if let (Some(domain), Some(tensor)) = (domain.as_deref(), input.as_tensor()) {
                if tensor_uses_backend_storage(tensor) {
                    return managed_cholesky(context, buffers, tensor, domain, provider);
                }
            }
            ensure_host_tensor_read("cholesky", &input)?;
            ensure_supported_linalg_dtype("cholesky", input.dtype())?;
            #[cfg(feature = "cpu-faer")]
            if provider == CpuLinalgProvider::Faer && faer_strided_read_ok(&input) {
                return cholesky_faer_view_entered(context, buffers, input.tensor_view());
            }
            context.with_materialized_tensor_read(buffers, "cholesky", input, |input, buffers| {
                cholesky_entered(provider, context, buffers, input)
            })
        })
    }

    fn lu_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor_read("lu", &input)?;
        ensure_supported_linalg_dtype("lu", input.dtype())?;
        let provider = linalg_provider_kind(self.kind(), "lu")?;
        self.with_linalg_pool_fresh(move |context, buffers| {
            #[cfg(feature = "cpu-faer")]
            if provider == CpuLinalgProvider::Faer && faer_strided_read_ok(&input) {
                return lu_faer_view_entered(context, buffers, input.tensor_view());
            }
            context.with_materialized_tensor_read(buffers, "lu", input, |input, buffers| {
                lu_entered(provider, context, buffers, input)
            })
        })
    }

    fn full_piv_lu_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor_read("full_piv_lu", &input)?;
        ensure_supported_linalg_dtype("full_piv_lu", input.dtype())?;
        let provider = linalg_provider_kind(self.kind(), "full_piv_lu")?;
        self.with_linalg_pool_fresh(move |context, buffers| {
            #[cfg(feature = "cpu-faer")]
            if provider == CpuLinalgProvider::Faer && faer_strided_read_ok(&input) {
                return full_piv_lu_faer_view_entered(context, buffers, input.tensor_view());
            }
            context.with_materialized_tensor_read(
                buffers,
                "full_piv_lu",
                input,
                |input, buffers| full_piv_lu_entered(provider, context, buffers, input),
            )
        })
    }

    fn eig_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor_read("eig", &input)?;
        ensure_supported_linalg_dtype("eig", input.dtype())?;
        let provider = linalg_provider_kind(self.kind(), "eig")?;
        self.with_linalg_pool_fresh(move |context, buffers| {
            context.with_materialized_tensor_read(buffers, "eig", input, |input, buffers| {
                eig_entered(provider, context, buffers, input)
            })
        })
    }

    fn eigh_values(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        ensure_host_tensor("eigh_values", input)?;
        match linalg_provider_kind(self.kind(), "eigh_values")? {
            CpuLinalgProvider::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    self.with_linalg_pool_fresh(|ctx, buffers| match input {
                        Tensor::F32(t) => {
                            linalg::faer::eigh_values(ctx, buffers, t).map(Tensor::F32)
                        }
                        Tensor::F64(t) => {
                            linalg::faer::eigh_values(ctx, buffers, t).map(Tensor::F64)
                        }
                        Tensor::C32(t) => {
                            linalg::faer::eigh_values(ctx, buffers, t).map(Tensor::F32)
                        }
                        Tensor::C64(t) => {
                            linalg::faer::eigh_values(ctx, buffers, t).map(Tensor::F64)
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
                    self.with_linalg_pool_fresh(|_, buffers| match input {
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
        ensure_supported_linalg_dtype("eig", input.dtype())?;
        let provider = linalg_provider_kind(self.kind(), "eig")?;
        self.with_linalg_pool_fresh(|context, buffers| {
            eig_entered(provider, context, buffers, input)
        })
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
                    self.with_linalg_pool_fresh(|ctx, buffers| {
                        linalg::faer::eig_values(ctx, buffers, input)
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
                    self.with_linalg_pool_fresh(|_, buffers| {
                        linalg::blas::eig_values(buffers, input)
                    })
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
            return self.with_linalg_pool_fresh(|_, _| zeros_like_tensor(b));
        }

        let (rhs, restore_shape) = if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
            (
                self.reshape(b, &matrix_rhs_shape)?,
                Some(b.shape().to_vec()),
            )
        } else {
            (b.duplicate()?, None)
        };

        validate_lu_solve_prepared_shapes(packed_lu.shape(), pivots.shape(), rhs.shape())?;
        validate_nonsingular_u(packed_lu)?;
        let lu_op = if conjugate_a {
            self.conj(packed_lu)?
        } else {
            packed_lu.duplicate()?
        };
        let mut result = if transpose_a {
            let z = self.triangular_solve(&lu_op, &rhs, true, false, true, false)?;
            let y = self.triangular_solve(&lu_op, &z, true, true, true, true)?;
            apply_lu_pivots_cpu(&y, pivots, true)?
        } else {
            let pb = apply_lu_pivots_cpu(&rhs, pivots, false)?;
            let y = self.triangular_solve(&lu_op, &pb, true, true, false, true)?;
            self.triangular_solve(&lu_op, &y, true, false, false, false)?
        };
        result.tag_fresh(self.domain_id());

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
        let provider = linalg_provider_kind(self.kind(), "solve")?;
        self.with_linalg_pool_fresh(|context, buffers| {
            solve_entered(provider, context, buffers, a, b)
        })
    }

    fn solve_read(
        &mut self,
        a: TensorRead<'_>,
        b: TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        ensure_host_tensor_read("solve", &a)?;
        ensure_host_tensor_read("solve", &b)?;
        ensure_supported_linalg_dtypes("solve", a.dtype(), b.dtype())?;
        let provider = linalg_provider_kind(self.kind(), "solve")?;
        let direct = !has_zero_dim(a.shape())
            && !has_zero_dim(b.shape())
            && solve_shape_direct_eligible(a.shape(), b.shape());
        self.with_linalg_pool_fresh(move |context, buffers| {
            if direct {
                solve_from_views_entered(
                    provider,
                    context,
                    buffers,
                    a.tensor_view(),
                    b.tensor_view(),
                )
            } else {
                context.with_materialized_tensor_read(buffers, "solve", a, |a, buffers| {
                    context.with_materialized_tensor_read(buffers, "solve", b, |b, buffers| {
                        solve_entered(provider, context, buffers, a, b)
                    })
                })
            }
        })
    }

    fn solve_read_into(
        &mut self,
        a: TensorRead<'_>,
        b: TensorRead<'_>,
        out: TensorWrite<'_>,
    ) -> tenferro_tensor::Result<()> {
        crate::backend::validate_solve_read_into(&a, &b, &out)?;
        ensure_host_tensor_read("solve_read_into", &a)?;
        ensure_host_tensor_read("solve_read_into", &b)?;
        ensure_host_tensor_read("solve_read_into", &out.as_read())?;
        ensure_supported_linalg_dtypes("solve_read_into", a.dtype(), b.dtype())?;
        let provider = linalg_provider_kind(self.kind(), "solve_read_into")?;

        if has_zero_dim(a.shape())
            || has_zero_dim(b.shape())
            || !solve_read_into_direct_eligible(&a, &b, &out)
        {
            return crate::backend::solve_read_into_default(self, a, b, out);
        }

        let a = a.tensor_view();
        let b = b.tensor_view();
        self.with_linalg_pool(move |context, buffers| {
            solve_read_into_entered(provider, context, buffers, a, b, out)
        })
    }
}

fn solve_read_into_direct_eligible(
    a: &TensorRead<'_>,
    b: &TensorRead<'_>,
    out: &TensorWrite<'_>,
) -> bool {
    if !solve_shape_direct_eligible(a.shape(), b.shape()) {
        return false;
    }
    let out = out.as_read();
    if out.backend_family().is_some() || out.shape() != b.shape() {
        return false;
    }
    let Ok(strides) = out.strides() else {
        return false;
    };
    match out.shape() {
        [_] => strides == [1],
        [rows, cols] => {
            strides.first().copied() == Some(1)
                && strides.get(1).copied().is_some_and(|stride| {
                    stride >= isize::try_from(*rows).unwrap_or(isize::MAX)
                        && (*cols <= 1 || stride > 0)
                })
        }
        _ => false,
    }
}

fn solve_shape_direct_eligible(a_shape: &[usize], b_shape: &[usize]) -> bool {
    a_shape.len() == 2 && matches!(b_shape.len(), 1 | 2)
}

fn tensor_uses_backend_storage(input: &Tensor) -> bool {
    match input {
        Tensor::F32(input) => matches!(input.buffer(), StorageBuffer::Backend(_)),
        Tensor::F64(input) => matches!(input.buffer(), StorageBuffer::Backend(_)),
        Tensor::I32(input) => matches!(input.buffer(), StorageBuffer::Backend(_)),
        Tensor::I64(input) => matches!(input.buffer(), StorageBuffer::Backend(_)),
        Tensor::Bool(input) => matches!(input.buffer(), StorageBuffer::Backend(_)),
        Tensor::C32(input) => matches!(input.buffer(), StorageBuffer::Backend(_)),
        Tensor::C64(input) => matches!(input.buffer(), StorageBuffer::Backend(_)),
    }
}

fn managed_cholesky(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
    domain: &dyn SharedTensorAllocationDomain,
    provider: CpuLinalgProvider,
) -> tenferro_tensor::Result<Tensor> {
    match input {
        Tensor::F32(input) => managed_cholesky_typed(context, buffers, input, domain, provider),
        Tensor::F64(input) => managed_cholesky_typed(context, buffers, input, domain, provider),
        Tensor::C32(input) => managed_cholesky_typed(context, buffers, input, domain, provider),
        Tensor::C64(input) => managed_cholesky_typed(context, buffers, input, domain, provider),
        _ => Err(unsupported_dtype("cholesky", input.dtype())),
    }
}

trait ManagedCholeskyScalar: Copy + Send + Sync + TensorScalar + 'static {
    const DTYPE: DType;

    fn factor(
        context: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
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
                context: &CpuExecutionContext<'_>,
                buffers: &mut BufferPool,
                data: &[Self],
                n: usize,
                provider: CpuLinalgProvider,
            ) -> tenferro_tensor::Result<Vec<Self>> {
                match provider {
                    CpuLinalgProvider::Faer => {
                        #[cfg(feature = "cpu-faer")]
                        {
                            linalg::faer::cholesky_compact_data(context, buffers, data, n)
                        }
                        #[cfg(not(feature = "cpu-faer"))]
                        {
                            let _ = (context, buffers, data, n);
                            Err(unsupported_provider("cholesky", CpuBackendKind::Faer))
                        }
                    }
                    CpuLinalgProvider::Blas => {
                        #[cfg(feature = "cpu-blas")]
                        {
                            let _ = (context, buffers);
                            linalg::blas::cholesky_compact_data(data, n)
                        }
                        #[cfg(not(feature = "cpu-blas"))]
                        {
                            let _ = (context, buffers, data, n);
                            Err(unsupported_provider("cholesky", CpuBackendKind::Blas))
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
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    domain: &dyn SharedTensorAllocationDomain,
    provider: CpuLinalgProvider,
) -> tenferro_tensor::Result<Tensor>
where
    T: ManagedCholeskyScalar,
{
    let n = validate_managed_cholesky_input(input, domain.id())?;
    let StorageBuffer::Backend(buffer) = input.buffer() else {
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
        T::factor(context, buffers, &read, n, provider)?
    };
    let mut typed = T::take_output(domain.allocate(T::DTYPE, &[n, n])?)?;
    write_managed_cholesky_output(&mut typed, domain.id(), &values)?;
    Ok(T::wrap(typed))
}

fn validate_managed_cholesky_input<T: Copy + Send + Sync + TensorScalar + 'static>(
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
    let StorageBuffer::Backend(buffer) = input.buffer() else {
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
    output: &mut TypedTensor<T>,
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
    let Some(buffer) = output.backend_buffer_mut() else {
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

#[derive(Clone, Copy)]
struct TriangularSolveOptions {
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
}

fn triangular_solve_entered(
    provider: CpuLinalgProvider,
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: &Tensor,
    b: &Tensor,
    options: TriangularSolveOptions,
) -> tenferro_tensor::Result<Tensor> {
    match provider {
        CpuLinalgProvider::Faer => {
            #[cfg(feature = "cpu-faer")]
            {
                match (a, b) {
                    (Tensor::F32(a), Tensor::F32(b)) => linalg::faer::triangular_solve(
                        context,
                        buffers,
                        a,
                        b,
                        options.left_side,
                        options.lower,
                        options.transpose_a,
                        options.unit_diagonal,
                    )
                    .map(Tensor::F32),
                    (Tensor::F64(a), Tensor::F64(b)) => linalg::faer::triangular_solve(
                        context,
                        buffers,
                        a,
                        b,
                        options.left_side,
                        options.lower,
                        options.transpose_a,
                        options.unit_diagonal,
                    )
                    .map(Tensor::F64),
                    (Tensor::C32(a), Tensor::C32(b)) => linalg::faer::triangular_solve(
                        context,
                        buffers,
                        a,
                        b,
                        options.left_side,
                        options.lower,
                        options.transpose_a,
                        options.unit_diagonal,
                    )
                    .map(Tensor::C32),
                    (Tensor::C64(a), Tensor::C64(b)) => linalg::faer::triangular_solve(
                        context,
                        buffers,
                        a,
                        b,
                        options.left_side,
                        options.lower,
                        options.transpose_a,
                        options.unit_diagonal,
                    )
                    .map(Tensor::C64),
                    _ => unsupported_pair("triangular_solve", a, b),
                }
            }
            #[cfg(not(feature = "cpu-faer"))]
            {
                let _ = (context, buffers, a, b, options);
                Err(unsupported_provider(
                    "triangular_solve",
                    CpuBackendKind::Faer,
                ))
            }
        }
        CpuLinalgProvider::Blas => {
            #[cfg(feature = "cpu-blas")]
            {
                let _ = context;
                match (a, b) {
                    (Tensor::F32(a), Tensor::F32(b)) => linalg::blas::triangular_solve(
                        buffers,
                        a,
                        b,
                        options.left_side,
                        options.lower,
                        options.transpose_a,
                        options.unit_diagonal,
                    )
                    .map(Tensor::F32),
                    (Tensor::F64(a), Tensor::F64(b)) => linalg::blas::triangular_solve(
                        buffers,
                        a,
                        b,
                        options.left_side,
                        options.lower,
                        options.transpose_a,
                        options.unit_diagonal,
                    )
                    .map(Tensor::F64),
                    (Tensor::C32(a), Tensor::C32(b)) => linalg::blas::triangular_solve(
                        buffers,
                        a,
                        b,
                        options.left_side,
                        options.lower,
                        options.transpose_a,
                        options.unit_diagonal,
                    )
                    .map(Tensor::C32),
                    (Tensor::C64(a), Tensor::C64(b)) => linalg::blas::triangular_solve(
                        buffers,
                        a,
                        b,
                        options.left_side,
                        options.lower,
                        options.transpose_a,
                        options.unit_diagonal,
                    )
                    .map(Tensor::C64),
                    _ => unsupported_pair("triangular_solve", a, b),
                }
            }
            #[cfg(not(feature = "cpu-blas"))]
            {
                let _ = (context, buffers, a, b, options);
                Err(unsupported_provider(
                    "triangular_solve",
                    CpuBackendKind::Blas,
                ))
            }
        }
    }
}

fn solve_entered(
    provider: CpuLinalgProvider,
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: &Tensor,
    b: &Tensor,
) -> tenferro_tensor::Result<Tensor> {
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        return zeros_like_tensor(b);
    }

    let (rhs, restore_shape) = if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
        (
            context.reshape_tensor(b, &matrix_rhs_shape)?,
            Some(b.shape().to_vec()),
        )
    } else {
        (b.duplicate()?, None)
    };

    let result = match provider {
        CpuLinalgProvider::Faer => {
            #[cfg(feature = "cpu-faer")]
            {
                match (a, &rhs) {
                    (Tensor::F32(a), Tensor::F32(b)) => {
                        linalg::faer::solve(context, buffers, a, b, false).map(Tensor::F32)
                    }
                    (Tensor::F64(a), Tensor::F64(b)) => {
                        linalg::faer::solve(context, buffers, a, b, false).map(Tensor::F64)
                    }
                    (Tensor::C32(a), Tensor::C32(b)) => {
                        linalg::faer::solve(context, buffers, a, b, false).map(Tensor::C32)
                    }
                    (Tensor::C64(a), Tensor::C64(b)) => {
                        linalg::faer::solve(context, buffers, a, b, false).map(Tensor::C64)
                    }
                    _ => unsupported_pair("solve", a, &rhs),
                }
            }
            #[cfg(not(feature = "cpu-faer"))]
            {
                let _ = (context, buffers, a, &rhs);
                Err(unsupported_provider("solve", CpuBackendKind::Faer))
            }
        }
        CpuLinalgProvider::Blas => {
            #[cfg(feature = "cpu-blas")]
            {
                let _ = context;
                match (a, &rhs) {
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
                }
            }
            #[cfg(not(feature = "cpu-blas"))]
            {
                let _ = (context, buffers, a, &rhs);
                Err(unsupported_provider("solve", CpuBackendKind::Blas))
            }
        }
    }?;

    if let Some(shape) = restore_shape {
        context.reshape_tensor(&result, &shape)
    } else {
        Ok(result)
    }
}

fn solve_from_views_entered(
    provider: CpuLinalgProvider,
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: TensorView<'_>,
    b: TensorView<'_>,
) -> tenferro_tensor::Result<Tensor> {
    match provider {
        CpuLinalgProvider::Faer => {
            #[cfg(feature = "cpu-faer")]
            {
                match (a, b) {
                    (TensorView::F32(a), TensorView::F32(b)) => {
                        linalg::faer::solve_from_views(context, buffers, a, b, false)
                            .map(Tensor::F32)
                    }
                    (TensorView::F64(a), TensorView::F64(b)) => {
                        linalg::faer::solve_from_views(context, buffers, a, b, false)
                            .map(Tensor::F64)
                    }
                    (TensorView::C32(a), TensorView::C32(b)) => {
                        linalg::faer::solve_from_views(context, buffers, a, b, false)
                            .map(Tensor::C32)
                    }
                    (TensorView::C64(a), TensorView::C64(b)) => {
                        linalg::faer::solve_from_views(context, buffers, a, b, false)
                            .map(Tensor::C64)
                    }
                    _ => Err(Error::invalid_argument(
                        "solve",
                        "inputs",
                        "solve inputs must have the same dtype",
                    )),
                }
            }
            #[cfg(not(feature = "cpu-faer"))]
            {
                let _ = (context, buffers, a, b);
                Err(unsupported_provider("solve", CpuBackendKind::Faer))
            }
        }
        CpuLinalgProvider::Blas => {
            #[cfg(feature = "cpu-blas")]
            {
                let _ = context;
                match (a, b) {
                    (TensorView::F32(a), TensorView::F32(b)) => {
                        linalg::blas::solve_from_views(buffers, a, b, false).map(Tensor::F32)
                    }
                    (TensorView::F64(a), TensorView::F64(b)) => {
                        linalg::blas::solve_from_views(buffers, a, b, false).map(Tensor::F64)
                    }
                    (TensorView::C32(a), TensorView::C32(b)) => {
                        linalg::blas::solve_from_views(buffers, a, b, false).map(Tensor::C32)
                    }
                    (TensorView::C64(a), TensorView::C64(b)) => {
                        linalg::blas::solve_from_views(buffers, a, b, false).map(Tensor::C64)
                    }
                    _ => Err(Error::invalid_argument(
                        "solve",
                        "inputs",
                        "solve inputs must have the same dtype",
                    )),
                }
            }
            #[cfg(not(feature = "cpu-blas"))]
            {
                let _ = (context, buffers, a, b);
                Err(unsupported_provider("solve", CpuBackendKind::Blas))
            }
        }
    }
}

fn solve_read_into_entered(
    provider: CpuLinalgProvider,
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: TensorView<'_>,
    b: TensorView<'_>,
    out: TensorWrite<'_>,
) -> tenferro_tensor::Result<()> {
    let out = tensor_write_view(out);
    match provider {
        CpuLinalgProvider::Faer => {
            #[cfg(feature = "cpu-faer")]
            {
                match (a, b, out) {
                    (TensorView::F32(a), TensorView::F32(b), TensorViewMut::F32(mut out)) => {
                        linalg::faer::solve_into(context, buffers, a, b, &mut out, false)
                    }
                    (TensorView::F64(a), TensorView::F64(b), TensorViewMut::F64(mut out)) => {
                        linalg::faer::solve_into(context, buffers, a, b, &mut out, false)
                    }
                    (TensorView::C32(a), TensorView::C32(b), TensorViewMut::C32(mut out)) => {
                        linalg::faer::solve_into(context, buffers, a, b, &mut out, false)
                    }
                    (TensorView::C64(a), TensorView::C64(b), TensorViewMut::C64(mut out)) => {
                        linalg::faer::solve_into(context, buffers, a, b, &mut out, false)
                    }
                    _ => Err(Error::invalid_argument(
                        "solve_read_into",
                        "out",
                        "destination dtype does not match the solve inputs",
                    )),
                }
            }
            #[cfg(not(feature = "cpu-faer"))]
            {
                let _ = (context, buffers, a, b, out);
                Err(unsupported_provider(
                    "solve_read_into",
                    CpuBackendKind::Faer,
                ))
            }
        }
        CpuLinalgProvider::Blas => {
            #[cfg(feature = "cpu-blas")]
            {
                let _ = context;
                match (a, b, out) {
                    (TensorView::F32(a), TensorView::F32(b), TensorViewMut::F32(mut out)) => {
                        linalg::blas::solve_into(buffers, a, b, &mut out, false)
                    }
                    (TensorView::F64(a), TensorView::F64(b), TensorViewMut::F64(mut out)) => {
                        linalg::blas::solve_into(buffers, a, b, &mut out, false)
                    }
                    (TensorView::C32(a), TensorView::C32(b), TensorViewMut::C32(mut out)) => {
                        linalg::blas::solve_into(buffers, a, b, &mut out, false)
                    }
                    (TensorView::C64(a), TensorView::C64(b), TensorViewMut::C64(mut out)) => {
                        linalg::blas::solve_into(buffers, a, b, &mut out, false)
                    }
                    _ => Err(Error::invalid_argument(
                        "solve_read_into",
                        "out",
                        "destination dtype does not match the solve inputs",
                    )),
                }
            }
            #[cfg(not(feature = "cpu-blas"))]
            {
                let _ = (context, buffers, a, b, out);
                Err(unsupported_provider(
                    "solve_read_into",
                    CpuBackendKind::Blas,
                ))
            }
        }
    }
}

fn tensor_write_view(out: TensorWrite<'_>) -> TensorViewMut<'_> {
    match out {
        TensorWrite::Tensor(tensor) => match tensor {
            Tensor::F32(tensor) => TensorViewMut::F32(tensor.as_view_mut()),
            Tensor::F64(tensor) => TensorViewMut::F64(tensor.as_view_mut()),
            Tensor::I32(tensor) => TensorViewMut::I32(tensor.as_view_mut()),
            Tensor::I64(tensor) => TensorViewMut::I64(tensor.as_view_mut()),
            Tensor::Bool(tensor) => TensorViewMut::Bool(tensor.as_view_mut()),
            Tensor::C32(tensor) => TensorViewMut::C32(tensor.as_view_mut()),
            Tensor::C64(tensor) => TensorViewMut::C64(tensor.as_view_mut()),
        },
        TensorWrite::View(view) => view,
    }
}

fn cholesky_entered(
    provider: CpuLinalgProvider,
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Tensor> {
    match provider {
        CpuLinalgProvider::Faer => {
            #[cfg(feature = "cpu-faer")]
            {
                match input {
                    Tensor::F32(t) => linalg::faer::cholesky(context, buffers, t).map(Tensor::F32),
                    Tensor::F64(t) => linalg::faer::cholesky(context, buffers, t).map(Tensor::F64),
                    Tensor::C32(t) => linalg::faer::cholesky(context, buffers, t).map(Tensor::C32),
                    Tensor::C64(t) => linalg::faer::cholesky(context, buffers, t).map(Tensor::C64),
                    _ => Err(unsupported_dtype("cholesky", input.dtype())),
                }
            }
            #[cfg(not(feature = "cpu-faer"))]
            {
                let _ = (context, buffers, input);
                Err(unsupported_provider("cholesky", CpuBackendKind::Faer))
            }
        }
        CpuLinalgProvider::Blas => {
            #[cfg(feature = "cpu-blas")]
            {
                let _ = context;
                match input {
                    Tensor::F32(t) => linalg::blas::cholesky(buffers, t).map(Tensor::F32),
                    Tensor::F64(t) => linalg::blas::cholesky(buffers, t).map(Tensor::F64),
                    Tensor::C32(t) => linalg::blas::cholesky(buffers, t).map(Tensor::C32),
                    Tensor::C64(t) => linalg::blas::cholesky(buffers, t).map(Tensor::C64),
                    _ => Err(unsupported_dtype("cholesky", input.dtype())),
                }
            }
            #[cfg(not(feature = "cpu-blas"))]
            {
                let _ = (context, buffers, input);
                Err(unsupported_provider("cholesky", CpuBackendKind::Blas))
            }
        }
    }
}

fn lu_entered(
    provider: CpuLinalgProvider,
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match provider {
        CpuLinalgProvider::Faer => {
            #[cfg(feature = "cpu-faer")]
            {
                match input {
                    Tensor::F32(t) => linalg::faer::lu(context, buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                    Tensor::F64(t) => linalg::faer::lu(context, buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                    Tensor::C32(t) => linalg::faer::lu(context, buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
                    Tensor::C64(t) => linalg::faer::lu(context, buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
                    _ => Err(unsupported_dtype("lu", input.dtype())),
                }
            }
            #[cfg(not(feature = "cpu-faer"))]
            {
                let _ = (context, buffers, input);
                Err(unsupported_provider("lu", CpuBackendKind::Faer))
            }
        }
        CpuLinalgProvider::Blas => {
            #[cfg(feature = "cpu-blas")]
            {
                let _ = context;
                match input {
                    Tensor::F32(t) => linalg::blas::lu(buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                    Tensor::F64(t) => linalg::blas::lu(buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                    Tensor::C32(t) => linalg::blas::lu(buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
                    Tensor::C64(t) => linalg::blas::lu(buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
                    _ => Err(unsupported_dtype("lu", input.dtype())),
                }
            }
            #[cfg(not(feature = "cpu-blas"))]
            {
                let _ = (context, buffers, input);
                Err(unsupported_provider("lu", CpuBackendKind::Blas))
            }
        }
    }
}

fn full_piv_lu_entered(
    provider: CpuLinalgProvider,
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match provider {
        CpuLinalgProvider::Faer => {
            #[cfg(feature = "cpu-faer")]
            {
                match input {
                    Tensor::F32(t) => linalg::faer::full_piv_lu(context, buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                    Tensor::F64(t) => linalg::faer::full_piv_lu(context, buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                    Tensor::C32(t) => linalg::faer::full_piv_lu(context, buffers, t)
                        .and_then(full_piv_lu_c32_outputs_to_public_tensors),
                    Tensor::C64(t) => linalg::faer::full_piv_lu(context, buffers, t)
                        .and_then(full_piv_lu_c64_outputs_to_public_tensors),
                    _ => Err(unsupported_dtype("full_piv_lu", input.dtype())),
                }
            }
            #[cfg(not(feature = "cpu-faer"))]
            {
                let _ = (context, buffers, input);
                Err(unsupported_provider("full_piv_lu", CpuBackendKind::Faer))
            }
        }
        CpuLinalgProvider::Blas => {
            #[cfg(feature = "cpu-blas")]
            {
                let _ = context;
                match input {
                    Tensor::F32(t) => linalg::blas::full_piv_lu(buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                    Tensor::F64(t) => linalg::blas::full_piv_lu(buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                    Tensor::C32(t) => linalg::blas::full_piv_lu(buffers, t)
                        .and_then(full_piv_lu_c32_outputs_to_public_tensors),
                    Tensor::C64(t) => linalg::blas::full_piv_lu(buffers, t)
                        .and_then(full_piv_lu_c64_outputs_to_public_tensors),
                    _ => Err(unsupported_dtype("full_piv_lu", input.dtype())),
                }
            }
            #[cfg(not(feature = "cpu-blas"))]
            {
                let _ = (context, buffers, input);
                Err(unsupported_provider("full_piv_lu", CpuBackendKind::Blas))
            }
        }
    }
}

fn svd_entered(
    provider: CpuLinalgProvider,
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match provider {
        CpuLinalgProvider::Faer => {
            #[cfg(feature = "cpu-faer")]
            {
                match input {
                    Tensor::F32(t) => linalg::faer::svd(context, buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                    Tensor::F64(t) => linalg::faer::svd(context, buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                    Tensor::C32(t) => linalg::faer::svd(context, buffers, t)
                        .and_then(svd_c32_outputs_to_public_tensors),
                    Tensor::C64(t) => linalg::faer::svd(context, buffers, t)
                        .and_then(svd_c64_outputs_to_public_tensors),
                    _ => Err(unsupported_dtype("svd", input.dtype())),
                }
            }
            #[cfg(not(feature = "cpu-faer"))]
            {
                let _ = (context, buffers, input);
                Err(unsupported_provider("svd", CpuBackendKind::Faer))
            }
        }
        CpuLinalgProvider::Blas => {
            #[cfg(feature = "cpu-blas")]
            {
                let _ = context;
                match input {
                    Tensor::F32(t) => linalg::blas::svd(buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                    Tensor::F64(t) => linalg::blas::svd(buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                    Tensor::C32(t) => {
                        linalg::blas::svd(buffers, t).and_then(svd_c32_outputs_to_public_tensors)
                    }
                    Tensor::C64(t) => {
                        linalg::blas::svd(buffers, t).and_then(svd_c64_outputs_to_public_tensors)
                    }
                    _ => Err(unsupported_dtype("svd", input.dtype())),
                }
            }
            #[cfg(not(feature = "cpu-blas"))]
            {
                let _ = (context, buffers, input);
                Err(unsupported_provider("svd", CpuBackendKind::Blas))
            }
        }
    }
}

fn qr_entered(
    provider: CpuLinalgProvider,
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match provider {
        CpuLinalgProvider::Faer => {
            #[cfg(feature = "cpu-faer")]
            {
                match input {
                    Tensor::F32(t) => linalg::faer::qr(context, buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                    Tensor::F64(t) => linalg::faer::qr(context, buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                    Tensor::C32(t) => linalg::faer::qr(context, buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
                    Tensor::C64(t) => linalg::faer::qr(context, buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
                    _ => Err(unsupported_dtype("qr", input.dtype())),
                }
            }
            #[cfg(not(feature = "cpu-faer"))]
            {
                let _ = (context, buffers, input);
                Err(unsupported_provider("qr", CpuBackendKind::Faer))
            }
        }
        CpuLinalgProvider::Blas => {
            #[cfg(feature = "cpu-blas")]
            {
                let _ = context;
                match input {
                    Tensor::F32(t) => linalg::blas::qr(buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                    Tensor::F64(t) => linalg::blas::qr(buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                    Tensor::C32(t) => linalg::blas::qr(buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
                    Tensor::C64(t) => linalg::blas::qr(buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
                    _ => Err(unsupported_dtype("qr", input.dtype())),
                }
            }
            #[cfg(not(feature = "cpu-blas"))]
            {
                let _ = (context, buffers, input);
                Err(unsupported_provider("qr", CpuBackendKind::Blas))
            }
        }
    }
}

fn eigh_entered(
    provider: CpuLinalgProvider,
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match provider {
        CpuLinalgProvider::Faer => {
            #[cfg(feature = "cpu-faer")]
            {
                match input {
                    Tensor::F32(t) => linalg::faer::eigh(context, buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                    Tensor::F64(t) => linalg::faer::eigh(context, buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                    Tensor::C32(t) => linalg::faer::eigh(context, buffers, t)
                        .and_then(eigh_c32_outputs_to_public_tensors),
                    Tensor::C64(t) => linalg::faer::eigh(context, buffers, t)
                        .and_then(eigh_c64_outputs_to_public_tensors),
                    _ => Err(unsupported_dtype("eigh", input.dtype())),
                }
            }
            #[cfg(not(feature = "cpu-faer"))]
            {
                let _ = (context, buffers, input);
                Err(unsupported_provider("eigh", CpuBackendKind::Faer))
            }
        }
        CpuLinalgProvider::Blas => {
            #[cfg(feature = "cpu-blas")]
            {
                let _ = context;
                match input {
                    Tensor::F32(t) => linalg::blas::eigh(buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                    Tensor::F64(t) => linalg::blas::eigh(buffers, t)
                        .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                    Tensor::C32(t) => {
                        linalg::blas::eigh(buffers, t).and_then(eigh_c32_outputs_to_public_tensors)
                    }
                    Tensor::C64(t) => {
                        linalg::blas::eigh(buffers, t).and_then(eigh_c64_outputs_to_public_tensors)
                    }
                    _ => Err(unsupported_dtype("eigh", input.dtype())),
                }
            }
            #[cfg(not(feature = "cpu-blas"))]
            {
                let _ = (context, buffers, input);
                Err(unsupported_provider("eigh", CpuBackendKind::Blas))
            }
        }
    }
}

fn eig_entered(
    provider: CpuLinalgProvider,
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match provider {
        CpuLinalgProvider::Faer => {
            #[cfg(feature = "cpu-faer")]
            {
                linalg::faer::eig(context, buffers, input)
            }
            #[cfg(not(feature = "cpu-faer"))]
            {
                let _ = (context, buffers, input);
                Err(unsupported_provider("eig", CpuBackendKind::Faer))
            }
        }
        CpuLinalgProvider::Blas => {
            #[cfg(feature = "cpu-blas")]
            {
                let _ = context;
                linalg::blas::eig(buffers, input)
            }
            #[cfg(not(feature = "cpu-blas"))]
            {
                let _ = (context, buffers, input);
                Err(unsupported_provider("eig", CpuBackendKind::Blas))
            }
        }
    }
}

#[cfg(feature = "cpu-faer")]
fn faer_strided_read_ok(input: &TensorRead<'_>) -> bool {
    match input {
        TensorRead::Tensor(Tensor::F32(tensor)) => linalg::faer::faer_strided_ok(&tensor.as_view()),
        TensorRead::Tensor(Tensor::F64(tensor)) => linalg::faer::faer_strided_ok(&tensor.as_view()),
        TensorRead::Tensor(Tensor::C32(tensor)) => linalg::faer::faer_strided_ok(&tensor.as_view()),
        TensorRead::Tensor(Tensor::C64(tensor)) => linalg::faer::faer_strided_ok(&tensor.as_view()),
        TensorRead::View(TensorView::F32(view)) => linalg::faer::faer_strided_ok(view),
        TensorRead::View(TensorView::F64(view)) => linalg::faer::faer_strided_ok(view),
        TensorRead::View(TensorView::C32(view)) => linalg::faer::faer_strided_ok(view),
        TensorRead::View(TensorView::C64(view)) => linalg::faer::faer_strided_ok(view),
        TensorRead::Tensor(Tensor::I32(_))
        | TensorRead::Tensor(Tensor::I64(_))
        | TensorRead::Tensor(Tensor::Bool(_))
        | TensorRead::View(TensorView::I32(_))
        | TensorRead::View(TensorView::I64(_))
        | TensorRead::View(TensorView::Bool(_)) => false,
    }
}

#[cfg(feature = "cpu-faer")]
fn svd_faer_view_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: TensorView<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match input {
        TensorView::F32(view) => linalg::faer::svd_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
        TensorView::F64(view) => linalg::faer::svd_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
        TensorView::C32(view) => linalg::faer::svd_view(context, buffers, view)
            .and_then(svd_c32_outputs_to_public_tensors),
        TensorView::C64(view) => linalg::faer::svd_view(context, buffers, view)
            .and_then(svd_c64_outputs_to_public_tensors),
        unsupported => Err(unsupported_dtype("svd", unsupported.dtype())),
    }
}

#[cfg(feature = "cpu-faer")]
fn qr_faer_view_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: TensorView<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match input {
        TensorView::F32(view) => linalg::faer::qr_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
        TensorView::F64(view) => linalg::faer::qr_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
        TensorView::C32(view) => linalg::faer::qr_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
        TensorView::C64(view) => linalg::faer::qr_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
        unsupported => Err(unsupported_dtype("qr", unsupported.dtype())),
    }
}

#[cfg(feature = "cpu-faer")]
fn eigh_faer_view_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: TensorView<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match input {
        TensorView::F32(view) => linalg::faer::eigh_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
        TensorView::F64(view) => linalg::faer::eigh_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
        TensorView::C32(view) => linalg::faer::eigh_view(context, buffers, view)
            .and_then(eigh_c32_outputs_to_public_tensors),
        TensorView::C64(view) => linalg::faer::eigh_view(context, buffers, view)
            .and_then(eigh_c64_outputs_to_public_tensors),
        unsupported => Err(unsupported_dtype("eigh", unsupported.dtype())),
    }
}

#[cfg(feature = "cpu-faer")]
fn cholesky_faer_view_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: TensorView<'_>,
) -> tenferro_tensor::Result<Tensor> {
    match input {
        TensorView::F32(view) => {
            linalg::faer::cholesky_view(context, buffers, view).map(Tensor::F32)
        }
        TensorView::F64(view) => {
            linalg::faer::cholesky_view(context, buffers, view).map(Tensor::F64)
        }
        TensorView::C32(view) => {
            linalg::faer::cholesky_view(context, buffers, view).map(Tensor::C32)
        }
        TensorView::C64(view) => {
            linalg::faer::cholesky_view(context, buffers, view).map(Tensor::C64)
        }
        unsupported => Err(unsupported_dtype("cholesky", unsupported.dtype())),
    }
}

#[cfg(feature = "cpu-faer")]
fn lu_faer_view_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: TensorView<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match input {
        TensorView::F32(view) => linalg::faer::lu_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
        TensorView::F64(view) => linalg::faer::lu_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
        TensorView::C32(view) => linalg::faer::lu_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
        TensorView::C64(view) => linalg::faer::lu_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
        unsupported => Err(unsupported_dtype("lu", unsupported.dtype())),
    }
}

#[cfg(feature = "cpu-faer")]
fn full_piv_lu_faer_view_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: TensorView<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match input {
        TensorView::F32(view) => linalg::faer::full_piv_lu_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
        TensorView::F64(view) => linalg::faer::full_piv_lu_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
        TensorView::C32(view) => linalg::faer::full_piv_lu_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
        TensorView::C64(view) => linalg::faer::full_piv_lu_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
        unsupported => Err(unsupported_dtype("full_piv_lu", unsupported.dtype())),
    }
}

fn ensure_host_typed_tensor<T: TensorScalar>(
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
    ensure_supported_linalg_dtype(op, lhs)
}

fn ensure_supported_linalg_dtype(op: &'static str, dtype: DType) -> tenferro_tensor::Result<()> {
    match dtype {
        DType::F32 | DType::F64 | DType::C32 | DType::C64 => Ok(()),
        DType::I32 | DType::I64 | DType::Bool => Err(unsupported_dtype(op, dtype)),
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

fn apply_lu_pivots_typed<T: Clone + TensorScalar>(
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
                data.push(input_data[batch_offset + source_row + col * rows]);
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
