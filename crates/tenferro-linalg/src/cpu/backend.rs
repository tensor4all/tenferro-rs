use crate::backend::{unsupported_dtype, CompactQrResult, LinalgBackend};
#[derive(Clone, Copy, Debug)]
struct TriangularSolveOptions {
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
}
use crate::extension::apply_qr_gauge;
use crate::rank_revealing_qr::validate_rank_revealing_qr_options;

use super::linalg;
use tenferro_cpu::same_variant_pair;

use num_complex::{Complex32, Complex64};
use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_cpu::{CpuExecSession, CpuExecutionContext};
use tenferro_tensor::{
    validate::validate_nonsingular_u, AllocationDomainId, DType, Error, HostAccessError,
    MemoryKind, SharedTensorAllocationDomain, Tensor, TensorRead, TensorScalar, TensorStructural,
    TensorView, TensorViewMut, TensorWrite, TypedTensor,
};

/// The Rust scalar type behind a preset variant name a macro received.
macro_rules! preset_scalar {
    (F32) => {
        f32
    };
    (F64) => {
        f64
    };
    (I32) => {
        i32
    };
    (I64) => {
        i64
    };
    (Bool) => {
        bool
    };
    (C32) => {
        num_complex::Complex32
    };
    (C64) => {
        num_complex::Complex64
    };
}
trait FreshLinalgOutput {
    fn tag_fresh(&mut self, domain: tenferro_tensor::CpuDomainId);
}

impl FreshLinalgOutput for Tensor {
    fn tag_fresh(&mut self, domain: tenferro_tensor::CpuDomainId) {
        match self.dtype() {
            DType::F32 => tag_fresh_typed::<f32>(self, domain),
            DType::F64 => tag_fresh_typed::<f64>(self, domain),
            DType::I32 => tag_fresh_typed::<i32>(self, domain),
            DType::I64 => tag_fresh_typed::<i64>(self, domain),
            DType::Bool => tag_fresh_typed::<bool>(self, domain),
            DType::C32 => tag_fresh_typed::<Complex32>(self, domain),
            DType::C64 => tag_fresh_typed::<Complex64>(self, domain),
            // A caller-owned payload has no pooled storage to tag, and a tag the accessor cannot
            // recover leaves the placement untouched, which is the same outcome.
            DType::External(_) => {}
        }
    }
}

/// Mark a freshly allocated output with the CPU domain its pool belongs to.
fn tag_fresh_typed<T: tenferro_tensor::TensorScalar>(
    output: &mut Tensor,
    domain: tenferro_tensor::CpuDomainId,
) {
    if let Some(tensor) = output.as_typed_mut::<T>() {
        tensor.set_cpu_affinity(Some(domain));
    }
}

impl FreshLinalgOutput for Vec<Tensor> {
    fn tag_fresh(&mut self, domain: tenferro_tensor::CpuDomainId) {
        for output in self {
            output.tag_fresh(domain);
        }
    }
}

impl FreshLinalgOutput for CompactQrResult {
    fn tag_fresh(&mut self, domain: tenferro_tensor::CpuDomainId) {
        self.packed.tag_fresh(domain);
        self.coeff.tag_fresh(domain);
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
        self.with_linalg_pool_fresh(move |context, buffers| {
            if tensor_uses_backend_storage(input)
                && let Some(domain) = domain.as_deref()
            {
                return managed_cholesky(context, buffers, TensorRead::from_tensor(input), domain);
            }
            ensure_host_tensor("cholesky", input)?;
            cholesky_entered(context, buffers, input)
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
        let options = TriangularSolveOptions {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        };
        self.with_linalg_pool_fresh(|context, buffers| {
            triangular_solve_entered(context, buffers, a, b, options)
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
        let options = TriangularSolveOptions {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        };
        self.with_linalg_pool_fresh(move |context, buffers| {
            // Both operands must be faer-eligible for the direct path: `a`
            // reaches faer as a strided `MatRef` and `b` is gathered straight
            // into the destructible right-hand side. If either is ineligible,
            // the scoped materializer below packs only what it must.
            #[cfg(feature = "native")]
            if faer_strided_read_ok(&a) && faer_rhs_read_ok(&b) {
                return triangular_solve_faer_view_entered(
                    context,
                    buffers,
                    a.tensor_view(),
                    b.tensor_view(),
                    options,
                );
            }
            context.with_materialized_tensor_read(buffers, "triangular_solve", a, |a, buffers| {
                context.with_materialized_tensor_read(
                    buffers,
                    "triangular_solve",
                    b,
                    |b, buffers| triangular_solve_entered(context, buffers, a, b, options),
                )
            })
        })
    }

    fn lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor("lu", input)?;
        self.with_linalg_pool_fresh(|context, buffers| lu_entered(context, buffers, input))
    }

    fn lu_factor(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor("lu_factor", input)?;
        {
            #[cfg(feature = "native")]
            {
                #[cfg(feature = "native")]
                {
                    self.with_linalg_pool_fresh(|ctx, buffers| match input.dtype() {
                        DType::F32 => {
                            let t = input
                                .as_typed::<f32>()
                                .ok_or_else(|| unsupported_dtype("lu_factor", input.dtype()))?;
                            linalg::faer::lu_factor(ctx, buffers, t).map(|(lu, pivots, parity)| {
                                vec![
                                    Tensor::from_typed::<f32>(lu),
                                    Tensor::from_typed::<i32>(pivots),
                                    Tensor::from_typed::<f32>(parity),
                                ]
                            })
                        }
                        DType::F64 => {
                            let t = input
                                .as_typed::<f64>()
                                .ok_or_else(|| unsupported_dtype("lu_factor", input.dtype()))?;
                            linalg::faer::lu_factor(ctx, buffers, t).map(|(lu, pivots, parity)| {
                                vec![
                                    Tensor::from_typed::<f64>(lu),
                                    Tensor::from_typed::<i32>(pivots),
                                    Tensor::from_typed::<f64>(parity),
                                ]
                            })
                        }
                        DType::C32 => {
                            let t = input
                                .as_typed::<Complex32>()
                                .ok_or_else(|| unsupported_dtype("lu_factor", input.dtype()))?;
                            linalg::faer::lu_factor(ctx, buffers, t).map(|(lu, pivots, parity)| {
                                vec![
                                    Tensor::from_typed::<Complex32>(lu),
                                    Tensor::from_typed::<i32>(pivots),
                                    Tensor::from_typed::<Complex32>(parity),
                                ]
                            })
                        }
                        DType::C64 => {
                            let t = input
                                .as_typed::<Complex64>()
                                .ok_or_else(|| unsupported_dtype("lu_factor", input.dtype()))?;
                            linalg::faer::lu_factor(ctx, buffers, t).map(|(lu, pivots, parity)| {
                                vec![
                                    Tensor::from_typed::<Complex64>(lu),
                                    Tensor::from_typed::<i32>(pivots),
                                    Tensor::from_typed::<Complex64>(parity),
                                ]
                            })
                        }
                        _ => Err(unsupported_dtype("lu_factor", input.dtype())),
                    })
                }
            }
            #[cfg(feature = "blas")]
            {
                #[cfg(feature = "blas")]
                {
                    self.with_linalg_pool_fresh(|ctx, buffers| match input.dtype() {
                        DType::F32 => {
                            let t = input
                                .as_typed::<f32>()
                                .ok_or_else(|| unsupported_dtype("lu_factor", input.dtype()))?;
                            linalg::blas::lu_factor(ctx, buffers, t).map(|(lu, pivots, parity)| {
                                vec![
                                    Tensor::from_typed::<f32>(lu),
                                    Tensor::from_typed::<i32>(pivots),
                                    Tensor::from_typed::<f32>(parity),
                                ]
                            })
                        }
                        DType::F64 => {
                            let t = input
                                .as_typed::<f64>()
                                .ok_or_else(|| unsupported_dtype("lu_factor", input.dtype()))?;
                            linalg::blas::lu_factor(ctx, buffers, t).map(|(lu, pivots, parity)| {
                                vec![
                                    Tensor::from_typed::<f64>(lu),
                                    Tensor::from_typed::<i32>(pivots),
                                    Tensor::from_typed::<f64>(parity),
                                ]
                            })
                        }
                        DType::C32 => {
                            let t = input
                                .as_typed::<Complex32>()
                                .ok_or_else(|| unsupported_dtype("lu_factor", input.dtype()))?;
                            linalg::blas::lu_factor(ctx, buffers, t).map(|(lu, pivots, parity)| {
                                vec![
                                    Tensor::from_typed::<Complex32>(lu),
                                    Tensor::from_typed::<i32>(pivots),
                                    Tensor::from_typed::<Complex32>(parity),
                                ]
                            })
                        }
                        DType::C64 => {
                            let t = input
                                .as_typed::<Complex64>()
                                .ok_or_else(|| unsupported_dtype("lu_factor", input.dtype()))?;
                            linalg::blas::lu_factor(ctx, buffers, t).map(|(lu, pivots, parity)| {
                                vec![
                                    Tensor::from_typed::<Complex64>(lu),
                                    Tensor::from_typed::<i32>(pivots),
                                    Tensor::from_typed::<Complex64>(parity),
                                ]
                            })
                        }
                        _ => Err(unsupported_dtype("lu_factor", input.dtype())),
                    })
                }
            }
        }
    }

    fn full_piv_lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor("full_piv_lu", input)?;
        self.with_linalg_pool_fresh(|context, buffers| full_piv_lu_entered(context, buffers, input))
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
            return self.with_linalg_pool_fresh(|_, _| zeros_like_tensor(b));
        }

        let (rhs, restore_shape) = if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
            (
                self.reshape_read(TensorRead::from_tensor(b), &matrix_rhs_shape)?,
                Some(b.shape().to_vec()),
            )
        } else {
            (b.duplicate()?, None)
        };

        let result = {
            #[cfg(feature = "native")]
            {
                #[cfg(feature = "native")]
                {
                    self.with_linalg_pool_fresh(|ctx, buffers| {
                        same_variant_pair!(
                            a,
                            &rhs,
                            |a, b| {
                                linalg::faer::full_piv_lu_solve(ctx, buffers, a, b, transpose_a)
                            },
                            unsupported_pair("full_piv_lu_solve", a, &rhs)
                        )
                    })
                }
            }
            #[cfg(feature = "blas")]
            {
                #[cfg(feature = "blas")]
                {
                    self.with_linalg_pool_fresh(|_, buffers| {
                        same_variant_pair!(
                            a,
                            &rhs,
                            |a, b| { linalg::blas::full_piv_lu_solve(buffers, a, b, transpose_a) },
                            unsupported_pair("full_piv_lu_solve", a, &rhs)
                        )
                    })
                }
            }
        }?;

        if let Some(shape) = restore_shape {
            self.reshape_read(TensorRead::from_tensor(&result), &shape)
        } else {
            Ok(result)
        }
    }

    fn svd(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor("svd", input)?;
        self.with_linalg_pool_fresh(|context, buffers| svd_entered(context, buffers, input))
    }

    fn svd_full(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor("svd_full", input)?;
        self.with_linalg_pool_fresh(|context, buffers| svd_full_entered(context, buffers, input))
    }

    fn svd_full_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor_read("svd_full", &input)?;
        ensure_supported_linalg_dtype("svd_full", input.dtype())?;
        self.with_linalg_pool_fresh(move |context, buffers| {
            #[cfg(feature = "native")]
            if faer_strided_read_ok(&input) {
                return svd_full_faer_view_entered(context, buffers, input.tensor_view());
            }
            context.with_materialized_tensor_read(buffers, "svd_full", input, |input, buffers| {
                svd_full_entered(context, buffers, input)
            })
        })
    }

    fn svd_values(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        ensure_host_tensor("svd_values", input)?;
        self.with_linalg_pool_fresh(|context, buffers| svd_values_entered(context, buffers, input))
    }

    fn svd_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor_read("svd", &input)?;
        ensure_supported_linalg_dtype("svd", input.dtype())?;
        self.with_linalg_pool_fresh(move |context, buffers| {
            #[cfg(feature = "native")]
            if faer_strided_read_ok(&input) {
                return svd_faer_view_entered(context, buffers, input.tensor_view());
            }
            context.with_materialized_tensor_read(buffers, "svd", input, |input, buffers| {
                svd_entered(context, buffers, input)
            })
        })
    }

    fn svd_values_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        ensure_host_tensor_read("svd_values", &input)?;
        ensure_supported_linalg_dtype("svd_values", input.dtype())?;
        self.with_linalg_pool_fresh(move |context, buffers| {
            #[cfg(feature = "native")]
            if faer_strided_read_ok(&input) {
                return svd_values_faer_view_entered(context, buffers, input.tensor_view());
            }
            context.with_materialized_tensor_read(buffers, "svd_values", input, |input, buffers| {
                svd_values_entered(context, buffers, input)
            })
        })
    }

    fn qr(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor("qr", input)?;
        self.with_linalg_pool_fresh(|context, buffers| qr_entered(context, buffers, input))
    }

    fn qr_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor_read("qr", &input)?;
        ensure_supported_linalg_dtype("qr", input.dtype())?;
        self.with_linalg_pool_fresh(move |context, buffers| {
            #[cfg(feature = "native")]
            if faer_strided_read_ok(&input) {
                return qr_faer_view_entered(context, buffers, input.tensor_view());
            }
            context.with_materialized_tensor_read(buffers, "qr", input, |input, buffers| {
                qr_entered(context, buffers, input)
            })
        })
    }

    fn rank_revealing_qr(
        &mut self,
        input: &Tensor,
        options: crate::RankRevealingQrOptions,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        validate_rank_revealing_qr_options("rank_revealing_qr", options)?;
        ensure_host_tensor("rank_revealing_qr", input)?;
        ensure_supported_linalg_dtype("rank_revealing_qr", input.dtype())?;
        self.with_linalg_pool_fresh(|context, buffers| {
            rank_revealing_qr_entered(context, buffers, input, options)
        })
    }

    fn rank_revealing_qr_read(
        &mut self,
        input: TensorRead<'_>,
        options: crate::RankRevealingQrOptions,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        validate_rank_revealing_qr_options("rank_revealing_qr", options)?;
        ensure_host_tensor_read("rank_revealing_qr", &input)?;
        ensure_supported_linalg_dtype("rank_revealing_qr", input.dtype())?;
        self.with_linalg_pool_fresh(move |context, buffers| {
            #[cfg(feature = "native")]
            if faer_strided_read_ok(&input) {
                return rank_revealing_qr_faer_view_entered(
                    context,
                    buffers,
                    input.tensor_view(),
                    options,
                );
            }
            context.with_materialized_tensor_read(
                buffers,
                "rank_revealing_qr",
                input,
                |input, buffers| rank_revealing_qr_entered(context, buffers, input, options),
            )
        })
    }

    fn householder_qr(&mut self, input: &Tensor) -> tenferro_tensor::Result<CompactQrResult> {
        ensure_host_tensor("householder_qr", input)?;
        self.with_linalg_pool_fresh(|context, buffers| {
            householder_qr_entered(context, buffers, input)
        })
    }

    fn householder_qr_from_factors(
        &mut self,
        q: &Tensor,
        r: &Tensor,
    ) -> tenferro_tensor::Result<CompactQrResult> {
        ensure_host_tensor("householder_qr_from_factors", q)?;
        ensure_host_tensor("householder_qr_from_factors", r)?;
        self.with_linalg_pool_fresh(|context, buffers| {
            householder_qr_from_factors_entered(context, buffers, q, r)
        })
    }

    fn householder_qr_append(
        &mut self,
        packed: &Tensor,
        coeff: &Tensor,
        block: &Tensor,
    ) -> tenferro_tensor::Result<CompactQrResult> {
        ensure_host_tensor("householder_qr_append", packed)?;
        ensure_host_tensor("householder_qr_append", coeff)?;
        ensure_host_tensor("householder_qr_append", block)?;
        self.with_linalg_pool_fresh(|context, buffers| {
            householder_qr_append_entered(context, buffers, packed, coeff, block)
        })
    }

    fn householder_qr_r(
        &mut self,
        packed: &Tensor,
        coeff: &Tensor,
        options: crate::QrOptions,
    ) -> tenferro_tensor::Result<Tensor> {
        ensure_host_tensor("householder_qr_r", packed)?;
        ensure_host_tensor("householder_qr_r", coeff)?;
        self.with_linalg_pool_fresh(|context, buffers| {
            householder_qr_r_entered(context, buffers, packed, coeff, options)
        })
    }

    fn householder_qr_q_columns(
        &mut self,
        packed: &Tensor,
        coeff: &Tensor,
        columns: std::ops::Range<usize>,
        options: crate::QrOptions,
    ) -> tenferro_tensor::Result<Tensor> {
        ensure_host_tensor("householder_qr_q_columns", packed)?;
        ensure_host_tensor("householder_qr_q_columns", coeff)?;
        self.with_linalg_pool_fresh(|context, buffers| {
            householder_qr_q_columns_entered(context, buffers, packed, coeff, columns, options)
        })
    }

    fn eigh(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor("eigh", input)?;
        self.with_linalg_pool_fresh(|context, buffers| eigh_entered(context, buffers, input))
    }

    fn eigh_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor_read("eigh", &input)?;
        ensure_supported_linalg_dtype("eigh", input.dtype())?;
        self.with_linalg_pool_fresh(move |context, buffers| {
            #[cfg(feature = "native")]
            if faer_strided_read_ok(&input) {
                return eigh_faer_view_entered(context, buffers, input.tensor_view());
            }
            context.with_materialized_tensor_read(buffers, "eigh", input, |input, buffers| {
                eigh_entered(context, buffers, input)
            })
        })
    }

    fn eigh_values_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        ensure_host_tensor_read("eigh_values", &input)?;
        ensure_supported_linalg_dtype("eigh_values", input.dtype())?;
        self.with_linalg_pool_fresh(move |context, buffers| {
            #[cfg(feature = "native")]
            if faer_strided_read_ok(&input) {
                return eigh_values_faer_view_entered(context, buffers, input.tensor_view());
            }
            context.with_materialized_tensor_read(
                buffers,
                "eigh_values",
                input,
                |input, buffers| eigh_values_entered(context, buffers, input),
            )
        })
    }

    fn cholesky_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        let domain = self.shared_allocation_domain();
        self.with_linalg_pool_fresh(move |context, buffers| {
            if let Some(domain) = domain.as_deref()
                && input.backend_family().is_some()
            {
                return managed_cholesky(context, buffers, input, domain);
            }
            ensure_host_tensor_read("cholesky", &input)?;
            ensure_supported_linalg_dtype("cholesky", input.dtype())?;
            #[cfg(feature = "native")]
            if faer_strided_read_ok(&input) {
                return cholesky_faer_view_entered(context, buffers, input.tensor_view());
            }
            context.with_materialized_tensor_read(buffers, "cholesky", input, |input, buffers| {
                cholesky_entered(context, buffers, input)
            })
        })
    }

    fn lu_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor_read("lu", &input)?;
        ensure_supported_linalg_dtype("lu", input.dtype())?;
        self.with_linalg_pool_fresh(move |context, buffers| {
            #[cfg(feature = "native")]
            if faer_strided_read_ok(&input) {
                return lu_faer_view_entered(context, buffers, input.tensor_view());
            }
            context.with_materialized_tensor_read(buffers, "lu", input, |input, buffers| {
                lu_entered(context, buffers, input)
            })
        })
    }

    fn full_piv_lu_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor_read("full_piv_lu", &input)?;
        ensure_supported_linalg_dtype("full_piv_lu", input.dtype())?;
        self.with_linalg_pool_fresh(move |context, buffers| {
            #[cfg(feature = "native")]
            if faer_strided_read_ok(&input) {
                return full_piv_lu_faer_view_entered(context, buffers, input.tensor_view());
            }
            context.with_materialized_tensor_read(
                buffers,
                "full_piv_lu",
                input,
                |input, buffers| full_piv_lu_entered(context, buffers, input),
            )
        })
    }

    fn eig_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor_read("eig", &input)?;
        ensure_supported_linalg_dtype("eig", input.dtype())?;
        self.with_linalg_pool_fresh(move |context, buffers| {
            #[cfg(feature = "native")]
            if faer_strided_read_ok(&input) {
                return linalg::faer::eig_view(context, buffers, input.tensor_view());
            }
            context.with_materialized_tensor_read(buffers, "eig", input, |input, buffers| {
                eig_entered(context, buffers, input)
            })
        })
    }

    fn eig_values_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        ensure_host_tensor_read("eig_values", &input)?;
        ensure_supported_linalg_dtype("eig_values", input.dtype())?;
        self.with_linalg_pool_fresh(move |context, buffers| {
            #[cfg(feature = "native")]
            if faer_strided_read_ok(&input) {
                return linalg::faer::eig_values_view(context, buffers, input.tensor_view());
            }
            context.with_materialized_tensor_read(buffers, "eig_values", input, |input, buffers| {
                eig_values_entered(context, buffers, input)
            })
        })
    }

    fn eigh_values(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        ensure_host_tensor("eigh_values", input)?;
        self.with_linalg_pool_fresh(|context, buffers| eigh_values_entered(context, buffers, input))
    }

    fn eig(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        ensure_host_tensor("eig", input)?;
        ensure_supported_linalg_dtype("eig", input.dtype())?;
        self.with_linalg_pool_fresh(|context, buffers| eig_entered(context, buffers, input))
    }

    fn eig_values(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        ensure_host_tensor("eig_values", input)?;
        ensure_supported_linalg_dtype("eig_values", input.dtype())?;
        self.with_linalg_pool_fresh(|context, buffers| eig_values_entered(context, buffers, input))
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
        if !matches!(pivots.dtype(), DType::I32) {
            return Err(Error::dtype_mismatch(OP, DType::I32, pivots.dtype()));
        }
        if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
            return self.with_linalg_pool_fresh(|_, _| zeros_like_tensor(b));
        }

        validate_lu_solve_prepared_shapes(
            packed_lu.shape(),
            pivots.shape(),
            &packed_lu::rhs_matrix_shape(a.shape(), b.shape()),
        )?;
        validate_nonsingular_u(packed_lu)?;
        // One pooled RHS copy becomes the output; the provider kernel applies
        // the stored pivots and both triangular solves per matrix in place.
        self.with_linalg_pool_fresh(|ctx, buffers| {
            packed_lu::lu_solve_prepared_entered(
                ctx,
                buffers,
                a,
                packed_lu,
                pivots,
                b,
                transpose_a,
                conjugate_a,
            )
        })
    }

    fn lu_factor_solve(&mut self, a: &Tensor, b: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        const OP: &str = "lu_factor_solve";

        ensure_host_tensor(OP, a)?;
        ensure_host_tensor(OP, b)?;
        ensure_supported_linalg_pair(OP, a, b)?;
        self.with_linalg_pool_fresh(|ctx, buffers| {
            packed_lu::lu_factor_solve_entered(ctx, buffers, a, b)
        })
    }

    fn solve(&mut self, a: &Tensor, b: &Tensor) -> tenferro_tensor::Result<Tensor> {
        ensure_host_tensor("solve", a)?;
        ensure_host_tensor("solve", b)?;
        ensure_supported_linalg_pair("solve", a, b)?;
        self.with_linalg_pool_fresh(|context, buffers| solve_entered(context, buffers, a, b))
    }

    fn solve_read(
        &mut self,
        a: TensorRead<'_>,
        b: TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        ensure_host_tensor_read("solve", &a)?;
        ensure_host_tensor_read("solve", &b)?;
        ensure_supported_linalg_dtypes("solve", a.dtype(), b.dtype())?;
        let direct = !has_zero_dim(a.shape())
            && !has_zero_dim(b.shape())
            && solve_shape_direct_eligible(a.shape(), b.shape());
        self.with_linalg_pool_fresh(move |context, buffers| {
            if direct {
                solve_from_views_entered(context, buffers, a.tensor_view(), b.tensor_view())
            } else {
                context.with_materialized_tensor_read(buffers, "solve", a, |a, buffers| {
                    context.with_materialized_tensor_read(buffers, "solve", b, |b, buffers| {
                        solve_entered(context, buffers, a, b)
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

        if has_zero_dim(a.shape())
            || has_zero_dim(b.shape())
            || !solve_read_into_direct_eligible(&a, &b, &out)
        {
            return crate::backend::solve_read_into_default(self, a, b, out);
        }

        let a = a.tensor_view();
        let b = b.tensor_view();
        self.with_linalg_pool(move |context, buffers| {
            solve_read_into_entered(context, buffers, a, b, out)
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
    input.is_backend_buffer()
}

fn managed_cholesky(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
    domain: &dyn SharedTensorAllocationDomain,
) -> tenferro_tensor::Result<Tensor> {
    let dtype = input.dtype();
    ensure_supported_linalg_dtype("cholesky", dtype)?;
    match input.tensor_view() {
        TensorView::F32(view) => managed_cholesky_typed(context, buffers, &view, domain),
        TensorView::F64(view) => managed_cholesky_typed(context, buffers, &view, domain),
        TensorView::C32(view) => managed_cholesky_typed(context, buffers, &view, domain),
        TensorView::C64(view) => managed_cholesky_typed(context, buffers, &view, domain),
        _ => Err(unsupported_dtype("cholesky", dtype)),
    }
}

trait ManagedCholeskyScalar: Copy + Send + Sync + TensorScalar + 'static {
    const DTYPE: DType;

    fn factor(
        context: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        data: &[Self],
        n: usize,
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
            ) -> tenferro_tensor::Result<Vec<Self>> {
                {
                    #[cfg(feature = "native")]
                    {
                        #[cfg(feature = "native")]
                        {
                            linalg::faer::cholesky_compact_data(context, buffers, data, n)
                        }
                    }
                    #[cfg(feature = "blas")]
                    {
                        #[cfg(feature = "blas")]
                        {
                            let _ = context;
                            linalg::blas::cholesky_compact_data(buffers, data, n)
                        }
                    }
                }
            }

            fn take_output(output: Tensor) -> tenferro_tensor::Result<TypedTensor<Self>> {
                let Ok(output) = output.into_typed::<preset_scalar!($variant)>() else {
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
                Tensor::from_typed::<preset_scalar!($variant)>(output)
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
    input: &tenferro_tensor::TypedTensorView<'_, T>,
    domain: &dyn SharedTensorAllocationDomain,
) -> tenferro_tensor::Result<Tensor>
where
    T: ManagedCholeskyScalar,
{
    let n = validate_managed_cholesky_input(input, domain.id())?;
    let values = if n == 0 {
        Vec::new()
    } else {
        input.with_host_read(|read| T::factor(context, buffers, read, n))??
    };
    let mut typed = T::take_output(domain.allocate(T::DTYPE, &[n, n])?)?;
    write_managed_cholesky_output(&mut typed, domain.id(), &values)?;
    Ok(T::wrap(typed))
}

fn validate_managed_cholesky_input<T: Copy + Send + Sync + TensorScalar + 'static>(
    input: &tenferro_tensor::TypedTensorView<'_, T>,
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
    if !input.is_col_major_contiguous()? {
        return Err(tenferro_tensor::Error::invalid_argument(
            "cholesky",
            "input layout",
            "managed rank-2 Cholesky requires a compact column-major descriptor",
        ));
    }
    rows.checked_mul(cols).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(
            "cholesky",
            "input shape",
            "matrix element count overflows usize",
        )
    })?;
    if input.placement().memory_kind != MemoryKind::Managed {
        return Err(tenferro_tensor::Error::host_access(
            "cholesky",
            HostAccessError::Unsupported {
                backend: if matches!(
                    input.placement().memory_kind,
                    MemoryKind::PinnedHost | MemoryKind::UnpinnedHost
                ) {
                    "host"
                } else {
                    "backend"
                },
            },
        ));
    }
    match input.allocation_domain() {
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
                HostAccessError::Unsupported { backend: "backend" },
            ));
        }
    }
    Ok(rows)
}

fn write_managed_cholesky_output<T: TensorScalar + Copy + Send + Sync + 'static>(
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
    if let Some(buffer) = output.backend_buffer_mut() {
        let mut write = buffer
            .map_write()
            .map_err(|source| tenferro_tensor::Error::host_access("cholesky", source))?;
        return write
            .copy_from_slice(values)
            .map_err(|source| tenferro_tensor::Error::host_access("cholesky", source));
    }
    output.with_host_write(|write| {
        if write.len() != values.len() {
            return Err(tenferro_tensor::Error::runtime_state(
                "cholesky",
                "shared allocation owner returned an output with the wrong length",
            ));
        }
        write.copy_from_slice(values);
        Ok(())
    })?
}

/// The typed host tensor behind `input`, or this file's standard refusal.
///
/// Callers reach this from a match on `input.dtype()`, so `None` here means the tag
/// table and the runtime dtype disagree rather than a caller mistake.
fn typed_host<'a, T: TensorScalar>(
    input: &'a Tensor,
    op: &'static str,
) -> tenferro_tensor::Result<&'a TypedTensor<T>> {
    input
        .as_typed::<T>()
        .ok_or_else(|| unsupported_dtype(op, input.dtype()))
}

/// The typed operands behind a same-dtype pair, or this entry point's refusal.
fn q_columns_operands<'a, T: TensorScalar>(
    packed: &'a Tensor,
    coeff: &'a Tensor,
) -> Result<TypedOperandPair<'a, T>, tenferro_tensor::Error> {
    let p = packed.as_typed::<T>().ok_or_else(|| {
        Error::dtype_mismatch("householder_qr_q_columns", packed.dtype(), coeff.dtype())
    })?;
    let c = coeff.as_typed::<T>().ok_or_else(|| {
        Error::dtype_mismatch("householder_qr_q_columns", packed.dtype(), coeff.dtype())
    })?;
    Ok((p, c))
}

/// A pair of borrowed typed operands that share one scalar.
type TypedOperandPair<'a, T> = (
    &'a TypedTensor<T, tenferro_tensor::DynRank>,
    &'a TypedTensor<T, tenferro_tensor::DynRank>,
);

/// The typed operands behind a same-dtype pair, or the refusal this entry point reports.
///
/// The result type is spelled out because this module's imports resolve `TypedTensor` to its
/// two-parameter form and `Result` to the standard one at file scope.
fn qr_import_operands<'a, T: TensorScalar>(
    q: &'a Tensor,
    r: &'a Tensor,
) -> Result<TypedOperandPair<'a, T>, tenferro_tensor::Error> {
    let q_t = q
        .as_typed::<T>()
        .ok_or_else(|| unsupported_dtype("householder_qr_from_factors", q.dtype()))?;
    let r_t = r
        .as_typed::<T>()
        .ok_or_else(|| unsupported_dtype("householder_qr_from_factors", r.dtype()))?;
    Ok((q_t, r_t))
}

/// A triple of borrowed typed operands that share one scalar.
type TypedOperandTriple<'a, T> = (
    &'a TypedTensor<T, tenferro_tensor::DynRank>,
    &'a TypedTensor<T, tenferro_tensor::DynRank>,
    &'a TypedTensor<T, tenferro_tensor::DynRank>,
);

/// The typed operands behind a same-dtype triple, or the refusal this entry point reports.
fn append_operands<'a, T: TensorScalar>(
    packed: &'a Tensor,
    coeff: &'a Tensor,
    block: &'a Tensor,
) -> Result<TypedOperandTriple<'a, T>, tenferro_tensor::Error> {
    let mismatch = || unsupported_dtype("householder_qr_append", packed.dtype());
    let packed_t = packed.as_typed::<T>().ok_or_else(mismatch)?;
    let coeff_t = coeff.as_typed::<T>().ok_or_else(mismatch)?;
    let block_t = block.as_typed::<T>().ok_or_else(mismatch)?;
    Ok((packed_t, coeff_t, block_t))
}

fn ensure_host_tensor(op: &'static str, input: &Tensor) -> tenferro_tensor::Result<()> {
    match input.dtype() {
        // A caller-owned payload is not a linalg operand.
        DType::External(type_id) => Err(unsupported_dtype(op, DType::External(type_id))),
        DType::F32 => ensure_host_typed_tensor(op, typed_host::<f32>(input, op)?),
        DType::F64 => ensure_host_typed_tensor(op, typed_host::<f64>(input, op)?),
        DType::I32 => ensure_host_typed_tensor(op, typed_host::<i32>(input, op)?),
        DType::I64 => ensure_host_typed_tensor(op, typed_host::<i64>(input, op)?),
        DType::Bool => ensure_host_typed_tensor(op, typed_host::<bool>(input, op)?),
        DType::C32 => ensure_host_typed_tensor(op, typed_host::<Complex32>(input, op)?),
        DType::C64 => ensure_host_typed_tensor(op, typed_host::<Complex64>(input, op)?),
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

fn triangular_solve_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: &Tensor,
    b: &Tensor,
    options: TriangularSolveOptions,
) -> tenferro_tensor::Result<Tensor> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                same_variant_pair!(
                    a,
                    b,
                    |a, b| {
                        linalg::faer::triangular_solve(
                            context,
                            buffers,
                            a,
                            b,
                            options.left_side,
                            options.lower,
                            options.transpose_a,
                            options.unit_diagonal,
                        )
                    },
                    unsupported_pair("triangular_solve", a, b)
                )
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = context;
                same_variant_pair!(
                    a,
                    b,
                    |a, b| {
                        linalg::blas::triangular_solve(
                            buffers,
                            a,
                            b,
                            options.left_side,
                            options.lower,
                            options.transpose_a,
                            options.unit_diagonal,
                        )
                    },
                    unsupported_pair("triangular_solve", a, b)
                )
            }
        }
    }
}

fn solve_entered(
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

    let result = {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                same_variant_pair!(
                    a,
                    &rhs,
                    |a, b| { linalg::faer::solve(context, buffers, a, b, false) },
                    unsupported_pair("solve", a, &rhs)
                )
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = context;
                same_variant_pair!(
                    a,
                    &rhs,
                    |a, b| { linalg::blas::solve(buffers, a, b, false) },
                    unsupported_pair("solve", a, &rhs)
                )
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
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: TensorView<'_>,
    b: TensorView<'_>,
) -> tenferro_tensor::Result<Tensor> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                match (a, b) {
                    (TensorView::F32(a), TensorView::F32(b)) => {
                        linalg::faer::solve_from_views(context, buffers, a, b, false)
                            .map(Tensor::from_typed::<f32>)
                    }
                    (TensorView::F64(a), TensorView::F64(b)) => {
                        linalg::faer::solve_from_views(context, buffers, a, b, false)
                            .map(Tensor::from_typed::<f64>)
                    }
                    (TensorView::C32(a), TensorView::C32(b)) => {
                        linalg::faer::solve_from_views(context, buffers, a, b, false)
                            .map(Tensor::from_typed::<Complex32>)
                    }
                    (TensorView::C64(a), TensorView::C64(b)) => {
                        linalg::faer::solve_from_views(context, buffers, a, b, false)
                            .map(Tensor::from_typed::<Complex64>)
                    }
                    _ => Err(Error::invalid_argument(
                        "solve",
                        "inputs",
                        "solve inputs must have the same dtype",
                    )),
                }
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = context;
                match (a, b) {
                    (TensorView::F32(a), TensorView::F32(b)) => {
                        linalg::blas::solve_from_views(buffers, a, b, false)
                            .map(Tensor::from_typed::<f32>)
                    }
                    (TensorView::F64(a), TensorView::F64(b)) => {
                        linalg::blas::solve_from_views(buffers, a, b, false)
                            .map(Tensor::from_typed::<f64>)
                    }
                    (TensorView::C32(a), TensorView::C32(b)) => {
                        linalg::blas::solve_from_views(buffers, a, b, false)
                            .map(Tensor::from_typed::<Complex32>)
                    }
                    (TensorView::C64(a), TensorView::C64(b)) => {
                        linalg::blas::solve_from_views(buffers, a, b, false)
                            .map(Tensor::from_typed::<Complex64>)
                    }
                    _ => Err(Error::invalid_argument(
                        "solve",
                        "inputs",
                        "solve inputs must have the same dtype",
                    )),
                }
            }
        }
    }
}

fn solve_read_into_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: TensorView<'_>,
    b: TensorView<'_>,
    out: TensorWrite<'_>,
) -> tenferro_tensor::Result<()> {
    let out = tensor_write_view(out);
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
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
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
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
        }
    }
}

fn tensor_write_view(out: TensorWrite<'_>) -> TensorViewMut<'_> {
    match out {
        TensorWrite::Tensor(tensor) => match tensor.dtype() {
            DType::F32 => TensorViewMut::F32(write_view_operand::<f32>(tensor).as_view_mut()),
            DType::F64 => TensorViewMut::F64(write_view_operand::<f64>(tensor).as_view_mut()),
            DType::I32 => TensorViewMut::I32(write_view_operand::<i32>(tensor).as_view_mut()),
            DType::I64 => TensorViewMut::I64(write_view_operand::<i64>(tensor).as_view_mut()),
            DType::Bool => TensorViewMut::Bool(write_view_operand::<bool>(tensor).as_view_mut()),
            DType::C32 => TensorViewMut::C32(write_view_operand::<Complex32>(tensor).as_view_mut()),
            DType::C64 => TensorViewMut::C64(write_view_operand::<Complex64>(tensor).as_view_mut()),
            // INVARIANT: linalg rejects an externally defined dtype before adapting a
            // write target, and `TensorViewMut` has no externally defined variant.
            DType::External(_) => unreachable!("linalg validates its input dtypes first"),
        },
        TensorWrite::View(view) => view,
    }
}

/// The typed write target behind a tensor, or the unwind site this module documents.
///
/// INVARIANT: linalg rejects a dtype it cannot adapt before it borrows a write target, and
/// `TensorViewMut` has no externally defined variant, so this cannot be reached.
fn write_view_operand<T: TensorScalar>(tensor: &mut Tensor) -> &mut TypedTensor<T> {
    tensor
        .as_typed_mut::<T>()
        .unwrap_or_else(|| unreachable!("linalg validates its input dtypes first"))
}

fn cholesky_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Tensor> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                match input.dtype() {
                    DType::F32 => linalg::faer::cholesky(
                        context,
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("cholesky", input.dtype()))?,
                    )
                    .map(Tensor::from_typed::<f32>),
                    DType::F64 => linalg::faer::cholesky(
                        context,
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("cholesky", input.dtype()))?,
                    )
                    .map(Tensor::from_typed::<f64>),
                    DType::C32 => linalg::faer::cholesky(
                        context,
                        buffers,
                        input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("cholesky", input.dtype()))?,
                    )
                    .map(Tensor::from_typed::<Complex32>),
                    DType::C64 => linalg::faer::cholesky(
                        context,
                        buffers,
                        input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("cholesky", input.dtype()))?,
                    )
                    .map(Tensor::from_typed::<Complex64>),
                    _ => Err(unsupported_dtype("cholesky", input.dtype())),
                }
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = context;
                match input.dtype() {
                    DType::F32 => linalg::blas::cholesky(
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("cholesky", input.dtype()))?,
                    )
                    .map(Tensor::from_typed::<f32>),
                    DType::F64 => linalg::blas::cholesky(
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("cholesky", input.dtype()))?,
                    )
                    .map(Tensor::from_typed::<f64>),
                    DType::C32 => linalg::blas::cholesky(
                        buffers,
                        input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("cholesky", input.dtype()))?,
                    )
                    .map(Tensor::from_typed::<Complex32>),
                    DType::C64 => linalg::blas::cholesky(
                        buffers,
                        input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("cholesky", input.dtype()))?,
                    )
                    .map(Tensor::from_typed::<Complex64>),
                    _ => Err(unsupported_dtype("cholesky", input.dtype())),
                }
            }
        }
    }
}

fn lu_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                match input.dtype() {
                    DType::F32 => linalg::faer::lu(
                        context,
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("lu", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
                    DType::F64 => linalg::faer::lu(
                        context,
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("lu", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
                    DType::C32 => linalg::faer::lu(
                        context,
                        buffers,
                        input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("lu", input.dtype()))?,
                    )
                    .map(|outputs| {
                        outputs
                            .into_iter()
                            .map(Tensor::from_typed::<Complex32>)
                            .collect()
                    }),
                    DType::C64 => linalg::faer::lu(
                        context,
                        buffers,
                        input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("lu", input.dtype()))?,
                    )
                    .map(|outputs| {
                        outputs
                            .into_iter()
                            .map(Tensor::from_typed::<Complex64>)
                            .collect()
                    }),
                    _ => Err(unsupported_dtype("lu", input.dtype())),
                }
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = context;
                match input.dtype() {
                    DType::F32 => linalg::blas::lu(
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("lu", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
                    DType::F64 => linalg::blas::lu(
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("lu", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
                    DType::C32 => linalg::blas::lu(
                        buffers,
                        input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("lu", input.dtype()))?,
                    )
                    .map(|outputs| {
                        outputs
                            .into_iter()
                            .map(Tensor::from_typed::<Complex32>)
                            .collect()
                    }),
                    DType::C64 => linalg::blas::lu(
                        buffers,
                        input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("lu", input.dtype()))?,
                    )
                    .map(|outputs| {
                        outputs
                            .into_iter()
                            .map(Tensor::from_typed::<Complex64>)
                            .collect()
                    }),
                    _ => Err(unsupported_dtype("lu", input.dtype())),
                }
            }
        }
    }
}

fn full_piv_lu_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                match input.dtype() {
                    DType::F32 => linalg::faer::full_piv_lu(
                        context,
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("full_piv_lu", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
                    DType::F64 => linalg::faer::full_piv_lu(
                        context,
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("full_piv_lu", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
                    DType::C32 => linalg::faer::full_piv_lu(
                        context,
                        buffers,
                        input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("full_piv_lu", input.dtype()))?,
                    )
                    .and_then(full_piv_lu_c32_outputs_to_public_tensors),
                    DType::C64 => linalg::faer::full_piv_lu(
                        context,
                        buffers,
                        input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("full_piv_lu", input.dtype()))?,
                    )
                    .and_then(full_piv_lu_c64_outputs_to_public_tensors),
                    _ => Err(unsupported_dtype("full_piv_lu", input.dtype())),
                }
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = context;
                match input.dtype() {
                    DType::F32 => linalg::blas::full_piv_lu(
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("full_piv_lu", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
                    DType::F64 => linalg::blas::full_piv_lu(
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("full_piv_lu", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
                    DType::C32 => linalg::blas::full_piv_lu(
                        buffers,
                        input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("full_piv_lu", input.dtype()))?,
                    )
                    .and_then(full_piv_lu_c32_outputs_to_public_tensors),
                    DType::C64 => linalg::blas::full_piv_lu(
                        buffers,
                        input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("full_piv_lu", input.dtype()))?,
                    )
                    .and_then(full_piv_lu_c64_outputs_to_public_tensors),
                    _ => Err(unsupported_dtype("full_piv_lu", input.dtype())),
                }
            }
        }
    }
}

fn svd_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                match input.dtype() {
                    DType::F32 => linalg::faer::svd(
                        context,
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("svd", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
                    DType::F64 => linalg::faer::svd(
                        context,
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("svd", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
                    DType::C32 => linalg::faer::svd(
                        context,
                        buffers,
                        input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("svd", input.dtype()))?,
                    )
                    .and_then(svd_c32_outputs_to_public_tensors),
                    DType::C64 => linalg::faer::svd(
                        context,
                        buffers,
                        input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("svd", input.dtype()))?,
                    )
                    .and_then(svd_c64_outputs_to_public_tensors),
                    _ => Err(unsupported_dtype("svd", input.dtype())),
                }
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = context;
                match input.dtype() {
                    DType::F32 => linalg::blas::svd(
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("svd", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
                    DType::F64 => linalg::blas::svd(
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("svd", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
                    DType::C32 => {
                        let t = input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("svd", input.dtype()))?;
                        linalg::blas::svd(buffers, t).and_then(svd_c32_outputs_to_public_tensors)
                    }
                    DType::C64 => {
                        let t = input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("svd", input.dtype()))?;
                        linalg::blas::svd(buffers, t).and_then(svd_c64_outputs_to_public_tensors)
                    }
                    _ => Err(unsupported_dtype("svd", input.dtype())),
                }
            }
        }
    }
}

fn svd_full_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                match input.dtype() {
                    DType::F32 => linalg::faer::svd_full(
                        context,
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("svd_full", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
                    DType::F64 => linalg::faer::svd_full(
                        context,
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("svd_full", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
                    DType::C32 => linalg::faer::svd_full(
                        context,
                        buffers,
                        input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("svd_full", input.dtype()))?,
                    )
                    .and_then(svd_c32_outputs_to_public_tensors),
                    DType::C64 => linalg::faer::svd_full(
                        context,
                        buffers,
                        input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("svd_full", input.dtype()))?,
                    )
                    .and_then(svd_c64_outputs_to_public_tensors),
                    _ => Err(unsupported_dtype("svd_full", input.dtype())),
                }
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = context;
                match input.dtype() {
                    DType::F32 => linalg::blas::svd_full(
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("svd_full", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
                    DType::F64 => linalg::blas::svd_full(
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("svd_full", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
                    DType::C32 => {
                        let t = input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("svd_full", input.dtype()))?;
                        linalg::blas::svd_full(buffers, t)
                            .and_then(svd_c32_outputs_to_public_tensors)
                    }
                    DType::C64 => {
                        let t = input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("svd_full", input.dtype()))?;
                        linalg::blas::svd_full(buffers, t)
                            .and_then(svd_c64_outputs_to_public_tensors)
                    }
                    _ => Err(unsupported_dtype("svd_full", input.dtype())),
                }
            }
        }
    }
}

fn svd_values_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Tensor> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                match input.dtype() {
                    DType::F32 => {
                        let t = input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("svd_values", input.dtype()))?;
                        linalg::faer::svd_values(context, buffers, t).map(Tensor::from_typed::<f32>)
                    }
                    DType::F64 => {
                        let t = input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("svd_values", input.dtype()))?;
                        linalg::faer::svd_values(context, buffers, t).map(Tensor::from_typed::<f64>)
                    }
                    DType::C32 => {
                        let t = input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("svd_values", input.dtype()))?;
                        linalg::faer::svd_values(context, buffers, t).map(Tensor::from_typed::<f32>)
                    }
                    DType::C64 => {
                        let t = input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("svd_values", input.dtype()))?;
                        linalg::faer::svd_values(context, buffers, t).map(Tensor::from_typed::<f64>)
                    }
                    _ => Err(unsupported_dtype("svd_values", input.dtype())),
                }
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = context;
                match input.dtype() {
                    DType::F32 => linalg::blas::svd_values(
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("svd_values", input.dtype()))?,
                    )
                    .map(Tensor::from_typed::<f32>),
                    DType::F64 => linalg::blas::svd_values(
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("svd_values", input.dtype()))?,
                    )
                    .map(Tensor::from_typed::<f64>),
                    DType::C32 => linalg::blas::svd_values(
                        buffers,
                        input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("svd_values", input.dtype()))?,
                    )
                    .map(Tensor::from_typed::<f32>),
                    DType::C64 => linalg::blas::svd_values(
                        buffers,
                        input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("svd_values", input.dtype()))?,
                    )
                    .map(Tensor::from_typed::<f64>),
                    _ => Err(unsupported_dtype("svd_values", input.dtype())),
                }
            }
        }
    }
}

fn qr_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                match input.dtype() {
                    DType::F32 => linalg::faer::qr(
                        context,
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("qr", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
                    DType::F64 => linalg::faer::qr(
                        context,
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("qr", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
                    DType::C32 => linalg::faer::qr(
                        context,
                        buffers,
                        input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("qr", input.dtype()))?,
                    )
                    .map(|outputs| {
                        outputs
                            .into_iter()
                            .map(Tensor::from_typed::<Complex32>)
                            .collect()
                    }),
                    DType::C64 => linalg::faer::qr(
                        context,
                        buffers,
                        input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("qr", input.dtype()))?,
                    )
                    .map(|outputs| {
                        outputs
                            .into_iter()
                            .map(Tensor::from_typed::<Complex64>)
                            .collect()
                    }),
                    _ => Err(unsupported_dtype("qr", input.dtype())),
                }
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = context;
                match input.dtype() {
                    DType::F32 => linalg::blas::qr(
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("qr", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
                    DType::F64 => linalg::blas::qr(
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("qr", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
                    DType::C32 => linalg::blas::qr(
                        buffers,
                        input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("qr", input.dtype()))?,
                    )
                    .map(|outputs| {
                        outputs
                            .into_iter()
                            .map(Tensor::from_typed::<Complex32>)
                            .collect()
                    }),
                    DType::C64 => linalg::blas::qr(
                        buffers,
                        input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("qr", input.dtype()))?,
                    )
                    .map(|outputs| {
                        outputs
                            .into_iter()
                            .map(Tensor::from_typed::<Complex64>)
                            .collect()
                    }),
                    _ => Err(unsupported_dtype("qr", input.dtype())),
                }
            }
        }
    }
}

fn rank_revealing_qr_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
    options: crate::RankRevealingQrOptions,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    macro_rules! map_result {
        ($result:expr, $variant:ident) => {{
            $result.map(|result| {
                vec![
                    Tensor::from_typed::<preset_scalar!($variant)>(result.q),
                    Tensor::from_typed::<preset_scalar!($variant)>(result.r),
                    Tensor::from_typed::<i64>(result.column_permutation),
                    Tensor::from_typed::<i64>(result.rank),
                ]
            })
        }};
    }
    let mut outputs = {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                match input.dtype() {
                    DType::F32 => map_result!(
                        linalg::faer::rank_revealing_qr(
                            context,
                            buffers,
                            input.as_typed::<f32>().ok_or_else(|| unsupported_dtype(
                                "rank_revealing_qr",
                                input.dtype()
                            ))?,
                            options
                        ),
                        F32
                    ),
                    DType::F64 => map_result!(
                        linalg::faer::rank_revealing_qr(
                            context,
                            buffers,
                            input.as_typed::<f64>().ok_or_else(|| unsupported_dtype(
                                "rank_revealing_qr",
                                input.dtype()
                            ))?,
                            options
                        ),
                        F64
                    ),
                    DType::C32 => {
                        map_result!(
                            linalg::faer::rank_revealing_qr(
                                context,
                                buffers,
                                input.as_typed::<Complex32>().ok_or_else(|| {
                                    unsupported_dtype("rank_revealing_qr", input.dtype())
                                })?,
                                options
                            ),
                            C32
                        )
                    }
                    DType::C64 => {
                        map_result!(
                            linalg::faer::rank_revealing_qr(
                                context,
                                buffers,
                                input.as_typed::<Complex64>().ok_or_else(|| {
                                    unsupported_dtype("rank_revealing_qr", input.dtype())
                                })?,
                                options
                            ),
                            C64
                        )
                    }
                    _ => Err(unsupported_dtype("rank_revealing_qr", input.dtype())),
                }
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = context;
                match input.dtype() {
                    DType::F32 => {
                        let t = input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("rank_revealing_qr", input.dtype()))?;
                        map_result!(linalg::blas::rank_revealing_qr(buffers, t, options), F32)
                    }
                    DType::F64 => {
                        let t = input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("rank_revealing_qr", input.dtype()))?;
                        map_result!(linalg::blas::rank_revealing_qr(buffers, t, options), F64)
                    }
                    DType::C32 => {
                        let t = input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("rank_revealing_qr", input.dtype()))?;
                        map_result!(linalg::blas::rank_revealing_qr(buffers, t, options), C32)
                    }
                    DType::C64 => {
                        let t = input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("rank_revealing_qr", input.dtype()))?;
                        map_result!(linalg::blas::rank_revealing_qr(buffers, t, options), C64)
                    }
                    _ => Err(unsupported_dtype("rank_revealing_qr", input.dtype())),
                }
            }
        }
    }?;
    apply_qr_gauge(options.gauge, &mut outputs[..2])?;
    Ok(outputs)
}

fn householder_qr_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<CompactQrResult> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                macro_rules! factor {
                    ($tensor:expr, $variant:ident) => {{
                        let (packed, coeff) =
                            linalg::faer::compact_factor_2d(context, buffers, $tensor)?;
                        Ok(CompactQrResult {
                            packed: Tensor::from_typed::<preset_scalar!($variant)>(packed),
                            coeff: Tensor::from_typed::<preset_scalar!($variant)>(coeff),
                        })
                    }};
                }
                match input.dtype() {
                    DType::F32 => factor!(
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("householder_qr", input.dtype()))?,
                        F32
                    ),
                    DType::F64 => factor!(
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("householder_qr", input.dtype()))?,
                        F64
                    ),
                    DType::C32 => factor!(
                        input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("householder_qr", input.dtype()))?,
                        C32
                    ),
                    DType::C64 => factor!(
                        input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("householder_qr", input.dtype()))?,
                        C64
                    ),
                    _ => Err(unsupported_dtype("householder_qr", input.dtype())),
                }
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = context;
                macro_rules! factor {
                    ($tensor:expr, $variant:ident) => {{
                        let (packed, coeff) = linalg::blas::householder_qr(buffers, $tensor)?;
                        Ok(CompactQrResult {
                            packed: Tensor::from_typed::<preset_scalar!($variant)>(packed),
                            coeff: Tensor::from_typed::<preset_scalar!($variant)>(coeff),
                        })
                    }};
                }
                match input.dtype() {
                    DType::F32 => factor!(
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("householder_qr", input.dtype()))?,
                        F32
                    ),
                    DType::F64 => factor!(
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("householder_qr", input.dtype()))?,
                        F64
                    ),
                    DType::C32 => factor!(
                        input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("householder_qr", input.dtype()))?,
                        C32
                    ),
                    DType::C64 => factor!(
                        input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("householder_qr", input.dtype()))?,
                        C64
                    ),
                    _ => Err(unsupported_dtype("householder_qr", input.dtype())),
                }
            }
        }
    }
}

fn householder_qr_from_factors_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    q: &Tensor,
    r: &Tensor,
) -> tenferro_tensor::Result<CompactQrResult> {
    if q.dtype() != r.dtype() {
        return Err(Error::dtype_mismatch(
            "householder_qr_from_factors",
            q.dtype(),
            r.dtype(),
        ));
    }
    #[cfg(feature = "native")]
    {
        #[cfg(feature = "native")]
        {
            macro_rules! import {
                ($q:expr, $r:expr, $variant:ident) => {{
                    let (packed, coeff) = linalg::faer::from_factors_2d(context, buffers, $q, $r)?;
                    return Ok(CompactQrResult {
                        packed: Tensor::from_typed::<preset_scalar!($variant)>(packed),
                        coeff: Tensor::from_typed::<preset_scalar!($variant)>(coeff),
                    });
                }};
            }
            match (q.dtype(), r.dtype()) {
                (DType::F32, DType::F32) => {
                    let (q_t, r_t) = qr_import_operands::<f32>(q, r)?;
                    import!(q_t, r_t, F32)
                }
                (DType::F64, DType::F64) => {
                    let (q_t, r_t) = qr_import_operands::<f64>(q, r)?;
                    import!(q_t, r_t, F64)
                }
                (DType::C32, DType::C32) => {
                    let (q_t, r_t) = qr_import_operands::<Complex32>(q, r)?;
                    import!(q_t, r_t, C32)
                }
                (DType::C64, DType::C64) => {
                    let (q_t, r_t) = qr_import_operands::<Complex64>(q, r)?;
                    import!(q_t, r_t, C64)
                }
                _ => Err(unsupported_dtype("householder_qr_from_factors", q.dtype())),
            }
        }
    }
    #[cfg(feature = "blas")]
    {
        let _ = context;
        macro_rules! import {
            ($q:expr, $r:expr, $variant:ident) => {{
                let (packed, coeff) = linalg::blas::householder_qr_from_factors(buffers, $q, $r)?;
                Ok(CompactQrResult {
                    packed: Tensor::from_typed::<preset_scalar!($variant)>(packed),
                    coeff: Tensor::from_typed::<preset_scalar!($variant)>(coeff),
                })
            }};
        }
        match (q.dtype(), r.dtype()) {
            (DType::F32, DType::F32) => {
                let (q, r) = qr_import_operands::<f32>(q, r)?;
                import!(q, r, F32)
            }
            (DType::F64, DType::F64) => {
                let (q, r) = qr_import_operands::<f64>(q, r)?;
                import!(q, r, F64)
            }
            (DType::C32, DType::C32) => {
                let (q, r) = qr_import_operands::<Complex32>(q, r)?;
                import!(q, r, C32)
            }
            (DType::C64, DType::C64) => {
                let (q, r) = qr_import_operands::<Complex64>(q, r)?;
                import!(q, r, C64)
            }
            _ => Err(unsupported_dtype("householder_qr_from_factors", q.dtype())),
        }
    }
}

fn householder_qr_append_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    packed: &Tensor,
    coeff: &Tensor,
    block: &Tensor,
) -> tenferro_tensor::Result<CompactQrResult> {
    if packed.dtype() != coeff.dtype() || packed.dtype() != block.dtype() {
        return Err(Error::dtype_mismatch(
            "householder_qr_append",
            packed.dtype(),
            block.dtype(),
        ));
    }
    #[cfg(feature = "native")]
    {
        #[cfg(feature = "native")]
        {
            macro_rules! append {
                ($packed:expr, $coeff:expr, $block:expr, $variant:ident) => {{
                    let (packed, coeff) =
                        linalg::faer::append_2d(context, buffers, $packed, $coeff, $block)?;
                    return Ok(CompactQrResult {
                        packed: Tensor::from_typed::<preset_scalar!($variant)>(packed),
                        coeff: Tensor::from_typed::<preset_scalar!($variant)>(coeff),
                    });
                }};
            }
            match (packed.dtype(), coeff.dtype(), block.dtype()) {
                (DType::F32, DType::F32, DType::F32) => {
                    let (p, c, b) = append_operands::<f32>(packed, coeff, block)?;

                    append!(p, c, b, F32)
                }
                (DType::F64, DType::F64, DType::F64) => {
                    let (p, c, b) = append_operands::<f64>(packed, coeff, block)?;

                    append!(p, c, b, F64)
                }
                (DType::C32, DType::C32, DType::C32) => {
                    let (p, c, b) = append_operands::<Complex32>(packed, coeff, block)?;

                    append!(p, c, b, C32)
                }
                (DType::C64, DType::C64, DType::C64) => {
                    let (p, c, b) = append_operands::<Complex64>(packed, coeff, block)?;

                    append!(p, c, b, C64)
                }
                _ => Err(unsupported_dtype("householder_qr_append", packed.dtype())),
            }
        }
    }
    #[cfg(feature = "blas")]
    {
        let _ = context;
        macro_rules! append {
            ($packed:expr, $coeff:expr, $block:expr, $variant:ident) => {{
                let (packed, coeff) =
                    linalg::blas::householder_qr_append(buffers, $packed, $coeff, $block)?;
                Ok(CompactQrResult {
                    packed: Tensor::from_typed::<preset_scalar!($variant)>(packed),
                    coeff: Tensor::from_typed::<preset_scalar!($variant)>(coeff),
                })
            }};
        }
        match (packed.dtype(), coeff.dtype(), block.dtype()) {
            (DType::F32, DType::F32, DType::F32) => {
                let (p, c, b) = append_operands::<f32>(packed, coeff, block)?;

                append!(p, c, b, F32)
            }
            (DType::F64, DType::F64, DType::F64) => {
                let (p, c, b) = append_operands::<f64>(packed, coeff, block)?;

                append!(p, c, b, F64)
            }
            (DType::C32, DType::C32, DType::C32) => {
                let (p, c, b) = append_operands::<Complex32>(packed, coeff, block)?;

                append!(p, c, b, C32)
            }
            (DType::C64, DType::C64, DType::C64) => {
                let (p, c, b) = append_operands::<Complex64>(packed, coeff, block)?;

                append!(p, c, b, C64)
            }
            _ => Err(unsupported_dtype("householder_qr_append", packed.dtype())),
        }
    }
}

fn householder_qr_r_entered(
    _context: &CpuExecutionContext<'_>,
    _buffers: &mut BufferPool,
    packed: &Tensor,
    coeff: &Tensor,
    options: crate::QrOptions,
) -> tenferro_tensor::Result<Tensor> {
    let positive = options.gauge == crate::QrGauge::PositiveDiagonal;
    #[cfg(feature = "native")]
    {
        #[cfg(feature = "native")]
        return same_variant_pair!(
            packed,
            coeff,
            |p, c| linalg::faer::raw_r_2d(p, c, positive),
            Err(Error::dtype_mismatch(
                "householder_qr_r",
                packed.dtype(),
                coeff.dtype(),
            ))
        );
    }
    #[cfg(feature = "blas")]
    {
        same_variant_pair!(
            packed,
            coeff,
            |p, c| linalg::blas::householder_qr_r(p, c, positive),
            Err(Error::dtype_mismatch(
                "householder_qr_r",
                packed.dtype(),
                coeff.dtype(),
            ))
        )
    }
}

fn householder_qr_q_columns_entered(
    _context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    packed: &Tensor,
    coeff: &Tensor,
    columns: std::ops::Range<usize>,
    options: crate::QrOptions,
) -> tenferro_tensor::Result<Tensor> {
    let positive = options.gauge == crate::QrGauge::PositiveDiagonal;
    #[cfg(feature = "native")]
    {
        #[cfg(feature = "native")]
        return match (packed.dtype(), coeff.dtype()) {
            (DType::F32, DType::F32) => {
                let (p, c) = q_columns_operands::<f32>(packed, coeff)?;
                linalg::faer::q_columns_2d(
                    _context,
                    buffers,
                    p,
                    c,
                    columns.start,
                    columns.end,
                    positive,
                )
                .map(Tensor::from_typed::<f32>)
            }
            (DType::F64, DType::F64) => {
                let (p, c) = q_columns_operands::<f64>(packed, coeff)?;
                linalg::faer::q_columns_2d(
                    _context,
                    buffers,
                    p,
                    c,
                    columns.start,
                    columns.end,
                    positive,
                )
                .map(Tensor::from_typed::<f64>)
            }
            (DType::C32, DType::C32) => {
                let (p, c) = q_columns_operands::<Complex32>(packed, coeff)?;
                linalg::faer::q_columns_2d(
                    _context,
                    buffers,
                    p,
                    c,
                    columns.start,
                    columns.end,
                    positive,
                )
                .map(Tensor::from_typed::<Complex32>)
            }
            (DType::C64, DType::C64) => {
                let (p, c) = q_columns_operands::<Complex64>(packed, coeff)?;
                linalg::faer::q_columns_2d(
                    _context,
                    buffers,
                    p,
                    c,
                    columns.start,
                    columns.end,
                    positive,
                )
                .map(Tensor::from_typed::<Complex64>)
            }
            _ => Err(Error::dtype_mismatch(
                "householder_qr_q_columns",
                packed.dtype(),
                coeff.dtype(),
            )),
        };
    }
    #[cfg(feature = "blas")]
    {
        macro_rules! columns {
            ($packed:expr, $coeff:expr, $variant:ident) => {
                linalg::blas::householder_qr_q_columns(
                    buffers,
                    $packed,
                    $coeff,
                    columns.start,
                    columns.end,
                    positive,
                )
                .map(Tensor::from_typed::<preset_scalar!($variant)>)
            };
        }
        match (packed.dtype(), coeff.dtype()) {
            (DType::F32, DType::F32) => {
                let (p, c) = q_columns_operands::<f32>(packed, coeff)?;
                columns!(p, c, F32)
            }
            (DType::F64, DType::F64) => {
                let (p, c) = q_columns_operands::<f64>(packed, coeff)?;
                columns!(p, c, F64)
            }
            (DType::C32, DType::C32) => {
                let (p, c) = q_columns_operands::<Complex32>(packed, coeff)?;
                columns!(p, c, C32)
            }
            (DType::C64, DType::C64) => {
                let (p, c) = q_columns_operands::<Complex64>(packed, coeff)?;
                columns!(p, c, C64)
            }
            _ => Err(Error::dtype_mismatch(
                "householder_qr_q_columns",
                packed.dtype(),
                coeff.dtype(),
            )),
        }
    }
}

fn eigh_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                match input.dtype() {
                    DType::F32 => linalg::faer::eigh(
                        context,
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("eigh", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
                    DType::F64 => linalg::faer::eigh(
                        context,
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("eigh", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
                    DType::C32 => linalg::faer::eigh(
                        context,
                        buffers,
                        input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("eigh", input.dtype()))?,
                    )
                    .and_then(eigh_c32_outputs_to_public_tensors),
                    DType::C64 => linalg::faer::eigh(
                        context,
                        buffers,
                        input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("eigh", input.dtype()))?,
                    )
                    .and_then(eigh_c64_outputs_to_public_tensors),
                    _ => Err(unsupported_dtype("eigh", input.dtype())),
                }
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = context;
                match input.dtype() {
                    DType::F32 => linalg::blas::eigh(
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("eigh", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
                    DType::F64 => linalg::blas::eigh(
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("eigh", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
                    DType::C32 => {
                        let t = input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("eigh", input.dtype()))?;
                        linalg::blas::eigh(buffers, t).and_then(eigh_c32_outputs_to_public_tensors)
                    }
                    DType::C64 => {
                        let t = input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("eigh", input.dtype()))?;
                        linalg::blas::eigh(buffers, t).and_then(eigh_c64_outputs_to_public_tensors)
                    }
                    _ => Err(unsupported_dtype("eigh", input.dtype())),
                }
            }
        }
    }
}

fn eigh_values_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Tensor> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                match input.dtype() {
                    DType::F32 => {
                        let t = input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("eigh_values", input.dtype()))?;
                        linalg::faer::eigh_values(context, buffers, t)
                            .map(Tensor::from_typed::<f32>)
                    }
                    DType::F64 => {
                        let t = input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("eigh_values", input.dtype()))?;
                        linalg::faer::eigh_values(context, buffers, t)
                            .map(Tensor::from_typed::<f64>)
                    }
                    DType::C32 => {
                        let t = input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("eigh_values", input.dtype()))?;
                        linalg::faer::eigh_values(context, buffers, t)
                            .map(Tensor::from_typed::<f32>)
                    }
                    DType::C64 => {
                        let t = input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("eigh_values", input.dtype()))?;
                        linalg::faer::eigh_values(context, buffers, t)
                            .map(Tensor::from_typed::<f64>)
                    }
                    _ => Err(unsupported_dtype("eigh_values", input.dtype())),
                }
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = context;
                match input.dtype() {
                    DType::F32 => linalg::blas::eigh_values(
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("eigh_values", input.dtype()))?,
                    )
                    .map(Tensor::from_typed::<f32>),
                    DType::F64 => linalg::blas::eigh_values(
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("eigh_values", input.dtype()))?,
                    )
                    .map(Tensor::from_typed::<f64>),
                    DType::C32 => linalg::blas::eigh_values(
                        buffers,
                        input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("eigh_values", input.dtype()))?,
                    )
                    .map(Tensor::from_typed::<f32>),
                    DType::C64 => linalg::blas::eigh_values(
                        buffers,
                        input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("eigh_values", input.dtype()))?,
                    )
                    .map(Tensor::from_typed::<f64>),
                    _ => Err(unsupported_dtype("eigh_values", input.dtype())),
                }
            }
        }
    }
}

fn eig_values_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Tensor> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                linalg::faer::eig_values(context, buffers, input)
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = context;
                linalg::blas::eig_values(buffers, input)
            }
        }
    }
}

fn eig_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                linalg::faer::eig(context, buffers, input)
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = context;
                linalg::blas::eig(buffers, input)
            }
        }
    }
}

#[cfg(feature = "native")]
fn faer_strided_read_ok(input: &TensorRead<'_>) -> bool {
    match input {
        TensorRead::Tensor(tensor) => match tensor.dtype() {
            DType::F32 => tensor
                .as_typed::<f32>()
                .is_some_and(|t| linalg::faer::faer_strided_ok(&t.as_view())),
            DType::F64 => tensor
                .as_typed::<f64>()
                .is_some_and(|t| linalg::faer::faer_strided_ok(&t.as_view())),
            DType::C32 => tensor
                .as_typed::<Complex32>()
                .is_some_and(|t| linalg::faer::faer_strided_ok(&t.as_view())),
            DType::C64 => tensor
                .as_typed::<Complex64>()
                .is_some_and(|t| linalg::faer::faer_strided_ok(&t.as_view())),
            // A caller-owned payload is not a faer operand, and neither are the
            // integer and boolean tags, which faer does not compute on here.
            _ => false,
        },
        TensorRead::View(TensorView::F32(view)) => linalg::faer::faer_strided_ok(view),
        TensorRead::View(TensorView::F64(view)) => linalg::faer::faer_strided_ok(view),
        TensorRead::View(TensorView::C32(view)) => linalg::faer::faer_strided_ok(view),
        TensorRead::View(TensorView::C64(view)) => linalg::faer::faer_strided_ok(view),
        TensorRead::View(TensorView::I32(_))
        | TensorRead::View(TensorView::I64(_))
        | TensorRead::View(TensorView::Bool(_)) => false,
    }
}

/// Can this right-hand side be gathered directly by the faer view path?
///
/// The RHS is copied element by element, so strides may be arbitrary; only host
/// placement, the matrix rank every provider requires, and a supported dtype
/// matter.
#[cfg(feature = "native")]
fn faer_rhs_read_ok(input: &TensorRead<'_>) -> bool {
    if input.backend_family().is_some() {
        return false;
    }
    if input.shape().len() != 2 {
        return false;
    }
    matches!(
        input.dtype(),
        DType::F32 | DType::F64 | DType::C32 | DType::C64
    )
}

#[cfg(feature = "native")]
fn rank_revealing_qr_faer_view_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: TensorView<'_>,
    options: crate::RankRevealingQrOptions,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    macro_rules! factor {
        ($view:expr, $scalar:ty) => {
            linalg::faer::rank_revealing_qr_view(context, buffers, $view, options).map(|result| {
                vec![
                    Tensor::from_typed::<$scalar>(result.q),
                    Tensor::from_typed::<$scalar>(result.r),
                    Tensor::from_typed::<i64>(result.column_permutation),
                    Tensor::from_typed::<i64>(result.rank),
                ]
            })
        };
    }
    let mut outputs = match input {
        TensorView::F32(view) => factor!(view, f32),
        TensorView::F64(view) => factor!(view, f64),
        TensorView::C32(view) => factor!(view, Complex32),
        TensorView::C64(view) => factor!(view, Complex64),
        unsupported => Err(unsupported_dtype("rank_revealing_qr", unsupported.dtype())),
    }?;
    apply_qr_gauge(options.gauge, &mut outputs[..2])?;
    Ok(outputs)
}

#[cfg(feature = "native")]
fn triangular_solve_faer_view_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: TensorView<'_>,
    b: TensorView<'_>,
    options: TriangularSolveOptions,
) -> tenferro_tensor::Result<Tensor> {
    let TriangularSolveOptions {
        left_side,
        lower,
        transpose_a,
        unit_diagonal,
    } = options;
    macro_rules! solve {
        ($a:expr, $b:expr, $scalar:ty) => {
            linalg::faer::triangular_solve_view(
                context,
                buffers,
                $a,
                $b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::from_typed::<$scalar>)
        };
    }
    match (a, b) {
        (TensorView::F32(a), TensorView::F32(b)) => solve!(a, b, f32),
        (TensorView::F64(a), TensorView::F64(b)) => solve!(a, b, f64),
        (TensorView::C32(a), TensorView::C32(b)) => solve!(a, b, Complex32),
        (TensorView::C64(a), TensorView::C64(b)) => solve!(a, b, Complex64),
        (a, b) => Err(Error::dtype_mismatch(
            "triangular_solve",
            a.dtype(),
            b.dtype(),
        )),
    }
}

#[cfg(feature = "native")]
fn svd_faer_view_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: TensorView<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match input {
        TensorView::F32(view) => linalg::faer::svd_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
        TensorView::F64(view) => linalg::faer::svd_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
        TensorView::C32(view) => linalg::faer::svd_view(context, buffers, view)
            .and_then(svd_c32_outputs_to_public_tensors),
        TensorView::C64(view) => linalg::faer::svd_view(context, buffers, view)
            .and_then(svd_c64_outputs_to_public_tensors),
        unsupported => Err(unsupported_dtype("svd", unsupported.dtype())),
    }
}

#[cfg(feature = "native")]
fn svd_full_faer_view_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: TensorView<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match input {
        TensorView::F32(view) => linalg::faer::svd_full_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
        TensorView::F64(view) => linalg::faer::svd_full_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
        TensorView::C32(view) => linalg::faer::svd_full_view(context, buffers, view)
            .and_then(svd_c32_outputs_to_public_tensors),
        TensorView::C64(view) => linalg::faer::svd_full_view(context, buffers, view)
            .and_then(svd_c64_outputs_to_public_tensors),
        unsupported => Err(unsupported_dtype("svd_full", unsupported.dtype())),
    }
}

#[cfg(feature = "native")]
fn svd_values_faer_view_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: TensorView<'_>,
) -> tenferro_tensor::Result<Tensor> {
    match input {
        TensorView::F32(view) => {
            linalg::faer::svd_values_view(context, buffers, view).map(Tensor::from_typed::<f32>)
        }
        TensorView::F64(view) => {
            linalg::faer::svd_values_view(context, buffers, view).map(Tensor::from_typed::<f64>)
        }
        TensorView::C32(view) => {
            linalg::faer::svd_values_view(context, buffers, view).map(Tensor::from_typed::<f32>)
        }
        TensorView::C64(view) => {
            linalg::faer::svd_values_view(context, buffers, view).map(Tensor::from_typed::<f64>)
        }
        unsupported => Err(unsupported_dtype("svd_values", unsupported.dtype())),
    }
}

#[cfg(feature = "native")]
fn qr_faer_view_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: TensorView<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match input {
        TensorView::F32(view) => linalg::faer::qr_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
        TensorView::F64(view) => linalg::faer::qr_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
        TensorView::C32(view) => linalg::faer::qr_view(context, buffers, view).map(|outputs| {
            outputs
                .into_iter()
                .map(Tensor::from_typed::<Complex32>)
                .collect()
        }),
        TensorView::C64(view) => linalg::faer::qr_view(context, buffers, view).map(|outputs| {
            outputs
                .into_iter()
                .map(Tensor::from_typed::<Complex64>)
                .collect()
        }),
        unsupported => Err(unsupported_dtype("qr", unsupported.dtype())),
    }
}

#[cfg(feature = "native")]
fn eigh_faer_view_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: TensorView<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match input {
        TensorView::F32(view) => linalg::faer::eigh_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
        TensorView::F64(view) => linalg::faer::eigh_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
        TensorView::C32(view) => linalg::faer::eigh_view(context, buffers, view)
            .and_then(eigh_c32_outputs_to_public_tensors),
        TensorView::C64(view) => linalg::faer::eigh_view(context, buffers, view)
            .and_then(eigh_c64_outputs_to_public_tensors),
        unsupported => Err(unsupported_dtype("eigh", unsupported.dtype())),
    }
}

#[cfg(feature = "native")]
fn eigh_values_faer_view_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: TensorView<'_>,
) -> tenferro_tensor::Result<Tensor> {
    match input {
        TensorView::F32(view) => {
            linalg::faer::eigh_values_view(context, buffers, view).map(Tensor::from_typed::<f32>)
        }
        TensorView::F64(view) => {
            linalg::faer::eigh_values_view(context, buffers, view).map(Tensor::from_typed::<f64>)
        }
        TensorView::C32(view) => {
            linalg::faer::eigh_values_view(context, buffers, view).map(Tensor::from_typed::<f32>)
        }
        TensorView::C64(view) => {
            linalg::faer::eigh_values_view(context, buffers, view).map(Tensor::from_typed::<f64>)
        }
        unsupported => Err(unsupported_dtype("eigh_values", unsupported.dtype())),
    }
}

#[cfg(feature = "native")]
fn cholesky_faer_view_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: TensorView<'_>,
) -> tenferro_tensor::Result<Tensor> {
    match input {
        TensorView::F32(view) => {
            linalg::faer::cholesky_view(context, buffers, view).map(Tensor::from_typed::<f32>)
        }
        TensorView::F64(view) => {
            linalg::faer::cholesky_view(context, buffers, view).map(Tensor::from_typed::<f64>)
        }
        TensorView::C32(view) => {
            linalg::faer::cholesky_view(context, buffers, view).map(Tensor::from_typed::<Complex32>)
        }
        TensorView::C64(view) => {
            linalg::faer::cholesky_view(context, buffers, view).map(Tensor::from_typed::<Complex64>)
        }
        unsupported => Err(unsupported_dtype("cholesky", unsupported.dtype())),
    }
}

#[cfg(feature = "native")]
fn lu_faer_view_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: TensorView<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match input {
        TensorView::F32(view) => linalg::faer::lu_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
        TensorView::F64(view) => linalg::faer::lu_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
        TensorView::C32(view) => linalg::faer::lu_view(context, buffers, view).map(|outputs| {
            outputs
                .into_iter()
                .map(Tensor::from_typed::<Complex32>)
                .collect()
        }),
        TensorView::C64(view) => linalg::faer::lu_view(context, buffers, view).map(|outputs| {
            outputs
                .into_iter()
                .map(Tensor::from_typed::<Complex64>)
                .collect()
        }),
        unsupported => Err(unsupported_dtype("lu", unsupported.dtype())),
    }
}

#[cfg(feature = "native")]
fn full_piv_lu_faer_view_entered(
    context: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: TensorView<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match input {
        TensorView::F32(view) => linalg::faer::full_piv_lu_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
        TensorView::F64(view) => linalg::faer::full_piv_lu_view(context, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
        TensorView::C32(view) => {
            linalg::faer::full_piv_lu_view(context, buffers, view).map(|outputs| {
                outputs
                    .into_iter()
                    .map(Tensor::from_typed::<Complex32>)
                    .collect()
            })
        }
        TensorView::C64(view) => {
            linalg::faer::full_piv_lu_view(context, buffers, view).map(|outputs| {
                outputs
                    .into_iter()
                    .map(Tensor::from_typed::<Complex64>)
                    .collect()
            })
        }
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
        DType::I32 | DType::I64 | DType::Bool | DType::External(_) => {
            Err(unsupported_dtype(op, dtype))
        }
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

fn batched_vector_rhs_shape(a: &Tensor, b: &Tensor) -> Option<Vec<usize>> {
    batched_vector_rhs_shape_of(a.shape(), b.shape())
}

/// The matrix RHS shape `[n, 1, batch...]` a vector RHS `b` is solved as.
fn batched_vector_rhs_shape_of(a_shape: &[usize], b_shape: &[usize]) -> Option<Vec<usize>> {
    if b_shape.len() == 1 {
        return Some(vec![b_shape[0], 1]);
    }

    let is_batched_vector_rhs = a_shape.len() == b_shape.len() + 1
        && !b_shape.is_empty()
        && b_shape[0] == a_shape[0]
        && b_shape[1..] == a_shape[2..];
    if !is_batched_vector_rhs {
        return None;
    }

    let mut rhs_shape = vec![b_shape[0], 1];
    rhs_shape.extend_from_slice(&b_shape[1..]);
    Some(rhs_shape)
}

fn zeros_like_tensor(input: &Tensor) -> tenferro_tensor::Result<Tensor> {
    Ok(match input.dtype() {
        // A caller-owned payload has no zero-like runtime tensor.
        DType::External(type_id) => {
            return Err(unsupported_dtype(
                "zeros_like_tensor",
                DType::External(type_id),
            ));
        }
        DType::F32 => Tensor::from_typed::<f32>(TypedTensor::zeros(input.shape().to_vec())?),
        DType::F64 => Tensor::from_typed::<f64>(TypedTensor::zeros(input.shape().to_vec())?),
        DType::I32 => Tensor::from_typed::<i32>(TypedTensor::zeros(input.shape().to_vec())?),
        DType::I64 => Tensor::from_typed::<i64>(TypedTensor::zeros(input.shape().to_vec())?),
        DType::Bool => {
            let t = typed_host::<bool>(input, "zeros_like_tensor")?;
            Tensor::from_typed::<bool>(TypedTensor::from_vec_col_major(
                t.shape().to_vec(),
                vec![false; t.n_elements()],
            )?)
        }
        DType::C32 => Tensor::from_typed::<Complex32>(TypedTensor::zeros(input.shape().to_vec())?),
        DType::C64 => Tensor::from_typed::<Complex64>(TypedTensor::zeros(input.shape().to_vec())?),
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
            Tensor::from_typed::<Complex32>(p),
            Tensor::from_typed::<Complex32>(l),
            Tensor::from_typed::<Complex32>(u),
            Tensor::from_typed::<Complex32>(q),
            Tensor::from_typed::<f32>(complex32_real_part_tensor(parity)?),
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
            Tensor::from_typed::<Complex64>(p),
            Tensor::from_typed::<Complex64>(l),
            Tensor::from_typed::<Complex64>(u),
            Tensor::from_typed::<Complex64>(q),
            Tensor::from_typed::<f64>(complex64_real_part_tensor(parity)?),
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
            Tensor::from_typed::<Complex32>(u),
            Tensor::from_typed::<f32>(complex32_real_part_tensor(values)?),
            Tensor::from_typed::<Complex32>(vt),
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
            Tensor::from_typed::<Complex64>(u),
            Tensor::from_typed::<f64>(complex64_real_part_tensor(values)?),
            Tensor::from_typed::<Complex64>(vt),
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
            Tensor::from_typed::<f32>(complex32_real_part_tensor(values)?),
            Tensor::from_typed::<Complex32>(vectors),
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
            Tensor::from_typed::<f64>(complex64_real_part_tensor(values)?),
            Tensor::from_typed::<Complex64>(vectors),
        ]),
        _ => Err(eigh_output_count_error(count)),
    }
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

mod packed_lu;

#[cfg(test)]
mod tests;
