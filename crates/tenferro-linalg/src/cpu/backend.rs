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
use crate::validation::validate_square_matrix;

use super::linalg;
use tenferro_cpu::same_variant_pair;

use super::linalg::HostView;
use num_complex::{Complex32, Complex64};
use tcontract_exec::Exec;
use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_cpu::CpuEngine;
use tenferro_tensor::{
    validate::validate_nonsingular_u, AllocationDomainId, DType, Error, Host, HostAccessError,
    MemoryKind, SharedTensorAllocationDomain, Tensor, TensorRead, TensorScalar, TensorStructural,
    TensorView, TensorViewMut, TensorWrite, TypedTensor,
};

/// A caller operand after the one host narrowing at the `LinalgBackend` boundary.
///
/// Every `*_entered` function below takes this instead of `&Tensor`: the type is the evidence
/// that the operand borrows host storage, so no function under the boundary checks placement
/// again.
type HostOperand<'a> = TensorView<'a, Host>;

/// Narrow a caller operand to host storage, attributing a refusal to `op`.
fn host_operand<'a>(
    op: &'static str,
    input: &'a Tensor,
) -> tenferro_tensor::Result<HostOperand<'a>> {
    TensorRead::from_tensor(input).try_host(op)
}

/// Narrow a value the CPU engine itself produced (a materialized or reshaped copy).
///
/// The engine writes those into host storage, so this narrowing cannot refuse in practice; it is
/// a separate function so a caller operand is never re-narrowed under the boundary by accident.
fn host_engine_output<'a>(
    op: &'static str,
    produced: &'a Tensor,
) -> tenferro_tensor::Result<HostOperand<'a>> {
    TensorRead::from_tensor(produced).try_host(op)
}

/// One typed arm of a [`HostOperand`].
trait HostArm: Sized {
    fn arm<'v, 'a>(view: &'v HostOperand<'a>) -> Option<&'v HostView<'a, Self>>;
}

macro_rules! impl_host_arm {
    ($($scalar:ty => $variant:ident),* $(,)?) => {$(
        impl HostArm for $scalar {
            fn arm<'v, 'a>(view: &'v HostOperand<'a>) -> Option<&'v HostView<'a, Self>> {
                match view {
                    TensorView::$variant(typed) => Some(typed),
                    _ => None,
                }
            }
        }
    )*};
}

impl_host_arm!(
    f32 => F32,
    f64 => F64,
    i32 => I32,
    i64 => I64,
    bool => Bool,
    Complex32 => C32,
    Complex64 => C64,
);

/// `as_typed` on a host operand, so the dtype tables read the same as they did on `&Tensor`.
trait HostOperandExt<'a> {
    fn as_typed<T: HostArm>(&self) -> Option<&HostView<'a, T>>;
}

impl<'a> HostOperandExt<'a> for HostOperand<'a> {
    fn as_typed<T: HostArm>(&self) -> Option<&HostView<'a, T>> {
        T::arm(self)
    }
}

/// Run `operation` on `input` laid out like an owned tensor, materializing it first if needed.
///
/// An operand that already covers its buffer in compact column-major order is passed through;
/// any other layout is gathered once by the CPU engine, as the borrowed-read routes always did.
fn with_owned_layout<R>(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    op: &'static str,
    input: HostOperand<'_>,
    operation: impl FnOnce(
        &HostOperand<'_>,
        &mut BufferPool,
        &mut Exec<'_>,
    ) -> tenferro_tensor::Result<R>,
) -> tenferro_tensor::Result<R> {
    if input.is_owned_layout() {
        return operation(&input, buffers, exec);
    }
    tenferro_cpu::with_materialized_tensor_read(
        exec,
        buffers,
        op,
        TensorRead::from_view(input.into_dynamic()),
        |materialized, buffers, exec| {
            operation(&host_engine_output(op, materialized)?, buffers, exec)
        },
    )
}

/// A metadata-only reshape of a compact host operand, keeping it host-marked.
///
/// Used for the vector right-hand side a solve treats as an `[n, 1, batch...]` matrix: the
/// operand already covers its buffer compactly, so no element moves.
fn reshape_host<'a>(
    input: &HostOperand<'a>,
    shape: &[usize],
) -> tenferro_tensor::Result<HostOperand<'a>> {
    Ok(match input {
        TensorView::F32(view) => TensorView::F32(view.reshape_view(shape)?),
        TensorView::F64(view) => TensorView::F64(view.reshape_view(shape)?),
        TensorView::I32(view) => TensorView::I32(view.reshape_view(shape)?),
        TensorView::I64(view) => TensorView::I64(view.reshape_view(shape)?),
        TensorView::Bool(view) => TensorView::Bool(view.reshape_view(shape)?),
        TensorView::C32(view) => TensorView::C32(view.reshape_view(shape)?),
        TensorView::C64(view) => TensorView::C64(view.reshape_view(shape)?),
    })
}

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
trait CpuBackendLinalgAffinityExt {
    /// Every BLAS linalg entry passes here, including the direct caller-output
    /// path that does not allocate a fresh result.
    fn with_linalg_pool_checked<R>(
        &mut self,
        op: impl FnOnce(&mut Exec<'_>, &mut BufferPool) -> tenferro_tensor::Result<R>,
    ) -> tenferro_tensor::Result<R>;
}

impl CpuBackendLinalgAffinityExt for CpuEngine {
    fn with_linalg_pool_checked<R>(
        &mut self,
        op: impl FnOnce(&mut Exec<'_>, &mut BufferPool) -> tenferro_tensor::Result<R>,
    ) -> tenferro_tensor::Result<R> {
        self.with_linalg_pool(move |exec, buffers| {
            #[cfg(feature = "blas")]
            super::tlinalg_blas::reject_forced_batch_strategy(exec)?;
            op(exec, buffers)
        })
    }
}

impl LinalgBackend for CpuEngine {
    fn cholesky(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        let domain = self.shared_allocation_domain().cloned();
        self.with_linalg_pool_checked(move |exec, buffers| {
            if tensor_uses_backend_storage(input)
                && let Some(domain) = domain.as_deref()
            {
                return managed_cholesky(exec, buffers, TensorRead::from_tensor(input), domain);
            }
            let input = &host_operand("cholesky", input)?;
            cholesky_entered(exec, buffers, input)
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
        let a = &host_operand("triangular_solve", a)?;
        let b = &host_operand("triangular_solve", b)?;
        let options = TriangularSolveOptions {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        };
        self.with_linalg_pool_checked(|exec, buffers| {
            triangular_solve_entered(exec, buffers, a, b, options)
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
        let a = a.try_host("triangular_solve")?;
        let b = b.try_host("triangular_solve")?;
        ensure_supported_linalg_dtypes("triangular_solve", a.dtype(), b.dtype())?;
        let options = TriangularSolveOptions {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        };
        self.with_linalg_pool_checked(move |exec, buffers| {
            // Both operands must be faer-eligible for the direct path: `a`
            // reaches faer as a strided `MatRef` and `b` is gathered straight
            // into the destructible right-hand side. If either is ineligible,
            // the scoped materializer below packs only what it must.
            #[cfg(feature = "native")]
            if faer_strided_host_ok(&a) && faer_rhs_host_ok(&b) {
                return triangular_solve_faer_view_entered(exec, buffers, a, b, options);
            }
            with_owned_layout(exec, buffers, "triangular_solve", a, |a, buffers, exec| {
                with_owned_layout(exec, buffers, "triangular_solve", b, |b, buffers, exec| {
                    triangular_solve_entered(exec, buffers, a, b, options)
                })
            })
        })
    }

    fn lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = &host_operand("lu", input)?;
        self.with_linalg_pool_checked(|exec, buffers| lu_entered(exec, buffers, input))
    }

    fn lu_factor(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = &host_operand("lu_factor", input)?;
        {
            #[cfg(feature = "native")]
            {
                #[cfg(feature = "native")]
                {
                    self.with_linalg_pool_checked(|exec, buffers| match input.dtype() {
                        DType::F32 => {
                            let t = input
                                .as_typed::<f32>()
                                .ok_or_else(|| unsupported_dtype("lu_factor", input.dtype()))?;
                            linalg::faer::lu_factor(exec, buffers, t).map(|(lu, pivots, parity)| {
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
                            linalg::faer::lu_factor(exec, buffers, t).map(|(lu, pivots, parity)| {
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
                            linalg::faer::lu_factor(exec, buffers, t).map(|(lu, pivots, parity)| {
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
                            linalg::faer::lu_factor(exec, buffers, t).map(|(lu, pivots, parity)| {
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
                    self.with_linalg_pool_checked(|exec, buffers| match input.dtype() {
                        DType::F32 => {
                            let t = input
                                .as_typed::<f32>()
                                .ok_or_else(|| unsupported_dtype("lu_factor", input.dtype()))?;
                            linalg::blas::lu_factor(exec, buffers, t).map(|(lu, pivots, parity)| {
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
                            linalg::blas::lu_factor(exec, buffers, t).map(|(lu, pivots, parity)| {
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
                            linalg::blas::lu_factor(exec, buffers, t).map(|(lu, pivots, parity)| {
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
                            linalg::blas::lu_factor(exec, buffers, t).map(|(lu, pivots, parity)| {
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
        let input = &host_operand("full_piv_lu", input)?;
        self.with_linalg_pool_checked(|exec, buffers| full_piv_lu_entered(exec, buffers, input))
    }

    fn full_piv_lu_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        transpose_a: bool,
    ) -> tenferro_tensor::Result<Tensor> {
        let a = &host_operand("full_piv_lu_solve", a)?;
        let b = &host_operand("full_piv_lu_solve", b)?;
        ensure_supported_linalg_pair("full_piv_lu_solve", a, b)?;
        validate_square_matrix("full_piv_lu_solve", a.shape())?;
        if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
            return self.with_linalg_pool_checked(|_, _| zeros_like_tensor(b.dtype(), b.shape()));
        }

        // A vector right-hand side is solved as a one-column matrix: a metadata reshape of the
        // compact host operand, restored on the result below.
        let (rhs, restore_shape) = match batched_vector_rhs_shape(a, b) {
            Some(matrix_rhs_shape) => (
                reshape_host(b, &matrix_rhs_shape)?,
                Some(b.shape().to_vec()),
            ),
            None => (b.clone(), None),
        };

        // The same complete-pivoting kernel as `full_piv_lu` in both CPU builds.
        let result = self.with_linalg_pool_checked(|exec, buffers| {
            same_variant_pair!(
                a,
                &rhs,
                |a, b| { linalg::faer::full_piv_lu_solve(exec, buffers, a, b, transpose_a) },
                unsupported_pair("full_piv_lu_solve", a, &rhs)
            )
        })?;

        if let Some(shape) = restore_shape {
            self.reshape_read(TensorRead::from_tensor(&result), &shape)
        } else {
            Ok(result)
        }
    }

    fn svd(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = &host_operand("svd", input)?;
        self.with_linalg_pool_checked(|exec, buffers| svd_entered(exec, buffers, input))
    }

    fn svd_full(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = &host_operand("svd_full", input)?;
        self.with_linalg_pool_checked(|exec, buffers| svd_full_entered(exec, buffers, input))
    }

    fn svd_full_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.try_host("svd_full")?;
        ensure_supported_linalg_dtype("svd_full", input.dtype())?;
        self.with_linalg_pool_checked(move |exec, buffers| {
            #[cfg(feature = "native")]
            if faer_strided_host_ok(&input) {
                return svd_full_faer_view_entered(exec, buffers, input);
            }
            with_owned_layout(exec, buffers, "svd_full", input, |input, buffers, exec| {
                svd_full_entered(exec, buffers, input)
            })
        })
    }

    fn svd_values(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        let input = &host_operand("svd_values", input)?;
        self.with_linalg_pool_checked(|exec, buffers| svd_values_entered(exec, buffers, input))
    }

    fn svd_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.try_host("svd")?;
        ensure_supported_linalg_dtype("svd", input.dtype())?;
        self.with_linalg_pool_checked(move |exec, buffers| {
            #[cfg(feature = "native")]
            if faer_strided_host_ok(&input) {
                return svd_faer_view_entered(exec, buffers, input);
            }
            with_owned_layout(exec, buffers, "svd", input, |input, buffers, exec| {
                svd_entered(exec, buffers, input)
            })
        })
    }

    fn svd_read_with_policy(
        &mut self,
        input: TensorRead<'_>,
        policy: tenferro_tensor::MaterializationPolicy,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        if policy == tenferro_tensor::MaterializationPolicy::Forbid {
            // Same predicate `svd_read` branches on: the native build reads a
            // rank-2 non-negative-stride view in place; everything else is
            // gathered by `with_owned_layout` unless already owned-layout.
            // Checked before the pool is entered, so a refusal has no side effect.
            let host = input.clone().try_host("svd")?;
            ensure_supported_linalg_dtype("svd", host.dtype())?;
            #[cfg(feature = "native")]
            let direct = faer_strided_host_ok(&host) || host.is_owned_layout();
            #[cfg(not(feature = "native"))]
            let direct = host.is_owned_layout();
            if !direct {
                return Err(Error::input_materialization_forbidden("svd", 0));
            }
        }
        self.svd_read(input)
    }

    fn svd_values_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        let input = input.try_host("svd_values")?;
        ensure_supported_linalg_dtype("svd_values", input.dtype())?;
        self.with_linalg_pool_checked(move |exec, buffers| {
            #[cfg(feature = "native")]
            if faer_strided_host_ok(&input) {
                return svd_values_faer_view_entered(exec, buffers, input);
            }
            with_owned_layout(
                exec,
                buffers,
                "svd_values",
                input,
                |input, buffers, exec| svd_values_entered(exec, buffers, input),
            )
        })
    }

    fn qr(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = &host_operand("qr", input)?;
        self.with_linalg_pool_checked(|exec, buffers| qr_entered(exec, buffers, input))
    }

    fn qr_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.try_host("qr")?;
        ensure_supported_linalg_dtype("qr", input.dtype())?;
        self.with_linalg_pool_checked(move |exec, buffers| {
            #[cfg(feature = "native")]
            if faer_strided_host_ok(&input) {
                return qr_faer_view_entered(exec, buffers, input);
            }
            with_owned_layout(exec, buffers, "qr", input, |input, buffers, exec| {
                qr_entered(exec, buffers, input)
            })
        })
    }

    fn rank_revealing_qr(
        &mut self,
        input: &Tensor,
        options: crate::RankRevealingQrOptions,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        validate_rank_revealing_qr_options("rank_revealing_qr", options)?;
        let input = &host_operand("rank_revealing_qr", input)?;
        ensure_supported_linalg_dtype("rank_revealing_qr", input.dtype())?;
        self.with_linalg_pool_checked(|exec, buffers| {
            rank_revealing_qr_entered(exec, buffers, input, options)
        })
    }

    fn rank_revealing_qr_read(
        &mut self,
        input: TensorRead<'_>,
        options: crate::RankRevealingQrOptions,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        validate_rank_revealing_qr_options("rank_revealing_qr", options)?;
        let input = input.try_host("rank_revealing_qr")?;
        ensure_supported_linalg_dtype("rank_revealing_qr", input.dtype())?;
        self.with_linalg_pool_checked(move |exec, buffers| {
            #[cfg(feature = "native")]
            if faer_strided_host_ok(&input) {
                return rank_revealing_qr_faer_view_entered(exec, buffers, input, options);
            }
            with_owned_layout(
                exec,
                buffers,
                "rank_revealing_qr",
                input,
                |input, buffers, exec| rank_revealing_qr_entered(exec, buffers, input, options),
            )
        })
    }

    fn householder_qr(&mut self, input: &Tensor) -> tenferro_tensor::Result<CompactQrResult> {
        let input = &host_operand("householder_qr", input)?;
        self.with_linalg_pool_checked(|exec, buffers| householder_qr_entered(exec, buffers, input))
    }

    fn householder_qr_from_factors(
        &mut self,
        q: &Tensor,
        r: &Tensor,
    ) -> tenferro_tensor::Result<CompactQrResult> {
        let q = &host_operand("householder_qr_from_factors", q)?;
        let r = &host_operand("householder_qr_from_factors", r)?;
        self.with_linalg_pool_checked(|exec, buffers| {
            householder_qr_from_factors_entered(exec, buffers, q, r)
        })
    }

    fn householder_qr_append(
        &mut self,
        packed: &Tensor,
        coeff: &Tensor,
        block: &Tensor,
    ) -> tenferro_tensor::Result<CompactQrResult> {
        let packed = &host_operand("householder_qr_append", packed)?;
        let coeff = &host_operand("householder_qr_append", coeff)?;
        let block = &host_operand("householder_qr_append", block)?;
        self.with_linalg_pool_checked(|exec, buffers| {
            householder_qr_append_entered(exec, buffers, packed, coeff, block)
        })
    }

    fn householder_qr_r(
        &mut self,
        packed: &Tensor,
        coeff: &Tensor,
        options: crate::QrOptions,
    ) -> tenferro_tensor::Result<Tensor> {
        let packed = &host_operand("householder_qr_r", packed)?;
        let coeff = &host_operand("householder_qr_r", coeff)?;
        self.with_linalg_pool_checked(|exec, buffers| {
            householder_qr_r_entered(exec, buffers, packed, coeff, options)
        })
    }

    fn householder_qr_q_columns(
        &mut self,
        packed: &Tensor,
        coeff: &Tensor,
        columns: std::ops::Range<usize>,
        options: crate::QrOptions,
    ) -> tenferro_tensor::Result<Tensor> {
        let packed = &host_operand("householder_qr_q_columns", packed)?;
        let coeff = &host_operand("householder_qr_q_columns", coeff)?;
        self.with_linalg_pool_checked(|exec, buffers| {
            householder_qr_q_columns_entered(exec, buffers, packed, coeff, columns, options)
        })
    }

    fn eigh(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = &host_operand("eigh", input)?;
        self.with_linalg_pool_checked(|exec, buffers| eigh_entered(exec, buffers, input))
    }

    fn eigh_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.try_host("eigh")?;
        ensure_supported_linalg_dtype("eigh", input.dtype())?;
        self.with_linalg_pool_checked(move |exec, buffers| {
            #[cfg(feature = "native")]
            if faer_strided_host_ok(&input) {
                return eigh_faer_view_entered(exec, buffers, input);
            }
            with_owned_layout(exec, buffers, "eigh", input, |input, buffers, exec| {
                eigh_entered(exec, buffers, input)
            })
        })
    }

    fn eigh_values_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        let input = input.try_host("eigh_values")?;
        ensure_supported_linalg_dtype("eigh_values", input.dtype())?;
        self.with_linalg_pool_checked(move |exec, buffers| {
            #[cfg(feature = "native")]
            if faer_strided_host_ok(&input) {
                return eigh_values_faer_view_entered(exec, buffers, input);
            }
            with_owned_layout(
                exec,
                buffers,
                "eigh_values",
                input,
                |input, buffers, exec| eigh_values_entered(exec, buffers, input),
            )
        })
    }

    fn cholesky_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        let domain = self.shared_allocation_domain().cloned();
        self.with_linalg_pool_checked(move |exec, buffers| {
            if let Some(domain) = domain.as_deref()
                && input.backend_family().is_some()
            {
                return managed_cholesky(exec, buffers, input, domain);
            }
            let input = input.try_host("cholesky")?;
            ensure_supported_linalg_dtype("cholesky", input.dtype())?;
            #[cfg(feature = "native")]
            if faer_strided_host_ok(&input) {
                return cholesky_faer_view_entered(exec, buffers, input);
            }
            with_owned_layout(exec, buffers, "cholesky", input, |input, buffers, exec| {
                cholesky_entered(exec, buffers, input)
            })
        })
    }

    fn lu_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.try_host("lu")?;
        ensure_supported_linalg_dtype("lu", input.dtype())?;
        self.with_linalg_pool_checked(move |exec, buffers| {
            #[cfg(feature = "native")]
            if faer_strided_host_ok(&input) {
                return lu_faer_view_entered(exec, buffers, input);
            }
            with_owned_layout(exec, buffers, "lu", input, |input, buffers, exec| {
                lu_entered(exec, buffers, input)
            })
        })
    }

    fn full_piv_lu_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.try_host("full_piv_lu")?;
        ensure_supported_linalg_dtype("full_piv_lu", input.dtype())?;
        self.with_linalg_pool_checked(move |exec, buffers| {
            if faer_strided_host_ok(&input) {
                return full_piv_lu_faer_view_entered(exec, buffers, input);
            }
            with_owned_layout(
                exec,
                buffers,
                "full_piv_lu",
                input,
                |input, buffers, exec| full_piv_lu_entered(exec, buffers, input),
            )
        })
    }

    fn eig_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = input.try_host("eig")?;
        ensure_supported_linalg_dtype("eig", input.dtype())?;
        self.with_linalg_pool_checked(move |exec, buffers| {
            #[cfg(feature = "native")]
            if faer_strided_host_ok(&input) {
                return linalg::faer::eig_view(exec, buffers, input.into_dynamic());
            }
            with_owned_layout(exec, buffers, "eig", input, |input, buffers, exec| {
                eig_entered(exec, buffers, input)
            })
        })
    }

    fn eig_values_read(&mut self, input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor> {
        let input = input.try_host("eig_values")?;
        ensure_supported_linalg_dtype("eig_values", input.dtype())?;
        self.with_linalg_pool_checked(move |exec, buffers| {
            #[cfg(feature = "native")]
            if faer_strided_host_ok(&input) {
                return linalg::faer::eig_values_view(exec, buffers, input.into_dynamic());
            }
            with_owned_layout(
                exec,
                buffers,
                "eig_values",
                input,
                |input, buffers, exec| eig_values_entered(exec, buffers, input),
            )
        })
    }

    fn eigh_values(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        let input = &host_operand("eigh_values", input)?;
        self.with_linalg_pool_checked(|exec, buffers| eigh_values_entered(exec, buffers, input))
    }

    fn eig(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = &host_operand("eig", input)?;
        ensure_supported_linalg_dtype("eig", input.dtype())?;
        self.with_linalg_pool_checked(|exec, buffers| eig_entered(exec, buffers, input))
    }

    fn eig_values(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor> {
        let input = &host_operand("eig_values", input)?;
        ensure_supported_linalg_dtype("eig_values", input.dtype())?;
        self.with_linalg_pool_checked(|exec, buffers| eig_values_entered(exec, buffers, input))
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

        let a = &host_operand(OP, a)?;
        let packed_lu_tensor = packed_lu;
        let packed_lu = &host_operand(OP, packed_lu)?;
        let pivots = &host_operand(OP, pivots)?;
        let b = &host_operand(OP, b)?;
        ensure_supported_linalg_pair(OP, a, b)?;
        ensure_supported_linalg_pair(OP, a, packed_lu)?;
        if !matches!(pivots.dtype(), DType::I32) {
            return Err(Error::dtype_mismatch(OP, DType::I32, pivots.dtype()));
        }
        if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
            return self.with_linalg_pool_checked(|_, _| zeros_like_tensor(b.dtype(), b.shape()));
        }

        validate_lu_solve_prepared_shapes(
            packed_lu.shape(),
            pivots.shape(),
            &packed_lu::rhs_matrix_shape(a.shape(), b.shape()),
        )?;
        validate_nonsingular_u(packed_lu_tensor)?;
        // One pooled RHS copy becomes the output; the provider kernel applies
        // the stored pivots and both triangular solves per matrix in place.
        self.with_linalg_pool_checked(|exec, buffers| {
            packed_lu::lu_solve_prepared_entered(
                exec,
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

        let a = &host_operand(OP, a)?;
        let b = &host_operand(OP, b)?;
        ensure_supported_linalg_pair(OP, a, b)?;
        self.with_linalg_pool_checked(|exec, buffers| {
            packed_lu::lu_factor_solve_entered(exec, buffers, a, b)
        })
    }

    fn solve(&mut self, a: &Tensor, b: &Tensor) -> tenferro_tensor::Result<Tensor> {
        let a = &host_operand("solve", a)?;
        let b = &host_operand("solve", b)?;
        ensure_supported_linalg_pair("solve", a, b)?;
        self.with_linalg_pool_checked(|exec, buffers| solve_entered(exec, buffers, a, b))
    }

    fn solve_read(
        &mut self,
        a: TensorRead<'_>,
        b: TensorRead<'_>,
    ) -> tenferro_tensor::Result<Tensor> {
        let a = a.try_host("solve")?;
        let b = b.try_host("solve")?;
        ensure_supported_linalg_dtypes("solve", a.dtype(), b.dtype())?;
        let direct = !has_zero_dim(a.shape())
            && !has_zero_dim(b.shape())
            && solve_shape_direct_eligible(a.shape(), b.shape());
        self.with_linalg_pool_checked(move |exec, buffers| {
            if direct {
                solve_from_views_entered(exec, buffers, a, b)
            } else {
                with_owned_layout(exec, buffers, "solve", a, |a, buffers, exec| {
                    with_owned_layout(exec, buffers, "solve", b, |b, buffers, exec| {
                        solve_entered(exec, buffers, a, b)
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
        let a = a.try_host("solve_read_into")?;
        let b = b.try_host("solve_read_into")?;
        let out = out.try_mut_host("solve_read_into")?;
        ensure_supported_linalg_dtypes("solve_read_into", a.dtype(), b.dtype())?;

        if has_zero_dim(a.shape())
            || has_zero_dim(b.shape())
            || !solve_read_into_direct_eligible(&a, &b, &out)
        {
            // The default composes the public `solve_read`, which is a boundary of its own.
            return crate::backend::solve_read_into_default(
                self,
                TensorRead::from_view(a.into_dynamic()),
                TensorRead::from_view(b.into_dynamic()),
                TensorWrite::from_view(out.into_dynamic()),
            );
        }

        self.with_linalg_pool_checked(move |exec, buffers| {
            solve_read_into_entered(exec, buffers, a, b, out)
        })
    }
}

fn solve_read_into_direct_eligible(
    a: &HostOperand<'_>,
    b: &HostOperand<'_>,
    out: &TensorViewMut<'_, Host>,
) -> bool {
    if !solve_shape_direct_eligible(a.shape(), b.shape()) || out.shape() != b.shape() {
        return false;
    }
    let strides = out.strides();
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
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
    domain: &dyn SharedTensorAllocationDomain,
) -> tenferro_tensor::Result<Tensor> {
    let dtype = input.dtype();
    ensure_supported_linalg_dtype("cholesky", dtype)?;
    match linalg_view("cholesky", input)? {
        TensorView::F32(view) => managed_cholesky_typed(exec, buffers, &view, domain),
        TensorView::F64(view) => managed_cholesky_typed(exec, buffers, &view, domain),
        TensorView::C32(view) => managed_cholesky_typed(exec, buffers, &view, domain),
        TensorView::C64(view) => managed_cholesky_typed(exec, buffers, &view, domain),
        _ => Err(unsupported_dtype("cholesky", dtype)),
    }
}

trait ManagedCholeskyScalar: Copy + Send + Sync + TensorScalar + 'static {
    const DTYPE: DType;

    /// The factor as a pooled tensor: the staging copy into the managed domain borrows it, and its
    /// storage returns to the session pool when the copy is done.
    fn factor(
        exec: &mut Exec<'_>,
        buffers: &mut BufferPool,
        data: &[Self],
        n: usize,
    ) -> tenferro_tensor::Result<TypedTensor<Self>>;

    fn take_output(output: Tensor) -> tenferro_tensor::Result<TypedTensor<Self>>;
    fn wrap(output: TypedTensor<Self>) -> Tensor;
}

macro_rules! impl_managed_cholesky_scalar {
    ($scalar:ty, $dtype:ident, $variant:ident) => {
        impl ManagedCholeskyScalar for $scalar {
            const DTYPE: DType = DType::$dtype;

            fn factor(
                exec: &mut Exec<'_>,
                buffers: &mut BufferPool,
                data: &[Self],
                n: usize,
            ) -> tenferro_tensor::Result<TypedTensor<Self>> {
                {
                    #[cfg(feature = "native")]
                    {
                        #[cfg(feature = "native")]
                        {
                            linalg::faer::cholesky_compact_data(exec, buffers, data, n)
                        }
                    }
                    #[cfg(feature = "blas")]
                    {
                        #[cfg(feature = "blas")]
                        {
                            let _ = &exec;
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
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: &tenferro_tensor::TypedTensorView<'_, T>,
    domain: &dyn SharedTensorAllocationDomain,
) -> tenferro_tensor::Result<Tensor>
where
    T: ManagedCholeskyScalar,
{
    let n = validate_managed_cholesky_input(input, domain.id())?;
    // The factor keeps its pool recycler: the copy below is its only use, and the staging storage
    // returns to the session pool immediately afterwards.
    let values = if n == 0 {
        None
    } else {
        Some(input.with_host_read(|read| T::factor(exec, buffers, read, n))??)
    };
    let mut typed = T::take_output(domain.allocate(T::DTYPE, &[n, n])?)?;
    let staged: &[T] = match &values {
        Some(values) => values.host_data()?,
        None => &[],
    };
    write_managed_cholesky_output(&mut typed, domain.id(), staged)?;
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

/// The typed operands behind a same-dtype pair, or this entry point's refusal.
fn q_columns_operands<'v, 'a, T: HostArm>(
    packed: &'v HostOperand<'a>,
    coeff: &'v HostOperand<'a>,
) -> Result<TypedOperandPair<'v, 'a, T>, tenferro_tensor::Error> {
    let p = packed.as_typed::<T>().ok_or_else(|| {
        Error::dtype_mismatch("householder_qr_q_columns", packed.dtype(), coeff.dtype())
    })?;
    let c = coeff.as_typed::<T>().ok_or_else(|| {
        Error::dtype_mismatch("householder_qr_q_columns", packed.dtype(), coeff.dtype())
    })?;
    Ok((p, c))
}

/// A pair of borrowed typed operands that share one scalar.
type TypedOperandPair<'v, 'a, T> = (&'v HostView<'a, T>, &'v HostView<'a, T>);

/// The typed operands behind a same-dtype pair, or the refusal this entry point reports.
///
/// The result type is spelled out because this module's imports resolve `TypedTensor` to its
/// two-parameter form and `Result` to the standard one at file scope.
fn qr_import_operands<'v, 'a, T: HostArm>(
    q: &'v HostOperand<'a>,
    r: &'v HostOperand<'a>,
) -> Result<TypedOperandPair<'v, 'a, T>, tenferro_tensor::Error> {
    let q_t = q
        .as_typed::<T>()
        .ok_or_else(|| unsupported_dtype("householder_qr_from_factors", q.dtype()))?;
    let r_t = r
        .as_typed::<T>()
        .ok_or_else(|| unsupported_dtype("householder_qr_from_factors", r.dtype()))?;
    Ok((q_t, r_t))
}

/// A triple of borrowed typed operands that share one scalar.
type TypedOperandTriple<'v, 'a, T> = (
    &'v HostView<'a, T>,
    &'v HostView<'a, T>,
    &'v HostView<'a, T>,
);

/// The typed operands behind a same-dtype triple, or the refusal this entry point reports.
fn append_operands<'v, 'a, T: HostArm>(
    packed: &'v HostOperand<'a>,
    coeff: &'v HostOperand<'a>,
    block: &'v HostOperand<'a>,
) -> Result<TypedOperandTriple<'v, 'a, T>, tenferro_tensor::Error> {
    let mismatch = || unsupported_dtype("householder_qr_append", packed.dtype());
    let packed_t = packed.as_typed::<T>().ok_or_else(mismatch)?;
    let coeff_t = coeff.as_typed::<T>().ok_or_else(mismatch)?;
    let block_t = block.as_typed::<T>().ok_or_else(mismatch)?;
    Ok((packed_t, coeff_t, block_t))
}

fn triangular_solve_entered(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    a: &HostOperand<'_>,
    b: &HostOperand<'_>,
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
                            exec,
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
                let _ = &exec;
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
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    a: &HostOperand<'_>,
    b: &HostOperand<'_>,
) -> tenferro_tensor::Result<Tensor> {
    validate_square_matrix("solve", a.shape())?;
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        return zeros_like_tensor(b.dtype(), b.shape());
    }

    // A vector right-hand side is solved as a one-column matrix: a metadata reshape of the
    // compact host operand, restored on the result below.
    let (rhs, restore_shape) = match batched_vector_rhs_shape(a, b) {
        Some(matrix_rhs_shape) => (
            reshape_host(b, &matrix_rhs_shape)?,
            Some(b.shape().to_vec()),
        ),
        None => (b.clone(), None),
    };

    let result = {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                same_variant_pair!(
                    a,
                    &rhs,
                    |a, b| { linalg::faer::solve(exec, buffers, a, b, false) },
                    unsupported_pair("solve", a, &rhs)
                )
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = &exec;
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
        tenferro_cpu::reshape_tensor(exec, buffers, &result, &shape)
    } else {
        Ok(result)
    }
}

fn solve_from_views_entered(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    a: HostOperand<'_>,
    b: HostOperand<'_>,
) -> tenferro_tensor::Result<Tensor> {
    // The provider's view kernels are shared with the dynamic `with_exec` entries.
    let a = a.into_dynamic();
    let b = b.into_dynamic();
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                match (a, b) {
                    (TensorView::F32(a), TensorView::F32(b)) => {
                        linalg::faer::solve_from_views(exec, buffers, a, b, false)
                            .map(Tensor::from_typed::<f32>)
                    }
                    (TensorView::F64(a), TensorView::F64(b)) => {
                        linalg::faer::solve_from_views(exec, buffers, a, b, false)
                            .map(Tensor::from_typed::<f64>)
                    }
                    (TensorView::C32(a), TensorView::C32(b)) => {
                        linalg::faer::solve_from_views(exec, buffers, a, b, false)
                            .map(Tensor::from_typed::<Complex32>)
                    }
                    (TensorView::C64(a), TensorView::C64(b)) => {
                        linalg::faer::solve_from_views(exec, buffers, a, b, false)
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
                let _ = &exec;
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
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    a: HostOperand<'_>,
    b: HostOperand<'_>,
    out: TensorViewMut<'_, Host>,
) -> tenferro_tensor::Result<()> {
    // The provider's view kernels are shared with the dynamic `with_exec` entries.
    let a = a.into_dynamic();
    let b = b.into_dynamic();
    let out = out.into_dynamic();
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                match (a, b, out) {
                    (TensorView::F32(a), TensorView::F32(b), TensorViewMut::F32(mut out)) => {
                        linalg::faer::solve_into(exec, buffers, a, b, &mut out, false)
                    }
                    (TensorView::F64(a), TensorView::F64(b), TensorViewMut::F64(mut out)) => {
                        linalg::faer::solve_into(exec, buffers, a, b, &mut out, false)
                    }
                    (TensorView::C32(a), TensorView::C32(b), TensorViewMut::C32(mut out)) => {
                        linalg::faer::solve_into(exec, buffers, a, b, &mut out, false)
                    }
                    (TensorView::C64(a), TensorView::C64(b), TensorViewMut::C64(mut out)) => {
                        linalg::faer::solve_into(exec, buffers, a, b, &mut out, false)
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
                let _ = &exec;
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

fn cholesky_entered(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: &HostOperand<'_>,
) -> tenferro_tensor::Result<Tensor> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                match input.dtype() {
                    DType::F32 => linalg::faer::cholesky(
                        exec,
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("cholesky", input.dtype()))?,
                    )
                    .map(Tensor::from_typed::<f32>),
                    DType::F64 => linalg::faer::cholesky(
                        exec,
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("cholesky", input.dtype()))?,
                    )
                    .map(Tensor::from_typed::<f64>),
                    DType::C32 => linalg::faer::cholesky(
                        exec,
                        buffers,
                        input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("cholesky", input.dtype()))?,
                    )
                    .map(Tensor::from_typed::<Complex32>),
                    DType::C64 => linalg::faer::cholesky(
                        exec,
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
                let _ = &exec;
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
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: &HostOperand<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                match input.dtype() {
                    DType::F32 => linalg::faer::lu(
                        exec,
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("lu", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
                    DType::F64 => linalg::faer::lu(
                        exec,
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("lu", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
                    DType::C32 => linalg::faer::lu(
                        exec,
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
                        exec,
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
                let _ = &exec;
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
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: &HostOperand<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    // One complete-pivoting kernel (tlinalg, faer) serves both CPU builds: LAPACK has no
    // rank-revealing complete-pivot LU (`?getc2` errors on singular input, perturbs small
    // pivots and breaks ties toward the last maximum), and the factorization is level-2 work.
    match input.dtype() {
        DType::F32 => linalg::faer::full_piv_lu(
            exec,
            buffers,
            input
                .as_typed::<f32>()
                .ok_or_else(|| unsupported_dtype("full_piv_lu", input.dtype()))?,
        )
        .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
        DType::F64 => linalg::faer::full_piv_lu(
            exec,
            buffers,
            input
                .as_typed::<f64>()
                .ok_or_else(|| unsupported_dtype("full_piv_lu", input.dtype()))?,
        )
        .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
        DType::C32 => linalg::faer::full_piv_lu(
            exec,
            buffers,
            input
                .as_typed::<Complex32>()
                .ok_or_else(|| unsupported_dtype("full_piv_lu", input.dtype()))?,
        )
        .and_then(full_piv_lu_c32_outputs_to_public_tensors),
        DType::C64 => linalg::faer::full_piv_lu(
            exec,
            buffers,
            input
                .as_typed::<Complex64>()
                .ok_or_else(|| unsupported_dtype("full_piv_lu", input.dtype()))?,
        )
        .and_then(full_piv_lu_c64_outputs_to_public_tensors),
        _ => Err(unsupported_dtype("full_piv_lu", input.dtype())),
    }
}

fn svd_entered(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: &HostOperand<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                match input.dtype() {
                    DType::F32 => linalg::faer::svd(
                        exec,
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("svd", input.dtype()))?,
                    )
                    .map(svd_erased),
                    DType::F64 => linalg::faer::svd(
                        exec,
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("svd", input.dtype()))?,
                    )
                    .map(svd_erased),
                    DType::C32 => linalg::faer::svd(
                        exec,
                        buffers,
                        input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("svd", input.dtype()))?,
                    )
                    .map(svd_erased),
                    DType::C64 => linalg::faer::svd(
                        exec,
                        buffers,
                        input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("svd", input.dtype()))?,
                    )
                    .map(svd_erased),
                    _ => Err(unsupported_dtype("svd", input.dtype())),
                }
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = &exec;
                match input.dtype() {
                    DType::F32 => linalg::blas::svd(
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("svd", input.dtype()))?,
                    )
                    .map(svd_erased),
                    DType::F64 => linalg::blas::svd(
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("svd", input.dtype()))?,
                    )
                    .map(svd_erased),
                    DType::C32 => {
                        let t = input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("svd", input.dtype()))?;
                        linalg::blas::svd(buffers, t).map(svd_erased)
                    }
                    DType::C64 => {
                        let t = input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("svd", input.dtype()))?;
                        linalg::blas::svd(buffers, t).map(svd_erased)
                    }
                    _ => Err(unsupported_dtype("svd", input.dtype())),
                }
            }
        }
    }
}

fn svd_full_entered(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: &HostOperand<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                match input.dtype() {
                    DType::F32 => linalg::faer::svd_full(
                        exec,
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("svd_full", input.dtype()))?,
                    )
                    .map(svd_erased),
                    DType::F64 => linalg::faer::svd_full(
                        exec,
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("svd_full", input.dtype()))?,
                    )
                    .map(svd_erased),
                    DType::C32 => linalg::faer::svd_full(
                        exec,
                        buffers,
                        input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("svd_full", input.dtype()))?,
                    )
                    .map(svd_erased),
                    DType::C64 => linalg::faer::svd_full(
                        exec,
                        buffers,
                        input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("svd_full", input.dtype()))?,
                    )
                    .map(svd_erased),
                    _ => Err(unsupported_dtype("svd_full", input.dtype())),
                }
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = &exec;
                match input.dtype() {
                    DType::F32 => linalg::blas::svd_full(
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("svd_full", input.dtype()))?,
                    )
                    .map(svd_erased),
                    DType::F64 => linalg::blas::svd_full(
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("svd_full", input.dtype()))?,
                    )
                    .map(svd_erased),
                    DType::C32 => {
                        let t = input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("svd_full", input.dtype()))?;
                        linalg::blas::svd_full(buffers, t).map(svd_erased)
                    }
                    DType::C64 => {
                        let t = input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("svd_full", input.dtype()))?;
                        linalg::blas::svd_full(buffers, t).map(svd_erased)
                    }
                    _ => Err(unsupported_dtype("svd_full", input.dtype())),
                }
            }
        }
    }
}

fn svd_values_entered(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: &HostOperand<'_>,
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
                        linalg::faer::svd_values(exec, buffers, t).map(Tensor::from_typed::<f32>)
                    }
                    DType::F64 => {
                        let t = input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("svd_values", input.dtype()))?;
                        linalg::faer::svd_values(exec, buffers, t).map(Tensor::from_typed::<f64>)
                    }
                    DType::C32 => {
                        let t = input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("svd_values", input.dtype()))?;
                        linalg::faer::svd_values(exec, buffers, t).map(Tensor::from_typed::<f32>)
                    }
                    DType::C64 => {
                        let t = input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("svd_values", input.dtype()))?;
                        linalg::faer::svd_values(exec, buffers, t).map(Tensor::from_typed::<f64>)
                    }
                    _ => Err(unsupported_dtype("svd_values", input.dtype())),
                }
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = &exec;
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
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: &HostOperand<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                match input.dtype() {
                    DType::F32 => linalg::faer::qr(
                        exec,
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("qr", input.dtype()))?,
                    )
                    .map(pair_erased),
                    DType::F64 => linalg::faer::qr(
                        exec,
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("qr", input.dtype()))?,
                    )
                    .map(pair_erased),
                    DType::C32 => linalg::faer::qr(
                        exec,
                        buffers,
                        input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("qr", input.dtype()))?,
                    )
                    .map(pair_erased),
                    DType::C64 => linalg::faer::qr(
                        exec,
                        buffers,
                        input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("qr", input.dtype()))?,
                    )
                    .map(pair_erased),
                    _ => Err(unsupported_dtype("qr", input.dtype())),
                }
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = &exec;
                match input.dtype() {
                    DType::F32 => linalg::blas::qr(
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("qr", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed).collect()),
                    DType::F64 => linalg::blas::qr(
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("qr", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed).collect()),
                    DType::C32 => linalg::blas::qr(
                        buffers,
                        input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("qr", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed).collect()),
                    DType::C64 => linalg::blas::qr(
                        buffers,
                        input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("qr", input.dtype()))?,
                    )
                    .map(|outputs| outputs.into_iter().map(Tensor::from_typed).collect()),
                    _ => Err(unsupported_dtype("qr", input.dtype())),
                }
            }
        }
    }
}

fn rank_revealing_qr_entered(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: &HostOperand<'_>,
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
                            exec,
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
                            exec,
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
                                exec,
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
                                exec,
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
                let _ = &exec;
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
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: &HostOperand<'_>,
) -> tenferro_tensor::Result<CompactQrResult> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                macro_rules! factor {
                    ($tensor:expr, $variant:ident) => {{
                        let (packed, coeff) =
                            linalg::faer::compact_factor_2d(exec, buffers, $tensor)?;
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
                let _ = &exec;
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
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    q: &HostOperand<'_>,
    r: &HostOperand<'_>,
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
                    let (packed, coeff) = linalg::faer::from_factors_2d(exec, buffers, $q, $r)?;
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
        let _ = &exec;
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
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    packed: &HostOperand<'_>,
    coeff: &HostOperand<'_>,
    block: &HostOperand<'_>,
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
                        linalg::faer::append_2d(exec, buffers, $packed, $coeff, $block)?;
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
        let _ = &exec;
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
    _context: &mut Exec<'_>,
    buffers: &mut BufferPool,
    packed: &HostOperand<'_>,
    coeff: &HostOperand<'_>,
    options: crate::QrOptions,
) -> tenferro_tensor::Result<Tensor> {
    let positive = options.gauge == crate::QrGauge::PositiveDiagonal;
    #[cfg(feature = "native")]
    {
        #[cfg(feature = "native")]
        return same_variant_pair!(
            packed,
            coeff,
            |p, c| linalg::faer::raw_r_2d(buffers, p, c, positive),
            Err(Error::dtype_mismatch(
                "householder_qr_r",
                packed.dtype(),
                coeff.dtype(),
            ))
        );
    }
    #[cfg(feature = "blas")]
    {
        // The LAPACK reflector state has no pooled R destination; its extraction builds the
        // response vector itself.
        let _ = buffers;
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
    _context: &mut Exec<'_>,
    buffers: &mut BufferPool,
    packed: &HostOperand<'_>,
    coeff: &HostOperand<'_>,
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
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: &HostOperand<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                match input.dtype() {
                    DType::F32 => linalg::faer::eigh(
                        exec,
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("eigh", input.dtype()))?,
                    )
                    .map(pair_erased),
                    DType::F64 => linalg::faer::eigh(
                        exec,
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("eigh", input.dtype()))?,
                    )
                    .map(pair_erased),
                    DType::C32 => linalg::faer::eigh(
                        exec,
                        buffers,
                        input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("eigh", input.dtype()))?,
                    )
                    .map(pair_erased),
                    DType::C64 => linalg::faer::eigh(
                        exec,
                        buffers,
                        input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("eigh", input.dtype()))?,
                    )
                    .map(pair_erased),
                    _ => Err(unsupported_dtype("eigh", input.dtype())),
                }
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = &exec;
                match input.dtype() {
                    DType::F32 => linalg::blas::eigh(
                        buffers,
                        input
                            .as_typed::<f32>()
                            .ok_or_else(|| unsupported_dtype("eigh", input.dtype()))?,
                    )
                    .map(pair_erased),
                    DType::F64 => linalg::blas::eigh(
                        buffers,
                        input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("eigh", input.dtype()))?,
                    )
                    .map(pair_erased),
                    DType::C32 => {
                        let t = input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("eigh", input.dtype()))?;
                        linalg::blas::eigh(buffers, t).map(pair_erased)
                    }
                    DType::C64 => {
                        let t = input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("eigh", input.dtype()))?;
                        linalg::blas::eigh(buffers, t).map(pair_erased)
                    }
                    _ => Err(unsupported_dtype("eigh", input.dtype())),
                }
            }
        }
    }
}

fn eigh_values_entered(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: &HostOperand<'_>,
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
                        linalg::faer::eigh_values(exec, buffers, t).map(Tensor::from_typed::<f32>)
                    }
                    DType::F64 => {
                        let t = input
                            .as_typed::<f64>()
                            .ok_or_else(|| unsupported_dtype("eigh_values", input.dtype()))?;
                        linalg::faer::eigh_values(exec, buffers, t).map(Tensor::from_typed::<f64>)
                    }
                    DType::C32 => {
                        let t = input
                            .as_typed::<Complex32>()
                            .ok_or_else(|| unsupported_dtype("eigh_values", input.dtype()))?;
                        linalg::faer::eigh_values(exec, buffers, t).map(Tensor::from_typed::<f32>)
                    }
                    DType::C64 => {
                        let t = input
                            .as_typed::<Complex64>()
                            .ok_or_else(|| unsupported_dtype("eigh_values", input.dtype()))?;
                        linalg::faer::eigh_values(exec, buffers, t).map(Tensor::from_typed::<f64>)
                    }
                    _ => Err(unsupported_dtype("eigh_values", input.dtype())),
                }
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = &exec;
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
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: &HostOperand<'_>,
) -> tenferro_tensor::Result<Tensor> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                linalg::faer::eig_values(exec, buffers, input)
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = &exec;
                linalg::blas::eig_values(buffers, input)
            }
        }
    }
}

fn eig_entered(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: &HostOperand<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    {
        #[cfg(feature = "native")]
        {
            #[cfg(feature = "native")]
            {
                linalg::faer::eig(exec, buffers, input)
            }
        }
        #[cfg(feature = "blas")]
        {
            #[cfg(feature = "blas")]
            {
                let _ = &exec;
                linalg::blas::eig(buffers, input)
            }
        }
    }
}

fn faer_strided_host_ok(input: &HostOperand<'_>) -> bool {
    match input {
        TensorView::F32(view) => linalg::faer::faer_strided_ok(view),
        TensorView::F64(view) => linalg::faer::faer_strided_ok(view),
        TensorView::C32(view) => linalg::faer::faer_strided_ok(view),
        TensorView::C64(view) => linalg::faer::faer_strided_ok(view),
        // faer does not compute on the integer and boolean tags here.
        TensorView::I32(_) | TensorView::I64(_) | TensorView::Bool(_) => false,
    }
}

/// Can this right-hand side be gathered directly by the faer view path?
///
/// The RHS is copied element by element, so strides may be arbitrary; only the
/// matrix rank every provider requires and a supported dtype matter (the host
/// type already rules out a backend buffer).
#[cfg(feature = "native")]
fn faer_rhs_host_ok(input: &HostOperand<'_>) -> bool {
    input.shape().len() == 2
        && matches!(
            input.dtype(),
            DType::F32 | DType::F64 | DType::C32 | DType::C64
        )
}

#[cfg(feature = "native")]
fn rank_revealing_qr_faer_view_entered(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: HostOperand<'_>,
    options: crate::RankRevealingQrOptions,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    // The provider's view kernels are shared with the dynamic `with_exec` entries.
    let input = input.into_dynamic();
    macro_rules! factor {
        ($view:expr, $scalar:ty) => {
            linalg::faer::rank_revealing_qr_view(exec, buffers, $view, options).map(|result| {
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
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    a: HostOperand<'_>,
    b: HostOperand<'_>,
    options: TriangularSolveOptions,
) -> tenferro_tensor::Result<Tensor> {
    // The provider's view kernels are shared with the dynamic `with_exec` entries.
    let a = a.into_dynamic();
    let b = b.into_dynamic();
    let TriangularSolveOptions {
        left_side,
        lower,
        transpose_a,
        unit_diagonal,
    } = options;
    macro_rules! solve {
        ($a:expr, $b:expr, $scalar:ty) => {
            linalg::faer::triangular_solve_view(
                exec,
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
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: HostOperand<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    // The provider's view kernels are shared with the dynamic `with_exec` entries.
    let input = input.into_dynamic();
    match input {
        TensorView::F32(view) => linalg::faer::svd_view(exec, buffers, view).map(svd_erased),
        TensorView::F64(view) => linalg::faer::svd_view(exec, buffers, view).map(svd_erased),
        TensorView::C32(view) => linalg::faer::svd_view(exec, buffers, view).map(svd_erased),
        TensorView::C64(view) => linalg::faer::svd_view(exec, buffers, view).map(svd_erased),
        unsupported => Err(unsupported_dtype("svd", unsupported.dtype())),
    }
}

#[cfg(feature = "native")]
fn svd_full_faer_view_entered(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: HostOperand<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    // The provider's view kernels are shared with the dynamic `with_exec` entries.
    let input = input.into_dynamic();
    match input {
        TensorView::F32(view) => linalg::faer::svd_full_view(exec, buffers, view).map(svd_erased),
        TensorView::F64(view) => linalg::faer::svd_full_view(exec, buffers, view).map(svd_erased),
        TensorView::C32(view) => linalg::faer::svd_full_view(exec, buffers, view).map(svd_erased),
        TensorView::C64(view) => linalg::faer::svd_full_view(exec, buffers, view).map(svd_erased),
        unsupported => Err(unsupported_dtype("svd_full", unsupported.dtype())),
    }
}

#[cfg(feature = "native")]
fn svd_values_faer_view_entered(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: HostOperand<'_>,
) -> tenferro_tensor::Result<Tensor> {
    // The provider's view kernels are shared with the dynamic `with_exec` entries.
    let input = input.into_dynamic();
    match input {
        TensorView::F32(view) => {
            linalg::faer::svd_values_view(exec, buffers, view).map(Tensor::from_typed::<f32>)
        }
        TensorView::F64(view) => {
            linalg::faer::svd_values_view(exec, buffers, view).map(Tensor::from_typed::<f64>)
        }
        TensorView::C32(view) => {
            linalg::faer::svd_values_view(exec, buffers, view).map(Tensor::from_typed::<f32>)
        }
        TensorView::C64(view) => {
            linalg::faer::svd_values_view(exec, buffers, view).map(Tensor::from_typed::<f64>)
        }
        unsupported => Err(unsupported_dtype("svd_values", unsupported.dtype())),
    }
}

#[cfg(feature = "native")]
fn qr_faer_view_entered(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: HostOperand<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    // The provider's view kernels are shared with the dynamic `with_exec` entries.
    let input = input.into_dynamic();
    match input {
        TensorView::F32(view) => linalg::faer::qr_view(exec, buffers, view).map(pair_erased),
        TensorView::F64(view) => linalg::faer::qr_view(exec, buffers, view).map(pair_erased),
        TensorView::C32(view) => linalg::faer::qr_view(exec, buffers, view).map(pair_erased),
        TensorView::C64(view) => linalg::faer::qr_view(exec, buffers, view).map(pair_erased),
        unsupported => Err(unsupported_dtype("qr", unsupported.dtype())),
    }
}

#[cfg(feature = "native")]
fn eigh_faer_view_entered(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: HostOperand<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    // The provider's view kernels are shared with the dynamic `with_exec` entries.
    let input = input.into_dynamic();
    match input {
        TensorView::F32(view) => linalg::faer::eigh_view(exec, buffers, view).map(pair_erased),
        TensorView::F64(view) => linalg::faer::eigh_view(exec, buffers, view).map(pair_erased),
        TensorView::C32(view) => linalg::faer::eigh_view(exec, buffers, view).map(pair_erased),
        TensorView::C64(view) => linalg::faer::eigh_view(exec, buffers, view).map(pair_erased),
        unsupported => Err(unsupported_dtype("eigh", unsupported.dtype())),
    }
}

#[cfg(feature = "native")]
fn eigh_values_faer_view_entered(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: HostOperand<'_>,
) -> tenferro_tensor::Result<Tensor> {
    // The provider's view kernels are shared with the dynamic `with_exec` entries.
    let input = input.into_dynamic();
    match input {
        TensorView::F32(view) => {
            linalg::faer::eigh_values_view(exec, buffers, view).map(Tensor::from_typed::<f32>)
        }
        TensorView::F64(view) => {
            linalg::faer::eigh_values_view(exec, buffers, view).map(Tensor::from_typed::<f64>)
        }
        TensorView::C32(view) => {
            linalg::faer::eigh_values_view(exec, buffers, view).map(Tensor::from_typed::<f32>)
        }
        TensorView::C64(view) => {
            linalg::faer::eigh_values_view(exec, buffers, view).map(Tensor::from_typed::<f64>)
        }
        unsupported => Err(unsupported_dtype("eigh_values", unsupported.dtype())),
    }
}

#[cfg(feature = "native")]
fn cholesky_faer_view_entered(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: HostOperand<'_>,
) -> tenferro_tensor::Result<Tensor> {
    // The provider's view kernels are shared with the dynamic `with_exec` entries.
    let input = input.into_dynamic();
    match input {
        TensorView::F32(view) => {
            linalg::faer::cholesky_view(exec, buffers, view).map(Tensor::from_typed::<f32>)
        }
        TensorView::F64(view) => {
            linalg::faer::cholesky_view(exec, buffers, view).map(Tensor::from_typed::<f64>)
        }
        TensorView::C32(view) => {
            linalg::faer::cholesky_view(exec, buffers, view).map(Tensor::from_typed::<Complex32>)
        }
        TensorView::C64(view) => {
            linalg::faer::cholesky_view(exec, buffers, view).map(Tensor::from_typed::<Complex64>)
        }
        unsupported => Err(unsupported_dtype("cholesky", unsupported.dtype())),
    }
}

#[cfg(feature = "native")]
fn lu_faer_view_entered(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: HostOperand<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    // The provider's view kernels are shared with the dynamic `with_exec` entries.
    let input = input.into_dynamic();
    match input {
        TensorView::F32(view) => linalg::faer::lu_view(exec, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
        TensorView::F64(view) => linalg::faer::lu_view(exec, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
        TensorView::C32(view) => linalg::faer::lu_view(exec, buffers, view).map(|outputs| {
            outputs
                .into_iter()
                .map(Tensor::from_typed::<Complex32>)
                .collect()
        }),
        TensorView::C64(view) => linalg::faer::lu_view(exec, buffers, view).map(|outputs| {
            outputs
                .into_iter()
                .map(Tensor::from_typed::<Complex64>)
                .collect()
        }),
        unsupported => Err(unsupported_dtype("lu", unsupported.dtype())),
    }
}

fn full_piv_lu_faer_view_entered(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    input: HostOperand<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    // The provider's view kernels are shared with the dynamic `with_exec` entries.
    let input = input.into_dynamic();
    match input {
        TensorView::F32(view) => linalg::faer::full_piv_lu_view(exec, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f32>).collect()),
        TensorView::F64(view) => linalg::faer::full_piv_lu_view(exec, buffers, view)
            .map(|outputs| outputs.into_iter().map(Tensor::from_typed::<f64>).collect()),
        TensorView::C32(view) => {
            linalg::faer::full_piv_lu_view(exec, buffers, view).map(|outputs| {
                outputs
                    .into_iter()
                    .map(Tensor::from_typed::<Complex32>)
                    .collect()
            })
        }
        TensorView::C64(view) => {
            linalg::faer::full_piv_lu_view(exec, buffers, view).map(|outputs| {
                outputs
                    .into_iter()
                    .map(Tensor::from_typed::<Complex64>)
                    .collect()
            })
        }
        unsupported => Err(unsupported_dtype("full_piv_lu", unsupported.dtype())),
    }
}

fn ensure_supported_linalg_pair(
    op: &'static str,
    lhs: &HostOperand<'_>,
    rhs: &HostOperand<'_>,
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

/// Borrow `read` as a preset-scalar view for `op`.
///
/// [`TensorRead::tensor_view`] answers `None` for a caller-owned payload
/// (#2486). The entries call [`ensure_supported_linalg_dtype`] first, which
/// already refuses that payload, so this reports the same typed error should a
/// view ever be requested without that check.
fn linalg_view<'a>(
    op: &'static str,
    read: TensorRead<'a>,
) -> tenferro_tensor::Result<TensorView<'a>> {
    let dtype = read.dtype();
    read.tensor_view()
        .ok_or_else(|| unsupported_dtype(op, dtype))
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

fn batched_vector_rhs_shape(a: &HostOperand<'_>, b: &HostOperand<'_>) -> Option<Vec<usize>> {
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

fn zeros_like_tensor(dtype: DType, shape: &[usize]) -> tenferro_tensor::Result<Tensor> {
    Ok(match dtype {
        // A caller-owned payload has no zero-like runtime tensor.
        DType::External(type_id) => {
            return Err(unsupported_dtype(
                "zeros_like_tensor",
                DType::External(type_id),
            ));
        }
        DType::F32 => Tensor::from_typed::<f32>(TypedTensor::zeros(shape.to_vec())?),
        DType::F64 => Tensor::from_typed::<f64>(TypedTensor::zeros(shape.to_vec())?),
        DType::I32 => Tensor::from_typed::<i32>(TypedTensor::zeros(shape.to_vec())?),
        DType::I64 => Tensor::from_typed::<i64>(TypedTensor::zeros(shape.to_vec())?),
        DType::Bool => {
            let len = shape
                .iter()
                .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
                .ok_or_else(|| {
                    Error::invalid_argument(
                        "zeros_like_tensor",
                        "shape",
                        "element count overflows usize",
                    )
                })?;
            Tensor::from_typed::<bool>(TypedTensor::from_vec_col_major(
                shape.to_vec(),
                vec![false; len],
            )?)
        }
        DType::C32 => Tensor::from_typed::<Complex32>(TypedTensor::zeros(shape.to_vec())?),
        DType::C64 => Tensor::from_typed::<Complex64>(TypedTensor::zeros(shape.to_vec())?),
    })
}

fn complex32_real_part_tensor(
    values: TypedTensor<Complex32>,
) -> tenferro_tensor::Result<TypedTensor<f32>> {
    let mut out = TypedTensor::from_vec_col_major(
        values.shape().to_vec(),
        values.host_data()?.iter().map(|value| value.re).collect(),
    )?;
    out.set_placement(values.placement().clone())?;
    Ok(out)
}

fn complex64_real_part_tensor(
    values: TypedTensor<Complex64>,
) -> tenferro_tensor::Result<TypedTensor<f64>> {
    let mut out = TypedTensor::from_vec_col_major(
        values.shape().to_vec(),
        values.host_data()?.iter().map(|value| value.re).collect(),
    )?;
    out.set_placement(values.placement().clone())?;
    Ok(out)
}

/// Erase a typed `(U, S, Vᴴ)`; the spectrum is already in the real scalar.
fn svd_erased<T: TensorScalar, R: TensorScalar>(
    (u, s, vt): (TypedTensor<T>, TypedTensor<R>, TypedTensor<T>),
) -> Vec<Tensor> {
    vec![
        Tensor::from_typed(u),
        Tensor::from_typed(s),
        Tensor::from_typed(vt),
    ]
}

/// Erase a typed pair of factorization outputs (`(Q, R)` or `(values, vectors)`).
fn pair_erased<A: TensorScalar, B: TensorScalar>(
    (first, second): (TypedTensor<A>, TypedTensor<B>),
) -> Vec<Tensor> {
    vec![Tensor::from_typed(first), Tensor::from_typed(second)]
}

fn full_piv_lu_output_count_error(count: usize) -> Error {
    Error::Internal(format!(
        "full_piv_lu produced an invalid output count: expected 5, got {count}"
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
    lhs: &HostOperand<'_>,
    rhs: &HostOperand<'_>,
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
