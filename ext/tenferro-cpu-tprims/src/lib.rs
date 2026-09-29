#![deny(missing_docs)]

//! Optional tprims-backed GEMM, `dot_general` and linear-algebra providers
//! for `tenferro-cpu`.
//!
//! [tprims](https://github.com/tensor4all/tprims-rs) takes an explicit
//! execution context that borrows a Rayon pool. [`TprimsProvider`] builds one
//! from the provider's [`CpuExecutionContext`]: the pool of the inner parallel
//! region ([`CpuExecutionContext::rayon_pool`]) with the context's thread
//! budget, or serial execution otherwise. Everything it does not handle is
//! reported as unsupported before any output is written, so the selected
//! `tenferro-cpu` backend runs it.
//!
//! # Examples
//!
//! ```
//! use std::sync::Arc;
//! use tenferro_cpu::{CpuBackend, CpuBackendKind, CpuProviderBundle};
//! use tenferro_cpu_tprims::TprimsProvider;
//!
//! let builder = CpuProviderBundle::builder(CpuBackendKind::default_compiled())
//!     .gemm_provider(Arc::new(TprimsProvider::new()))
//!     .prefer_general_contraction_provider(Arc::new(TprimsProvider::new()));
//! // Linear algebra (tenferro_linalg::cpu_kernels): Cholesky, solves, SVD, QR, eigh.
//! let bundle =
//!     tenferro_linalg::cpu_kernels::install_linalg_kernels(builder, Arc::new(TprimsProvider::new()))
//!         .build()?;
//! let backend = CpuBackend::new().with_provider_bundle(bundle)?;
//! # let _ = backend;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use num_complex::{Complex32, Complex64};
use strided_view::{StridedView, StridedViewMut};
use tenferro_cpu::provider::{
    CpuBatchedMatrixLayout, CpuDotGeneralRequest, CpuExecutionContext, CpuGemmProvider,
    CpuGemmRequest, CpuGeneralContractionProvider, CpuGroupedGemmRequest, CpuOperand,
    CpuProviderOutcome, CpuProviderUnsupported, CpuVendorBatch,
};
use tenferro_cpu::{CpuPlacementControl, CpuProviderExecutionCapabilities, CpuThreadCountControl};
use tenferro_tensor::{
    col_major_strides, ContractionScalar, DType, Error, Result, TensorRead, TensorScalar,
    TensorView, TensorViewMut, TensorWrite, TypedTensorView, TypedTensorViewMut,
};
use tprims_blas::{BatchIn, BatchStrategy, Conj, GroupedJob, MatIn};
use tprims_contract::{ContractPlan, DotGeneral, Flags, Strategy};
use tprims_exec::{Exec, Pool};

mod linalg;

/// tprims implementation of `tenferro-cpu`'s GEMM and general-contraction
/// provider slots.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::CpuGemmProvider;
/// use tenferro_cpu_tprims::TprimsProvider;
/// let provider: &dyn CpuGemmProvider = &TprimsProvider::new();
/// let _ = provider.execution_capabilities();
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct TprimsProvider;

impl TprimsProvider {
    /// Create the provider.
    ///
    /// # Examples
    ///
    /// ```
    /// let _ = tenferro_cpu_tprims::TprimsProvider::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// Parallel work runs on the engine's workers, within the per-call budget.
fn capabilities() -> CpuProviderExecutionCapabilities {
    CpuProviderExecutionCapabilities {
        thread_count: CpuThreadCountControl::PerCallUpperBound,
        placement: CpuPlacementControl::EngineWorkers,
        worker_local_sequential: true,
        accepts_sequential: true,
        accepts_outer: true,
        accepts_inner: true,
    }
}

/// Run `f` with the tprims execution context for `ctx`: the inner region's
/// pool bounded by the thread budget, or serial.
fn with_exec<R>(ctx: &CpuExecutionContext<'_>, f: impl FnOnce(&Exec<'_>) -> R) -> R {
    match ctx.rayon_pool() {
        Some(pool) => {
            let pool = Pool::borrow(pool);
            let exec = Exec::rayon(&pool)
                .with_budget(ctx.thread_budget().get())
                .unwrap_or(Exec::Serial);
            f(&exec)
        }
        None => f(&Exec::Serial),
    }
}

/// The four element types tprims supports, with their tenferro views.
trait Elem: TensorScalar + tprims_blas::Scalar {
    /// Complex conjugate (identity for real types).
    fn conj_elem(self) -> Self;
    /// A column-major tensor of this type's real counterpart.
    fn real_tensor(
        shape: Vec<usize>,
        data: Vec<<Self as tprims_blas::Scalar>::Re>,
    ) -> Result<tenferro_tensor::Tensor>;
    fn scalar(s: ContractionScalar) -> Option<Self>;
    fn view<'b, 'a>(v: &'b TensorView<'a>) -> Option<&'b TypedTensorView<'a, Self>>;
    fn view_mut<'b, 'a>(
        v: &'b mut TensorViewMut<'a>,
    ) -> Option<&'b mut TypedTensorViewMut<'a, Self>>;
}

macro_rules! elem {
    ($t:ty, $v:ident, $conj:expr) => {
        impl Elem for $t {
            fn conj_elem(self) -> Self {
                let conj: fn(Self) -> Self = $conj;
                conj(self)
            }
            fn real_tensor(
                shape: Vec<usize>,
                data: Vec<<Self as tprims_blas::Scalar>::Re>,
            ) -> Result<tenferro_tensor::Tensor> {
                tenferro_tensor::Tensor::from_vec_col_major(shape, data)
            }
            fn scalar(s: ContractionScalar) -> Option<Self> {
                match s {
                    ContractionScalar::$v(x) => Some(x),
                    _ => None,
                }
            }
            fn view<'b, 'a>(v: &'b TensorView<'a>) -> Option<&'b TypedTensorView<'a, Self>> {
                match v {
                    TensorView::$v(x) => Some(x),
                    _ => None,
                }
            }
            fn view_mut<'b, 'a>(
                v: &'b mut TensorViewMut<'a>,
            ) -> Option<&'b mut TypedTensorViewMut<'a, Self>> {
                match v {
                    TensorViewMut::$v(x) => Some(x),
                    _ => None,
                }
            }
        }
    };
}
elem!(f32, F32, |x| x);
elem!(f64, F64, |x| x);
elem!(Complex32, C32, |x| x.conj());
elem!(Complex64, C64, |x| x.conj());

/// A read operand: its whole backing storage and the layout of the tensor in
/// it (element strides and offset).
struct In<'a, T> {
    data: &'a [T],
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
}

fn read<'a, T: Elem>(r: &'a TensorRead<'_>) -> Result<Option<In<'a, T>>> {
    Ok(match r {
        TensorRead::Tensor(t) => match t.as_typed::<T>() {
            Some(t) => Some(In {
                data: t.host_data()?,
                strides: col_major_strides(t.shape())?,
                shape: t.shape().to_vec(),
                offset: 0,
            }),
            None => None,
        },
        TensorRead::View(v) => match T::view(v) {
            Some(v) => Some(In {
                data: v.host_storage()?,
                shape: v.shape().to_vec(),
                strides: v.strides().to_vec(),
                offset: v.offset(),
            }),
            None => None,
        },
    })
}

/// The written operand, as [`In`].
struct Out<'a, T> {
    data: &'a mut [T],
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
}

fn write<'a, T: Elem>(w: &'a mut TensorWrite<'_>) -> Result<Option<Out<'a, T>>> {
    Ok(match w {
        TensorWrite::Tensor(t) => match t.as_typed_mut::<T>() {
            Some(t) => {
                let shape = t.shape().to_vec();
                let strides = col_major_strides(&shape)?;
                Some(Out {
                    data: t.host_data_mut()?,
                    shape,
                    strides,
                    offset: 0,
                })
            }
            None => None,
        },
        TensorWrite::View(v) => match T::view_mut(v) {
            Some(v) => {
                let (shape, strides, offset) =
                    (v.shape().to_vec(), v.strides().to_vec(), v.offset());
                Some(Out {
                    data: v.host_storage_mut()?,
                    shape,
                    strides,
                    offset,
                })
            }
            None => None,
        },
    })
}

fn conj(c: bool) -> Conj {
    if c {
        Conj::Yes
    } else {
        Conj::No
    }
}

fn failure(op: &'static str, e: impl std::fmt::Display) -> Error {
    Error::backend_failure(op, format!("tprims: {e}"))
}

const UNSUPPORTED_LAYOUT_OUT: CpuProviderOutcome =
    CpuProviderOutcome::Unsupported(CpuProviderUnsupported::Layout(CpuOperand::Output));

/// dtype dispatch for the four supported element types.
macro_rules! by_dtype {
    ($dtype:expr, $f:ident($($arg:expr),*)) => {
        match $dtype {
            DType::F32 => $f::<f32>($($arg),*),
            DType::F64 => $f::<f64>($($arg),*),
            DType::C32 => $f::<Complex32>($($arg),*),
            DType::C64 => $f::<Complex64>($($arg),*),
            other => Ok(CpuProviderOutcome::Unsupported(CpuProviderUnsupported::DType(other))),
        }
    };
}

/// A `[rows, cols]` or `[rows, cols, batch]` view of `layout` in `data`.
fn mat_dims(
    rows: usize,
    cols: usize,
    batch: Option<usize>,
    l: CpuBatchedMatrixLayout,
) -> (Vec<usize>, Vec<isize>) {
    match batch {
        None => (vec![rows, cols], vec![l.row_stride(), l.column_stride()]),
        Some(b) => (
            vec![rows, cols, b],
            vec![l.row_stride(), l.column_stride(), l.batch_stride()],
        ),
    }
}

fn gemm_typed<T: Elem>(
    ctx: &CpuExecutionContext<'_>,
    mut request: CpuGemmRequest<'_, '_, '_>,
    batched: bool,
) -> Result<CpuProviderOutcome> {
    const OP: &str = "gemm";
    if request.vendor_batch() == CpuVendorBatch::Required {
        return Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::RuntimeUnavailable,
        ));
    }
    let acc = request.accumulation();
    let (Some(alpha), Some(beta)) = (T::scalar(acc.alpha), T::scalar(acc.beta)) else {
        return Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::Accumulation,
        ));
    };
    let (m, n, k, count) = (
        request.rows(),
        request.columns(),
        request.contracted(),
        request.batch_count(),
    );
    let (la, lb, lc) = (
        request.lhs_layout(),
        request.rhs_layout(),
        request.output_layout(),
    );
    let (lhs, rhs) = (request.lhs().clone(), request.rhs().clone());
    let (Some(a), Some(b)) = (read::<T>(&lhs)?, read::<T>(&rhs)?) else {
        return Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::DType(lhs.dtype()),
        ));
    };
    let batch = (batched || count != 1).then_some(count);
    let (da, sa) = mat_dims(m, k, batch, la);
    let (db, sb) = mat_dims(k, n, batch, lb);
    let (dc, sc) = mat_dims(m, n, batch, lc);
    let av = StridedView::new(a.data, &da, &sa, la.offset()).map_err(|e| failure(OP, e))?;
    let bv = StridedView::new(b.data, &db, &sb, lb.offset()).map_err(|e| failure(OP, e))?;
    let output = request.output();
    let Some(c) = write::<T>(output)? else {
        return Ok(UNSUPPORTED_LAYOUT_OUT);
    };
    let mut cv = StridedViewMut::new(c.data, &dc, &sc, lc.offset()).map_err(|e| failure(OP, e))?;
    let (ca, cb) = (conj(acc.lhs_conj), conj(acc.rhs_conj));
    // tprims validates before writing, so an error leaves the output intact.
    with_exec(ctx, |exec| match batch {
        None => {
            let (ai, bi) = (
                MatIn {
                    view: &av,
                    conj: ca,
                },
                MatIn {
                    view: &bv,
                    conj: cb,
                },
            );
            tprims_blas::gemm(exec, alpha, ai, bi, beta, &mut cv).map(|_| ())
        }
        Some(_) => {
            let (ai, bi) = (
                BatchIn {
                    view: &av,
                    conj: ca,
                },
                BatchIn {
                    view: &bv,
                    conj: cb,
                },
            );
            tprims_blas::gemm_batched(exec, alpha, ai, bi, beta, &mut cv, BatchStrategy::Auto)
                .map(|_| ())
        }
    })
    .map_err(|e| failure(OP, e))?;
    Ok(CpuProviderOutcome::Executed)
}

fn grouped_typed<T: Elem>(
    ctx: &CpuExecutionContext<'_>,
    mut request: CpuGroupedGemmRequest<'_, '_, '_>,
) -> Result<CpuProviderOutcome> {
    const OP: &str = "grouped_gemm";
    let acc = request.accumulation();
    let (Some(alpha), Some(beta)) = (T::scalar(acc.alpha), T::scalar(acc.beta)) else {
        return Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::Accumulation,
        ));
    };
    let (lhs, rhs) = (request.lhs().clone(), request.rhs().clone());
    let (Some(a), Some(b)) = (read::<T>(&lhs)?, read::<T>(&rhs)?) else {
        return Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::DType(lhs.dtype()),
        ));
    };
    // Job offsets are relative to each operand's view offset.
    let base = |o: isize| usize::try_from(o).map_err(|_| failure(OP, "negative operand offset"));
    let (oa, ob) = (base(a.offset)?, base(b.offset)?);
    let jobs_in: Vec<_> = request.jobs().to_vec();
    let output = request.output();
    let Some(c) = write::<T>(output)? else {
        return Ok(UNSUPPORTED_LAYOUT_OUT);
    };
    let oc = base(c.offset)?;
    let jobs: Vec<GroupedJob> = jobs_in
        .iter()
        .map(|j| GroupedJob {
            a_offset: oa + j.lhs_offset(),
            b_offset: ob + j.rhs_offset(),
            c_offset: oc + j.out_offset(),
            rows: j.rows(),
            inner: j.contracted(),
            cols: j.cols(),
        })
        .collect();
    with_exec(ctx, |exec| {
        tprims_blas::gemm_grouped(
            exec,
            alpha,
            a.data,
            conj(acc.lhs_conj),
            b.data,
            conj(acc.rhs_conj),
            beta,
            c.data,
            &jobs,
        )
    })
    .map_err(|e| failure(OP, e))?;
    Ok(CpuProviderOutcome::Executed)
}

fn dot_general_typed<T: Elem>(
    ctx: &CpuExecutionContext<'_>,
    request: CpuDotGeneralRequest<'_, '_, '_>,
) -> Result<CpuProviderOutcome> {
    const OP: &str = "dot_general";
    let (lhs, rhs, output, axes, acc) = request.into_parts();
    let (Some(alpha), Some(beta)) = (T::scalar(acc.alpha), T::scalar(acc.beta)) else {
        return Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::Accumulation,
        ));
    };
    let (Some(a), Some(b)) = (read::<T>(lhs)?, read::<T>(rhs)?) else {
        return Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::DType(lhs.dtype()),
        ));
    };
    let (lc, rc): (Vec<usize>, Vec<usize>) = axes.contracting_pairs().unzip();
    let (lb, rb): (Vec<usize>, Vec<usize>) = axes.batch_pairs().unzip();
    let cfg = DotGeneral::new(&lc, &rc, &lb, &rb);
    let Some(c) = write::<T>(output)? else {
        return Ok(UNSUPPORTED_LAYOUT_OUT);
    };
    // Output order [lhs free, rhs free, batch] is the same in both libraries.
    // A plan the library cannot build is unsupported, not an error: nothing
    // has been written yet.
    let Ok(plan) = ContractPlan::<T>::new(
        &cfg,
        (&a.shape, &a.strides),
        (&b.shape, &b.strides),
        (&c.shape, &c.strides),
        (conj(acc.lhs_conj), conj(acc.rhs_conj)),
        Strategy::Auto,
        Flags::default(),
    ) else {
        return Ok(UNSUPPORTED_LAYOUT_OUT);
    };
    let av =
        StridedView::new(a.data, &a.shape, &a.strides, a.offset).map_err(|e| failure(OP, e))?;
    let bv =
        StridedView::new(b.data, &b.shape, &b.strides, b.offset).map_err(|e| failure(OP, e))?;
    let mut cv =
        StridedViewMut::new(c.data, &c.shape, &c.strides, c.offset).map_err(|e| failure(OP, e))?;
    with_exec(ctx, |exec| {
        plan.execute(exec, alpha, &av, &bv, beta, &mut cv)
    })
    .map_err(|e| failure(OP, e))?;
    Ok(CpuProviderOutcome::Executed)
}

impl CpuGemmProvider for TprimsProvider {
    fn execution_capabilities(&self) -> CpuProviderExecutionCapabilities {
        capabilities()
    }

    /// One GEMM through `tprims_blas::gemm`.
    ///
    /// # Errors
    ///
    /// [`Error::BackendFailure`] when tprims rejects the validated request
    /// (output untouched) or a host buffer is unavailable.
    fn gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> Result<CpuProviderOutcome> {
        by_dtype!(request.lhs().dtype(), gemm_typed(context, request, false))
    }

    /// A strided batch through `tprims_blas::gemm_batched`.
    ///
    /// # Errors
    ///
    /// As [`CpuGemmProvider::gemm`].
    fn strided_batched_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> Result<CpuProviderOutcome> {
        by_dtype!(request.lhs().dtype(), gemm_typed(context, request, true))
    }

    /// Variable-size jobs through `tprims_blas::gemm_grouped`.
    ///
    /// # Errors
    ///
    /// As [`CpuGemmProvider::gemm`].
    fn grouped_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> Result<CpuProviderOutcome> {
        by_dtype!(request.lhs().dtype(), grouped_typed(context, request))
    }
}

impl CpuGeneralContractionProvider for TprimsProvider {
    fn execution_capabilities(&self) -> CpuProviderExecutionCapabilities {
        capabilities()
    }

    /// A binary contraction through a `tprims_contract::ContractPlan`
    /// (`Strategy::Auto`).
    ///
    /// # Errors
    ///
    /// As [`CpuGemmProvider::gemm`]; layouts tprims cannot plan are reported
    /// as unsupported instead.
    fn dot_general(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuDotGeneralRequest<'_, '_, '_>,
    ) -> Result<CpuProviderOutcome> {
        by_dtype!(request.lhs().dtype(), dot_general_typed(context, request))
    }
}
