//! Which prepared plan the BLAS build runs.
//!
//! `cpueinsum-blas` prepares a vendor-first plan: every problem its layout
//! inference can shape into a GEMM runs on the linked CBLAS, and tcontract's
//! own plan is only the fallback. That bypasses tcontract's c64 `FaerLimit`,
//! which is right for a reference BLAS but not for Accelerate's Complex64
//! kernel: it pays a per right-operand-element read that a thin row role never
//! amortises (tensor4all-rs#1099, M5 Max, thread budget of one, 300 interleaved
//! calls, medians, us):
//!
//! | product | f64 Accelerate | c64 Accelerate | c64 faer |
//! |---|---|---|---|
//! | `1x141 * 141x282` | 2.6 | 62.7 | 15.7 |
//! | `2x141 * 141x282` | 6.7 | 81.0 | 19.2 |
//! | `8x141 * 141x282` | 6.6 | 78.6 | 57.0 |
//! | `256x256 * 256x256` | 82 | 518 | 2137 |
//!
//! Accelerate wins by 4x at `256^3` and loses by 4x at the thin c64 shapes, so
//! this module keeps Accelerate as the default and takes tcontract's plan for a
//! c64 product whose GEMM row role has collapsed to one or two, when tcontract
//! would fuse it onto faer. Every other product keeps the vendor plan: f32, f64
//! and c32 (faer is 1.8x to 3x slower than the vendor for them here) and every
//! product with a real row role.
//!
//! The gate is compiled only for an Accelerate link ([`links_accelerate`]): the
//! numbers above are Accelerate's, and a build that links a reference BLAS must
//! not be re-routed on the strength of them.
//!
//! A routed product can differ from the vendor kernel, because faer accumulates
//! in its own order: measured max |difference| 1.4e-14 to 3.6e-14 against the
//! vendor kernel for the shapes above, which is 2.2e-16 to 1.3e-15 of the output
//! scale. Individual components near cancellation move by many ULP at that
//! scale, the effect the forward-error test in `contraction/tests.rs` documents.
//! This is therefore a c64 numerical-policy change for thin products on an
//! Accelerate link, made deliberately rather than as a side effect of another
//! fix, and every other dtype and shape keeps its current accumulation order.

use cpueinsum::tcontract::api::{DType, Problem};
use cpueinsum::tcontract::{Algorithm, PlanConfig, SliceAccumulationSource};
use cpueinsum::Exec;
use cpueinsum_blas::{BlasError, BlasScalar, ExecutionRoute};

/// A prepared plan for the BLAS build.
///
/// Both arms take the same arguments and report the same route, so the
/// contraction lane stays one code path whichever plan was prepared.
//
// INVARIANT: both plans stay inline in the cached entry, so the split costs one
// discriminant per retained cache slot and no allocation; `large_enum_variant`
// would have this arm boxed, which allocates once per prepared routed plan.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum BinaryPlan<T: BlasScalar> {
    /// tcontract's own plan, prepared exactly as the native build prepares it.
    Native(cpueinsum::BinaryPlan<T>),
    /// The vendor-first adapter.
    Vendor(cpueinsum_blas::BinaryPlan<T>),
}

/// The largest GEMM row role routed to tcontract's plan.
///
/// faer is 4.0x to 4.2x faster at `m = 1` and `m = 2`. `m = 8` measured 1.4x
/// faster on faer as well, but it stays on the vendor kernel: the vendor's
/// advantage grows with the row role (4.1x at `256^3`) and no measurement here
/// locates the crossover.
const THIN_ROW_ROLE: usize = 2;

impl<T: BlasScalar> BinaryPlan<T> {
    /// Prepare the plan this product should run.
    ///
    /// # Errors
    ///
    /// Returns [`BlasError::Prepare`] when the selected plan rejects the dtype,
    /// configuration or problem.
    pub(crate) fn new(problem: &Problem, config: &PlanConfig) -> Result<Self, BlasError> {
        if takes_faer_plan(problem) {
            let native = cpueinsum::BinaryPlan::<T>::new(problem, config)
                .map_err(|source| BlasError::Prepare { source })?;
            // Only the plan tcontract would have chosen is taken; a product
            // tcontract declines keeps the vendor kernel it runs today.
            if native.report().algorithm == Algorithm::Faer {
                return Ok(Self::Native(native));
            }
        }
        cpueinsum_blas::BinaryPlan::new(problem, config).map(Self::Vendor)
    }

    /// Caller packing workspace in elements. tcontract's plan packs into the
    /// `Exec` workspace instead, so it asks for none.
    pub(crate) fn work_len(&self) -> usize {
        match self {
            Self::Native(_) => 0,
            Self::Vendor(plan) => plan.work_len(),
        }
    }

    /// Update initialized output, returning the route that ran.
    ///
    /// # Errors
    ///
    /// As [`cpueinsum_blas::BinaryPlan::execute_slices_accum`] for the vendor
    /// plan, and as [`cpueinsum::BinaryPlan::execute_slices_accum`] for
    /// tcontract's.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_slices_accum(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        a: (&[T], isize),
        b: (&[T], isize),
        beta: T,
        source: SliceAccumulationSource<'_, T>,
        d: (&mut [T], isize),
        work: &mut [T],
    ) -> Result<ExecutionRoute, BlasError> {
        match self {
            Self::Native(plan) => plan
                .execute_slices_accum(exec, alpha, a, b, beta, source, d)
                .map(ExecutionRoute::Native)
                .map_err(|source| BlasError::Native { source }),
            Self::Vendor(plan) => {
                plan.execute_slices_accum(exec, alpha, a, b, beta, source, d, work)
            }
        }
    }

    /// Overwrite fresh output, returning it initialized plus the route that ran.
    ///
    /// # Errors
    ///
    /// As [`Self::execute_slices_accum`], plus the fresh-output coverage checks
    /// of the plan that runs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_uninit_slices<'d>(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        a: (&[T], isize),
        b: (&[T], isize),
        beta: T,
        source: SliceAccumulationSource<'_, T>,
        d: (&'d mut [core::mem::MaybeUninit<T>], isize),
        work: &mut [T],
    ) -> Result<(&'d mut [T], ExecutionRoute), BlasError> {
        match self {
            Self::Native(plan) => plan
                .execute_uninit_slices(exec, alpha, a, b, beta, source, d)
                .map(|(initialized, route)| (initialized, ExecutionRoute::Native(route)))
                .map_err(|source| BlasError::Native { source }),
            Self::Vendor(plan) => {
                plan.execute_uninit_slices(exec, alpha, a, b, beta, source, d, work)
            }
        }
    }
}

/// Whether this product runs tcontract's plan instead of the vendor one.
fn takes_faer_plan(problem: &Problem) -> bool {
    problem.dtype() == DType::C64 && row_role(problem) <= THIN_ROW_ROLE && links_accelerate()
}

/// The product of the GEMM row role's extents, saturating.
fn row_role(problem: &Problem) -> usize {
    problem
        .roles()
        .m()
        .iter()
        .try_fold(1usize, |rows, axis| rows.checked_mul(axis.extent()))
        .unwrap_or(usize::MAX)
}

/// The degenerate kernel is Accelerate's, so the gate needs an Accelerate link.
fn links_accelerate() -> bool {
    cfg!(feature = "blas-accelerate")
}
