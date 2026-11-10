//! A borrowed-resources CPU engine and lane splitting (#2552 S2b).
//!
//! `BorrowedCpuEngine` wraps a caller's `&mut Exec` and `&mut BufferPool` and is an [`Engine`],
//! so a call written against `&mut dyn Engine` can run on resources the caller already holds.
//! It is a *carrier*: it implements the whole `Engine` surface, but every operation answers
//! `Unsupported`; only the linalg factorizations svd, svdvals, qr, eigh, eigvalsh and eigvals
//! run on it, through [`with_borrowed_cpu_engine`], which tenferro-linalg's lane dispatch calls.
//!
//! [`CpuEngine::with_lanes`] builds on it: it forks the engine's token with
//! `threadteam::Exec::fork_sized_slots` and hands each lane a `BorrowedCpuEngine` over the child
//! token and a caller-provided lane pool.

use cpueinsum::Exec;
use tenferro_tensor::backend::Engine;
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, ElementwiseReadOp, GatherConfig, NativeSessionRef,
    PadConfig, ScatterConfig, SliceConfig, TensorRead, TensorWrite,
};
use tenferro_tensor::{
    SessionCachedDot, TensorAnalytic, TensorBuffer, TensorDeviceTransfer, TensorDot,
    TensorElementwise, TensorFusion, TensorIndexing, TensorReduction, TensorStructural,
};

use crate::buffer_pool::BufferPool;
use crate::CpuEngine;

/// Native-engine marker for [`BorrowedCpuEngine`]; private to this crate.
struct BorrowedCpuMarker;

/// A CPU engine that borrows its execution bundle and buffer pool from the caller.
///
/// `'a` is the borrow of the bundle and pool, `'e` the bundle's own lifetime. The engine stores
/// the two `&mut` borrows and nothing else, so it cannot outlive either and cannot be kept past
/// the call that built it.
pub(crate) struct BorrowedCpuEngine<'a, 'e> {
    exec: &'a mut Exec<'e>,
    pool: &'a mut BufferPool,
}

impl<'a, 'e> BorrowedCpuEngine<'a, 'e> {
    /// Borrow `exec` and `pool` for as long as this engine lives.
    pub(crate) fn new(exec: &'a mut Exec<'e>, pool: &'a mut BufferPool) -> Self {
        Self { exec, pool }
    }
}

impl std::fmt::Debug for BorrowedCpuEngine<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BorrowedCpuEngine")
            .field("budget", &self.exec.budget())
            .field("pool", &self.pool.stats())
            .finish()
    }
}

/// Visit the bundle and pool behind a type-erased engine, if it is a lane engine.
///
/// An owner-scoped bridge owned by `tenferro-cpu` for `tenferro-linalg`'s lane dispatch: inside
/// [`CpuEngine::with_lanes`] the `&mut dyn Engine` a lane receives is such an engine, and this
/// function lends the lane's execution bundle and buffer pool to the callback. For any other
/// engine, including a [`CpuEngine`], it returns `None` without calling `f`.
///
/// The callback is higher-ranked over the bundle's lifetime, so it cannot observe or extend the
/// lifetimes the engine was built with; the erased pointer is only reconstructed for the visit,
/// while `engine` stays exclusively borrowed.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::linalg_interop::BufferPool;
/// use tenferro_cpu::{with_borrowed_cpu_engine, CpuEngine};
/// use tenferro_tensor::Engine;
///
/// let mut engine = CpuEngine::serial();
/// let mut pools = [BufferPool::new()];
/// let lane_budgets = engine.with_lanes(&[1], &mut pools, |_, lane| {
///     with_borrowed_cpu_engine(lane, |exec, _pool| exec.budget())
/// })?;
/// assert_eq!(lane_budgets, vec![Some(1)]);
///
/// // A plain engine is not a lane.
/// let erased: &mut dyn Engine = &mut engine;
/// assert!(with_borrowed_cpu_engine(erased, |_, _| ()).is_none());
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[doc(hidden)]
pub fn with_borrowed_cpu_engine<B, R>(
    engine: &mut B,
    f: impl FnOnce(&mut Exec<'_>, &mut BufferPool) -> R,
) -> Option<R>
where
    B: Engine + ?Sized,
{
    let data = engine
        .native_engine()?
        .into_marked_ptr::<BorrowedCpuMarker>()?;
    // SAFETY: only `BorrowedCpuEngine::native_engine` creates a token carrying the crate-private
    // `BorrowedCpuMarker`, and that token points at the live `BorrowedCpuEngine` (of some
    // lifetimes `'x`, `'y`) it was made from, so the marker proves the pointee type. Elided
    // lifetimes in the cast are unconstrained inference variables that are never exposed: `f` is
    // higher-ranked over the bundle lifetime (`Exec<'_>` in its signature), so it must work for
    // every lifetime and cannot name, store or extend the real `'y`; the returned `R` is chosen by
    // the caller before the borrow exists and so cannot hold a reference derived from it. The
    // token borrowed `*engine` exclusively and this function holds `engine: &mut B` for the whole
    // visit, so no other reference to the pointee exists while `this` is alive.
    let this = unsafe { data.cast::<BorrowedCpuEngine<'_, '_>>().as_mut() };
    Some(f(&mut *this.exec, &mut *this.pool))
}

fn refused<T>(op: &'static str) -> tenferro_tensor::Result<T> {
    Err(tenferro_tensor::Error::unsupported(
        op,
        "a lane of CpuEngine::with_lanes runs only the linalg factorizations \
         svd/svdvals/qr/eigh/eigvalsh/eigvals; this operation needs a CpuEngine",
    ))
}
impl SessionCachedDot for BorrowedCpuEngine<'_, '_> {}
impl TensorFusion for BorrowedCpuEngine<'_, '_> {}
impl TensorBuffer for BorrowedCpuEngine<'_, '_> {}

impl Engine for BorrowedCpuEngine<'_, '_> {
    fn native_engine(&mut self) -> Option<NativeSessionRef<'_>> {
        // SAFETY: `BorrowedCpuMarker` is private to this crate and this is the only place a token
        // carrying it is created; it always points to a `BorrowedCpuEngine`.
        Some(unsafe { NativeSessionRef::new::<BorrowedCpuMarker, _>(self) })
    }
}

impl TensorStructural for BorrowedCpuEngine<'_, '_> {
    fn transpose_read(
        &mut self,
        _a0: TensorRead<'_>,
        _a1: &[usize],
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("transpose_read")
    }

    fn reshape_read(
        &mut self,
        _a0: TensorRead<'_>,
        _a1: &[usize],
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("reshape_read")
    }

    fn broadcast_in_dim_read(
        &mut self,
        _a0: TensorRead<'_>,
        _a1: &[usize],
        _a2: &[usize],
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("broadcast_in_dim_read")
    }

    fn cast(
        &mut self,
        _a0: &tenferro_tensor::Tensor,
        _a1: DType,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("cast")
    }

    fn extract_diagonal(
        &mut self,
        _a0: &tenferro_tensor::Tensor,
        _a1: usize,
        _a2: usize,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("extract_diagonal")
    }

    fn embed_diagonal(
        &mut self,
        _a0: &tenferro_tensor::Tensor,
        _a1: usize,
        _a2: usize,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("embed_diagonal")
    }

    fn tril(
        &mut self,
        _a0: &tenferro_tensor::Tensor,
        _a1: i64,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("tril")
    }

    fn triu(
        &mut self,
        _a0: &tenferro_tensor::Tensor,
        _a1: i64,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("triu")
    }
}

impl TensorElementwise for BorrowedCpuEngine<'_, '_> {
    fn elementwise_read_into(
        &mut self,
        _a0: ElementwiseReadOp,
        _a1: &[TensorRead<'_>],
        _a2: TensorWrite<'_>,
    ) -> Result<(), tenferro_tensor::Error> {
        refused("elementwise_read_into")
    }

    fn add_read(
        &mut self,
        _a0: TensorRead<'_>,
        _a1: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("add_read")
    }

    fn sub_read(
        &mut self,
        _a0: TensorRead<'_>,
        _a1: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("sub_read")
    }

    fn mul_read(
        &mut self,
        _a0: TensorRead<'_>,
        _a1: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("mul_read")
    }

    fn neg_read(
        &mut self,
        _a0: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("neg_read")
    }

    fn conj_read(
        &mut self,
        _a0: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("conj_read")
    }

    fn div_read(
        &mut self,
        _a0: TensorRead<'_>,
        _a1: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("div_read")
    }

    fn abs_read(
        &mut self,
        _a0: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("abs_read")
    }

    fn sign_read(
        &mut self,
        _a0: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("sign_read")
    }

    fn maximum_read(
        &mut self,
        _a0: TensorRead<'_>,
        _a1: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("maximum_read")
    }

    fn minimum_read(
        &mut self,
        _a0: TensorRead<'_>,
        _a1: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("minimum_read")
    }

    fn compare_read(
        &mut self,
        _a0: TensorRead<'_>,
        _a1: TensorRead<'_>,
        _a2: &CompareDir,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("compare_read")
    }

    fn select_read(
        &mut self,
        _a0: TensorRead<'_>,
        _a1: TensorRead<'_>,
        _a2: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("select_read")
    }

    fn clamp_read(
        &mut self,
        _a0: TensorRead<'_>,
        _a1: TensorRead<'_>,
        _a2: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("clamp_read")
    }
}

impl TensorAnalytic for BorrowedCpuEngine<'_, '_> {
    fn exp_read(
        &mut self,
        _a0: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("exp_read")
    }

    fn log_read(
        &mut self,
        _a0: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("log_read")
    }

    fn sin_read(
        &mut self,
        _a0: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("sin_read")
    }

    fn cos_read(
        &mut self,
        _a0: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("cos_read")
    }

    fn tanh_read(
        &mut self,
        _a0: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("tanh_read")
    }

    fn sqrt_read(
        &mut self,
        _a0: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("sqrt_read")
    }

    fn rsqrt_read(
        &mut self,
        _a0: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("rsqrt_read")
    }

    fn pow_read(
        &mut self,
        _a0: TensorRead<'_>,
        _a1: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("pow_read")
    }

    fn expm1_read(
        &mut self,
        _a0: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("expm1_read")
    }

    fn log1p_read(
        &mut self,
        _a0: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("log1p_read")
    }

    fn erf_read(
        &mut self,
        _a0: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("erf_read")
    }
}

impl TensorReduction for BorrowedCpuEngine<'_, '_> {
    fn reduce_sum_read(
        &mut self,
        _a0: TensorRead<'_>,
        _a1: &[usize],
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("reduce_sum_read")
    }

    fn reduce_prod_read(
        &mut self,
        _a0: TensorRead<'_>,
        _a1: &[usize],
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("reduce_prod_read")
    }

    fn reduce_max_read(
        &mut self,
        _a0: TensorRead<'_>,
        _a1: &[usize],
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("reduce_max_read")
    }

    fn reduce_min_read(
        &mut self,
        _a0: TensorRead<'_>,
        _a1: &[usize],
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("reduce_min_read")
    }
}

impl TensorIndexing for BorrowedCpuEngine<'_, '_> {
    fn gather(
        &mut self,
        _a0: &tenferro_tensor::Tensor,
        _a1: &tenferro_tensor::Tensor,
        _a2: &GatherConfig,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("gather")
    }

    fn scatter(
        &mut self,
        _a0: &tenferro_tensor::Tensor,
        _a1: &tenferro_tensor::Tensor,
        _a2: &tenferro_tensor::Tensor,
        _a3: &ScatterConfig,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("scatter")
    }

    fn slice(
        &mut self,
        _a0: &tenferro_tensor::Tensor,
        _a1: &SliceConfig,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("slice")
    }

    fn dynamic_slice(
        &mut self,
        _a0: &tenferro_tensor::Tensor,
        _a1: &tenferro_tensor::Tensor,
        _a2: &[usize],
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("dynamic_slice")
    }

    fn dynamic_update_slice(
        &mut self,
        _a0: &tenferro_tensor::Tensor,
        _a1: &tenferro_tensor::Tensor,
        _a2: &tenferro_tensor::Tensor,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("dynamic_update_slice")
    }

    fn pad(
        &mut self,
        _a0: &tenferro_tensor::Tensor,
        _a1: &PadConfig,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("pad")
    }

    fn concatenate(
        &mut self,
        _a0: &[&tenferro_tensor::Tensor],
        _a1: usize,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("concatenate")
    }

    fn reverse(
        &mut self,
        _a0: &tenferro_tensor::Tensor,
        _a1: &[usize],
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("reverse")
    }
}

impl TensorDot for BorrowedCpuEngine<'_, '_> {
    fn dot_general_read(
        &mut self,
        _a0: TensorRead<'_>,
        _a1: TensorRead<'_>,
        _a2: &DotGeneralConfig,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("dot_general_read")
    }
}

impl TensorDeviceTransfer for BorrowedCpuEngine<'_, '_> {
    fn download_to_host(
        &mut self,
        _a0: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("download_to_host")
    }

    fn upload_host_tensor(
        &mut self,
        _a0: TensorRead<'_>,
    ) -> Result<tenferro_tensor::Tensor, tenferro_tensor::Error> {
        refused("upload_host_tensor")
    }
}

/// One lane's slot while the token is forked: its pool and, afterwards, its result.
struct Lane<'p, R> {
    pool: &'p mut BufferPool,
    out: Option<R>,
}

impl CpuEngine {
    /// Run `f` once per lane, concurrently, each lane an engine over its own share of this
    /// engine's thread budget and its own caller-owned buffer pool.
    ///
    /// A *lane* is one concurrent slice of the engine: `shares[i]` threads of the thread budget
    /// (the shares must be non-zero and sum to at most [`CpuEngine::num_threads`]) and the pool
    /// `pools[i]`. `f(i, lane)` runs on lane `i` and receives the lane as `&mut dyn Engine`;
    /// results come back in lane order. The split uses the engine's `threadteam` fork only, so
    /// lanes never oversubscribe the budget and never touch another pool.
    ///
    /// The caller owns the pools and should keep them across calls: a lane draws every output,
    /// workspace and gather buffer from its pool, so reusing the same pool is what makes
    /// repeated calls allocation-free in steady state. A pool is never shared between lanes.
    ///
    /// Only the linalg factorizations svd, svdvals, qr, eigh, eigvalsh and eigvals run in a lane
    /// (through the ordinary `tenferro-linalg` methods, for example `svd_read`); every other
    /// operation on a lane engine returns [`tenferro_tensor::Error::Unsupported`]. A negative-stride
    /// input is gathered once into a buffer from the lane's pool; the steady-state zero-allocation
    /// property is only promised for inputs without negative strides.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] when `shares` and `pools` differ in length, a share
    /// is zero, or the shares sum to more than the thread budget; no work runs and the engine
    /// stays usable. Returns [`crate::Error::RuntimeState`] when this engine is broken.
    ///
    /// # Panics
    ///
    /// A panic in a lane is re-raised after every lane has finished, and the engine becomes
    /// broken, exactly as for a panic inside [`CpuEngine::with_exec`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::linalg_interop::BufferPool;
    /// use tenferro_cpu::{CpuEngine, Pool};
    ///
    /// let pool = Pool::builder().threads(2).build()?;
    /// let mut engine = CpuEngine::new(pool);
    /// // The caller owns one pool per lane and reuses them across calls.
    /// let mut pools = [BufferPool::new(), BufferPool::new()];
    /// let lane_ids = engine.with_lanes(&[1, 1], &mut pools, |index, _lane| index)?;
    /// assert_eq!(lane_ids, vec![0, 1]);
    ///
    /// // An invalid split is a validation error and leaves the engine usable.
    /// assert!(engine.with_lanes(&[2, 2], &mut pools, |index, _lane| index).is_err());
    /// assert_eq!(engine.num_threads(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn with_lanes<R: Send>(
        &mut self,
        shares: &[usize],
        pools: &mut [BufferPool],
        f: impl Fn(usize, &mut dyn Engine) -> R + Sync,
    ) -> crate::Result<Vec<R>> {
        const OP: &str = "CpuEngine::with_lanes";
        if shares.len() != pools.len() {
            return Err(tenferro_tensor::Error::invalid_argument(
                OP,
                "pools",
                format!(
                    "need one pool per lane: {} shares but {} pools",
                    shares.len(),
                    pools.len()
                ),
            ));
        }
        if shares.contains(&0) {
            return Err(tenferro_tensor::Error::invalid_argument(
                OP,
                "shares",
                format!("every lane needs at least one thread, got {shares:?}"),
            ));
        }
        let budget = self.num_threads();
        let total = shares
            .iter()
            .try_fold(0_usize, |sum, &share| sum.checked_add(share));
        if total.is_none_or(|total| total > budget) {
            return Err(tenferro_tensor::Error::invalid_argument(
                OP,
                "shares",
                format!("the shares {shares:?} exceed the engine's thread budget of {budget}"),
            ));
        }
        self.with_exec(|exec| {
            let workspace = exec.workspace();
            let policy = *exec.batch_policy();
            let mut lanes: Vec<Lane<'_, R>> = pools
                .iter_mut()
                .map(|pool| Lane { pool, out: None })
                .collect();
            exec.token()
                .fork_sized_slots(shares, &mut lanes, |index, lane, child| {
                    let mut bundle = Exec::from_token(child, workspace, policy);
                    let mut engine = BorrowedCpuEngine::new(&mut bundle, &mut *lane.pool);
                    lane.out = Some(f(index, &mut engine));
                })
                .map_err(|error| {
                    tenferro_tensor::Error::invalid_argument(OP, "shares", error.to_string())
                })?;
            lanes
                .into_iter()
                .map(|lane| {
                    lane.out.ok_or_else(|| {
                        tenferro_tensor::Error::Internal("a lane produced no result".into())
                    })
                })
                .collect()
        })?
    }
}

#[cfg(test)]
mod tests;
