//! A borrowed-resources CPU engine and lane splitting (prototype for tenferro-rs#2552, P7).
//!
//! [`BorrowedCpuEngine`] wraps a caller's `&mut Exec` and `&mut BufferPool` and is an
//! [`Engine`], so a call written against `&mut dyn Engine` can run on resources the caller
//! already holds (the case the `*_with_exec` entries of tenferro-linalg exist for, #2462).
//!
//! It is a *carrier*: it implements the whole `Engine` surface, but every operation family other
//! than the borrowed-resource hook ([`with_borrowed_cpu_engine`]) answers `Unsupported`. See
//! `ext/api-grammar-proof/findings/p7.md` for why that is the measured cost of this design and
//! what the full-engine alternative would need.
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
/// the call that built it (the borrow checker rejects both; see the compile-fail tests in the
/// proof crate).
#[doc(hidden)]
pub struct BorrowedCpuEngine<'a, 'e> {
    exec: &'a mut Exec<'e>,
    pool: &'a mut BufferPool,
}

impl<'a, 'e> BorrowedCpuEngine<'a, 'e> {
    /// Borrow `exec` and `pool` for as long as this engine lives.
    pub fn new(exec: &'a mut Exec<'e>, pool: &'a mut BufferPool) -> Self {
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

/// Visit the bundle and pool behind a type-erased engine, if it is a [`BorrowedCpuEngine`].
///
/// The callback is higher-ranked over the bundle's lifetime, so it cannot observe or extend the
/// lifetimes the engine was built with; the erased pointer is only reconstructed for the visit,
/// while `engine` stays exclusively borrowed.
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
    // SAFETY: only `BorrowedCpuEngine::native_engine` creates a token with the crate-private
    // `BorrowedCpuMarker`, and it points at a live `BorrowedCpuEngine` (of some lifetimes
    // `'x`, `'y`). The token borrowed `*engine` exclusively and this function holds `engine`
    // for the whole visit. `f` is higher-ranked over the bundle lifetime, so by parametricity it
    // works for the real `'y` without being able to name or lengthen it.
    let this = unsafe { data.cast::<BorrowedCpuEngine<'_, '_>>().as_mut() };
    Some(f(&mut *this.exec, &mut *this.pool))
}

fn refused<T>(op: &'static str) -> tenferro_tensor::Result<T> {
    Err(tenferro_tensor::Error::unsupported(
        op,
        "BorrowedCpuEngine carries a caller's execution bundle and buffer pool for the \
         borrowed factorization entries only; this operation needs a CpuEngine",
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

struct Lane<'p, R> {
    pool: &'p mut BufferPool,
    out: Option<R>,
}

impl CpuEngine {
    /// Run `f` once per lane, concurrently, each lane an engine over its own share of this
    /// engine's thread budget and its own caller-provided buffer pool.
    ///
    /// Lane `i` gets `shares[i]` slots of the budget (`threadteam::Exec::fork_sized_slots`: the
    /// shares must be non-zero and sum to at most the budget) and `pools[i]`. Results come back in
    /// lane order. The lane engine is a [`BorrowedCpuEngine`]: borrowed-resource factorizations
    /// run on it; other operations are `Unsupported`. A panic in a lane propagates after every
    /// lane has finished and marks this engine broken, like a panic in `with_exec`.
    ///
    /// # Errors
    ///
    /// `Error::RuntimeState` when this engine is broken; `Error::Validation` when `shares` and
    /// `pools` differ in length, a share is zero, or the shares exceed the budget.
    #[doc(hidden)]
    pub fn with_lanes<R: Send>(
        &mut self,
        shares: &[usize],
        pools: &mut [BufferPool],
        f: impl Fn(usize, &mut dyn Engine) -> R + Sync,
    ) -> crate::Result<Vec<R>> {
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
                    tenferro_tensor::Error::invalid_argument(
                        "CpuEngine::with_lanes",
                        "shares",
                        error.to_string(),
                    )
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
