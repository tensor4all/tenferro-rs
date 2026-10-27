//! Tensor-boundary adapters; numerical plans and scheduling belong to cpueinsum.

#[cfg(test)]
mod tests;
mod workspaces;
pub use workspaces::ContractionWorkspaces;

use crate::buffer_pool::{BufferPool, PoolScalar};
use cpueinsum::tcontract::{
    api::{DotGeneral, LayoutSpec, OperandSpec, Problem},
    PlanConfig,
};
use cpueinsum::{ArenaProvider, Exec, SliceAccumulationSource};
use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tenferro_cpu_basic::PooledUninitOutput;
use tenferro_tensor::{
    ContractionScalar, DotGeneralAccumulation, DotGeneralConfig, Tensor, TensorRead, TensorView,
    TensorViewMut, TensorWrite, TypedTensorView, TypedTensorViewMut,
};
use tinyvec::TinyVec;

#[cfg(feature = "native")]
use cpueinsum::BinaryPlan;
#[cfg(feature = "blas")]
use cpueinsum_blas::BinaryPlan;

/// The numerical route a binary contraction took, for allocation/route tests.
///
/// Only the vendor/native distinction is exposed: the native alternatives keep
/// their own finer [`cpueinsum::ExecutionRoute`] variants, which are not a
/// stable test contract.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ContractionRoute {
    #[cfg_attr(not(feature = "blas"), allow(dead_code))]
    Vendor,
    Native,
}

#[cfg(test)]
thread_local! {
    static LAST_CONTRACTION_ROUTE: std::cell::Cell<Option<ContractionRoute>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn take_last_contraction_route() -> Option<ContractionRoute> {
    LAST_CONTRACTION_ROUTE.with(|last| last.replace(None))
}

#[cfg(all(test, feature = "blas"))]
fn record_contraction_route(route: cpueinsum_blas::ExecutionRoute) {
    use cpueinsum_blas::ExecutionRoute;
    let route = match route {
        ExecutionRoute::Vendor => ContractionRoute::Vendor,
        ExecutionRoute::Native(_) => ContractionRoute::Native,
    };
    LAST_CONTRACTION_ROUTE.with(|last| last.set(Some(route)));
}

#[cfg(all(test, not(feature = "blas")))]
fn record_contraction_route(_route: cpueinsum::ExecutionRoute) {
    LAST_CONTRACTION_ROUTE.with(|last| last.set(Some(ContractionRoute::Native)));
}

#[cfg(feature = "native")]
trait Scalar: PoolScalar + cpueinsum::Scalar {}
#[cfg(feature = "native")]
impl<T: PoolScalar + cpueinsum::Scalar> Scalar for T {}
#[cfg(feature = "blas")]
trait Scalar: PoolScalar + cpueinsum_blas::BlasScalar {}
#[cfg(feature = "blas")]
impl<T: PoolScalar + cpueinsum_blas::BlasScalar> Scalar for T {}

/// Translate a lower numerical failure, keeping a typed route decline distinct.
///
/// A lower `Unsupported`/declined step is the lower library refusing a route it
/// owns, not a tenferro backend failure, so it maps to `Unsupported` while the
/// lower typed error is preserved as the source (not formatted into a string).
pub(crate) trait LowerContractionError {
    fn into_crate_error(self, op: &'static str) -> crate::Error;
}

/// A bare tcontract API failure already carries the same decline classification.
impl LowerContractionError for cpueinsum::tcontract::Error {
    fn into_crate_error(self, op: &'static str) -> crate::Error {
        if self.is_unsupported() {
            return tenferro_tensor::Error::extension(
                op,
                "contraction",
                tenferro_tensor::ErrorKind::Unsupported,
                self,
            );
        }
        crate::Error::backend_source(op, self)
    }
}

impl LowerContractionError for cpueinsum::Error {
    fn into_crate_error(self, op: &'static str) -> crate::Error {
        // A caller-owned arena shortage is a runtime-state failure and keeps its
        // typed `cpueinsum` source rather than becoming a text-only backend error.
        if let cpueinsum::Error::Workspace { .. } = &self {
            return crate::Error::runtime_state_source(op, self);
        }
        let declined = matches!(
            &self,
            cpueinsum::Error::Contract { source, .. } if source.is_unsupported()
        );
        if declined {
            return tenferro_tensor::Error::extension(
                op,
                "contraction",
                tenferro_tensor::ErrorKind::Unsupported,
                self,
            );
        }
        crate::Error::backend_source(op, self)
    }
}

#[cfg(feature = "blas")]
impl LowerContractionError for cpueinsum_blas::BlasError {
    fn into_crate_error(self, op: &'static str) -> crate::Error {
        // A short work buffer is a runtime-state failure with a typed source.
        if let cpueinsum_blas::BlasError::Work { .. } = &self {
            return crate::Error::runtime_state_source(op, self);
        }
        let declined = match &self {
            cpueinsum_blas::BlasError::Prepare { source }
            | cpueinsum_blas::BlasError::Native { source } => matches!(
                source,
                cpueinsum::Error::Contract { source, .. } if source.is_unsupported()
            ),
            _ => false,
        };
        if declined {
            return tenferro_tensor::Error::extension(
                op,
                "contraction",
                tenferro_tensor::ErrorKind::Unsupported,
                self,
            );
        }
        crate::Error::backend_source(op, self)
    }
}

/// How an engine places its work.
enum Placement {
    /// No pool: every operation runs on the calling thread.
    Serial,
    /// A one-worker pool: tokens have a budget of one and never dispatch.
    Single(threadteam::Pool),
    /// A pool of two or more workers, moved into a [`threadteam::Team`]: the
    /// members stay reserved for the engine's lifetime, so every parallel
    /// region (the strided kernels, tcontract batches, faer's `Par::Team`) is a
    /// team section and no work goes through Rayon.
    Team(threadteam::Team),
}

// Exactly one retained workspace per engine, kept together with the pool (or
// team) it serves. The engine starts no task of its own; placement is the
// `threadteam::Exec` token it mints for each operation.
//
// An engine is single-threaded by contract: operations take `&mut CpuEngine`,
// and the team has one leader at a time. Two threads that share an engine
// through their own lock are serialized by it; one that reaches the team while
// another leads waits for that leader (`Team::exec`), it never gets a second
// team.
pub(crate) struct ExecutionResources {
    placement: Placement,
    /// The pool's dispatch costs (host default without a pool), kept for the
    /// pre-entry gates that decide before a token is minted.
    costs: threadteam::DispatchCosts,
    /// Scratch for every token of this engine, serial or team.
    workspace: ArenaProvider,
    /// The thread that minted the latest team token. A token that is still
    /// alive when the same thread asks again is a re-entrant call, which gets
    /// a serial token instead of waiting for itself.
    last_leader: std::sync::Mutex<Option<std::thread::ThreadId>>,
    pub(crate) nary: ContractionWorkspaces,
    /// Reads canonicalized out of a view through `to_contiguous_read`, which is
    /// the operand staging copy a read hook must not make. Counted so a
    /// regression test can assert that a read hook stages nothing.
    read_stagings: AtomicU64,
    /// Ledger roots of the tokens minted since the last
    /// [`ExecutionResources::take_ledger_roots`] (test support).
    #[cfg(feature = "ledger")]
    ledger_roots: std::sync::Mutex<Vec<u64>>,
}
impl fmt::Debug for ExecutionResources {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContractionResources")
            .field("threads", &self.thread_budget())
            .field("workspace", &self.workspace().stats())
            .finish()
    }
}
impl ExecutionResources {
    /// Take over `pool`, or run serially without one.
    ///
    /// A pool of two or more workers becomes a team. When its members cannot
    /// all be reserved the engine runs serially (never through Rayon): the
    /// failure is a construction-time refusal, so the pool is dropped and
    /// [`ExecutionResources::thread_budget`] reports one.
    pub(crate) fn new(pool: Option<threadteam::Pool>) -> Self {
        let costs = pool
            .as_ref()
            .map_or_else(threadteam::DispatchCosts::host_default, |pool| {
                pool.dispatch_costs()
            });
        let placement = match pool {
            None => Placement::Serial,
            Some(pool) if pool.width().get() == 1 => Placement::Single(pool),
            Some(pool) => Self::reserve_team(pool),
        };
        Self {
            costs,
            placement,
            workspace: ArenaProvider::new(),
            last_leader: std::sync::Mutex::new(None),
            nary: ContractionWorkspaces::default(),
            read_stagings: AtomicU64::new(0),
            #[cfg(feature = "ledger")]
            ledger_roots: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Reserve every member of `pool` for the engine. Workers that were
    /// started a moment ago may not have reached their idle state yet (a busy
    /// host, several engines built at once), and a refused reservation hands the
    /// pool back, so retry briefly before giving up and running serially.
    fn reserve_team(mut pool: threadteam::Pool) -> Placement {
        const ATTEMPTS: u32 = 50;
        const PAUSE: std::time::Duration = std::time::Duration::from_millis(10);
        let width = pool.width().get();
        for attempt in 0..ATTEMPTS {
            match pool.into_team(width) {
                Ok(team) => return Placement::Team(team),
                Err(refused) => pool = refused.into_pool(),
            }
            if attempt + 1 < ATTEMPTS {
                std::thread::sleep(PAUSE);
            }
        }
        Placement::Serial
    }

    pub(crate) fn count_read_staging(&self) {
        self.read_stagings.fetch_add(1, Ordering::Relaxed);
    }

    /// Views canonicalized by `to_contiguous_read` since construction.
    #[cfg(test)]
    pub(crate) fn read_stagings(&self) -> u64 {
        self.read_stagings.load(Ordering::Relaxed)
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.workspace().retained_bytes()
    }
    pub(crate) fn trim(&self) {
        use cpueinsum::WorkspaceProvider;
        self.workspace().trim();
    }
    pub(crate) fn retention_guard(&self, max_bytes: usize) -> WorkspaceRetention<'_> {
        WorkspaceRetention {
            arena: self.workspace(),
            max_bytes,
        }
    }
    /// Number of participants of the engine's team (its pool width), or one.
    pub(crate) fn thread_budget(&self) -> std::num::NonZeroUsize {
        match &self.placement {
            Placement::Serial | Placement::Single(_) => std::num::NonZeroUsize::MIN,
            Placement::Team(team) => team.size(),
        }
    }
    /// Dispatch costs of the owned pool, for decisions made before a token exists.
    pub(crate) fn dispatch_costs(&self) -> &threadteam::DispatchCosts {
        &self.costs
    }
    fn workspace(&self) -> &ArenaProvider {
        &self.workspace
    }
    /// Mint the one provider bundle of an operation: every member of the team
    /// as the budget, the workspace, and `policy`. A serial or one-worker
    /// engine mints a budget-one token and never dispatches.
    ///
    /// A team has one leader at a time. A thread that asks while another thread
    /// leads waits for it (an engine is not meant to be shared, and a lock
    /// around it already serializes); a thread that asks while it leads itself
    /// (a re-entrant call from inside an operation) gets a serial token with
    /// the same workspace, so faer runs `Par::Seq` there and nothing panics.
    pub(crate) fn exec(&self, policy: crate::CpuBatchPolicy) -> Exec<'_> {
        #[cfg_attr(not(feature = "ledger"), allow(unused_mut))]
        let mut exec = match &self.placement {
            Placement::Serial => Exec::serial_with_workspace(&self.workspace),
            Placement::Single(pool) => Exec::from_token(
                // INVARIANT: a width-one budget is non-zero.
                threadteam::Exec::new(pool, 1).unwrap_or_else(|_| threadteam::Exec::serial()),
                Some(&self.workspace),
                cpueinsum::BatchPolicy::default(),
            ),
            Placement::Team(team) => match self.lead(team) {
                Some(token) => Exec::from_token(
                    token,
                    Some(&self.workspace),
                    cpueinsum::BatchPolicy::default(),
                ),
                None => Exec::serial_with_workspace(&self.workspace),
            },
        };
        #[cfg(feature = "ledger")]
        if let Some(root) = exec.token().ledger_root() {
            self.ledger_roots
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(root);
        }
        exec.with_batch_policy(policy)
    }
    /// The team token, or `None` when this thread leads the team already.
    fn lead<'a>(&'a self, team: &'a threadteam::Team) -> Option<threadteam::Exec<'a>> {
        let me = std::thread::current().id();
        let mut last = self
            .last_leader
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let token = match team.try_exec(team.size().get()) {
            Ok(token) => token,
            // The latest leader is the holder of the live token: if that is this
            // thread, waiting would wait for itself.
            Err(_busy) if *last == Some(me) => return None,
            Err(_busy) => {
                // Another thread leads: wait for it, without holding the lock.
                drop(last);
                let token = team.exec(team.size().get());
                *self
                    .last_leader
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(me);
                return Some(token);
            }
        };
        *last = Some(me);
        Some(token)
    }
    /// Ledger roots of every token minted since the previous call.
    #[cfg(feature = "ledger")]
    pub(crate) fn take_ledger_roots(&self) -> Vec<u64> {
        std::mem::take(
            &mut *self
                .ledger_roots
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }
}

pub(crate) struct WorkspaceRetention<'a> {
    arena: &'a ArenaProvider,
    max_bytes: usize,
}
impl Drop for WorkspaceRetention<'_> {
    fn drop(&mut self) {
        use cpueinsum::WorkspaceProvider;
        if self.max_bytes != usize::MAX && self.arena.retained_bytes() > self.max_bytes {
            self.arena.trim();
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Layout {
    dims: TinyVec<[usize; 8]>,
    strides: TinyVec<[isize; 8]>,
    offset: isize,
}
impl Layout {
    fn new(dims: &[usize], strides: &[isize], offset: isize) -> Self {
        Self {
            dims: dims.into(),
            strides: strides.into(),
            offset,
        }
    }
    fn spec(&self) -> crate::Result<OperandSpec> {
        LayoutSpec::new(&self.dims, &self.strides, self.offset)
            .map(OperandSpec::new)
            .map_err(|error| error.into_crate_error("dot_general"))
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Key {
    dtype: cpueinsum::tcontract::api::DType,
    lhs: Layout,
    rhs: Layout,
    out: Layout,
    lhs_conj: bool,
    rhs_conj: bool,
}
#[cfg(any(feature = "native", feature = "blas"))]
struct CachedBinary {
    key: Key,
    config: DotGeneralConfig,
    plan: Arc<dyn Any + Send + Sync>,
}
#[derive(Default)]
#[cfg(any(feature = "native", feature = "blas"))]
pub(crate) struct BinaryCache {
    slots: Vec<Option<CachedBinary>>,
    hits: u64,
    misses: u64,
    evictions: u64,
    clears: u64,
}
#[cfg(any(feature = "native", feature = "blas"))]
impl fmt::Debug for BinaryCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BinaryCache")
            .field("slots", &self.slots.len())
            .finish()
    }
}
#[cfg(any(feature = "native", feature = "blas"))]
impl BinaryCache {
    pub(crate) fn clear(&mut self) {
        self.slots.clear();
        self.slots.shrink_to_fit();
        self.clears = self.clears.saturating_add(1);
    }
    pub(crate) fn stats(&self) -> tenferro_tensor::CacheStats {
        let mut stats = tenferro_tensor::CacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            clears: self.clears,
            retained_bytes: self.slots.capacity() * std::mem::size_of::<Option<CachedBinary>>(),
            ..Default::default()
        };
        for entry in self.slots.iter().flatten() {
            stats.entries += 1;
            // Known owned descriptor payload and opaque lower plan headers.
            // Lower-private metadata heaps need a lower-library footprint report.
            stats.retained_bytes = stats
                .retained_bytes
                .saturating_add(std::mem::size_of_val(entry.plan.as_ref()));
            for layout in [&entry.key.lhs, &entry.key.rhs, &entry.key.out] {
                if layout.dims.is_heap() {
                    stats.retained_bytes = stats
                        .retained_bytes
                        .saturating_add(layout.dims.capacity() * std::mem::size_of::<usize>());
                }
                if layout.strides.is_heap() {
                    stats.retained_bytes = stats
                        .retained_bytes
                        .saturating_add(layout.strides.capacity() * std::mem::size_of::<isize>());
                }
            }
            for axes in [
                &entry.config.lhs_contracting_dims,
                &entry.config.rhs_contracting_dims,
                &entry.config.lhs_batch_dims,
                &entry.config.rhs_batch_dims,
            ] {
                if axes.is_heap() {
                    stats.retained_bytes = stats
                        .retained_bytes
                        .saturating_add(axes.capacity() * std::mem::size_of::<usize>());
                }
            }
        }
        stats
    }
    pub(crate) fn set_capacity(&mut self, capacity: usize) {
        self.evictions = self.evictions.saturating_add(
            self.slots
                .iter()
                .skip(capacity)
                .filter(|slot| slot.is_some())
                .count() as u64,
        );
        self.slots.truncate(capacity);
        self.slots.shrink_to(capacity);
    }
    /// The plan for `key`, cached in `slot` when the slot is within `capacity`.
    ///
    /// The cache only hands out shared, immutable plans: the caller runs the
    /// returned plan after releasing whatever lock guards this cache, so two
    /// engines sharing one plan cache never wait for each other's numerical
    /// work.
    fn plan<T: Scalar>(
        &mut self,
        slot: Option<usize>,
        capacity: usize,
        key: Key,
        config: &DotGeneralConfig,
    ) -> crate::Result<Arc<BinaryPlan<T>>> {
        let prepare = || {
            let lhs = if key.lhs_conj {
                key.lhs.spec()?.conj()
            } else {
                key.lhs.spec()?
            };
            let rhs = if key.rhs_conj {
                key.rhs.spec()?.conj()
            } else {
                key.rhs.spec()?
            };
            let dot = DotGeneral::new(
                &config.lhs_contracting_dims,
                &config.rhs_contracting_dims,
                &config.lhs_batch_dims,
                &config.rhs_batch_dims,
            );
            let problem = Problem::from_dot_general(key.dtype, lhs, rhs, key.out.spec()?, &dot)
                .map_err(|error| error.into_crate_error("dot_general"))?;
            BinaryPlan::<T>::new(&problem, &PlanConfig::default())
                .map_err(|error| error.into_crate_error("dot_general"))
        };
        let Some(slot) = slot.filter(|&slot| slot < capacity) else {
            return Ok(Arc::new(prepare()?));
        };
        let matches = self
            .slots
            .get(slot)
            .and_then(Option::as_ref)
            .is_some_and(|entry| entry.key == key && entry.config == *config);
        if matches {
            self.hits = self.hits.saturating_add(1);
        } else {
            self.misses = self.misses.saturating_add(1);
            let plan = prepare()?;
            if self.slots.len() <= slot {
                self.slots
                    .try_reserve(slot + 1 - self.slots.len())
                    .map_err(|error| crate::Error::backend_source("dot_general", error))?;
                self.slots.resize_with(slot + 1, || None);
            }
            self.slots[slot] = Some(CachedBinary {
                key,
                config: config.clone(),
                plan: Arc::new(plan),
            });
        }
        let plan = self.slots[slot]
            .as_ref()
            .map(|entry| Arc::clone(&entry.plan))
            .ok_or_else(|| {
                crate::Error::runtime_state("dot_general", "cached numerical plan slot is empty")
            })?;
        plan.downcast::<BinaryPlan<T>>().map_err(|_| {
            crate::Error::runtime_state(
                "dot_general",
                "cached numerical plan has incompatible scalar type",
            )
        })
    }
}

#[cfg(any(feature = "native", feature = "blas"))]
fn key<T: Scalar>(
    lhs: &TypedTensorView<'_, T>,
    rhs: &TypedTensorView<'_, T>,
    out: Layout,
    accumulation: DotGeneralAccumulation,
) -> Key {
    Key {
        dtype: <T as cpueinsum::tcontract::api::Scalar>::STORAGE,
        lhs: Layout::new(lhs.shape(), lhs.strides(), lhs.offset()),
        rhs: Layout::new(rhs.shape(), rhs.strides(), rhs.offset()),
        out,
        lhs_conj: accumulation.lhs_conj,
        rhs_conj: accumulation.rhs_conj,
    }
}

#[cfg(any(feature = "native", feature = "blas"))]
#[allow(clippy::too_many_arguments)]
fn initialized<T: Scalar>(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    plans: &crate::CpuPlanCache,
    slot: Option<usize>,
    lhs: TypedTensorView<'_, T>,
    rhs: TypedTensorView<'_, T>,
    mut out: TypedTensorViewMut<'_, T>,
    config: &DotGeneralConfig,
    accumulation: DotGeneralAccumulation,
    alpha: T,
    beta: T,
) -> crate::Result<()> {
    let key = key(
        &lhs,
        &rhs,
        Layout::new(out.shape(), out.strides(), out.offset()),
        accumulation,
    );
    let output_offset = out.offset();
    let a = lhs.host_storage()?;
    let b = rhs.host_storage()?;
    let d = out.host_storage_mut()?;
    let plan = plans.binary_plan::<T>(slot, key, config)?;
    {
        {
            #[cfg(feature = "blas")]
            let mut work = T::pool_acquire_zeroed(buffers, plan.work_len());
            #[cfg(feature = "native")]
            let _ = buffers;
            let result = plan.execute_slices_accum(
                exec,
                alpha,
                (a, lhs.offset()),
                (b, rhs.offset()),
                beta,
                SliceAccumulationSource::Output,
                (d, output_offset),
                #[cfg(feature = "blas")]
                &mut work,
            );
            #[cfg(test)]
            if let Ok(route) = &result {
                record_contraction_route(*route);
            }
            let result = result
                .map(|_| ())
                .map_err(|error| error.into_crate_error("dot_general"));
            #[cfg(feature = "blas")]
            T::pool_release(buffers, work);
            result
        }
    }
}

#[cfg(any(feature = "native", feature = "blas"))]
#[allow(clippy::too_many_arguments)]
fn fresh<T: Scalar>(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    plans: &crate::CpuPlanCache,
    slot: Option<usize>,
    lhs: TypedTensorView<'_, T>,
    rhs: TypedTensorView<'_, T>,
    shape: tenferro_tensor::ShapeVec,
    config: &DotGeneralConfig,
    accumulation: DotGeneralAccumulation,
    alpha: T,
    beta: T,
) -> crate::Result<Tensor> {
    let strides = tenferro_tensor::col_major_strides(&shape)?;
    let key = key(&lhs, &rhs, Layout::new(&shape, &strides, 0), accumulation);
    let a = lhs.host_storage()?;
    let b = rhs.host_storage()?;
    let plan = plans.binary_plan::<T>(slot, key, config)?;
    {
        {
            #[cfg(feature = "blas")]
            let mut work = T::pool_acquire_zeroed(buffers, plan.work_len());
            let mut output = PooledUninitOutput::<T>::new(buffers, shape)?;
            let result = plan.execute_uninit_slices(
                exec,
                alpha,
                (a, lhs.offset()),
                (b, rhs.offset()),
                beta,
                SliceAccumulationSource::Absent,
                (output.as_uninit_slice_mut(), 0),
                #[cfg(feature = "blas")]
                &mut work,
            );
            #[cfg(test)]
            if let Ok((_, route)) = &result {
                record_contraction_route(*route);
            }
            let result = result
                .map(|_| ())
                .map_err(|error| error.into_crate_error("dot_general"));
            #[cfg(feature = "blas")]
            T::pool_release(buffers, work);
            result?;
            // SAFETY: BinaryPlan succeeded on MaybeUninit storage and proves complete
            // physical coverage. No initialized output reference existed before it.
            unsafe { output.assume_init_recycled() }.map(Tensor::from_typed::<T>)
        }
    }
}

#[cfg(any(feature = "native", feature = "blas"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn dot_fresh(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    plans: &crate::CpuPlanCache,
    slot: Option<usize>,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
    config: &DotGeneralConfig,
    accumulation: DotGeneralAccumulation,
    shape: tenferro_tensor::ShapeVec,
) -> crate::Result<Tensor> {
    crate::dot_runtime::validate_dot_operands(&lhs, &rhs, config, accumulation)?;
    macro_rules! dispatch {
        ($($variant:ident),*) => { match (lhs.tensor_view(), rhs.tensor_view(), accumulation.alpha, accumulation.beta) {
            $((TensorView::$variant(a), TensorView::$variant(b), ContractionScalar::$variant(alpha), ContractionScalar::$variant(beta)) =>
                fresh(exec, buffers, plans, slot, a, b, shape, config, accumulation, alpha, beta),)*
            _ => Err(crate::Error::runtime_state("dot_general", "operands and accumulation scalars must have matching supported dtypes")),
        } };
    }
    dispatch!(F32, F64, C32, C64)
}

fn writable_view<'a>(out: TensorWrite<'a>, op: &'static str) -> crate::Result<TensorViewMut<'a>> {
    Ok(match out {
        TensorWrite::View(view) => view,
        TensorWrite::Tensor(tensor) => {
            macro_rules! view { ($($variant:ident => $ty:ty),*) => { match tensor.dtype() {
                $(tenferro_tensor::DType::$variant => TensorViewMut::$variant(tensor.as_typed_mut::<$ty>()
                    .ok_or_else(|| crate::Error::runtime_state(op, "incompatible typed output"))?.as_view_mut()),)*
                dtype => return Err(crate::Error::unsupported_dtype(op, dtype, crate::cpu_contraction_unsupported_dtype_message(dtype))),
            } }; }
            view!(F32 => f32, F64 => f64, C32 => num_complex::Complex32, C64 => num_complex::Complex64)
        }
    })
}

#[cfg(any(feature = "native", feature = "blas"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn dot_into(
    exec: &mut Exec<'_>,
    buffers: &mut BufferPool,
    plans: &crate::CpuPlanCache,
    slot: Option<usize>,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
    config: &DotGeneralConfig,
    accumulation: DotGeneralAccumulation,
    out: TensorWrite<'_>,
) -> crate::Result<()> {
    crate::dot_runtime::validate_dot_general(&lhs, &rhs, &out, config, accumulation)?;
    if lhs.dtype() != rhs.dtype() {
        return Err(crate::Error::dtype_mismatch(
            "dot_general",
            lhs.dtype(),
            rhs.dtype(),
        ));
    }
    if lhs.dtype() != out.dtype() {
        return Err(crate::Error::dtype_mismatch(
            "dot_general",
            lhs.dtype(),
            out.dtype(),
        ));
    }
    let shape = tenferro_tensor::backend::dot_general_output_shape(
        lhs.shape(),
        rhs.shape(),
        config,
        "dot_general",
    )?;
    if shape.as_slice() != out.shape() {
        return Err(crate::Error::shape_mismatch(
            "dot_general",
            shape.to_vec(),
            out.shape().to_vec(),
        ));
    }
    for scalar in [accumulation.alpha, accumulation.beta] {
        if scalar.dtype() != lhs.dtype() {
            return Err(crate::Error::dtype_mismatch(
                "dot_general",
                lhs.dtype(),
                scalar.dtype(),
            ));
        }
    }
    let out = writable_view(out, "dot_general")?;
    macro_rules! dispatch {
        ($($variant:ident),*) => { match (lhs.tensor_view(), rhs.tensor_view(), out, accumulation.alpha, accumulation.beta) {
            $((TensorView::$variant(a), TensorView::$variant(b), TensorViewMut::$variant(d), ContractionScalar::$variant(alpha), ContractionScalar::$variant(beta)) =>
                initialized(exec, buffers, plans, slot, a, b, d, config, accumulation, alpha, beta),)*
            _ => Err(crate::Error::runtime_state("dot_general", "operands and accumulation scalars must have matching supported dtypes")),
        } };
    }
    dispatch!(F32, F64, C32, C64)
}

#[cfg(any(feature = "native", feature = "blas"))]
impl crate::CpuPlanCache {
    /// The shared contraction plan for `key`, looked up under the cache lock
    /// and returned for execution after the lock is released.
    fn binary_plan<T: Scalar>(
        &self,
        slot: Option<usize>,
        key: Key,
        config: &DotGeneralConfig,
    ) -> crate::Result<Arc<BinaryPlan<T>>> {
        let mut state = self.lock();
        let capacity = state.gemm.capacity();
        state.gemm.binary.plan::<T>(slot, capacity, key, config)
    }
}

#[cfg(test)]
mod retention_tests {
    use super::*;
    use cpueinsum::WorkspaceProvider;
    #[test]
    fn retained_lower_workspace_is_trimmed_on_normal_and_unwind_exit() {
        let resources = ExecutionResources::new(None);
        {
            let _guard = resources.retention_guard(0);
            let mut team = resources.workspace().take_team(&Default::default(), 1, 1);
            team.panel(512);
            drop(team);
            assert!(resources.workspace().retained_bytes() > 0);
        }
        assert_eq!(resources.workspace().retained_bytes(), 0);
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = resources.retention_guard(0);
            let mut team = resources.workspace().take_team(&Default::default(), 1, 1);
            team.panel(512);
            drop(team);
            panic!("exercise workspace cleanup");
        }));
        assert!(unwind.is_err());
        assert_eq!(resources.workspace().retained_bytes(), 0);
    }
}

/// The typed refusal of every numerical route in a no-backend build.
#[cfg(not(any(feature = "native", feature = "blas")))]
pub(crate) fn no_backend(op: &'static str) -> crate::Error {
    crate::Error::unsupported(
        op,
        "this build selects no CPU numerical backend; enable the `native` or `blas` feature",
    )
}

/// Plan cache placeholder for a no-backend build.
#[cfg(not(any(feature = "native", feature = "blas")))]
#[derive(Debug, Default)]
pub(crate) struct BinaryCache;

#[cfg(not(any(feature = "native", feature = "blas")))]
impl BinaryCache {
    pub(crate) fn clear(&mut self) {}
    pub(crate) fn stats(&self) -> tenferro_tensor::CacheStats {
        tenferro_tensor::CacheStats::default()
    }
    pub(crate) fn set_capacity(&mut self, _capacity: usize) {}
}

#[cfg(not(any(feature = "native", feature = "blas")))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn dot_fresh(
    _exec: &mut Exec<'_>,
    _buffers: &mut BufferPool,
    _plans: &crate::CpuPlanCache,
    _slot: Option<usize>,
    _lhs: TensorRead<'_>,
    _rhs: TensorRead<'_>,
    _config: &DotGeneralConfig,
    _accumulation: DotGeneralAccumulation,
    _shape: tenferro_tensor::ShapeVec,
) -> crate::Result<Tensor> {
    Err(no_backend("dot_general"))
}

#[cfg(not(any(feature = "native", feature = "blas")))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn dot_into(
    _exec: &mut Exec<'_>,
    _buffers: &mut BufferPool,
    _plans: &crate::CpuPlanCache,
    _slot: Option<usize>,
    _lhs: TensorRead<'_>,
    _rhs: TensorRead<'_>,
    _config: &DotGeneralConfig,
    _accumulation: DotGeneralAccumulation,
    _out: TensorWrite<'_>,
) -> crate::Result<()> {
    Err(no_backend("dot_general"))
}
